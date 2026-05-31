# -*- coding: utf-8 -*-
# Author: Serhii Barskyi | https://www.linkedin.com/in/serhii-barskyi/
# Data Science Course: https://preply.com/en/tutor/7756455
# Framework: Complex Spectral Surrogate Framework (CSSF)
#
# Licensed under the Apache License, Version 2.0.
# You may not use this file except in compliance with the License.
# Full license text: https://www.apache.org/licenses/LICENSE-2.0
#
# Attribution required: if you use this code, please cite:
# Serhii Barskyi, Complex Spectral Surrogate Framework (CSSF),
# https://www.linkedin.com/in/serhii-barskyi/
"""
qaoa/circuit.py — parametric QAOA circuit.

Mathematics (formula 1.12):
    |ψ_p(γ,β)⟩ = Π_{l=1}^p e^{-iβ_l H_M} e^{-iγ_l H_C} |+⟩^{⊗K}

Gate implementation:
    e^{-iγ h_i Z_i}     = Rz(2γ h_i)          on qubit i
    e^{-iγ J_ij Z_i Z_j} = RZZ(2γ J_ij)        on qubits i,j
    e^{-iβ LSF_i X_i}   = Rx(2β·LSF_i)        on qubit i  ← LSF-weighted!

Qiskit bit ordering (little-endian):
    Qubit 0 = LSB: bitstring index i → x_k = (i >> k) & 1

Feasibility: Σ_k x_k = B_max (exactly B_max ones).

Decoding:
    For case14/30 (K≤22): full statevector → all 2^K probabilities →
    best feasible bitstring.
    For case57 (K=42): statevector infeasible (dim=4×10^12) →
    use CSNN-T^QAOA.

Optimizers:
    COBYLA (scipy): K≤14, exact, ~30 iterations
    SPSA (qiskit-algorithms): K>14, stochastic, ~200 iterations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from qaoa.hamiltonian import HamiltonianSpec


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class QAOAResult:
    """
    QAOA optimization result.

    Attributes
    ----------
    buses_opt   : list of global bus indices (optimal placement)
    x_opt       : (K,) binary vector
    energy_opt  : E(γ*,β*) — optimal value
    feasible    : Σx_i = B_max
    gamma_opt   : (p,) optimal γ parameters
    beta_opt    : (p,) optimal β parameters
    n_iter      : number of optimizer iterations
    backend     : 'statevector' or 'GPU'
    case        : case name
    """
    buses_opt:  List[int]
    x_opt:      NDArray[np.floating]
    energy_opt: float
    feasible:   bool
    gamma_opt:  NDArray[np.floating]
    beta_opt:   NDArray[np.floating]
    n_iter:     int
    backend:    str
    case:       str


# ── Circuit construction ──────────────────────────────────────────────────────

def build_qaoa_circuit(
    ham:   HamiltonianSpec,
    gamma: NDArray,
    beta:  NDArray,
    p:     int,
) -> object:
    """
    Builds a parametric QAOA Qiskit circuit.

    |ψ_p⟩ = Π_{l=1}^p e^{-iβ_l H_M} e^{-iγ_l H_C} |+⟩^K

    Parameters
    ----------
    ham   : HamiltonianSpec
    gamma : (p,) float
    beta  : (p,) float
    p     : depth

    Returns
    -------
    QuantumCircuit with save_statevector()
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator as _AerSim  # registers save_statevector

    K = ham.K
    qc = QuantumCircuit(K)

    # Initial state: |+⟩^K = H^⊗K |0⟩^K
    qc.h(range(K))

    for layer in range(p):
        g = float(gamma[layer])
        b = float(beta[layer])

        # ── Cost unitary: e^{-iγ H_C} ────────────────────────────────────
        # e^{-iγ h_i Z_i} = Rz(2γ h_i)
        for i in range(K):
            if abs(ham.h_vec[i]) > 1e-12:
                qc.rz(2.0 * g * ham.h_vec[i], i)

        # e^{-iγ J_ij Z_i Z_j} = RZZ(2γ J_ij)
        for i in range(K):
            for j in range(i + 1, K):
                if abs(ham.J_mat[i, j]) > 1e-12:
                    qc.rzz(2.0 * g * ham.J_mat[i, j], i, j)

        # ── Mixer unitary: e^{-iβ H_M} ───────────────────────────────────
        # e^{-iβ LSF_i X_i} = Rx(2β·LSF_i)  ← LSF-weighted mixer
        for i in range(K):
            if abs(ham.lsf_w[i]) > 1e-12:
                qc.rx(2.0 * b * ham.lsf_w[i], i)

    qc.save_statevector()
    return qc


# ── Statevector decoding ──────────────────────────────────────────────────────

def _decode_statevector(
    sv:           NDArray[np.complexfloating],
    ham:          HamiltonianSpec,
    n_shots:      int,
    ising_const:  float = 0.0,
) -> Tuple[NDArray, float, bool]:
    """
    Finds the best feasible bitstring from the statevector.

    Qiskit ordering (little-endian): qubit k → bit k (LSB).
    Feasible: Σ_k x_k = B_max.

    energy = Ising_no_const + ising_const = QUBO energy (consistent with brute_force)

    Returns
    -------
    x_opt    : (K,) binary vector
    energy   : QUBO energy = Ising_no_const + const
    feasible : bool
    """
    K     = ham.K
    B_max = ham.B_max
    probs = np.abs(sv) ** 2         # (2^K,)

    best_e = np.inf
    best_x = None

    for idx in range(len(probs)):
        # little-endian: qubit k = (idx >> k) & 1
        x = np.array([(idx >> k) & 1 for k in range(K)], dtype=float)
        if int(x.sum()) != B_max:
            continue
        if probs[idx] < 1e-10:
            continue

        # Expected value of H_C for this bitstring: <s|H_C|s>
        # = Σ h_i s_i + Σ J_ij s_i s_j,  s_i = 1-2x_i
        s = 1.0 - 2.0 * x
        e = float(np.dot(ham.h_vec, s))
        for i in range(K):
            for j in range(i + 1, K):
                e += ham.J_mat[i, j] * s[i] * s[j]

        if e < best_e:
            best_e = e
            best_x = x.copy()

    if best_x is None:
        # Fallback: take bitstring with highest probability among feasible
        best_prob = -1.0
        for idx in range(len(probs)):
            x = np.array([(idx >> k) & 1 for k in range(K)], dtype=float)
            if int(x.sum()) == B_max and probs[idx] > best_prob:
                best_prob = probs[idx]
                best_x = x.copy()
                best_e = np.inf  # will recompute

        if best_x is not None:
            s = 1.0 - 2.0 * best_x
            best_e = float(np.dot(ham.h_vec, s))
            for i in range(K):
                for j in range(i + 1, K):
                    best_e += ham.J_mat[i, j] * s[i] * s[j]

    feasible = best_x is not None and int(best_x.sum()) == B_max
    # Add Ising constant → convert to QUBO scale (consistent with brute_force)
    best_e_qubo = (best_e + ising_const) if best_e != np.inf else np.inf
    return best_x, best_e_qubo, feasible


# ── QAOA cost function ────────────────────────────────────────────────────────

def _make_cost_fn(ham: HamiltonianSpec, p: int, sim):
    """
    Closure: cost_fn(params) → E(γ,β) via statevector.

    params : (2p,) — [γ_1,...,γ_p, β_1,...,β_p]
    """
    call_count = [0]

    def cost_fn(params: NDArray) -> float:
        call_count[0] += 1
        gamma = np.array(params[:p])
        beta  = np.array(params[p:])
        qc = build_qaoa_circuit(ham, gamma, beta, p)
        sv = np.asarray(sim.run(qc).result().get_statevector())

        # Expected value <H_C>
        probs = np.abs(sv) ** 2
        E = 0.0
        K = ham.K
        for idx in range(len(probs)):
            if probs[idx] < 1e-12:
                continue
            x = np.array([(idx >> k) & 1 for k in range(K)], dtype=float)
            s = 1.0 - 2.0 * x
            e_loc = float(np.dot(ham.h_vec, s))
            for i in range(K):
                for j in range(i + 1, K):
                    e_loc += ham.J_mat[i, j] * s[i] * s[j]
            E += probs[idx] * e_loc
        return E

    return cost_fn, call_count


# ── Main run_qaoa function ────────────────────────────────────────────────────

def run_qaoa(
    ham:          HamiltonianSpec,
    p:            int = 1,
    n_shots:      int = 1024,
    optimizer:    str = 'COBYLA',
    seed:         int = 42,
    backend:      str = 'statevector',
    ising_const:  float = 0.0,
) -> QAOAResult:
    """
    Runs QAOA for the BESS placement problem.

    Algorithm:
        1. Create AerSimulator (statevector or GPU)
        2. cost_fn(γ,β) = <ψ_p(γ,β)|H_C|ψ_p(γ,β)> via statevector
        3. Optimize γ,β (COBYLA or SPSA)
        4. Decode: statevector → best feasible bitstring

    Parameters
    ----------
    ham       : HamiltonianSpec
    p         : QAOA depth (number of layers)
    n_shots   : not used for statevector (kept for compatibility)
    optimizer : 'COBYLA' (K≤14) or 'SPSA' (K>14)
    seed      : random seed
    backend   : 'statevector' or 'GPU'

    Returns
    -------
    QAOAResult
    """
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize as sp_minimize

    # ── Simulator ─────────────────────────────────────────────────────────────
    if backend == 'GPU':
        try:
            sim = AerSimulator(method='statevector', device='GPU')
            # GPU test
            from qiskit import QuantumCircuit as _QC
            _qc = _QC(1); _qc.h(0); _qc.save_statevector()
            sim.run(_qc).result()
        except Exception:
            sim = AerSimulator(method='statevector')
            backend = 'statevector'
    else:
        sim = AerSimulator(method='statevector')

    # ── Initial parameters ────────────────────────────────────────────────────
    rng    = np.random.default_rng(seed)
    params0 = rng.uniform(0, 2 * np.pi, 2 * p)

    # ── Cost function ─────────────────────────────────────────────────────────
    cost_fn, call_count = _make_cost_fn(ham, p, sim)

    # ── Optimization ──────────────────────────────────────────────────────────
    if optimizer == 'SPSA':
        try:
            from qiskit_algorithms.optimizers import SPSA as QiskitSPSA
            spsa = QiskitSPSA(maxiter=200)
            res_spsa = spsa.minimize(cost_fn, params0)
            params_opt = res_spsa.x
            n_iter = int(res_spsa.nfev) if hasattr(res_spsa, 'nfev') else 200
        except ImportError:
            # Fallback to scipy Powell
            res = sp_minimize(cost_fn, params0, method='Powell',
                              options={'maxiter': 200})
            params_opt = res.x
            n_iter = int(res.nfev)
    else:
        res = sp_minimize(
            cost_fn, params0, method='COBYLA',
            options={'maxiter': 300, 'rhobeg': 0.5},
        )
        params_opt = res.x
        n_iter = call_count[0]

    gamma_opt = np.array(params_opt[:p])
    beta_opt  = np.array(params_opt[p:])

    # ── Final decoding via statevector ────────────────────────────────────────
    qc_final = build_qaoa_circuit(ham, gamma_opt, beta_opt, p)
    sv_final = np.asarray(sim.run(qc_final).result().get_statevector())
    x_opt, energy_opt, feasible = _decode_statevector(
        sv_final, ham, n_shots, ising_const=ising_const
    )

    if x_opt is None:
        x_opt = np.zeros(ham.K)
        feasible = False
        energy_opt = float('inf')

    buses_opt = [ham.candidates[k] for k in range(ham.K) if x_opt[k] > 0.5]

    return QAOAResult(
        buses_opt=buses_opt,
        x_opt=x_opt,
        energy_opt=energy_opt,
        feasible=feasible,
        gamma_opt=gamma_opt,
        beta_opt=beta_opt,
        n_iter=n_iter,
        backend=backend,
        case=ham.case,
    )


def qaoa_quality_metric(
    energy_qaoa:  float,
    energy_opt:   float,
    energy_worst: float,
) -> float:
    """
    Quality metric r ∈ [0,1] (formula 1.14):
        r = (E_QAOA - E_worst) / (E_opt - E_worst)

    r=1: QAOA found the optimum.
    r=0: QAOA found the worst solution.
    """
    denom = energy_opt - energy_worst
    if abs(denom) < 1e-12:
        return 1.0
    return float(np.clip((energy_qaoa - energy_worst) / denom, 0.0, 1.0))
