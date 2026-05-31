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
qubo/ising.py — exact algebraic transformation QUBO → Ising.

Mathematics (formula 1.11 of the document):

Variable substitution: x_i = (1 - s_i)/2,  s_i ∈ {-1, +1}

Expanding Q(x) = x^T Q x through spin variables:

    Q(x) = const + Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j

where:
    J_ij  = Q_ij / 2                          (off-diagonal)
    h_i   = -(Q @ 1)_i / 2                    (half row sum)
    const = (Σ_i Q_ii + Σ_{i<j} Q_ij) / 2    (scalar, does not affect optimization)

Derivation:
    x_i x_j = (1-s_i)(1-s_j)/4
    x^T Q x = Σ_{i,j} Q_ij x_i x_j
            = Σ_i Q_ii x_i + Σ_{i<j} 2 Q_ij x_i x_j  (Q symm, x_i^2=x_i)
    Substituting x_i=(1-s_i)/2:
    = Σ_i Q_ii(1-s_i)/2 + Σ_{i<j} Q_ij(1-s_i)(1-s_j)/2
    = [Σ Q_ii/2 + Σ_{i<j} Q_ij/2]     ← const
    - [Σ Q_ii s_i/2 + Σ_{j≠i} Q_ij s_i/2]  ← h_i s_i
    + [Σ_{i<j} Q_ij s_i s_j / 2]       ← J_ij s_i s_j

The document writes J_ij = A_ij/4, where A = 2Q (matrix without the factor 2).
Our notation: Q(x) = x^T Q x with symmetric Q → J_ij = Q_ij/2.

Verification (formula 1.11):
    max_{x ∈ {0,1}^K, Σx=B_max} |Q(x) - [const + Σh_i s_i + Σ J_ij s_i s_j]| < ε_mach
    for case14: all C(13,1) = 13 configurations B_max=1
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List

import numpy as np
from numpy.typing import NDArray

from qubo.qubo_builder import QUBOProblem


@dataclass
class IsingProblem:
    """
    Ising formulation: H = Σ h_i Z_i + Σ J_ij Z_i Z_j + const.

    Attributes
    ----------
    h          : (K,) float — local fields
    J          : (K, K) float — interaction matrix (J_ii = 0)
    const      : float — constant (does not affect QAOA optimization)
    candidates : (K,) int — global bus indices
    K          : number of qubits
    B_max      : number of BESS units
    case       : case name
    """
    h:          NDArray[np.floating]
    J:          NDArray[np.floating]
    const:      float
    candidates: List[int]
    K:          int
    B_max:      int
    case:       str

    def energy_ising(self, s: NDArray) -> float:
        """
        Ising Hamiltonian energy: const + Σ h_i s_i + Σ J_ij s_i s_j.

        Parameters
        ----------
        s : (K,) float — spin vector s_i ∈ {-1, +1}
        """
        e = self.const + float(np.dot(self.h, s))
        for i in range(self.K):
            for j in range(i + 1, self.K):
                e += self.J[i, j] * s[i] * s[j]
        return e


def qubo_to_ising(prob: QUBOProblem) -> IsingProblem:
    """
    Exact algebraic transformation QUBO → Ising.

    J_ij  = Q_ij / 2
    h_i   = -(Q @ ones)_i / 2
    const = (Σ Q_ii + Σ_{i<j} Q_ij) / 2

    Parameters
    ----------
    prob : QUBOProblem

    Returns
    -------
    IsingProblem
    """
    Q = prob.Q
    K = prob.K

    # Local fields
    h = -(Q @ np.ones(K)) / 2.0             # (K,)

    # Interaction matrix
    J = Q.copy() / 2.0
    np.fill_diagonal(J, 0.0)                # J_ii = 0

    # Constant
    const = (np.sum(np.diag(Q)) + np.sum(np.triu(Q, 1))) / 2.0

    return IsingProblem(
        h=h, J=J, const=const,
        candidates=prob.candidates,
        K=K, B_max=prob.B_max,
        case=prob.case,
    )


def verify_ising_identity(
    prob_qubo:  QUBOProblem,
    prob_ising: IsingProblem,
    max_configs: int = 1000,
) -> dict:
    """
    Verifies the QUBO = Ising identity for all feasible configurations.

    max|Q(x) - H_Ising(s)| < ε_mach

    For case14 (K=13, B_max=1): C(13,1)=13 configurations — full enumeration.
    For K>20: limited to max_configs random configurations.

    Parameters
    ----------
    prob_qubo  : QUBOProblem
    prob_ising : IsingProblem
    max_configs: maximum number of configurations

    Returns
    -------
    dict with keys:
        passed      : bool
        max_err     : float
        n_tested    : int
        is_exhaustive: bool — whether full enumeration was performed
    """
    K     = prob_qubo.K
    B_max = prob_qubo.B_max

    errors = []
    n_total = 0

    # Generate feasible configurations
    all_combos = list(combinations(range(K), B_max))
    is_exhaustive = len(all_combos) <= max_configs

    if is_exhaustive:
        combos = all_combos
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_combos), size=max_configs, replace=False)
        combos = [all_combos[i] for i in idx]

    for combo in combos:
        x = np.zeros(K)
        x[list(combo)] = 1.0
        s = 1.0 - 2.0 * x                   # s_i = 1-2x_i

        f_qubo  = prob_qubo.energy(x)
        f_ising = prob_ising.energy_ising(s)
        errors.append(abs(f_qubo - f_ising))
        n_total += 1

    max_err = float(max(errors)) if errors else 0.0

    return {
        'passed':       max_err < 1e-8,
        'max_err':      max_err,
        'n_tested':     n_total,
        'is_exhaustive': is_exhaustive,
    }
