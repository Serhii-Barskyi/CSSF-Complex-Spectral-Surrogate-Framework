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
experiments/verify_h2.py — verification of hypothesis H2.

H2: GP+CSNN-T prior achieves target r with M0 < M0^COBYLA.

r(M0) curves for three strategies:
    1. csnn_prior   — GP with CSNN-T^QAOA prior (our method)
    2. zero_prior   — GP without prior (zero prior)
    3. cobyla       — direct COBYLA optimization

r = (E_best - E_worst) / (E_opt - E_worst) ∈ [0,1]

Usage:
    python experiments/verify_h2.py
    python experiments/verify_h2.py --M0_max 30 --n_seeds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset import load_dataset
from core.csnn_t import fit_csnn_t
from core.mpf import compute_mpf
from qubo.screener import screen_candidates
from qubo.qubo_builder import build_qubo
from qubo.ising import qubo_to_ising
from qaoa.hamiltonian import build_hamiltonian
from qaoa.circuit import run_qaoa, qaoa_quality_metric, build_qaoa_circuit
from qaoa.gp_optimizer import optimize_landscape_gp

import kagglehub
import pathlib as _pl
# Download latest version
_kaggle_path = kagglehub.dataset_download("serhiibarskyi/ac-loss-sensitivity-factor-cases-143057118")
print("Path to dataset files:", _kaggle_path)
# Locate JSON files — kagglehub path structure varies by environment
_base = _pl.Path(_kaggle_path)
_all_json = list(_base.rglob("*.json"))
print("JSON files found:", [f.name for f in _all_json[:10]])
_probe = [f for f in _all_json if f.name == "case14_full_modeA_Barskyi_Serhii.json"]
if not _probe:
    raise FileNotFoundError(f"case14_full_modeA_Barskyi_Serhii.json not found. All JSON files: {[f.name for f in _all_json]}")
DATA_DIR = _probe[0].parent
print("DATA_DIR:", DATA_DIR)


def make_cost_fn(ham, p: int, sim):
    """Cost function E(γ,β) via statevector."""
    def cost_fn(params):
        gamma = params[:p]
        beta  = params[p:]
        qc = build_qaoa_circuit(ham, gamma, beta, p)
        sv = np.asarray(sim.run(qc).result().get_statevector())
        probs = np.abs(sv) ** 2
        K = ham.K
        E = 0.0
        for idx in range(len(probs)):
            if probs[idx] < 1e-12:
                continue
            x = np.array([(idx >> k) & 1 for k in range(K)], dtype=float)
            s = 1.0 - 2.0 * x
            e_loc = float(np.dot(ham.h_vec, s))
            for i in range(K):
                for j in range(i+1, K):
                    e_loc += ham.J_mat[i, j] * s[i] * s[j]
            E += probs[idx] * e_loc
        return E
    return cost_fn


def verify_h2(
    case:     str = 'case14',
    p:        int = 1,
    M0_max:   int = 40,
    n_seeds:  int = 3,
) -> dict:
    """
    Builds r(M0) curves for three strategies.

    Parameters
    ----------
    case    : case name
    p       : QAOA depth
    M0_max  : maximum evaluation budget
    n_seeds : number of random seeds for averaging
    """
    print(f"\n{'='*60}")
    print(f"H2 Verification: {case}, p={p}, M0_max={M0_max}")
    print(f"{'='*60}")

    path = DATA_DIR / f"{case}_full_modeA_Barskyi_Serhii.json"
    if not path.exists():
        print(f"  SKIP: dataset not found")
        return {}

    # Pipeline
    ds    = load_dataset(path)
    mdl   = fit_csnn_t(ds)
    mpf   = compute_mpf(ds.n, ds.edges, ds.slack_buses)
    K     = min(13, len(ds.non_slack))
    cands = screen_candidates(ds, mdl, mpf, K=K)
    prob  = build_qubo(ds, cands, mdl, mpf, B_max=1)

    if prob.K > 20:
        print(f"  SKIP: K={prob.K} > 20")
        return {}

    ising   = qubo_to_ising(prob)
    lsf_raw = np.abs(mdl.mean_lsf(ds.X_train)[cands.candidates])
    ham     = build_hamiltonian(ising, lsf_raw)

    # Brute force
    bf      = prob.brute_force()
    E_opt   = bf['energy_opt']
    E_worst = bf['energy_worst']
    print(f"  E_opt={E_opt:.4f}, E_worst={E_worst:.4f}")

    # Guard: CSNN-T^QAOA surrogate requires n_init >= 5 points
    # n_init = min(10, M0//5), so M0 >= 25 is needed
    if M0_max < 25:
        print(f"  WARNING: M0_max={M0_max} < 25 → CSNN prior not built "
              f"(n_init={min(10,M0_max//5)} < 5). csnn_prior ≡ zero_prior.")

    # Simulator
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method='statevector')

    # Checkpoint: M0 values
    M0_values = list(range(10, M0_max + 1, 5))

    curves = {'csnn_prior': [], 'zero_prior': [], 'cobyla': []}

    for seed in range(n_seeds):
        print(f"\n  Seed {seed+1}/{n_seeds}:")
        cost_fn = make_cost_fn(ham, p, sim)

        # 1. csnn_prior
        gp_csnn = optimize_landscape_gp(
            cost_fn, ham, p=p, M0=M0_max, seed=seed,
            strategy='csnn_prior',
            energy_opt=E_opt, energy_worst=E_worst,
        )
        r_csnn = qaoa_quality_metric(gp_csnn.energy_opt, E_opt, E_worst)
        curves['csnn_prior'].append(r_csnn)
        print(f"    csnn_prior: r={r_csnn:.4f} (M0={M0_max})")

        # 2. zero_prior
        gp_zero = optimize_landscape_gp(
            cost_fn, ham, p=p, M0=M0_max, seed=seed,
            strategy='zero_prior',
            energy_opt=E_opt, energy_worst=E_worst,
        )
        r_zero = qaoa_quality_metric(gp_zero.energy_opt, E_opt, E_worst)
        curves['zero_prior'].append(r_zero)
        print(f"    zero_prior: r={r_zero:.4f} (M0={M0_max})")

        # 3. cobyla
        res_cobyla = run_qaoa(ham, p=p, optimizer='COBYLA', seed=seed)
        r_cobyla = qaoa_quality_metric(res_cobyla.energy_opt, E_opt, E_worst)
        curves['cobyla'].append(r_cobyla)
        print(f"    cobyla:     r={r_cobyla:.4f} (n_iter={res_cobyla.n_iter})")

    # Averaging
    print(f"\n  {'='*40}")
    print(f"  Mean r (M0={M0_max}, {n_seeds} seeds):")
    for strategy, rs in curves.items():
        r_mean = np.mean(rs)
        r_std  = np.std(rs)
        print(f"    {strategy:<15}: r={r_mean:.4f} ± {r_std:.4f}")

    # H2: csnn_prior >= zero_prior?
    h2_holds = np.mean(curves['csnn_prior']) >= np.mean(curves['zero_prior']) - 1e-6
    print(f"\n  H2: csnn_prior >= zero_prior? {'✓' if h2_holds else '✗'}")

    return {
        'case': case, 'p': p, 'M0_max': M0_max,
        'E_opt': E_opt, 'E_worst': E_worst,
        'curves': curves, 'H2_holds': h2_holds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',    default='case14',
                        choices=['case14','case30','case57'])
    parser.add_argument('--p',       type=int, default=1)
    parser.add_argument('--M0_max',  type=int, default=30)
    parser.add_argument('--n_seeds', type=int, default=3)
    args = parser.parse_args()

    print("CSSF Framework — Experiment H2: GP+CSNN prior vs zero prior vs COBYLA")
    result = verify_h2(args.case, args.p, args.M0_max, args.n_seeds)

    if result:
        print(f"\nH2 result: {'VERIFIED' if result['H2_holds'] else 'NOT VERIFIED'}")


if __name__ == "__main__":
    main()
