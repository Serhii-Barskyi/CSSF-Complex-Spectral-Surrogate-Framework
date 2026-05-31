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
experiments/verify_h1.py — verification of hypothesis H1.

H1: r(LSF-mixer) >= r(uniform-mixer) at equal p.

r = (E_QAOA - E_worst) / (E_opt - E_worst) ∈ [0,1]  (formula 1.14)

E_opt, E_worst — from brute force (case14: 13 configurations, seconds).

Usage:
    python experiments/verify_h1.py
    python experiments/verify_h1.py --p 1 2 --case case14
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
from qaoa.hamiltonian import build_hamiltonian, HamiltonianSpec
from qaoa.circuit import run_qaoa, qaoa_quality_metric, build_qaoa_circuit

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


def run_qaoa_uniform_mixer(
    ham:       HamiltonianSpec,
    p:         int,
    seed:      int = 42,
    ising_const: float = 0.0,
) -> float:
    """
    Runs QAOA with the standard X-mixer (not LSF-weighted).

    Temporarily creates a copy of ham with uniform lsf_w.
    """
    import copy
    ham_uniform = copy.copy(ham)
    # Uniform mixer: all weights = 1.0
    ham_uniform = HamiltonianSpec(
        h_vec=ham.h_vec,
        J_mat=ham.J_mat,
        lsf_w=np.ones(ham.K),   # uniform!
        lsf_raw=ham.lsf_raw,
        K=ham.K, B_max=ham.B_max,
        candidates=ham.candidates,
        case=ham.case + '_uniform',
    )
    res = run_qaoa(ham_uniform, p=p, optimizer='COBYLA', seed=seed, ising_const=ising_const)
    return res.energy_opt


def verify_h1(case: str = 'case14', p_values: list = None) -> dict:
    """
    Verifies H1 for the given case and set of depths p.

    Returns
    -------
    dict with comparison table
    """
    if p_values is None:
        p_values = [1, 2]

    print(f"\n{'='*60}")
    print(f"H1 Verification: {case}")
    print(f"  LSF-mixer vs Uniform-mixer, p ∈ {p_values}")
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
    ising = qubo_to_ising(prob)
    lsf_raw = np.abs(mdl.mean_lsf(ds.X_train)[cands.candidates])
    ham   = build_hamiltonian(ising, lsf_raw)

    # Brute force: E_opt and E_worst
    if prob.K > 20:
        print(f"  SKIP: K={prob.K} > 20, brute force not available")
        return {}
    bf = prob.brute_force()
    E_opt   = bf['energy_opt']
    E_worst = bf['energy_worst']
    print(f"  Brute force: E_opt={E_opt:.4f}, E_worst={E_worst:.4f}")
    print(f"  n_configs={bf['n_configs']}, bus_opt={bf['buses_opt']}")

    rows = []
    for p in p_values:
        print(f"\n  p={p}:")

        # LSF-weighted mixer
        res_lsf = run_qaoa(ham, p=p, optimizer='COBYLA', seed=42)
        r_lsf   = qaoa_quality_metric(res_lsf.energy_opt, E_opt, E_worst)
        print(f"    LSF-mixer:     E={res_lsf.energy_opt:.4f}, r={r_lsf:.4f}, "
              f"feasible={res_lsf.feasible}, bus={res_lsf.buses_opt}")

        # Uniform mixer
        E_uni = run_qaoa_uniform_mixer(ham, p=p, seed=42, ising_const=ising.const)
        r_uni = qaoa_quality_metric(E_uni, E_opt, E_worst)
        print(f"    Uniform-mixer: E={E_uni:.4f}, r={r_uni:.4f}")

        h1_holds = r_lsf >= r_uni - 1e-6
        delta_r  = r_lsf - r_uni
        print(f"    H1: r_LSF={r_lsf:.4f} >= r_uniform={r_uni:.4f}? "
              f"{'✓' if h1_holds else '✗'} (Δr={delta_r:+.4f})")

        rows.append({
            'p': p, 'r_lsf': r_lsf, 'r_uniform': r_uni,
            'delta_r': delta_r, 'H1_holds': h1_holds,
            'E_lsf': res_lsf.energy_opt, 'E_uniform': E_uni,
        })

    h1_verified = all(row['H1_holds'] for row in rows)
    print(f"\n  H1 {'VERIFIED ✓' if h1_verified else 'NOT VERIFIED ✗'} "
          f"(all p ∈ {p_values})")

    return {
        'case': case, 'E_opt': E_opt, 'E_worst': E_worst,
        'rows': rows, 'H1_verified': h1_verified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',   default='case14', choices=['case14','case30','case57'])
    parser.add_argument('--p',      nargs='+', type=int, default=[1, 2])
    args = parser.parse_args()

    print("CSSF Framework — Experiment H1: LSF-mixer vs Uniform-mixer")
    result = verify_h1(args.case, args.p)

    if result:
        print(f"\nH1 result for {args.case}: {'VERIFIED' if result['H1_verified'] else 'FAILED'}")


if __name__ == "__main__":
    main()
