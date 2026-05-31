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
experiments/run_qaoa.py — Steps 2-4: QUBO + QAOA pipeline.

case14/30: statevector simulator.
case57: CSNN-T^QAOA (statevector infeasible at N_q=42).

Usage:
    python experiments/run_qaoa.py
    python experiments/run_qaoa.py --cases case14 --p 1 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset import load_dataset
from core.csnn_t import fit_csnn_t
from core.mpf import compute_mpf
from qubo.screener import screen_candidates
from qubo.qubo_builder import build_qubo, verify_qubo
from qubo.ising import qubo_to_ising, verify_ising_identity
from qaoa.hamiltonian import build_hamiltonian
from qaoa.circuit import run_qaoa, qaoa_quality_metric
from qaoa.csnn_t_qaoa import build_csnn_t_qaoa
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
CASES    = ["case14", "case30", "case57"]


def run_case(case: str, p_values: list, backend: str = 'statevector') -> dict:
    print(f"\n{'='*60}")
    print(f"QAOA PIPELINE: {case}")
    print(f"{'='*60}")

    path = DATA_DIR / f"{case}_full_modeA_Barskyi_Serhii.json"
    if not path.exists():
        print(f"  SKIP: dataset not found")
        return {}

    # ── Pipeline ──────────────────────────────────────────────────────────────
    ds    = load_dataset(path)
    mdl   = fit_csnn_t(ds)
    mpf   = compute_mpf(ds.n, ds.edges, ds.slack_buses)
    K     = min(13, len(ds.non_slack))
    cands = screen_candidates(ds, mdl, mpf, K=K)
    prob  = build_qubo(ds, cands, mdl, mpf, B_max=1)
    ising = qubo_to_ising(prob)
    lsf_raw = np.abs(mdl.mean_lsf(ds.X_train)[cands.candidates])
    ham   = build_hamiltonian(ising, lsf_raw)

    # ── QUBO and Ising verification ───────────────────────────────────────────
    vq = verify_qubo(prob)
    print(f"  QUBO: K={prob.K}, λ={prob.lam_pen:.4f}, "
          f"symmetric={vq['Q_symmetric']}, c≥0={vq['c_nonneg']}, "
          f"λ_ok={vq['lam_sufficient']}, connected={vq['n_connected']}")

    vi = verify_ising_identity(prob, ising)
    print(f"  Ising: J_ij=Q_ij/2, max|Q(x)-H(s)|={vi['max_err']:.2e}, "
          f"exhaustive={vi['is_exhaustive']}, passed={vi['passed']}")

    # ── Brute force (only K≤20) ───────────────────────────────────────────────
    bf = None
    if prob.K <= 20:
        bf = prob.brute_force()
        print(f"  Brute force: E_opt={bf['energy_opt']:.4f}, "
              f"E_worst={bf['energy_worst']:.4f}, bus={bf['buses_opt']}")

    # ── case57: statevector feasible at K=13 (2^13=8192) ─────────────────────
    # NOTE: N_q=42 only when K=len(non_slack)=56 — code uses K=min(13,56)=13
    # At K=13: dim=2^13=8192 — statevector trivial in seconds
    # The N_q=42 limitation applies only if K=42 (not used here)
    if case == 'case57' and prob.K > 30:
        print(f"\n  case57: K={prob.K} > 30 → statevector not practical")
        from qaoa.csnn_t_qaoa import build_frequency_set
        for p in p_values:
            freqs = build_frequency_set(p=p, k_max=3)
            print(f"  p={p}: |Λ_p|={len(freqs)} frequencies, speedup ~{30**(2*p)/50:.1e}×")
        return {'case': case, 'note': 'csnn_t_qaoa_only'}

    # ── QAOA for case14/30 ────────────────────────────────────────────────────
    results_by_p = {}
    for p in p_values:
        opt = 'COBYLA' if prob.K <= 14 else 'SPSA'
        print(f"\n  QAOA p={p}, optimizer={opt}, backend={backend}:")
        t0 = time.time()
        res = run_qaoa(ham, p=p, optimizer=opt, seed=42, backend=backend, ising_const=ising.const)
        dt = time.time() - t0

        r = 0.0
        if bf:
            r = qaoa_quality_metric(res.energy_opt, bf['energy_opt'],
                                    bf['energy_worst'])
            gap_pct = 100 * (res.energy_opt - bf['energy_opt']) / abs(bf['energy_opt'] + 1e-10)
            print(f"    E_QAOA={res.energy_opt:.4f}, E_opt={bf['energy_opt']:.4f}")
            print(f"    gap={gap_pct:.2f}%, r={r:.4f}")
            print(f"    buses_opt={res.buses_opt}, feasible={res.feasible}")
        else:
            print(f"    E_QAOA={res.energy_opt:.4f}, feasible={res.feasible}")
            print(f"    buses_opt={res.buses_opt}")

        print(f"    n_iter={res.n_iter}, time={dt:.1f}s")
        results_by_p[p] = {'res': res, 'r': r, 'time': dt}

    return {
        'case': case, 'K': prob.K, 'B_max': prob.B_max,
        'bf': bf, 'results_by_p': results_by_p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases',   nargs='+', default=['case14'],
                        choices=CASES)
    parser.add_argument('--p',       nargs='+', type=int, default=[1, 2])
    parser.add_argument('--backend', default='statevector',
                        choices=['statevector', 'GPU'])
    args = parser.parse_args()

    print("CSSF Framework — Steps 2-4: QUBO + QAOA")
    results = [run_case(c, args.p, args.backend) for c in args.cases]

    print(f"\n{'='*60}")
    print("SUMMARY TABLE:")
    for r in results:
        if not r or 'results_by_p' not in r:
            continue
        for p, pr in r['results_by_p'].items():
            res = pr['res']
            print(f"  {r['case']} p={p}: r={pr['r']:.4f}, "
                  f"bus={res.buses_opt}, t={pr['time']:.1f}s")


if __name__ == "__main__":
    main()
