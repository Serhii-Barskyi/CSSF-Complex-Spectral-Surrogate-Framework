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
experiments/run_csnn_t.py — Step 1: full CSNN-T^OPF pipeline.

Reproduces the ρ table from the document for all three cases:
    case14: ρ_rei ≈ 0.999  (document: 0.9993)
    case30: ρ_rei ≈ 0.94   (document: 0.9415)
    case57: ρ_rei ≈ 0.989  (document: 0.9890)

Note: v3 datasets are richer (M=40/82/156 vs 28/46/98 in the document),
so actual ρ values may be higher than in the table.

Usage:
    python experiments/run_csnn_t.py
    python experiments/run_csnn_t.py --cases case14
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset import load_dataset, BESSDataset
from core.gcv import gcv_lambda, effective_rank
from core.csnn_t import fit_csnn_t, CSNNTModel
from core.mpf import compute_mpf
from core.metrics import compute_metrics, pearson_rho

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


def run_case(case: str) -> dict:
    """
    Full Step 1 for a single case.

    Returns
    -------
    dict with results
    """
    print(f"\n{'='*60}")
    print(f"CASE: {case}")
    print(f"{'='*60}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    path = DATA_DIR / f"{case}_full_modeA_Barskyi_Serhii.json"
    if not path.exists():
        print(f"  SKIP: dataset not found ({path})")
        return {}

    t0 = time.time()
    ds = load_dataset(path)
    print(f"  n={ds.n}, M={ds.M}, M_complex={ds.M_complex}")
    print(f"  N_train={ds.N_train}, N_test={ds.N_test}")
    print(f"  delta_r(json)={ds.delta_r}, stability_margin={ds.stability_margin}")
    print(f"  rho_dc_vs_ac={ds.params['rho_dc_vs_ac']:.4f}")
    print(f"  meta_train={dict(__import__('collections').Counter(ds.meta_train))}")
    print(f"  meta_test= {dict(__import__('collections').Counter(ds.meta_test))}")

    # ── GCV ───────────────────────────────────────────────────────────────────
    print(f"\n  [GCV] selecting λ...")
    lam_opt, lam_grid, gcv_vals = gcv_lambda(
        ds.X_train, ds.y_train, n_lambdas=100, lam_range=(-15, 2)
    )
    eff_rank = effective_rank(ds.X_train, lam_opt)
    print(f"  λ_opt = {lam_opt:.2e}")
    print(f"  effective_rank(λ_opt) = {eff_rank:.2f} / rank={ds.rank}")

    # ── CSNN-T ────────────────────────────────────────────────────────────────
    print(f"\n  [CSNN-T] training surrogate...")
    t_fit = time.time()
    model = fit_csnn_t(ds)
    t_fit = time.time() - t_fit
    print(f"  Training time: {t_fit:.3f}s")
    print(f"  H shape: {model.H.shape}")

    # ── MPF ───────────────────────────────────────────────────────────────────
    mpf_result = compute_mpf(ds.n, ds.edges, ds.slack_buses)
    ns = ds.non_slack
    corr_mpf_lsf = float(np.corrcoef(
        mpf_result.mpf[ns],
        np.abs(model.mean_lsf(ds.X_train)[ns])
    )[0, 1])
    print(f"\n  [MPF] |corr(MPF, |LSF|)| = {abs(corr_mpf_lsf):.4f}")

    # ── Inference speed ───────────────────────────────────────────────────────
    t_infer = time.time()
    for _ in range(100):
        _ = model.predict(ds.X_test[:1])
    t_infer = (time.time() - t_infer) / 100
    print(f"  Inference time (1 scenario): {t_infer*1000:.3f} ms")

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_pred = model.predict(ds.X_test)
    metrics = compute_metrics(
        y_pred, ds.y_test, ds.meta_test, ds.non_slack,
        rho_dc_vs_ac=ds.params['rho_dc_vs_ac'],
    )

    print(f"\n  [METRICS] test = {list(metrics['by_type'].keys())}")
    print(f"  ρ_global = {metrics['rho_global']:.4f}")
    print(f"  expl_var = {metrics['expl_var_global']:.4f}")
    print(f"  beats_dc = {metrics['beats_dc']} "
          f"(ρ={metrics['rho_global']:.4f} > ρ_dc={metrics['rho_dc_vs_ac']:.4f})")

    for stype, vals in metrics['by_type'].items():
        print(f"  ρ_{stype:<6} = {vals['rho']:.4f}  "
              f"(n={vals['n']}, expl_var={vals['expl_var']:.4f})")

    # ── Top-3 buses ───────────────────────────────────────────────────────────
    mean_lsf = model.mean_lsf(ds.X_train)
    top3_buses = sorted(ns, key=lambda b: mean_lsf[b])[:3]
    top3_scores = [float(-mean_lsf[b]) for b in top3_buses]
    print(f"\n  Top-3 buses by |E[LSF]|: {list(zip(top3_buses, [f'{s:.4f}' for s in top3_scores]))}")

    t_total = time.time() - t0
    print(f"\n  Total time: {t_total:.2f}s")

    return {
        'case':         case,
        'n':            ds.n,
        'M_complex':    ds.M_complex,
        'rank':         ds.rank,
        'delta_r':      ds.delta_r,
        'lam_opt':      lam_opt,
        'eff_rank':     eff_rank,
        'rho_global':   metrics['rho_global'],
        'expl_var':     metrics['expl_var_global'],
        'beats_dc':     metrics['beats_dc'],
        'rho_dc_vs_ac': metrics['rho_dc_vs_ac'],
        'by_type':      metrics['by_type'],
        'corr_mpf_lsf': corr_mpf_lsf,
        't_fit_s':      t_fit,
        't_infer_ms':   t_infer * 1000,
        'top3_buses':   top3_buses,
    }


def print_summary(results: list) -> None:
    """Prints summary table."""
    print(f"\n{'='*60}")
    print("SUMMARY TABLE (Theorem 1: ρ > ρ_DC)")
    print(f"{'='*60}")
    print(f"{'Case':<8} {'ρ_test':<8} {'ρ_DC':<8} {'beats_DC':<10} {'λ_opt':<12} {'t_fit':<8}")
    print("-" * 60)
    for r in results:
        if not r:
            continue
        print(f"{r['case']:<8} {r['rho_global']:.4f}   {r['rho_dc_vs_ac']:.4f}   "
              f"{'✓' if r['beats_dc'] else '✗':<10} {r['lam_opt']:.2e}   {r['t_fit_s']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Run CSNN-T^OPF experiments')
    parser.add_argument('--cases', nargs='+', default=CASES,
                        choices=CASES, help='Cases to run')
    args = parser.parse_args()

    print("CSSF Framework — Step 1: CSNN-T^OPF Surrogate")
    print(f"Cases: {args.cases}")

    results = [run_case(case) for case in args.cases]
    print_summary(results)
    return results


if __name__ == "__main__":
    main()
