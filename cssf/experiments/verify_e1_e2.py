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
experiments/verify_e1_e2.py — verification of E1 and E2.

E1: ΔL_i = L0 - L_i — direct AC OPF confirmation.
    Buses with high LSF/MPF ranking produce the largest loss reduction.
    ρ(score_CSNN, ΔL_AC) — measure of economic value of the surrogate.

E2: ‖k‖_1 ≤ 3 covers >95% of energy E(γ,β) at p ≤ 3.
    Verifies Hypothesis E2 (document).

Usage:
    python experiments/verify_e1_e2.py
    python experiments/verify_e1_e2.py --case case14 --bess_mw 5.0
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
from qaoa.csnn_t_qaoa import build_csnn_t_qaoa, build_frequency_set

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


# ── E1: Direct AC OPF confirmation ───────────────────────────────────────────

def compute_delta_L(case: str, candidates: list, bess_mw: float = 5.0) -> dict:
    """
    Computes ΔL_i = L0 - L_i for each bus via AC OPF (pandapower).

    Parameters
    ----------
    case       : 'case14', 'case30', 'case57'
    candidates : list of candidate buses
    bess_mw    : BESS power in MW

    Returns
    -------
    dict {bus: delta_L_mw}
    """
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError:
        raise ImportError("pandapower: pip install pandapower")

    net_fn = {'case14': pn.case14, 'case30': pn.case30, 'case57': pn.case57}
    if case not in net_fn:
        raise ValueError(f"Unknown case: {case}")

    # Base losses
    net0 = net_fn[case]()
    pp.runpp(net0, algorithm='nr', numba=False)
    L0 = float(net0.res_line['pl_mw'].sum())

    delta_L = {'L0': L0}
    for bus in candidates:
        net = net_fn[case]()
        pp.create_sgen(net, bus=bus, p_mw=bess_mw, q_mvar=0.0, name='BESS')
        try:
            pp.runpp(net, algorithm='nr', numba=False)
            L_bess = float(net.res_line['pl_mw'].sum())
            delta_L[bus] = L0 - L_bess
        except Exception:
            delta_L[bus] = 0.0

    return delta_L


def verify_e1(case: str = 'case14', bess_mw: float = 5.0) -> dict:
    """
    Verifies E1: ρ(score_CSNN, ΔL_AC) > 0.

    Document: top-3 buses by LSF/MPF should systematically produce
    the largest ΔL when BESS is installed.
    """
    print(f"\n{'='*60}")
    print(f"E1 Verification: {case}  (BESS={bess_mw} MW)")
    print(f"{'='*60}")

    path = DATA_DIR / f"{case}_full_modeA_Barskyi_Serhii.json"
    if not path.exists():
        print(f"  SKIP: dataset not found")
        return {}

    ds    = load_dataset(path)
    mdl   = fit_csnn_t(ds)
    mpf   = compute_mpf(ds.n, ds.edges, ds.slack_buses)
    K     = min(13, len(ds.non_slack))
    cands = screen_candidates(ds, mdl, mpf, K=K)

    print(f"  Computing ΔL for {K} candidates (AC OPF)...")
    dL_map = compute_delta_L(case, cands.candidates, bess_mw)
    L0     = dL_map['L0']
    dLs    = np.array([dL_map[b] for b in cands.candidates])
    scores = cands.scores

    print(f"  L0 = {L0:.4f} MW")

    # Correlation score vs ΔL
    rho = float(np.corrcoef(scores, dLs)[0, 1])
    print(f"  ρ(score_CSNN, ΔL_AC) = {rho:.4f}")

    # Top-3 comparison
    top3_dL    = sorted(cands.candidates,
                        key=lambda b: -dL_map[b])[:3]
    top3_score = cands.candidates[:3]
    overlap    = len(set(top3_dL) & set(top3_score))

    print(f"\n  Top-3 by ΔL_AC:    {top3_dL}")
    print(f"  Top-3 by score:    {top3_score}")
    print(f"  Overlap: {overlap}/3")

    # Table
    print(f"\n  {'Bus':<6} {'score':<8} {'ΔL(MW)':<10} {'rank_score':<12} {'rank_dL'}")
    print(f"  {'-'*50}")
    rank_score = {b: i+1 for i, b in enumerate(cands.candidates)}
    rank_dL    = {b: i+1 for i, b in
                  enumerate(sorted(cands.candidates, key=lambda b: -dL_map[b]))}
    for i, bus in enumerate(cands.candidates[:K]):
        print(f"  {bus:<6} {scores[i]:<8.4f} {dL_map[bus]:<10.4f} "
              f"{rank_score[bus]:<12} {rank_dL[bus]}")

    e1_holds = rho > 0.5
    print(f"\n  E1: ρ={rho:.4f} > 0.5? {'✓' if e1_holds else '✗'}")

    return {
        'case': case, 'L0': L0, 'rho': rho,
        'top3_dL': top3_dL, 'top3_score': top3_score,
        'overlap': overlap, 'E1_holds': e1_holds,
        'dL_map': dL_map,
    }


# ── E2: Spectral concentration ────────────────────────────────────────────────

def verify_e2(case: str = 'case14', p_values: list = None) -> dict:
    """
    Verifies E2: ‖k‖_1 ≤ 3 covers >95% of energy E(γ,β) at p≤3.

    Method: run QAOA at M0=50 points, train CSNN-T^QAOA,
    analyze energy fraction by ‖k‖_1.
    """
    if p_values is None:
        p_values = [1, 2]

    print(f"\n{'='*60}")
    print(f"E2 Verification: {case}")
    print(f"  Hypothesis E2: ‖k‖_1 ≤ 3 covers >95% of energy")
    print(f"{'='*60}")

    path = DATA_DIR / f"{case}_full_modeA_Barskyi_Serhii.json"
    if not path.exists():
        print(f"  SKIP: dataset not found")
        return {}

    ds    = load_dataset(path)
    mdl   = fit_csnn_t(ds)
    mpf   = compute_mpf(ds.n, ds.edges, ds.slack_buses)
    K     = min(13, len(ds.non_slack))
    cands = screen_candidates(ds, mdl, mpf, K=K)
    prob  = build_qubo(ds, cands, mdl, mpf, B_max=1)
    ising = qubo_to_ising(prob)
    lsf_raw = np.abs(mdl.mean_lsf(ds.X_train)[cands.candidates])
    ham   = build_hamiltonian(ising, lsf_raw)

    from qiskit_aer import AerSimulator
    sim = AerSimulator(method='statevector')

    results = {}
    for p in p_values:
        print(f"\n  p={p}:")
        dim = 2 * p
        rng = np.random.default_rng(42)
        M0  = 50

        # Compute E(γ,β) at M0 points
        pts    = rng.uniform(0, 2 * np.pi, (M0, dim))
        E_vals = []
        for pt in pts:
            gamma = pt[:p]; beta = pt[p:]
            qc = build_qaoa_circuit(ham, gamma, beta, p)
            sv = np.asarray(sim.run(qc).result().get_statevector())
            probs = np.abs(sv) ** 2
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
            E_vals.append(E)

        E_vals = np.array(E_vals)
        print(f"    E(γ,β): min={E_vals.min():.4f}, max={E_vals.max():.4f}")

        # Train CSNN-T^QAOA with different k_max values
        energy_by_kmax = {}
        for k_max in range(4):
            model = build_csnn_t_qaoa(pts, E_vals, p=p, k_max=k_max)
            h_abs2 = np.abs(model.h) ** 2
            total  = h_abs2.sum() + 1e-15
            norms  = np.abs(model.freqs).sum(axis=1)
            frac   = float(h_abs2[norms <= k_max].sum() / total)
            energy_by_kmax[k_max] = frac
            print(f"    k_max={k_max}: {frac*100:.1f}% energy")

        e2_holds = energy_by_kmax.get(3, 0.0) >= 0.95
        print(f"    E2 (k_max=3 ≥ 95%): {'✓' if e2_holds else '✗'}")
        results[p] = {'energy_by_kmax': energy_by_kmax, 'E2_holds': e2_holds}

    return {'case': case, 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',    default='case14',
                        choices=['case14', 'case30', 'case57'])
    parser.add_argument('--bess_mw', type=float, default=5.0)
    parser.add_argument('--p',       nargs='+', type=int, default=[1])
    args = parser.parse_args()

    print("CSSF Framework — Experiments E1 and E2")

    r_e1 = verify_e1(args.case, args.bess_mw)
    r_e2 = verify_e2(args.case, args.p)

    print(f"\n{'='*60}")
    print("SUMMARY:")
    if r_e1:
        print(f"  E1: ρ(score,ΔL)={r_e1['rho']:.4f}, "
              f"top3 overlap={r_e1['overlap']}/3, "
              f"{'✓' if r_e1['E1_holds'] else '✗'}")
    if r_e2:
        for p, res in r_e2.get('results', {}).items():
            print(f"  E2 p={p}: {res['energy_by_kmax'].get(3,0)*100:.1f}% at k_max=3, "
                  f"{'✓' if res['E2_holds'] else '✗'}")


if __name__ == "__main__":
    main()
