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
experiments/milp_baseline.py — MILP-DC baseline for DOE comparison.

Implements three BESS placement methods:

1. CSSF (ours): CSNN-T surrogate + QUBO optimization (brute force for K≤20)
2. SR: Sensitivity Ranking — industry standard (LSF screening)
3. MILP-DC: Mixed Integer Linear Program with DC power flow
   Solver: Pyomo + HiGHS (DOE-recommended open-source solver)

MILP-DC mathematics:
    min  Σ_{(i,j)∈E} g_ij · l_ij                     (DC losses, L1 surrogate)
    s.t. Σ_j b_ij (θ_i - θ_j) = P_i^{inj} + P^{BESS} x_i  (DC power balance)
         θ_slack = 0                                   (reference bus)
         l_ij ≥ +(θ_i - θ_j)                          (L1 linearisation)
         l_ij ≥ -(θ_i - θ_j)
         Σ_i x_i = B_max                              (exactly B_max BESS)
         x_i ∈ {0,1}                                   (binary placement)

Usage:
    python experiments/milp_baseline.py
    python experiments/milp_baseline.py --cases case14 case30 --b_max 1 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset import load_dataset
from core.csnn_t import fit_csnn_t
from core.mpf import compute_mpf
from qubo.screener import screen_candidates
from qubo.qubo_builder import build_qubo
from qubo.ising import qubo_to_ising

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


def _get_network_data(case: str):
    """Returns network data: edges, P_nom, n_buses, slack."""
    import pandapower.networks as pn
    import pandapower as pp

    net_fn = {'case14': pn.case14, 'case30': pn.case30, 'case57': pn.case57}
    net    = net_fn[case]()
    pp.runpp(net, algorithm='nr', numba=False)

    n      = len(net.bus)
    slack  = int(net.ext_grid['bus'].iloc[0])
    sn_mva = float(net.sn_mva)

    # Edges: (from, to, b_pu, g_pu) — lines AND transformers
    edges = []
    for _, line in net.line.iterrows():
        fi    = int(line['from_bus'])
        ti    = int(line['to_bus'])
        vn    = float(net.bus.loc[fi, 'vn_kv'])
        base_z = vn**2 / sn_mva
        x_pu  = float(line['x_ohm_per_km'] * line['length_km']) / base_z
        r_pu  = float(line.get('r_ohm_per_km', 0.01) * line['length_km']) / base_z
        b_pu  = 1.0 / max(x_pu, 1e-9)
        g_pu  = r_pu / max(x_pu**2 + r_pu**2, 1e-12)
        edges.append((fi, ti, b_pu, g_pu))

    # Transformers as edges (DC approximation: b = 1/x_pu)
    for _, trafo in net.trafo.iterrows():
        fi    = int(trafo['hv_bus'])
        ti    = int(trafo['lv_bus'])
        vn    = float(net.bus.loc[fi, 'vn_kv'])
        base_z = vn**2 / sn_mva
        # x_pu = vk_percent / 100 * vn_kv^2 / sn_mva / sn_trafo_mva
        sn_t   = float(trafo['sn_mva'])
        vk     = float(trafo['vk_percent']) / 100.0
        x_pu   = vk * vn**2 / (sn_t * base_z) if sn_t > 0 else 1e-3
        b_pu   = 1.0 / max(x_pu, 1e-6)
        g_pu   = 0.0   # transformer without active losses in DC
        edges.append((fi, ti, b_pu, g_pu))

    # Nominal injections P_inj = P_gen - P_load (per-unit)
    P_nom = np.zeros(n)
    for _, load in net.load.iterrows():
        P_nom[int(load['bus'])] -= float(load['p_mw']) / sn_mva
    for _, gen in net.gen.iterrows():
        P_nom[int(gen['bus'])] += float(gen['p_mw']) / sn_mva
    # ext_grid (slack) absorbs imbalance — its P is not set

    return {
        'edges':   edges,
        'P_nom':   P_nom,
        'n':       n,
        'slack':   slack,
        'sn_mva':  sn_mva,
        'net':     net,
    }


def solve_milp_dc(
    case:       str,
    candidates: List[int],
    B_max:      int,
    bess_mw:    float = 5.0,
    load_scale: float = 1.0,
    verbose:    bool = False,
) -> Dict:
    """
    MILP-DC via Pyomo + HiGHS.

    Formulation:
        min  Σ g_ij · l_ij
        s.t. DC power balance (all buses except slack)
             θ_slack = 0
             L1: l_ij ≥ ±(θ_i - θ_j)
             Σ x_i = B_max,  x_i ∈ {0,1}

    Returns dict: buses_opt, obj_val, solve_time, status, n_vars
    """
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    nd    = _get_network_data(case)
    edges = nd['edges']
    P_nom = nd['P_nom'].copy() * load_scale
    n     = nd['n']
    slack = nd['slack']
    sn    = nd['sn_mva']

    non_slack = [i for i in range(n) if i != slack]
    CANDS_IDX = list(range(len(candidates)))
    EDGES_IDX = list(range(len(edges)))

    m = pyo.ConcreteModel()
    m.x      = pyo.Var(CANDS_IDX, domain=pyo.Binary)
    m.theta  = pyo.Var(range(n),  domain=pyo.Reals,
                       bounds=(-np.pi/2, np.pi/2))
    m.loss   = pyo.Var(EDGES_IDX, domain=pyo.NonNegativeReals)

    # Objective: minimize DC losses (L1 surrogate)
    m.obj = pyo.Objective(
        expr=sum(g * m.loss[e] for e, (fi, ti, b, g) in enumerate(edges))
    )

    # Constraints
    m.balance = pyo.ConstraintList()
    m.lpos    = pyo.ConstraintList()
    m.lneg    = pyo.ConstraintList()

    # DC power balance: Σ B_ij(θ_i-θ_j) = P_i + BESS_i  (for non-slack)
    for i in non_slack:
        bess_i = 0.0
        if i in candidates:
            k = candidates.index(i)
            bess_i = (bess_mw / sn) * m.x[k]

        net_inj = P_nom[i] + bess_i
        net_flow = sum(
            b * (m.theta[fi] - m.theta[ti]) for fi, ti, b, g in edges if fi == i
        ) + sum(
            b * (m.theta[ti] - m.theta[fi]) for fi, ti, b, g in edges if ti == i
        )
        m.balance.add(net_flow == net_inj)

    # Slack: θ_slack = 0
    m.slack_ref = pyo.Constraint(expr=m.theta[slack] == 0.0)

    # L1 linearisation of losses
    for e, (fi, ti, b, g) in enumerate(edges):
        m.lpos.add(m.loss[e] >=  (m.theta[fi] - m.theta[ti]))
        m.lneg.add(m.loss[e] >= -(m.theta[fi] - m.theta[ti]))

    # Exactly B_max BESS
    m.bmax = pyo.Constraint(
        expr=sum(m.x[k] for k in CANDS_IDX) == B_max
    )

    # Solve
    solver = SolverFactory('appsi_highs')
    if not solver.available():
        raise RuntimeError("appsi_highs not found. Install pyomo>=6.4: pip install pyomo")

    t0     = time.time()
    result = solver.solve(m, tee=verbose)
    dt     = time.time() - t0
    status = str(result.solver.termination_condition)

    if 'optimal' not in status.lower() and 'feasible' not in status.lower():
        return {
            'buses_opt': [], 'obj_val': None,
            'solve_time': dt, 'status': status,
            'n_vars': len(CANDS_IDX) + n + len(EDGES_IDX),
        }

    x_vals    = np.array([pyo.value(m.x[k]) for k in CANDS_IDX])
    buses_opt = [candidates[k] for k in CANDS_IDX if x_vals[k] > 0.5]
    obj_val   = float(pyo.value(m.obj)) * sn   # MW

    return {
        'buses_opt':  buses_opt,
        'obj_val':    obj_val,
        'solve_time': dt,
        'status':     status,
        'n_vars':     len(CANDS_IDX) + n + len(EDGES_IDX),
        'n_binary':   len(CANDS_IDX),
    }


def solve_sr(ds, B_max: int) -> Dict:
    """Sensitivity Ranking: argmax |E[LSF_i]| over train data."""
    ns   = np.array(ds.non_slack)
    mlsf = ds.y_train[:, ns].mean(axis=0)
    top  = ns[np.argsort(-np.abs(mlsf))[:B_max]].tolist()
    return {'buses_opt': top, 'method': 'SR_sensitivity_ranking'}


def measure_delta_L(
    case:    str,
    buses:   List[int],
    bess_mw: float = 5.0,
    scaling: float = 1.0,
) -> Optional[float]:
    """ΔL = L0 - L_bess (MW) via pandapower AC OPF."""
    import pandapower as pp
    import pandapower.networks as pn
    net_fn = {'case14': pn.case14, 'case30': pn.case30, 'case57': pn.case57}

    def run_safe(net):
        for alg in ['nr', 'iwamoto_nr', 'bfsw']:
            try:
                pp.runpp(net, algorithm=alg, numba=False)
                return True
            except Exception:
                pass
        return False

    net0 = net_fn[case]()
    net0.load['p_mw']   *= scaling
    net0.load['q_mvar'] *= scaling
    if not run_safe(net0):
        return None
    L0 = float(net0.res_line['pl_mw'].sum())

    net1 = net_fn[case]()
    net1.load['p_mw']   *= scaling
    net1.load['q_mvar'] *= scaling
    for b in buses:
        pp.create_sgen(net1, bus=b, p_mw=bess_mw, q_mvar=0.0, name='BESS')
    if not run_safe(net1):
        return None
    return L0 - float(net1.res_line['pl_mw'].sum())


def run_comparison(
    case:       str,
    b_max_list: List[int],
    bess_mw:    float = 5.0,
    scenarios:  Optional[List[Tuple[str, float]]] = None,
) -> Dict:
    """Runs CSSF vs SR vs MILP-DC for the given case."""
    if scenarios is None:
        scenarios = [
            ('low',     0.70),
            ('normal',  1.00),
            ('peak',    1.15),
            ('ood_130', 1.30),
            ('ood_145', 1.45),
        ]

    print(f"\n{'='*65}")
    print(f"COMPARISON: {case}")
    print(f"Methods: CSSF (QUBO/brute-force) | SR (LSF heuristic) | MILP-DC (HiGHS)")
    print(f"{'='*65}")

    path = DATA_DIR / f"{case}_full_modeA_Barskyi_Serhii.json"
    if not path.exists():
        print(f"  Dataset not found: {path}")
        return {}

    ds    = load_dataset(path)
    mdl   = fit_csnn_t(ds)
    mpf_r = compute_mpf(ds.n, ds.edges, ds.slack_buses)
    K     = min(13, len(ds.non_slack))

    results = {}

    for B_max in b_max_list:
        print(f"\n  B_max={B_max}:")

        # CSSF
        cands  = screen_candidates(ds, mdl, mpf_r, K=K)
        prob   = build_qubo(ds, cands, mdl, mpf_r, B_max=B_max)
        bf     = prob.brute_force()
        buses_cssf = bf['buses_opt']

        # SR
        buses_sr = solve_sr(ds, B_max)['buses_opt']

        # MILP-DC
        try:
            milp = solve_milp_dc(case, cands.candidates, B_max, bess_mw)
            buses_milp = milp['buses_opt'] if milp['buses_opt'] else buses_sr
            milp_time  = milp['solve_time']
            milp_ok    = bool(milp['buses_opt'])
            milp_obj   = milp['obj_val']
        except Exception as e:
            print(f"    MILP error: {e}")
            buses_milp = buses_sr
            milp_time  = 0.0
            milp_ok    = False
            milp_obj   = None

        methods = {
            'CSSF_QUBO': buses_cssf,
            'SR':        buses_sr,
            'MILP_DC':   buses_milp,
        }

        print(f"    CSSF   : {buses_cssf}")
        print(f"    SR     : {buses_sr}")
        milp_info = f"ok={milp_ok}, obj={milp_obj:.4f}MW, t={milp_time:.2f}s" if milp_ok else "infeasible→fallback to SR"
        print(f"    MILP-DC: {buses_milp} ({milp_info})")

        # Measure ΔL
        dL = {m: [] for m in methods}
        print(f"\n    {'Scenario':<12} {'CSSF':>9} {'SR':>9} {'MILP_DC':>9}")
        print(f"    {'-'*42}")
        for sname, sc in scenarios:
            row = {}
            for mname, buses in methods.items():
                v = measure_delta_L(case, buses, bess_mw, sc)
                row[mname] = v
                if v is not None:
                    dL[mname].append(v)
            def fmt(v): return f"{v:>+9.4f}" if v is not None else "      N/A"
            print(f"    {sname:<12} {fmt(row['CSSF_QUBO'])} {fmt(row['SR'])} {fmt(row['MILP_DC'])}")

        means = {m: np.mean(v) if v else 0.0 for m, v in dL.items()}
        print(f"    {'Mean (MW)':<12} {means['CSSF_QUBO']:>+9.4f} {means['SR']:>+9.4f} {means['MILP_DC']:>+9.4f}")
        best = max(means, key=means.get)
        print(f"\n    Best by mean ΔL: {best}")

        results[B_max] = {
            'buses':       {m: methods[m] for m in methods},
            'mean_dL_mw':  means,
            'best_method': best,
            'milp_time_s': milp_time,
            'milp_ok':     milp_ok,
        }

    return {'case': case, 'results': results}


def main():
    parser = argparse.ArgumentParser(
        description='CSSF vs SR vs MILP-DC production benchmark'
    )
    parser.add_argument('--cases',   nargs='+', default=['case14'],
                        choices=CASES)
    parser.add_argument('--b_max',   nargs='+', type=int, default=[1, 3])
    parser.add_argument('--bess_mw', type=float, default=5.0)
    args = parser.parse_args()

    print("="*65)
    print("CSSF PRODUCTION BENCHMARK: CSSF vs SR vs MILP-DC")
    print("Classical solver: Pyomo + HiGHS (DOE-recommended, open-source)")
    print("Industry heuristic: SR — Sensitivity Ranking (LSF-based)")
    print("Reference: Ajagekar & You, Energy 2019, DOE OTC Challenge 2026")
    print("="*65)

    all_results = {}
    for case in args.cases:
        r = run_comparison(case, args.b_max, args.bess_mw)
        if r:
            all_results[case] = r

    # Summary table
    print(f"\n{'='*65}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*65}")
    print(f"{'Case':<8} {'B_max':>6} {'CSSF(MW)':>10} {'SR(MW)':>8} {'MILP_DC(MW)':>12} {'Best':>12}")
    print("-"*58)
    for case, cr in all_results.items():
        for B_max, br in cr['results'].items():
            d = br['mean_dL_mw']
            print(f"{case:<8} {B_max:>6} {d.get('CSSF_QUBO',0):>+10.4f} "
                  f"{d.get('SR',0):>+8.4f} {d.get('MILP_DC',0):>+12.4f} "
                  f"{br['best_method']:>12}")

    return all_results


if __name__ == "__main__":
    main()
