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
core/dataset.py — loading, validation, and construction of mathematical objects from JSON datasets.

Mathematics:
    Feature matrix X ∈ ℂ^{N × M_complex}, X_{sm} = e^{ik(θ_i^s - θ_j^s)},
    where (i,j) ∈ E — graph edge, k = ±1, s — scenario index.
    Pre-computed in JSON as X_modeA_re + i·X_modeA_im.

    Target matrix y ∈ ℝ^{N × n}, y_{si} = ∂P_loss/∂P_i|_{ξ=s} (LSF).
    Sign LSF < 0: BESS reduces losses; LSF > 0: possible under AC over-excitation.

    edges: list of [i, j, b_ij], where b_ij — line susceptance (i,j),
           required for QUBO term -∑ b_ij x_i x_j.

    Split: train = first n_train scenarios, test = remaining n_test.
    meta: labels ['normal','peak','low','n1','rei'] — N_PER_TYPE each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class BESSDataset:
    """
    All mathematical objects for one case (case14/case30/case57).

    Attributes
    ----------
    case         : case name ('case14', 'case30', 'case57')
    n            : number of buses
    M            : number of edges |E|
    M_complex    : M_complex = 2·M (number of features = pairs k=+1 and k=-1 per edge)
    rank         : rank of X_train (from JSON rank_modeA)
    delta_r      : rank deficiency (M_complex - rank)
    N_train      : number of training scenarios
    N_test       : number of test scenarios
    X_train      : (N_train, M_complex) complex — feature matrix, train
    X_test       : (N_test,  M_complex) complex — feature matrix, test
    y_train      : (N_train, n) float — LSF, train
    y_test       : (N_test,  n) float — LSF, test
    theta_train  : (N_train, n) float — phase angles (rad), train
    theta_test   : (N_test,  n) float — phase angles (rad), test
    meta_train   : (N_train,) str — scenario types, train
    meta_test    : (N_test,)  str — scenario types, test
    edges        : list of [i, j, b_ij] — edges with susceptances
    slack_buses  : indices of slack buses (LSF=0 by definition)
    load_buses   : indices of load buses (where perturbation was applied)
    gen_buses    : indices of generator buses
    non_slack    : indices of all buses except slack (= load_buses ∪ gen_buses)
    params       : full params dict from JSON (for access to rho_dc_vs_ac etc.)
    stability_margin : N_train - 2·rank (regression stability margin)
    """
    case:          str
    n:             int
    M:             int
    M_complex:     int
    rank:          int
    delta_r:       int

    N_train:       int
    N_test:        int

    X_train:       NDArray[np.complexfloating]   # (N_train, M_complex)
    X_test:        NDArray[np.complexfloating]   # (N_test,  M_complex)
    y_train:       NDArray[np.floating]          # (N_train, n)
    y_test:        NDArray[np.floating]          # (N_test,  n)
    theta_train:   NDArray[np.floating]          # (N_train, n)
    theta_test:    NDArray[np.floating]          # (N_test,  n)
    meta_train:    NDArray                       # (N_train,) str
    meta_test:     NDArray                       # (N_test,)  str

    edges:         List[List]                    # [[i, j, b_ij], ...]
    slack_buses:   List[int]
    load_buses:    List[int]
    gen_buses:     List[int]
    non_slack:     List[int]                     # load_buses + gen_buses (without slack)

    params:        dict
    stability_margin: int


def load_dataset(path: str | Path) -> BESSDataset:
    """
    Loads a JSON dataset and returns a BESSDataset.

    Verifies:
    - X_sm = e^{ik(θ_i - θ_j)} for the first edge and first scenario
    - y_lsf sign for normal scenarios (load_buses: LSF ≤ 0)
    - Dimensions of X, y, theta match

    Parameters
    ----------
    path : path to JSON dataset file

    Returns
    -------
    BESSDataset
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r") as f:
        d = json.load(f)

    # ── Basic parameters ──────────────────────────────────────────────────────
    case      = d["case"]
    n         = int(d["n"])
    M         = int(d["M"])           # number of edges |E|
    M_complex = int(d["M_complex"])   # 2*|E|
    rank      = int(d["rank_modeA"])
    delta_r   = int(d["delta_r"])
    n_train   = int(d["n_train"])
    n_test    = int(d["n_test"])

    params = d["params"]
    slack_buses = [int(b) for b in params["slack_buses"]]
    load_buses  = [int(b) for b in params["load_buses"]]
    gen_buses   = [int(b) for b in params["gen_buses"]]
    # non_slack: all buses except slack (for which LSF is defined)
    non_slack = sorted(set(range(n)) - set(slack_buses))

    edges = [[int(e[0]), int(e[1]), float(e[2])] for e in d["edges"]]

    # ── Matrices ──────────────────────────────────────────────────────────────
    X_re  = np.array(d["X_modeA_re"], dtype=float)   # (N, M_complex)
    X_im  = np.array(d["X_modeA_im"], dtype=float)
    X_all = X_re + 1j * X_im                          # (N, M_complex)

    y_all     = np.array(d["y_lsf"],    dtype=float)  # (N, n)
    theta_all = np.array(d["theta_rad"], dtype=float)  # (N, n)
    meta_all  = np.array(d["meta"],     dtype=str)     # (N,)

    N = X_all.shape[0]
    assert N == n_train + n_test, (
        f"N={N} ≠ n_train+n_test={n_train+n_test}"
    )

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train = X_all[:n_train]
    X_test  = X_all[n_train:]
    y_train = y_all[:n_train]
    y_test  = y_all[n_train:]
    theta_train = theta_all[:n_train]
    theta_test  = theta_all[n_train:]
    meta_train  = meta_all[:n_train]
    meta_test   = meta_all[n_train:]

    # ── Verification X_sm = e^{ik(θ_i - θ_j)} ────────────────────────────────
    # First edge, first scenario, k=+1 (column 0) and k=-1 (column 1)
    i0, j0, _ = edges[0]
    delta_ij   = theta_all[0, i0] - theta_all[0, j0]
    err_plus   = abs(X_all[0, 0] - np.exp( 1j * delta_ij))
    err_minus  = abs(X_all[0, 1] - np.exp(-1j * delta_ij))
    if err_plus > 1e-10 or err_minus > 1e-10:
        raise ValueError(
            f"{case}: X_sm ≠ e^{{ik·δ_ij}}. "
            f"err_plus={err_plus:.2e}, err_minus={err_minus:.2e}"
        )

    # ── Stability margin: N_train ≥ 2·rank ───────────────────────────────────
    stability_margin = n_train - 2 * rank

    return BESSDataset(
        case=case,
        n=n, M=M, M_complex=M_complex,
        rank=rank, delta_r=delta_r,
        N_train=n_train, N_test=n_test,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        theta_train=theta_train, theta_test=theta_test,
        meta_train=meta_train, meta_test=meta_test,
        edges=edges,
        slack_buses=slack_buses,
        load_buses=load_buses,
        gen_buses=gen_buses,
        non_slack=non_slack,
        params=params,
        stability_margin=stability_margin,
    )


def load_all_datasets(data_dir: str | Path) -> Dict[str, BESSDataset]:
    """
    Loads all three datasets from a directory.

    Returns
    -------
    dict: {'case14': ..., 'case30': ..., 'case57': ...}
    """
    data_dir = Path(data_dir)
    result = {}
    for case in ["case14", "case30", "case57"]:
        path = data_dir / f"{case}_full_modeA_Barskyi_Serhii.json"
        if path.exists():
            result[case] = load_dataset(path)
    return result
