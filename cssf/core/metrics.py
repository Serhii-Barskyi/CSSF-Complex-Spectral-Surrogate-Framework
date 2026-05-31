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
core/metrics.py — quality metrics for the CSNN-T^OPF surrogate.

The primary metric is the Pearson correlation between ŷ and y across buses and scenarios,
computed SEPARATELY for five scenario types:
    'normal', 'peak', 'low', 'n1', 'rei'

This is critically important:
- 'rei' — out-of-distribution (load 115–145%), degradation is expected
- 'n1'  — N-1 contingency, LSF may change sign (ρ_n1 ∈ [0.75, 0.94])
- CSNN-T must outperform DC approximation: ρ > rho_dc_vs_ac

ρ_Pearson = Σ(ŷ-ȳ)(y-ȳ) / √(Σ(ŷ-ȳ)² · Σ(y-ȳ)²)

Explained variance is also computed (1 - Var(residual)/Var(y)).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


def pearson_rho(
    y_pred: NDArray[np.floating],
    y_true: NDArray[np.floating],
    buses: Optional[List[int]] = None,
) -> float:
    """
    Pearson correlation between prediction and ground truth.

    Parameters
    ----------
    y_pred : (N, n) or (N*n,) — predicted values
    y_true : (N, n) or (N*n,) — true values
    buses  : list of bus indices to compute over (None = all)

    Returns
    -------
    ρ ∈ [-1, 1]
    """
    if buses is not None:
        y_pred = y_pred[:, buses] if y_pred.ndim == 2 else y_pred
        y_true = y_true[:, buses] if y_true.ndim == 2 else y_true

    yp = y_pred.flatten()
    yt = y_true.flatten()

    if len(yp) < 2:
        return float("nan")

    rho = float(np.corrcoef(yp, yt)[0, 1])
    return rho if np.isfinite(rho) else 0.0


def explained_variance(
    y_pred: NDArray[np.floating],
    y_true: NDArray[np.floating],
    buses: Optional[List[int]] = None,
) -> float:
    """
    Explained variance: 1 - Var(y - ŷ) / Var(y).
    """
    if buses is not None:
        y_pred = y_pred[:, buses]
        y_true = y_true[:, buses]

    residual = y_true - y_pred
    var_res = float(np.var(residual))
    var_y   = float(np.var(y_true))

    if var_y < 1e-15:
        return 1.0
    return float(1.0 - var_res / var_y)


def compute_metrics(
    y_pred: NDArray[np.floating],
    y_true: NDArray[np.floating],
    meta:   NDArray,
    non_slack: List[int],
    rho_dc_vs_ac: float = 0.0,
) -> Dict:
    """
    Full quality metrics for the surrogate.

    Computes ρ and explained_variance:
    - globally (all scenarios, all non_slack buses)
    - separately for each scenario type

    Parameters
    ----------
    y_pred       : (N, n) float — predicted LSF
    y_true       : (N, n) float — true LSF (AC)
    meta         : (N,) str — scenario labels
    non_slack    : list of non-slack bus indices
    rho_dc_vs_ac : ρ(LSF_DC, LSF_AC) from dataset — lower bound

    Returns
    -------
    dict with keys:
        'rho_global'          : float
        'expl_var_global'     : float
        'by_type'             : dict[str → {'rho': float, 'expl_var': float, 'n': int}]
        'beats_dc'            : bool — ρ_global > rho_dc_vs_ac
        'rho_dc_vs_ac'        : float
    """
    scenario_types = ["normal", "peak", "low", "n1", "rei"]

    # Global metrics
    rho_global  = pearson_rho(y_pred, y_true, buses=non_slack)
    expl_global = explained_variance(y_pred, y_true, buses=non_slack)

    # Per scenario type
    by_type = {}
    for stype in scenario_types:
        mask = meta == stype
        if mask.sum() == 0:
            continue
        rho_s  = pearson_rho(y_pred[mask], y_true[mask], buses=non_slack)
        expl_s = explained_variance(y_pred[mask], y_true[mask], buses=non_slack)
        by_type[stype] = {
            "rho":      rho_s,
            "expl_var": expl_s,
            "n":        int(mask.sum()),
        }

    return {
        "rho_global":      rho_global,
        "expl_var_global": expl_global,
        "by_type":         by_type,
        "beats_dc":        rho_global > rho_dc_vs_ac,
        "rho_dc_vs_ac":    rho_dc_vs_ac,
    }
