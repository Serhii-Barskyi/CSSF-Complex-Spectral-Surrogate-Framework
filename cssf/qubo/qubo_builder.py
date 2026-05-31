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
qubo/qubo_builder.py — QUBO matrix construction.

Mathematics (formula 1.10 of the document) — FULL with susceptance:

    Q(x) = Σ_i c_i x_i - Σ_{(i,j)∈E} b_ij x_i x_j + λ(Σ_i x_i - B_max)²

    where c_i = 1 - α·E[LSF_i] - β·MPF_i  (normalized capital costs)

Expanding the penalty term and reducing to the form x^T Q x:

    Q_ii = c_i + λ(1 - 2·B_max)
    Q_ij = λ - b_ij/2   for (i,j) ∈ E among candidates  (susceptance!)
    Q_ij = λ            otherwise

NOTE: b_ij/2 — because the term -b_ij x_i x_j in QUBO with symmetric Q
is written as Q_ij x_i x_j + Q_ji x_j x_i = 2*Q_ij x_i x_j,
so 2*Q_ij = -b_ij → Q_ij = -b_ij/2.
The penalty adds +λ to each off-diagonal element.
Total: Q_ij = λ - b_ij/2.

The penalty λ must be large enough:
    λ > max(c_i) + max(b_ij)/2  → constraint violation is always suboptimal.
Default: λ = 3·max(c_i).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from core.dataset import BESSDataset
from core.csnn_t import CSNNTModel
from core.mpf import MPFResult
from qubo.screener import CandidateResult


@dataclass
class QUBOProblem:
    """
    QUBO problem: Q(x) = x^T Q x, x ∈ {0,1}^K, Σx_i = B_max.

    Attributes
    ----------
    Q          : (K, K) float — QUBO matrix (symmetric)
    c          : (K,) float — linear coefficients c_i (all > 0)
    candidates : (K,) int — global indices of candidate buses
    K          : number of candidates (qubits)
    B_max      : number of BESS units to place
    lam_pen    : penalty parameter λ
    edge_map   : dict {(ki,kj): b_ij} — susceptances between candidates
    case       : case name
    """
    Q:          NDArray[np.floating]
    c:          NDArray[np.floating]
    candidates: List[int]
    K:          int
    B_max:      int
    lam_pen:    float
    edge_map:   Dict[Tuple[int, int], float]
    case:       str

    def energy(self, x: NDArray) -> float:
        """Q(x) = x^T Q x for binary vector x ∈ {0,1}^K."""
        return float(x @ self.Q @ x)

    def is_feasible(self, x: NDArray) -> bool:
        """Checks the constraint Σx_i = B_max."""
        return int(np.round(x.sum())) == self.B_max

    def brute_force(self) -> dict:
        """
        Full enumeration for small K (K ≤ 20).
        Finds the optimal and worst feasible solution.
        """
        from itertools import combinations
        assert self.K <= 20, f"brute_force: K={self.K} > 20"

        best_e, worst_e = np.inf, -np.inf
        best_x, worst_x = None, None

        for combo in combinations(range(self.K), self.B_max):
            x = np.zeros(self.K)
            x[list(combo)] = 1.0
            e = self.energy(x)
            if e < best_e:
                best_e, best_x = e, x.copy()
            if e > worst_e:
                worst_e, worst_x = e, x.copy()

        return {
            'energy_opt':   best_e,
            'energy_worst': worst_e,
            'buses_opt':    [self.candidates[i] for i in range(self.K)
                             if best_x[i] > 0.5],
            'buses_worst':  [self.candidates[i] for i in range(self.K)
                             if worst_x[i] > 0.5],
            'n_configs':    len(list(combinations(range(self.K), self.B_max))),
        }


def build_qubo(
    ds:          BESSDataset,
    candidates:  CandidateResult,
    csnn_model:  CSNNTModel,
    mpf_result:  MPFResult,
    B_max:       int = 1,
    alpha:       float = 1.0,
    beta:        float = 0.1,
    lam_pen:     Optional[float] = None,
) -> QUBOProblem:
    """
    Builds the QUBO matrix according to formula 1.10 of the document.

    Q_ii = c_i + λ(1 - 2·B_max)
    Q_ij = λ - b_ij/2   if (i,j) ∈ E among candidates
    Q_ij = λ            otherwise

    c_i = 1 - α·E[LSF_i] - β·MPF_i

    Parameters
    ----------
    ds         : BESSDataset
    candidates : CandidateResult from screener
    csnn_model : CSNNTModel (for E[LSF])
    mpf_result : MPFResult (for MPF_i)
    B_max      : number of BESS units
    alpha      : LSF weight
    beta       : MPF weight
    lam_pen    : penalty (None = 3·max(c_i))

    Returns
    -------
    QUBOProblem
    """
    K     = candidates.K
    buses = candidates.candidates              # global indices

    # ── Linear coefficients c_i ───────────────────────────────────────────────
    mean_lsf = csnn_model.mean_lsf(ds.X_train)  # (n,)
    mpf_all  = mpf_result.mpf                    # (n,)

    # CORRECT FORMULA: c_i = 1 + alpha*E[LSF_i] - beta*MPF_i
    #
    # Document: minimize Σ c_i x_i. A bus should be attractive
    # (small c_i) if it has large |LSF_i| AND large MPF_i.
    #
    # E[LSF_i] < 0 → alpha*E[LSF_i] = -alpha*|LSF_i| decreases c_i ✓
    # MPF_i > 0 → -beta*MPF_i decreases c_i ✓
    #
    # min c_i = argmax(alpha*|LSF_i| + beta*MPF_i) = argmax score_i
    # Consistent with screener formula: score_i = alpha*|LSF_i| + beta*MPF_i ✓
    c = np.array([
        1.0 + alpha * mean_lsf[bus] - beta * mpf_all[bus]
        for bus in buses
    ])

    # c_i can be negative (good buses with large |LSF_i|) — this is correct
    # Physical meaning: c_i < 0 → QUBO favors placing BESS at this bus

    # ── Penalty ───────────────────────────────────────────────────────────────
    if lam_pen is None:
        lam_pen = float(3.0 * c.max())

    # Verify λ is sufficient: λ > max(c) + max(b)/2
    max_b = max((b for _, _, b in ds.edges), default=0.0)
    if lam_pen < c.max() + max_b / 2:
        lam_pen = float(c.max() + max_b / 2 + 1.0)

    # ── Susceptance between candidates ────────────────────────────────────────
    # edge_map: (ki, kj) → b_ij, ki < kj (indices in candidates, not global)
    bus_to_kidx = {bus: ki for ki, bus in enumerate(buses)}
    edge_map: Dict[Tuple[int, int], float] = {}

    for gi, gj, b in ds.edges:
        ki = bus_to_kidx.get(gi)
        kj = bus_to_kidx.get(gj)
        if ki is not None and kj is not None:
            key = (min(ki, kj), max(ki, kj))
            edge_map[key] = b

    # ── Build Q ───────────────────────────────────────────────────────────────
    Q = np.zeros((K, K))

    # Diagonal: c_i + λ(1 - 2·B_max)
    np.fill_diagonal(Q, c + lam_pen * (1.0 - 2.0 * B_max))

    # Off-diagonal
    for ki in range(K):
        for kj in range(ki + 1, K):
            key = (ki, kj)
            if key in edge_map:
                # Connected candidates: λ - b_ij/2
                val = lam_pen - edge_map[key] / 2.0
            else:
                # Unconnected: λ
                val = lam_pen
            Q[ki, kj] = val
            Q[kj, ki] = val

    return QUBOProblem(
        Q=Q, c=c, candidates=buses,
        K=K, B_max=B_max, lam_pen=lam_pen,
        edge_map=edge_map, case=ds.case,
    )


def verify_qubo(prob: QUBOProblem) -> dict:
    """
    Verification of mathematical properties of the QUBO matrix.

    Returns
    -------
    dict with keys:
        Q_symmetric  : bool
        c_nonneg     : bool
        lam_sufficient: bool — λ > max(c) + max(b)/2
        n_connected  : int — number of connected candidate pairs
        diag_range   : (min, max) of Q diagonal
    """
    K = prob.K
    n_connected = len(prob.edge_map)

    # Is λ sufficient?
    max_b = max(prob.edge_map.values(), default=0.0)
    lam_ok = prob.lam_pen > prob.c.max() + max_b / 2

    return {
        'Q_symmetric':    bool(np.allclose(prob.Q, prob.Q.T, atol=1e-10)),
        'c_nonneg':       bool(np.all(prob.c >= -1e-10)),
        'lam_sufficient': lam_ok,
        'n_connected':    n_connected,
        'diag_range':     (float(np.diag(prob.Q).min()),
                           float(np.diag(prob.Q).max())),
        'lam_pen':        prob.lam_pen,
    }
