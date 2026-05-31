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
core/mpf.py — Modal Participation Factor (MPF) via Laplacian spectrum.

Mathematics (Step 2, formula 1.9):
    MPF_i = (B+)_ii = Σ_m v_m[i]² / λ_m  (diagonal of the pseudo-inverse Laplacian)

    B = D - A — weighted network Laplacian:
        D_ii = Σ_j b_ij  (diagonal of susceptances)
        A_ij = b_ij       (adjacency matrix)

    Eigendecomposition: B V = V diag(λ), λ_0=0 (excluded)

    imp_m — NOT used. Correct formula: MPF_i = (B+)_ii = Σ v_m[i]²/λ_m
    (rationale: large eigenvalues correspond to high-frequency
     local modes with high participation in power flows)

    Physical meaning: MPF_i characterises the topological importance of bus i
    independently of the operating regime.
    |corr(MPF, LSF)| ≈ 0.04–0.16 — informational complementarity.

    The screener uses a combined ranking:
        score_i = α·|E[LSF_i]| + β·MPF_i
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray


@dataclass
class MPFResult:
    """
    MPF computation result.

    Attributes
    ----------
    mpf        : (n,) float — Modal Participation Factor for each bus
    eigenvals  : (n-1,) float — non-zero eigenvalues of B (λ_1,...,λ_{n-1})
    eigenvecs  : (n, n-1) float — corresponding eigenvectors
    B          : (n, n) float — weighted Laplacian
    """
    mpf:       NDArray[np.floating]
    eigenvals: NDArray[np.floating]
    eigenvecs: NDArray[np.floating]
    B:         NDArray[np.floating]


def build_laplacian(n: int, edges: List[List]) -> NDArray[np.floating]:
    """
    Builds the weighted Laplacian B = D - A from an edge list.

    B_ii = Σ_j b_ij   (sum of susceptances of incident edges)
    B_ij = -b_ij      (negative susceptance of edge (i,j))

    Parameters
    ----------
    n     : number of buses
    edges : list of [i, j, b_ij]

    Returns
    -------
    B : (n, n) float
    """
    B = np.zeros((n, n), dtype=float)
    for i, j, b in edges:
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b
    return B


def compute_mpf(
    n: int,
    edges: List[List],
    slack_buses: List[int],
) -> MPFResult:
    """
    Computes MPF_i = Σ_m v_m[i]² / λ_m · imp_m for all buses.

    Algorithm:
        1. Build B from edges
        2. Eigendecomp: B V = V diag(λ), sort by λ
        3. Exclude λ_0 = 0 (zero eigenvector — constant mode)
        4. MPF_i = (B+)_ii = Σ_m v_m[i]² / λ_m  (diagonal of B+, no imp_m)

    Parameters
    ----------
    n           : number of buses
    edges       : list of [i, j, b_ij]
    slack_buses : indices of slack buses (MPF=0 not enforced, but they have
                  low participation due to the structure of B)

    Returns
    -------
    MPFResult
    """
    B = build_laplacian(n, edges)

    # Eigendecomp: eigh for symmetric matrix (guarantees real λ)
    eigenvals_all, eigenvecs_all = np.linalg.eigh(B)

    # Sort by ascending λ (eigh already sorts)
    # Exclude zero eigenvalue λ_0 ≈ 0
    pos_mask = eigenvals_all > 1e-8
    lam = eigenvals_all[pos_mask]          # (n-1,) or fewer
    V   = eigenvecs_all[:, pos_mask]       # (n, n-1)

    assert len(lam) > 0, "No non-zero eigenvalues of B"

    # CORRECT FORMULA: MPF_i = (B+)_ii = Σ_m v_m[i]² / λ_m
    #
    # BUG in previous version: imp_m = λ_m/Σλ gave
    # MPF_i = Σ_m v_m[i]²*(λ_m/Σλ)/λ_m = Σ_m v_m[i]²/Σλ = const ∀i
    # (algebraically constant, zero discriminating power).
    #
    # Correct: MPF_i = (B+)_ii = diagonal of the pseudo-inverse Laplacian
    # = effective graph resistance = commute time between i and uniform
    # = Σ_m v_m[i]² / λ_m  (without imp_m)
    # Physically: large MPF_i ↔ bus is weakly connected to the network → important for BESS.
    mpf = np.zeros(n)
    for i in range(n):
        mpf[i] = float(np.sum(V[i, :] ** 2 / lam))

    # Normalise MPF to [0, 1]
    mpf_max = mpf.max()
    if mpf_max > 1e-15:
        mpf = mpf / mpf_max

    return MPFResult(
        mpf=mpf,
        eigenvals=lam,
        eigenvecs=V,
        B=B,
    )
