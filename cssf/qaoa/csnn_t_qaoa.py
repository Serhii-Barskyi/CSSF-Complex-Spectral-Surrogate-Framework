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
qaoa/csnn_t_qaoa.py — CSNN-T^QAOA: surrogate of the energy landscape E(γ,β).

Mathematics (Step 4, formula 1.13, Theorem 3):
    E(γ,β) = Σ_{k∈Λ_p} f̂_p(k) e^{ik·(γ,β)}  — finite trig polynomial on T^{2p}

    Surrogate: h = (X_γ^H X_γ + λI)^{-1} X_γ^H E_vals
    X_γ^{n,m} = e^{i k_m · (γ_n, β_n)},  k_m ∈ Λ_p

    Λ_p = {k ∈ Z^{2p} : ‖k‖_1 ≤ k_max}
    Hypothesis E2: k_max=3 covers >95% of energy at p≤3.

    Speedup: 30^{2p}/M_0 ≈ 1.5×10^7 at p=3, M_0=50.

Barren plateau criterion (formula 1.7):
    ‖PSF_j^β‖²_{L²} = Σ_{k∈Λ_p} k_{p+j}² |f̂_p(k)|² ≥ 0
    Computed analytically from h without additional simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iproduct
from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray

from core.gcv import gcv_lambda, tikhonov_solve


@dataclass
class CSNNTQAOAModel:
    """
    Trained CSNN-T^QAOA model — surrogate of E(γ,β).

    Attributes
    ----------
    h        : (|Λ_p|,) complex — spectral coefficients
    freqs    : (|Λ_p|, 2p) int — frequencies k ∈ Λ_p
    lam_opt  : float
    p        : QAOA depth
    M0       : number of evaluations used
    residual : float
    """
    h:       NDArray[np.complexfloating]
    freqs:   NDArray[np.int_]
    lam_opt: float
    p:       int
    M0:      int
    residual: float

    def predict(self, gamma_beta: NDArray) -> NDArray[np.floating]:
        """
        Prediction of E(γ,β) via trigonometric polynomial.

        Parameters
        ----------
        gamma_beta : (N, 2p) float — points in T^{2p}

        Returns
        -------
        (N,) float
        """
        X = _build_feature_matrix(gamma_beta, self.freqs)
        return np.real(X @ self.h)


def build_frequency_set(p: int, k_max: int = 3) -> NDArray[np.int_]:
    """
    Builds Λ_p = {k ∈ Z^{2p} : ‖k‖_1 ≤ k_max}.

    Hypothesis E2: k_max=3 covers >95% of energy at p≤3.

    Parameters
    ----------
    p     : QAOA depth (number of layers)
    k_max : maximum L1-norm of frequency

    Returns
    -------
    freqs : (|Λ_p|, 2p) int
    """
    dim = 2 * p
    freq_range = range(-k_max, k_max + 1)
    freqs = [
        k for k in iproduct(freq_range, repeat=dim)
        if sum(abs(ki) for ki in k) <= k_max
    ]
    return np.array(freqs, dtype=int)


def _build_feature_matrix(
    gamma_beta: NDArray,
    freqs: NDArray,
) -> NDArray[np.complexfloating]:
    """
    X_{n,m} = e^{i k_m · (γ_n, β_n)}.

    Parameters
    ----------
    gamma_beta : (N, 2p) float
    freqs      : (|Λ_p|, 2p) int

    Returns
    -------
    X : (N, |Λ_p|) complex
    """
    phases = gamma_beta @ freqs.T   # (N, |Λ_p|)
    return np.exp(1j * phases)


def build_csnn_t_qaoa(
    sample_points: NDArray,
    energy_samples: NDArray,
    p: int = 1,
    k_max: int = 3,
    n_lambdas: int = 80,
    lam_range: Tuple[float, float] = (-12, 4),
) -> CSNNTQAOAModel:
    """
    Trains the CSNN-T^QAOA surrogate of the energy landscape E(γ,β).

    h = (X_γ^H X_γ + λI)^{-1} X_γ^H E_vals

    Parameters
    ----------
    sample_points  : (M0, 2p) float — points in T^{2p}
    energy_samples : (M0,) float — E(γ,β) at those points
    p              : QAOA depth
    k_max          : maximum ‖k‖_1

    Returns
    -------
    CSNNTQAOAModel
    """
    freqs = build_frequency_set(p, k_max)
    X = _build_feature_matrix(sample_points, freqs)  # (M0, |Λ_p|)
    y = energy_samples.astype(float)

    lam_opt, _, _ = gcv_lambda(X, y, n_lambdas=n_lambdas, lam_range=lam_range)
    h = tikhonov_solve(X, y, lam_opt).flatten()

    y_pred = np.real(X @ h)
    norm_y = np.linalg.norm(y)
    residual = float(np.linalg.norm(y - y_pred) / (norm_y + 1e-15))

    return CSNNTQAOAModel(
        h=h, freqs=freqs, lam_opt=lam_opt,
        p=p, M0=len(sample_points), residual=residual,
    )


def barren_plateau_criterion(
    model: CSNNTQAOAModel,
    p: int = 1,
) -> Dict:
    """
    Analytical barren plateau criterion (formula 1.7):
        ‖PSF_j^β‖²_{L²} = Σ_{k∈Λ_p} k_{p+j}² |f̂_p(k)|²

    Computed from model coefficients without simulations.
    If ‖PSF_j^β‖² ≈ 0 → barren plateau in β_j.

    Returns
    -------
    dict with keys 'psf_sq', 'barren_flags'
    """
    h_abs2 = np.abs(model.h) ** 2  # (|Λ_p|,)
    freqs  = model.freqs            # (|Λ_p|, 2p)
    total  = h_abs2.sum() + 1e-15

    psf_sq = {}
    for j in range(p):
        # k_{p+j} — component along β_j
        k_beta_j = freqs[:, p + j].astype(float)
        psf_sq[j] = float(np.sum(k_beta_j ** 2 * h_abs2))

    return {
        "psf_sq":         psf_sq,
        "psf_sq_norm":    {j: v / total for j, v in psf_sq.items()},
        "barren_flags":   {j: v / total < 0.01 for j, v in psf_sq.items()},
    }
