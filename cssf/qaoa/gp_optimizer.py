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
qaoa/gp_optimizer.py — Gaussian Process + EI for QAOA landscape optimization.

Mathematics (formula 1.13, Step 4):
    (γ*,β*) = argmax_{γ,β} EI(γ,β; μ_CSNN, k_Matérn-5/2)

    EI(x) = E[max(0, f* - f(x))]  — Expected Improvement

    Matérn-5/2 kernel corresponds to twice-differentiable functions —
    which exactly matches the trigonometric polynomial E(γ,β) (Theorem 3).

    GP operates on residuals: r(x) = E(x) - μ_CSNN(x)
    where μ_CSNN is the prior from CSNN-T^QAOA.

    Speedup: M0=50 evaluations instead of 30^{2p} (≈ 1.5×10^7 at p=3).

Hypothesis H2:
    GP+CSNN-T prior achieves target r with M0 < M0^COBYLA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from qaoa.hamiltonian import HamiltonianSpec
from qaoa.circuit import qaoa_quality_metric


@dataclass
class GPResult:
    """
    GP landscape optimization result.

    Attributes
    ----------
    gamma_opt   : (p,) optimal parameters
    beta_opt    : (p,) optimal parameters
    energy_opt  : minimum energy found
    r_quality   : quality metric r ∈ [0,1]
    M0_used     : actual number of evaluations
    strategy    : 'csnn_prior' | 'zero_prior'
    """
    gamma_opt:  NDArray[np.floating]
    beta_opt:   NDArray[np.floating]
    energy_opt: float
    r_quality:  float
    M0_used:    int
    strategy:   str


def optimize_landscape_gp(
    cost_fn:    Callable[[NDArray], float],
    ham:        HamiltonianSpec,
    p:          int = 1,
    M0:         int = 50,
    k_max:      int = 3,
    seed:       int = 42,
    strategy:   str = 'csnn_prior',
    energy_opt: Optional[float] = None,
    energy_worst: Optional[float] = None,
) -> GPResult:
    """
    Optimizes E(γ,β) via GP with CSNN-T^QAOA prior and EI.

    Algorithm:
        1. n_init=10 random evaluations
        2. Train CSNN-T^QAOA surrogate (if strategy='csnn_prior')
        3. Bayesian Optimization iterations (M0-n_init steps):
           a. GP fit on residuals r(x) = E(x) - μ_CSNN(x)
           b. Maximize EI → next point
           c. Evaluate cost_fn → update data
        4. Return best point found

    Parameters
    ----------
    cost_fn      : callable(params: ndarray(2p,)) → float
    ham          : HamiltonianSpec (for LSF weights and candidates)
    p            : QAOA depth
    M0           : evaluation budget
    k_max        : ‖k‖_1 ≤ k_max for CSNN-T^QAOA
    seed         : random seed
    strategy     : 'csnn_prior' or 'zero_prior'
    energy_opt   : optimal energy (for r-metric)
    energy_worst : worst energy (for r-metric)

    Returns
    -------
    GPResult
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
    except ImportError:
        raise ImportError("scikit-learn: pip install scikit-learn")

    from qaoa.csnn_t_qaoa import build_csnn_t_qaoa

    rng = np.random.default_rng(seed)
    dim = 2 * p  # dimensionality of T^{2p}

    # ── Initial sample ────────────────────────────────────────────────────────
    n_init = min(10, M0 // 5)
    pts    = rng.uniform(0, 2 * np.pi, (n_init, dim))
    E_vals = np.array([cost_fn(pt) for pt in pts])

    best_e  = float(E_vals.min())
    best_pt = pts[int(E_vals.argmin())].copy()

    # ── CSNN-T^QAOA surrogate (prior) ─────────────────────────────────────────
    csnn_model = None
    if strategy == 'csnn_prior' and n_init >= 5:
        try:
            csnn_model = build_csnn_t_qaoa(pts, E_vals, p=p, k_max=k_max)
        except Exception:
            csnn_model = None

    # ── Bayesian Optimization ─────────────────────────────────────────────────
    for step in range(n_init, M0):
        # Prior mean
        if csnn_model is not None:
            mu_prior = csnn_model.predict(pts)
        else:
            mu_prior = np.zeros(len(pts))

        residuals = E_vals - mu_prior

        # GP fit on residuals
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6,
            normalize_y=True, n_restarts_optimizer=2,
        )
        try:
            gp.fit(pts, residuals)
        except Exception:
            next_pt = rng.uniform(0, 2 * np.pi, dim)
        else:
            next_pt = _maximize_ei(gp, csnn_model, best_e, dim,
                                   n_restarts=5, rng=rng)

        # Evaluate
        next_e = cost_fn(next_pt)
        pts    = np.vstack([pts, next_pt[None]])
        E_vals = np.append(E_vals, next_e)

        if next_e < best_e:
            best_e  = next_e
            best_pt = next_pt.copy()

        # Update CSNN surrogate every 10 steps
        if csnn_model is not None and (step + 1) % 10 == 0:
            try:
                csnn_model = build_csnn_t_qaoa(pts, E_vals, p=p, k_max=k_max)
            except Exception:
                pass

    # ── Result ────────────────────────────────────────────────────────────────
    gamma_opt = best_pt[:p]
    beta_opt  = best_pt[p:]

    r = 0.0
    if energy_opt is not None and energy_worst is not None:
        r = qaoa_quality_metric(best_e, energy_opt, energy_worst)

    return GPResult(
        gamma_opt=gamma_opt,
        beta_opt=beta_opt,
        energy_opt=best_e,
        r_quality=r,
        M0_used=M0,
        strategy=strategy,
    )


def _maximize_ei(
    gp,
    csnn_model,
    best_energy: float,
    dim:         int,
    n_restarts:  int,
    rng:         np.random.Generator,
) -> NDArray:
    """
    Maximizes EI(x) = E[max(0, f* - f(x))].

    f(x) = μ_CSNN(x) + GP_residual(x)
    """
    from scipy.stats import norm
    from scipy.optimize import minimize as sp_minimize

    def neg_ei(x: NDArray) -> float:
        x2 = x.reshape(1, -1)
        mu_csnn = float(csnn_model.predict(x2)[0]) if csnn_model else 0.0
        try:
            mu_gp, sigma = gp.predict(x2, return_std=True)
            mu_total = mu_csnn + float(mu_gp[0])
            sigma_val = float(sigma[0]) + 1e-9
        except Exception:
            return 0.0

        z  = (best_energy - mu_total) / sigma_val
        ei = sigma_val * (z * norm.cdf(z) + norm.pdf(z))
        return -float(ei)

    best_val = np.inf
    best_x   = rng.uniform(0, 2 * np.pi, dim)

    for _ in range(n_restarts):
        x0     = rng.uniform(0, 2 * np.pi, dim)
        bounds = [(0, 2 * np.pi)] * dim
        res    = sp_minimize(neg_ei, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 50, 'ftol': 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_x   = res.x

    return best_x
