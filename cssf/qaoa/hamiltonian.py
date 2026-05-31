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
qaoa/hamiltonian.py — construction of QAOA quantum Hamiltonians.

Mathematics:

Problem Hamiltonian (from Ising):
    H_C = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j

LSF-weighted mixer (our contribution, Theorem 3):
    H_M = Σ_i LSF_i · X_i

    spec(H_M) = {Σ_i s_i LSF_i : s_i = ±1}  (Theorem 3)
    This determines the admissible frequencies of landscape E(γ,β) on T^{2p}.

Analytical barren plateau criterion (formula 1.7):
    ‖PSF_j^β‖²_{L²} = Σ_{k∈Λ_p} k_{p+j}² |f̂_p(k)|²
    Computed from CSNN-T^QAOA coefficients without simulations.

Implementation:
    e^{-iγH_C}  via Rz(2γh_i) and RZZ(2γJ_ij)
    e^{-iβH_M}  via Rx(2β·LSF_i)  ← LSF-weighted, not uniform!
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from qubo.ising import IsingProblem


@dataclass
class HamiltonianSpec:
    """
    Hamiltonian specification for QAOA.

    Attributes
    ----------
    h_vec      : (K,) float — local fields of H_C
    J_mat      : (K, K) float — interaction matrix (J_ii=0)
    lsf_w      : (K,) float — normalized LSF weights for the mixer
    lsf_raw    : (K,) float — unnormalized |E[LSF_i]| for candidate buses
    K          : number of qubits
    B_max      : number of BESS units
    candidates : list of global bus indices
    case       : case name
    """
    h_vec:      NDArray[np.floating]
    J_mat:      NDArray[np.floating]
    lsf_w:      NDArray[np.floating]
    lsf_raw:    NDArray[np.floating]
    K:          int
    B_max:      int
    candidates: List[int]
    case:       str


def build_hamiltonian(
    ising:    IsingProblem,
    lsf_raw:  NDArray[np.floating],
) -> HamiltonianSpec:
    """
    Assembles HamiltonianSpec from an Ising problem and LSF weights.

    LSF weights are normalized to [0,1] by their maximum for numerical
    stability during β optimization. Normalization does not change the
    spectral structure of H_M — only the scale of β.

    Parameters
    ----------
    ising   : IsingProblem
    lsf_raw : (K,) float — |E[LSF_i]| for candidate buses (unnormalized)

    Returns
    -------
    HamiltonianSpec
    """
    lsf_max = float(lsf_raw.max())
    assert lsf_max > 1e-15, "LSF weights are all zero"
    lsf_w = lsf_raw / lsf_max   # normalize to [0,1]

    return HamiltonianSpec(
        h_vec=ising.h.copy(),
        J_mat=ising.J.copy(),
        lsf_w=lsf_w,
        lsf_raw=lsf_raw.copy(),
        K=ising.K,
        B_max=ising.B_max,
        candidates=list(ising.candidates),
        case=ising.case,
    )


def mixer_spectrum(lsf_w: NDArray, K: int) -> NDArray:
    """
    Theoretical spectrum of H_M = Σ_i LSF_i X_i (Theorem 3):
        spec(H_M) = {Σ_i s_i LSF_i : s_i ∈ {-1,+1}}

    Returns array of all 2^K values (with repetitions).
    """
    from itertools import product
    return np.array([
        sum(s * lsf_w[i] for i, s in enumerate(signs))
        for signs in product([-1, 1], repeat=K)
    ])
