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
qubo/screener.py — candidate screening for BESS placement.

Mathematics (document, Step 2-3):
    score_i = α·|E_ξ[LSF_i]| + β·MPF_i

    Buses are ranked in descending order of score_i.
    Top-K candidates are selected among non_slack buses.

    Physical meaning:
    - α·|E[LSF_i]|: operational value (loss reduction)
    - β·MPF_i: topological value (structural importance)
    - |corr(LSF, MPF)| ≈ 0.09-0.47 → two independent criteria

    Combinatorial space reduction:
    case57: C(56,5) ≈ 3.8×10^6 → C(22,5) ≈ 2.6×10^4 (at K=22)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from core.dataset import BESSDataset
from core.csnn_t import CSNNTModel
from core.mpf import MPFResult


@dataclass
class CandidateResult:
    """
    Candidate screening result.

    Attributes
    ----------
    candidates   : (K,) int — global indices of candidate buses,
                   sorted in descending order of score
    scores       : (K,) float — score_i = α|LSF_i| + β·MPF_i
    lsf_scores   : (K,) float — α·|E[LSF_i]|
    mpf_scores   : (K,) float — β·MPF_i
    K            : number of selected candidates
    case         : case name
    """
    candidates:  List[int]
    scores:      NDArray[np.floating]
    lsf_scores:  NDArray[np.floating]
    mpf_scores:  NDArray[np.floating]
    K:           int
    case:        str


def screen_candidates(
    ds:         BESSDataset,
    csnn_model: CSNNTModel,
    mpf_result: MPFResult,
    K:          int,
    alpha:      float = 1.0,
    beta:       float = 0.1,
) -> CandidateResult:
    """
    Selects top-K candidate buses by combined LSF+MPF ranking.

    score_i = α·|E_ξ[LSF_i]| + β·MPF_i

    Slack buses are excluded (LSF=0 by definition).

    Parameters
    ----------
    ds          : BESSDataset
    csnn_model  : trained CSNNTModel (for E[LSF])
    mpf_result  : MPFResult (for MPF_i)
    K           : number of candidates
    alpha       : LSF criterion weight
    beta        : MPF criterion weight

    Returns
    -------
    CandidateResult
    """
    assert 1 <= K <= len(ds.non_slack), (
        f"K={K} out of range [1, {len(ds.non_slack)}]"
    )

    ns = np.array(ds.non_slack)                   # global non-slack indices

    # E_ξ[LSF_i] over train scenarios via surrogate
    mean_lsf = csnn_model.mean_lsf(ds.X_train)   # (n,) — all buses

    # score for non_slack buses
    lsf_part = alpha * np.abs(mean_lsf[ns])       # (len(ns),)
    mpf_part = beta  * mpf_result.mpf[ns]         # (len(ns),)
    scores   = lsf_part + mpf_part                # (len(ns),)

    # Top-K in descending order
    top_idx  = np.argsort(-scores)[:K]            # indices in ns
    top_buses = ns[top_idx].tolist()

    return CandidateResult(
        candidates=top_buses,
        scores=scores[top_idx],
        lsf_scores=lsf_part[top_idx],
        mpf_scores=mpf_part[top_idx],
        K=K,
        case=ds.case,
    )
