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
config.py — central hyperparameter registry of the BESS Placement framework.

All modules import parameters from here — single point of change.
Mathematical notation corresponds to the document aizenberg_technologies_for_bess_placement.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).parent
DATA_DIR  = ROOT_DIR / "data"

import kagglehub
# Download latest version
_kaggle_path = kagglehub.dataset_download("serhiibarskyi/ac-loss-sensitivity-factor-cases-143057118")
print("Path to dataset files:", _kaggle_path)

import pathlib as _pl
DATASETS = {
    "case14":  _pl.Path(_kaggle_path) / "case14_full_modeA_Barskyi_Serhii.json",
    "case30":  _pl.Path(_kaggle_path) / "case30_full_modeA_Barskyi_Serhii.json",
    "case57":  _pl.Path(_kaggle_path) / "case57_full_modeA_Barskyi_Serhii.json",
    "case118": _pl.Path(_kaggle_path) / "case118_full_modeA_Barskyi_Serhii.json",
}

# ── QUBO hyperparameters (formula 1.10) ──────────────────────────────────────

ALPHA   = 1.0    # weight of LSF criterion: alpha * E[LSF_i]
BETA    = 0.1    # weight of MPF criterion: beta * MPF_i
LAMBDA_PEN = None  # penalty coefficient (None = auto: 3 * max(c_k))

# Default K for each case (number of BESS units to place)
K_DEFAULT = {
    "case14": 3,   # binom(11,3)=165 — full enumeration in seconds
    "case30": 3,
    "case57": 5,
    "case118": 5,  # binom(20,5)=15504 — QUBO/QAOA required
}

# ── QAOA parameters ──────────────────────────────────────────────────────────

QAOA_P_DEFAULT  = 1       # circuit depth p (number of cost+mixer layers)
QAOA_SHOTS      = 1024    # number of measurements (for qasm backend)
QAOA_K_MAX      = 22      # maximum number of qubits on statevector simulator
QAOA_OPTIMIZER  = "COBYLA"  # parameter optimizer (COBYLA / SPSA)
QAOA_SEED       = 42

# Limits by case (N_q qubits, statevector feasibility)
# case14:  N_q=11,  dim=2048       ✓
# case30:  N_q=20,  dim≈10^6      ✓ (slow)
# case57:  N_q=42,  dim=4*10^12   — statevector infeasible → CSNN-T^QAOA only
# case118: N_q=20,  dim=2^20≈10^6 ✓ GPU required (K=20, C(20,5)=15504 configs)

# ── CSNN-T^QAOA parameters (Step 4, formula 1.13) ───────────────────────────

CSNN_QAOA_M0     = 50    # number of evaluations for surrogate training
CSNN_QAOA_K_MAX  = 3     # ||k||_1 <= k_max for frequency set Λ_p
                          # Hypothesis E2: covers >95% of energy for p<=3

# ── GCV parameters (formula 2.12) ───────────────────────────────────────────

GCV_N_LAMBDAS  = 100     # number of points on the logarithmic grid λ
GCV_LAM_RANGE  = (-12, 4)  # (log10_min, log10_max) for λ

# ── Candidate screener ───────────────────────────────────────────────────────

SCREENER_N_CANDIDATES = 22   # maximum number of candidates for QAOA (≤ QAOA_K_MAX)

# ── Scenario types and split ─────────────────────────────────────────────────

SCENARIO_TYPES = ["normal", "peak", "low", "n1", "rei"]
# rei = out-of-distribution (REI = Random Equipment Influence)
# n1  = N-1 contingency (single line outage)

# ── Random seeds ─────────────────────────────────────────────────────────────

RANDOM_SEED = 42
