# CSSF: Complex Spectral Surrogate Framework

## Complex Spectral Neural Network Surrogates for Stochastic BESS Siting in Transmission Networks: Overcoming DC Approximation Limitations

### An LSF-Informed End-to-End Pipeline for Strategic BESS Placement: From AC OPF Surrogates to QUBO Formulations for QAOA

*By SigmaPublishinQ Team* · [LinkedIn](https://www.linkedin.com/company/sigma-publishinq)

📖 **Full Framework Description:** https://www.kaggle.com/code/serhiibarskyi/aizenberg-technologies-for-optimal-bess-placement

---



Siting BESS at transmission-network buses is a **long-term capital decision** with a 5–20 year horizon [DOE OTC 2026; FERC Order No. 1920, 2024]. The physically correct objective is:

$$\mathbf{x}^{*} = \arg\min_{\mathbf{x}\in\{0,1\}^{n},\;\mathbf{1}^{\top}\mathbf{x}=B_{\max}} \mathbb{E}_{s\sim P}\!\bigl[\Delta L(\mathbf{x},s)\bigr], \qquad \Delta L(\mathbf{x},s)\approx -P_{\text{BESS}}\sum_{i}x_{i}\cdot\mathrm{LSF}_{i}(s)$$

The production standard — **MILP-DC** — replaces this with two sequential approximations: the expectation $\mathbb{E}_{s\sim P}[\cdot]$ is collapsed to a single nominal scenario $P_0$, and the AC loss function is replaced by its DC linearization ($\sin\theta\approx\theta$, $\cos\theta\approx 1$). This substitution solves a **structurally different problem**: its solution may diverge from $\mathbf{x}^*$ even with unlimited computation.

The DC approximation error scales as $O(s^3)$ with load factor $s$ — verified numerically with exponent $\alpha=2.986$, $R^2=0.9999$. Moving from nominal load $s=1.15$ to overload $s=1.45$, ranking error grows by $(1.45/1.15)^3\approx 27\times$. Precisely these modes — peak renewable integration, post-contingency dispatch, heat-wave demand — are most critical for long-term BESS planning.

**CSSF is designed for this regime.** CSNN-T is trained on full AC OPF data and retains ranking accuracy under extreme loads where DC-LSF degrades cubically. The result: $\tau_\text{CSSF}=0.987$ versus $\tau_\text{SR/MILP-DC}=0.761$ on IEEE-118-bus at OOD scenarios $s\sim U(1.32, 1.50)$ — zero overlap with training.

---

## Project Overview

CSSF is a hybrid classical-quantum framework for optimal placement of Battery Energy Storage Systems (BESS) in power grids. Its central element is **CSNN-T** — a single-hidden-layer complex-valued neural network of the Extreme Learning Machine class with fixed hidden layer $\phi_m(\theta)=e^{ik_m\cdot\theta}$ over the torus $\mathbb{T}^{|E|}$ and trained complex output weights $H^*\in\mathbb{C}^{M\times n}$, obtained in one closed-form step:

```math
H^{*}=(X^{\mathrm{H}}X+\lambda^* I)^{-1}X^{\mathrm{H}}Y_{\mathrm{LSF}}
```
The stochastically-averaged LSF vector $`\hat{\mathbb{E}}_P[\mathrm{LSF}_i]`$ produced by CSNN-T is the **LSF-informed end-to-end invariant** connecting all five pipeline stages — from AC OPF regression to qubit rotation angles:

$$\underbrace{H^* \leftarrow Y_{\text{LSF}}}_{\text{T1: CSNN-T}} \;\to\; \underbrace{\text{score}_i = \alpha|\mathbb{E}[\text{LSF}_i]| + \beta\,\text{MPF}_i}_{\text{T2: MPF-Screener}} \;\to\; \underbrace{c_i = 1 + \alpha\,\mathbb{E}[\text{LSF}_i] - \beta\,\text{MPF}_i}_{\text{T3: QUBO}} \;\to\; \underbrace{R_x\!\left(2\beta\,|\text{LSF}_i|\right)}_{\text{T5: LSF-mixer}}$$

MILP-DC produces none of these components in a unified architecture.

---

## Key Results

### OOD Ranking Accuracy (Kendall τ)

Trained on $s\sim U(0.85, 1.15)$, tested on strict OOD scenarios $s\sim U(1.32, 1.50)$ — **zero distribution overlap**:

| System | τ (CSSF) | τ (SR / MILP-DC†) | Δτ | CI (Bootstrap 2000) |
|--------|----------|-------------------|----|----------------------|
| IEEE 14-bus | **0.974** | 0.872 | +0.103 | Bootstrap, 2000 |
| IEEE 30-bus | **0.990** | 0.862 | +0.128 | Bootstrap, 2000 |
| IEEE 57-bus | **0.995** | 0.897 | +0.097 | Bootstrap, 2000 |
| IEEE 118-bus | **0.987** | 0.761 | +0.226 | [0.979, 0.993] |

† $\tau_\text{MILP-DC} \equiv \tau_\text{SR}$ by definition: both methods rank all $n$ buses by DC-LSF at the nominal scenario $P_0$. The DC approximation error at OOD ($s>1.30$) degrades both identically — $\rho_\text{DC}\in[0.349, 0.684]$ vs $\rho_\text{CSNN}\in[0.947, 0.9998]$ on the same test set.

### Active Loss Reduction ΔL_AC (MW) — Verified by Full AC OPF

BESS capacity 5 MW per unit. Experiment S9-B, PandaPower Newton–Raphson:

| System | B | p | Scenario | CSSF ΔL | SR ΔL | MILP-DC ΔL | Winner |
|--------|---|---|----------|---------|-------|------------|--------|
| IEEE-30 | 2 | 2 | OOD mean | **+1.008 MW** | +0.955 MW | +0.979 MW | **CSSF** |
| IEEE-30 | 2 | 2 | s=1.45 | **+1.135 MW** | +1.092 MW | +1.097 MW | **CSSF** |
| IEEE-14 | 3 | 2 | OOD mean | **+2.839 MW** | +2.839 MW | +2.689 MW | **CSSF** |
| IEEE-14 | 3 | 2 | s=1.45 | **+3.055 MW** | +3.055 MW | +2.896 MW | **CSSF** |

### Topological Safety Index (TSI) — N-1 Resilience

Fraction of selected buses with nodal degree ≥ 2 (degree-1 bus isolated from network under N-1 line outage). TSI is an **emergent property** of the LSF+MPF composite score — no explicit topological filter in the code:

| System | B | CSSF buses | CSSF TSI | SR buses | SR TSI | MILP-DC buses | MILP-DC TSI |
|--------|---|------------|----------|----------|--------|----------------|-------------|
| IEEE-118 | 2 | [108, 107] | **1.00** | [111, 40] | 0.50 | [111, 40] | 0.50 |
| IEEE-30 | 2 | [18, 17] | **1.00** | [29, 25] | 0.50 | [18, 16] | 1.00 |
| IEEE-14 | 3 | [7, 13, 2] | **0.67** | [7, 13, 2] | 0.67 | [13, 12, 9] | **1.00** |

Across all 16 experiment configurations (4 systems × $B\in\{2,3\}$ × $p\in\{1,2\}$): CSSF selects a degree-1 bus in **1/16** configurations, SR in **12/16**, MILP-DC in **4/16**.

On IEEE-118 — the most operationally significant system — CSSF achieves TSI=1.00 while both SR and MILP-DC invariably include bus 111 (degree=1), which becomes fully isolated under an N-1 line outage.

### Pipeline Speedup vs AC OPF Scanning

| System | Speedup (CSNN-T inference vs single AC OPF) |
|--------|----------------------------------------------|
| IEEE 14-bus | **4.6M×** |
| IEEE 30-bus | **7.2M×** |
| IEEE 57-bus | **8.0M×** |
| IEEE 118-bus | **3.6M×** |

Single matrix multiplication $O(M\cdot n)$: 0.003–0.020 ms inference vs 11–97 s per AC OPF solve. This makes stochastic evaluation of $\mathbb{E}_P[\mathrm{LSF}_i]$ over 400 scenarios computationally trivial.

---

## 📈 Performance Summary

| Metric | CSSF | SR | MILP-DC |
|--------|------|----|---------|
| Kendall τ — IEEE 14-bus (OOD) | **0.974** | 0.872 | ≡ SR = 0.872† |
| Kendall τ — IEEE 30-bus (OOD) | **0.990** | 0.862 | ≡ SR = 0.862† |
| Kendall τ — IEEE 57-bus (OOD) | **0.995** | 0.897 | ≡ SR = 0.897† |
| Kendall τ — IEEE 118-bus (OOD) | **0.987** | 0.761 | ≡ SR = 0.761† |
| ρ CSNN-T surrogate (OOD test) | **[0.947, 0.9998]** | — | ρ_DC = 0.35–0.68 |
| ΔL_AC — IEEE-30, B=2, OOD mean | **+1.008 MW** | +0.955 MW | +0.979 MW |
| ΔL_AC — IEEE-14, B=3, OOD mean | **+2.839 MW** | +2.839 MW | +2.689 MW |
| TSI — IEEE-118, B=2 | **1.00** | 0.50 | 0.50 |
| Degree-1 bus selections (16 configs) | **1/16** | 12/16 | 4/16 |
| Speedup vs AC scan | **3.6M–8.0M×** | — | — |
| CSNN-T^QAOA vs grid search (p=1) | **36× fewer evals** | — | — |
| CSNN-T^QAOA vs grid search (p=2) | **6,279× fewer evals** | — | — |
| Multi-scenario coverage | **5 scenario types** | 1 nominal | 1 nominal |

† $\tau_\text{MILP-DC}\equiv\tau_\text{SR}$ by mathematical definition: both rank all $n$ buses by DC-LSF at a single nominal scenario $P_0$. The DC approximation error at OOD ($s>1.30$) degrades both identically, which is precisely the limitation CSSF is designed to overcome.

---

## Honest Disclosure: Boundaries of Quantum Advantage

In compliance with DOE OTC 2026 requirements, quantum advantage boundaries are explicitly stated:

- At $K=13$ (IEEE-14/30/57-bus), classical enumeration of $2^{13}=8{,}192$ states is faster than QAOA — **quantum advantage does not exist in this regime**
- The quantum advantage threshold begins at $K\geq 50$, corresponding to networks with $n\geq 300$ buses
- For Phase 3 (IBM Heron r2) at $K=20$: circuit depth 210, shots 8,192, state space $2^{20}=1{,}048{,}576$ (GPU verification)
- COBYLA on CSNN-T^QAOA surrogate and GP+EI reach identical solutions; LSF-weighted mixer reduces COBYLA iterations but does not change the optimum

---

## 🛠️ Technologies

- **CSNN-T:** complex spectral neural network — Tikhonov regularization, GCV $\lambda^*$ selection, closed-form training
- **QAOA LSF-mixer:** original LSF-weighted mixer (`H_M = Σᵢ |LSF_i|·Xᵢ`, gate `Rₓ(2·β·|LSFᵢ|)`)
- **QUBO / Ising transform:** automatic assembly, error verified $<10^{-12}$
- **GP+EI Bayesian optimization:** Gaussian process with Matérn-2.5 kernel, $M_0=50$ circuit evaluations
- **Qiskit / IBM Heron r2:** quantum execution (Phase 3, $K=20$, depth=210, shots=8,192)
- **PandaPower / PyPower:** AC/DC OPF computations and training data generation (Newton–Raphson)
- **Pyomo + HiGHS:** MILP-DC power flow (DOE-recommended baseline)
- **Google Colab / GPU (NVIDIA A4):** development and execution environment

---

## 📋 Requirements

### Environment

- Google Colab (GPU required for Phase 3, $K=20$) or local Python environment
- Python 3.8+

### Dependencies

```bash
# Quantum
pip install qiskit==1.2.4 qiskit-aer-gpu==0.14.2
# Power-systems
pip install pandapower==3.4.0 numba
# MILP solver (DOE-listed: HiGHS)
pip install pyomo>=6.4.2 highspy
# ML / optimisation
pip install scikit-learn scipy
```

---

## 🚀 Quick Start

### Production Benchmark

```bash
# Open CSSF_Production_Benchmark.ipynb in Google Colab
# Execute all cells sequentially
```

---

## 📁 Project Structure

### Production Benchmark

```
├── CSSF_Production_Benchmark.ipynb   # End-to-end benchmark: CSSF vs SR vs MILP-DC
```

### Full CSSF Framework

```
cssf/
├── core/
│   ├── csnn_t.py          # CSNN-T: H* = (X^H X + λI)^{-1} X^H Y
│   ├── gcv.py             # GCV λ* selection
│   └── mpf.py             # MPF: (B⁺)_{ii} = Σ v_m[i]² / λ_m
├── qubo/
│   ├── screener.py        # Top-K: score_i = α|LSF_i| + β·MPF_i
│   ├── qubo_builder.py    # QUBO assembly, λ sufficiency verified
│   └── ising.py           # Ising transform, error < 1e-12
├── qaoa/
│   ├── hamiltonian.py     # LSF-weighted mixer Hamiltonian
│   ├── circuit.py         # QAOA circuit + COBYLA optimizer
│   ├── csnn_t_qaoa.py     # QAOA energy surrogate (|Λ_p|=25/129/377 at p=1/2/3)
│   └── gp_optimizer.py    # GP+EI Bayesian optimization
└── config.py              # Centralized configuration
```

---

## 🗂️ Dataset

**[AC Loss Sensitivity Factor — cases 14/30/57/118](https://www.kaggle.com/datasets/serhiibarskyi/ac-loss-sensitivity-factor-cases-143057118)**
*(Kaggle — open access)*

Ground-truth AC Loss Sensitivity Factors for four standard IEEE benchmark networks, computed via full nonlinear AC OPF (PandaPower Newton–Raphson) across diverse operating conditions including N-1 contingency scenarios.

| System | Buses | Lines | $N_\text{train}$ | $N_\text{test}$ (OOD) | OOD range |
|--------|-------|-------|------------------|------------------------|-----------|
| IEEE 14-bus | 14 | 20 | 280 | 70 | $s\sim U(1.32, 1.50)$ |
| IEEE 30-bus | 30 | 41 | 280 | 70 | $s\sim U(1.32, 1.50)$ |
| IEEE 57-bus | 57 | 80 | 400 | 100 | $s\sim U(1.32, 1.50)$ |
| IEEE 118-bus | 118 | 186 | 400 | 100 | $s\sim U(1.32, 1.50)$ |

Training scenarios: $s\sim U(0.85, 1.15)$ — normal, peak, low-load, N-1 contingency (70–100 per type). OOD scenarios: $s\sim U(1.32, 1.50)$ — **zero distribution overlap** with training.

Why this dataset is valuable:
- **Ground-truth AC accuracy:** LSFs from full nonlinear AC OPF, not DC approximation ($\rho_\text{DC}=0.35$–$0.68$ vs $\rho_\text{CSNN}=0.947$–$0.9998$)
- **Strict OOD split:** zero overlap between training and test distributions
- **N-1 contingency coverage:** single-element outages in all training scenarios
- **Reproducible benchmark:** deterministic generation pipeline, JSON metadata for every split

---

## What Has Been Verified

| Statement | Experiment | Status |
|-----------|------------|--------|
| $\tau_\text{CSNN}=0.974$–$0.995$ vs $\tau_\text{SR/MILP-DC}=0.862$–$0.897$ (OOD, IEEE-14/30/57) | Bootstrap CI, 2000 samples | ✅ |
| $\tau_\text{CSNN}=0.987$ vs $\tau_\text{SR/MILP-DC}=0.761$, CI [0.979, 0.993] (OOD, IEEE-118) | Bootstrap CI | ✅ |
| $\rho_\text{CSNN}\in[0.947, 0.9998]$ vs $\rho_\text{DC}\in[0.35, 0.68]$ | Test set | ✅ |
| $\Delta L_{AC}$: CSSF +1.008 MW vs SR +0.955 MW vs MILP-DC +0.979 MW (IEEE-30, $B=2$, $p=2$) | Experiment S9-B | ✅ |
| TSI=1.00 on IEEE-118 $B=2$; SR and MILP-DC TSI=0.50 | Experiment S9-B | ✅ |
| Degree-1 bus selections: CSSF 1/16, SR 12/16, MILP-DC 4/16 | Experiment S9 | ✅ |
| $Q(\mathbf{x})=H_C(\mathbf{z})+\text{const}$, error $<10^{-12}$ | Configuration enumeration | ✅ |
| DC error exponent $\alpha=2.986$, $R^2=0.9999$ (cubic scaling) | Lemma C verification | ✅ |
| CSNN-T^QAOA: $36\times$ ($p=1$), $6{,}279\times$ ($p=2$) vs grid search | Theorem 3 + GP+EI | ✅ |
| 1.61× COBYLA acceleration (LSF vs uniform mixer, 10 seeds, case57) | Experiment S9 | ✅ |
| Quantum advantage absent at $K=13$; threshold at $K\geq 50$ | Theoretical + S9 | ⏳ Phase 3 |

---

## 🎯 Applications

### Strategic BESS Placement

- Multi-scenario stochastic optimization: 5 scenario types vs 1 nominal for SR/MILP-DC
- N-1 contingency-aware bus ranking (emergent from LSF+MPF score)
- AI datacenter load integration support

### Quantum Hardware Scaling (Phase 3)

- IBM Heron r2: $K=20$ qubits, depth=210, shots=8,192
- State space $2^{20}=1{,}048{,}576$; GPU verification (NVIDIA A4)

---

## 🔮 Future Work

- Extension to IEEE 300-bus and real-world transmission networks ($n\geq300$ where $K\geq50$ unlocks quantum advantage)
- Full QAOA execution on quantum devices (IBM, IonQ)
- N-1 security constraints natively embedded in QUBO
- Online adaptation to shifting load distributions (AI datacenter demand spikes)

---

## 📄 License

Apache License 2.0 — see LICENSE file for details.

---

## 🙏 Acknowledgments

- Professor Igor Aizenberg https://scholar.google.com/citations?hl=en&user=ZjfN_9AAAAAJ
- Potomac Quantum Innovation Center https://www.pqic.org/
- Aqora https://aqora.io/
- Connected DMV https://www.connecteddmv.org/
- qBraid https://www.qbraid.com/
- Google Colab platform

---


## 📞 Contact

**Serhii Barskyi**
Data Scientist (Spectral Methods) / Quantum Optimization Specialist
|ψ⟩⊗|φ⟩ QAOA | QUBO | Spectral Analysis | Fourier-based ML |⇅〉 Qiskit | Django REST | Energy, Logistics, Industrial AIoT | Fourier Neural Operators

https://preply.com/en/tutor/7756455

https://www.linkedin.com/in/serhii-barskyi/

https://www.linkedin.com/company/sigma-publishinq

---

*Note: This project is optimized for Google Colab (GPU) but can be adapted for local execution and deployment on quantum hardware.*
