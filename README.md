# CSSF: Complex Spectral Surrogate Framework

**Quantum-Enhanced Strategic Siting of Battery Energy Storage Systems (BESS)**

*By SigmaPublishinQ Team* · [LinkedIn](https://www.linkedin.com/company/sigma-publishinq)

📖 **Full Framework Description:** https://www.kaggle.com/code/serhiibarskyi/aizenberg-technologies-for-optimal-bess-placement

---

CSSF is a hybrid classical-quantum framework for optimal placement of Battery Energy Storage Systems (BESS) in power grids. At its core is the **CSNN-T** surrogate (Complex Spectral Neural Network on the Torus), a Tikhonov-regularized regression on the toric basis of the AC power flow manifold **𝕋^{n−1}**, replacing expensive AC OPF computations with an analytic spectral approximation. QAOA parameters are predicted via the **CSNN-T^QAOA** surrogate, while the BESS placement problem is formulated as a QUBO and solved using the original **LSF-weighted mixer**. The framework achieves speedups from **4.6M×** to **8.1M×** over AC scanning, with ranking quality exceeding the industry-standard Sensitivity Ranking across all IEEE test systems.

---

## 🚀 Project Overview

CSSF consists of two main components:

### 1. CSSF Production Benchmark ⚡

A reproducible end-to-end pipeline — from dataset generation to QUBO/QAOA optimization and three-way comparative analysis (CSSF vs. Sensitivity Ranking vs. MILP DC). Benchmarked on IEEE 14/30/57/118-bus systems under strict OOD conditions (load scenarios with zero overlap with the training distribution).

### 2. Full CSSF Framework ⚡⚡⚡

A scalable multi-system platform with modular architecture, GPU support, Bayesian optimization (GP+EI), quantum execution on IBM Heron r2, and extended spectral analysis — including Theorem 2 (spectral sparsity of LSF), Theorem 3 (QAOA landscape frequency structure), and Lemma C (OOD error scaling).

---

## 📊 Key Features

- **Spectral surrogate modeling:** CSNN-T fits a Tikhonov-regularized model in the toric Fourier basis of the AC power flow manifold, capturing the analytic structure of LSF via edge harmonics {±e_ij}
- **Quantum optimization:** QUBO + QAOA with the original LSF-weighted mixer
- **Strict OOD regime:** testing on load scenarios U(1.32, 1.50) — zero overlap with training peak U(1.10, 1.30)
- **Multi-system benchmarking:** IEEE 14, 30, 57, 118-bus with N-1 contingency
- **Scalability:** up to K=20 batteries, C(20,5)=15,504 configurations, statevector 2²⁰=1M (GPU)
- **Honest disclosure:** DOE-compliant documentation with explicit statements on the boundaries of quantum advantage
- **Theorem verification:** numerical evidence for spectral structure theorems (Theorems 1, 2, 3 / Lemma C)

---

## 🛠️ Technologies

- **CSNN-T:** complex spectral neural network (Tikhonov regularization, GCV λ selection)
- **QAOA LSF-mixer:** original LSF-weighted mixer (`H_M = Σᵢ LSF_i · Xᵢ`, gate `Rₓ(2·β·LSFᵢ)`)
- **QUBO / Ising transform:** automatic assembly of the BESS placement problem
- **GP+EI Bayesian Opt:** Gaussian surrogate optimization of QAOA parameters (M₀=50 circuit evaluations)
- **Qiskit / IBM Heron r2:** quantum execution (Phase 3, K=20, depth=210, shots=8192)
- **PyPower / Pandapower:** AC/DC OPF computations and training data generation
- **Pyomo + HiGHS:** MILP DC power flow (DOE-recommended baseline)
- **Google Colab / GPU:** development and execution environment

---

## 📋 Requirements

### Environment

- Google Colab (GPU recommended) or local Python environment
- Python 3.8+

### Dependencies

```bash
# Quantum
pip('qiskit==1.2.4', 'qiskit-aer-gpu==0.14.2')
# Power-systems
pip('pandapower==3.4.0', 'numba')
# Classical MILP solver (DOE-listed: HiGHS)
pip('pyomo>=6.4.2', 'highspy')
# ML / optimisation
pip('scikit-learn', 'scipy')
```

---

## 🚀 Quick Start

### Production Benchmark

**Clone the Repository**

```bash
# In Google Colab
# Copy to: /content/drive/MyDrive/cssf/
```


**Run the Benchmark**

```bash
# Open CSSF_Production_Benchmark.ipynb in Google Colab
# Execute all cells sequentially
```

### Expected Results

| System | Kendall τ (CSSF) | Kendall τ (SR) | Δ | Speedup vs AC Scan |
|--------|-----------------|----------------|---|---------------------|
| IEEE 14-bus | **0.974** | 0.872 | +0.103 | **4.6M×** |
| IEEE 30-bus | **0.990** | 0.862 | +0.128 | **7.2M×** |
| IEEE 57-bus | **0.995** | 0.897 | +0.097 | **8.1M×** |
| IEEE 118-bus | **0.987** | 0.761 | +0.226 | **4.7M×** |

---

## 📁 Project Structure

### Production Benchmark

```
├── CSSF_Production_Benchmark.ipynb          # Extended benchmark notebook (+ theorem verification blocks)

```

### Full CSSF Framework

```
cssf/
├── core/
│   ├── csnn_t.py              # CSNN-T surrogate: h = (X^H X + λI)^{-1} X^H y
│   ├── gcv.py                 # GCV λ selection
│   └── mpf.py                 # Moore-Penrose Factor: MPF_i = (B⁺)_{ii}
├── qubo/
│   ├── screener.py            # Top-K bus selection: score_i = α|LSF_i| + β·MPF_i
│   ├── qubo_builder.py        # QUBO matrix assembly
│   └── ising.py               # Ising transform (const verified < 1e-12)
├── qaoa/
│   ├── hamiltonian.py         # LSF-weighted mixer Hamiltonian
│   ├── circuit.py             # QAOA circuit + COBYLA optimizer
│   ├── csnn_t_qaoa.py         # QAOA energy surrogate (25 frequencies, p=1)
│   └── gp_optimizer.py        # GP+EI Bayesian optimization
└── config.py                  # Centralized configuration
```

---

## 📈 Performance Metrics

| Metric | CSSF | SR (Sensitivity Ranking) | MILP DC |
|--------|------|--------------------------|---------| 
| Kendall τ — IEEE 14-bus | **0.974** | 0.872 | — |
| Kendall τ — IEEE 30-bus | **0.990** | 0.862 | — |
| Kendall τ — IEEE 57-bus | **0.995** | 0.897 | — |
| Kendall τ — IEEE 118-bus | **0.987** | 0.761 | — |
| ρ CSNN-T surrogate (mean) | **0.999** | — | 0.35–0.68 |
| Speedup vs AC scan | **4.6M–8.1M×** | — | — |
| CSNN-T^QAOA vs grid search | **36× fewer evals** | — | — |
| Multi-scenario coverage | **5 scenario types** | 1 nominal | — |

---

## 🗂️ Dataset

**[AC Loss Sensitivity Factor — cases 14/30/57/118](https://www.kaggle.com/datasets/serhiibarskyi/ac-loss-sensitivity-factor-cases-143057118)**
*(Kaggle — open access)*

This dataset provides **ground-truth AC Loss Sensitivity Factors (LSF)** for the four standard IEEE benchmark networks used in CSSF: 14-bus, 30-bus, 57-bus, and 118-bus systems. All data are generated via full nonlinear AC power flow simulations across diverse operating conditions, including N-1 contingency scenarios.

**What's included:**

| System | Buses | Lines | N_train | N_test | OOD scenarios |
|--------|-------|-------|---------|--------|---------------|
| IEEE 14-bus | 14 | 20 | 280 | 70 | U(1.32–1.50) |
| IEEE 30-bus | 30 | 41 | 280 | 70 | U(1.32–1.50) |
| IEEE 57-bus | 57 | 80 | 400 | 100 | U(1.32–1.50) |
| IEEE 118-bus | 118 | 186 | 400 | — | U(1.32–1.50) |

**Why this dataset is valuable:**

- **Ground-truth AC accuracy:** LSFs computed from full nonlinear AC OPF (not DC approximation), providing significantly higher fidelity than DC-derived sensitivity factors commonly used in the literature (ρ_DC = 0.35–0.68 vs. ρ_CSSF ≈ 0.999)
- **Strict OOD split:** test scenarios drawn from U(1.32, 1.50) — zero distribution overlap with training peak U(1.10, 1.30) — enabling genuine out-of-distribution generalization benchmarking
- **N-1 contingency coverage:** all scenarios include single-element outages, reflecting real operational stress conditions required by grid reliability standards
- **Multi-system standardization:** four canonical IEEE test cases in a unified format, enabling direct cross-system comparison and ML model evaluation
- **Reproducible benchmark:** fully deterministic generation pipeline; JSON metadata included for every split — making it a plug-and-play foundation for ML-for-power-systems research

---

## 🧪 Testing

### Input Data

- IEEE 14/30/57/118-bus topologies (PyPower)
- N-1 contingency scenarios
- OOD load: U(1.32, 1.50) — strict out-of-distribution regime

### Output Data

- Bus ranking for BESS placement
- Auto-generated JSON files with reproducible metrics
- Comparative tables: CSSF vs SR vs MILP DC

### Verification

After running `CSSF_Production_Benchmark.ipynb`, the notebook output will display:
- Kendall τ across all systems and methods
- ρ CSNN-T (train/test reported separately)
- Timing metrics and JSON reports

---

## 🔬 Advanced Features

### CSNN-T Surrogate (OPF)

- Complex spectral approximation of AC power flow distribution on the toric manifold **𝕋^{n−1}**
- Tikhonov regularization with GCV λ selection (optimal: λ=10⁻¹²)
- Matrix ranks: 35 (case14) → 138 (case57)
- Stability margins: 124–210

### LSF-Weighted QAOA Mixer (Original)

- Physically motivated mixer: `H_M = Σᵢ LSF_i · Xᵢ`
- Gate: `Rₓ(2·β·LSFᵢ)` — adaptive rotation by bus sensitivity
- Achieves deeper exploration of the optimization landscape compared to uniform mixing
- LSF-mixer converges in mean **55.7 COBYLA iterations** vs. 89.8 for uniform mixer (1.61× speedup), both reaching r = 1.000

### Theorem 3: Spectral Structure of the QAOA Landscape

- `E(γ,β) = Σ_{k∈Λ_p} f̂(k)·e^{ik·(γ,β)}`
- |Λ_p|=25/129/377 frequencies at p=1/2/3, k_max=3
- LSF-weighted mixer expands β-frequency support to **|W₁| = 2K²+2K+1 = 365** vs. 5 for uniform mixer (**73× richer**)
- CSNN-T^QAOA surrogate operates in Fourier space

### GP+EI Bayesian Optimization

- GP prior: CSNN-T^QAOA surrogate
- Expected Improvement acquisition function
- M₀=50 circuit evaluations — **36× more efficient** than grid search at p=1

---

## 🎯 Applications

### Strategic BESS Placement

Optimal selection of K buses for battery installation, minimizing load loss (ΔL):
- Multi-scenario analysis (5 scenario types vs. 1 nominal for SR)
- N-1 contingency-aware ranking
- AI datacenter load integration support

### Grid Security Assessment

- OPF surrogate for rapid operational state evaluation in near real time
- Robust testing under strict OOD load scenarios

### Quantum Hardware Scaling (Phase 3)

- IBM Heron r2: 20 qubits, depth=210, shots=8192
- K=20 batteries, C(20,5)=15,504 configurations

---

## 🔮 Future Prospects

- **Scalability:** extension to IEEE 300-bus and real-world transmission networks
- **Quantum integration:** full QAOA execution on quantum devices (IBM, IonQ)
- **Dynamic scenarios:** online adaptation to changing load graphs
- **AI datacenters:** integration of non-stochastic load spikes into the OPF surrogate
- **Real-world deployment:** industrial AIoT, energy logistics networks

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Professor Igor Aizenberg https://scholar.google.com/citations?hl=en&user=ZjfN_9AAAAAJ&view_op=list_works&sortby=pubdate
- Quantum computing community
- Google Colab platform
- Potomac Quantum Innovation Center https://www.pqic.org/
- Aqora https://aqora.io/
- Connected DMV https://www.connecteddmv.org/
- qBraid https://www.qbraid.com/

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
