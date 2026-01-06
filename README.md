# Spectral Archaeology: The Causal Topology of Model Evolution

> **"Attention geometry retains forensic evidence of training history."**

This repository provides the official implementation for detecting **Passive-Triggered Connectivity Collapse (PTCC)**—a macroscopic "synthetic scar" found in models trained on specific synthetic curricula. Our tools enable training-free model auditing and surgical connectivity repair via graph-spectral analysis.

## 🌟 Key Capabilities

1.  **Forensic Identification:** Achieve perfect separation between model families (e.g., Phi, Llama, Mistral, Qwen) using a single forward pass and fixed Layer 2 smoothness rules.
2.  **PTCC Diagnosis:** Detect catastrophic connectivity failure ($\Delta\lambda_2 \approx -0.76$) triggered by English passive voice in synthetic-heavy models.
3.  **Cross-Linguistic Auditing:** Analyze 10 languages and 12 model lineages to uncover the **Tokenization Topology Law**, which correlates topological regimes with information density rather than language identity.
4.  **Mechanistic Repair:** Surgically restore information flow by steering the sparse "compensatory patch" of induction heads at Layer 2.

## 🚀 Reproducing The Results

We provide modular notebooks to reproduce the core results and figures of the paper.

### 📊 [01_Spectral_Mechanisms.ipynb](notebooks/01_Spectral_Mechanisms.ipynb)
**"The Capacity vs. Simplification Dichotomy"**
- **Goal:** Reproduce Figure 3 (The Recovery Pathways).
- **Insight:** Validates that Activation Steering restores internal capacity, whereas Chain-of-Thought (CoT) merely bypasses connectivity requirements by externalizing reasoning into context.

### 🛠️ [02_Surgical_Repair_Verified.ipynb](notebooks/02_Surgical_Repair_Verified.ipynb)
**"Spectral Surgery and Gain Metrics"**
- **Goal:** Reproduce Table 5 and Table 6 (Intervention Efficacy).
- **Stats:** Generates Perplexity (PPL) reduction data and the 38% connectivity recovery metrics for the Phi-3 lineage.

### 🌍 [03_Cross_Linguistic_Linages.ipynb](notebooks/03_Cross_Linguistic_Linages.ipynb)
**"Developmental Trajectory & Script Gearbox"**
- **Goal:** Reproduce Appendix H results and the "Tokenization Topology Law".
- **Analysis:** Tracks the Phi lineage from "Infancy" (Fragmented) to "Adolescence" (Scarred) to "Maturity" (Adaptive Gearbox).

## 📂 Repository Structure

- `src/`: Our custom `spectral-trust` engine for graph-spectral diagnostics.
- `notebooks/`: Self-contained experiments for reproduction.
- `data/`: Pre-computed spectral statistics for 12 models and 10 languages.
- `figures/`: Raw data and LaTeX-ready visualizations for PTCC signatures.

## 🔧 Installation & Requirements

```bash
# Install the core spectral-trust library and dependencies
pip install -r requirements.txt
```

*Hardware Note:* To reproduce the full Layer 2 ablation and steering experiments, an A100 (80GB VRAM) or equivalent is recommended, though Phi-3-mini experiments can run on a single 16GB GPU.
