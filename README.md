# Multilingual Business Entity Resolution & Record Linkage System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ML%20Stack-PyTorch%20%7C%20LightGBM%20%7C%20Transformers-orange.svg)](https://pytorch.org/)
[![Evaluation](https://img.shields.io/badge/Macro%20F0.5%20Score-99.09%25%20(OOF)-success.svg)]()
[![Test Benchmark](https://img.shields.io/badge/Leaderboard%20Score-98.68%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, high-precision Machine Learning pipeline for large-scale **Business Entity Resolution (BER)** and deduplication across heterogeneous, noisy data sources without shared identifiers. 

> **Project Context:**  
> Developed as an **Academic Machine Learning Mini Project**, leveraging the benchmark data and real-world industrial constraints from the *Amazon ML Challenge 2026 (Business Entity Resolution)*. The system tackles extreme real-world challenges: native Indic script transliteration, OCR/spelling noise, legal entity variations, and zero-shot transfer to previously unseen countries (France).

---

## 🎯 Project Overview & Key Results

Entity matching in industrial databases is critical for deduplication, fraud detection, and master data management. When databases lack primary keys, records must be linked using noisy attributes such as business names, addresses, and postal codes.

This system achieves state-of-the-art precision and recall using a multi-stage architecture: **high-throughput inverted-index candidate blocking**, **hybrid feature engineering**, **fine-tuned Transformer cross-encoders**, **gradient-boosted trees (LightGBM)**, and **Bayesian expected-F₀.₅ decision decoding**.

### Key Performance Benchmarks

| Metric / Stage | Performance | Description |
| :--- | :---: | :--- |
| **Top Validation Score (Macro F₀.₅)** | **99.09%** (`0.99093`) | Out-Of-Fold (OOF) score on multi-source entity deduplication |
| **Public Test Benchmark** | **98.68%** (`0.98676`) | Evaluated on out-of-distribution test sets (including unseen countries) |
| **Pipeline A Baseline** | **98.77%** (`0.98765`) | Standalone LightGBM + XLM-RoBERTa Cross-Encoder pipeline |
| **Candidate Blocking Pair Coverage** | **98.25%** | Two-channel TF-IDF inverted index covering >98.2% true matches |
| **Candidate Search Space Reduction** | **>99.9%** | Reduces billions of pairwise comparisons to ~69 candidates per entity |

*Note on Metric:* The primary metric is **Macro-Averaged F₀.₅**, which weights **Precision 4× higher than Recall**. In business record linkage, false positives (falsely merging two separate legal companies) cause catastrophic data corruption, making precision-heavy optimization paramount.

---

## 🏗️ System Architecture

The pipeline is structured into four decoupled, memory-efficient stages:

```
                  Raw Multi-Source Data (Source 1, Source 2, Source 3)
                                          │
    ┌─────────────────────────────────────┴─────────────────────────────────────┐
    │                                                                           │
    ▼                                                                           ▼
[Stage 1: Normalization]                                            [Stage 2: Candidate Blocking]
• Sound-alike skeleton hashing (NYSIIS/Metaphone)                   • Two-channel TF-IDF inverted indices (fw + hh)
• Indic native script transliteration                               • Within-country search space pruning
• Legal suffix canonicalization (Pvt Ltd ↔ Private Limited)         • N-gram & word cosine similarity (top-K = 10)
• Landmark stripping & zero-padded number normalization             • Pair coverage: 98.25% (153.5M candidate pool)
    │                                                                           │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
                                          ▼
[Stage 3: Hybrid Feature Extraction & Multilingual Cross-Encoders]
• 40+ string & token similarity features (Levenshtein, Monge-Elkan, token-set ratios, IDF-Jaccard)
• Address & house-number consistency verification
• Pretrained Transformer Cross-Encoders:
    - XLM-RoBERTa-base (multilingual pair scoring)
    - Multilingual-E5 (pairwise and cluster cross-encoders)
• Graph & sibling margin features (relative margin against 2nd-best entity match)
                                          │
                                          ▼
[Stage 4: Gradient Boosting & Decision Decoding]
• Multi-stage LightGBM classifiers (Round 1 fast ranking + Round 2 hard-pair refinement)
• Isotonic Probability Calibration (per-fold cross-fitting)
• Bayesian Expected-F₀.₅ Threshold Decoding with learned "any match" singleton gating
• Twin-world decoy filtering (rejects near-identical address/name false positives)
                                          │
                                          ▼
                          output/matching_results.tsv
                  (Final Scored & Deduplicated Record Matches)
```

---

## 🔬 Core Innovations

1. **Two-Channel Inverted Index Blocking:**
   Avoids expensive all-pairs comparison ($O(N \times M)$) by combining:
   * **Full-Word Channel (`fw`):** TF-IDF on token n-grams and address phrases.
   * **Half-and-Half Channel (`hh`):** Split cosine scoring on 50% name / 50% address tokens.
2. **Out-of-Sample Cross-Encoder Training:**
   Fine-tuned `xlm-roberta-base` and `multilingual-e5` models split in honest out-of-fold halves, ensuring downstream tree models learn un-leaked generalization probabilities.
3. **Twin-World Decoy Defense:**
   In real-world business registries, decoys share near-identical street numbers or generic name suffixes. A synthetic twin generator trains the classifiers specifically to penalize false-twin false positives.
4. **Expected-F₀.₅ Bayesian Decoding:**
   Rather than applying an arbitrary fixed probability cutoff, the decoder maximizes the mathematical expectation $E[F_{0.5}]$ under calibrated probabilities for each reference entity.

---

## 📁 Repository Structure

```text
├── Documentation_template.md        # Comprehensive 33 KB academic methodology write-up
├── README.md                        # Project overview and documentation
├── context.md                       # Research foundations, paper citations & disclosures
├── output/
│   └── matching_results.tsv         # Final predicted match table (1.73M records, MD5 f1754ce2)
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt         # Production dependencies (Python 3.11 / LightGBM / PyTorch)
│       ├── requirements_ce_gpu.txt  # GPU requirements for transformer cross-encoders
│       └── src/
│           ├── run_all.sh           # Master pipeline execution script
│           ├── configs/             # YAML configuration files (paths, hyperparameters)
│           ├── ber/                 # Pipeline A core engine:
│           │   ├── normalize.py     # String sanitization & transliteration
│           │   ├── blocking/        # TF-IDF candidate generation & probe reports
│           │   ├── features/        # RapidFuzz, Monge-Elkan, graph & house number extractors
│           │   ├── models/          # LightGBM training & stacked models
│           │   ├── decode.py        # Isotonic calibration & Expected-F0.5 decoder
│           │   ├── calibrate.py     # Unseen domain / country calibration
│           │   └── blend.py         # Multi-model ensemble blending
│           ├── gpu/
│           │   └── ce_train_predict.py # Transformer Cross-Encoder training (XLM-RoBERTa / E5)
│           ├── razan/               # Pipeline C modules (QTpies cluster encoders & fuse logic)
│           └── tests/               # Unit and regression test suites
└── student_resource/                # (Optional / Gitignored) Raw benchmark data and caches
```

---

## 🚀 Quick Start & Execution

### 1. Environment Setup

It is recommended to use Python 3.11 in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/d3vSoup/MLC-Zon.git
cd MLC-Zon

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install pipeline dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run Tests & Validation

Verify mathematical correctness, metric implementations, and I/O handlers:

```bash
cd code/business_entity_resolution/src
python -m pytest -q tests/
```

### 3. Run Pipeline Stages

Each stage can be executed independently or sequentially via the CLI:

```bash
cd code/business_entity_resolution/src

# 1. Normalize data
python -m ber.cli normalize --split train

# 2. Generate candidate pairs via TF-IDF blocking
python -m ber.cli block --split train

# 3. Extract features
python -m ber.cli features --split train

# 4. Train LightGBM model
python -m ber.cli train

# 5. Decode matches with Expected-F0.5 optimization
python -m ber.cli decode
```

Alternatively, run the complete automated pipeline:
```bash
bash run_all.sh
```

---

## 📊 Detailed Error & Loss Anatomy

Analysis of residual error reveals the true bottleneck in record linkage:
* **48% of loss** stems from **Candidate Blocking Misses** (mitigated by our two-channel index).
* **33% of loss** is due to **False Negatives in sparse address records** (mitigated by cross-encoder semantic embeddings).
* **6.3% of loss** stems from **False Positives on generic business tokens** (e.g., "Enterprises", "Holdings", "Group").

---

## 🛠️ Technology Stack

* **Core Languages:** Python 3.11+
* **Machine Learning & NLP:** LightGBM, PyTorch, Hugging Face Transformers (`xlm-roberta-base`, `multilingual-e5-base`, `multilingual-e5-small`)
* **String & Graph Processing:** RapidFuzz, Scikit-learn, Sparse-dot-topn, Indic-Transliteration
* **Data Processing & I/O:** Polars, Pandas, PyArrow, Parquet, Zstandard

---

## 📜 Citations & Theoretical Grounding

This project builds on principles from leading research in entity resolution and record linkage:
1. **Paulsen et al. (PVLDB 2023):** *Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching* — Inverted-index TF-IDF candidate selection.
2. **Conneau et al. (ACL 2020):** *XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale* — Cross-lingual semantic similarity.
3. **Bai et al. (2023):** *Optimal F-score Clustering for Bipartite Record Linkage* — Mathematical Bayes estimator for expected F-score maximization.
4. **Ke et al. (NeurIPS 2017):** *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* — Scalable tree boosting over heterogeneous tabular features.

Full bibliography and literature review available in [`context.md`](context.md).

---

## 👤 Author & Acknowledgments

* **Souparno Chakraborty** ([@d3vSoup](https://github.com/d3vSoup))
* Developed as an academic Mini Project in Machine Learning, incorporating collaborative challenge engineering with team members: *Muhammed Razan, Muaz Ismail Mohammed, and Ankit Ranjan*.
* **License:** [MIT License](LICENSE)
