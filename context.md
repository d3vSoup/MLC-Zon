# MLC-Zon · Project Context

**Working Title:**
> *"Resolving Business Entities Across Borders: A Hybrid Blocking, Gradient-Boosting, and Multilingual Cross-Encoder Pipeline"*

**Short-form title (for slides / reports):**
> *"Cross-Border Business Entity Resolution with LightGBM and XLM-RoBERTa"*

---

## 0. What This Project Is

This is a mini-project built on top of the **QTpies** team submission for the **ML Challenge 2026: Business Entity Resolution** (MLC-Zon).

The task:
- Three noisy business-record sources (S1 = reference, S2 + S3 = to be matched).
- No shared identifiers. Names and addresses have typos, Indic-script transliterations, legal-form variants, abbreviated or reordered addresses, zero-padded house numbers.
- Train set covers **US and India** only. Test set adds **France** (unseen country, 15% of test).
- Evaluation metric: **macro F0.5** over S1 entities (precision-weighted).

The final public score was **0.986758** — achieved through a three-layer system:

1. **Pipeline A** (ours): TF-IDF blocking → 42 hand-crafted features → LightGBM x2 + xlm-roberta-base cross-encoder → expected-F0.5 decoder.
2. **Pipeline C / QTpies** (Razan): char-4-gram blocking → multi-stage LightGBM → pairwise + cluster cross-encoders (multilingual-e5).
3. **Fuse layer**: pair-level blend of A + C probabilities → per-country post-process with vetoes, adds, and twin-signature rules.

---

## 1. Repository Map (key paths)

| Path | What |
|---|---|
| `code/business_entity_resolution/src/ber/` | Pipeline A (Python package): normalize → block → features → train → decode → blend |
| `code/business_entity_resolution/src/configs/default.yaml` | All knobs (K, thresholds, model names, blend weight, France k) |
| `code/business_entity_resolution/src/run_all.sh` | Single entry point for Pipeline A |
| `code/business_entity_resolution/src/gpu/ce_train_predict.py` | XLM-RoBERTa cross-encoder fine-tune + score (GPU) |
| `code/business_entity_resolution/src/razan/` | Pipeline C + fuse: `src/fuse.py`, `merge_by_country.py` |
| `code/business_entity_resolution/src/tests/` | 12 unit tests: metric, IO, decoder |
| `Documentation_template.md` | Full methodology write-up (ships in the zip) |
| `output/matching_results.tsv` | Final submission file, md5 `f1754ce2ef7dad7a8df904636eec20c2` |

---

## 2. Pipeline A — Stage by Stage

| # | Stage | Key module | What it does |
|---|---|---|---|
| 1 | normalize | `ber/normalize.py`, `ber/dictionaries.py` | Unicode fold, Indic→Latin (indic-transliteration), state→codes, address short forms, zero-strip, DBA/landmark split, sound-alike skeleton |
| 2 | folds | `ber/folds.py` | 5 folds grouped by S1 (seed 42) + 20% training sample flag |
| 3 | block | `ber/blocking/tfidf.py` | Two sparse TF-IDF cosine channels per country: `fw` (all tokens) and `hh` (half-name / half-address), top-10 S1 per record |
| 4 | blockreport | `ber/blocking/report.py` | Pair completeness 0.9825, reduction ratio 0.9999933 |
| 5 | features | `ber/features/strings.py`, `context.py` | 42 pair features: rapidfuzz name/address similarities, IDF-Jaccard, postal/house-number, legal form, channel scores/ranks, context margins |
| 6 | train/predict | `ber/models/lgbm.py` | LightGBM round 1 (5 folds OOF) |
| 7 | graph, r2feats | `ber/features/graph.py`, `numbers.py`, `names2.py` | 13 graph features + 15 soft house-number + 8 v2 name features on hard pairs (p >= 0.01) |
| 8 | ce | `ber/ce_ingest.py` | Joins shipped xlm-roberta-base cross-encoder score `ce_p` |
| 9 | stack | `ber/models/stack.py` | LightGBM round 2 (3000 rounds, 5 folds), OOF 0.98765 |
| 10 | decode | `ber/decode.py` | Owner rule → isotonic calibration → any-match gate → exact expected-F0.5 decoder |
| 11 | report/write | `ber/analysis.py`, `ber/decode.py` | Loss buckets; writes both TSVs + runs official validator |
| 12 | blend | `ber/blend.py`, `ber/calibrate.py` | Pair-level blend with Razan's p; label-free France calibration (odds factor k by empty-rate matching) |

---

## 3. Pipeline C (QTpies) — Model Lineage

| Model | What adds | OOF (DROP16 world) |
|---|---|---|
| M5 | char-4-gram + word TF-IDF blocking, 50 features, 2-stage LGBM, 2 pairwise CEs (multilingual-e5-small) | 0.98820 |
| M8 | M5 + twin-world training (synthetic decoys) + cluster CE | 0.98849 |
| M9 | M8 + cluster CE on multilingual-e5-base | — |
| M10 | M9 + stage-3 LGBM recalibration with label-free same-name rival counts | 0.98898 |
| M11h | M10 + Pipeline-A `p` as feature (US/India stacker) | 0.98956 |

The **fuse** (`src/razan/src/fuse.py`) applies per-country vetoes, adds, twin-signature vetoes, and France fills, using ~10 public LB probes to set thresholds.

---

## 4. Key Numbers

| Quantity | Value |
|---|---|
| Train S1 | 2,206,821 (US 60% / India 40%) |
| Test S1 | 1,732,544 (US 38% / India 47% / France 15%) |
| Blocking pair completeness | 0.9825 |
| Train pairs | 153.46 M |
| Blocking ceiling F0.5 | 0.99407 |
| Pipeline A OOF (with CE) | 0.98765 |
| Blend OOF (M11h, w = 0.0) | 0.99093 |
| Final public score (FINAL 2) | **0.986758** |
| Clean-room runtime | 320 min, peak 10.4 GB RAM |

---

## 5. What the Mini-Project Explores

The mini-project focuses on understanding and partially re-implementing or analysing one or more of the following sub-problems from the pipeline:

1. **Blocking quality vs. speed trade-off** — how does dual-channel TF-IDF (fw + hh) compare to single-channel, and how sensitive is pair completeness to K?
2. **Feature importance in gradient-boosted matching** — which of the 42 pair features matter most, and how do the round-2 graph / house-number features change prediction on hard pairs?
3. **Cross-encoder as a stacked feature** — how much does adding `ce_p` (xlm-roberta-base) improve round-2 LightGBM versus using only hand-crafted features?
4. **Unseen-country generalisation** — the leave-one-country-out (LOCO) study shows India-unseen 0.91571 → +0.0099 with num+names2 features; what drives this gap?
5. **Expected-F0.5 decoding** — comparing the threshold-free expected-F0.5 decoder against a fixed-threshold baseline on calibrated probabilities.

---

## 6. Related Research Papers

### 6.1 Entity Matching & Record Linkage (Foundations)

| # | Paper | Venue / Year | Relevance |
|---|---|---|---|
| 1 | **Ditto: Deep Entity Matching with Pre-Trained Language Models** — Li, Li, Suhara, Doan, Tan | VLDB 2020 ([arXiv:2004.00584](https://arxiv.org/abs/2004.00584)) | Cross-encoder architecture for entity matching; BERT fine-tuning for pairwise classification — directly analogous to the `ce_train_predict.py` step. |
| 2 | **DeepMatcher: Deep Learning for Entity Matching** — Mudgal et al. | SIGMOD 2018 | Predecessor to Ditto; established deep learning baselines for EM; motivates why transformer-based CEs beat RNN models. |
| 3 | **A Survey of Entity Resolution with Deep Learning** — Thirumuruganathan et al. | IEEE TKDE 2021 | Comprehensive overview of blocking, features, and classification for ER; grounding for the full pipeline architecture. |

### 6.2 Blocking

| # | Paper | Venue / Year | Relevance |
|---|---|---|---|
| 4 | **Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching** — Paulsen, Govind, Doan | PVLDB 2023 | Direct inspiration for the TF-IDF top-K blocking in `ber/blocking/tfidf.py`; shows simple inverted-index blocking rivals complex neural blockers. |
| 5 | **Blocking and Filtering Techniques for Entity Resolution: A Survey** — Papadakis et al. | ACM CSUR 2020 | Comprehensive taxonomy of blocking methods; contextualises the two-channel (fw + hh) design choice. |

### 6.3 Multilingual Encoders

| # | Paper | Venue / Year | Relevance |
|---|---|---|---|
| 6 | **XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale** — Conneau et al. | ACL 2020 ([arXiv:1911.02116](https://arxiv.org/abs/1911.02116)) | The backbone of the cross-encoder in Pipeline A (`xlm-roberta-base`); enables multilingual entity matching without translation. |
| 7 | **Multilingual E5 Text Embeddings: A Technical Report** — Wang et al. | arXiv:2402.05672, 2024 | Backbone of Razan's `multilingual-e5-small/base` pairwise and cluster cross-encoders (Pipeline C). |
| 8 | **BERT: Pre-training of Deep Bidirectional Transformers** — Devlin et al. | NAACL 2019 ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805)) | Foundational transformer paper; all cross-encoders above are BERT-family variants. |

### 6.4 Gradient Boosting for Tabular Data

| # | Paper | Venue / Year | Relevance |
|---|---|---|---|
| 9 | **LightGBM: A Highly Efficient Gradient Boosting Decision Tree** — Ke et al. | NeurIPS 2017 | The classifier used in both round-1 and round-2 of Pipeline A; justifies the choice of GBDT over a neural classifier for the 42 hand-crafted + graph features. |
| 10 | **Why do tree-based models still outperform deep learning on tabular data?** — Grinsztajn et al. | NeurIPS 2022 ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) | Explains why LightGBM remains competitive for heterogeneous feature tables (mix of string similarities, counts, ranks). |

### 6.5 Decision Decoding & F-Score Optimisation

| # | Paper | Venue / Year | Relevance |
|---|---|---|---|
| 11 | **Optimal F-score Clustering for Bipartite Record Linkage** — Bai et al. | arXiv 2023 | Theoretical grounding for the expected-F0.5 decoder in `ber/decode.py`; proves that maximising expected F_β under calibrated probabilities is a Bayes estimator. |
| 12 | **Isotonic Regression under Lipschitz Constraint** — Luss & Teboulle | JMLR 2012 | Basis for the isotonic calibration step applied to raw LightGBM scores before decoding. |

### 6.6 Generalisation to Unseen Domains / Countries

| # | Paper | Venue / Year | Relevance |
|---|---|---|---|
| 13 | **Zero-Shot Transfer Learning for Entity Resolution** — Zhao & Ng | EMNLP 2021 | Studies transfer of ER models to domains not seen in training — directly relevant to the France LOCO study. |
| 14 | **Domain Adaptation of NLP Models** — Ramponi & Plank | arXiv:2007.01654, 2020 | Survey of domain shift; motivates why country-agnostic features and LOCO validation are needed for robust out-of-distribution performance. |

---

## 7. Disclosures (must stay in any write-up)

1. **Test-set label-free statistics used**: France empty-rate for calibration factor k; per-split generic-word / IDF sets; twin parameters measured on US test (DBIN histogram, 89% positive shifts, 94% suffixes, 30% pairs).
2. **Country-keyed settings**: twin rates (US 0.072 / 0.03, India 0.03); fuse per-country thresholds; inactive France branch in `calibrate.py:87`.
3. **Leaderboard use**: fuse thresholds, hybrid rule, and FINAL 2 per-country merge were chosen from ~10 public probes.
4. **SOLO caveat**: clean-room rebuild changes 174–176 fused rows (0.01%); reference file shipped as fixed input.
5. **Hand-written dictionaries**: all maps in `ber/dictionaries.py` (disclosed in Documentation 4.3).
6. **Licences**: xlm-roberta-base (MIT, 278 M), multilingual-e5-small (MIT, 118 M), multilingual-e5-base (MIT, 278 M), LightGBM / polars / rapidfuzz (MIT), sparse_dot_topn (Apache-2.0), indic-transliteration (MIT). All below 8 B limit; no external data or APIs.

---

## 8. Team

| Name | Role |
|---|---|
| Muhammed Razan | Leader; Pipeline C (QTpies) + fuse |
| Muaz Ismail Mohammed | Pipeline A, integration, packaging |
| Ankit Ranjan | Cross-encoder (Kaggle GPU), Documentation |
| Souparno Chakraborty | Cross-encoder (Kaggle GPU), Documentation |

---

*Last updated: 2026-10-16. All numbers traceable to `EXPERIMENTS.md` / `HANDOFF_RAZAN.md`.*
