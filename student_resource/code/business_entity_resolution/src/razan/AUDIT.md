# QTpies — Business Entity Resolution: Full Pipeline Audit

| | |
|---|---|
| **Snapshot** | Sat 26 Sep 2026, ~14:30 IST · branch `er-pipeline` @ `b7e2a37` |
| **Best public F0.5** | **0.979** (`sub11_m4_shift`) · staged but not submitted: `sub12_m5_shift` (expected ≈ 0.9795–0.980) |
| **Public leader** | 0.985884 at start → **0.99056** (Sat midday) · our gap: 0.0116 |
| **Best CV (OOF F0.5)** | **0.98820** (M5, test-like density; test-mix 0.98900) |
| **Deadline** | Sun 27 Sep ~17:30 IST (5 public submissions/day, reset ~05:30 IST) |
| **Purpose** | A self-contained record of the QTpies pipeline (data → blocking → features → models → decoding → calibration → submission). It covers every experiment (including failures), what we learned about the test set, where the remaining loss sits, and known audit risks. It is written to be cross-referenced against the other QTpies pipeline. |
| **Sources** | Code in `code/business_entity_resolution/`, `EXPERIMENTS.md`, Kaggle kernel logs (`runs/`), analysis-script outputs and session logs. Every number is **measured** unless marked *estimate* or *assumption*. |

---

## Contents
0. [Executive summary](#0-executive-summary)
1. [Scoreboard: leaderboard, CV, and the climb](#1-scoreboard)
2. [Problem, metric and verified data facts](#2-problem-metric-and-verified-data-facts)
3. [Pipeline, stage by stage](#3-pipeline-stage-by-stage)
4. [Infrastructure, compute and orchestration](#4-infrastructure-compute-and-orchestration)
5. [Chronological experiment log](#5-chronological-experiment-log)
6. [What worked, ranked by measured gain](#6-what-worked-ranked-by-measured-gain)
7. [What did not work, and why](#7-what-did-not-work-and-why)
8. [What we learned about the test set](#8-what-we-learned-about-the-test-set)
9. [Where the remaining loss is](#9-where-the-remaining-loss-is)
10. [Candidate search (blocking) deep-dive](#10-candidate-search-blocking-deep-dive)
11. [Audit findings: risks, gaps and bugs](#11-audit-findings-risks-gaps-and-bugs)
12. [Recommended next steps](#12-recommended-next-steps)
13. [Checklist for comparing with the other pipeline](#13-checklist-for-comparing-with-the-other-pipeline)
14. [Deliverables status and compliance](#14-deliverables-status-and-compliance)
15. [Lessons and mistakes](#15-lessons-and-mistakes)
- [Appendix A: Feature dictionary](#appendix-a-feature-dictionary)
- [Appendix B: Constants and hyperparameters](#appendix-b-constants-and-hyperparameters)
- [Appendix C: Files, artefacts and paths](#appendix-c-files-artefacts-and-paths)
- [Appendix D: How each model was produced (Kaggle DAG)](#appendix-d-how-each-model-was-produced-kaggle-dag)
- [Appendix E: Glossary](#appendix-e-glossary)

---

## 0. Executive summary

**What we built.** A five-stage entity-resolution pipeline.
1. **Normalisation** (`norm.py`). Rule-based cleaning plus two learned components, both label-free on test:
   - a native-script → Latin token table learned from train pairs;
   - an "admin-level" address-component vocabulary learned per split.
2. **Candidate generation** (`block.py`). Within each country, every S2/S3 row retrieves its top-15 S1s by a combined TF-IDF cosine: name char-4-grams ⊕ address word 1–2-grams. A second, wider pass runs for uncertain rows.
3. **Pair features** (`feats.py`). 50 language-agnostic features: rapidfuzz similarities, number agreement, rarity counts, and candidate-list context (ranks, gaps, margins).
4. **Two-stage LightGBM** with 2-fold cross-fitting (`model.py`, `stage2.py`).
   - Stage 1 scores pairs.
   - Stage 2 ("collective") adds features computed from the other rows of the same S1 and the other S1s of the same row: owner competition, cluster mass, sibling agreement, rival twin groups.
   - Stage 2 also takes two fine-tuned **cross-encoders** (`ce.py`, `intfloat/multilingual-e5-small`, MIT, 118M params, trained on Kaggle T4×2): CE-1 on normalised text, CE-2 on raw text.
5. **Decision** (`decide.py`). Each S2/S3 row gets one owner, then each S1 keeps the subset that maximises exact expected F0.5 (numba Poisson-binomial DP).
   - Then **label-free test-time calibration** (`calibrate.py`): a difference-in-differences prior-shift correction for seen countries, and empty-rate matching for the unseen country.

**Score journey (public):** 0.969 → 0.971 → 0.977473 → **0.979**.

**CV journey (OOF F0.5):**

| Model | Stage 1 | Stage 2 | With CE |
|---|---|---|---|
| M1 (natural density) | 0.97269 | 0.98092 | 0.98715 (model2) |
| M2 onwards (test-like density) | 0.97123 | — | 0.98641 → 0.98749 (M4) → **0.98820 (M5)** |

**Biggest wins:**

| Change | Gain |
|---|---|
| Collective stage 2 | +0.0082 CV |
| Cross-encoder feature | +0.0062 CV |
| France calibration | +0.0020 public |
| US prior-shift calibration | +0.0015 public |
| Hard-negative CE scoring all stage-2 pairs | +0.0010 CV |
| Raw-text CE-2 | +0.0007 CV |
| Blocking pass 2 | +0.45% recall |

**Flat or failed:**

| Experiment | Result |
|---|---|
| Generic-word features | +0.0001 |
| Stage-2 hyperparameter tuning | flat |
| Averaging M2 and M3 | flat |
| Stage 3 | worse on smoke |
| Blocking v3 tweaks | +0.0006 recall |
| Isotonic band recalibration | failed its null test |
| Single-difference band-k | biased null |
| Stratified decode | ≈ +0.0004 |
| Row-count prior | weak |
| Address-only blocking pass | ≈ +0.0007 (analysis only) |

**The test set is not train-like.**
- About 40% of test S2/S3 rows are distractors, against 26% in train.
- **India** test behaves exactly like train.
- **US** test has about 8× more **twin decoys**: same core name, a legal suffix added (94%), and the house number shifted by +1…+20 (median 7).
- **France** is unseen in training. It over-links on "same address, one generic word swapped" twins, and its true links score lower.
- US test is also **less crowded** than train (fewer same-name S1s).

**Remaining loss.** On train (M4 oracle budget), the main losses are:
- blocking misses: +0.0053
- stage-2 not-kept: +0.0053
- false links: +0.0016

On test, there is ≈ **0.009 extra** from US twins plus France. On train, fixing every *solvable* error would reach 0.9943.

**Gap to the leader (0.99056).** Our train-level OOF at test-like density (0.9882) is already below the leader's **test** score. So the leader's base matcher is stronger *and* they handle the test shift.

**Top audit risks:**
1. The documented `run.py all` path does **not** reproduce the submitted file. That comes from the Kaggle DAG plus `calibrate.py`.
2. Cross-encoder dependencies and the model revision are unpinned.
3. The README is stale.
4. CV has two mild optimism sources: early stopping on the OOF fold, and folds grouped by S1 only.
5. The 2-argument mode of `calibrate.py` names "France" explicitly.
6. Folds and drop sets depend on polars' `hash()`, so they are tied to the pinned polars version.

**Most promising next step (designed, not implemented).** Retrain on a **test-matched synthetic "twin world"**: twin rows derived from real train rows, with parameters measured label-free on test. Expected gain is +0.0015 to +0.004. After that come the blocking additions (≈ +0.0007) and France-specific twin handling.

---

## 1. Scoreboard

### 1.1 Public leaderboard: every score that was reported back

| # | When (IST) | File (`submissions/…`) | Model | Test-time decode | Predicted | **Public** | Δ vs previous | What it taught us |
|---|---|---|---|---|---|---|---|---|
| 1 | Sat 02:10 | `sub01_s2` | M1 | raw | 0.976 (0.972–0.979) | **0.969** | — | CV→public gap of −0.0124 (CV mix 0.98144) |
| 2 | Sat 02:48 | `probe_fr_empty` | sub01 with every France list emptied | — | ~0.83 | **0.836** | probe | France F ≈ 0.94; US+India ≈ 0.973 (vs CV 0.980) |
| 3 | Sat 03:19 | `sub03_frk015` | M1 | France-only odds ×0.15 | — | **0.971** | +0.002 | France over-links; the removed links were mostly twins |
| 4 | Sat ~10:00 | `sub08_m4_frcal` | M4 | France k=0.05 (empty-rate matching) | 0.982 (0.978–0.985) | **0.977473** | +0.0065 | CV gains transfer; the shift gap stays ≈ −0.011 |
| 5 | Sat ~11:30 | `sub11_m4_shift` | M4 | + US k=0.060 (DiD prior shift) | 0.981 (0.979–0.983) | **0.979** | +0.0015 | US shift is real, but the gain was ⅓ of the predicted +0.0043 |
| — | staged | `sub12_m5_shift` | M5 | US k=0.060, France k=0.10 | ≈ 0.9795–0.980 | not submitted | — | Recommended as Sunday's first upload |

- Only these five scores came back to the modelling loop; any other uploads were never used in analysis.
- Our pre-submission **predictions were optimistic every time**: by 0.007, 0.0045 and 0.002. CV→public mapping was never well calibrated (§15).
- Staged files never scored publicly, all validator PASS:
  - `sub02_k06`: M1, all countries odds ×0.6.
  - `sub02_s2ce`: model2, raw decode.
  - `sub03_m2`, `sub04_m3`, `sub08_m4`, `sub09_m3t`, `sub10_m5`: raw decodes.
  - `sub05_ce_frcal`, `sub06_m2_frcal`, `sub07_m3_frcal`: France-calibrated.

### 1.2 Cross-validation: OOF F0.5 on the full train S1 set, exact metric

"Mix" weights countries to the test mix (US 38.3 / India 46.8 / France 15.0), with France proxied by the mean of US and India. **Mix is therefore not a France estimate.**

| Run (Kaggle slug) | Git | Density world | Stage 1 | **Stage 2** | Mix | US | India | Stage-2 pairs (pos) | True links outside stage-2 set |
|---|---|---|---|---|---|---|---|---|---|
| Smoke, 3% subset | 8eddd0e | tiny, not representative | 0.9824 | 0.9869 | 0.9877 | 0.9876 | 0.9860 | 0.28M | 988 |
| **M1** `qtpies-model` | 3eef518 | natural (26% distractors) | **0.97269** | **0.98092** | 0.98144 | 0.98220 | 0.97900 | 10.30M (7.51M) | 124,235 (1.63%) |
| **model2** `qtpies-model2` | 65f88cc | natural | (M1's) | **0.98715** | 0.98798 | 0.98761 | 0.98647 | 10.30M | 124,235 |
| **M2** `qtpies-m2-s2` | 65f88cc | **DROP16** (~38% distractors) | 0.97123 | **0.98641** | 0.98726 | 0.98681 | 0.98582 | 8.81M (6.31M) | 104,470 (1.63%) |
| **M3** `qtpies-m3` | 94704b5 | DROP16 | 0.97123 | **0.98652** | 0.98736 | 0.98693 | 0.98590 | 8.81M | 104,470 |
| avg(M2, M3) | — | DROP16 | — | 0.98651 | 0.98735 | 0.98693 | 0.98589 | — | — |
| **m3t** `qtpies-m3t` | bb51728 | DROP16 | 0.97123 | 0.98654 | 0.98738 | 0.98695 | 0.98593 | 8.81M | 104,470 |
| **M4** `qtpies-m4` | 38e0b5e/bb51728 | DROP16 | 0.97123 | **0.98749** | 0.98832 | 0.98793 | 0.98682 | 8.81M | 104,470 |
| **M5** `qtpies-m5` | ab98fa8 | DROP16 | 0.97123 | **0.98820** | **0.98900** | 0.98874 | 0.98739 | 8.81M | 104,470 |

Comparability rules:
- M1 and model2 are measured at **natural** train density. M2 onwards are measured at **test-like** density: 16% of train S1s are hash-dropped, so their rows become distractors. The second world is harsher, so **never compare CVs across the two worlds**.
- In the M2 world, M1's architecture scores 0.97123 at stage 1 against 0.97269 in the natural world.
- The **decoder's own expected-F on test** is not a score predictor. It was 0.98267 (M1), 0.99039 (model2), 0.99010 (M2), 0.99020 (M3), 0.99038 (m3t), 0.99141 (M4) and 0.99274 (M5), all far above public.

### 1.3 The climb, attributed

| Step | Public | Source of the gain | Evidence |
|---|---|---|---|
| M1 baseline | 0.969 | Normalisation, 2-pass TF-IDF blocking, 50 features, stage 1 → collective stage 2, expected-F0.5 decode | CV 0.98092 |
| + France odds ×0.15 | 0.971 (+0.002) | France over-linking: 38.6k of its 894k links removed, mostly generic-word-swap twins | Probe; France ≈ 0.94 → 0.955 |
| + model line M1 → M4, France re-calibrated | 0.977473 (+0.0065) | CE-1 feature (+0.0062 CV), rival-group and sibling-margin features, density matching, generic features, hard-negative CE scoring all stage-2 pairs (+0.0010); France k=0.05 by empty-rate matching | CV 0.98092 (natural) → 0.98749 (DROP16) |
| + US prior-shift correction | 0.979 (+0.0015) | Label-free DiD vs India: US k=0.060; 70,370 US S1 rows changed; India and France identical to sub08 | Clean A/B |
| (staged) M5 | ≈ 0.9795–0.980 | CE-2 raw text (+0.0007 CV); same calibration (US 0.060, France 0.10) | Not yet submitted |

---

## 2. Problem, metric and verified data facts

### 2.1 Task and metric
- **The task:** for every S1 entity (a deduplicated reference list), output all matching S2/S3 rows, or an empty list.
- **Scoring:** macro F0.5 per S1, singletons included: F0.5 = 1.25·P·R / (0.25·P + R).
  - An empty prediction on a true singleton scores 1.0.
  - Any link on a singleton scores 0.
- **Cost asymmetry:** for an S1 with 3 true links,
  - one false link scores 0.789;
  - one miss scores 0.909;
  - top-1 only scores 0.714.
- **Implication:** decode per S1 by expected F0.5, not with a global threshold. For independent probabilities, adding a link pays off only if p > F*/(1+β²) ≈ F*/1.25, which is **≈ 0.78 at F*≈0.98**. That is why true links with p ≈ 0.4–0.7 are correctly left out.
- **Output:** one row per test S1 (1,732,544 rows), tab-separated, comma-joined IDs, no quoting.
  - `candidate_pairs.tsv` must be exactly the final model's input.
  - Matches must be a subset of candidates.

### 2.2 Scale and structure (full data)

| Split | S1 | S2 | S3 | S2+S3 rows per S1 |
|---|---|---|---|---|
| Train | 2,206,821 (US 1,323,633 · India 883,188) | 5,034,616 | 5,285,603 | 4.68 |
| Test | 1,732,544 (US 663,106 · India 809,986 · France 259,452) | 4,887,273 | 5,082,316 | 5.75 |

**Rows per S1 by country:**

| | S2/S1 | S3/S1 |
|---|---|---|
| Train (both countries) | 2.28 | 2.39–2.40 |
| Test US | 2.82 | 2.93 |
| Test India | 2.86 | 2.97 |
| Test France | 2.71 | 2.82 |

**Country mix of S1:** train 60% US / 40% India; test **38.3% US / 46.8% India / 15.0% France**.

**Generator constants (train; identical for US and India, which points to a programmatic generator):**
- 7,638,365 true links, which is also 7,638,365 distinct S2/S3 IDs. So **each S2/S3 row has at most one owner**.
- The country always agrees within a true pair.
- **3.46 links per S1.** Rows with no true link: **5.58%** of S1s (US 5.583%, India 5.588%).
- Distribution of links per S1, for 0…11: 5.6, 5.4, 17.0, 24.1 (mode 3), 21.9, 14.6, 7.5, 2.9, 0.8, 0.2, <0.03, <0.01 %.
- **Caps:** at most 5 S2 rows and at most 6 S3 rows per S1.
- Per-source truth among S1s with ≥1 link:

  | Rows from source | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|---|
  | S2 | 138,001 | 663,037 | 548,426 | 280,255 | 100,088 | 20,291 | — |
  | S3 | 120,275 | 601,965 | 561,161 | 313,022 | 121,674 | 29,651 | 2,350 |

- About **26% of train S2/S3 rows are distractors**, i.e. match nothing (1.34M in each source).
  - These are businesses whose S1 is absent.
  - Only 5.4% of them share an exact normalised name with some S1, against 29.2% of true matches; 0% share both name and address.
- **Ambiguity inside S1:** 39% of S1s share an exact normalised name within their country ("Primary Care Group" ×253; "Physical Therapy Associates" has 615 same-core S1s). 5.3% share an exact address. None share both.
- **Missing fields:**
  - S1 is always complete, and S1 names are always Latin script.
  - S2/S3 addresses are missing in ≈3.4% of train rows and ≈2.7% of test rows. By country: US train 3.59% vs test 2.89%; India 2.97% vs 2.37%; France 3.00%.

### 2.3 Noise catalogue: what the generator does

**Name noise.** Frequencies are measured over all 20.3M S2/S3 names in train and test.

| Pattern | Share of names |
|---|---|
| Parentheses | 5.52% |
| `&` | 4.18% |
| `.com/.net/.org/…` | 3.61% |
| `[` | 2.83% |
| Apostrophe | 1.57% |
| OCR 0↔O inside a word | 1.03% |
| `#` | 0.60% |
| `M/s` | 0.58% |
| `+` | 0.53% |
| `@` | 0.30% |
| `(ID: …)` | 0.29% |
| `\| www…` | 0.28% |
| 7+ digit phone numbers | 0.28% |
| `***`, `>>`, `--` junk affixes | ≈0.28% each |
| Alias wrappers: `dba`, `formerly`, `f/k/a`, `aka`, `trading as`, `t/a`, `d/b/a`, `doing business as`, `née`, `fka` | 0.06–0.27% each |

Other name noise:
- Legal suffix added, dropped, swapped or bracketed.
- Word shuffles and typos.
- Filler words: Center, Services, Group, Dr, The, Shri, M/s.
- Doubled words and injected accents.
- **Full replacement** by an invented brand ("Korflux", "Umbraumbra", "Ectotavo"), a domain ("salazarscatering.com"), a handle plus phone ("@hinklegoucher - 5121399883"), an acronym, or an acronym-domain.
- **Indian scripts** in India rows: Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi. These are phonetic spellings of English business words.

**Address noise:**
- Component reordering and abbreviations (Rd/Road, NC/North Carolina, TG, DL).
- House numbers perturbed, truncated or dropped (2645→264, 2048→048, 4605→4790). **Rows from the same source often share the same perturbed number**, which is the basis of `sib_pert`.
- Prefixes: zeros, #, No, Door No, H.no, HN, BLOCK F-261.
- Added PO Box, PMB, Unit or Fl.
- City swapped for a neighbour, county or CDP.
- State written in native script; `null/<NULL>/N/A` tokens; missing address.

**Noise profile on 200k true pairs** (token-set ratio after lowercase and accent strip):

| Source · country | Name exact | Name ≥90 | Name <50 | Native script | Address missing | Address ≥80 | Shares a number |
|---|---|---|---|---|---|---|---|
| S2 · US | 30% | 79% | 3% | 0% | 4.9% | 93% | 80% |
| S3 · US | 31% | 81% | 3% | 0% | 4.6% | 81% | 84% |
| S2 · India | 18% | 59% | 26% | 23% | 3.9% | 97% | 87% |
| S3 · India | 20% | 65% | 17% | 13% | 4.0% | 90% | 85% |

**Complementarity:** a weak Latin name *and* a missing address together occur in only **0.014%** of true pairs, so one field almost always carries the signal.

**Source styles:** S2 uses UPPERCASE and state codes. S3 uses Title Case, full state names, "Door No", and PO Box/PMB.

**France (test only):**
- A few dense cities: Bordeaux, Nantes, Lille, Saint-Nazaire, Tourcoing, Dunkerque, Pornic, La Baule.
- **39.4% of France S1 core names are duplicates** after normalisation.
- Abbreviations: R/BD/AV/PL/ALL/CHE, bis/ter, N°.
- Département ↔ région swaps.
- Legal forms SARL/SAS/SASU/SCI/SA/EURL, often dotted: 70,464 France names contain single-letter runs such as "S.A.R.L.".
- The same generator families as the other countries: domains, invented brands, junk affixes.

### 2.4 Leakage checks (none found; re-checked Sat 12:19)

| Check | Result |
|---|---|
| Spearman (row order of S2/S3 vs its S1's row order) | S2 0.0011 · S3 −0.0016 |
| Spearman (ID number vs S1 ID number) | S2 −0.0027 · S3 0.0057 |
| Rows of the same entity adjacent in the file? | Median gap equals the random median (1.48M vs 1.48M); gap ≤10 occurs 0.000% |

### 2.5 Early ceiling estimate (EDA, before any model)
- 1.7–2.3% of true links look **pairwise-irreducible**: either no address plus a name shared by several S1s, or an alias name plus a shared address.
- An oracle that is perfect elsewhere scores **0.993–0.995**.
- The later measured train ceiling, fixing every *solvable* error on M4, is **0.9943** (§9).

---

## 3. Pipeline, stage by stage

### 3.0 Data flow

```
raw TSV (train/test × S1/S2/S3)  → common.raw(): parquet, quote_char=None, utf8-lossy
   │
   ▼
norm.py   learn_translit() [train true pairs only] · admin_vocab(split) [label-free, per split]
   │      → n_{split}_s{k}.parquet : id, country, nm, core, al_l, al_r, ad, nums, f_handle, f_domain, f_native, f_alias, f_noaddr
   ▼
block.py  per country (sharded): pass 1 (df-cap 3k, top-15) ∪ pass 2 (df-cap 10k, top-15, only rows with best cc < 0.6)
   │      → cand_{split}/{country}_{chunk}.parquet : ri, si, cn, ca, cc, cc2      (~16 candidates per row)
   ▼
feats.py  BER_DROP=16 on train: dropped S1s removed before ranks, aggregates and rarity counts
   │      → F_{split}/*.parquet : 50 features per (ri, si)
   ▼
model.py  Stage 1 LightGBM, 2 folds by hash(si): fit1 → pred_shard ×5 → gather1
   │      → oof_m.parquet (train OOF) · pred_test_m.parquet (mean of both fold models)
   ├───► ce.py (GPU T4×2): CE-1 (normalised text), CE-2 (raw text); hard negatives from oof_m
   │      → ce_{split}.parquet / ce2_{split}.parquet  (every pair with stage-1 p ≥ 0.002)
   ▼
stage2.py + model.train2("s2ce")  pairs with p1 ≥ 0.002  ← this set is candidate_pairs.tsv
   │      50 stage-1 features + p + collective (owner, cluster, sibling, rival) + generic + ce + ce2
   │      → oof_s2ce.parquet · pred_test_s2ce.parquet · output/s2ce/{matching_results,candidate_pairs}.tsv
   ▼
calibrate.py (laptop)  seen countries: DiD prior-shift k vs anchor country; unseen: empty-rate k
   │      → decide.py (one owner per row → exact expected-F0.5 top-k per S1)
   ▼
submissions/<name>/matching_results.tsv  → official validator (student_resource/utils/validate_submission.py)
```

**Index conventions** (used everywhere):
- `si` is the row number in `n_{split}_s1.parquet`.
- `ri` is the row number in `concat(n_{split}_s2, n_{split}_s3)`.
- `fold = hash(si, seed 42) % 2`.
- Density drop: `hash(si, seed 49) % 100 < BER_DROP`, applied to train only.

### 3.1 Code map (`code/business_entity_resolution/`)

| File | Lines | Role |
|---|---|---|
| `src/common.py` | 40 | Paths (`BER_WORK`, `BER_DATA`), `SEED=42`, `THREADS`, `DROP` and `dropped()`, `raw()` (TSV→parquet), `gt_pairs()` |
| `src/norm.py` | 235 | Normalisation; transliteration table; admin vocabulary |
| `src/block.py` | 119 | TF-IDF blocking (two passes), sharded by country and chunk, atomic writes |
| `src/feats.py` | 132 | Pair features (context, rarity, rapidfuzz), sharded, atomic writes |
| `src/model.py` | 198 | LightGBM stage 1 and stage 2; cross-fitting; OOF evaluation; sharded stage-1 prediction; stage-2 table |
| `src/stage2.py` | 118 | Collective features; generic-word features |
| `src/ce.py` | 136 | Cross-encoder training and scoring (GPU) |
| `src/decide.py` | 73 | One-owner rule plus exact expected-F0.5 decoding (numba), with self-checks |
| `src/calibrate.py` | 113 | Test-time prior-shift calibration (CLI) |
| `src/score.py` | 38 | Exact metric (checks against the README example 0.714) |
| `src/run.py` | 99 | Entry point, steps, TSV writer |
| `src/probe.py` | 19 | Leaderboard probe: empties one country's lists |
| `kaggle_runner.py` | 224 | Kaggle kernel batches, DAG chaining, push/status/fetch/wait |
| `finalize.py` | 45 | Fetches a kernel's `matching_results.tsv` and log, checks row count, runs the validator, stages to `submissions/` |
| `memguard.py` | 25 | Kills a laptop job when free RAM drops below a floor |
| `aws/launch.py`, `aws/job.py`, `aws/iam-policy.json` | — | Spot-VM launcher with budget guard. **Never launched** because of the vCPU quota. |

### 3.2 Normalisation (`norm.py`)

**Names**, producing `nm`, `core`, `al_l`, `al_r`:
1. Lowercase.
2. Strip `(id: 123)`, everything after `|`, URLs and `www.`, and runs of 7+ digits (phones).
3. Unicode NFKD, then remove combining marks (accents).
4. `&` and `+` → ` and `.
5. OCR fix: `0` next to a letter → `o`.
6. Drop TLD suffixes (`.com .net .org .in .co .fr .biz .info .us .io`).
7. Non-alphanumerics → space.
8. Remove consecutive duplicate tokens ("the the").
9. **Collapse single-letter runs**, so dotted legal forms become one token: `s a r l` → `sarl`, `l l c` → `llc`.
10. **Alias detection:** the regex covers `aka | a k a | nee | doing business as | dba | d b a | formerly known as | formerly | fka | f k a | t a | trading as`.
    - `al_l` is the part before the marker (usually an invented brand).
    - `al_r` is the part after it: the **real name, used downstream**.
11. `core` = tokens minus the `LEGAL` and `FILLER` lists; if that leaves nothing, `core = nm`.
    - `LEGAL`: inc, llc, ltd, pvt, private, limited, corp, co, company, llp, lp, pllc, pc, plc, esq, sarl, sas, sasu, sci, sa, eurl, snc, selarl, cie, gmbh, ets…
    - `FILLER`: the, and, of, dr, mr, shri, sri, smt, m, s, center, services, group, associates, fils, associes, plus French articles de/du/des/la/le/les/l/d/et/au/aux/en/sur.
12. **Flags:**
    - `f_native`: Indic script in the name, computed *before* transliteration (a bug fixed on Fri).
    - `f_handle`: `@` or `#`.
    - `f_domain`: TLD or `www`.
    - `f_alias`, `f_noaddr`.

**Transliteration** (`learn_translit`, from **train true pairs only**):
- **Names:** position alignment when the token counts of S1 and the native-script row agree. Each native token maps to its most frequent aligned Latin token.
- **Addresses:** co-occurrence over ≤400k sampled pairs. A native token *y* maps to argmax_t P(Latin t in S1 address | *y*), kept only if P ≥ 0.9 and *y* was seen ≥ 5 times; ties go to the rarer *t*.
- Name mappings win on conflicts. The v1 table held **1,347 name tokens (1,363 total)**. The EDA prototype (1,498 tokens) covered 96.4% of native-script tokens in test.
- Unknown native tokens fall back to **anyascii** (ISC licence).

**Addresses**, producing `ad` and `nums`:
1. Lowercase; remove `<null> null n/a none`; `N°` → `no`.
2. Split on commas into components.
3. **Drop admin-level components**, learned per split and per country, label-free (`admin_vocab`):
   - S1 components that appear last ≥ 70% of the time with ≥ 200 occurrences (states, régions);
   - plus frequent (≥ 200) last components of S2/S3 that never appear in S1 (départements, state spellings);
   - excluding anything with digits or shorter than 2 characters.
   - The learned France list: `gironde, hauts de france, loire atlantique, nord, nouvelle aquitaine, pas de calais, pays de la loire, st herblain, st nazaire`. See finding A10: the last two are city spellings.
4. Multi-word state names → codes (US and Indian states).
5. `nums`: all digit runs with leading zeros stripped, unique, in order.
6. Token map, applied to both sides so direction doesn't matter:
   - English: rd→road, st→street, ave→avenue, dr→drive, …
   - Compass points: n/s/e/w.
   - French: r→rue, bd→boulevard, av→avenue, all→allee, che→chemin, imp→impasse, fbg→faubourg, crs→cours, bis→b.
   - Noise tokens (`city cdp no door hno hn h`) removed.
   - City aliases: bombay→mumbai, gurgaon→gurugram, bengaluru→bangalore.
   - Single-word state names → codes.
7. Remove consecutive duplicate tokens.
8. `f_noaddr = (ad == "")`.

**v1 → v2 changes (Fri 19:00–21:00):** dotted legal forms collapsed, French stopwords, learned admin-component removal, the `f_native` fix, and a new `admin` job so the six file jobs can run in parallel.

### 3.3 Candidate generation (`block.py`)

**What gets vectorised:**
- **Name string:** `core`, or the core of `al_r` if an alias is present, **with spaces removed** ("greatindustries"). Char 4-grams make it robust to word shuffles, splits and typos.
- **Address string:** `ad`, as word 1–2-grams.

**How candidates are retrieved:**
- Two TF-IDF vectorisers (sublinear tf, float32) are fitted **per country on S1 only**, with `max_df = cap`. Features in more than *cap* S1 documents are dropped: they cost the most and weigh the least.
- Each part is L2-normalised, then the parts are concatenated and scaled by 1/√2. So the **combined cosine `cc` = (name cosine + address cosine)/2** when both are present.
- `sparse_dot_topn` retrieves the **top-15 S1s per query row** (threshold 0.02, multi-threaded).
- **Pass 1:** every row, cap 3,000.
- **Pass 2:** only rows whose best pass-1 `cc` < 0.6 (US 26.3%, India 23.1%), cap 10,000. The candidate set is the union.
- Every pair stores `cn` and `ca` (pass-1 name and address parts), `cc` (pass 1) and `cc2` (pass 2).
- Chunks of 250k query rows; shards run as separate processes (no fork pools, which deadlock with polars); outputs are written atomically.

**Benchmarks** that led here (India, 50k query rows against the full India S1 list):

| ID | Config | Recall @1 / @5 / @20 | Cost |
|---|---|---|---|
| B0 | Exact keys (sorted first-3 name words ∪ number+street), EDA | @20 = 0.768 (48.4% + 53.9% for the two keys) | 20 candidates/S1 |
| B1 | TF-IDF char-3 name + word address + combo, max_df 5% | — | 32 ms/query, too slow |
| per-view | name char-3 · name char-4 · address word-1/2 · address char-4 (df cap 2k) | @20 = 0.490 · 0.659 · 0.883 · 0.858 | 0.05–0.34 ms/q |
| B2 | Combined, per-row top-m pruning 16/12 | 0.942 / 0.963 / 0.974 | 2.4 ms/q (pruning doesn't bound postings) |
| union | name ∪ address ∪ combined | @20 = 0.9785 | 3 views |
| **B3** | **Combined, global df cap 3k / 10k / 30k** | @20 = **0.9736 / 0.9790 / 0.9830** | 0.39 / 1.05 / 3.4 ms/q |
| **B4 (v2, chosen)** | Cap-3k top-15 ∪ cap-10k top-15 for uncertain rows | See next table | 168.7M train pairs, ~16/row, 7 Kaggle 4-core sessions, ~14 min wall |
| v3a | + legal-suffix strip | union 0.9789 (v2 0.9790) | Plateau |
| v3b | + number×token pair features | union 0.9796 | +38% time (152 s vs 110 s) → not adopted |

**v2 recall on the full train set** (7,638,365 true links):

| | Recall |
|---|---|
| **All links** | **0.98436** |
| US S2 | 0.9881 |
| US S3 | 0.9893 |
| India S2 | 0.9845 |
| **India S3** | **0.9743** (weakest) |
| True S1 ranked by `cc` @1 / @3 / @5 / @10 / @15 | 0.9493 / 0.9662 / 0.9717 / 0.9771 / 0.9797 |
| Pass 2's contribution | about +0.45% recall |

Candidates per row: mean 15.98, median 15, max 30. The reduction ratio against the within-country cross product is about 1 − 1.7×10⁸ / 1.2×10¹³ ≈ 0.99999. The miss taxonomy and what could recover misses are in §10.

### 3.4 Pair features (`feats.py`): 50 per pair
- **Context** (per query row): ranks and gaps to the row's best candidate for `cc`, `cn`, `ca` and `cc2`; `n_cand`; `margin_cc`, the difference to the best *other* S1. Pairs are kept if `rk_cc ≤ 15` or `rk_cc2 ≤ 15`.
- **Per-S1 aggregates** over all candidate rows: `n_cand_s1`, `n_top1_s1` (rows for which this S1 is the top-`cc` candidate), and gaps to the S1's best `cc`, `cn` and `ca`.
- **Rarity**, counting only *live* S1s:
  - `k_core`: S1s with the same core name;
  - `k_addr`: same address;
  - `k_h1`: same first house number;
  - `k_rcore`: S1s whose core equals the row's core.
- **Structure:** `core_eq`, `acr_eq` (row equals the S1's acronym), lengths, counts of numbers on each side, numbers in common, `h_exact` (the row contains the S1's house number).
- **Flags:** `src3` (source S3), `f_handle`, `f_domain`, `f_native`, `f_alias`, `f_noaddr`.
- **13 rapidfuzz scores:**
  - on names: ratio, token-set, token-sort, partial, Jaro-Winkler, and ratio/partial on space-concatenated cores;
  - on addresses: ratio, token-set, token-sort, partial;
  - on house numbers: partial and ratio.
  - The row name is the alias right side if present.
- Full definitions are in Appendix A.

**Separation check** (India chunk, true vs non-true pairs):

| Feature | Non-true | True |
|---|---|---|
| `cc` | 0.35 | 0.85 |
| `ca` | 0.06 | 0.82 |
| `rk_cc` | 8.4 | 1.03 |
| `h_exact` | 0.06 | 0.82 |
| `nn_common` | 0.16 | 1.73 |
| `a_tset` | 42 | 94 |
| `n_tset` | 70 | 94 |

### 3.5 Stage-1 model (`model.py`)
- **LightGBM (MIT):** binary objective, lr 0.08, 255 leaves, `min_data_in_leaf` 200, `feature_fraction` 0.8, bagging 0.7 each round, λ₂ 1.0, `max_bin` 255, seed 42.
  - Early stopping after 100 rounds, capped at 6,000.
  - The validation set is **2M rows of the other fold**.
- **Training pool:** a *uniform* sample of `BER_POOL` pairs (8M on Kaggle), which keeps probabilities calibrated to the base rate.
  - M1: 167.3M pairs → 8M pool with 358,925 positives.
  - Iterations 271 / 270; logloss 0.00587 / 0.00605.
- **2-fold cross-fitting by S1 hash.**
  - Model *k* trains on fold *k*.
  - OOF predictions for fold *k* come from model 1−*k*.
  - Test = **mean of both fold models**.
- **Top gain (M1):** `rk_cc2` 4.1M ≫ `margin_cc` 0.56M > `h_part` 0.43M > `rk_cc` > `gap_cc2` > `nn_r` > `a_tset` > `n_ratio`…
- **Sharding (M2 onwards):** `fit1` trains the two fold models. Five `pred_shard` kernels predict train OOF and test in parallel, then `gather1` concatenates.
  - On 4 cores, stage-1 OOF prediction on 167M pairs takes 37 min, and test prediction (2 models × ~165M pairs) about 75 min.

### 3.6 Cross-encoders (`ce.py`, GPU)

| | CE-1 (v1: model2, M2, M3) | CE-1 hard-negative (M4, M5) | CE-2 raw text (M5) |
|---|---|---|---|
| Base | `intfloat/multilingual-e5-small` (MIT, 118M), 1-logit head | same | same |
| Text | normalised `nm \| ad` of S1 vs row | same | **raw** `business_name \| business_address` (casing, native script, source formatting kept) |
| Positives (per fold) | 1.2M true pairs among the top-5 `cc` candidates | same | same |
| Negatives (per fold) | 1.2M random non-true top-5 `cc` look-alikes | **600k random look-alikes + 600k hardest stage-1 OOF false candidates (highest p)** | same as hard-negative |
| Training | 1 epoch, batch 256, lr 5e-5, AdamW (wd 0.01), linear warm-up 3%, grad clip 1.0, fp16, DataParallel T4×2, max 128 tokens, 9,375 steps | same | same |
| Folds | Same `hash(si) % 2` as the GBDT. Train pairs are scored by the *other* fold's model; test = mean logit of both. | same | same |
| Pairs scored | Stage-1 uncertain band only (M1's p ∈ [0.001, 0.999]): **6.27M train + 6.88M test** | **every stage-2 pair** (band 0.002–1.01) | every stage-2 pair |
| Validation | **AUC 0.99982, logloss 0.0175 / 0.0189** on 100k other-fold top-5 pairs | — | — |
| Wall time | about 50 min per fold (training), 25 + 53 min scoring | train + score ≈ 2 h 15 min | train ≈ 3 h (both folds in parallel), score 1.8 h |
| Stage-2 top gain | model2: `ce` 17M ≫ `q` 1.2M | M4: `ce` 19.4M ≫ `q` 1.4M | M5: `ce2` 18.6M ≫ `ce` 2.9M ≫ `q` 0.38M |

**Why it helps:** the uncertain band is dominated by typos, transliterations, shuffles and abbreviations, where hand-built string features saturate.

**Why it cannot fix everything:** pairwise-ambiguous links, such as no address plus a crowded name, are not decidable from a single pair.

### 3.7 Stage-2 collective model (`stage2.py`, `model.train2`)

**Input set.**
- Every stage-1 pair with **p1 ≥ 0.002** (`P_MIN`). This is exactly the set written to `candidate_pairs.tsv`, 9,531,458 test IDs (5.50 per S1) for M2–M5.
- Pairs with p1 < 1e-4 (`P_FLOOR`) are dropped before aggregates are computed.
- On train, 4,183 true links (0.07%) are cut at this step.

**Features added on top of the 50 stage-1 features:**
- **Owner competition** (per row *r*):
  - `p`: stage-1 probability;
  - `prk`: rank of this S1 for the row;
  - `psum_r`: the row's total stage-1 mass;
  - `p_other`: best *other* S1's p;
  - **`q = p / max(1, psum_r)`**.
- **Cluster context** (per S1 *s*). The "owners" are rows whose top S1 is *s* with p ≥ 0.3 (`OWN_MIN`).
  - `cl_mass` (sum of their q), `cl_n`, `cl_n2`, `cl_n3`;
  - `cl_mass_ex` and `cl_n_ex`, which exclude the row itself. They are compared against the S2 ≤ 5 / S3 ≤ 6 caps.
- **Sibling evidence** (pair (*r*, *s*) against the other confident owners *r′* of *s*, bucketed by S1 to bound RAM):
  - `sib_n_max` / `sib_a_max`: best name and address token-set similarity to a sibling;
  - `sib_nq_max` / `sib_aq_max`: the same, weighted by the sibling's q;
  - `sib_n95` / `sib_a95`: sibling counts at ≥ 95;
  - **`sib_pert`**: the number of siblings that share a number with *r* which *s* lacks, i.e. **the same perturbed house number means the same generator instance**;
  - `sib_pert_same`: the same, restricted to the same source.
- **Sibling margins** (M2 onwards): `gap_sib_nq`, `gap_sib_aq`, `gap_sib_pert`, `gap_cl_mass` and `rk_sib`. They answer: which of the row's competing, same-named S1s has siblings that look like this row?
- **Rival groups** (M2 onwards): plausible candidates of *s* (p ≥ 0.01) that are not its confident owners.
  - `riv_n`, `riv_mass`, `riv_n95`;
  - **`riv_pert`**: rivals sharing a non-S1 number with *r*;
  - `riv_twin`: a shared non-S1 number plus name ≥ 85.
  - Near-twin distractor entities arrive as groups that share *their own* house number.
- **Generic-word features** (M3 onwards, `BER_GENERIC=1`). The vocabulary is learned per country from that split's S1 names, label-free: a token is generic if it appears in ≥ 0.2% of S1 names (India 183 tokens, France 109, US 244).
  - `gen_drop`, `gen_add`, `gen_swap`;
  - `dist_common`, `dist_jacc`, `dist_eq` on the distinctive tokens.
- **Cross-encoder logits:** `ce` and `ce2`. Nulls are allowed, and LightGBM handles them.

**Training.** Same LightGBM parameters and folds, trained on stage-1 **OOF** p. From M2 onwards the pool is 12M, which covers all 8.81M pairs; M1 used an 8M pool drawn from 10.3M pairs.

| Run | Iterations | Top gain after `ce`/`q`/`p` |
|---|---|---|
| M1 | 974 / 871 | `sib_pert`, `psum_r`, `cl_mass_ex`, `n_top1_s1`, `p_other` |
| M4 | 274 / 304 | — |
| M5 | 242 / 241 | `prk`, `rk_sib`, `sib_pert`, `gap_sib_nq` |

**Measured value:** stage 1 → stage 2 is **+0.0082** on M1. Misses fell from 2.52% to 1.77% of links, and false links from 0.44% to 0.25%, once CE and the collective features were in (M3).

### 3.8 Decision layer (`decide.py`)
1. Drop pairs with p < 1e-4.
2. Compute `q = p / max(1, Σ_s p)` per row.
3. **One owner per S2/S3 row:** keep only the row's argmax-p S1.
4. For each S1, sort its owned rows by q (descending) and choose the prefix length *k* (0 allowed) that maximises **exact expected F0.5 under independence**:

   E[F] = Σ_{a,b} P(TP-in-prefix = a) · P(true-outside = b) · 1.25a / (0.25(a+b) + k)

   Both distributions are Poisson-binomial, computed by dynamic programming in numba.
   - *k* = 0 is scored as P(no true links).
   - Self-checks: [0.99, 0.98, 0.97] → k=3; [0.05] → k=0 (predict empty); [0.99, 0.9, 0.3] → k=2.
5. Return the kept pairs and each S1's expected F.

On 10M pairs this takes seconds, so the decoder runs in every CV evaluation.

### 3.9 Test-time calibration (`calibrate.py`)

**Why it is needed.** The model is calibrated to train's distractor rate and to train's kinds of look-alikes. Test has more distractors (≈ 40% vs 26%) and new twin types. Density matching (DROP16) fixed the rate but not the *type* of twins.

**Seen countries** (in the run's train OOF): the difference-in-differences prior shift.
1. Take each row's **best pair**. Bin by p into 47 logit-spaced bands, p ≈ 0.01 … 0.9999, each 0.3 logit wide.
2. For each band, compute ratio_c[b] = (test rows per S1) / (train-OOF rows per live S1).
   - The generator's true links per S1 are the same in both splits. So **excess rows in a band are test-only distractors**.
3. **Anchor** = the seen country with the smallest *mid-band excess*: the row-weighted mean ratio over p ∈ [0.3, 0.93]. It gets k = 1.
   - India was the anchor: excess 1.005 on M4, 0.980 on M5.
4. For the other country *c*, compute precision_c[b] = p̄_c[b] × ratio_anchor[b] / ratio_c[b].
   - Take log k_b = **min(0,** logit(precision) − logit(p̄)**)**, because a deficit is not evidence.
   - **k_c** = exp of the row-weighted mean over bands p ∈ [0.9, 0.995] with ≥ 200 rows in both countries, capped at 1.
   - The DiD cancels artifacts common to both countries: test p is the mean of two fold models while OOF p comes from one model, and the density is simulated.
5. Rescale odds: p′ = kp / (kp + 1 − p). Then decode.

**Unseen countries** (not in the train OOF): empty-rate matching.
- Pick k from the grid `KS = (1, 0.7, 0.5, 0.35, 0.25, 0.18, 0.13, 0.1, 0.07, 0.05)` so that the decoded empty-S1 rate is closest to the **seen-country empty rate + DELTA (0.0015)**.
- On M1 this rule picked k = 0.13, next to the leaderboard-validated 0.15.

**Null tests** ("test" = train-OOF fold 1, "train" = fold 0, so there is no shift):

| Estimator | Null result | Verdict |
|---|---|---|
| Isotonic band→precision map | US F 0.98791 → 0.97872 (−85k links); India 0.98682 → 0.98525 | **Failed.** Fold-model sharpness at p > 0.93 made the fit pool to a flat 0.75. |
| Single-difference band-k over [0.3, 0.99] | US k = 0.683, India k = 0.452 | **Biased** |
| **DiD vs anchor with per-band clip** | **k = 1.000, F unchanged** | **Adopted** |

**Results:**

| Model | Country | Estimate | Outcome |
|---|---|---|---|
| M4 | US | Per-band k 0.068 0.060 0.053 0.060 0.053 0.065 0.054 0.067 0.061 → **k = 0.060**. The flat profile marks a genuine prior shift. | Links/S1 3.434 → 3.309 |
| M4 | France | k = 0.05, **at the grid floor** | Empty 5.07% → 5.98%; −65k links |
| M5 | US | Bands 0.047–0.077 → k = 0.060 | Empty 5.89%, 3.3171 links/S1 |
| M5 | France | k = 0.10 (M5 over-links France less) | Empty 5.91%, 3.1666 links/S1 |
| M5 | India | Anchor, unchanged | Empty 5.86%, 3.3466 links/S1 |

**Measured versus predicted.** For sub08 → sub11, the model predicted **+0.0043** (global k) and a stratified back-test predicted +0.0033. The leaderboard gave **+0.0015**. Decode-time calibration therefore over-estimates the shift by 2–3×. §8.3 explains why.

### 3.10 Output, validation and staging
- `run.write_tsv` writes one row per S1 in file order, with sorted comma-joined IDs, `quote_style="never"`.
- `predict_and_write` writes `output/<tag>/{matching_results,candidate_pairs}.tsv`. The candidates are the stage-2 input set, so matches are a subset by construction.
- `finalize.py <slug> <name> [tag] [--all]`:
  1. fetches only `matching_results.tsv` and the kernel log (≈ 40 MB instead of ≈ 5 GB);
  2. retries up to 3 times, because Kaggle downloads can truncate;
  3. requires ≥ 1,732,545 lines;
  4. prints CV lines from the log;
  5. runs the **official validator**;
  6. stages to `submissions/<name>/`.
- `calibrate.py` writes only `matching_results.tsv`, using the same pairs, so the model kernel's `candidate_pairs.tsv` remains valid.

### 3.11 Density matching (`BER_DROP=16`)
- About 16% of train S1s are treated as **absent**, via a hash of `si` with seed 49. Their S2/S3 rows become distractors and their links leave the ground truth.
- The distractor share goes from 26% to ≈ 37.8%, against ≈ 40% on test. Live train S1s: US 1,112,057 and India 741,709; live links 6.41M.
- It is applied in `feats.py`:
  - candidates of dropped S1s are removed **before** ranks, gaps, margins and per-S1 aggregates are computed;
  - rarity counts use only live S1s.
- `model.gt_idx` removes dropped links, and `evaluate` excludes dropped S1s.
- Effect: CV became honest (harsher) and test predictions more conservative (5.891M → 5.872M links).
- Caveat: blocking was **not** re-run without the dropped S1s (finding A6).

---

## 4. Infrastructure, compute and orchestration

### 4.1 Hardware
| Resource | What it was used for | Notes |
|---|---|---|
| **Laptop** (Windows 11, 8 cores, 21 GB RAM, no GPU) | Orchestration, calibration, light analysis | Every heavy local job runs under `memguard.py` with a 1.5–3 GB free-RAM floor |
| **Kaggle CPU sessions** (5 concurrent; 4 vCPU Xeon 2.2 GHz, 31 GB, 20 GB `/kaggle/working`) | Blocking, features, all LightGBM | Free. GPU sessions (2 concurrent, also 4 vCPU) were used as extra CPU boxes. |
| **Kaggle GPU** (T4×2, 30 h/week, reset Sat 05:30 IST) | CE training and scoring | **≈ 4.5 h used in the current quota week** (4.54 h used / 25.46 h left), plus ≈ 3–5 h the previous week. Never the bottleneck. |
| **AWS** (us-east-1) | Planned for 64-vCPU spot runs | vCPU quota was 5, raised only to 8; the 64-vCPU request stayed open. **No instance launched; $0 compute.** A private S3 bucket holds ≈ 1 GB (pennies). |

### 4.2 The laptop crash (Fri ~19:40 IST) and the re-plan
- **Cause:** RAM exhaustion. Three heavy jobs ran at once with about 9 GB free:
  - v1 test blocking (~3.5 GB);
  - v1 train features (~6 GB);
  - a full-train recall query (a window rank over ~155M pairs).
- **Response:**
  - All heavy work moved to Kaggle.
  - Processes are sharded, not forked (polars plus fork can deadlock).
  - Every parquet write is atomic (`.tmp` then `os.replace`).
  - `memguard.py` guards laptop jobs.
  - Kaggle's API cannot provision the big TPU host: a `TpuV5E8` push silently gets the 4-vCPU CPU image (kaggle-cli #1197).

### 4.3 Kaggle orchestration (`kaggle_runner.py`)
- **Kernels:** each is a private script kernel that:
  - embeds the exact `src/` snapshot as a base64 tarball;
  - pip-installs the pinned requirements;
  - symlinks every input parquet (private datasets plus upstream kernel outputs) into `BER_WORK`;
  - runs its job list with per-batch env (`BATCH_ENV`);
  - exports only newly written files — **even when a job fails**.
- **DAG edges:** `kernel_sources`. `chain(*batches)` pushes, retrying for up to 2 h while the 5-session cap is full, waits for COMPLETE, and moves on. A `=batch` means wait only.
- **Private datasets:** `qtpies-er-data` (raw parquet), `qtpies-er-norm` (v2 normalised), `qtpies-er-cand-test` (test candidates).
- **Lessons learned:**
  - Kaggle output downloads sometimes arrive **truncated** (a corrupt parquet footer). A chunk validator re-fetches.
  - Status right after a push can still show the previous version, hence a 90 s delay.
  - Running kernels expose no logs through the CLI.

### 4.4 Stage timings (4 vCPU Kaggle sessions)

| Stage | Wall time |
|---|---|
| Train blocking | 7 shards, ~14 min |
| Test blocking | France and India on the laptop; US in 3 Kaggle shards |
| Train features | 5 shards, ~15 min |
| Test features | 4 shards |
| M1 kernel (single, un-sharded) | Stage-1 fit 12 min · OOF 37 min · stage 2 ~44 min · test predict ~95 min → **3 h 10 min total** |
| M2 onwards (sharded stage 1, cached) | Stage-2 kernels 30–45 min (M4 39 min, M5 30 min) |
| CE-1 training | ≈ 50 min per fold (T4×2) |
| CE-4 train + score | ≈ 2 h 15 min |
| CE-2 train | ≈ 3 h |
| CE-2 score | ≈ 1.8 h |

---

## 5. Chronological experiment log

All times IST.

| When | Experiment | Hypothesis / goal | Result | Verdict |
|---|---|---|---|---|
| Fri 01:20–03:00 | EDA plus briefing: exact-key blocking, ceiling analysis, noise profiling | Understand the generator | Exact keys stall at 76.8%; ceiling 0.993–0.995; facts in §2 | Foundation |
| Fri 17:40 | Mission start: deadline Sun 17:30, target 0.99, turn-based loop | — | — | — |
| Fri 18:00–18:40 | Normaliser v1, scorer, blocking benchmarks B1–B3, decoder, models | Fast, high-recall blocking | Global df cap wins (B3); cap 3k for v1 | Kept |
| Fri 18:40 | v1 full-train blocking (local) | — | First India chunk recall@15 = 0.9793 | Superseded by v2 |
| Fri 19:00–19:40 | Normaliser v2, `k_h1`, stage-2 smoke on synthetic data | French forms, admin swaps | — | Kept |
| Fri ~19:40 | **Laptop crash (OOM)** | — | v1 abandoned mid-run | Moved compute to Kaggle |
| Fri 21:00 | v2 end-to-end on a 3% subset | Plumbing | Stage 1 0.9824 → stage 2 0.9869 | Plumbing OK (subset too easy) |
| Fri 21:25–22:00 | **v2 blocking on Kaggle** | ≥ 98% recall | **0.98436** | Kept |
| Fri 21:54–22:55 | **CE-1 training** | Uncertain-band discrimination | AUC 0.99982 | Kept |
| Sat 00:54 | "Quick" stage-1-only submission path | Faster first submission | Reverted at the user's request ("focus on the full submission") | Dropped |
| Sat 01:58 | **M1** (stage 1 + stage 2) | First full model | s1 0.97269 → **s2 0.98092 (+0.0082)** | Kept; **sub01** |
| Sat 02:00 | OOF error anatomy (M1) | Where is the loss? | 87% of false links are distractor rows; misses are 52–57% address-less; 91% of address-bearing misses are the row's top candidate but kept out (median p 0.44) | Led to M2 features |
| Sat 02:10 | **sub01 → 0.969** | CV mix 0.9814 → predicted 0.976 | Gap −0.0124 | Diagnose |
| Sat 02:11 | Naive density simulation on OOF (decode-time drop) | Explain the gap | Blind to the shift (0.98088 vs 0.98092): features still saw the dropped S1 | Replaced by DROP16 in features |
| Sat 02:13 | Odds ×0.6 for all countries (`sub02_k06`) | Global over-confidence | Superseded by DROP16 | Not submitted |
| Sat 02:21 | **M2 code**: DROP16, sibling-margin features, sharded stage 1 | Test-like density | — | Kept |
| Sat 02:26 | **Stage 3** (collective features from stage-2 p) | Iterate | Smoke 0.98408 vs 0.98571 (s2ce) | **Dropped** |
| Sat 02:29 | **Rival-group features** | Near-twin groups share their own number | Smoke +0.0003 | Kept |
| Sat 02:37 | Blocking v3 bench (suffix strip, number×token) | More recall | +0.0006 at +38% cost | **Dropped** |
| Sat 02:48 | **France-empty probe → 0.836** | Measure France F | France ≈ 0.94; US+India ≈ 0.973 | Informative |
| Sat 03:19 | **France odds ×0.15 → 0.971** | France over-links | +0.002 | Kept (became `calibrate.py`) |
| Sat 03:20 | Generic-word swap analysis on train | Can train teach "swap = twin"? | Swap rate: true links 0.52% vs false links 2.11%; identical-distinctive swaps 0.22% vs 0.57% | Built M3 |
| Sat 03:22 | `calibrate.py`: France empty-rate matching | Label-free France k | Reproduces k ≈ 0.13–0.15 on M1 | Kept |
| Sat 03:53 | **model2**: M1 stage 1 + CE + rival + sibling-margin | CE value | **0.98715 (+0.0062)** | Kept (ablation) |
| Sat 03:57 | **M2** | Test-like density | 0.98641 (harsher world) | Kept |
| Sat 04:14 | **M3**: + generic features | Curb France twins | +0.0001; France empty 5.05% vs 5.12% (unchanged) | Weak |
| Sat 04:15 | avg(M2, M3) | Ensemble | 0.98651 (flat: same stage 1, correlated) | **Dropped** |
| Sat 04:18 | Twin-signature audit (M3 test) | Is the France twin excess real? | France 3.26% of kept links vs India 0.89%, US 0.27%; train precision of such links 99.65–99.75% (normal noise in US/India) | Targeted France drop only ≈ +0.0005 → not pursued |
| Sat 05:35 | **m3t**: stage-2 lr 0.04, 511 leaves, min_data 100, ff 0.7, patience 200 | Capacity | 0.98654 (flat) | Stage 2 is information-limited |
| Sat 05:40 | M3 error breakdown | — | FN_block 104,470 · FN_model 113,734 · FP 15,661 · TP 6,196,661 | — |
| Sat 07:13 | **M4**: CE with hard negatives, scoring all stage-2 pairs | Better CE | **0.98749 (+0.0010)** | Kept; **sub08** |
| Sat ~10:00 | **sub08_m4_frcal → 0.977473** | — | +0.0065; gap to CV ≈ −0.011 unchanged | Diagnose |
| Sat 10:10 | Decoder self-estimate and density diagnostics | Over-confidence on test? | India test = train; US 19× cap violations; France 50× | Found the US shift |
| Sat 10:15 | US cap-violating clusters inspected | — | Twins: name + Co/LLC/Ltd/Corp, number +1…20, p 0.78–0.94 | Twin fingerprint |
| Sat 10:42 | Isotonic band map | Map p → test precision | **Failed** the null test | Dropped |
| Sat 10:45 | Single-difference band-k | — | **Biased** null (0.68 / 0.45) | Dropped |
| Sat 10:48 | **DiD vs anchor** | Cancel common artifacts | Null k = 1.000; US k = 0.060 | Kept; **sub11** (staged 10:50) |
| Sat ~11:30 | **sub11 → 0.979** | Predicted +0.0043 | **+0.0015** | Kept; lever near-exhausted |
| Sat 11:30–12:30 | Loss budget, anatomy, signature×band excess, US audit, stratified back-test | Where the 0.021 goes | §8–§9 | Analysis |
| Sat 11:53 | **M5**: + CE-2 raw text | Diversity | **0.98820 (+0.0007)** | Kept; **sub12** staged |
| Sat 12:17 | Leader at 0.99056 | — | Our "≈0.005 unfixable" claim was **wrong** (§15) | Re-examined |
| Sat 12:19 | Leakage re-check (IDs, order, adjacency) | Hidden signal? | None | — |
| Sat 12:20 | Ambiguity bucket: model ranking and row-count prior | Is "ambiguous" really ambiguous? | 1-candidate rows: 12,462 at median p 0.56 (fixable); 2 candidates: ranked right 62%; count prior weak | Opportunity noted |
| Sat 12:22 | Blocking-miss taxonomy | What could re-blocking recover? | §10 | — |
| Sat 12:24 | Address-only pass on misses | — | Recall@10 India 20.4%, US 32.7% → ≈ 10–13k rows | ≈ +0.0007 |
| Sat 12:26 | Crowding, train vs test | — | US test is less crowded | Noted |
| Sat 12:30 | **Twin fingerprint** (label-free) | Parameters for a synthetic twin world | §8.3 | Basis of the next plan |

---

## 6. What worked, ranked by measured gain

| Rank | Change | Gain | Measured on | Why it worked |
|---|---|---|---|---|
| 1 | **Collective stage 2** (owner competition, cluster mass, sibling agreement) | **+0.0082** | CV (M1 s1 → s2) | Resolves *which* same-named S1 owns an address-less or ambiguous row from the other rows; `sib_pert` exploits the generator reusing a perturbed number within a source |
| 2 | **Cross-encoder feature** (e5-small, 2-fold OOF) | **+0.0062** | CV (M1 s2 → model2) | Learns typos, transliteration, abbreviations and shuffles beyond hand-built similarity; AUC 0.99982 on hard look-alikes |
| 3 | **France calibration** (odds ×0.15, then empty-rate matching) | **+0.0020** | Public (probe A/B) | France over-linked on twins the model never saw in training |
| 4 | **US prior-shift calibration** (DiD vs India) | **+0.0015** | Public (clean A/B) | US test contains ~8× more twin decoys scored 0.78–0.98 |
| 5 | **Hard-negative CE scoring all stage-2 pairs** | +0.0010 | CV (M3 → M4, same world) | The CE learns exactly the stage-1 model's confusions; no missing CE values |
| 6 | **Raw-text CE-2** | +0.0007 | CV (M4 → M5) | Casing, native script and source formatting carry signal that normalisation removes; diversity from CE-1 |
| 7 | **Blocking pass 2** (cap 10k for uncertain rows) | ≈ +0.45% recall (≈ +0.0013 F, *estimate*) | Recall | Common n-grams dropped by the 3k cap still matter for short or generic names |
| 8 | **Density matching (DROP16)** | Honest CV; test links 5.891M → 5.872M | CV/test | The model and features see test-like distractor density; the public effect was never isolated |
| 9 | **Expected-F0.5 decoding** plus one owner | Foundational (not ablated) | — | The metric-optimal subset per S1, including the empty set; respects one-owner |
| 10 | **Learned transliteration** plus admin-component removal | Foundational | — | Native-script names become matchable; département/région swaps are neutralised |

## 7. What did not work, and why

| Experiment | Result | Root cause / lesson |
|---|---|---|
| **Generic-word swap features** (M3) | +0.0001 CV; France over-linking unchanged | Train has almost no "generic-word swap = twin" signal: swaps occur in only 0.52% of true links, and twin-signature links are 99.7% precise in US/India. You cannot learn a pattern the training data doesn't contain. |
| **Stage-2 hyperparameter tuning** (m3t) | 0.98654 vs 0.98652 | Stage 2 is *information*-limited, not capacity-limited |
| **Averaging M2 and M3** | 0.98651 (flat) | Same stage 1; highly correlated |
| **Stage 3** (collective features recomputed from stage-2 p) | Smoke 0.98408 vs 0.98571 | Over-fitting to stage-2 decisions; fewer pairs survive |
| **Blocking v3** (suffix strip; number×token pairs) | Recall 0.9789 / 0.9796 vs 0.9790 | Misses are not solvable by n-gram tweaks (§10) |
| **Naive decode-time density simulation** | Blind to the shift | Features were computed with the dropped S1s present; the drop must happen before features |
| **Odds ×0.6 for all countries** | Not submitted | Superseded by DROP16; India needed no correction |
| **Isotonic band recalibration** | Null test: US F −0.009 | Fold-model sharpness artifacts at p > 0.93 |
| **Single-difference band-k** | Biased null (k 0.68 / 0.45) | The same artifacts; needs a difference-in-differences |
| **Stratified (signature × band) decode** | Back-test +0.00084 over sub11, likely ≈ +0.0004 real | The calibration lever is exhausted; the DiD over-states test false links by 2× |
| **Row-count prior** for ambiguous no-address rows | The true S1 has the fewest kept rows in only 15–42% of cases | The generator's per-source counts are too flat (§2.2) |
| **Address-only blocking pass** (analysis) | ≈ 10–13k of 100k misses recovered (≈ +0.0007) | 64% of misses have no address, or only a city, plus a crowded name |
| **Quick stage-1-only path** | Reverted at the user's request | Focus on the full two-stage submission |
| **AWS** | Never launched | vCPU quota (5 → 8; 64 never approved) |

---

## 8. What we learned about the test set

### 8.1 More distractors
- Test has 5.75 S2/S3 rows per S1 against 4.68 in train.
- With the generator constant of 3.46 links per S1, **≈ 40% of test rows are distractors**, against 26% in train.
- Sub01 assigned only 58–62% of test rows (train: 74% matched).
- The uncertain band (0.1 < p < 0.9) holds 7.3% of test pairs against 5.5% on train.

### 8.2 India behaves like train
- Links per S1: 3.346 on test vs 3.347 on train OOF.
- The per-band test/train rows-per-S1 ratio is ≈ 1 in every band.
- The decoder's expected F is the same (0.9959 vs 0.9960).
- India is therefore the calibration **anchor**.
- *Assumption:* India's test F ≈ its OOF F (0.9868 for M4), which is used in the leaderboard accounting (§8.6).

### 8.3 US: twin decoys, a prior shift in one cell
- **Symptoms:**
  - US links per S1 were 3.434 on test vs 3.354 on train OOF.
  - Clusters breaking the S2 cap occur **19×** as often as in train (101 such clusters on M4).
  - Rows per S1 run about **2×** train at p 0.6–0.97.
- **Cell analysis** by signature × band, comparing test/train rows per 1k S1 with M4:
  - The excess sits almost entirely in **"house number differs, core name identical"**: 49.6 excess rows per 1k S1 at p 0.3–0.9 (7.8×) and 22.7 at p 0.9–0.98 (5.2×).
  - A smaller excess is in "same number, same name" at p 0.3–0.9 (6.9×).
  - There is no excess in address-less or number-free cells.
- **Twin fingerprint**, measured label-free on M5 in the cell (num≠, name=, p 0.3–0.98):

| | Train OOF | **US test** |
|---|---|---|
| Rows per 1,000 S1 | 8.7 | **71.6 (8.2×)** |
| Legal suffix added (LLC/Co/Ltd/Corp…) | 69.5% | **94.2%** |
| In a group sharing its own number | 41.4% | 46.3% |
| House-number shift \|δ\| ≤ 20 | 41.5% | **79.6%** |
| Shift is positive (δ > 0) | 43.2% | **89.3%** |
| Median \|δ\| | 100 | **7** |
| \|δ\| histogram (≤2, 2–5, 5–10, 10–20, 20–50, 50–200, >200) | 2123 / 610 / 515 / 759 / 335 / 1372 / 3944 | 9604 / 12288 / 7809 / 8120 / 4105 / 1110 / 4469 |

- **Example.** The S1 is "Bright Chemical Innovations | 6182 Diamond Ave".
  - True rows at 6182 score p = 1.000.
  - Twins "… Corp / LLC / Ltd | 6183 Diamond Ave" score p 0.86–0.88 and were linked before the correction.
- **Why the global correction under-delivered.** sub11 removed 83,165 US links. Of those, ≈ 43,909 were estimated false, i.e. twins, and ≈ **39k were probably true** links from unshifted cells: address-less rows and "same number, different name".
  - About 8.5k estimated false links remain among the kept links with p < 0.999.
- **US test is less crowded:**

  | | Train (live S1s) | US test |
  |---|---|---|
  | S1s sharing a core name | 48.5% | 42.7% |
  | Shared by ≥ 5 | 26.7% | 22.2% |
  | Mean same-core S1s | 27.0 | 16.9 |
  | Address-less rows | 3.59% | 2.89% |

  The model is probably slightly too timid on name-only US rows, a small effect in the opposite direction.

### 8.4 France: unseen country, a different twin type
- **Probe:** France F ≈ 0.05 + (0.969 − 0.836)/0.15 ≈ **0.94** on M1, rising to ≈ **0.955** after odds ×0.15.
- **Over-linking** (M1): 3.447 links per S1 and 5.09% empty, against 5.58% singletons.
  - Share of kept links with p < 0.9: France 2.43% vs India 1.88% and US 1.59%.
- **The removed links**: 38.6k of 894k at k = 0.15, with p quantiles q10/50/90 of 0.78/0.89/0.95.
  - They share the distinctive token and the **same address**, with **one generic French word swapped**: Graines *Ecole* → *Parents*, *Collège* → *Ehpad*, *Union* → *Élémentaire*, *Jeunes* → *Musique*, *Sport* → *Primaire*.
  - Some also shift the house number.
- **Signature excess** (M4):
  - "same number, different name": **12×** at p 0.3–0.9, 6.5× at 0.9–0.98, 1.8× at 0.98–0.999;
  - "same number and name": 8.5× / 7.4× / **4.6×**;
  - "number differs, name identical": 11.4× at p 0.3–0.9;
  - with a **deficit** in the top band. So France's true links also score **lower** than in train, and band matching is not valid for France.
- **Twin-signature links** (generic swap with identical distinctive tokens): 3.26% of France kept links at k=1, 1.76% at k=0.07. Of those kept at k=0.07, 6,564 are grouped and 22,064 lone.
- **Calibration history:** k = 0.15 (probe) → 0.13 (M1 rule) → 0.07 (model2, M3) → 0.10 (M2) → **0.05, at the floor** (M4) → 0.10 (M5).

### 8.5 Diagnostics that proved useful (all label-free)

| Diagnostic | Test | Train OOF |
|---|---|---|
| Predicted S2 per S1 | France 1.64 · India 1.63 · US 1.66 | 1.62–1.63 |
| Predicted S3 per S1 | France 1.75 · India 1.72 · US 1.77 | 1.72–1.74 |
| Share of S1s exceeding the S2 cap | US 0.015% · France 0.040% | 0.0008% |
| Decoder expected F | India 0.9959 · US 0.9917 · France 0.9788 | US 0.9945 · India 0.9960 |

The train expected-F values are themselves over-confident against the actual 0.9879 / 0.9868.

### 8.6 Leaderboard accounting (sub11 = 0.979)
- Test S1 shares: India 46.75%, US 38.27%, France 14.98%.
- With India ≈ 0.9868 (*assumption*): 0.3827·F_US + 0.1498·F_FR = 0.5177.
- So **US ≈ 0.975–0.983 and France ≈ 0.945–0.965**.
- Test-only loss relative to CV is ≈ **0.009**, split between US twins (≈ 0.002–0.005) and France (≈ 0.005–0.008).

---

## 9. Where the remaining loss is

### 9.1 Train-side loss budget (M4 OOF 0.98749, DROP16)
"Oracle gain" = the F0.5 gained by fixing only that error class.

| Category | Rows | Oracle gain (all) | US | India |
|---|---|---|---|---|
| **A. Not blocked** (true S1 never a candidate) | 100,287 | **+0.00529** | +0.00383 | +0.00748 |
| B. Cut at stage 1 (p1 < 0.002) | 4,183 | +0.00024 | +0.00024 | +0.00023 |
| C. Stage 2 not kept, row has an address | 27,088 | +0.00151 | +0.00173 | +0.00117 |
| **C. Stage 2 not kept, no address** | 81,288 | **+0.00384** | +0.00449 | +0.00287 |
| D. Wrong S1, other name | 434 | +0.00010 | | |
| D. Wrong S1, same name | 485 | +0.00015 | | |
| E. False link, natural distractor, with address | 3,785 | +0.00073 | | |
| E. False link, natural distractor, no address | 633 | +0.00009 | | |
| E. False link, simulated (dropped-S1) distractor | 5,357 | +0.00076 | | |

**Ambiguous vs solvable** (ambiguous = no address *and* the true S1's core name is shared by ≥ 2 live S1s):

| | Ambiguous | Solvable |
|---|---|---|
| A (not blocked) | 47,606 rows, +0.00239 | 52,681 rows, +0.00298 |
| B (cut at stage 1) | 1,500 rows, +0.00008 | 2,683 rows, +0.00016 |
| C (stage 2 not kept) | 71,977 rows, +0.00344 | 36,399 rows, +0.00190 |
| D (wrong S1) | — | +0.00025 |
| E (false links) | — | +0.00158 |

→ **Ceiling if every solvable error were fixed: 0.99431.**

Caveat: the leader's 0.9906 on *test* shows that the "ambiguous" definition, core-name equality, is too blunt. Rows still carry the legal form, the domain or handle, and sibling patterns (§15).

### 9.2 Anatomy of each bucket

**A. Blocking misses** (100,287):
- 57.7% have no address; 10.0% have a native-script name.
- The true S1's core name is shared by ≥ 2 live S1s in 78.2% of cases (≥ 5 in 67.1%).
- The number agrees in only 32.5%.
- Examples: "Physical Therapy Associates Incorporated | no address", with 615 same-core S1s; Bengali "শ্যাম সার্ভিসেস" = Shyam Services.
- Full taxonomy in §10.

**C. No-address stage-2 misses** (81,288): 88.5% crowded names. Of these, 72,699 have their true S1 among candidates with p ≥ 0.05, grouped by how many plausible S1s they have:

| Plausible S1s | Rows | Model ranks the true S1 first | Median p_true |
|---|---|---|---|
| **1** | **12,462** | 100% | **0.56** (fixable: the model is too cautious) |
| 2 | 26,475 | 61.9% | 0.49 |
| 3 | 14,766 | 46.1% | 0.33 |
| 4+ | 18,996 | 31.5% | 0.19 |

**C. Address-bearing stage-2 misses** (27,088):
- Name token-set median is 70 (p10 17): the **name was replaced by an unrelated brand at the same address** ("Cure Auto Body!" → "Halonex | 9227 Root Drive").
- In the M1 analysis, 91% of such misses were the row's top candidate but not kept (median p 0.44, against a threshold of ≈ 0.78).

**E. False links:**
- 87% are distractor rows (M1): only 13.3% had a true owner elsewhere.
- They are confident: p quantiles q10/50/90 = 0.75/0.91/0.99.
- They are near-copies of real S1s: same street, a one-letter name change, a shifted number ("zephix nano care" vs "zephay nano care" at 2 Kimberly Court).

### 9.3 Error counts across runs

| Run | TP | Blocking FN | Model FN | FP |
|---|---|---|---|---|
| M1 (natural) | 7,321,594 | 124,235 | 192,536 | 32,397 |
| M3 (DROP16) | 6,196,661 | 104,470 | 113,734 | 15,661 |

**M3 by country and source:**

| | Model FN | Blocking FN | FP |
|---|---|---|---|
| US S2 | 39,131 | 24,082 | 4,891 |
| US S3 | 37,997 | 23,156 | 5,125 |
| India S2 | 16,669 | — | 2,456 |
| India S3 | — | 36,407 | 3,189 |

Cells marked — were not captured in the logged output.

**Flag rates by error class:**

| Error class | No address (M1 → M3) | Native script |
|---|---|---|
| Blocking FN | 57% | ≈ 10% |
| Model FN | 52% → 72% | — |

### 9.4 The gap to the leader (0.99056)
- Our public score is 0.979, a gap of 0.0116.
- Our **train-level** OOF at test-like density is 0.9882. Even with zero test shift we would sit about 0.002 below the leader.
- So the leader must be better at the **base matcher**: blocking recall and the ambiguous no-address bucket. They must also handle the **test shift**: US twins and France.
- Where to look:
  - **Blocking plus the no-address bucket:** ≈ 0.009 of train loss, most of it recoverable according to the leader's score.
  - **Test shift:** ≈ 0.009.

---

## 10. Candidate search (blocking) deep-dive

### 10.1 Design rationale
- The name and address views are **complementary**: they almost never fail together (0.014%).
- The combined cosine lets a strong address rescue a mangled name, and vice versa.
- Char 4-grams over the *concatenated* core are robust to shuffles, typos and word splits.
- A **global df cap** is the only cost control that bounds postings; per-row pruning does not.
- **Pass 2** relaxes the cap only for uncertain rows (≈ 25%), keeping the cost manageable.

### 10.2 Miss taxonomy (M4, 100,287 train blocking misses)

| Category | Rows | Share | Median same-core S1s | Name similarity | Address similarity | Recoverable by… |
|---|---|---|---|---|---|---|
| Name perturbed, address weak or none | 40,392 | **40.3%** | 0 | 100* | 0 | Hard: no address and a mangled name |
| Exact core name, no address, crowded | 24,310 | **24.2%** | 51 | 100 | 0 | Not by retrieval: needs the collective or ambiguity model |
| Name perturbed, address "strong" (token-set ≥ 80) | 20,222 | 20.2% | 0 | 78 | 92 | *Partly.* Token-set over-states it; real address-pass recovery is ≈ 13% of misses |
| Native-script name | 10,049 | 10.0% | 48 | 100 | 89 | Better transliteration, or a name-agnostic address view |
| Exact core + shared number → unique | 3,301 | 3.3% | 46 | 100 | 84 | **(core, number) exact key** |
| Exact core, address without a shared number, crowded | 1,450 | 1.4% | 48 | 100 | 79 | Hard |
| Exact core, unique name | 563 | 0.6% | 1 | 100 | 0 | Exact-name key |

\*Token-set similarity between cores; the full names differ.

### 10.3 What an extra pass would recover (measured)

**Address-only TF-IDF pass** (word 1–2-grams, same family, K = 5/10/20):

| | n | Recall @5 | @10 | @20 |
|---|---|---|---|---|
| India blocking misses with an address | 32,153 | 16.9% | **20.4%** | 25.8% |
| India random true rows with an address (reference) | 50,000 | 90.7% | 92.0% | 93.1% |
| US blocking misses with an address | 10,298 | 27.7% | **32.7%** | 42.3% |
| US random true rows with an address (reference) | 50,000 | 94.6% | 96.4% | 97.5% |

That recovers ≈ **10k rows at K=10 and ≈ 13k at K=20**. The (core, number) exact key adds ≈ 3.3k. Together they are worth ≈ **+0.0007** F.

They would require re-running blocking, features, stage 1, CE scoring and stage 2 **on both train and test**: about 2 h of extra GPU time plus risk.

### 10.4 What the blocking numbers mean for the other pipeline

Compare recall on the **same definition**: the share of the 7,638,365 train links whose true S1 appears among a row's candidates. Also compare per country × source, at the same candidates-per-row budget.

| | Our value |
|---|---|
| Recall | 0.98436 at ~16 candidates per row |
| After the stage-1 cut (candidate_pairs.tsv) | ≈ 98.3% of links, at 5.50 candidates per S1 |

---

## 11. Audit findings: risks, gaps and bugs

| ID | Severity | Area | Finding | Recommendation |
|---|---|---|---|---|
| **A1** | **High** | Reproducibility | `README.md` documents `python run.py all` (norm → block → feats → train → train2 → predict). That is the **M1 recipe**: no DROP16, no cross-encoders, no generic features, no calibration. The submitted files come from the **Kaggle DAG** (Appendix D, with per-batch env `BER_DROP=16`, `BER_POOL=12M`, `BER_GENERIC=1`, `BER_CE_*`) plus `calibrate.py` on the laptop. | Document the exact DAG, env and calibrate command; add an M5 step list to `run.py`, or a `reproduce.md` |
| **A2** | **High** | Reproducibility / licence | CE dependencies are unpinned: torch and transformers are the "Kaggle image" versions; the HF model is fetched by name (`intfloat/multilingual-e5-small`) with no revision hash; `requirements.txt` lacks torch and transformers. | Pin torch/transformers, the model revision and a weights SHA; record them in `LICENSES.md`. Never commit weights. |
| **A3** | Medium | Docs | The README says "CPU only; no GPU, no internet, no pretrained models" and that artefacts came from the laptop. Its file table omits `ce.py`, `calibrate.py`, `probe.py`, `kaggle_runner.py`, `finalize.py`, `memguard.py` and `aws/`. | Rewrite before the zip |
| **A4** | Medium | CV validity | Early stopping of fold-*k*'s model uses 2M rows of fold 1−*k*, the same fold its OOF predictions are scored on. The iteration count is therefore selected on evaluation data: slight optimism, unmeasured and probably ≤ 1e-4. | Early-stop on a held-out slice of the *training* fold |
| **A5** | Medium | CV validity (stacking) | Folds are grouped by S1 only. A row's candidates span both folds. Stage-2 features of pair (*r*, *s*) in fold *k* (`p_other`, `psum_r`, `q`) use stage-1 p of (*r*, *s′*), which was produced by the model trained on fold *k*, and that model saw (*r*, *s*)'s label. This is second-order leakage; India test ≈ train suggests it is small. | Group folds by connected component of the candidate graph, or by row cluster |
| **A6** | Medium | Simulation fidelity | DROP16 removes dropped S1s *after* blocking. Candidate lists, and the pass-2 trigger (best cc < 0.6), were computed with the dropped S1 present. Simulated distractors therefore have ≤ 14 candidates and often skipped pass 2, so their context features differ subtly from real distractors'. | Re-block the train side without the dropped S1s (needed anyway for the twin world) |
| **A7** | Medium | Rules | The first `calibrate.py` (commit `2d78219`), used for `sub05`–`sub08_frcal`, selected the `"France"` label by name, as did the inline `sub03_frk015` probe. The current 2-argument fallback still does (`c != "France"`). **`sub11` and `sub12` used the 3-argument OOF mode**, where "unseen" means "absent from train OOF", derived from data. `score.py` also hard-codes `TEST_MIX`, but only for reporting. | Use only the OOF mode for the final; delete the fallback branch |
| **A8** | Medium | Calibration | France k comes from a fixed grid with a floor of 0.05, and M4 hit the floor. `DELTA=0.0015` was set by hand and validated by a single leaderboard probe (k = 0.15 on M1). | Extend the grid (a continuous search); document DELTA's provenance |
| **A9** | Medium | Determinism | Folds (`hash(si) % 2`) and the drop set (`hash(si, SEED+7)`) use polars' `hash()`, which is **not guaranteed stable across polars versions**. Reproducing the folds requires polars 1.44.2 exactly. | Keep the pin, or switch to a stable hash (e.g. xxhash of the ID string) |
| A10 | Low | Normalisation | `admin_vocab` treated `st herblain` and `st nazaire` (city spellings used by S2/S3) as admin components. The city is stripped only on the abbreviated side, making those France addresses asymmetric. | Exclude tokens that are city names in S1 (e.g. with a `saint`→`st` canonicalisation first) |
| A11 | Low | Feature coverage | In model2, M2 and M3, CE-1 scored only M1's stage-1 uncertain band. In M2 and M3 stage 1 changed (DROP16), so part of the stage-2 set had null CE. | Fixed in M4 (all stage-2 pairs scored) |
| A12 | Low | Decoder | Expected F assumes independent link probabilities and is over-confident (train: 0.9945 predicted vs 0.9879 actual for US; test: 0.9927 vs public ≈ 0.98). | Never use the self-estimate as a leaderboard predictor |
| A13 | Low | Compliance | Kaggle kernels ran with **internet on**, for pip installs and the HF model download only. Competition data was uploaded to **private** Kaggle datasets and a **private** S3 bucket (unused). Nothing is public. | State this in the methodology; delete the datasets and bucket after the challenge |
| A14 | Info | Transductive, label-free statistics used on test | Admin vocabulary, generic-word vocabulary (from test S1 names), per-band densities (calibration), France empty rate. No labels, no external data. | Document in `Documentation_template.md` |
| A15 | Info | Leaderboard exposure | Leaderboard feedback shaped exactly two things: the France diagnosis (the emptied probe) and the France direction/DELTA (the k = 0.15 probe). Everything else was chosen on CV or label-free test statistics. | Overfitting exposure: **low** |
| A16 | Info | Code hygiene | Unscheduled or legacy paths: stage 3 (`train3`/`predict3`), `predict1`, the M1 `train`/`predict` without CE. The quick path was reverted. | Keep them documented or prune before the zip |
| A17 | Info | Analysis provenance | The key analysis scripts (`calib_diag`, `bands`, `null_check`, `gain`, `budget`, `budget2`, `anatomy`, `shift3`, `us_audit`, `strat_eval`, `ambig`, `blocking`, `addr_pass`, `twins`) live only in the **session scratchpad**, a temp folder that is not versioned. `work/` is gitignored. | Copy them into the repo (e.g. `code/business_entity_resolution/analysis/`) |
| A18 | Info | Deliverables | Not done yet: the final model's `candidate_pairs.tsv` (not fetched), `Documentation_template.md` (not filled), the zip. | §14 |

---

## 12. Recommended next steps

### 12.1 Test-matched synthetic "twin world" (highest value; designed, not implemented)

**Idea.** Calibration can only drop whole confidence bands. Retraining teaches the model to separate twins from true links using shared-number, suffix and sibling evidence.

**Grounding** (the anti-overfitting rules):
1. **Every parameter comes from test statistics, label-free** (§8.3 table): δ mostly +1…+20 (median 7, 89% positive), a legal suffix added in 94% of cases, groups sharing their own number (~46%).
2. The **rate per country comes from moment-matching** the signature × band densities. India shows no excess, so it gets ≈ 0 twins, and no country is hard-coded.
3. **No invented text.** Convert 1–2 of an S1's *existing* true rows into a twin group: shift the number by δ (consistent within the group), add a suffix, and remove them from the ground truth as distractors. All of the generator's natural noise is kept.

**Pipeline.** Re-run the train side only: norm → **block** (also fixes A6) → feats → stage 1 → CE scoring of new pairs → stage 2. Test-side artefacts are reused.

**Acceptance tests** (before spending a submission):
- (a) CV on unaffected cells does not drop.
- (b) After retraining, test's twin-cell density per band looks like train's, as India's already does.
- (c) India predictions are unchanged.
- (d) The DiD US k moves toward 1.

**Cost and gain.** About 1.5 h local plus 5–6 h on Kaggle. Expected **+0.0015 to +0.004** over sub12, and it removes sub11's collateral damage (≈ 39k true links).

### 12.2 Other items

| # | Item | Expected gain | Cost / risk |
|---|---|---|---|
| 2 | Blocking additions: address-only pass (K 10–20) plus a (core, number) exact key | ≈ +0.0007 | Full train and test re-run (~2 h extra GPU) |
| 3 | Over-caution on single-candidate no-address rows (12,462 rows, median p 0.56). Candidate fixes: a "sole plausible S1 in country" feature (`k_core`-conditional), or a collective rule. | Part of C-solvable (+0.0019 total) | Low |
| 4 | France: a "same address, different name" twin world or a signature-aware decode; generic-word groups | +0.001–0.003 (least certain) | Needs 1–2 leaderboard probes |
| 5 | CV hygiene (A4, A5), stable hashing (A9), re-block without dropped S1s (A6) | Trust in CV | Low |
| 6 | Reproducibility and deliverables (A1–A3, A17, A18) | Audit readiness | **Must do before Sun 17:30** |
| — | *Not worth it:* more decode-time calibration, stage-2 tuning, more CE variants of the same kind | ≤ +0.0004 | — |

---

## 13. Checklist for comparing with the other pipeline

Report these on **identical definitions** so the two pipelines can be compared cell by cell.

| Metric | Our value (M5 unless noted) | How to compute |
|---|---|---|
| Blocking recall (links) | **0.98436** (US S2 .9881, US S3 .9893, India S2 .9845, India S3 .9743) | Share of the 7,638,365 train links whose true S1 is among the row's candidates |
| Candidates per row / per S1 | ~16 per row (blocking); 5.50 per S1 (final candidate set) | — |
| Links lost before the final model | 1.63% (104,470 of 6.41M live links, DROP16) | Blocking misses plus the stage-1 cut |
| OOF F0.5, natural density | 0.98715 (model2, best natural-density run) | Exact metric over all train S1 |
| OOF F0.5, test-like density | **0.98820** (mix 0.98900; US 0.98874; India 0.98739) | Hash-drop 16% of S1s *before* features |
| Test links per S1 / empty rate (final decode) | France 3.167 / 5.91% · US 3.317 / 5.89% · India 3.347 / 5.86% | From `matching_results.tsv`; generator truth 3.46 / 5.58% |
| US twin-cell density on test | 71.6 rows per 1k S1 (M5, raw) vs 8.7 on train | Rows' best pairs with p 0.3–0.98, number differs, core name equal |
| France "same number, different name" excess | 6–12× (M4) | Signature × band rows per S1, test/train |
| Public scores | 0.969 · 0.971 · 0.977473 · **0.979** | — |

**Suggested joint analyses:**
1. Diff the two `matching_results.tsv` files: Jaccard of link sets per country, and disagreement per signature cell. This shows which pipeline handles twins, France and no-address rows better.
2. If the other pipeline has pair probabilities, **average or stack** them with ours and decode once with `decide.py` plus calibration. Decorrelated models (different blocking or features) are where ensembling pays; our M2+M3 averaging was flat only because they shared stage 1.
3. Compare blocking miss taxonomies (§10.2). If their recall on "name perturbed, address weak" or "native script" is much higher, adopt their retrieval view as an extra pass.

---

## 14. Deliverables status and compliance

| Deliverable | Status |
|---|---|
| `matching_results.tsv` (best staged) | `submissions/sub12_m5_shift/` (validator PASS). Best scored so far: `sub11_m4_shift` (0.979). |
| `candidate_pairs.tsv` | In the final model kernel's output, `output/s2ce/candidate_pairs.tsv` (9,531,458 IDs for M2–M5). **Not yet fetched**: `python finalize.py qtpies-m5 <name> s2ce --all`. Calibration keeps matches a subset. |
| `code/business_entity_resolution/{src,README.md,requirements.txt}` | Code committed at `b7e2a37`. **README and requirements need updating** (A1–A3). |
| `Documentation_template.md` | **Not filled yet**. This audit covers every section it asks for. |
| `LICENSES.md` | Present. Add pinned torch/transformers versions and the model revision. |
| `QTpies_submission.zip` | Not built |

**Compliance summary:**
- **No external lookup.** No registries, geocoding, ER services or internet augmentation.
- **Only permissive licences:**
  - e5-small: MIT, 118M params (≤ 8B).
  - LightGBM: MIT.
  - polars and rapidfuzz: MIT.
  - scikit-learn, numpy, scipy, numba: BSD.
  - sparse_dot_topn and pyarrow: Apache-2.0.
  - anyascii: ISC.
- **Not used:** libpostal, Unidecode, paid LLM APIs.
- **No hard-coded countries** in the final decision path: `sub11` and `sub12` derive "unseen" from the train OOF. The earlier France-calibrated files named France explicitly (A7).
- **Learned artefacts come from training data only:** the transliteration table and the LightGBM and CE weights.
- **Data** lives only in **private** Kaggle datasets and a private S3 bucket. No secrets are committed; `kaggle.json`, `.env`, parquet and model weights are gitignored.

---

## 15. Lessons and mistakes

1. **Every public prediction was optimistic** (by 0.007, 0.0045 and 0.002), and the decoder's self-estimate even more so.
   - CV is measured on train-like twins. Test has different twins, which no CV built from train can see.
   - Measure **label-free test statistics** (per-band densities, signature cells) early.
2. **"Ambiguous = unfixable" was wrong.** On Sat 12:00 we said ≈ 0.005 of the loss was unfixable by anyone and 0.99 was unrealistic. The leader's 0.9906 disproves both. Our ambiguity definition (core-name equality) ignores the legal form, domain or handle patterns, sibling structure and the 1-candidate cases.
3. **Null tests save you.** Two of three calibration estimators looked plausible and failed the fold-vs-fold null. Only the DiD passed.
4. **You can't learn what the training data doesn't contain.** Generic-swap features were useless because train has no such twins. The fix is to add them to training (the synthetic twin world), not to add features.
5. **Information beats capacity.** Tuning and stage 3 were flat. CE and collective features — new information — gave every big gain.
6. **Operational:**
   - The crash came from running three heavy jobs at once.
   - Kaggle downloads truncate silently, so validate every file.
   - The API cannot give a big CPU host.
   - AWS quota requests take days, so file them early.
7. **Keep one variable per submission when attribution matters.** The sub08 → sub11 clean A/B (only US rows changed) made the +0.0015 interpretable.

---

## Appendix A: Feature dictionary

**Stage 1** (50 features; `ri`/`si` keys excluded):

| Group | Features | Definition |
|---|---|---|
| Blocking cosines | `cn`, `ca`, `cc`, `cc2` | Name part, address part and combined cosine (pass 1); combined cosine in the pass-2 (cap 10k) space |
| Row context | `rk_cc`, `rk_cn`, `rk_ca`, `rk_cc2` | Rank of this S1 among the row's candidates |
| | `gap_cc`, `gap_cn`, `gap_ca`, `gap_cc2` | Difference to the row's best candidate |
| | `n_cand` | Number of candidates for the row |
| | `margin_cc` | cc minus the best *other* S1's cc (for the top candidate: minus the 2nd best) |
| S1 context | `n_cand_s1` | Candidate rows of this S1 |
| | `n_top1_s1` | Rows for which this S1 is the top-cc candidate |
| | `gap_s1_cc`, `gap_s1_cn`, `gap_s1_ca` | Difference to this S1's best candidate row |
| Row flags | `src3`, `f_handle`, `f_domain`, `f_native`, `f_alias`, `f_noaddr` | Source is S3; `@`/`#`; domain/www; Indic script; alias wrapper; empty address |
| Rarity (live S1s) | `k_core`, `k_addr`, `k_h1`, `k_rcore` | S1s in the country sharing the S1's core; its address; its first number; the row's core |
| Structure | `core_eq`, `acr_eq` | Cores equal; row core equals the S1 acronym |
| | `len_r`, `len_1` | Core lengths |
| | `nn_r`, `nn_1`, `nn_common` | Number counts on each side and in common |
| | `h_exact` | The row's numbers contain the S1's first number |
| Name similarity (rapidfuzz) | `n_ratio`, `n_tset` | Normalised names |
| | `n_tsort`, `n_part`, `n_jw` | Cores |
| | `n_cat`, `n_catpart` | Space-removed cores |
| Address similarity | `a_ratio`, `a_tset`, `a_tsort`, `a_part` | Normalised addresses |
| Number similarity | `h_part`, `h_ratio` | S1 first number vs the row's numbers (partial) / the row's first number (ratio) |

**Stage 2** adds 37 features (87 in total for M4 and M5):
- `p`, `prk`, `psum_r`, `p_other`, `q`
- `cl_mass`, `cl_n`, `cl_n2`, `cl_n3`, `cl_mass_ex`, `cl_n_ex`
- `sib_n_max`, `sib_a_max`, `sib_nq_max`, `sib_aq_max`, `sib_n95`, `sib_a95`, `sib_pert`, `sib_pert_same`
- `riv_n`, `riv_mass`, `riv_n95`, `riv_pert`, `riv_twin`
- `gap_sib_nq`, `gap_sib_aq`, `gap_sib_pert`, `gap_cl_mass`, `rk_sib`
- `gen_drop`, `gen_add`, `gen_swap`, `dist_common`, `dist_jacc`, `dist_eq` (M3 onwards)
- `ce` (model2 onwards), `ce2` (M5)

Definitions are in §3.7.

## Appendix B: Constants and hyperparameters

| Where | Constant | Value |
|---|---|---|
| `common.py` | `SEED`; fold hash; drop hash; `BER_DROP` | 42; `hash(si, 42) % 2`; `hash(si, 49) % 100 < DROP`; 16 for M2 onwards |
| `block.py` | `K`, `CAPS`, `UNSURE`, `CHUNK`, matmul threshold | 15; (3000, 10000); 0.6; 250,000; 0.02 |
| `feats.py` | `KEEP_RANK` | 15 |
| `model.py` | LightGBM | objective binary, lr 0.08, 255 leaves, min_data 200, ff 0.8, bagging 0.7/1, λ₂ 1.0, max_bin 255, early stop 100, max 6000 rounds; validation 2M rows of the other fold |
| `model.py` | `BER_POOL` | 16M code default · 8M Kaggle default (M1 stage 1 and 2; stage 1 from M2 onwards) · 12M for stage 2 from M2 onwards |
| `stage2.py` | `P_MIN`, `P_FLOOR`, `OWN_MIN`, `RIV_MIN`, `SIB_BUCKETS`, `GEN_DF` | 0.002, 1e-4, 0.3, 0.01, 16, 0.002 |
| `ce.py` | `MAXLEN`, `N_POS`, `N_NEG`, `BS`, `LR`, `BAND` | 128, 1.2M, 1.2M (½ hard when `BER_CE_HARD=1`), 256, 5e-5, [0.001, 0.999] v1 → [0.002, 1.01] M4 onwards |
| `decide.py` | Minimum p; one-owner; β | 1e-4; argmax p; 0.5 |
| `calibrate.py` | `EDGES`, `MID`, `DEC`, `KS`, `DELTA` | 47 bands (logit −4.6…9.2); [0.3, 0.93]; [0.9, 0.995]; (1…0.05); 0.0015 |
| m3t override | `BER_LGB2` | lr 0.04, 511 leaves, min_data 100, ff 0.7, patience 200 |

## Appendix C: Files, artefacts and paths

| Path | Content |
|---|---|
| `code/business_entity_resolution/` | Pipeline code (§3.1) |
| `EXPERIMENTS.md` | Chronological experiment log (plan, blocking log, runs M1–M5, diagnoses, compute and cost) |
| `LICENSES.md` | Licence ledger |
| `work2/` (gitignored) | v2 normalised parquet, `translit.parquet`, `admin_{split}.parquet`, `cand_train/`, `cand_test/`, `gt_pairs.parquet` |
| `work/` (gitignored) | EDA and bench scripts: `eda1–5.py`, `block1–3.py`, `ceiling.py`, `bench_block.py`, `bench2.py`, `bench3.py`, `errors.py`, `diag.py`, `shift.py`, `recall_v2.py` |
| `runs/qtpies-{model,model2,m2-s2,m3,m3t,m4,m5}/` | Downloaded OOF and test predictions (`oof_s2*.parquet`, `pred_test_s2*.parquet`) plus kernel logs |
| `submissions/<name>/matching_results.tsv` | Every staged file (§1.1) |
| `kaggle/k/<slug>/` | Generated kernel sources and metadata |
| Private Kaggle datasets | `qtpies-er-data`, `qtpies-er-norm`, `qtpies-er-cand-test` |

## Appendix D: How each model was produced (Kaggle DAG)

Batches are defined in `kaggle_runner.py` and run with `python kaggle_runner.py chain <batch> …` from `code/business_entity_resolution/`.

| Batch | Kernels | Jobs | Env | Upstream |
|---|---|---|---|---|
| `block` (+ `block_gpu`) | 14 (+2 on the GPU pool) | `block.py <split> <country> <i> <n>` (train India 3, US 4; test France 1, India 3, US 3) | — | Dataset `qtpies-er-norm` |
| `feats_test` | 4 | `feats.py test i 4` | — | Test blocking; dataset `qtpies-er-cand-test` |
| `ftd_train` | 5 | `feats.py train i 5` | `BER_DROP=16` | Train blocking |
| `m2_fit1` | 1 | `run.py fit1` | DROP16 | `ftd_train` |
| `m2_pred` | 5 | `run.py predshard:i:5` | DROP16 | `ftd_train`, `feats_test`, `m2_fit1` |
| `m2_s2` (M2) | 1 (GPU pool) | `gather1`, `train2ce`, `predictce` | DROP16, POOL 12M | `m2_pred`, CE-1 scores |
| `ce4_train` | 2 GPU | `ce.py train k` | `BER_CE_HARD=1` | Train blocking, `m2_s2` (for `oof_m`) |
| `ce4_score` | 2 GPU | `ce.py score {train,test} …` | `BER_CE_BAND=0.002,1.01` | `ce4_train`, `m2_s2` |
| `m4` (M4) | 1 | `gather1`, `train2ce`, `predictce` | DROP16, POOL 12M, `GENERIC=1` | `m2_pred`, features, `ce4_score` |
| `ce2_train` | 2 GPU | `ce.py train k` | HARD, `RAW=1`, `TAG=ce2` | Train blocking, `m2_s2`, dataset `qtpies-er-data` |
| `ce2_score` | 2 GPU | `ce.py score …` | band 0.002–1.01, RAW, `TAG=ce2` | `ce2_train`, `m2_s2` |
| **`m5` (M5)** | 1 | `gather1`, `train2ce`, `predictce` | DROP16, POOL 12M, `GENERIC=1` | M4's upstream plus `ce2_score` |

**Final step (laptop):**

```
BER_WORK=work2 BER_DROP=16 python memguard.py 3 python src/calibrate.py \
    ../../runs/qtpies-m5/pred_test_s2ce.parquet ../../submissions/sub12_m5_shift ../../runs/qtpies-m5/oof_s2ce.parquet
```

Then run the official validator.

## Appendix E: Glossary

| Term | Meaning |
|---|---|
| **S1 / S2 / S3** | Reference source / noisy sources |
| **Row** | An S2/S3 record |
| **Link** | A true (S1, row) pair |
| **Distractor** | A row with no true S1 |
| **Twin** | A distractor built as a near-copy of a real S1 (same name, shifted number, suffix; or in France the same address with a swapped generic word) |
| **OOF** | Out-of-fold prediction (2-fold cross-fitting by S1 hash) |
| **DROP16** | Test-like density world: 16% of train S1s hash-dropped |
| **Mix** | CV weighted to the test country mix, with France proxied |
| **CE** | Cross-encoder |
| **Stage-2 set** | Pairs with stage-1 p ≥ 0.002; equal to `candidate_pairs.tsv` |
| **q** | Stage-1 p normalised by the row's total mass |
| **k** | Odds multiplier applied at decode time |
| **DiD** | Difference-in-differences of test/train rows-per-S1 ratios against an anchor country |
| **Anchor** | The seen country with the smallest mid-band excess (India) |
| **Signature cell** | Pair type (no-addr / addr-no-num / num= / num≠) × (core name = / ≠) |
| **Oracle gain** | F0.5 gained if one error class were fixed perfectly |
| **Expected F** | The decoder's own estimate under independence |
