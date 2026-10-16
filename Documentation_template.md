# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** QTpies  
**Team Members:** Muhammed Razan (leader), Muaz Ismail Mohammed, Ankit Ranjan, Souparno Chakraborty  
**Submission Date:** 27 September 2026

| Requested information | Where |
|---|---|
| Methodology used | section 2 (problem analysis, solution strategy), section 4.4 (decoding, calibration, final fuse) |
| Candidate Generation / Blocking Strategy | section 3 |
| Model Architecture and Feature Engineering | section 4 (architecture overview, 4.1 features, 4.2 models, 4.3 hand-written dictionaries) |
| Any other relevant information | 4.5 use of test-set statistics, 5 results and error analysis, Appendix A code, reproducibility and fair play |

---

## 1. Executive Summary

We solve the task as blocking, pairwise classification and set decoding, with two independent pipelines. Each blocks
within country with TF-IDF, scores pairs with LightGBM on name, address, number and group features plus multilingual
cross-encoder scores (all MIT), and decodes each entity's match list with a calibrated expected-F0.5 decoder; the second
pipeline also trains against synthetic "twin" decoys and adds set-aware cluster cross-encoders and a stage-3 stacker.
The two pipelines' pair probabilities are blended on the union of their candidate sets (out-of-fold macro F0.5 0.99093
on density-matched train entities), a fuse with fixed per-country rules corrects the blend, and the final file scores
**0.986763** on the public leaderboard. No external data, APIs or lookups are used.

---

## 2. Methodology Used

### 2.1 Problem Analysis

Findings from EDA on train (stage `eda`, `src/ber/eda.py`):

- **Countries.** Train has US (60%) and India (40%) only. Test adds **France: 15% of test S1, never seen in train.**
  Generalising to an unseen country became the main design constraint: in pipeline A country is never a model
  feature and no code path hard-codes a country list (countries absent from train are found by a data rule). Pipeline
  C's training augmentation and the final fuse are keyed by country name (disclosed in section 4.5).
- **Singletons.** 5.6% of train S1 have no true match. The metric scores them 1.0 only for an empty prediction, so
  false merges on singletons are expensive.
- **Group sizes.** A mean of 3.46 true matches per S1; 77% of S1 have 2 or more matches from the same source.
- **One-owner rule.** No S2/S3 record belongs to more than one S1 (0 of 7,638,365 matched ids), and no true pair crosses
  countries. Both facts are used: each record is kept only on its best S1, and blocking runs within each country.
- **Noise.** Names differ a lot, most of all in India. The median name token-set similarity of true S2 pairs is 91 in
  India and 98 in the US; the 5th percentile is 9 in India. Names appear in native scripts (Devanagari, Bengali,
  Kannada, Malayalam and others). Other noise: invented brand names, legal-form changes (Pvt Ltd / Private Limited /
  Prívate), suffix words ("Services", "Partners"), obfuscated domains ("iadvis0rs.com"), abbreviated or re-ordered
  addresses, typos in street names, zero-padded or cut house numbers ("004859" vs "4859", "099" vs "5099"), and
  missing addresses.
- **Unmatched records and decoys.** About 26% of train S2/S3 records match nothing; test has more (about 38-40%), and it
  holds near-twin decoys that train almost lacks: copies of a real record with the house number shifted and a legal
  suffix added (US), or with one generic name word swapped at the same address (France).

### 2.2 Solution Strategy

**Approach Type:** Hybrid: blocking + two gradient-boosted pair classifiers with cross-encoder features + a calibrated
set decoder, blended at pair level and corrected by a rule-based fuse.  
**Core Innovation:** (1) a decoder that picks, per S1, the number of matches that maximises the expected F0.5 under the
calibrated pair probabilities, with a learned "any match" gate for the empty case; (2) cross-encoders trained in
halves or folds, so every train pair gets an honest out-of-sample score that LightGBM can learn from; (3) a "twin
world": synthetic decoy records generated from train so the model learns to reject test's near-twins, plus a cluster
cross-encoder that reads an entity with all its candidate records at once; (4) country-agnostic, data-driven features,
validated with a leave-one-country-out study to predict behaviour on unseen France.

Pipeline (`code/business_entity_resolution/README.md`, steps A-D):

- **A** (`src/run_all.sh`, CPU): normalize -> block (fw + hh channels) -> 42 features -> LightGBM round 1 -> round-2
  features (group/graph, soft house numbers, v2 names) -> cross-encoder score (step B) -> LightGBM round 2 -> decode ->
  blend with C.
- **C** (`src/razan/`): normalise -> block -> stage-1 LightGBM -> two pairwise cross-encoders -> twin world -> two
  cluster cross-encoders -> stage-2 LightGBM (M9) -> stage 3 (M10; M11 with A's probability).
- **Blend** (in A): union of both candidate sets, weighted pair probability, calibration, decode.
- **D. Fuse**: per-country corrections of the blend with C's probabilities -> `output/matching_results.tsv`.

---

## 3. Candidate Generation / Blocking Strategy

**Normalization** before blocking (A: `src/ber/normalize.py`): Unicode folding; Indic scripts transliterated to Latin
with `indic-transliteration` (MIT); US and Indian state names to codes; address short forms (street -> st, and so on);
zero-stripping of numbers; glued-token splitting; ordinals; a DBA / "trading as" split (both sides kept); landmark
phrases ("near ...", "opposite ...") split out; and a sound-alike consonant skeleton. C (`src/razan/src/norm.py`) learns
a native-script -> Latin token table from train pairs (anyascii fallback) and derives a core name (legal forms and
filler words removed), a normalised address and number sets. All hand-written maps are disclosed in section 4.3.

**Blocking keys used:** sparse TF-IDF cosines, computed **within each country** (zero true cross-country pairs in
train); top-K S1 per S2/S3 record from a sparse top-K matrix product (`sparse_dot_topn`, Apache-2.0).

| pipeline | channel | tokens | K |
|---|---|---|---|
| A | fw | all tokens: name words, any-order name word pairs, glued name, skeleton, address words + bigrams (tokens in > 0.3% of documents dropped) | 10 |
| A | hh | half name / half address cosine | 10 |
| C | combined | name character 4-grams + address word 1-2-grams (each part L2-normalised); features with S1 document frequency > 3,000 dropped, a second pass with cap 10,000 for rows whose best cosine < 0.6 | 15 |

- **Candidate pairs generated:** A: 153,461,986 train (69.5 per S1) and 148,566,883 test; no S1 has zero candidates.
  C blocks 160,008,083 test pairs and keeps the 9,531,458 with stage-1 p >= 0.002 for stage 2. **The final candidate
  set (`output/candidate_pairs.tsv`) is the union of A's test candidates and C's stage-2 pairs: 149,201,425 test
  pairs** (634,542 of them found only by C). Every final link is in it.
- **Reduction ratio (A):** 0.9999933 vs all S1 x (S2+S3) pairs (train); 0.9999870 vs within-country pairs.
- **Pair completeness (A, train):** **0.9825** overall (US 0.9895, India 0.9720; S2 0.9839, S3 0.9812). A perfect
  decoder on this candidate set scores F0.5 0.99407.

**How we ensured true matches were not lost:**

- Complementary channels and pipelines: each of A's two channels alone loses 0.5-0.6 points of PC, and C's
  character-4-gram blocking adds pairs A's word-level channels miss.
- The skeleton and transliteration tokens catch cross-script and sound-alike names; any-order word pairs catch
  re-ordered names.
- K was chosen on the PC-at-K curve. Every larger option was measured against the F0.5 ceiling first:

| option (measured, not adopted) | recovered misses | extra train pairs | ceiling gain |
|---|---|---|---|
| fw K=20 | 20k | +92M | +0.0009 |
| fw + hh K=20 | 30k | +155M | +0.0014 |
| fw + hh K=30 | 48k | +306M | +0.0022 |
| char 3-gram channel K=10 | 13k | +85M | +0.0006 |
| sibling expansion K=5 (orphan record -> sure records -> their S1) | 25k | +8.75M | +0.0013 |

The real gain is below the ceiling, because the recovered pairs are the hardest ones; no option was worth 2-3x the pairs.

**Reproducibility check:** re-blocking 20k random train records per country with the final code gave 100% the same
pairs as the production candidate set, and a full clean-room run reproduced `candidate_pairs.tsv` byte for byte.

---

## 4. Model Architecture and Feature Engineering

**Architecture overview** (A = pipeline A, `src/ber/`; C = pipeline C, `src/razan/src/`):

```
                     challenge data (train: US + India; test: US + India + France)
                  /                                                   \
  A: normalize -> TF-IDF blocking (fw + hh, K 10 + 10)      C: normalise -> TF-IDF blocking (char 4-grams + address words, K 15)
     42 pair features -> LightGBM round 1                      pair features -> LightGBM stage 1
     group / house-number / name features                      2 pairwise cross-encoders (e5-small), twin world,
       + xlm-roberta-base cross-encoder score                   2 cluster cross-encoders (e5-small, e5-base)
     -> LightGBM round 2: p_A                                   -> LightGBM stage 2 (M9) -> stage 3: M10; M11 = M10 + p_A
                  \                                                   /
   blend: union of candidates, p = w p_A + (1 - w) p_C, isotonic + gate + expected-F0.5 decoder
          (M10 blend with France k 0.6 -> France rows; M11h blend -> US / India rows)
                                         |
   fuse: fixed per-country vetoes, adds, twin-signature vetoes, France fills -> output/matching_results.tsv
```

### 4.1 Features

**Pipeline A, round 1: 42 pair features** (`ber/features/strings.py`, `context.py`):

- **Name features:** rapidfuzz ratio / partial / token-sort / token-set (on the core name, and token-set on the
  normalized name); Levenshtein; Jaro-Winkler; skeleton token-set and ratio; IDF-weighted Jaccard; Monge-Elkan;
  first-token match; legal-form equality; best match over DBA parts; digit equality; name length ratio.
- **Address features:** ratio / token-set / token-sort, token-set without landmark; IDF-weighted Jaccard; length
  ratio; empty-address flag; postal-code tri-state (same / differ / missing); house-number equality; number-set
  Jaccard.
- **Other:** blocking channel scores and ranks; number of channels; source (S2/S3); context features without labels:
  each record's margin vs its best and second-best S1, per-S1 score gap, list sizes; and in-split frequency shares
  (how common a name or address is, which flags chains and malls).

**Pipeline A, round 2** (round-1 p >= 0.01: 9.84M train / 8.78M test pairs):

- **13 group (graph) features** (`ber/features/graph.py`): the p share, rank and competition of the pair inside its
  record and inside its S1; counts of confident matches; the best name / address similarity to sibling records (other
  records already confidently on the same S1); p-weighted sibling similarity; an exact-address sibling; the best other
  S1 fit.
- **15 soft house-number features** (`ber/features/numbers.py`): digits only and country-agnostic: zero-stripped and
  joined forms ("17-03" = "1703"), a one-digit typo, a cut prefix or suffix, the absolute and relative difference, and
  "other number taken as house number".
- **8 v2 name features** (`ber/features/names2.py`): dotted acronyms joined ("S.A.S.U." -> "sasu"); legal forms and
  EN/FR function words removed; generic words found from the data (a token in more than 0.1% of the split's S1 names;
  no hand-written list); rare-part similarity.
- **Cross-encoder score `ce_p`** (section 4.2).

**Pipeline C** (`src/razan/src/feats.py`, `stage2.py`, `stage3.py`):

- **Stage 1:** language-agnostic name / address / number similarities and the blocking cosines.
- **Stage 2 (collective):** owner competition (best other S1 of the record, its total mass, rank), cluster context
  (the S1's confident mass per source), sibling evidence (similarity to the S1's confident records, shared numbers),
  generic-word features, twin features (house-number shift `dn`, legal-form added / dropped), house-number shape
  features, and the scores of two pairwise and two cluster cross-encoders.
- **Stage 3 (M10):** stage-2 p, row and S1 context (rank, top and second p, probability mass, confident rows per S1,
  candidate counts) and **same-name rival-row counts**: for a record's core name, the number of addressed records that
  no S1 claims (best p < 0.5), that are claimed, and that have no address. A no-address record whose name matches an
  S1 is either that S1's record or a record of an absent same-name entity; unclaimed addressed records with that name
  are evidence for the absent entity. **M11** adds pipeline A's round-2 p on the same pairs (its value, whether A
  scored the pair, A's row top p and rank, the gap between the two).

### 4.2 Models

**Pipeline A.**
- **LightGBM round 1** (MIT): binary objective, learning rate 0.1, 1,000 rounds with early stopping, feature fraction
  0.8, bagging 0.8, L2 1.0. It trains on 20% of train S1 groups (5 folds by S1) and predicts every train pair OOF; test
  uses the fold-0 model. Thread count is fixed at 16, because LightGBM output depends on it.
- **Cross-encoder** (`src/gpu/ce_train_predict.py`): `xlm-roberta-base` (MIT, about 278M parameters), fine-tuned as a
  pair classifier on "name ; address" text pairs: max length 128, learning rate 2e-5, batch size 32, 1 epoch, fp16,
  2x NVIDIA T4 on Kaggle. **Halves design:** train S1 is split into two fixed halves; one model trains on H1 pairs and
  scores the H2 band plus test, the other the reverse, so every train pair gets an out-of-sample score; test gets the
  mean of both (their test scores correlate at 0.981). Training pairs: 600,000 per half (40% positives, half of them
  the hardest by round-1 p; 60% negatives: half the hardest, a quarter chain-name pairs, the rest easy top-3 pairs).
  Band scored: 0.03 < p < 0.97, or the top-3 pairs of a record with p >= 0.001 (10.78M train and 10.32M test pairs).
  Compute: 5.61 h (half 1) and 5.80 h (half 2) on 2x T4.
- **LightGBM round 2:** the same parameters, 3,000 rounds with early stopping (best iteration ~870-1,050). It trains on
  all hard train pairs, 5 folds by S1; test uses the fold-0 model. Pairs below the hard threshold keep round-1 p.

**Pipeline C** (private Kaggle kernels; `src/razan/README.md`). Training uses a density-matched world: 16% of train
S1 are hash-dropped so their records act as distractors, as on test.
- **Stage 1 and stage 2:** LightGBM, 2-fold cross-fitting grouped by S1; test p = mean of the two fold models.
- **Pairwise cross-encoders:** `intfloat/multilingual-e5-small` (MIT, 118M), one on normalised and one on raw text,
  trained with the stage-1 model's confusions as hard negatives, 2 folds, scoring every stage-2 pair.
- **Twin world:** 90,665 house-number twins (US; shift drawn from a label-free test histogram, legal suffix added) and
  26,498 generic-word twins (US, India) generated from train records and appended as negatives; 116,106 twin pairs
  enter stage 2.
- **Cluster cross-encoders:** one sequence = an S1 plus up to 12 of its stage-2 candidate records, one logit per
  record; `multilingual-e5-small` (400,000 units per fold) and `intfloat/multilingual-e5-base` (MIT, 278M; 200,000
  units per fold); 2 folds, each fold's model scores the other fold's train S1 and test S1.
- **M9** = stage 2 with all of the above. **Stage 3:** LightGBM (learning rate 0.05, 63 leaves, seed 7) on M9 p and the
  stage-3 features, cross-fitted by S1 hash on live train S1; the test model is fit on all live train pairs. **M10**
  uses C's features only; **M11** adds A's p. **M11h** = M11 for US/India pairs and M10 for France (the stacker never
  saw France).

**Pair-level blend** (`src/ber/blend.py`). Candidates = union of both pipelines' candidate sets; p = w x p_A +
(1 - w) x p_C, and a pair scored by one side only keeps that side's p (this beat "missing = 0" on OOF). Isotonic
calibration, the gate and the expected-F0.5 decoder are fitted on C's 1,853,766 live train S1. Two blends feed the fuse:
- **M10 blend**, w = 0.3 (C weight 0.7), chosen on OOF: C weight 0.5 / 0.7 / 0.85 = 0.99068 / **0.99070** / 0.99055.
- **M11h blend**, w = 0.0: M11 already contains A's p, and pairs only A scored keep A's p. OOF **0.99093**.

**Models and libraries (licenses):** xlm-roberta-base (MIT, 278M); multilingual-e5-small (MIT, 118M);
multilingual-e5-base (MIT, 278M); LightGBM (MIT); polars (MIT); indic-transliteration (MIT); anyascii (ISC);
sparse_dot_topn (Apache-2.0); rapidfuzz (MIT); numba (BSD-2-Clause); scikit-learn, pandas, numpy, scipy (BSD-3-Clause);
pyarrow (Apache-2.0); regex (Apache-2.0); PyTorch (BSD-3-Clause) and Hugging Face transformers (Apache-2.0), GPU steps
only. Every model is far below the 8B-parameter limit. All pins are in the `requirements*.txt` files.

### 4.3 Hand-written dictionaries (disclosed)

Pipeline A keeps all hand-written domain knowledge in `src/ber/dictionaries.py`:

| name | size | content |
|---|---|---|
| LEGAL | 31 | legal-form tokens -> canonical class: US (inc, incorporated, corp, llc, llp, lp, pc, pllc, co, company), India (pvt, private, ltd, limited, opc, plc, incl. transliteration variants praivet / piraivet / limirrad), France (sa, sas, sasu, sarl, eurl, sci, scop, selarl, snc) |
| NAME_STOP | 5 | the, com, www, net, org |
| NAME_STOP2 | 15 | EN + FR function words for v2 names (and, of, de, du, la, le, les, des, et, l, d, en, a, au, aux) |
| DBA | 1 regex | dba, doing business as, trading as, aka, also known as |
| PLACE_PHRASES | 26 | multi-word US / Indian place names -> short code (new york -> ny, district of columbia -> dc, ...) |
| ADDR_TOKENS | 117 | single address tokens -> canonical short form: US state names -> codes, Indian state names incl. spelling / transliteration variants, street types (street, road, avenue, ...; FR chemin, allee, impasse, route, faubourg), unit words, directions, ordinals |
| ADDR_FR | 1 | r -> rue |
| ADDR_STOP | 8 | null, cedex, no, number, india, usa, us, france |
| CHILLU / CANDRA | 6 / 4 | Malayalam chillu letters and Devanagari candra vowels (script normalization) |
| SKEL | 26 | letter -> consonant class for the sound-alike skeleton |

Pipeline C (`src/razan/src/norm.py`, `twins.py`): legal-form tokens (US, India, France), filler words (EN / FR
function words, honorifics, generic business words), an alias-marker regex (aka, dba, formerly, trading as), address
abbreviations (EN and FR street types, directions, unit words, three city renamings), state names -> codes, and the
legal suffixes used for synthetic twins (LLC, Inc, Co, Corp, Ltd, LP). These are general language and format
knowledge, not lookups of business identities.

### 4.4 Decoding and threshold selection

1. **Owner rule:** each S2/S3 record is kept only on the S1 where it has the highest p.
2. **Isotonic calibration** of p -> q, cross-fitted by S1 fold so OOF q stays honest.
3. **Gate:** a small LightGBM on 8 per-S1 features (top-1 and top-2 q, sum of q, count of q > 0.5, owned and candidate
   counts, name frequency) predicts P(S1 has any true match), OOF by fold.
4. **Expected-F0.5 decoder:** for each S1, among its top 20 owned candidates sorted by q, pick the k that maximises the
   exact expected F0.5 under independent Bernoulli(q) (dynamic programming). For k = 0 the expected F0.5 is 1 - gate.

**Threshold selection method:** the pipelines and the blend use no hand-set threshold: four decoders are compared on
OOF (single tau, gate + tau2 grid, expected-F, expected-F + gate) and expected-F + gate is best.

**Label-free test-time calibration** (`src/ber/calibrate.py`). Before isotonic calibration, the odds of p are
multiplied by a factor k per S1 country. For countries absent from train, k is set by bisection so that their share of
S1 with an empty prediction equals the seen countries' test empty share + 0.0015 (empty-rate matching; k 0.432 on the
M10 blend). For seen countries a difference-in-differences prior shift is used only if a train fold-vs-fold null test
passes; it fails, so k = 1 for the US and India.

**Final fuse** (`src/razan/src/fuse.py`, fixed parameters). Applied to a blend file with q = p / max(1, sum of p over
the record's candidates); p is C's M9 p for France and the p of the C model inside the blend (M11h) for the US and India:

| rule | threshold | definition |
|---|---|---|
| veto | France 0.7, US / India 0.3 | drop a blend link with p below the threshold |
| add | 0.99 | add a pair with q >= 0.99 on a record the blend leaves unowned |
| solo veto / add | veto France 0.9, US / India 0.7; add 0.9 | a third opinion from pipeline A's single-pipeline file: veto links it does not make, add q in [0.9, 0.99) where it agrees |
| twin veto (G) | France 0.99 | drop a link whose names differ by one swapped core word at the same address |
| twin veto (N) | US 0.7 | drop a link with the same core name, a changed legal part and disjoint numbers |
| fill | France 0.5 | an S1 left empty gets its best non-twin candidate with q >= 0.5 |

The final file takes France rows from the fuse on the M10 blend at France k 0.6 and US / India rows from the fuse on
the M11h blend (`merge_by_country.py`; macro F0.5 is a sum over S1, each S1 in one country). Against its base blends,
the fuse changes 10,047 France S1 rows (9,241 vetoes, 1,183 adds), 121 US rows (133 vetoes) and 98 India rows
(101 vetoes). The fuse is a test-time step with no out-of-fold counterpart: its thresholds, the per-country choice of
blend and France k 0.6 were selected with public-leaderboard feedback.

### 4.5 Use of test-set statistics and country keys (disclosure)

No test labels are used anywhere. Label-free statistics of the unlabeled test set shape these parts:
- **Empty-rate matching** for countries absent from train (section 4.4). Generic-word sets and IDF weights for name
  features are computed per split, so on test from test S1 names.
- **Density matching (C):** 16% of train S1 are treated as absent so train matches the test distractor share.
- **Twin world (C):** the twin parameters were measured label-free on US test predictions: the shift-size histogram,
  89% positive shifts, 94% added suffixes and 30% two-record groups. The twin rates are keyed by country name
  (US 0.072 for number twins; 0.03 for generic-word twins in the US and India). They shape training only; every test
  entity, France included, goes through the same pipeline.
- **Fuse:** the twin-signature vetoes target the label-free excess of twin signatures among test links over a twin-free
  baseline at the same model score; its thresholds are keyed by country name and were set with public-leaderboard
  feedback (section 4.4).
- **Reporting only:** C's score script weights countries by the test S1 mix (US 0.383, India 0.468, France 0.150)
  when it reports cross-validation.

Apart from these, country enters only as an opaque grouping key. No external data, APIs or lookups are used.

---

## 5. Results & Error Analysis (other relevant information)

### 5.1 Validation design

- **Pipeline A:** grouped 5-fold CV by S1 entity (seed 42). The OOF macro F0.5 is computed exactly as the metric: per
  S1 (singletons included), averaged over all 2,206,821 train S1.
- **Pipeline C and the blend:** 2-fold CV by S1 hash, evaluated on C's 1,853,766 live train S1 (the density-matched
  world). A's context features were computed with the dropped S1 present, so A is slightly optimistic there (0.98765
  natural vs 0.98764 live).
- **Leave-one-country-out (LOCO)** simulates unseen France: train on one country, calibrate and decode on it only,
  then score the other country.

### 5.2 Progress

Pipeline A (OOF macro F0.5, all train S1):

| step | PC | OOF | US | India | singleton-only |
|---|---|---|---|---|---|
| v0: one TF-IDF channel, 12 features, single threshold | 0.9654 | 0.8891 | 0.8915 | 0.8854 | 0.7355 |
| decoder: isotonic + gate + expected-F | 0.9654 | 0.8932 | 0.8954 | 0.8900 | 0.7056 |
| normalization v1 + two-channel blocking | 0.9825 | 0.9305 | 0.9393 | 0.9174 | 0.8124 |
| 42 pair features | 0.9825 | 0.9715 | 0.9750 | 0.9661 | 0.9522 |
| round 2 (group features), 3000 rounds | 0.9825 | 0.9773 | 0.9797 | 0.9724 | 0.9679 |
| + soft house numbers + v2 names | 0.9825 | 0.98233 | 0.98532 | 0.97786 | 0.97962 |
| + cross-encoder (both halves) | 0.9825 | **0.98765** | 0.98971 | 0.98456 | 0.99374 |

Pipeline C and the blends (live train S1):

| model | OOF | US | India |
|---|---|---|---|
| C: stage 2 with pairwise cross-encoders | 0.98820 | | |
| C: + twin world + cluster cross-encoder (e5-small) | 0.98849 | 0.98913 | 0.98754 |
| C: M9 (+ e5-base cluster cross-encoder) | 0.98852 | 0.98915 | 0.98757 |
| C: M10 (stage 3, rival-row counts) | 0.98898 | 0.98969 | 0.98792 |
| C: M11 (stage 3 + A's p) | 0.98956 | | |
| blend A + M10 (C weight 0.7) | 0.99070 | 0.99091 | 0.99039 |
| blend A + M11h (w 0.0) | **0.99093** | | |

- **F_0.5 Score (macro):** validation (OOF, live train S1) **0.99093** for the M11h blend and 0.99070 for the M10
  blend; pipeline A alone 0.98765 (all train S1). **Final file: 0.986763 public** (the fuse is test-only, so it has no
  OOF).

### 5.3 Unseen-country findings (LOCO, pipeline A)

| model | US -> India (India unseen) | India -> US (US unseen) |
|---|---|---|
| round 2, no num/names2 | 0.90584 (in-distribution 0.9724) | 0.95473 (in-distribution 0.9797) |
| + soft house numbers + v2 names | **0.91571 (+0.0099)** | **0.96935 (+0.0146)** |

- An unseen country costs 2.5-7 points, mostly through over-matching: the unseen side is over-confident (q 0.8 ->
  true 0.72), with extra false positives at 40% and singleton false positives at 15% of the loss.
- The source-country OOF is perfectly calibrated, so the gap is discrimination, not calibration.
- Country-agnostic number and name features help the unseen side about 2-3x more than they help OOF.
- **Error anatomy of the unseen side (US->IN):** errors grow 4-15x (false negatives 53 -> 241, false positives
  10 -> 156 per 1k S1), and the false positives are confident (median q 0.985). 58% of them fit none of our twin types;
  of those, 98.7% are decoy records (they belong to no S1) with a near-identical address (median token-set similarity
  99.3) and one business word changed in the name.
- **Country-invariant models do not close the gap:** a transfer model on country-invariant features, monotone
  constraints, and dropping the most shifted format features all lost on the unseen side in both directions.
- LOCO predicts the effect of feature changes on an unseen country well, but it preferred no France down-scaling
  (k = 1); the final uses France k 0.6 (section 4.4).

### 5.4 Loss anatomy (pipeline A OOF)

Share of total loss: blocking misses 48%, extra false negatives in partly matched S1 33%, missed entities 9.6%,
extra false positives 6.3%, singleton false positives 2.8%. Blocking dominates in India (64.5% of its loss).

- **Common false positives (wrong merges):** chain or franchise names at nearby addresses; generic-word names
  ("... Services", "... Holdings", "... Group", "... Center", "... Partners"); records with a missing address where
  only the name agrees; on test, near-twin decoys (house number shifted with a legal suffix added, or one generic
  word swapped at the same address), which the twin world and the fuse's twin vetoes target.
- **Common false negatives (missed matches):** records renamed to invented brand names with an abbreviated address;
  cross-script names with a missing address; noisy or conflicting house numbers (62% of in-candidate US false
  negatives had a house conflict before the soft number features); true pairs that blocking never proposed.

### 5.5 What we tried and dropped

| idea | result | decision |
|---|---|---|
| unseen-country calibration correction (isotonic / tempering fitted on LOCO) | +0.0008 / +0.0010 per direction, precision up, recall down | not adopted |
| pseudo-labelling the unseen side (p > 0.95 -> 1, p < 0.02 -> 0) | -0.0072 (US->IN), -0.0094 (IN->US) | dropped |
| per-country percentile copies of the top-15 similarity features | -0.0014 unseen (US->IN) | dropped |
| bigger blocking (section 3 table) | ceiling gain at most +0.0022 for 2-3x pairs | not built |
| more round-2 data / rounds | 50% -> 100% of S1: +0.0003; 1000 -> 3000 rounds: +0.0005 | 3000 rounds kept |
| self-supervised synthetic pairs for an unseen country | LOCO US->IN: 0.91571 -> 0.91364 / 0.91132 | dropped |
| twin-decoy decode rule in pipeline A alone | these signatures are true links in train (precision 0.986-1.000); dropping them costs OOF 0.0003 / 0.0039 | dropped (the fuse vetoes twins only where C's p is low) |
| seen-country DiD calibration | fails fold-vs-fold null test (k 0.31-0.33) | not applied |
| second cross-encoder in A (multilingual-e5-base) | +0.00015 OOF on its half (gate +0.0003) | dropped |
| v2 name generic sets + IDF per country group | OOF +-0.00002; LOCO -0.0007 / -0.0003 | dropped |
| drop the most country-shifted house-number format features | LOCO -0.0052 / -0.0028 | dropped |
| transfer model on 58 country-invariant features, mixed with the full model | LOCO unseen down at every mix weight in both directions | dropped |
| monotone constraints on 24 similarity features | LOCO unseen -0.0010 / +0.0005, seen -0.0004 / -0.0008 | dropped |
| stage 3 with more of A's context and rival-S1 claims (C) | +0.00006 OOF over M11 | not used |

---

## 6. Conclusion

Two independent pipelines, each built on careful normalization, within-country TF-IDF blocking, gradient-boosted pair
models with cross-encoder features and an expected-F0.5 decoder, complement each other: their blend reaches 0.99093 OOF,
and a fixed-rule fuse corrects it on test to 0.986763 public. Training against synthetic twin decoys and set-aware
cross-encoders handled the test's decoy records, and country-agnostic features checked with a leave-one-country-out
study carried the model to the unseen country.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds the code only; `README.md` gives the run order, the exact commands and the
md5 every intermediate file had in our run:

- **Step A** `src/run_all.sh`: the entry point of pipeline A (`src/ber/`: `normalize.py`, `dictionaries.py`,
  `blocking/`, `features/`, `models/` (round 1 `lgbm.py`, round 2 `stack.py`), `ce_export.py` / `ce_ingest.py`,
  `decode.py`, `calibrate.py`, `blend.py`, `metric.py`, `loco.py`, `analysis.py`, `cli.py`). It regenerates
  `output/candidate_pairs.tsv`, the single-pipeline file and the blend files from the dataset plus the B and C outputs;
  config in `src/configs/default.yaml`; tests in `src/tests/`. Environment `requirements.txt` (Python 3.11).
- **Step B** `src/gpu/ce_train_predict.py`: cross-encoder fine-tuning and scoring (GPU) -> `src/ce_scores/`.
  Environment `requirements_ce_gpu.txt`.
- **Step C** `src/razan/`: pipeline C's code (`src/razan/src/`) and its Kaggle job runner; `src/razan/README.md`
  regenerates its outputs (M9, M10, M11h, id maps) from the dataset. Environments `requirements_qtpies.txt` and
  `requirements_qtpies_gpu.txt` (Python 3.12).
- **Step D** `src/razan/src/fuse.py` + `merge_by_country.py`: three commands that build `output/matching_results.tsv`
  from the step A and C outputs in minutes.

**Reproducibility.** From the intermediate files of our run (md5s in the READMEs), step D rebuilds the final file byte
for byte (md5 `f1754ce2ef7dad7a8df904636eec20c2`, verified). A clean-room run of step A (fresh clone, fresh environment,
the dataset and the B and C outputs) reproduced the candidate file byte for byte and the round-2 OOF to 1e-5 (0.98766
vs 0.98765). Pipeline C's stage 3 and M11h rebuild byte for byte from its stage-2 outputs (the M10 train OOF to one
float32 ulp in 888 of 8.8M values). The GPU steps and the multi-threaded LightGBM stages reproduce up to fp16 / cuDNN
and thread-order noise (about 1e-4 of OOF F0.5), so a full rerun from the data gives a near-identical, not
byte-identical, final file; for example, a clean-room rebuild of the single-pipeline file (the fuse's `--solo` input)
changes 0.01% of the final rows.

**Fair play:** no external data, databases, APIs, geocoding or internet lookups are used anywhere. The only network
use was downloading pip packages and pretrained model weights from Hugging Face. All learning uses the provided
training data only.

### B. Additional Results

**Final file** (`output/matching_results.tsv`):

| country | S1 | links | empty share | links per S1 | rows from |
|---|---|---|---|---|---|
| US | 663,106 | 2,248,985 | 5.77% | 3.392 | fuse on the M11h blend |
| India | 809,986 | 2,730,559 | 5.81% | 3.371 | fuse on the M11h blend |
| France | 259,452 | 827,430 | 5.92% | 3.189 | fuse on the M10 blend, France k 0.6 |
| all | 1,732,544 | 5,806,974 | | | md5 `f1754ce2ef7dad7a8df904636eec20c2` |

**Round-2 feature importance, pipeline A** (clean-room `lgbm2_ce`, mean gain / 1e6 over 5 folds, top 20): ce_p 17.48,
p1 14.90, p_share_rec 0.46, p_rank_rec 0.43, n_s1_c50_rec 0.16, fw_margin_rec 0.11, sib_w_other_rec 0.11,
hn_absdiff_log 0.10, p_other_rec 0.09, sib_addr_max 0.06, adnl_tset 0.05, p_sum_s1_other 0.05, sib_addr_eq 0.04,
hn_suffix 0.04, hn_reldiff 0.04, f_s1_name_share 0.03, n_s1_c80_other 0.02, p_other_s1 0.02, sib_wmax 0.02,
hn_in_other 0.02.
