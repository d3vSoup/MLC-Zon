# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TO CONFIRM]  
**Team Members:** [TO CONFIRM: full names of Muaz, Razan, Soup, Ankit]  
**Submission Date:** 27 September 2026

> Draft status (Sat 26 Sep): numbers below are from `EXPERIMENTS.md`. Items marked [TO CONFIRM] or [FINAL] are filled
> after the final configuration is chosen on Sunday.

---

## 1. Executive Summary

We solve the task as blocking plus pairwise classification plus set decoding. Two TF-IDF blocking channels, run
within each country, keep 98.25% of the true train pairs while scoring only 0.0007% of all S1 x (S2+S3) pairs. Two
LightGBM rounds score the pairs: 42 pair features first, then graph/group features, soft house-number features, v2
name features and the score of a multilingual cross-encoder (xlm-roberta-base, MIT) fine-tuned on train pairs. A
calibrated expected-F0.5 decoder with a per-entity "has any match" gate then turns the pair scores into match lists.
Grouped out-of-fold (OOF) macro F0.5 on all 2,206,821 train S1 entities is 0.98765 [FINAL], and the public
leaderboard score is 0.981 [FINAL]. No external data, APIs or lookups are used.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on train (reports in `work/reports/phase0_facts.md`):

- **Countries.** Train has US (60%) and India (40%) only. Test adds **France: 15% of test S1, never seen in train.**
  Generalising to an unseen country became the main design constraint: country is never a model feature, and no
  code path hard-codes a country list.
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
- **Unmatched records.** About 26% of S2/S3 records match nothing.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-round gradient-boosted classifier (with a cross-encoder score as a feature) + a
graph-aware calibrated set decoder.  
**Core Innovation:** (1) a decoder that picks, per S1, the number of matches that maximises the expected F0.5 under the
calibrated pair probabilities, with a learned "any match" gate for the empty case; (2) a cross-encoder trained in two
halves, so every train pair gets an honest out-of-sample score that LightGBM can learn from; (3) country-agnostic,
data-driven features, validated with a leave-one-country-out study to predict behaviour on unseen France.

Pipeline (`src/run_all.sh`, one command, each stage cached under `work/`):

1. normalize (v1) -> 2. block (fw + hh channels) -> 3. features (42) -> 4. LightGBM round 1 -> 5. round-2 features
(group/graph, soft house numbers, v2 names) -> 6. cross-encoder scores (keyed by entity ids) -> 7. LightGBM round 2 ->
8. decode (owner rule, isotonic calibration, gate, expected-F0.5) -> 9. write both TSVs + official validator.

---

## 3. Candidate Generation (Blocking)

**Normalization (v1)** before blocking: Unicode folding; Indic scripts transliterated to Latin with
`indic-transliteration` (MIT); US and Indian state names to codes (including transliterated names); address short
forms (street -> st, and so on); zero-stripping of numbers; glued-token splitting; ordinals; a DBA / "trading as"
split (both sides kept); landmark phrases ("near ...", "opposite ...") split out; and a sound-alike consonant
skeleton for transliteration-robust matching. All hand-written maps are disclosed in section 4.3.

**Blocking keys used:** two sparse TF-IDF cosine channels, computed **within each country** (zero true cross-country
pairs in train). The top-K S1 for every S2/S3 record comes from a sparse top-K matrix product (`sparse_dot_topn`,
Apache-2.0):

| channel | tokens | K | alone PC | PC lost if dropped |
|---|---|---|---|---|
| fw | all tokens: name words, any-order name word pairs, glued name, skeleton, address words + bigrams (tokens in > 0.3% of documents dropped) | 10 | 0.9778 | -0.0061 |
| hh | half name / half address cosine | 10 | 0.9763 | -0.0046 |

- **Candidate pairs generated:** 153,461,986 train (69.5 per S1, p95 211) and 148,566,883 test. No S1 has zero
  candidates.
- **Reduction ratio:** 0.9999933 vs all S1 x (S2+S3) pairs (train); 0.9999870 vs within-country pairs.
- **Pair completeness (PC):** **0.9825** overall (US 0.9895, India 0.9720; S2 0.9839, S3 0.9812).
- **Ceiling:** a perfect decoder on this candidate set scores F0.5 0.99407.

**How we ensured true matches were not lost:**

- Two complementary channels: each one alone loses 0.5-0.6 points of PC.
- The skeleton and transliteration tokens catch cross-script and sound-alike names. Any-order word pairs catch
  re-ordered names.
- K was chosen on the PC-at-K curve (`work/reports/blocking_pc_at_k.txt`). We then measured every larger option
  against the F0.5 ceiling before building it:

| option (measured, not adopted) | recovered misses | extra train pairs | ceiling gain |
|---|---|---|---|
| fw K=20 | 20k | +92M | +0.0009 |
| fw + hh K=20 | 30k | +155M | +0.0014 |
| fw + hh K=30 | 48k | +306M | +0.0022 |
| char 3-gram channel K=10 | 13k | +85M | +0.0006 |
| sibling expansion K=5 (orphan record -> sure records -> their S1) | 25k | +8.75M | +0.0013 |

The real gain is below the ceiling, because the recovered pairs are the hardest ones. No option was worth 2-3x the
pairs, so K stays at 10 + 10.

**Reproducibility check (Sat):** re-blocking 20k random train records per country with the final code gave 100% the
same pairs as the production candidate set (296,109 US / 299,895 India), and two runs of the same search were
identical. Re-normalizing 50k rows per source file gave 0 differing rows.

---

## 4. Matching Model

### 4.1 Features

**Round 1: 42 pair features** (`ber/features/strings.py`, `context.py`):

- **Name features:** rapidfuzz ratio / partial / token-sort / token-set (on the core name, and token-set on the
  normalized name); Levenshtein; Jaro-Winkler; skeleton token-set and ratio; IDF-weighted Jaccard; Monge-Elkan;
  first-token match; legal-form equality; best match over DBA parts; digit equality; name length ratio.
- **Address features:** ratio / token-set / token-sort, token-set without landmark; IDF-weighted Jaccard; length
  ratio; empty-address flag; postal-code tri-state (same / differ / missing); house-number equality; number-set
  Jaccard.
- **Other:** blocking channel scores and ranks; number of channels; source (S2/S3); context features without labels:
  each record's margin vs its best and second-best S1, per-S1 score gap, list sizes; and in-split frequency shares
  (how common a name or address is, which flags chains and malls).

**Round 2: extra features on hard pairs only** (round-1 p >= 0.01: 9.84M train / 8.78M test pairs):

- **13 group (graph) features** (`ber/features/graph.py`): the p share, rank and competition of the pair inside its
  record and inside its S1; counts of confident matches; the best name / address similarity to sibling records (other
  records already confidently on the same S1); p-weighted sibling similarity; an exact-address sibling; and the best
  other S1 fit.
- **15 soft house-number features** (`ber/features/numbers.py`): digits only and country-agnostic. They cover
  zero-stripped and joined forms ("17-03" = "1703"), a one-digit typo, a cut prefix or suffix, the absolute and
  relative difference, and "other number taken as house number".
- **8 v2 name features** (`ber/features/names2.py`): dotted acronyms joined ("S.A.S.U." -> "sasu"); legal forms and
  EN/FR function words removed; generic words found from the data (a token is generic if it is in more than 0.1% of
  the split's S1 names, so there is no hand-written list); rare-part similarity. A name made only of generic words
  gives "missing", not "mismatch".
- **Cross-encoder score `ce_p`** (section 4.2).

### 4.2 Models

**LightGBM round 1** (MIT): binary objective, learning rate 0.1, 1,000 rounds with early stopping, feature fraction
0.8, bagging 0.8, L2 1.0. It trains on 20% of train S1 groups (5 folds by S1) and predicts every train pair OOF;
test uses the fold-0 model. Thread count is fixed at 16, because LightGBM output depends on it.

**Cross-encoder** (`src/gpu/ce_train_predict.py`): `xlm-roberta-base` (MIT, about 278M parameters), fine-tuned as a
pair classifier on "name ; address" text pairs. Settings: max length 128, learning rate 2e-5, batch size 32, 1
epoch, fp16, 2x NVIDIA T4 on Kaggle.

- **Halves design.** Train S1 is split into two fixed halves (H1 = folds 0-1 + even keys of fold 2; H2 = the rest).
  One model trains on H1 pairs and scores the H2 band plus the test band. The other trains on H2 and scores the H1
  band plus the test band. Every train pair therefore gets an out-of-sample score, like OOF. Test gets the mean of
  both models (their test scores correlate at 0.981).
- **Training pairs:** 600,000 per half. 40% are positives (half of them the hardest by round-1 p). 60% are negatives:
  half the hardest, a quarter chain-name pairs, the rest easy top-3 pairs.
- **Band (pairs scored):** 0.03 < p < 0.97, or the top-3 pairs of a record with p >= 0.001. That is 10.78M train and
  10.32M test pairs.
- **Coverage:** CE scores reach 93% of round-2 pairs and 98.5-99.3% of the top-3 pairs per S1 (100% of top-3 pairs
  with p > 0.5).
- **Training time:** [TO CONFIRM: GPU hours per half, from Ankit's and Soup's Kaggle logs].
- **Second cross-encoder:** `intfloat/multilingual-e5-base` (MIT, about 278M parameters), seed 43, same halves and
  bands. [FINAL: used or not, and in which form (separate feature or averaged)].

**CE scores ship as inputs.** GPU inference is not part of the CPU pipeline. The scores are shipped in
`code/business_entity_resolution/src/ce_scores/ce_scores_{train,test}.parquet` (columns s1, cand, ce_p), keyed by
entity ids, so any run joins them whatever its row order. To regenerate them: run `python -m ber.cli ce_export`, run
`src/gpu/ce_train_predict.py --half 1` and `--half 2` on a GPU, then run `python -m ber.ce_ingest <half1_dir>
<half2_dir>`.

**LightGBM round 2:** the same parameters, 3,000 rounds with early stopping (best iteration ~870-1,050 with CE). It
trains on all hard train pairs, 5 folds by S1; test uses the fold-0 model. Pairs below the hard threshold keep their
round-1 p. Seed bagging (3 extra seeds) was started and dropped for time; the base model is kept.

**Pair-level blend with a second pipeline (QTpies M5)** [FINAL: used or not]. A teammate's independent pipeline
(char-4-gram + word TF-IDF blocking, 50 features, a collective two-stage LightGBM and two `intfloat/multilingual-e5-small`
cross-encoders, MIT, 118M parameters; its own REPRODUCE.md is shipped in `src/razan/`) provides stage-2 pair
probabilities for train (out-of-fold) and test, keyed by entity ids and shipped as inputs in `src/razan/`.
- Candidate set = union of both pipelines' final candidates (ours + 634,542 test pairs only its blocking found).
- p = 0.5 x ours + 0.5 x theirs; a pair scored by one side only keeps that side's p (this beat "missing = 0" at every
  weight on OOF: 0.99032 vs 0.98950 at w = 0.5).
- The weight is fixed at 0.5, not tuned: on the OOF the curve is flat (w 0.3 / 0.5 / 0.7 = 0.99012 / 0.99032 /
  0.99041), and our side is slightly optimistic in their evaluation world (see 5.1).
- Evaluated on their 1,853,766 "live" train S1 (their density-matched world: 16% of train S1 hash-dropped so their
  records act as distractors, as on test). Their M5 OOF reproduced exactly (0.98820) before blending.

**All models and libraries (licenses):** xlm-roberta-base (MIT); multilingual-e5-base (MIT) [if used];
multilingual-e5-small (MIT, blend partner's cross-encoders) [if blend used]; LightGBM
(MIT); indic-transliteration (MIT); sparse_dot_topn (Apache-2.0); rapidfuzz (MIT); scikit-learn (BSD-3-Clause);
pandas / numpy (BSD-3-Clause); pyarrow (Apache-2.0); regex (Apache-2.0); PyTorch (BSD-3-Clause) and Hugging Face
transformers (Apache-2.0), GPU script only. Every model is far below the 8B-parameter limit. All pins are in
`requirements.txt`.

### 4.3 Hand-written dictionaries (disclosed)

All hand-written domain knowledge lives in one file, `src/ber/dictionaries.py`:

| name | size | content |
|---|---|---|
| LEGAL | 31 | legal-form tokens -> canonical class: US (inc, incorporated, corp, llc, llp, lp, pc, pllc, co, company), India (pvt, private, ltd, limited, opc, plc, incl. transliteration variants praivet / piraivet / limirrad), France (sa, sas, sasu, sarl, eurl, sci, scop, selarl, snc) |
| NAME_STOP | 5 | the, com, www, net, org |
| NAME_STOP2 | 15 | EN + FR function words for v2 names (and, of, de, du, la, le, les, des, et, l, d, en, a, au, aux) |
| DBA | 1 regex | dba, doing business as, trading as, aka, also known as |
| PLACE_PHRASES | 26 | multi-word US / Indian place names -> short code (new york -> ny, district of columbia -> dc, ...) |
| ADDR_TOKENS | 117 | single address tokens -> canonical short form: US state names -> codes, Indian state names incl. spelling / transliteration variants, street types (street, road, avenue, drive, lane, boulevard, ...; FR chemin, allee, impasse, route, faubourg), unit words, directions, ordinals |
| ADDR_FR | 1 | r -> rue |
| ADDR_STOP | 8 | null, cedex, no, number, india, usa, us, france |
| CHILLU | 6 | Malayalam chillu letters -> base consonant + virama (script normalization) |
| CANDRA | 4 | Devanagari candra vowels -> plain vowels (script normalization) |
| SKEL | 26 | letter -> consonant class for the sound-alike skeleton (vowels, h, w, y dropped) |

These maps are general language and format knowledge, not lookups of business identities.

### 4.4 Decoding and threshold selection

1. **Owner rule:** each S2/S3 record is kept only on the S1 where it has the highest p (the one-owner rule is exact in
   train).
2. **Isotonic calibration** of p -> q, cross-fitted by S1 fold so OOF q stays honest.
3. **Gate:** a small LightGBM on 8 per-S1 features (top-1 and top-2 q, sum of q, count of q > 0.5, owned and candidate
   counts, name frequency). It predicts P(S1 has any true match), OOF by fold.
4. **Expected-F0.5 decoder:** for each S1, among its top L = 20 owned candidates sorted by q, pick the k that maximises
   the exact expected F0.5 under independent Bernoulli(q) (dynamic programming over hits inside and outside the
   top-k). For k = 0, the expected F0.5 is replaced by 1 - gate.

**Threshold selection method:** there is no hand-set threshold. Four decoders are compared on OOF (single tau at the
plateau centre, gate + tau2 grid, expected-F, expected-F + gate), and the best is kept: expected-F + gate in every
run since phase 5.

**Label-free test-time calibration** (`src/ber/calibrate.py`, adapted from the blend partner's method). The model is
calibrated to train, but test has more distractor records (about 40% vs 26%) and a country absent from train. Before
isotonic calibration, the odds of p are multiplied by a factor k per S1 country:
- **Countries in train: difference-in-differences prior shift.** Per p band of each record's best pair, compare test
  vs train rows per S1. The anchor is selected automatically by a data rule (the country with the smallest mid-band
  excess), and verified against label-free diagnostics. The shift is used only if a null test (train fold 1 vs fold
  0, where no shift exists) returns k >= 0.95. **Our models fail that null test** (India k 0.31-0.33): near p = 0.99,
  a 1% difference in band counts between fold models already means k about 0.5. So k = 1 for seen countries in the
  final file. On test the data-rule anchor was India, and the US showed a twin-decoy excess (2-3x rows at p 0.4-0.5,
  DiD k 0.13), but its per-band profile was sloped (an artifact sign) and the null test failed, so it is not applied.
- **Countries absent from train (open-set rule, derived from data): empty-rate matching.** k is set by bisection so
  that the share of S1 with an empty prediction equals the seen countries' test empty share + 0.0015 (France:
  k 0.184, empty 5.25% -> 5.92%) [FINAL: k after the leaderboard A/B].

---

## 5. Results & Error Analysis

### 5.1 Validation design

- Grouped 5-fold CV by S1 entity (fold file with seed 42). The OOF macro F0.5 is computed exactly as the metric does:
  per S1 (singletons included), then averaged over all 2,206,821 train S1.
- **Blend evaluation** uses the partner's density-matched world (their 1,853,766 live train S1). Caveat: our
  context features (margins vs the best S1 of a record) were computed with their dropped S1 present, so our side is
  slightly optimistic there (our OOF barely moves: 0.98765 natural vs 0.98764 live); the blend weight is therefore
  fixed at 0.5 instead of the OOF optimum.
- **Leave-one-country-out (LOCO)** simulates unseen France: train on one country, calibrate and decode on it only,
  then score the other country.

### 5.2 Progress (OOF macro F0.5 on all train S1)

| step | PC | OOF | US | India | singleton-only | public |
|---|---|---|---|---|---|---|
| v0: norm v0, one TF-IDF channel, 12 features, tau | 0.9654 | 0.8891 | 0.8915 | 0.8854 | 0.7355 | - |
| decode v1: isotonic + gate + expected-F | 0.9654 | 0.8932 | 0.8954 | 0.8900 | 0.7056 | - |
| norm v1 + two-channel blocking | 0.9825 | 0.9305 | 0.9393 | 0.9174 | 0.8124 | - |
| 42 pair features | 0.9825 | 0.9715 | 0.9750 | 0.9661 | 0.9522 | - |
| round 2 (group features), 3000 rounds | 0.9825 | 0.9773 | 0.9797 | 0.9724 | 0.9679 | 0.965 |
| + soft house numbers + v2 names | 0.9825 | 0.98233 | 0.98532 | 0.97786 | 0.97962 | 0.977 |
| + cross-encoder, half 1 only (scored on H2) | 0.9825 | 0.98744 (H2) | 0.98958 | 0.98425 | 0.99367 | 0.981 |
| + cross-encoder, both halves | 0.9825 | **0.98765** | 0.98971 | 0.98456 | 0.99374 | [FINAL] |

Fold standard deviation of the final OOF: 0.00007.

- **F_0.5 Score (macro):** 0.98765 OOF [FINAL]; public 0.981 [FINAL].

### 5.3 Unseen-country findings (LOCO)

| model | US -> India (India unseen) | India -> US (US unseen) |
|---|---|---|
| round 2, no num/names2 | 0.90584 (in-distribution 0.9724) | 0.95473 (in-distribution 0.9797) |
| + soft house numbers + v2 names | **0.91571 (+0.0099)** | **0.96935 (+0.0146)** |

- An unseen country costs 2.5-7 points, mostly through over-matching: the unseen side is over-confident (q 0.8 ->
  true 0.72), with extra false positives at 40% and singleton false positives at 15% of the loss.
- The source-country OOF is perfectly calibrated, so the gap is discrimination, not calibration.
- Country-agnostic number and name features help the unseen side about 2-3x more than they help OOF. The public
  leaderboard showed the same: +0.012 public vs +0.005 OOF, with France at 15% of test.
- The cross-encoder gave +0.004 public vs +0.005 OOF. That is about the US/India share of the gain, so it is roughly
  neutral on France.

### 5.4 Loss anatomy (final OOF)

Share of total loss: blocking misses 48%, extra false negatives in partly matched S1 33%, missed entities 9.6%,
extra false positives 6.3%, singleton false positives 2.8%. Blocking dominates in India (64.5% of its loss).

- **Common false positives (wrong merges):** chain or franchise names at nearby addresses; generic-word names
  ("... Services", "... Holdings", "... Group", "... Center", "... Partners"); records with a missing address where
  only the name agrees.
- **Common false negatives (missed matches):** records renamed to invented brand names with an abbreviated address;
  cross-script names with a missing address; noisy or conflicting house numbers (62% of in-candidate US false
  negatives had a house conflict before the soft number features); and true pairs that blocking never proposed.

### 5.5 What we tried and dropped

| idea | result | decision |
|---|---|---|
| unseen-country calibration correction (isotonic / tempering fitted on LOCO) | +0.0008 / +0.0010 per direction, precision up, recall down | not adopted |
| pseudo-labelling the unseen side (p > 0.95 -> 1, p < 0.02 -> 0) | -0.0072 (US->IN), -0.0094 (IN->US): pseudo-labels copy the over-matching | dropped |
| per-country percentile copies of the top-15 similarity features | -0.0014 unseen (US->IN) | dropped |
| bigger blocking (see section 3 table) | ceiling gain at most +0.0022 for 2-3x pairs | not built |
| more round-2 data / rounds | 50% -> 100% of S1: +0.0003; 1000 -> 3000 rounds: +0.0005 | 3000 rounds kept |
| round-2 seed bagging (3 extra seeds) | started, stopped for time | dropped |
| self-supervised synthetic pairs for an unseen country (noise model from US+IN true pairs, noisy copies of unlabeled S1, other S1 as true negatives thanks to S1 deduplication; same blocking/features; synthetic feature distributions close to real) | LOCO US->IN: India unseen 0.91571 -> 0.91364 (weight 0.3) / 0.91132 (weight 1.0) | dropped |
| twin-decoy decode rule (drop links with core name equal + house number +-1..20 + legal suffix added, or one generic word swapped at the same address) | blend already rejects test twins (US 3.6 vs 1.75 such links per 1k S1 on train); these signatures are true links in train (precision 0.986-1.000); dropping them costs OOF 0.0003 / 0.0039 | dropped |
| seen-country DiD calibration | fails fold-vs-fold null test (k 0.31-0.33) | not applied |

---

## 6. Conclusion

Careful normalization and two blocking channels made the candidate set nearly complete (PC 0.9825, ceiling 0.994)
at 70 pairs per entity. A second, graph-aware LightGBM round and an expected-F0.5 decoder with an any-match gate
turned pair scores into precise match lists. A cross-encoder trained in two halves added about +0.005. The main
lesson: with an unseen test country, country-agnostic features and a leave-one-country-out study predicted public
gains better than in-distribution OOF did.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `README.md`: exact commands. Environment: Python 3.11, `uv venv` + `uv pip install -r requirements.txt` (all
  dependencies pinned, including transitive ones).
- `src/run_all.sh`: the single entry point. It regenerates `output/candidate_pairs.tsv` and
  `output/matching_results.tsv` from the dataset, then runs the official validator. It runs on a 16 GB RAM laptop
  in about [TO CONFIRM: clean-room wall time] h; each stage is cached under `work/`.
- `src/ber/`: `normalize.py`, `dictionaries.py`, `blocking/` (TF-IDF channels, PC reports), `features/` (strings,
  context, graph, numbers, names2, r2extra), `models/` (LightGBM round 1 `lgbm.py`, round 2 `stack.py`),
  `ce_export.py` / `ce_ingest.py` (cross-encoder data out / scores in), `decode.py`, `metric.py`, `loco.py`
  (leave-one-country-out), `analysis.py` (loss buckets), `cli.py` (stage runner).
- `src/configs/default.yaml`: every knob (K, thresholds, model names, seeds).
- `src/gpu/ce_train_predict.py`: cross-encoder fine-tuning and scoring (Kaggle T4, resumable).
- `src/ce_scores/`: the shipped cross-encoder scores (inputs).
- `src/tests/`: metric, IO and decoder tests.

**Reproducibility:** the final file was regenerated on Sunday by a clean-room run: a fresh clone, a fresh environment
from `requirements.txt`, and the dataset only. [FINAL: md5 match / differing rows and OOF vs reference.]

**Fair play:** no external data, databases, APIs, geocoding or internet lookups are used anywhere. The only network
use was downloading pip packages and the pretrained cross-encoder weights from Hugging Face. All learning uses the
provided training data only.

### B. Additional Results

[FINAL: per-country test statistics of the final file (empty share, mean matches), LOCO tables, feature importance
top 20 from `work/reports/importance_<model>.tsv`.]
