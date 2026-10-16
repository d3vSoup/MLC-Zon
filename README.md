# MLC-Zon: Business Entity Resolution (team QTpies)

Final submission for the **ML Challenge 2026: Business Entity Resolution**. This repository is the code and the final
output of that submission, nothing else.

| | |
|---|---|
| **Final output** | [`output/matching_results.tsv`](output/matching_results.tsv), md5 `f1754ce2ef7dad7a8df904636eec20c2` |
| **Write-up** | [`Documentation_template.md`](Documentation_template.md): methodology, blocking, models, features, disclosures, results |
| **Code** | [`code/business_entity_resolution/`](code/business_entity_resolution/), run order in its [README](code/business_entity_resolution/README.md) |
| **Team** | Muhammed Razan (leader), Muaz Ismail Mohammed, Ankit Ranjan, Souparno Chakraborty |

## The task

Three business-record sources with no shared identifiers. Source 1 is the deduplicated reference. For every Source 1
entity (S1), predict the matching records in Source 2 and Source 3 (zero, one or many). Score: macro **F0.5** over S1
entities, where a singleton S1 scores 1.0 only for an empty prediction.

- Train covers **US and India**; test adds **France**, a country never seen in training (15% of test S1).
- Names and addresses are noisy: typos, transliteration from Indic scripts, legal-form changes, abbreviated and
  re-ordered addresses, truncated or zero-padded house numbers.
- Test contains near-twin decoys (a real record with a shifted house number plus a legal suffix, or with one generic
  name word swapped at the same address) that train almost lacks.

Output format (tab-separated, one row per S1, 1,732,544 rows):

```
source1_entity_id	matched_entity_ids
S1-714132312	S2-637340732,S3-625880872
```

## Approach

Blocking, pairwise classification and set decoding, run as two independent pipelines whose pair probabilities are
blended and then corrected by a rule-based fuse.

```
 train: US + India          test: US + India + France
        |                              |
  Pipeline A (src/ber)          Pipeline C "QTpies" (src/razan)
  normalise, TF-IDF blocking    normalise, char-4-gram + address blocking
  42 pair features              pair features -> LightGBM stage 1
  LightGBM round 1              2 pairwise + 2 cluster cross-encoders (e5)
  group / house-number feats    twin world (synthetic decoys) -> stage 2 (M9)
  + xlm-roberta-base CE score   stage 3 (M10; M11 = M10 + p_A)
  LightGBM round 2 -> p_A
        \                              /
         blend on the union of candidates: p = w*p_A + (1-w)*p_C
         isotonic calibration + gate + expected-F0.5 decoder
                          |
         fuse: per-country vetoes / adds / twin-signature vetoes / France fills
                          |
                output/matching_results.tsv
```

Key ideas (details in [`Documentation_template.md`](Documentation_template.md), sections 2-4):

1. **Expected-F0.5 decoder.** Per S1, choose the number of matches that maximises expected F0.5 under the calibrated
   pair probabilities, with a learned "any match" gate for the empty case.
2. **Out-of-sample cross-encoders.** Trained in halves/folds so every train pair has an honest cross-encoder score
   for LightGBM to learn from (multilingual models, all MIT).
3. **Twin world.** Synthetic decoys generated from train teach the model to reject test's near-twins; a cluster
   cross-encoder reads an entity with all its candidate records at once.
4. **Country-agnostic features**, validated with a leave-one-country-out study that simulates the unseen country.
5. **Exact country-wise merge.** Macro F0.5 is a sum over S1 and every S1 is in one country, so France rows (fuse on
   the M10 blend, France k 0.6) and US/India rows (fuse on the M11h blend) combine exactly.

Validation (out-of-fold macro F0.5, density-matched train S1): pipeline A alone 0.98765, M10 blend 0.99070,
M11h blend 0.99093. The fuse is applied to test only, so the final file has no out-of-fold score; its public
leaderboard score is 0.986763.

## Repository layout

```
output/matching_results.tsv           final submission file
Documentation_template.md             full methodology write-up
code/business_entity_resolution/
  README.md                           run order, commands, md5 of every intermediate
  requirements*.txt                   pinned environments (A: py3.11; C/D: py3.12, polars 1.44.2; GPU variants)
  src/
    run_all.sh                        step A entry point
    configs/default.yaml              paths and parameters (root override: BER_ROOT)
    ber/                              pipeline A: normalize, blocking, features, models, decode, calibrate, blend
    gpu/ce_train_predict.py           step B: cross-encoder fine-tune + score (GPU)
    razan/                            pipeline C (QTpies) + step D fuse
      README.md                       how to regenerate C's outputs, with md5s
      src/fuse.py, merge_by_country.py   step D
    tests/                            metric / io / decode tests
```

## Reproduction

| Step | What | Hardware | Where |
|---|---|---|---|
| A | pipeline A + blends | CPU, 16 GB RAM, ~5 h | `src/run_all.sh`, `python -m ber.cli blend` |
| B | xlm-roberta cross-encoder scores | 2x T4 GPU, ~6 h per half | `src/gpu/ce_train_predict.py` |
| C | pipeline C: M9, M10, M11, M11h | Kaggle CPU+GPU, laptop | `src/razan/README.md` |
| D | final fuse + country merge | CPU, ~5 GB RAM, minutes | `src/razan/src/fuse.py`, `merge_by_country.py` |

Run order: C up to M10 and B, then A, then C (export, M11, M11h), then A again (second blend), then D. Full commands,
configs and the md5 of every intermediate are in
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

Step D from the intermediate files of our run rebuilds the final file byte for byte (md5 above). A full rerun from the
raw data is near-identical but not byte-identical (fp16 / cuDNN and multi-threaded LightGBM ordering noise, about
1e-4 of OOF F0.5).

Quick checks:

```bash
md5sum output/matching_results.tsv        # f1754ce2ef7dad7a8df904636eec20c2
cd code/business_entity_resolution/src
python -m pytest -q tests                 # metric, io, decode (pipeline A environment)
python razan/src/fuse.py --selftest       # fuse rules on a toy example (QTpies environment)
```

## Not in this repository

The repo is code plus the final output only. Excluded because of size or because they are regenerated:

- **Challenge data** (`dataset/`, `utils/validate_submission.py`): supplied by the organisers; place at the root as
  described in the code README. Validate with
  `python utils/validate_submission.py --matching output/matching_results.tsv --test-dir dataset/test`.
- **`output/candidate_pairs.tsv`** (1.9 GB, union candidate set, 149,201,425 test pairs): regenerated by step A
  (`python -m ber.cli blend`, md5 `5021548f879c1f8ad6fbf157e0462778`).
- **Intermediate artefacts** (cross-encoder scores, stage-2 probabilities, normalised parquet files, blend files):
  regenerated by steps B, C and A; md5s are listed in the code READMEs.

## Disclosures

Stated once here, in full in section 4.5 of the write-up:

- No external data, APIs, geocoding or internet lookups. The network is used only for pip packages and pretrained
  weights (multilingual-e5, xlm-roberta-base, both MIT).
- No test labels are used. Label-free statistics of the unlabeled test set shape: empty-rate matching for countries
  absent from train, per-split generic-word/IDF sets, the twin-world rates, and the fuse's twin-signature rules.
- The France calibration factor and the fuse thresholds are keyed by country name and were set with public-leaderboard
  feedback.
- The fuse's third-opinion input is our single-pipeline file (`ce_full_v1`, md5 `3a0d4c771a1203acf207811ee2c8a2fe`), a
  fixed input of the final step. A clean-room rebuild changes 0.01% of final rows.

## Licences

Models and libraries with their licences are listed in section 4.2 of the write-up; everything is MIT, BSD, Apache-2.0 or
similar. Only the provided training data is learned from.
