# QTpies M5 share bundle

Built from branch `er-pipeline` @ `544d5df` (polars 1.44.2). Model **M5** (`qtpies-m5`): 2-stage LightGBM + two
cross-encoders; train OOF F0.5 0.98820 in the density-matched (DROP16) world. Calibrated test decode = `sub12_m5_shift`.

## Files
| File | Columns | Meaning |
|---|---|---|
| `oof_s2ce.parquet` | ri, si, y, p | Train stage-2 **out-of-fold** probabilities (2-fold by `hash(si, 42) % 2`). DROP16 world: pairs of the dropped S1s are absent (their rows act as distractors); `y` = 1 for a true link |
| `pred_test_s2ce.parquet` | ri, si, p | Test stage-2 probabilities, **raw** (before `calibrate.py`), mean of the two fold models. This pair set is our `candidate_pairs.tsv` (stage-1 p ≥ 0.002) |
| `ids/n_{split}_s1.parquet` | si, id, country | `si` = 0-based row of our normalised S1 file |
| `ids/n_{split}_s2.parquet`, `ids/n_{split}_s3.parquet` | ri, id, country | `ri` = 0-based row of concat(S2, S3): S3 rows continue after the last S2 row |
| `dropped_train_s1.parquet` | si, id, country | Train S1s treated as absent (BER_DROP=16) |
| `src/calibrate.py`, `src/decide.py` | | Test-time calibration and expected-F0.5 decoder; `common.py` and `run.py` are included because `calibrate.py` imports them |

## Conventions
- si/ri are row numbers of **our** normalised files, which are not guaranteed to follow raw-TSV order: always map through `ids/`.
- Dropped S1s: `(pl.col("si").cast(pl.Int32).hash(49) % 100) < 16`, polars 1.44.2 (polars hashes are version-dependent).
  Same set with UInt32 si: True. Dropped 353,055; live 1,853,766.
- Join example: `oof.join(pl.read_parquet("ids/n_train_s1.parquet"), on="si")` gives the S1 entity id;
  `pl.concat([pl.read_parquet("ids/n_train_s2.parquet"), pl.read_parquet("ids/n_train_s3.parquet")])` joined on `ri` gives the S2/S3 id.

## Reproduce `sub12_m5_shift`
```
cd src
BER_WORK=../ids BER_DROP=16 python calibrate.py ../pred_test_s2ce.parquet <out_dir> ../oof_s2ce.parquet
```
Needs polars 1.44.2, numpy, numba 0.63.1. Result checked byte-identical to `sub12_m5_shift/matching_results.tsv`: **True**.
(US k=0.060 via difference-in-differences vs India; France k=0.10 via empty-rate matching; India unchanged.)

## Row counts and SHA-256 (first 16 hex)
| File | Rows | SHA-256 |
|---|---|---|
| `oof_s2ce.parquet` | 8,812,682 | `693af2a673dcd4f9…` |
| `pred_test_s2ce.parquet` | 9,531,458 | `88c7e7fadc507710…` |
| `dropped_train_s1.parquet` | 353,055 | `4ec879abb94a0c2c…` |
| `ids/n_train_s1.parquet` | 2,206,821 | `cbf8ab8d6042ad4c…` |
| `ids/n_train_s2.parquet` | 5,034,616 | `82b1317856dfb342…` |
| `ids/n_train_s3.parquet` | 5,285,603 | `12b3fa5cf6cb302a…` |
| `ids/n_test_s1.parquet` | 1,732,544 | `5fceb484aaad0e9d…` |
| `ids/n_test_s2.parquet` | 4,887,273 | `5402275dbb9870a7…` |
| `ids/n_test_s3.parquet` | 5,082,316 | `88cc8b948b7eab3e…` |
