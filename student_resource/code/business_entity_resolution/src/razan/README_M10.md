# QTpies M10 bundle: drop-in for `src/razan/` (re-blend)

M10 = M9 + stage 3 (`stage3.py`): a LightGBM that recalibrates M9's stage-2 pair probability with same-name rival-row
counts (addressed rows with the same core name that no S1 claims). Label-free features, cross-fitted by S1 on train.
Train CV (live S1s, same metric as before): **0.98898** vs M9 0.98852 (India 0.98792 vs 0.98757, US 0.98969 vs 0.98915).

## What changes vs the M9 bundle you already have
Only two files. Everything else in `src/razan/` stays exactly as in the M9 (= M8 = M5) bundle.

| File | Rows | md5 | Columns |
|---|---|---|---|
| `oof_s2ce.parquet` | 8,812,682 | `5e7eaaa23bcd2a44565d53222d2c4dc2` | ri Int32, si Int32, y Int8, p Float32 (same pairs and schema as M9) |
| `pred_test_s2ce.parquet` | 9,531,458 | `16d67ec72c03e196071c6b1319ef693c` | ri Int32, si Int32, p Float64 (same pairs as M9) |

Unchanged (verify md5): `ids/n_test_s1` 54d8729130a3a994f83a0095e9a72915, `ids/n_test_s2` 85f009cbe17a1f82a295eeeff6d2e602,
`ids/n_test_s3` 82cfd1f0bcac38452c8866be7f5fc012, `ids/n_train_s1` 79415a426916e570211e43d2637489f7,
`ids/n_train_s2` 2681072cf8210e527d6cf2609172cf26, `ids/n_train_s3` 6b0f7a688a00dd374910cee963712bf0,
`dropped_train_s1.parquet` cd705017c5c541892db8f1b56e30ce99. No twin rows (max ri = 10,320,218, as in M9).

## Blend
1. Copy the two parquets over `student_resource/code/business_entity_resolution/src/razan/`.
2. `default.yaml`: `blend.w: 0.5`, `mode: asis`, `unseen_k: auto` (as in `tools/wip/m8_reblend.sh`); keep `n_jobs: 16`.
3. `cd .../src && python -m ber.cli blend` (then optionally w 0.7).
4. Send back, per w: `oof_live` overall + US + India (M5 blend was 0.99032), the France k chosen, and
   `output/candidates/blend_cal/matching_results.tsv` (+ its md5). We run `fuse.py` on top and use one probe.

Also please report the clean-room md5s of `output/candidates/blend_cal/matching_results.tsv` (M5 bundle; expected
0a0b14806bcddf3370258467ae3fddd9) and `output/candidates/ce_full_v1/matching_results.tsv` (expected
3a0d4c771a1203acf207811ee2c8a2fe): the current best upload (0.986611) is `fuse.py` on exactly those two files.

Reproduce M10: `BER_WORK=<work with our n_{train,test}_s*.parquet> python stage3.py <M9 share> <out>` (deterministic, ~10 min, 6 GB).
Licence: LightGBM (MIT); no new models or data.
