# QTpies fuse bundle: rebuild #32 / #33 / #34 / #35 on the teammate's machine

Everything `fuse.py` reads that the teammate does not already have. Verified 19:31: #32, #33 and #34 below are rebuilt
byte for byte from THIS bundle's paths (plus his own blend files and ce_full_v1).

| Path | Content | md5 |
|---|---|---|
| `m9/pred_test_s2ce.parquet` | QTpies M9 stage-2 test p (ri, si, p) | 8ea20cc57dbd275973e17602725e4ae9 |
| `m9/ids/n_test_s{1,2,3}.parquet` | si / ri -> entity id (identical to every QTpies bundle) | see MD5.txt |
| `norm/n_test_s{1,2,3}.parquet` | QTpies normalised test records (id, country, nm, core, ad, nums, f_noaddr) | see MD5.txt |
| `m10/pred_test_s2ce.parquet` | M10 test p | 16d67ec72c03e196071c6b1319ef693c |
| `m11h/pred_test_s2ce.parquet` | M11h test p (M11 US/India, M10 France) | e9ba15537cdc045408f2a4fe7e6594fd |
| `m12h/pred_test_s2ce.parquet` | M12h test p (M12 US/India, M10 France) | 34091257c95991d272847ca42143c45d |
| `code/fuse.py` | final post-processing (self-check: `python code/fuse.py --selftest`) | 0e784338e4955030d92e5c58925f3c4e |
| `code/merge_by_country.py` | country-wise merge for FINAL 2 | see MD5.txt |

Teammate inputs (his clean room):
| Name | File | md5 |
|---|---|---|
| SOLO | `output/candidates/ce_full_v1/matching_results.tsv` | 3a0d4c771a1203acf207811ee2c8a2fe |
| BLEND_M5 | M5-bundle `blend_cal/matching_results.tsv` | 0a0b14806bcddf3370258467ae3fddd9 |
| BLEND_M10 | M10 bundle, `blend.w: 0.3` (`blend_cal_m10_w03/matching_results.tsv`) | d754f06f8800aa5516a2a67d7a1ab2e4 |
| BLEND_M11H | M11h bundle, `blend.w: 0.0` | e7a6eda3527f643798c82680d47d2fed |

Run from the bundle root (Python >= 3.11, polars >= 1.20; ~2 min and ~5 GB RAM each):
```bash
COMMON="--pred m9/pred_test_s2ce.parquet --ids m9/ids \
  --veto France=0.7,US=0.3,India=0.3 --add France=0.99,US=0.99,India=0.99 \
  --solo $SOLO --veto-solo France=0.9,US=0.7,India=0.7 --add-solo France=0.9,US=0.9,India=0.9 \
  --norm norm --veto-g France=0.99 --veto-n US=0.7"
# #32 (public 0.986611) on the M5 blend
python code/fuse.py $BLEND_M5  out32 $COMMON --pred2 m10/pred_test_s2ce.parquet --add2 US=0.95,India=0.95 --fill France=0.5     # -> dbbc1482af916fdcf250f8425cbe1ab0
# #33 (public 0.986655) on the M10 blend: US/India rules read M10 p
python code/fuse.py $BLEND_M10 out33 $COMMON --pred-alt m10/pred_test_s2ce.parquet  --alt-countries US,India --fill France=0.5   # -> c52d413c7404e9be4c368aa49d1d4de0
# #34 (public 0.986731) on the M10 blend: US/India rules read M12h p
python code/fuse.py $BLEND_M10 out34 $COMMON --pred-alt m12h/pred_test_s2ce.parquet --alt-countries US,India --fill France=0.5   # -> c51d385383f2fca625aa8ff24eb6d573
# #35 (public 0.986666) on the M11h blend: US/India rules read M11h p
python code/fuse.py $BLEND_M11H out35 $COMMON --pred-alt m11h/pred_test_s2ce.parquet --alt-countries US,India --fill France=0.5  # -> 10f5696b16da21ba9634ad6040110b3f
# FINAL 2 = France rows from #34, US/India rows from #35 (macro F is a sum over S1s, each S1 in one country)
python code/merge_by_country.py out34/matching_results.tsv out35/matching_results.tsv m9/ids/n_test_s1.parquet out_final2 --a-countries France  # -> 50c695de7ae6c03d19a70ee75040f983
```
Each writes `<out>/matching_results.tsv` (md5 also in `<out>/params.json`), then
`python utils/validate_submission.py --matching <out>/matching_results.tsv --test-dir dataset/test` -> PASS.
`candidate_pairs.tsv` for the zip = the chosen base blend's own `candidate_pairs.tsv` (fuse only adds M9 stage-2 pairs,
which are all in every union: M5/M9/M10/M11h/M12h share the same 9,531,458 test pairs).

## SOLO input (disclosed)
`$SOLO` must be the **production** `ce_full_v1/matching_results.tsv`, md5 `3a0d4c771a1203acf207811ee2c8a2fe`, shipped in the
zip as a fuse input (e.g. `src/razan/ce_full_v1_matching_results.tsv`). With it, #33 / #34 / #35 / FINAL 2 rebuild byte for
byte (verified by the teammate on his laptop for #33 and #34). A clean-room rebuild of ce_full_v1 (md5 677f56f9...) differs
slightly and changes 174-176 of 1,732,544 fused rows (0.01%; empty shares and links per S1 equal to 4 decimals).

## Finals
- FINAL 1 = #34 (public 0.986731): `out34/matching_results.tsv`, md5 c51d385383f2fca625aa8ff24eb6d573.
- FINAL 2 = France from #34 + US/India from #35: `out_final2/matching_results.tsv`, md5 50c695de7ae6c03d19a70ee75040f983.
