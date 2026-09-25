# Upload failure diagnosis: output/matching_results.tsv

File under test: `output/matching_results.tsv`, not modified.
- md5 `e7dbc8357c3d2035228854ddf8497556`
- sha256 `e748991ef5f40a50f68b2d77495d2179a652a0522bbdf1bd5e7d1a9febde6e20`
- 97,367,936 bytes (97.37 MB, 92.86 MiB)

`~/uploaded.tsv` does not exist on this laptop, so the exact uploaded bytes could not be compared. Razan reports his Drive copy has the same md5.

Raw numbers are in `output/diagnosis/part1_checks.json`. Dataset fingerprints are in `output/diagnosis/dataset_md5.txt`.

## Hypotheses

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Parser limits | **RULED OUT** | The longest line is 155 characters. The most IDs in one row is 11. No field is near csv's 131,072 limit. csv.reader with the default comma: 1,732,545 rows, no error. csv.reader with tab: 1,732,545 rows, 0 bad rows. pandas with default settings: 1,732,544 rows, 2 str columns, 0 NaN S1, and 98,302 NaN matches, which are exactly the empty rows (the spec allows empty). pandas with `dtype=str, keep_default_na=False`: same values, 0 differences once NaN is read as "". polars is not installed, so it was not tested. |
| H2 | Hidden bytes | **RULED OUT** | No BOM. 0 CR. 0 NUL. 0 non-ASCII bytes. 0 spaces. 0 quotes. 0 trailing whitespace. Every data line has exactly 1 tab. 0 tabs inside lists. 0 leading, trailing, or double commas. 0 spaces after commas. The file ends with a single `\n`. The `cat -A` samples (below) look clean. |
| H3 | Header and order | **RULED OUT** | The header bytes are exactly `source1_entity_id\tmatched_entity_ids`. Row order is identical to `test_source1.tsv`. S1 rows are unique and all are in test. An order-dependent evaluator is unlikely, and it would not affect us, since our order already matches. |
| H4 | ID semantics | **RULED OUT** | 5,814,202 IDs. All match `S[23]-\d+`. 0 lowercase. 0 S1 IDs inside lists. 0 IDs appear in more than one row (checked separately from the validator). 0 IDs are missing from local `test_source2/3`. 0 train IDs appear, and train and test IDs do not overlap at all. |
| H5 | Dataset version | **CAN'T TELL (top suspect)** | Every local dataset file is dated **2026-09-18**, before the 25 Sep start. The official validator only checks against these local files. If the portal's test set differs, for example a re-issued set with new IDs, the file fails there but passes here. The md5 list is below, for Razan to compare against a fresh portal download. |
| H6 | Size or timeout | **CAN'T TELL (unlikely)** | 97.4 MB raw, 41.9 MB as gzip or zip. Parsing takes 1.4–1.6 s (pandas) and 2.6 s (csv). Scoring against a synthetic ground truth with our vectorized metric takes 13 s. A naive O(n²) evaluator over 1.73M rows or 5.8M IDs would take hours for **every** team, and teams do get scores, so the evaluator must be efficient. Any team that predicts about 3.5 matches per S1 has a file this size. A portal upload size limit below 97 MB is possible, but unknown. |
| H7 | Which file was uploaded | **CAN'T TELL** | At 18:20 IST, 5 files existed (see the table). The only one in `output/` was the fallback, e7db…, written at 17:59. Razan's Drive copy has the same md5. Whether that exact copy was the one chosen in the portal upload dialog cannot be checked from here. |

### H7 inventory (all existed at 18:20 IST)

| mtime | bytes | md5 | path | what it is |
|---|---|---|---|---|
| 09-25 08:02 | 94,960,462 | cf617c1d… | work/v0/matching_results_phase5.tsv | Phase 5, OOF 0.8932 |
| 09-25 11:42 | 95,608,800 | db6bffa1… | work/v1_phase2/matching_results_phase2.tsv | Phase 2, OOF 0.9305 |
| 09-25 14:48 | 95,958,895 | 5f36e265… | work/v2_phase3/matching_results_phase3.tsv | Phase 3, OOF 0.9715 |
| 09-25 17:50 | 97,329,340 | 7e95d320… | work/v3_phase9/matching_results_phase9_lgbm1.tsv | Phase 9, OOF 0.9768 |
| 09-25 17:59 | 97,367,936 | **e7dbc835…** | **output/matching_results.tsv** | **r3000, OOF 0.9773 (fallback)** |

### cat -A samples

```
source1_entity_id^Imatched_entity_ids$
S1-714132312^IS2-637340732,S3-625880872$
S1-106407869^IS3-585937637,S3-613056593$
S1-156285671^IS2-172049742,S2-648730048,S3-339493107,S3-788875930,S3-871416040$
S1-689823050^IS2-65478038,S2-212601842,S2-319486263,S2-829077756,S3-506456729$
S1-280204013^I$                (empty match, line 9)
S1-750279127^I$                (empty match, line 19)
S1-897975128^I$                (empty match, line 22)
S1-379595891^IS2-219263878,S2-245820002,...,S3-$   (longest line, 155 chars, cut at 120)
S1-331082266^IS2-306362124,S2-418056273,S3-541573833,S3-914496393$
S1-752426175^IS2-65303828,S2-667342751,S2-671228947,S3-405377239,S3-780565363$
```

## Top 3 likely causes (none confirmed as a defect in our file)

1. **The portal's test set differs from our local copy (H5).** A validator PASS cannot catch this.
   - Smallest safe fix: Razan downloads the test files fresh from the portal and compares their md5 with the list below. **No upload is needed for this check.**
   - If they differ: re-run inference on the new test set. Blocking + features + prediction take about 3 h on the laptop, and no retraining is needed if train is unchanged.
2. **An upload or transport problem** (H6/H7): a 97 MB upload dropped midway, a portal size limit or timeout, a transient failure at 18:20, or the wrong file picked in the dialog.
   - Smallest safe fix: ask the organizers or forum what "Failed" means and whether there is a size limit. Then retry once with this exact file, named exactly `matching_results.tsv`, md5-checked right before upload, over a stable connection.
   - Upload a `.zip` only if the portal says it accepts one.
3. **A portal-side quota or processing error.** "Failed" with no reason may also mean the submission limit was hit or the queue broke.
   - Smallest safe fix: ask the organizers. Do not burn retries.

**No corrected file was produced.** None of H1–H4 found a defect, so any "fixed" variant would be a guess, and a blind retry may cost a submission.

## md5 list for Razan (compare with a fresh portal download)

| file | md5 | bytes | lines incl. header |
|---|---|---|---|
| test/test_source1.tsv | 51a155b8c84bfefcf049e983f06c3cf4 | 175,022,086 | 1,732,545 |
| test/test_source2.tsv | ff8aba5701829a622ac05b2765c4eb5b | 509,456,422 | 4,887,274 |
| test/test_source3.tsv | 5f48d48594dea3ea1455549eb5233ad8 | 506,002,772 | 5,082,317 |
| train/train_ground_truth.tsv | 3643443570b2c881c425d69c2d46e95d | 127,015,583 | 2,206,822 |
| train/train_source1.tsv | 1a0c98e43dad1babed3c99bca2c7b5bd | 210,069,713 | 2,206,822 |
| train/train_source2.tsv | a7bddba33ead4a85e6c088ad3b377fc2 | 489,301,488 | 5,034,617 |
| train/train_source3.tsv | 5df07bf109f55a292bdfaa8c983cfa20 | 503,705,637 | 5,285,604 |

All local mtimes are 2026-09-18 18:17:50.
