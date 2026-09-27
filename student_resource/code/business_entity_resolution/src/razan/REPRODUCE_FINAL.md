# QTpies final bundle: reproduce the team's best submission

Best public: **0.986655** = `fuse.py` on the teammate's **M10 re-blend** (step 3b), md5 **`c52d413c7404e9be4c368aa49d1d4de0`**.
Previous best 0.986611 = `fuse.py` on the M5 `blend_cal` (step 3), md5 `dbbc1482af916fdcf250f8425cbe1ab0`.
Verified: both md5s are reproduced exactly from this bundle plus the teammate's files.
All leaderboard probes of 27 Sep: `LB_LOG_2026-09-27.md` in the QTpies repo (github.com/razancodes/MLC-raZON, main).

## Bundle layout
| Path | What | Used by |
|---|---|---|
| `m9/oof_s2ce.parquet`, `m9/pred_test_s2ce.parquet` | QTpies M9 stage-2 p (train OOF without twin rows; test) | stage3.py, fuse.py |
| `m9/ids/n_{train,test}_s{1,2,3}.parquet`, `m9/dropped_train_s1.parquet` | si/ri -> entity id; density-matching dropped S1s | everything (identical to the M5/M8/M9 bundles) |
| `m10/oof_s2ce.parquet`, `m10/pred_test_s2ce.parquet` | M10 = M9 + stage 3 (same pairs and schema as M9) | teammate blend (`src/razan/`), fuse.py `--pred2` |
| `norm/n_test_s{1,2,3}.parquet` | QTpies normalised test records, columns id, country, nm, core, ad, nums, f_noaddr (row order = ri/si) | fuse.py `--norm`, stage3.py |
| `norm/n_train_s{1,2,3}.parquet` | normalised train records, columns id, country, core, f_noaddr | stage3.py |
| `code/fuse.py`, `code/stage3.py`, `code/export_for_razan.py` | final post-processing, M10/M11 stacker, teammate export for M11 | |
| `MD5.txt` | md5 of every file above | |

`norm/` are column subsets of QTpies `norm.py` output (REPRODUCE.md step 1); the full pipeline from the raw dataset is
REPRODUCE.md -> REPRODUCE_M8.md -> REPRODUCE_M9.md (M9 bundle) -> step 1 here (M10) -> teammate blend -> step 3 here.

## 1. M10 from M9 (optional: the bundle already contains the output)
```
cd code
BER_WORK=../norm python stage3.py ../m9 ../m10_rebuilt
```
Deterministic (LightGBM seed 7, 8 threads), ~10 min, ~6 GB RAM. Must give `oof_s2ce.parquet` = `5e7eaaa23bcd2a44565d53222d2c4dc2`
and `pred_test_s2ce.parquet` = `16d67ec72c03e196071c6b1319ef693c`. Train CV (live S1s): M9 0.98852 -> M10 0.98898.

## 2. Teammate files (his repo, clean room, tag freeze-2026-09-27, M5 bundle in src/razan/)
| File | md5 |
|---|---|
| `output/candidates/blend_cal/matching_results.tsv` | `0a0b14806bcddf3370258467ae3fddd9` |
| `output/candidates/ce_full_v1/matching_results.tsv` | `3a0d4c771a1203acf207811ee2c8a2fe` |
| `output/candidates/blend_cal/candidate_pairs.tsv` | the final candidate file, unchanged by fuse (every added link is an M9 stage-2 pair; M9 scores exactly the same 9,531,458 test pairs as the M5 bundle in the union) |

## 3. Final matching file
```
cd code
python fuse.py <blend_cal>/matching_results.tsv <out_dir> \
  --pred ../m9/pred_test_s2ce.parquet --ids ../m9/ids \
  --veto France=0.7,US=0.3,India=0.3 --add France=0.99,US=0.99,India=0.99 \
  --solo <ce_full_v1>/matching_results.tsv --veto-solo France=0.9,US=0.7,India=0.7 --add-solo France=0.9,US=0.9,India=0.9 \
  --norm ../norm --veto-g France=0.99 --veto-n US=0.7 \
  --pred2 ../m10/pred_test_s2ce.parquet --add2 US=0.95,India=0.95 --fill France=0.5
```
-> `<out_dir>/matching_results.tsv` md5 **`dbbc1482af916fdcf250f8425cbe1ab0`** (also `out_md5` in `<out_dir>/params.json`,
next to every input md5 and parameter). ~2 min, ~5 GB RAM, polars >= 1.20 (tested 1.44.2, MIT). Then
`python utils/validate_submission.py --matching <out_dir>/matching_results.tsv --test-dir dataset/test` -> PASS.
Self-check: `python fuse.py --selftest`. Copy the file to `output/matching_results.tsv`; `candidate_pairs.tsv` = the
blend's union file.

What fuse does (per country, on M9's raw stage-2 p; q = p / max(1, sum of p over the row's candidates); every rule
was measured on the public leaderboard, one change per upload, see LB log):
- veto: drop a team link M9 scores below τ (US/IN 0.3, France 0.7);
- add: add an M9 pair with q ≥ 0.99 on a row the team file leaves unowned (one owner per row);
- solo: veto team links M9 < τ that `ce_full_v1` does not link (only M5 backed them); add q in [0.9, 0.99) where it does;
- twin signatures (label-free test excess, AUDIT §8.3–8.4): France one swapped core word at the same address (M9 < 0.99),
  US same core + changed legal part + disjoint numbers (M9 < 0.7);
- M10 adds q ≥ 0.95 (US/IN); France S1s left empty get their best non-twin M9 candidate with q ≥ 0.5.

## 3b. Current best (#33, public 0.986655): fuse on the M10 re-blend
Teammate: M10 re-blend = `m10/*.parquet` in `src/razan/`, **`blend.w: 0.3`** (teammate weight 0.3 = QTpies M10 weight 0.7), `mode: asis`,
`unseen_k: auto` (France k 0.43217), `python -m ber.cli blend` (oof_live 0.9907; the `blend.w: 0.5` variant is md5 `660ee2f653e58f9dbb97a335f6e9e7dc`, oof_live 0.99068)
-> `output/candidates/blend_cal/matching_results.tsv` md5 `d754f06f8800aa5516a2a67d7a1ab2e4` (public 0.986456 on its own);
its `candidate_pairs.tsv` is the final candidate file (M10 has the same pairs as M5/M9, so the union is unchanged).
```
cd code
python fuse.py <M10 re-blend blend_cal>/matching_results.tsv <out_dir>   --pred ../m9/pred_test_s2ce.parquet --pred-alt ../m10/pred_test_s2ce.parquet --alt-countries US,India --ids ../m9/ids   --veto France=0.7,US=0.3,India=0.3 --add France=0.99,US=0.99,India=0.99   --solo <ce_full_v1>/matching_results.tsv --veto-solo France=0.9,US=0.7,India=0.7 --add-solo France=0.9,US=0.9,India=0.9   --norm ../norm --veto-g France=0.99 --veto-n US=0.7 --fill France=0.5
```
-> md5 **`c52d413c7404e9be4c368aa49d1d4de0`**, validator PASS. On a base that already blends M10 the US/India rules read
M10 p (`--pred-alt`); with M9 p they would veto ~3,500 links M10 scores > 0.7. France keeps the M9-based rules measured
on the leaderboard (6,229 vetoes, 1,826 adds, 520 fills on this base).

## 4. In progress: re-blends (need one probe each before becoming the final)
- **M10 re-blend (done):** copy `m10/*.parquet` over `src/razan/` (ids/dropped unchanged), `blend.w: 0.3` (as uploaded, `d754f06f…`), `mode: asis`,
  `unseen_k: auto`, `python -m ber.cli blend`; report `oof_live` (M5 blend: 0.99032); then step 3 with the new `blend_cal`.
- **M11 (stacker with the teammate's p):** `cp code/export_for_razan.py <his src>/ && python export_for_razan.py` ->
  send `work/razan_export_{train,test}.parquet`; QTpies runs `stage3.py ../m9 <out> <gt_pairs> <dir with the exports>`
  (A/B vs the blend's average printed); if it wins, the M11 bundle is blended with **`w: 0.0`** (w = weight on the
  teammate's model; M11 already contains it).

Licences: LightGBM, polars (MIT); CE models per REPRODUCE_M9.md (MIT / Apache-2.0, ≤ 8B). No external data or lookups.
