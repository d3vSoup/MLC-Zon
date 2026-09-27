# Final step: submission-level fusion (`src/fuse.py`)

Runs **after** the team blend (`python -m ber.cli blend` → `output/candidates/blend_cal/matching_results.tsv`).
It edits only the matching file; `candidate_pairs.tsv` stays the blend's union file unchanged (every added link is an
M9 stage-2 pair, and M9 scores exactly the same 9,531,458 test pairs as the M5 bundle inside the union — verified).

## What it does (fixed parameters, deterministic)
Per country, on our M9 stage-2 test probabilities (raw, pre-calibration):
- **veto**: drop a team link that M9 scored with p < τ_veto. Pairs M9 never scored are kept.
- **add**: add an M9 pair with q = p / max(1, Σ_s p over the row's M9 candidates) ≥ τ_add, only on an S2/S3 row that has
  no owner in the team file after vetoes (one owner per row).
- optional **solo** (teammate's `ce_full_v1` matching file): also veto a team link with M9 p < τ_veto_solo that the solo
  file does not link; also add an M9 pair with τ_add_solo ≤ q < τ_add that the solo file links.
- An S1 left with no links is written empty; untouched rows are copied byte for byte (no parameters → input md5).

Leaderboard evidence (27 Sep, each probe = one change on top of `blend_cal` 0.985131): US+India veto p<0.3 + add
q≥0.99 +0.000260; France veto p<0.1 +0.000310, 0.1–0.3 +0.000244, 0.3–0.7 +0.000225 (0.7–0.9 −0.000108, not used);
France add q≥0.99 +0.000112, 0.95–0.99 +0.000025; US+India veto 0.3–0.5 −0.000020 (not used). See `runs/lb_log.md`.

## Inputs (md5)
| File | md5 |
|---|---|
| `output/candidates/blend_cal/matching_results.tsv` (clean room, tag `freeze-2026-09-27`) | `0a0b14806bcddf3370258467ae3fddd9` |
| `output/candidates/ce_full_v1/matching_results.tsv` (only for the solo variants) | `3a0d4c771a1203acf207811ee2c8a2fe` |
| M9 bundle `pred_test_s2ce.parquet` (= `src/razan/pred_test_s2ce.parquet` of `qtpies_m9_share.zip`) | `8ea20cc57dbd275973e17602725e4ae9` |
| M9 bundle `ids/n_test_s1.parquet` / `n_test_s2` / `n_test_s3` | `54d87291…` / `85f009cb…` / `82cfd1f0…` |
| `src/fuse.py` | `9ca79e7ee2a78b4d596d9bae2e843ce0` |
| M10 bundle `pred_test_s2ce.parquet` (`src/stage3.py` on the M9 share; only for #32) | `16d67ec72c03e196071c6b1319ef693c` |
| our normalised test records `n_test_s1/s2/s3.parquet` (norm.py output; only for #31) | `77fae2dc…` / `87cbd261…` / `c129477e…` |

Environment: Python ≥ 3.11 with `polars>=1.20` (tested 1.44.2; MIT). Peak RAM ≈ 3.5 GB, ~2 min.

## Commands (run from `code/business_entity_resolution/`; R = the M9 bundle folder)
```
BASE="python src/fuse.py <blend_cal>/matching_results.tsv <out_dir> --pred $R/pred_test_s2ce.parquet --ids $R/ids"
# C2 (no solo)            -> 0b3e1691a85da6befe4ec9683cfdcba3
$BASE --veto France=0.7,US=0.3,India=0.3 --add France=0.95,US=0.99,India=0.99
# sub29 (C2 + solo vetoes) -> 4c61ed3ff4880537f4550da1df1b7f49
$BASE --veto France=0.7,US=0.3,India=0.3 --add France=0.95,US=0.99,India=0.99 \
      --solo <ce_full_v1>/matching_results.tsv --veto-solo France=0.9,US=0.7,India=0.7
# sub30 (sub29 + solo-gated adds) -> 16409c785f42e4f4b2780ddc6448a221
$BASE --veto France=0.7,US=0.3,India=0.3 --add France=0.99,US=0.99,India=0.99 \
      --solo <ce_full_v1>/matching_results.tsv --veto-solo France=0.9,US=0.7,India=0.7 --add-solo France=0.9,US=0.9,India=0.9
# sub31 (#30 + twin-signature vetoes) -> 1d5fe7d603541cb83c0b7995aaf27596
$BASE --veto France=0.7,US=0.3,India=0.3 --add France=0.99,US=0.99,India=0.99 \
      --solo <ce_full_v1>/matching_results.tsv --veto-solo France=0.9,US=0.7,India=0.7 --add-solo France=0.9,US=0.9,India=0.9 \
      --norm <dir with our n_test_s1/s2/s3.parquet> --veto-g France=0.99 --veto-n US=0.7
# sub32 (#31 + M10 adds + France fills) -> dbbc1482af916fdcf250f8425cbe1ab0
$BASE <sub31 flags> --pred2 <M10 bundle>/pred_test_s2ce.parquet --add2 US=0.95,India=0.95 --fill France=0.5
```
M10 bundle: `BER_WORK=<work with n_{train,test}_s*.parquet> python src/stage3.py <M9 share> <M10 share>` (deterministic;
train CV 0.98898 vs M9 0.98852).
Public: C2 = 0.986307 (predicted, sum of measured parts), sub29 = 0.986358, sub30 = 0.986404, sub31 = **0.986579**, sub32 = pending.

The chosen variant's output md5 must equal the one above (also written to `<out_dir>/params.json` as `out_md5`,
next to the input md5s); then copy it to `output/matching_results.tsv` and run
`python utils/validate_submission.py --matching output/matching_results.tsv --test-dir dataset/test`.
Self-check: `python src/fuse.py --selftest`.
