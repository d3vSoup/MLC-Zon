# Reproduce the final QTpies submission (M10 bundle + fuse)

Best public so far: **0.986611** = `sub32_m10_fill` (md5 `dbbc1482af916fdcf250f8425cbe1ab0`), built by `src/fuse.py`
on top of the teammate's `blend_cal`. Leaderboard history and every probe: `LB_LOG_2026-09-27.md`.

## Pipeline overview
```
QTpies pipeline (REPRODUCE.md -> REPRODUCE_M8.md -> REPRODUCE_M9.md)        teammate pipeline (tag freeze-2026-09-27)
  norm -> block -> stage 1 -> CE / cluster CE -> stage 2 (twin world)           ... -> lgbm2_ce -> ce_full_v1 (solo file)
  = M9 bundle: oof_s2ce / pred_test_s2ce (+ ids/, dropped_train_s1)              blend (his p + our M5 bundle, w 0.5) -> blend_cal
        |
  src/stage3.py (rival-row recalibration)  = M10 bundle
        |
  src/fuse.py:  blend_cal  +  M9 p  +  ce_full_v1  +  our n_test_s*.parquet  +  M10 p   ->  matching_results.tsv
```

## 1. M9 bundle (unchanged)
Follow `REPRODUCE_M9.md`. Outputs (md5):
| File | md5 |
|---|---|
| `oof_s2ce.parquet` (share version, no twin rows) | from `qtpies_m9_share.zip` |
| `pred_test_s2ce.parquet` | `8ea20cc57dbd275973e17602725e4ae9` |
| `ids/n_test_s1` / `s2` / `s3` | `54d8729130a3a994f83a0095e9a72915` / `85f009cbe17a1f82a295eeeff6d2e602` / `82cfd1f0bcac38452c8866be7f5fc012` |
| `ids/n_train_s1` / `s2` / `s3` | `79415a426916e570211e43d2637489f7` / `2681072cf8210e527d6cf2609172cf26` / `6b0f7a688a00dd374910cee963712bf0` |
| `dropped_train_s1.parquet` | `cd705017c5c541892db8f1b56e30ce99` |
| our normalised test records `work/n_test_s1/s2/s3.parquet` (norm.py) | `77fae2dcc777aeb25b4e5ae77f4d6c77` / `87cbd26172269abe053486e695c102e6` / `c129477e2c178c66bc17f8c49e4ad546` |

## 2. M10 bundle (stage 3)
```
cd code/business_entity_resolution/src
BER_WORK=<work dir with n_{train,test}_s{1,2,3}.parquet> python stage3.py <M9 share dir> <M10 share dir> [<work>/gt_pairs.parquet]
```
Deterministic (LightGBM seed 7, 8 threads), ~10 min, peak ~6 GB. Writes `oof_s2ce.parquet` (ri, si, y Int8, p Float32;
8,812,682 rows; md5 `5e7eaaa23bcd2a44565d53222d2c4dc2`), `pred_test_s2ce.parquet` (ri, si, p; 9,531,458 rows; md5
`16d67ec72c03e196071c6b1319ef693c`) and `stage3.txt`. With `gt_pairs` it prints train CV: M9 0.98852 -> **M10 0.98898**
(India 0.98757 -> 0.98792, US 0.98915 -> 0.98969). `ids/` and `dropped_train_s1.parquet` are the M9 ones.

What it adds: a no-address row whose name matches a live S1 is either that S1's row or a row of an absent same-name
entity. Addressed rows with the same core name that no S1 claims (best M9 p < 0.5) are evidence for the absent entity.
Features: M9 p, row/S1 context (rank, top/second p, masses, candidate counts) and those rival counts; cross-fitted by
S1 hash on live train S1s. All label-free at test time.

## 3. Teammate files (his repo, tag `freeze-2026-09-27`, clean room)
| File | md5 |
|---|---|
| `output/candidates/blend_cal/matching_results.tsv` (blend with our M5 bundle, France k 0.18386) | `0a0b14806bcddf3370258467ae3fddd9` |
| `output/candidates/ce_full_v1/matching_results.tsv` (his solo model) | `3a0d4c771a1203acf207811ee2c8a2fe` |

`candidate_pairs.tsv` = his blend's union file, unchanged: every link fuse adds is an M9 stage-2 pair, and M9 scores
exactly the same 9,531,458 test pairs as the M5 bundle inside that union (verified).

## 4. Fuse (final matching file)
```
cd code/business_entity_resolution
python src/fuse.py <blend_cal>/matching_results.tsv <out_dir> \
  --pred <M9 share>/pred_test_s2ce.parquet --ids <M9 share>/ids \
  --veto France=0.7,US=0.3,India=0.3 --add France=0.99,US=0.99,India=0.99 \
  --solo <ce_full_v1>/matching_results.tsv --veto-solo France=0.9,US=0.7,India=0.7 --add-solo France=0.9,US=0.9,India=0.9 \
  --norm <work dir with our n_test_s1/s2/s3.parquet> --veto-g France=0.99 --veto-n US=0.7 \
  --pred2 <M10 share>/pred_test_s2ce.parquet --add2 US=0.95,India=0.95 --fill France=0.5
```
-> `<out_dir>/matching_results.tsv` md5 **`dbbc1482af916fdcf250f8425cbe1ab0`** (also in `params.json` as `out_md5`,
with every input md5 and parameter). ~2 min, peak ~5 GB. Then
`python utils/validate_submission.py --matching <out_dir>/matching_results.tsv --test-dir dataset/test` -> PASS.
Self-check: `python src/fuse.py --selftest`. `src/fuse.py` md5 `9ca79e7ee2a78b4d596d9bae2e843ce0`.

Rules (per country, on M9's raw stage-2 p; q = p / max(1, sum of p over the row's candidates)):
| Rule | Meaning | Public evidence (one change per probe) |
|---|---|---|
| veto | drop a team link M9 scores below τ | US/IN <.3 +0.00026; France <.1 +0.00031, .1–.3 +0.00024, .3–.7 +0.00023 (.7–.9 −0.00011, not used) |
| add | add an M9 pair with q ≥ τ on a row the team file leaves unowned (one owner per row) | France q ≥ .99 +0.00011 |
| veto-solo | drop a team link M9 < τ that the teammate's solo file does not link (only M5 supported it) | +0.00005 |
| add-solo | add M9 pairs with q in [τ_solo, τ) that the solo file also links | +0.00005 |
| veto-g / veto-n | drop twin-signature links: France one swapped core word at the same address (M9 < .99); US same core, changed legal part, disjoint numbers (M9 < .7) | +0.00018 |
| add2 / fill | M10 pairs q ≥ .95 (US/IN); France S1s left empty get their best non-twin M9 candidate (q ≥ .5) | +0.00003 |

## Licences and rules
LightGBM (MIT), polars (MIT), CE models as in REPRODUCE_M9.md (MIT/Apache-2.0, ≤ 8B). No external data or lookups.
Label-free test statistics used: twin-signature excess rates (AUDIT §8.3–8.4) and the France empty rate.
