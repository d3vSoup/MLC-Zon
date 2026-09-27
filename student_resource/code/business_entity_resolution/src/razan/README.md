# Partner pipeline inputs (QTpies M10) for the blend

The blend (`ber/blend.py`, stage `blend`) reads only these files (md5):

| file | rows | md5 |
|---|---|---|
| `oof_s2ce.parquet` (train out-of-fold stage-2 p; live S1 only; no synthetic twin rows) | 8,812,682 | `5e7eaaa23bcd2a44565d53222d2c4dc2` |
| `pred_test_s2ce.parquet` (test stage-2 p, raw, before any calibration) | 9,531,458 | `16d67ec72c03e196071c6b1319ef693c` |
| `ids/n_test_s1` / `s2` / `s3` | | `54d8729130a3a994f83a0095e9a72915` / `85f009cbe17a1f82a295eeeff6d2e602` / `82cfd1f0bcac38452c8866be7f5fc012` |
| `ids/n_train_s1` / `s2` / `s3` | | `79415a426916e570211e43d2637489f7` / `2681072cf8210e527d6cf2609172cf26` / `6b0f7a688a00dd374910cee963712bf0` |
| `dropped_train_s1.parquet` (their density-matching drop, 16% of train S1) | | `cd705017c5c541892db8f1b56e30ce99` |

**M10** = M9 + stage 3. M9 = M8 + a second cluster cross-encoder feature (`cceb`, multilingual-e5-base, MIT). M8 = M5 +
twin-world training + a cluster cross-encoder (`cce`, multilingual-e5-small, MIT). Stage 3 (`src/stage3.py`, LightGBM)
recalibrates M9's stage-2 p with label-free same-name rival-row counts, cross-fitted by S1 on train. Train CV (their
decoder, live S1): M9 0.98852, M10 0.98898. The same pairs, ids and dropped S1 as M5 / M8, so the blend union is unchanged.

## How these files are made
- `REPRODUCE.md`: steps 1-12 (normalise, block, features, stage 1, pairwise cross-encoders, M5).
- `REPRODUCE_M8.md`: steps 13-19 (twin world, cluster cross-encoder, M8).
- `REPRODUCE_M9.md`: the `cceb` cluster cross-encoder and M9.
- `REPRODUCE_M10.md` + `README_M10.md`: stage 3 (`python stage3.py <M9 share> <M10 share>`, deterministic, ~10 min).

## Code (git commit `0aa2e02` of github.com/razancodes/MLC-raZON)
- `src/`: the partner's source at `0aa2e02`: M9 code, `stage3.py` (M12 version), `fuse.py`, `merge_by_country.py`.
  The M9 Kaggle kernels ran snapshot `3313ebdb1a60` and the M8 kernels `4594535af76c` (the same code with the
  kernels' own line endings; see REPRODUCE_M8 / M9).
- `code/`: the exact versions that made the shipped partner files.
  - `stage3_m10.py` (M10, md5 `7a913c23…`) and `stage3_m11.py` (M11, md5 `c740279f…`); M12 is `src/stage3.py` (`316cd967…`).
  - `make_m11h.py`: the h-variants (US/India from the stacker, France from M10).
  - `export_for_razan.py`: our clean-room p for the stacker.
- `kaggle_runner.py`: the job runner. `AUDIT.md`: the partner's label-free test audit (source of the twin parameters).
- `LB_LOG_2026-09-27.md`: every leaderboard probe of 27 Sep. The fuse thresholds were set from these probes.

## Fuse inputs (`fuse/`, md5s in `fuse/MD5.txt`)
- `m9/pred_test_s2ce.parquet` (M9 test p), `m11h/`, `m12h/` (stacker bundles: train OOF + test p).
- `norm/n_test_s{1,2,3}.parquet`: the partner's normalised test records.
- `solo/ce_full_v1_matching_results.tsv`: our reference-run single-pipeline file (fixed input, see the code README
  section D).
- `blends/`: our two clean-room blend files the fuse runs on.
- Recipes: `REPRODUCE_FUSE.md`, `REPRODUCE_FINAL.md`, `FUSE.md`; exact commands in the code README, section D.
