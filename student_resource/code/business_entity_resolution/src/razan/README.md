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

## Code
- `src/`: the partner's source at git commit `16a61c9` of github.com/razancodes/MLC-raZON (M10: stage 3 + fuse). It
  holds 16 files. SHA-256 over the sorted `src/*.py` (contents concatenated), first 12 hex, is `24bed6efd449` for
  these git (LF) files; without `stage3.py` / `fuse.py` it is `9f0cc9ee1551`.
- The M9 Kaggle kernels ran snapshot `3313ebdb1a60`: the same code with the kernel's mixed CRLF/LF line endings
  (REPRODUCE_M9.md). The M8 kernels ran `4594535af76c` (tag `m8`).
- `fuse.py` is not used by this submission's blend.
- `kaggle_runner.py`: the job runner at the same commit. `AUDIT.md`: the partner's label-free test audit (source of the
  twin parameters, disclosed in the Documentation, section 4.5).
