# QTpies pipeline (step C): stage-2 pair probabilities for the blend and the fuse

This folder holds the second pipeline of the team (QTpies): its code and its job runner. Step C writes its outputs
here, where steps A and D (main README) read them. Everything below regenerates them from the challenge data.

## Outputs (location under `src/razan/`, md5 in our run)

| File | Content | Read by | md5 |
|---|---|---|---|
| `oof_s2ce.parquet` | M10 train out-of-fold stage-2 p (ri, si, y, p; live S1 only, 8,812,682 pairs) | step A blend | `5e7eaaa23bcd2a44565d53222d2c4dc2` |
| `pred_test_s2ce.parquet` | M10 test stage-2 p (ri, si, p; 9,531,458 pairs) | step A blend, step D | `16d67ec72c03e196071c6b1319ef693c` |
| `ids/n_test_s1` / `s2` / `s3` | si / ri -> entity id, country | steps A, D | `54d8729130a3a994f83a0095e9a72915` / `85f009cbe17a1f82a295eeeff6d2e602` / `82cfd1f0bcac38452c8866be7f5fc012` |
| `ids/n_train_s1` / `s2` / `s3` | the same for train | step A | `79415a426916e570211e43d2637489f7` / `2681072cf8210e527d6cf2609172cf26` / `6b0f7a688a00dd374910cee963712bf0` |
| `dropped_train_s1.parquet` | train S1 treated as absent (density matching, 16%) | step A | `cd705017c5c541892db8f1b56e30ce99` |
| `fuse/m9/pred_test_s2ce.parquet` | M9 test stage-2 p | step D | `8ea20cc57dbd275973e17602725e4ae9` |
| `fuse/m11h/oof_s2ce.parquet` / `pred_test_s2ce.parquet` | M11h train OOF / test p | step A (second blend), step D | `a381bdfd1558eb15b2aaa1f28493173d` / `e9ba15537cdc045408f2a4fe7e6594fd` |
| `fuse/norm/n_test_s1` / `s2` / `s3` | `norm.py` test output (copy of `<work>/n_test_s*.parquet`) | step D | `77fae2dcc777aeb25b4e5ae77f4d6c77` / `87cbd26172269abe053486e695c102e6` / `c129477e2c178c66bc17f8c49e4ad546` |

Intermediate files (not read by steps A or D): M9 kernel outputs `oof_s2ce.parquet` (twin rows included)
`e896e397d4e7809f2cfe343dfb6b7ea4` and `pred_test_s2ce.parquet` `8ea20cc57dbd275973e17602725e4ae9`; share OOF (twin
rows removed) `d56378be750f3fa5b048a86df0eeb8ee`; M11 `oof_s2ce.parquet` / `pred_test_s2ce.parquet`
`a381bdfd1558eb15b2aaa1f28493173d` / `31f05ada3219216d2d72aa4629e134ca`; step-A export `razan_export_train.parquet` /
`razan_export_test.parquet` `8a1896904d006e939ed4cc97499f135a` / `390c9fa88c31b8317e1ab302d80d1dd5`.

si = row of our normalised S1 file; ri = row of concat(S2, S3). Always map through `ids/`.

## Code

`src/` (run from `src/`):

| File | Role |
|---|---|
| `common.py` | paths (`BER_WORK`), threads, the density-matching drop (`BER_DROP`) |
| `norm.py` | normalisation: transliteration, legal forms, fillers, core name, address, numbers |
| `block.py` | TF-IDF blocking within country, top-15 S1 per record |
| `feats.py` | pair features; `stage2.py`: collective stage-2 features (owner competition, sibling context, twin and number features) |
| `model.py`, `run.py` | stage 1 and stage 2 LightGBM (2 folds by S1 hash), job entry points |
| `ce.py`, `cce.py` | pairwise and cluster cross-encoders (GPU) |
| `twins.py` | synthetic twin negatives generated from train rows (twin world) |
| `decide.py`, `score.py` | expected-F0.5 decoder, the challenge metric |
| `make_share.py` | M9 kernel outputs -> share bundle (id maps, dropped S1, OOF without twin rows) |
| `stage3.py` | stage 3: M10 (rival-row recalibration) and M11 (stacker with the step-A model's p) |
| `make_m11h.py` | M11h = M11 for US/India, M10 for France |
| `fuse.py`, `merge_by_country.py` | step D (main README) |

`kaggle_runner.py` pushes each step as a private Kaggle kernel that embeds `src/` as a tarball.
`export_for_razan.py` runs in the step-A environment (C.5 below).

## Environment

- CPU: `requirements_qtpies.txt` (Python 3.12). **polars must be exactly 1.44.2**: the folds (`hash(si, 42) % 2`), the
  density drop (`hash(si, 49) % 100 < 16`) and the stage-3 folds use polars `hash()`, which is version-dependent.
- GPU (`ce.py`, `cce.py`): `requirements_qtpies_gpu.txt` (the Kaggle GPU image: torch 2.10.0+cu128, transformers 5.0.0).
  Base models: `intfloat/multilingual-e5-small` (MIT, 118M, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`) and
  `intfloat/multilingual-e5-base` (MIT, 278M). The scripts load them by name; set `BER_CE_BASE` / `BER_CCE_BASE` to a
  local snapshot to pin a revision.
- Kaggle kernels: private, CPU (4 vCPU, 31 GB) or GPU (2x T4). Internet is used only for `pip` and the model download.

## Reproduce from the challenge data

Working-copy layout (the paths `common.py` and `kaggle_runner.py` expect): this `src/` at
`<repo>/code/business_entity_resolution/src/`, `kaggle_runner.py` at `<repo>/code/business_entity_resolution/`, the
data at `<repo>/student_resource/dataset/`. `export PYTHONUTF8=1`.

**C.1 Normalise (laptop).** `BER_WORK=<work> python norm.py` writes `n_{train,test}_s{1,2,3}.parquet`, `translit.parquet`,
`admin_{train,test}.parquet` and `gt_pairs.parquet`. Copy `<work>/n_test_s{1,2,3}.parquet` to `fuse/norm/`. Test blocking
for France and India also runs locally: `BER_WORK=<work> BER_THREADS=8 python block.py test France` (then `India`).

**C.2 Private Kaggle datasets** (`kaggle datasets create`, private by default; the account and dataset names are set at
the top of `kaggle_runner.py`):

| Dataset | Content |
|---|---|
| `qtpies-er-norm` | `n_*.parquet`, `translit.parquet`, `admin_*.parquet`, `gt_pairs.parquet` |
| `qtpies-er-data` | raw `{train,test}_s{1,2,3}.parquet` and `gt_pairs.parquet` (cached by `norm.py` in `<work>`) |
| `qtpies-er-cand-test` | the France and India test candidates from C.1, named `cand_test__{France,India}_{nnn}.parquet` |

**C.3 Kaggle job chain.** From `code/business_entity_resolution/`, `python kaggle_runner.py chain <batch> ...`. A chain
pushes each batch, waits for its kernels and feeds their outputs to later batches (`kernel_sources`), so the order
matters. Per-batch settings are in `BATCH_ENV` in `kaggle_runner.py`.

| # | Batches | Hardware | Output |
|---|---|---|---|
| 1 | `block`, `block_test_us` | CPU | train candidates, US test candidates |
| 2 | `feats_test feats_train` | CPU | pair features |
| 3 | `ce_train model ce_score` | GPU / CPU / GPU | a first cross-encoder and stage-1 model (inputs of `m2_s2`) |
| 4 | `ftd_train m2_fit1 m2_pred m2_s2` | CPU | density-matched features and stage 1 (`BER_DROP=16`) |
| 5 | `ce4_train ce4_score ce2_train ce2_score` | GPU | two pairwise cross-encoders (e5-small; normalised and raw text) scoring every stage-2 pair |
| 6 | `t_gen t_feats t_ce4 t_ce2 m7b` | CPU / CPU / GPU / CPU / CPU | the twin world: 90,665 house-number twins (US) and 26,498 generic-word twins (US, India) generated from train, blocked, scored; stage 1 on it |
| 7 | `cce cceb` | GPU (2x T4 per kernel) | two cluster cross-encoders (e5-small, e5-base): one sequence = an S1 and all its stage-2 rows |
| 8 | `m9` | CPU | M9 stage 2: `oof_s2ce.parquet` (twin rows included), `pred_test_s2ce.parquet` |

Fetch: `kaggle kernels output <account>/qtpies-m9 -p <m9> --file-pattern "pred_test_s2ce\.parquet$|oof_s2ce\.parquet$"`.

**C.4 Share bundle and M10 (laptop, `requirements_qtpies.txt`).**
```
cd code/business_entity_resolution/src
BER_WORK=<work> BER_DROP=16 python make_share.py <m9> <share>        # ids/, dropped_train_s1, OOF without twin rows
BER_WORK=<work> python stage3.py <share> <m10> <work>/gt_pairs.parquet    # M10 (~10 min, ~6 GB)
```
Copy `<m10>/oof_s2ce.parquet`, `<m10>/pred_test_s2ce.parquet`, `<share>/ids/` and `<share>/dropped_train_s1.parquet`
to `src/razan/`, and `<share>/pred_test_s2ce.parquet` to `src/razan/fuse/m9/`. Step A can now run.

**C.5 M11 and M11h (after step A).**
```
cd code/business_entity_resolution/src && PYTHONPATH=. python razan/export_for_razan.py   # step-A env -> work/razan_export_*.parquet
BER_WORK=<work> python stage3.py <share> <m11> <work>/gt_pairs.parquet <dir with razan_export_*.parquet>
python make_m11h.py <share> <m10> <m11> ../fuse/m11h                  # from src/razan/src
```

## Determinism

- Byte-exact given the pins: normalisation, folds and density drop, twin generation (seed 42), `make_share.py`,
  `stage3.py` (LightGBM seed 7, 8 threads), `make_m11h.py`, `fuse.py`, `merge_by_country.py`. From the M9 kernel
  outputs and the `norm.py` files, C.4 and C.5 reproduce `ids/`, `dropped_train_s1.parquet`, the M10 test p, M11 and
  M11h byte for byte; the M10 train OOF matches to one float32 ulp (888 of 8,812,682 values differ, by at most 2.3e-10).
- Blocking: deterministic; top-15 selection could break exact score ties differently at another thread count.
- Not byte-exact on a rerun from scratch: the Kaggle LightGBM stages (4 threads, `deterministic` not set) and the
  cross-encoders (fp16, `DataParallel`, cuDNN). A full rerun lands within about 1e-4 of the OOF F0.5.
