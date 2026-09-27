# Business Entity Resolution — pipeline

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data.
No network access is needed after the pip install. No external data, APIs, or geocoding are used.

## Layout expected

```
<root>/                       # student_resource/ in dev, or the zip root
  dataset/{train,test}/*.tsv  # challenge data (not shipped)
  utils/validate_submission.py
  output/                     # written here
  work/                       # caches (created)
  code/business_entity_resolution/{src,README.md,requirements.txt}
```

Paths are set in `src/configs/default.yaml`, relative to `<root>`. Override the root with `BER_ROOT=/path`.

## What regenerates what

| step | where | hardware | inputs | output shipped in this folder |
|---|---|---|---|---|
| A. our pipeline + blend (`src/run_all.sh`) | this folder | CPU, 16 GB RAM | challenge data + the two shipped inputs below | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| B. our cross-encoder scores | `src/gpu/ce_train_predict.py` | Kaggle GPU (2x T4) | challenge data (via `ber.cli ce_export`) | `src/ce_scores/ce_scores_{train,test}.parquet` |
| C. partner pipeline predictions (QTpies) | `src/razan/src/` + `src/razan/kaggle_runner.py` | Kaggle CPU + GPU (2x T4) | challenge data | `src/razan/{oof_s2ce,pred_test_s2ce,dropped_train_s1}.parquet`, `src/razan/ids/` |
| D. final file (fuse, CPU, ~2 min each) | `src/razan/src/fuse.py`, `merge_by_country.py` | CPU, ~5 GB RAM | step A blend files + partner p (M9, M11h, M12h) + partner normalised test records + fixed SOLO file | `output/matching_results.tsv` (final) |

Step A regenerates both output files from the challenge data plus the shipped B and C outputs. A clean-room run
(fresh clone of the frozen tag, fresh environment, dataset only, 320 min on 16 GB RAM + swap) reproduced the
candidate set byte for byte and the blend OOF exactly (0.99032). 3,451 of 1,732,544 S1 rows (0.2%) of the final
file differed from the reference run, with links added and removed in balance. That drift comes from the model
stages, not from blocking. B and C are GPU steps; their code,
settings and commands are in this folder, and re-running them reproduces the inputs up to GPU / multi-thread noise
(fp16 cuDNN kernels, LightGBM without `deterministic` in C: about 1e-4 of OOF F0.5), so a full rerun of B or C gives a
near-identical, not byte-identical, final file.

## Environments

- `requirements.txt`: step A (Python 3.11, all transitive pins).
- `requirements_ce_gpu.txt`: step B, on the Kaggle GPU image.
- `requirements_qtpies.txt` (+ `requirements_qtpies_gpu.txt` for its cross-encoders): step C, Python 3.12. It is a
  separate environment because its pins conflict with step A's (lightgbm, numpy, pyarrow, scikit-learn, scipy).

## B. Cross-encoder scores (only if `round2.use_ce: true`)

`src/ce_scores/ce_scores_{train,test}.parquet` (s1, cand, ce_p) come from `src/gpu/ce_train_predict.py`
(xlm-roberta-base, MIT) run on 2x Tesla T4 GPUs (Kaggle, torch 2.10.0+cu128): 5.61 h (half 1) and 5.80 h (half 2) per run, then
`python -m ber.ce_ingest <dir_half1> <dir_half2>`. The scores are keyed by entity ids, so the CPU pipeline uses them
whatever the row order of its candidate list. To regenerate them: `python -m ber.cli ce_export`, run the GPU script
with `--half 1` and `--half 2`, then `ber.ce_ingest`.

## C. Partner pipeline predictions (only if `blend.enabled: true`)

`src/razan/` ships the stage-2 pair probabilities of a teammate's independent pipeline (QTpies M10: train
out-of-fold `oof_s2ce.parquet`, test `pred_test_s2ce.parquet`), its id maps (`ids/`) and its dropped-S1 list
(`src/razan/README.md`: files and SHA-256). Its code is in `src/razan/src/` (the partner's
source at git commit `16a61c9` of their repo: M9 code + `stage3.py`; kernel snapshot hashes in `src/razan/README.md`),
with the job runner `src/razan/kaggle_runner.py`. To regenerate:
1. Put `src/razan/src/` at `code/business_entity_resolution/src/` and `kaggle_runner.py` at
   `code/business_entity_resolution/` of a working copy (the layout both files expect), with the challenge data at
   `student_resource/dataset/`.
2. Follow `src/razan/REPRODUCE.md` (steps 1-12: normalise, block, features, stage 1, pairwise cross-encoders, M5), then `REPRODUCE_M8.md` (steps 13-19: twin world, cluster cross-encoder, M8),
   `REPRODUCE_M9.md` (`cceb` cluster cross-encoder, M9) and `REPRODUCE_M10.md` (stage 3: `python stage3.py <M9 share> <M10 share>`).
   Each step is `python kaggle_runner.py chain <batch>`; the tables there list hardware, settings, inputs and outputs.
   Datasets used are uploads of the challenge data and of intermediate outputs; no external data.
3. Copy `oof_s2ce.parquet`, `pred_test_s2ce.parquet`, `dropped_train_s1.parquet` and `ids/` into `src/razan/`.

The `blend` stage (last in `run_all.sh`) joins them to ours by entity ids: candidates = union of both pipelines,
p = w x ours + (1 - w) x theirs (`blend.w` in the config; a pair only one side scored keeps its p), isotonic + gate +
expected-F decode fitted on their density-matched live train S1, then label-free test-time calibration
(`ber/calibrate.py`: odds factor for countries absent from train, `blend.unseen_k`: a fixed value or `auto` =
empty-rate matching; the seen-country shift is applied only if its fold-vs-fold null test passes). It writes
`output/candidates/blend_{raw,cal}/` and copies the chosen file to `output/`.

## D. Final file: submission-level fuse (the uploaded final)

The final `output/matching_results.tsv` is the partner's `fuse.py` applied on top of our blend files. It vetoes team
links that the partner's stage-2 p rejects and adds partner pairs on unowned records, with per-country thresholds (see
`src/razan/FUSE.md`, `REPRODUCE_FUSE.md` and the Documentation). Every input is shipped under `src/razan/`, and
`src/razan/fuse/MD5.txt` lists their md5s. Environment: `requirements_qtpies.txt` (polars 1.44.2).

```bash
cd code/business_entity_resolution/src/razan
COMMON="--pred fuse/m9/pred_test_s2ce.parquet --ids ids --veto France=0.7,US=0.3,India=0.3 --add France=0.99,US=0.99,India=0.99 \
  --solo fuse/solo/ce_full_v1_matching_results.tsv --veto-solo France=0.9,US=0.7,India=0.7 --add-solo France=0.9,US=0.9,India=0.9 \
  --norm fuse/norm --veto-g France=0.99 --veto-n US=0.7 --fill France=0.5"
# #34 (public 0.986731): fuse on the M10 blend, US/India rules read M12h p
python src/fuse.py fuse/blends/m10_w03_matching_results.tsv out34 $COMMON --pred-alt fuse/m12h/pred_test_s2ce.parquet --alt-countries US,India
#   -> out34/matching_results.tsv md5 c51d385383f2fca625aa8ff24eb6d573
# #35 (public 0.986666): fuse on the M11h blend, US/India rules read M11h p
python src/fuse.py fuse/blends/m11h_w00_matching_results.tsv out35 $COMMON --pred-alt fuse/m11h/pred_test_s2ce.parquet --alt-countries US,India
#   -> out35/matching_results.tsv md5 10f5696b16da21ba9634ad6040110b3f
# FINAL 2: France rows of #34 + US/India rows of #35
python src/merge_by_country.py out34/matching_results.tsv out35/matching_results.tsv ids/n_test_s1.parquet out_final2 --a-countries France
#   -> out_final2/matching_results.tsv md5 50c695de7ae6c03d19a70ee75040f983
```

- **Blend inputs** (`fuse/blends/`) are step A outputs. `m10_w03` (md5 `d754f06f8800aa5516a2a67d7a1ab2e4`) is
  `run_all.sh` with this config (M10 inputs in `src/razan/`, `blend.w: 0.3`, `unseen_k: auto`). `m11h_w00` (md5
  `e7a6eda3527f643798c82680d47d2fed`) is the same run with `fuse/m11h/*.parquet` copied into `src/razan/` and
  `blend.w: 0.0`. Both were made in our clean room from the frozen tag.
- **SOLO** (`fuse/solo/`, md5 `3a0d4c771a1203acf207811ee2c8a2fe`) is our single-pipeline file from the reference run
  (`ce_full_v1`). The clean-room rebuild gives md5 `677f56f9bdce08a61a77f6a478492b99`. With that file, 174-176 of
  1,732,544 fused rows (0.01%) change, and empty shares and links per S1 stay equal to 4 decimals. So it ships as a
  fixed input.
- **Partner p:** M9 comes from `REPRODUCE_M9.md`; M11h and M12h come from `code/stage3_m11.py` / `src/stage3.py`
  (M12) + `code/make_m11h.py`, see `README_M11.md` / `README_M12.md`. They are fed our clean-room p through
  `code/export_for_razan.py`.

## Run (CPU only)

```bash
cd code/business_entity_resolution
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cd src
PY=../.venv/bin/python ./run_all.sh          # every stage skips if its output exists
PY=../.venv/bin/python python -m pytest -q tests
```

Single stage: `$PY -m ber.cli <stage> [--split train|test] [--force]`.
Stages in order (as in `run_all.sh`): `normalize, folds, block, blockreport, features, train` (round 1, 5 folds,
OOF), `predict --split test`, `graph` (group features on pairs with round-1 p >= 0.01), `r2feats` (soft house-number + v2 name
features on those pairs), `ce` (cross-encoder score, only if `round2.use_ce`), `stack` (round 2, 3000
rounds), `decode` (isotonic + gate + expected-F, all out-of-fold), `report`, `write`, `blend` (only if `blend.enabled`). `eda` = Phase 0 facts.
LightGBM runs with a fixed 16 threads (`n_jobs` in the config): its output changes with the thread count, so do not
change it if you want byte-identical files.
`write` runs the official validator (the matching file whole, with `--check-ids`; the candidate file in row shards, using the validator's own functions, to stay inside 16 GB RAM).

## Hardware and time (measured)

Intel i7-11800H (16 threads), 16 GB RAM + swap (train blocking peaked at 10.4 GB RAM + 6.75 GB swap), no GPU used.
Clean-room run 27 Sep: 320 min end to end. Earlier development timings (about 5.2 h): normalize 7 min, blocking
92 min, features 60 min, round-1 train 52 min + OOF predict 14 min, test predict 13 min, graph 2 min, round 2 55 min,
decode 4 min, write + validation 8 min. Peak RAM about 10 GB; about 40 GB of disk for `work/`.
Per-stage times are appended to `work/reports/timings.tsv`.
