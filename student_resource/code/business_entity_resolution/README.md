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

## Cross-encoder scores (only if `round2.use_ce: true`)

`src/ce_scores/ce_scores_{train,test}.parquet` (s1, cand, ce_p) ship with this folder. They come from
`src/gpu/ce_train_predict.py` (xlm-roberta-base, MIT) run on 2x T4 GPUs (Kaggle), about X h per half, then
`python -m ber.ce_ingest <dir_half1> <dir_half2>`. The scores are keyed by entity ids, so the CPU pipeline
uses them whatever the row order of its candidate list. To regenerate them: `python -m ber.cli ce_export`,
run the GPU script with `--half 1` and `--half 2`, then `ber.ce_ingest`.

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
rounds), `decode` (isotonic + gate + expected-F, all out-of-fold), `report`, `write`. `eda` = Phase 0 facts.
LightGBM runs with a fixed 16 threads (`n_jobs` in the config): its output changes with the thread count, so do not
change it if you want byte-identical files.
`write` runs the official validator (the matching file whole, with `--check-ids`; the candidate file in row shards, using the validator's own functions, to stay inside 16 GB RAM).

## Hardware and time (measured)

Intel i7-11800H (16 threads), 16 GB RAM (+ swap), no GPU used. About 5.2 h end to end: normalize 7 min, blocking
92 min, features 60 min, round-1 train 52 min + OOF predict 14 min, test predict 13 min, graph 2 min, round 2 55 min,
decode 4 min, write + validation 8 min. Peak RAM about 10 GB; about 40 GB of disk for `work/`.
Per-stage times are appended to `work/reports/timings.tsv`.
