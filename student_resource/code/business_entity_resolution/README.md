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

## Run (CPU only, v0)

```bash
cd code/business_entity_resolution
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cd src
PY=../.venv/bin/python ./run_all.sh          # every stage skips if its output exists
PY=../.venv/bin/python python -m pytest -q tests
```

Single stage: `$PY -m ber.cli <stage> [--split train|test] [--force]`.
Stages: `normalize, folds, block, blockreport, features, train, predict, decode, report, write` (`eda` = Phase 0 facts).
`write` runs the official validator (the matching file whole, with `--check-ids`; the candidate file in row shards, using the validator's own functions, to stay inside 16 GB RAM).

## Hardware and time (v0, measured)

Intel i7-11800H (16 threads), 16 GB RAM, no GPU used. See `work/reports/timings.tsv` for the time of each stage.
