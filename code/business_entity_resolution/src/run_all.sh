#!/usr/bin/env bash
# full pipeline, each stage skips if its output exists (use --force on a stage to redo)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
run() { echo "== $*"; local t=$SECONDS; $PY -m ber.cli "$@"; echo "   wall $((SECONDS - t))s"; }
run normalize --split train
run normalize --split test
run folds
run block --split train
run block --split test
run blockreport --split train
run features --split train
run features --split test
run train
run predict --split test
run graph --split train
run graph --split test
run r2feats --split train
run r2feats --split test
run ce --split train     # no-op unless round2.use_ce
run ce --split test
run stack
run decode
run report
run write
run blend               # no-op unless blend.enabled (second pipeline's preds in src/razan/)
