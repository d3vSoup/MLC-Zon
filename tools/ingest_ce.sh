#!/usr/bin/env bash
# Retrain B: teammates' cross-encoder scores as one more round-2 feature, on top of Retrain A
# (house numbers + v2 names). Retrain round 2 (3000 rounds), re-tune decoder on OOF, loss buckets,
# write output/candidates/<name>/ (validated, md5 + per-country stats). Never touches output/matching_results.tsv.
# usage: tools/ingest_ce.sh <name> <ankit_output_dir> <soup_output_dir>   (run it through tools/queue.sh)
set -euo pipefail
NAME=$1; shift
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
$PY -m ber.ce_ingest "$@"
$PY -c "from ber.models import stack; stack.fit(name='lgbm2_ce', rounds=3000, test=True, use_ce=True, extra=('r2_num', 'r2_names2'))"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = 'lgbm2_ce'
import json, ber.decode as d
d.run()
print(json.dumps(json.load(open(d.work('reports', 'decode_lgbm2_ce.json')))['decoders'], indent=1))
d.write(out_dir='../../../output/candidates/$NAME')"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = 'lgbm2_ce'
import ber.analysis as a; a.run()"
