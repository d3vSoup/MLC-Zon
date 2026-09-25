#!/usr/bin/env bash
# merge teammates' cross-encoder outputs into round 2, retrain round 2 (3000 rounds), re-tune decoder on OOF,
# write output/candidates/<name>/ (validated, md5 + per-country stats). Never touches output/matching_results.tsv.
# usage: tools/ingest_ce.sh <name> <ankit_output_dir> <soup_output_dir>
set -euo pipefail
NAME=$1; shift
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
$PY -m ber.ce_ingest "$@"
$PY -c "from ber.models import stack; stack.fit(name='lgbm1_ce', rounds=3000, test=True, use_ce=True)"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = 'lgbm1_ce'
import json, ber.decode as d
d.run()
print(json.dumps(json.load(open(d.work('reports', 'decode_lgbm1_ce.json')))['decoders'], indent=1))
d.write(out_dir='../../../output/candidates/$NAME')"
