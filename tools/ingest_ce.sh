#!/usr/bin/env bash
# Retrain B: teammates' cross-encoder scores as one more round-2 feature, on top of Retrain A
# (house numbers + v2 names). Keys the scores by ids into ce_scores/ (ships with the code), retrains round 2
# (3000 rounds), re-tunes decoder on OOF, loss buckets, writes output/candidates/<name>/ (validated, md5 +
# per-country stats). Never touches output/matching_results.tsv. Config is not changed: adopt with round2.use_ce.
# HALF=2: only one teammate done (CE model trained on H1 scores H2 + test) -> round 2 trained, tuned and scored
# on H2 S1 only (model lgbm2_ceh), then compared with lgbm2_x on the same H2 S1.
# usage: [HALF=2] tools/ingest_ce.sh <name> <ce_output_dir>...   (run it through tools/queue.sh)
set -euo pipefail
NAME=$1; shift
HALF=${HALF:-}
M=lgbm2_ce${HALF:+h}
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
$PY -m ber.ce_ingest "$@"
$PY -c "
from ber.io import CFG; CFG['round2']['use_ce'] = True
import ber.ce_ingest as c; c.run('train', force=True); c.run('test', force=True)"
$PY -c "from ber.models import stack; stack.fit(name='$M', rounds=3000, test=True, use_ce=True, extra=('r2_num', 'r2_names2'), half=${HALF:-None})"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = '$M'; CFG['decode']['half'] = ${HALF:-None}
import json, ber.decode as d
d.run()
print(json.dumps(json.load(open(d.work('reports', 'decode_$M.json')))['decoders'], indent=1))
d.write(out_dir='../../../output/candidates/$NAME')"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = '$M'
import ber.analysis as a; a.run()"
[ -z "$HALF" ] || $PY -c "
# same-S1 check: base lgbm2_x decoded on all S1, restricted to half $HALF, vs $M
import json, numpy as np, pandas as pd
from ber.io import work
from ber.metric import f05
from ber.ce_export import halves
from ber.decode import s1_table
keys, country, _ = s1_table('train')
h = halves(keys) == $HALF
out = {}
for m, sel in (('lgbm2_x', h), ('$M', slice(None))):
    z = np.load(work('preds', f'oof_decoded_{m}.npz'), allow_pickle=True)
    f, c, G = f05(z['G'], z['k'], z['t'])[sel], z['country'][sel], z['G'][sel]
    assert len(f) == h.sum()
    out[m] = {'overall': f.mean(), 'singleton_only': f[G == 0].mean(), **{k: f[c == k].mean() for k in pd.unique(c)}}
out = pd.DataFrame(out).round(5)
out['gain'] = out['$M'] - out['lgbm2_x']
print(f'H$HALF S1: {h.sum()}'); print(out.to_string())
out.to_csv(work('reports', 'h${HALF}_compare_$M.tsv'), sep='\t')"
