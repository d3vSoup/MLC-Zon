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
$PY -c "
# same-S1 checks: vs Retrain A (lgbm2_x) on the S1 this model is scored on; full run also vs CE-half model on H2
import os, numpy as np, pandas as pd
from ber.io import work
from ber.metric import f05
from ber.ce_export import halves
from ber.decode import s1_table
keys = s1_table('train')[0]
def per(m, sel):
    z = np.load(work('preds', f'oof_decoded_{m}.npz'), allow_pickle=True)
    f, c, G = f05(z['G'], z['k'], z['t']), z['country'], z['G']
    if len(f) == len(keys):
        f, c, G = f[sel], c[sel], G[sel]
    assert len(f) == sel.sum()
    return pd.Series({'overall': f.mean(), 'singleton_only': f[G == 0].mean(), **{k: f[c == k].mean() for k in sorted(set(c))}})
h = halves(keys)
checks = [('H$HALF' if '$HALF' else 'all S1', h == int('${HALF:-0}') if '$HALF' else np.ones(len(keys), bool), 'lgbm2_x')]
if not '$HALF' and os.path.exists(work('preds', 'oof_decoded_lgbm2_ceh.npz')):
    checks.append(('H2', h == 2, 'lgbm2_ceh'))
for tag, sel, base in checks:
    out = pd.DataFrame({base: per(base, sel), '$M': per('$M', sel)}).round(5)
    out['gain'] = out['$M'] - out[base]
    print(f'{tag} ({sel.sum()} S1)'); print(out.to_string())
    out.to_csv(work('reports', f'compare_${M}_vs_{base}.tsv'), sep='\\t')"
