#!/usr/bin/env bash
# second cross-encoder (multilingual-e5-base, same halves + bands): keys its scores as ce2 in src/ce_scores/, then
# tries (a) ce2_p as its own round-2 feature and (b) one ce_p = mean of both models; keeps the better OOF.
# writes output/candidates/<name>/ (validated, md5 + per-country stats) from the winner. run through tools/queue.sh.
# usage: [EXTRA="r2_num r2_names2c"] tools/ingest_ce2.sh <name> <ce2_ankit_dir> <ce2_soup_dir>
set -euo pipefail
NAME=$1; shift
EX=$(python3 -c "print(tuple('${EXTRA:-r2_num r2_names2}'.split()))")
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
$PY -m ber.ce_ingest --tag ce2 "$@"
$PY -c "
from ber.io import CFG; CFG['round2'].update(use_ce=True, ce_models=['ce', 'ce2'])
import ber.ce_ingest as c; c.run('train', force=True); c.run('test', force=True)"
for MIX in sep mean; do
  M=lgbm2_ce2_$MIX
  $PY -c "
from ber.io import CFG; CFG['round2'].update(use_ce=True, ce_models=['ce', 'ce2'], ce_mix='$MIX')
from ber.models import stack; stack.fit(name='$M', rounds=3000, test=True, use_ce=True, extra=$EX)"
  $PY -c "
from ber.io import CFG; CFG['decode']['model'] = '$M'
import ber.decode as d; d.run()"
done
$PY -c "
# pick the better CE2 variant on OOF, compare with the one-CE model, write the candidate from the winner
import json
from ber.io import CFG, work
oof = lambda m: json.load(open(work('reports', f'decode_{m}.json')))
res = {m: oof(m)['decoders'][oof(m)['chosen']]['oof'] for m in ('lgbm2_ce', 'lgbm2_ce2_sep', 'lgbm2_ce2_mean')}
for m, r in res.items():
    print(m, r)
best = max(('lgbm2_ce2_sep', 'lgbm2_ce2_mean'), key=lambda m: res[m]['overall'])
print(f'winner {best}: overall {res[best][\"overall\"] - res[\"lgbm2_ce\"][\"overall\"]:+.5f} vs lgbm2_ce')
CFG['decode']['model'] = best
import ber.decode as d
d.NAME = best
d.write(out_dir='../../../output/candidates/$NAME')
import ber.analysis as a
a.NAME = best
a.run()"
