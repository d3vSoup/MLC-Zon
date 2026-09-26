#!/usr/bin/env bash
# round-2 seed bagging: 3 extra fits of the base model (other seed, feature/bagging fractions), mean with the base
# -> <base>_bag; decode + OOF vs base; writes output/candidates/<name>/ only if the OOF gain >= +0.0005.
# CE settings must match the base: CE_MODELS (default "['ce']") and CE_MIX (default sep).
# usage: [CE_MODELS="['ce', 'ce2']" CE_MIX=mean] tools/seed_bag.sh <base model, e.g. lgbm2_ce> <name>
set -euo pipefail
B=$1; NAME=$2
CEM=${CE_MODELS:-"['ce']"}; MIX=${CE_MIX:-sep}
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
V=("{'seed': 1, 'feature_fraction': 0.7, 'bagging_fraction': 0.8}"
   "{'seed': 2, 'feature_fraction': 0.8, 'bagging_fraction': 0.7}"
   "{'seed': 3, 'feature_fraction': 0.9, 'bagging_fraction': 0.9}")
for i in 1 2 3; do
  $PY -c "
from ber.io import CFG; CFG['round2'].update(use_ce=True, ce_models=$CEM, ce_mix='$MIX')
from ber.models import stack
stack.fit(name='${B}_s$i', rounds=3000, test=True, use_ce=True, extra=('r2_num', 'r2_names2'), over=${V[$((i - 1))]})"
done
$PY -c "from ber.models import stack; stack.blend(['$B', '${B}_s1', '${B}_s2', '${B}_s3'], '${B}_bag')"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = '${B}_bag'
import json, ber.decode as d
d.run()
oof = lambda m: (lambda j: j['decoders'][j['chosen']]['oof'])(json.load(open(d.work('reports', f'decode_{m}.json'))))
a, b = oof('$B'), oof('${B}_bag')
print('base', a); print('bag ', b)
gain = b['overall'] - a['overall']
print(f'bag gain {gain:+.5f} (keep if >= +0.0005)')
if gain >= 0.0005:
    d.write(out_dir='../../../output/candidates/$NAME')"
