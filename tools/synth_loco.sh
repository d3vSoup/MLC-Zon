#!/usr/bin/env bash
# honest test of synthetic unseen-country pairs (LOCO): synthetic dataset from train S1 (labels never used for the
# unseen side) -> same normalize / block / features in a separate root -> round-1 source model scores it -> round-2
# inputs -> round 2 on real source rows + synthetic unseen rows (weight 0.3 and 1.0) -> score the REAL unseen country.
# usage: tools/synth_loco.sh [frac]   (run through tools/queue.sh; ~2 h)
set -euo pipefail
FRAC=${1:-0.5}
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
W=$(cd ../../../work && pwd); SR=$W/synth/train_root
[ -e $SR/dataset/train/train_ground_truth.tsv ] || $PY -c "import ber.synth as s; s.build('train', '$SR', frac=$FRAC)"
for st in normalize block features; do BER_ROOT=$SR $PY -m ber.cli $st --split train; done
[ -e $W/reports/synth_feat_compare.done ] || $PY -c "
# key similarity features: real vs synthetic true pairs (and negatives), per country
import numpy as np, pandas as pd, pyarrow.parquet as pq
from ber.io import read_tsv, source_file, to_key
F = ['nc_tset', 'nn_tset', 'sk_tset', 'ad_tset', 'adnl_tset', 'house_eq', 'postal', 'num_jacc', 'fw_score', 'n_first']
out = []
for tag, root in (('real', '$W'), ('synth', '$SR/work')):
    pf = pq.ParquetFile(f'{root}/train/feats.parquet')
    lab = pq.read_table(f'{root}/train/cands.parquet', columns=['label']).column(0).to_numpy()
    off = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])]
    g = range(0, pf.num_row_groups, max(1, pf.num_row_groups // 12))  # ~12 row groups: low RAM
    x = pd.concat([pf.read_row_group(i, columns=['s1'] + F).to_pandas().assign(label=lab[off[i]:off[i + 1]])
                   for i in g], ignore_index=True)
    c = read_tsv(source_file('train', 1), usecols=['entity_id', 'country'])
    x['country'] = pd.Series(c.country.values, index=to_key(c.entity_id)).reindex(x.s1.values).values
    out.append(x.groupby(['country', 'label'])[F].mean().assign(src=tag))
print(pd.concat(out).reset_index().set_index(['country', 'label', 'src']).sort_index().round(3).to_string())"
for D in us2in in2us; do
  SRC=$([ $D = us2in ] && echo US || echo India)
  $PY -c "
import pandas as pd
from ber.io import read_tsv, source_file, to_key
from ber import loco
s1 = read_tsv(source_file('train', 1), usecols=['entity_id', 'country'])
loco.round1('$SRC', pd.Series(s1.country.values, index=to_key(s1.entity_id)), '$D', fold0_only=True)"
  BER_ROOT=$SR $PY -c "from ber.loco import synth_prep; synth_prep('$D', '$W/loco/$D/r1_fold0.txt')"
  for WT in 0.3 1.0; do
    $PY -c "from ber.loco import variant; variant('$D', ('num', 'names2'), synth='$SR', synth_w=$WT)"
  done
done
