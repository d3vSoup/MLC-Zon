#!/usr/bin/env bash
# P1: label-free test-time calibration of a decoded model (ber.calibrate): fold-vs-fold null test, DiD k for seen
# countries, empty-rate k for countries absent from train; writes output/candidates/<name>/ (validated, md5, stats)
# and rows changed vs <base candidate>. usage: tools/calibrate.sh <model> <name> <base candidate name>
set -euo pipefail
M=$1; NAME=$2; BASE=$3
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = '$M'
import ber.calibrate as c, ber.decode as d
r = c.fit()
d.write(out_dir='../../../output/candidates/$NAME', kc=r['k'])"
$PY -c "
import pandas as pd
from ber.io import read_tsv, source_file
r = lambda n: read_tsv(f'../../../output/candidates/{n}/matching_results.tsv').set_index('source1_entity_id').iloc[:, 0] \
    .map(lambda x: ','.join(sorted(x.split(','))) if x else '')
a, b = r('$BASE'), r('$NAME')
c = read_tsv(source_file('test', 1), usecols=['entity_id', 'country']).set_index('entity_id').country.reindex(a.index)
d = a != b.reindex(a.index)
print(f'rows changed vs $BASE: {int(d.sum())} ({d.mean():.4f}); by country', d.groupby(c).sum().to_dict())"
