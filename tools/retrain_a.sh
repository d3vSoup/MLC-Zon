#!/usr/bin/env bash
# Retrain A: round 2 (3000 rounds) + soft house-number features + v2 name features, on hard pairs only.
# Writes output/candidates/numbers_france_v1/ (validated, md5, per-country stats). Never touches the fallback.
set -euo pipefail
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
PY=../../../../.venv/bin/python
for s in train test; do
  $PY -c "
import pyarrow.parquet as pq
from ber.io import work
from ber.features import numbers, names2
rows = pq.read_table(work('$s', 'r2.parquet'), columns=['row']).column(0).to_numpy()
numbers.run('$s', force=True, rows=rows, out=work('$s', 'r2_num.parquet'))
names2.run('$s', rows, work('$s', 'r2_names2.parquet'))"
done
$PY -c "from ber.models import stack; stack.fit(name='lgbm2_x', rounds=3000, test=True, extra=('r2_num', 'r2_names2'))"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = 'lgbm2_x'
import ber.decode as d
d.run(); d.write(out_dir='../../../output/candidates/numbers_france_v1')"
$PY -c "
from ber.io import CFG; CFG['decode']['model'] = 'lgbm2_x'
import ber.analysis as a; a.run()"
