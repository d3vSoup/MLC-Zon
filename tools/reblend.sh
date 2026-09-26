#!/usr/bin/env bash
# re-blend after a new round-2 model (e.g. CE2): blend_raw + blend_cal (+ calfull if null passes) from <model>, then a
# fixed odds k for countries absent from train if <unseen_k> is a number. Writes output/candidates/<name>_*.
# usage: tools/reblend.sh <model e.g. lgbm2_ce2_sep> <name e.g. blend2> [unseen_k: auto|number]   (via tools/queue.sh)
set -euo pipefail
M=$1; NAME=$2; K=${3:-auto}
cd "$(dirname "$0")/../student_resource/code/business_entity_resolution/src"
../../../../.venv/bin/python -c "
import ber.blend as b
b.final('$M', 0.5, 'asis', '$NAME')
if '$K' != 'auto':
    b.unseen_variants('$M', 0.5, 'asis', ks=(float('$K'),), name='$NAME')"
