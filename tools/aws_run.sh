#!/usr/bin/env bash
# start the full pipeline on the AWS box, detached (survives ssh drop); log in work/reports/aws_run.log
# usage: tools/aws_run.sh [ssh-host]
set -euo pipefail
H=${1:-ber-aws}
ssh "$H" 'cd ~/MLC-Zon/student_resource/code/business_entity_resolution/src && mkdir -p ../../../work/reports \
  && nohup env PY=../../../../.venv/bin/python ./run_all.sh > ../../../work/reports/aws_run.log 2>&1 < /dev/null & echo started'
