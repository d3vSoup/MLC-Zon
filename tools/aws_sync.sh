#!/usr/bin/env bash
# copy repo + dataset to the AWS box and build the env there (no git push, no keys copied)
# usage: tools/aws_sync.sh [ssh-host]   (default host alias: ber-aws in ~/.ssh/config)
set -euo pipefail
H=${1:-ber-aws}
cd "$(dirname "$0")/.."
ssh "$H" "mkdir -p ~/MLC-Zon/student_resource/work"
rsync -a --info=progress2 --exclude .venv --exclude .env --exclude 'student_resource/work' \
      --exclude 'student_resource/output/*.tsv' ./ "$H":~/MLC-Zon/
rsync -a student_resource/work/folds.parquet "$H":~/MLC-Zon/student_resource/work/
ssh "$H" 'cd ~/MLC-Zon && (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) \
  && export PATH=$HOME/.local/bin:$PATH && uv venv --python 3.11 .venv \
  && uv pip install --python .venv/bin/python -r student_resource/code/business_entity_resolution/requirements.txt \
  && nproc && free -g'
