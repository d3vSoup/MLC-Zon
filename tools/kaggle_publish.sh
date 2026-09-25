#!/usr/bin/env bash
# publish the 3 PRIVATE Kaggle datasets for the cross-encoder (token must be at ~/.config/kaggle/kaggle.json or
# ~/.kaggle/kaggle.json, chmod 600; this script never reads or prints it). Sharing with teammates: Kaggle UI
# (dataset -> Settings -> Sharing), the CLI cannot add collaborators.
# usage: tools/kaggle_publish.sh <your_kaggle_username>
set -euo pipefail
U=$1
cd "$(dirname "$0")/.."
K=${KAGGLE_BIN:-.venv/bin/kaggle}
[ -x "$K" ] || uv pip install --python .venv/bin/python -q kaggle==2.2.4
S=student_resource/work/kaggle
W=student_resource/work/ce
rm -rf "$S" && mkdir -p "$S"/{ce-train-pairs,ce-test-pairs,ber-code}
ln "$W/ce_train_pairs.parquet" "$W/ce_band_train.parquet" "$S/ce-train-pairs/"   # hard links: no extra disk
ln "$W/ce_band_test.parquet" "$S/ce-test-pairs/"
cp student_resource/code/business_entity_resolution/src/gpu/ce_train_predict.py docs/CE_FOR_TEAMMATES.md "$S/ber-code/"
for d in ce-train-pairs ce-test-pairs ber-code; do
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$d" "$U" "$d" > "$S/$d/dataset-metadata.json"
  "$K" datasets create -p "$S/$d" -q    # private by default (no --public)
done
echo "published (private): https://www.kaggle.com/datasets/$U/{ce-train-pairs,ce-test-pairs,ber-code}"
