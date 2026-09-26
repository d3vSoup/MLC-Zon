#!/usr/bin/env bash
# clean-room run: fresh clone of origin + fresh uv env from requirements.txt only. never uses the main .venv
# or main work/ caches (only tools/make_sample.py reads them, to cut the sample).
#   tools/cleanroom.sh sample                          # ~3% S1 per country, stand-in CE scores: plumbing test
#   tools/cleanroom.sh full <reference matching.tsv>   # real data (linked read-only) + real CE scores; diff vs ref
# run it through tools/queue.sh. clone goes to ../MLC-Zon-clean-<mode>/ (wiped first).
set -euo pipefail
MODE=$1; REF=${2:-}
MAIN=$(cd "$(dirname "$0")/.." && pwd)
DEST=$(dirname "$MAIN")/MLC-Zon-clean-$MODE
case $MODE in sample|full) ;; *) echo "mode must be sample or full"; exit 1 ;; esac
rm -rf "$DEST"
git clone -q "$(git -C "$MAIN" remote get-url origin)" "$DEST"
echo "clone at $(git -C "$DEST" log --oneline -1)"
SR=$DEST/student_resource; CODE=$SR/code/business_entity_resolution
cd "$CODE"
uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
if [ "$MODE" = sample ]; then
  "$MAIN/.venv/bin/python" "$MAIN/tools/make_sample.py" "$SR"
  sed -i 's/^  name: lgbm2_x .*/  name: lgbm2_ce/; s/^  use_ce: false .*/  use_ce: true/; s/^  model: lgbm2_x .*/  model: lgbm2_ce/' \
    src/configs/default.yaml
  grep -c "lgbm2_ce\|use_ce: true" src/configs/default.yaml
else
  ln -s "$MAIN/student_resource/dataset" "$SR/dataset"
  cp -r "$MAIN/student_resource/code/business_entity_resolution/ce_scores" "$CODE/" 2>/dev/null || echo "no CE scores"
fi
cd src
t=$SECONDS
PY=../.venv/bin/python ./run_all.sh
echo "pipeline wall $(( (SECONDS - t) / 60 )) min"
../.venv/bin/python -m pytest -q tests
du -sh "$SR/work" "$SR/output"
md5sum "$SR/output/matching_results.tsv" "$SR/output/candidate_pairs.tsv"
if [ -n "$REF" ]; then
  md5sum "$REF"
  ../.venv/bin/python - "$REF" "$SR/output/matching_results.tsv" <<'EOF'
import sys
import pandas as pd
a, b = (pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        .set_index("source1_entity_id")["matched_entity_ids"] for f in sys.argv[1:])
srt = lambda s: s.map(lambda x: ",".join(sorted(x.split(","))) if x else "")
a, b = srt(a), srt(b.reindex(a.index).fillna(""))
d = a != b
print(f"S1 rows with a different match list: {int(d.sum())} of {len(a)}")
print(pd.DataFrame({"ref": a[d], "clean": b[d]}).head(20).to_string())
EOF
fi
