#!/usr/bin/env bash
# build <team>_submission.zip in the exact required layout. code = committed HEAD (git archive), so no data,
# caches, venv or credentials can slip in; big TSVs are hard-linked into the staging dir (no copy).
# usage: tools/make_zip.sh <team_name> [matching_results.tsv]   (default: student_resource/output/matching_results.tsv)
set -euo pipefail
TEAM=$1
cd "$(dirname "$0")/.."
MATCH=$(realpath "${2:-student_resource/output/matching_results.tsv}")
C=student_resource/code/business_entity_resolution
[ -z "$(git status --porcelain -- "$C" student_resource/Documentation_template.md)" ] ||
  { echo "uncommitted changes in code or doc: commit first (zip must match a commit)"; exit 1; }
ST=student_resource/work/zipstage
rm -rf "$ST"; mkdir -p "$ST/output"
git archive HEAD "$C/src" "$C/README.md" "$C/requirements.txt" | tar -x -C "$ST"
mv "$ST/student_resource/code" "$ST/code"; rmdir "$ST/student_resource"
# CE scores are inputs (made on Kaggle GPUs, gitignored parquet): ship them if this build uses them
[ -d "$C/src/ce_scores" ] && cp -r "$C/src/ce_scores" "$ST/code/business_entity_resolution/src/"
ln "$MATCH" "$ST/output/matching_results.tsv"
ln student_resource/output/candidate_pairs.tsv "$ST/output/candidate_pairs.tsv"
cp student_resource/Documentation_template.md "$ST/"
# refuse anything that looks like a secret or a cache
bad=$(find "$ST" \( -name '.env' -o -name 'kaggle.json' -o -name '*.pem' -o -name '__pycache__' -o -name '.pytest_cache' -o -name '.venv' \) -print)
[ -z "$bad" ] || { echo "refusing, found: $bad"; exit 1; }
OUT=$PWD/${TEAM}_submission.zip
rm -f "$OUT"
(cd "$ST" && python3 -m zipfile -c "$OUT" output code Documentation_template.md)
rm -rf "$ST"
python3 - "$OUT" <<'EOF'
import sys, zipfile, hashlib
z = zipfile.ZipFile(sys.argv[1])
names = sorted(n for n in z.namelist() if not n.endswith("/"))
dirs = {}
for n in names:
    d, _, f = n.rpartition("/")
    dirs.setdefault(d, []).append(f)
for d in sorted(dirs):
    print(f"{d or '.'}/  ({len(dirs[d])} files)" + ("" if len(dirs[d]) > 6 else ": " + ", ".join(dirs[d])))
print("matching_results.tsv md5 in zip:", hashlib.md5(z.read("output/matching_results.tsv")).hexdigest())
EOF
echo "commit $(git rev-parse --short HEAD); $(du -h "$OUT" | cut -f1) $OUT"
