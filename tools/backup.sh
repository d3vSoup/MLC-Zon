#!/usr/bin/env bash
# Sunday 11:00, before work/ caches go: final files + CE scores + logs + final model's OOF -> tar.zst + md5 list.
# second copy outside the repo (../MLC-Zon-backup/), checked by md5. Muaz uploads the tar to Drive in the browser.
# usage: tools/backup.sh <final matching_results.tsv> <final model name, e.g. lgbm2_x or lgbm2_ce>
set -euo pipefail
cd "$(dirname "$0")/.."
MATCH=$(realpath --relative-to=. "$1"); M=$2
W=student_resource/work; C=student_resource/code/business_entity_resolution
N=final_$(date +%F); B=backup; COPY=../MLC-Zon-backup
mkdir -p "$B" "$COPY"
files=("$MATCH" student_resource/output/candidate_pairs.tsv EXPERIMENTS.md HANDOFF.md "$W/reports"
       "$W/preds/oof_decoded_$M.npz" "$W/preds/oof_$M.parquet" student_resource/output/candidates)
[ -d "$C/src/ce_scores" ] && files+=("$C/src/ce_scores")
for f in "${files[@]}"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done
git rev-parse HEAD > "$B/$N.commit"
find "${files[@]}" -type f -print0 | sort -z | xargs -0 md5sum > "$B/$N.md5"
tar -I 'zstd -T0 -10' -cf "$B/$N.tar.zst" "${files[@]}" "$B/$N.md5" "$B/$N.commit"
zstd -tq "$B/$N.tar.zst"
cp "$B/$N.tar.zst" "$B/$N.md5" "$B/$N.commit" "$COPY/"
a=$(md5sum < "$B/$N.tar.zst"); b=$(md5sum < "$COPY/$N.tar.zst")
[ "$a" = "$b" ] || { echo "copy md5 differs"; exit 1; }
echo "backup ok: $B/$N.tar.zst ($(du -h "$B/$N.tar.zst" | cut -f1), $(wc -l < "$B/$N.md5") files, md5 ${a%% *})"
echo "copy: $COPY/$N.tar.zst; final matching md5 $(md5sum < "$MATCH" | cut -d' ' -f1); free disk $(df -h --output=avail . | tail -1)"
