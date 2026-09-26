#!/usr/bin/env bash
# Sunday 11:00, before work/ caches go: final files + CE scores + logs + final model's OOF -> tar.zst + md5 list.
# second copy = Drive: Muaz uploads the tar in the browser (disk is tight; no local second copy).
# usage: tools/backup.sh <final matching_results.tsv> <final model name, e.g. lgbm2_x or lgbm2_ce>
set -euo pipefail
cd "$(dirname "$0")/.."
MATCH=$(realpath --relative-to=. "$1"); M=$2
W=student_resource/work; C=student_resource/code/business_entity_resolution
N=final_$(date +%F); B=backup
mkdir -p "$B"
files=("$MATCH" student_resource/output/candidate_pairs.tsv EXPERIMENTS.md HANDOFF.md "$W/reports"
       "$W/preds/oof_decoded_$M.npz" student_resource/output/candidates)  # per-S1 OOF npz (80 MB) instead of the 1.6 GB pair file: disk
[ -d "$C/src/ce_scores" ] && files+=("$C/src/ce_scores")
for f in "${files[@]}"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done
git rev-parse HEAD > "$B/$N.commit"
find "${files[@]}" -type f -print0 | sort -z | xargs -0 md5sum > "$B/$N.md5"
tar -I 'zstd -T0 -10' -cf "$B/$N.tar.zst" "${files[@]}" "$B/$N.md5" "$B/$N.commit"
zstd -tq "$B/$N.tar.zst"
a=$(md5sum < "$B/$N.tar.zst")
echo "backup ok: $B/$N.tar.zst ($(du -h "$B/$N.tar.zst" | cut -f1), $(wc -l < "$B/$N.md5") files, md5 ${a%% *})"
echo "upload to Drive: $PWD/$B/$N.tar.zst (check md5 ${a%% *} after upload); final matching md5 $(md5sum < "$MATCH" | cut -d' ' -f1); free disk $(df -h --output=avail . | tail -1)"
