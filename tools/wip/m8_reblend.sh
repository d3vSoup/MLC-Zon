#!/usr/bin/env bash
# M8 variants inside the finished clean room: M5 blend outputs moved aside, src/razan swapped to the staged M8 bundle,
# blend stage re-run on the clean-room lgbm2_ce preds at w 0.5 / 0.7 / 0.3 (config value blend.w), France k by the
# empty-rate rule (unseen_k auto) each time -> output/candidates/cleanroom_m8_w05 / _w07 / _w03 (+ _raw = k 1)
set -euo pipefail
D=/home/Muaz/Documents/Software/MLC-Zon-clean-full
SR=$D/student_resource; C=$SR/code/business_entity_resolution; CAND=$SR/output/candidates
M8=/home/Muaz/Documents/Software/MLC-Zon-m8-staged/razan
free_ok() { [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 5 ] || { echo "free disk < 5 GB: stop"; exit 1; }; }
[ -e "$SR/output/candidates_m5" ] && { echo "M5 outputs already moved aside: stop"; exit 1; }
free_ok
mkdir "$SR/output/candidates_m5"
for d in blend_raw blend_cal blend_calfull blend_unseen_k0.183856; do
  [ -e "$CAND/$d" ] && mv "$CAND/$d" "$SR/output/candidates_m5/"
done
cp "$SR/output/matching_results.tsv" "$SR/output/candidates_m5/final_matching_results.tsv"
REF=$SR/output/candidates_m5/blend_raw/candidate_pairs.tsv  # M5 union; M8 has the same pair set -> same union
mv "$C/src/razan" "$C/src/razan_m5"
mkdir "$C/src/razan"
cp -r "$M8"/oof_s2ce.parquet "$M8"/pred_test_s2ce.parquet "$M8"/dropped_train_s1.parquet "$M8"/ids \
  "$M8"/README.md "$M8"/REPRODUCE.md "$M8"/REPRODUCE_M8.md "$C/src/razan/"
Y=$C/src/configs/default.yaml
sed -i 's/^  unseen_k: 0.18385635887853335 .*/  unseen_k: auto        # M8 variants: France k re-derived by empty-rate matching/' "$Y"
cd "$C/src"
for w in 0.5 0.7 0.3; do
  free_ok
  tag=w0${w#0.}
  sed -i "s/^  w: [0-9.]* .*/  w: $w                # M8 variant/" "$Y"
  grep -n "^  w:\|^  unseen_k:" "$Y"
  ../.venv/bin/python -m ber.cli blend
  mv "$CAND/blend_cal" "$CAND/cleanroom_m8_$tag"
  mv "$CAND/blend_raw" "$CAND/cleanroom_m8_${tag}_raw"
  [ -e "$CAND/blend_calfull" ] && mv "$CAND/blend_calfull" "$CAND/cleanroom_m8_${tag}_calfull"
  for d in "$CAND/cleanroom_m8_$tag" "$CAND/cleanroom_m8_${tag}_raw"; do  # one 2 GB union file for all variants
    cmp -s "$REF" "$d/candidate_pairs.tsv" && ln -f "$REF" "$d/candidate_pairs.tsv" || echo "union differs from M5: kept own copy in $d"
  done
  md5sum "$CAND/cleanroom_m8_$tag/matching_results.tsv"
  python3 -c "import json; s=json.load(open('$CAND/cleanroom_m8_$tag/stats.json')); print('$tag OOF', s['oof_live']['overall'], {c: v.get('k') for c, v in s.items() if isinstance(v, dict) and 'k' in v})"
done
