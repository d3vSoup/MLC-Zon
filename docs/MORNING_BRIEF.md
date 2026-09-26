# Morning brief — Sun 27 Sep (written 00:20, updated 01:00)

## Timeline (freeze moved to ~06:00)
- 05:30 upload blend_raw; Muaz sends its score + "go freeze" together.
- ~06:00 freeze: set config (round2.name lgbm2_ce, use_ce true, decode.model lgbm2_ce, blend.enabled true,
  blend.unseen_k = A/B winner), commit, tag `freeze-2026-09-27`, push main + tag, `tools/backup.sh` (tar to Drive),
  delete work/ caches (abort if free < 45 GB), `FREEZE_REF=freeze-2026-09-27 tools/cleanroom.sh full <ref tsv>`.
- ~11:30 clean room done: its blend stage writes France k 1 (`blend_unseen_k1`, = blend_raw), the exact blend_cal k
  0.18385635887853335 (`blend_unseen_k0.183856`) and k 0.5 (`blend_unseen_k0.5`), each validated with md5;
  accept if OOF within 0.0005 of the reference (0.99032).
- Upload the clean-room file at the chosen k (must be the LAST upload). Zip + doc done by 15:00.


## Night outcome in one line
Nothing new beat `blend_cal`'s model. names2c failed LOCO (-0.0007 / -0.0003), the CE2 half-ingest gained only
+0.00015 (gate +0.0003), Soup's CE2 half never arrived. The decode-rule LOCO test says: no France correction (k = 1)
is best for an unseen country. Phase B (drop format-count features) is being tested — see the last section.

## 05:30 upload: `blend_raw` (France k = 1) vs `blend_cal` (France k 0.184, public 0.985131)
- **LOCO (India held out, labels):** no correction is best; k 0.25 cost -0.0061 F on the unseen country, k 0.5 cost
  -0.0018, k 2 cost -0.0007. If France behaves like unseen India: France F rises by up to ~0.005 at k = 1 vs 0.184,
  i.e. **raw wins by ~+0.0005 to +0.001 public**.
- **QTpies M4 probe (Razan):** France down-scaling helped (+0.002 public), but on a model that over-linked France
  (links/S1 3.447, empty 5.09%). Our blend under-links France (links/S1 3.333, assigned ratio 0.963 vs 0.973-0.979),
  so the LOCO reading is more likely to apply.
- France F change = (raw - cal) / 0.1498.

## Upload order (slots 2 and 3) with decision rules
- **Branch A — raw > cal (by >= 0.0002):** France k = 1 is the final setting (`blend.unseen_k: 1.0`).
  Slot 2: optional confirmation `blend_unseen_k1.5` (LOCO predicts a small loss; skip it if you want to save slots).
  Slot 3: the clean-room final (must be the LAST upload).
- **Branch B — cal > raw:** France down-scaling helps here. Slot 2: `blend_unseen_k0.5`.
  If k 0.5 > both: final k 0.5. If k 0.5 is between: final = the better of 0.184 / 0.5. Slot 3: clean-room final.
- **Tie (|diff| < 0.0002):** keep k = 1 (the LOCO-supported, simpler setting).

## Candidates (all validated PASS + --check-ids, union candidates)
| name | md5 | France k | France empty | France links/S1 | note |
|---|---|---|---|---|---|
| blend_cal | 0a0b14806bcddf3370258467ae3fddd9 | 0.184 | 5.92% | 3.197 | public 0.985131 (best) |
| blend_raw | da71932fa9a3fb048715ae267d9a23e3 | 1.0 | 5.25% | 3.333 | 05:30 A/B |
| blend_unseen_k0.5 | 6e327bf79470ecc4bcbd851bd22aa1fd | 0.5 | 5.61% | 3.272 | branch B |
| blend_unseen_k1.5 | 2240d33a2fce773d812ba3e339243d41 | 1.5 | 5.06% | 3.375 | branch A (optional) |
| blend_unseen_k2 | 31363753439f13e061b82b3a9749ce64 | 2.0 | 4.94% | 3.412 | not recommended |
| ce_full_cal | f3fc911c3beb11f12489a6555c7d4180 | 0.196 | 6.02% | 3.184 | single-pipeline fallback |

OOF (Razan's live S1, same rows): blend 0.99032 (US 0.99049, India 0.99007, singleton 0.99372).

## Phase A / B (docs/LOCO_ERRORS.md)
- Unseen-country errors are 4-15x the seen ones, with confident false positives: a ranking gap, not calibration.
- Most-shifted features: house-number format (n_num_a/b, hn_in_other, hn_suffix; 0.6-1.0 SD).
- Tested fixes: names2c (name priors per country group) LOCO -0.0007 / -0.0003; dropping the format-count features
  LOCO -0.0052 / -0.0028. **Nothing passed the gates. No new candidate from the night.**
- France addresses look US-like (one house number), so the India-unseen format shift overstates France's problem;
  LOCO remains the best available proxy.

## Final-config reminder (for the 12:00 freeze)
round2.name lgbm2_ce, use_ce true, decode.model lgbm2_ce, blend.enabled true, blend.unseen_k = the A/B winner
(1.0 for Branch A / tie; 0.184 or 0.5 for Branch B). names2.by_group stays false. No CE2.
