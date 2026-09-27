## LIVE (newest first)
**Night 2 (21:20): no candidate passes the gate; FINAL 2 stays final. See [NIGHT2.md](NIGHT2.md).**

- 21:05 FINAL2 public 0.986758 = best AND last. 27f: Documentation FINAL = FINAL2 committed + tagged; zip build waits for disk (6.6 GB free < 8 GB floor). Night-2 Phase 1: country decomposition shows FINAL2 already = best known France (#34) + best US/IN (#35); diagnostics do not predict public deltas (conflicting signs).
- 20:40 27e ZIP VERIFIED: QTpies_submission.zip 2,581,827,078 bytes md5 0dd30abc9c6137e53f0a013a9860afdf (tag freeze-2026-09-27e eff3ceca; output = #34 c51d3853, candidates = union 5021548f). Unzipped: tree OK, no secrets/caches, fuse inputs MD5 OK, validator PASS + subset; #34 / #35 / FINAL2 regenerated FROM THE ZIP with matching md5s. Fallback kept: QTpies_submission_w07_27c.zip (9c6798a9).
- 20:28 27e package in progress: partner src @0aa2e02 + code/ (stage3 M10/M11, make_m11h, export), fuse inputs under src/razan/fuse (MD5.txt), README section D (exact commands -> #34 / #35 / FINAL2), Documentation FINAL = #34 (0.986731) with fuse/SOLO disclosures. M12h: evaluated, not used (0.99100 vs 0.99093).
- 20:22 MATCH table (production SOLO 3a0d4c77): #33 c52d413c, #34 c51d3853 (0.986731), #35 10f5696b (0.986666), FINAL2 50c695de (France #34 + US/IN #35) -- all MATCH, all PASS + --check-ids; #35/FINAL2 matches subset of union 5021548f (M10 union == M11h union). FINAL2 copy: ~/Downloads/final2/. Razan repo HEAD 0aa2e02; patch verified.
- 19:58 FUSE REPRODUCTION: #33 c52d413c MATCH, #34 c51d3853 (public 0.986731) MATCH, both with Razan's recipe = SOLO production ce_full_v1 3a0d4c77 (not clean-room 677f56f9). With clean SOLO: #33 3aeaeb5c / #34 9a7de36d, 174-176 rows differ (0.01%). Both PASS. M12h checks PASS (logloss 0.02476), blend w 0.0 running.
- 19:25 [BLEND] M11h w 0.0: OOF 0.99093 (US 0.99117 / IN 0.99059) vs M10 PW0.7 0.99070 -> passes Razan's go (> 0.99070). md5 e7a6eda3, France k 0.353, PASS + --check-ids, ~/Downloads/m11/. M11 alone 0.98956.
- 19:10 #33 (0.986655, md5 c52d413c) recipe in Razan's REPRODUCE_FINAL: base = M10 blend d754f06f (his doc says blend.w 0.5: wrong, it is blend.w 0.3) + solo 677f56f9 + M9 + M10 + norm; needs fuse.py @cb77fc3 (--pred-alt). Fuse venv ready (polars 1.44.2, selftest OK). Waiting for the M9 / final bundle.
- 18:45 NAMING: "M10 weight" = partner weight; config blend.w = our weight. M11 (Razan stacker on razan_export 8a189690/390c9fa8) auto-runner armed: on ~/Downloads/*m11*share*.zip -> checks vs M10 -> blend.w 0.0 (asis: pairs only we scored keep our p) + M11-alone OOF -> ~/Downloads/m11/. Log: work/queue/logs/m11_auto.log.
- 18:40 small M11 bundle ~/Downloads/for_razan_m11_small.zip 277 MB md5 e887d101; zipcheck dir deleted.
- 18:35 SAFETY-NET ZIP READY: QTpies_submission.zip 1,433,727,893 bytes md5 9c6798a9f1c929644fbbd11f58b1fe7e (from 27c 6610aff; output = d754f06f, candidates = union; unzipped check: tree OK, validator PASS incl. candidate shards + matches subset, no secrets/caches).
- 18:30 SAFETY NET 27c: src/razan -> M10 inputs (M5 inputs backed up in ../MLC-Zon-inputs/razan_m5), partner src @16a61c9, config blend.w 0.3 (partner 0.7) + unseen_k auto, Documentation [FINAL] filled (public 0.986456). Zip next.
- 18:25 M11 export for Razan: ~/Downloads/for_razan_m11.zip 4.45 GB md5 6803a670 (exports train 8,812,682 / test 9,531,458 pairs, M10 blends w0.7/w0.5, clean-room inputs).
- 18:20 blend_m10_w07 (d754f06f) public 0.986456; #32 0.986611 not reproducible -> cannot be final. Final = best of #33 / w0.7 / M11 (by 20:30).
- 18:05 [BLEND] M10 PW0.7 OOF 0.99070 (d754f06f), PW0.85 0.99055 (b2a34e78), PW0.5 0.99068 (660ee2f6); M10 alone 0.98899. Solo rebuilt 677f56f9 (exact). Razan fuse inputs in ~/Downloads/for_razan/ (blend_cal_cleanroom 9c36a54b, ce_full_v1_cleanroom 677f56f9). All PASS + --check-ids. No uploads until Muaz confirms.
- 17:15 CANDIDATES (labels): [BLEND] = our clean-room preds + partner stage-2 p, our decoder (has OOF on live S1). [FUSE] = Razan's fuse.py on top of a [BLEND] file + M9/M10 p + solo file, thresholds tuned on public LB (NO OOF: test-only). Partner alone (our decoder, live S1): M8 0.98843, M10 0.98899. [BLEND] M5 pw0.5 0.99032 (base, 9c36a54b), M8 pw0.5 0.99044 (a8942ebf), M10 pw0.7/0.85/0.5 running. [FUSE] #32 0.986611 public (production inputs), #33 = fuse on clean-room inputs (pending).
- 17:10 llm/ (Qwen2.5-7B-Instruct, Apache-2.0, 7.6B, local HF inference): not imported by fuse/stage3/src; no API calls; LB log says no-go (AUC 0.511). M10 zip md5 f23c9276 confirmed.
- 16:50 M10 bundle checks PASS (ids/dropped identical, same pairs, raw p, no twins, OOF logloss 0.0271 vs M8 0.0289). NOTE: per Razan's REPRODUCE_M10, 0.986611 = sub32 = fuse.py on PRODUCTION blend_cal (0a0b1480) + ce_full_v1 (3a0d4c77) + M9 + M10, not M10 alone; the M9 bundle is not here. M8 pw0.7 skipped; M10 blends (pw 0.7/0.85/0.5) chained after M8 pw0.5.
- 16:48 commit C 226035d, tag freeze-2026-09-27b pushed (pipeline code = freeze tag).
- 16:45 NEW BAR 0.986611 = Razan M10 alone. Base file is fallback only. Plan: M8 blends (partner w 0.5 / 0.7) as fallbacks; M10 blends (partner w 0.7 / 0.85 / 0.5) when the bundle arrives; model freeze 19:00; final LAST upload by 22:00; zip by 21:30.
- 16:30 CLEAN ROOM PASS: blend OOF 0.99032 (= ref), base final md5 9c36a54b (3,451 rows differ from blend_cal, 0.2%; candidates byte-identical). Copy: ~/Downloads/final_base/.
- 12:35 A (uploaded 05:30) = blend_raw (md5 da71932f) vs blend_cal: only France differs, 32,679 rows, A +35,347 links, 0 removed; US/India identical. France empty 5.25% vs 5.92%, links/S1 3.333 vs 3.197.
- 12:30 clean-room swap cap raised live to 14G (MemoryMax 13G unchanged).
- 12:15 train blocking done (peak 9.0 GB RAM + 8.0 GB swap = at the swap cap, survived). Test blocking running.
- 12:12 Documentation: variant-independent sections updated (uncommitted); [FINAL] slots left for the final variant.
- 12:10 CE compute from Kaggle logs: CE1 H1 5.61 h, H2 5.80 h (2x T4, torch 2.10.0+cu128); CE2 H1 5.65 h; CE2 H2 not accessible (Soup: ~12 h, team-reported).
- 11:54 clean room retry passed US train blocking (peak 10.4 GB RAM + 6.75 GB swap). Running; ETA ~17:00.
- 11:35 M8 rules cleared by Muaz (official PDF). Team QTpies; members Muhammed Razan (leader), Muaz Ismail Mohammed, Ankit Ranjan, Souparno Chakraborty.
- 11:09 clean room retry started (13G + 8G). 10:04 first run oom-killed at block --split train.

## UPLOAD RULES (11:20, Muaz)
- Clean room retry started 11:09 (same tag `freeze-2026-09-27`, MemoryMax 13G + MemorySwapMax 8G; first run died
  10:04 at `block --split train`, oom-kill at 11G + 2G). If it fails again: no retry, ship the production blend_cal zip.
- **Uploads:** M8 w0.5 first. Then w0.7 if w0.5 beats 0.985131, otherwise w0.3.
- **The final is the best M8 blend ONLY IF (a) the snapshot hash matched — DONE: `4594535af76c` matches — and (b) Muaz
  confirms the rules allow the country-keyed twin rates (see docs/M8_DISCLOSURE.md) — CLEARED 11:35 (official PDF:
  open country set, France fully processed; twin rates shape training augmentation only).** Otherwise the final
  is the base clean-room file (`blend_unseen_k0.183856`).
- If M8 wins: commit the staged M8 package (`../MLC-Zon-m8-staged/razan/` -> `src/razan/`) + `blend.w` +
  `blend.unseen_k: auto` to main, tag `freeze-2026-09-27b`, push, rebuild the zip from 27b.
- Team name QTpies -> `QTpies_submission.zip` (casing to confirm on Unstop).
- **The final must be the LAST upload, by 22:00. Zip submitted by 23:00. Deadline 23:59.**

**Hail mary (01:00-03:10): nothing passed, no hail-mary build. See [HAILMARY.md](HAILMARY.md).**

# Morning brief — Sun 27 Sep (written 00:20, updated 03:20)

## Result: blend_cal wins the France A/B
- blend_raw (France k 1) public **0.98488** vs blend_cal (France k 0.18385635887853335) **0.985131** -> France F
  about -0.0017 at k 1. **Final France k = exact blend_cal k 0.18385635887853335.** No k 0.5 probe.
- LOCO predicted the opposite, so treat LOCO as a weak proxy for France: only uploads confirm France changes.

## Timeline
- Freeze done early (~03:20, config flip committed, tag `freeze-2026-09-27` pushed with main).
  Config: round2.name lgbm2_ce, use_ce true, decode.model lgbm2_ce, blend.enabled true,
  blend.unseen_k 0.18385635887853335, extra_k [].
- Backup tar `backup/final_2026-09-27.tar.zst` (2.4 GB, md5 dafd9adf916023819d3ae98b43b63ef9): Muaz uploads it
  to Drive, then the local tar is deleted.
- work/ is deleted after the backup check. **Muaz runs the btrfs balance.** The clean room starts only when
  `btrfs filesystem usage /` shows >= 5 GB unallocated.
- Clean room from the tag: `FREEZE_REF=freeze-2026-09-27 tools/cleanroom.sh full
  student_resource/output/candidates/blend_cal/matching_results.tsv` (through tools/queue.sh). Full run_all, CE1
  only, Razan's files as inputs. The final output is `blend_unseen_k0.183856` (exact France k), validated with md5.
  Accept if the blend OOF is within 0.0005 of 0.99032 on the same rows. About 6 h (the old plan said done
  ~14:30-15:15 after an 08:45 freeze; an earlier start moves it forward).
- On failure: note the stage and error, no blind retry. Production candidates stay the fallback (blend_cal).
- Then: final upload (the clean-room file), then zip + doc.

## Uploads today (updated 10:00: M8 plan)
Clean room (M5 inputs, frozen tag) started 09:23, expected end ~15:30. Then, in the same clean-room folder, the blend
stage is re-run with QTpies **M8** inputs (staged at `../MLC-Zon-m8-staged/razan/`) at w 0.5 / 0.7 / 0.3 (config value
`blend.w`), France k re-derived by the empty-rate rule (`blend.unseen_k: auto`) each time. Outputs:
`candidates/cleanroom_m8_w05`, `_w07`, `_w03` (validated + md5; OOF on the same live S1 rows reported for each).

- Slot 1 (done): blend_raw 0.98488.
- Slot 2: **cleanroom_m8_w05**.
- Slot 3: **cleanroom_m8_w07** if w0.5 beats 0.985131, otherwise **cleanroom_m8_w03**.
- Last slot: **the final, and it MUST be the last upload.**
  - **If Muaz clears the rules for M8's country-keyed twin rates:** the best-scoring M8 blend (re-upload it if it is
    not already the last upload). Then commit the M8 inputs + `blend.w` + `unseen_k: auto`, tag, push, zip.
  - **Otherwise (not cleared, or no M8 blend beats 0.985131):** the base M5 clean-room file
    (`blend_unseen_k0.183856`, expected md5 = blend_cal 0a0b14806bcddf3370258467ae3fddd9), uploaded last.
- **M8 is NOT cleared yet.** Open items: rules check of the country-keyed twin rates and the test-fitted statistics
  (DBIN etc., Documentation draft in docs/M8_DISCLOSURE.md); the bundle's `src/` is not the M8 kernel
  snapshot (hash mismatch, `twins.py` / `cce.py` absent).

## Night outcome in one line
Nothing new beat `blend_cal`'s model. names2c failed LOCO (-0.0007 / -0.0003), the CE2 half-ingest gained only
+0.00015 (gate +0.0003), Soup's CE2 half never arrived. The decode-rule LOCO test says: no France correction (k = 1)
is best for an unseen country. Phase B (drop format-count features) is being tested — see the last section.

## History: the pre-upload prediction (wrong: cal won)
- **LOCO (India held out, labels):** no correction is best; k 0.25 cost -0.0061 F on the unseen country, k 0.5 cost
  -0.0018, k 2 cost -0.0007. If France behaves like unseen India: France F rises by up to ~0.005 at k = 1 vs 0.184,
  i.e. **raw wins by ~+0.0005 to +0.001 public**.
- **QTpies M4 probe (Razan):** France down-scaling helped (+0.002 public), but on a model that over-linked France
  (links/S1 3.447, empty 5.09%). Our blend under-links France (links/S1 3.333, assigned ratio 0.963 vs 0.973-0.979),
  so the LOCO reading is more likely to apply.
- France F change = (raw - cal) / 0.1498.

## History: the old decision rules (superseded by the result above; cal won = Branch B, no k 0.5 probe)
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

## Final config (frozen)
round2.name lgbm2_ce, use_ce true, decode.model lgbm2_ce, blend.enabled true, blend.unseen_k 0.18385635887853335,
extra_k []. names2.by_group stays false. No CE2. No hail-mary flag.
