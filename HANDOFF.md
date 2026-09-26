# HANDOFF — read this first, then EXPERIMENTS.md

## SUN 27 SEP 03:20 IST — FREEZE (read docs/MORNING_BRIEF.md first)
- A/B: blend_raw 0.98488 < blend_cal 0.985131 -> **final France k = exact 0.18385635887853335**. Hail mary (transfer
  model, monotone, FP bucket): nothing passed, see docs/HAILMARY.md (code on local branch `hailmary`, not merged).
- Frozen config: round2.name lgbm2_ce, use_ce true, decode.model lgbm2_ce, blend.enabled true, blend.unseen_k
  0.18385635887853335, extra_k []. Tag `freeze-2026-09-27` = this commit, pushed with main.
- Backup: backup/final_2026-09-27.tar.zst (2.4 GB, md5 dafd9adf916023819d3ae98b43b63ef9) -> Drive (Muaz), then delete
  local tar. Next: delete work/ -> Muaz runs btrfs balance -> >= 5 GB unallocated -> cleanroom full from the tag ->
  accept if blend OOF within 0.0005 of 0.99032 -> upload clean-room blend_unseen_k0.183856 (LAST) -> zip + doc.

## SUN 27 SEP 00:35 IST (read docs/MORNING_BRIEF.md + docs/AUTO_RESULTS.md first)
- Best public: blend_cal 0.985131. Night: names2c FAILED LOCO, CE2 half +0.00015 (< +0.0003 gate), Soup CE2 never
  arrived, drop-format-features FAILED LOCO. **No new candidate.** Decode-rule LOCO: k = 1 best for an unseen country.
- 05:30 upload blend_raw (France k 1) vs blend_cal; decision rules in docs/MORNING_BRIEF.md.
- Freeze config: round2.name lgbm2_ce, use_ce true, decode.model lgbm2_ce, blend.enabled true, blend.unseen_k = A/B
  winner; names2.by_group false; no CE2. Then backup.sh -> (Muaz "go freeze") delete work/ caches -> cleanroom full.
- Disk ~5 GB free before work/ is deleted. Git: local commits since 21:00 NOT pushed (Muaz said no push tonight).

## SAT 26 SEP ~20:15 IST — CURRENT STATE (supersedes older sections where they differ)
- **Deadline Sun 27 Sep 23:59 IST** (official). Upload budget: 1 Sat night (blend_cal, Razan), 5 on Sunday after ~05:30.
- Public: ce_full_v1 0.980518; ce_half_v1 0.981; **blend_cal 0.985131 = best** (md5 0a0b14806bcddf3370258467ae3fddd9, rank ~110).
- Sunday uploads: 05:30 blend_raw -> France k point 2 (k 2-3 if raw wins, else 0.5) -> point 3 to bracket -> ~11:00 CE2 + best France k -> ~19:00 clean-room final (LAST). Backup 2nd copy = Drive (backup.sh keeps no local copy).
- **Blend** (ber/blend.py): union candidates; p = 0.5 ours (lgbm2_ce) + 0.5 QTpies M5 (src/razan/, shipped inputs,
  sha256 checked, REPRODUCE.md in); single-source pairs as-is. OOF on his live S1: ours 0.98764, M5 0.98818 (reproduced
  exactly), blend 0.99032. w fixed 0.5 (our side optimistic in his world).
- **Calibration** (ber/calibrate.py): anchor by data rule; seen-country DiD only if fold-vs-fold null test passes -> it
  FAILS for our models (India k 0.31-0.33) -> seen k = 1. Unseen (absent from train) = empty-rate matching (France k
  0.184 on blend). Seen-k curve on blend: k = 1 best.
- **France A/B files** (candidates/): blend_raw (k 1) da71932f, blend_unseen_k0.5 6e327bf7, blend_unseen_k2 31363753,
  blend_unseen_k3 861a28ca, blend_cal (k 0.184). France F change = score difference / 0.1498.
- Dropped Sat: synthetic unseen-country pairs (LOCO -0.002/-0.004), twin-decoy rule (blend already rejects twins; costs
  OOF), seed bagging, quantile LOCO, seen-k>1.
- **Sunday**: 05:30 upload blend_raw (A/B vs blend_cal) -> pick France k (k>1 allowed) -> CE2 ingest when files arrive
  (tools/ingest_ce2.sh <name> <ce2_ankit> <ce2_soup>; cutoff validated by 11:00) -> re-blend (blend.final with the
  CE2 model) -> 12:00 FREEZE needs Muaz's "go freeze" -> config flip: round2.name lgbm2_ce (or CE2 model),
  use_ce true (+ce_models/ce_mix), decode.model same, blend.enabled true, blend.unseen_k = A/B winner -> backup.sh ->
  delete work/ caches (abort < 45 GB free) -> tools/cleanroom.sh full <ref> (~6 h) -> accept if OOF within 0.0005 ->
  final upload by 20:00 (LAST upload) -> make_zip.sh (~1.4 GB with razan inputs), submit by 21:30.
- Root-cause fix Sat: pyarrow iter_batches leaked ~90 MB/batch -> ber.models.lgbm.batches reads row groups.

## Files that matter

| what | path | md5 | score |
|---|---|---|---|
| fallback (never touch) | `student_resource/output/matching_results.tsv` (copy in `work/fallback/`) | e7dbc8357c3d2035228854ddf8497556 | public 0.965, OOF 0.9773 |
| previous best | `student_resource/output/candidates/numbers_france_v1/matching_results.tsv` | cd673b353beb3e431f0a23da1ce3bd40 | **public 0.977**, OOF 0.98233 |
| **current best = upload baseline** | `student_resource/output/candidates/ce_half_v1/matching_results.tsv` | 7083a967b1c2bd32abbd39f00bad72f5 | **public 0.981**, OOF 0.98744 (H2) |
| next upload (CE both halves) | `student_resource/output/candidates/ce_full_v1/matching_results.tsv` | 3a0d4c771a1203acf207811ee2c8a2fe | public pending, OOF **0.98765** |
| candidate set (all runs share it) | `student_resource/output/candidate_pairs.tsv` | — | PC 0.9825 |

Leaderboard (Sat morning): rank 1 ~0.988, rank 50 ~0.984, us ~200th. Target: +0.007. France (15% of test, unseen
country) is the biggest lever; public gain of Retrain A (+0.012) was bigger than OOF (+0.005).

## Current model (config `src/configs/default.yaml`)
Normalize v1 -> TF-IDF blocking (fw + hh channels, K=10 each, within country) -> 44 string features -> LightGBM
round 1 (20% S1, 5 folds) -> round 2 on hard pairs (p >= 0.01): graph/group features + 15 soft house-number + 8 v2-name
features (`r2feats`), 3000 rounds = model `lgbm2_x` -> decode: owner argmax, isotonic, gate, expected-F (`expf_gate`).
`run_all.sh` is the ONE path to the final file (stages incl. `r2feats`, `ce`). `round2.use_ce: false` until CE adopted.

## Standing rules (from Muaz)
- Never modify `output/matching_results.tsv`. New outputs only in `output/candidates/<name>/`, validated (PASS +
  `--check-ids`), with md5 + per-country stats (`decode.write(out_dir=...)` does all of this).
- Country is never a feature; never hard-code `{US, India}` or `France`; unseen-country logic keys on "not in train".
- No network calls except package/model downloads (and git/Kaggle/Drive when Muaz asks). Disk > 15 GB. Commit after each step; log every run in `EXPERIMENTS.md`. Short reports.
- **Memory-safe queue only**: one heavy job (> 3 GB) at a time, through `tools/queue.sh`. Never launch under 6 GB free.
- LightGBM threads fixed at 16 (output changes with thread count). CPU is protected by nice/CPUWeight instead.
- Commits: caveman style, no AI attribution. Do not commit `CLAUDE.md` or the problem-statement files in `docs/`.
- No searches over the whole home dir; stay in the project.

## Queue (`tools/queue.sh`)
- `tools/queue.sh add <name> '<cmd run from src/>' [mem cap, default 11G]` ; `tools/queue.sh status`
- Log: `student_resource/work/queue/queue.log` (START/END/FAIL with runtime, peak RAM, reason). Job logs: `work/queue/logs/`.
- Runner = systemd user unit `ber-queue`; each job = own unit with MemoryMax cap. Survives Claude session end.
- Stop a job: find unit with `systemctl --user list-units 'run-*' --state=running`, then `systemctl --user stop <unit>`.
- Edit `tools/queue.sh` only by writing a temp file + `mv` (the running runner reads the old inode).

## Running / queued now (check `tools/queue.sh status`)
- `seed_bag` (`tools/seed_bag.sh lgbm2_ce ce_full_bag_v1`): 3 extra round-2 seeds (other feature/bagging fractions),
  mean with lgbm2_ce. Kept only if OOF gain >= +0.0005. Ends ~16:30.
- Done today: quantile LOCO dropped (-0.0014); repro_check = blocking 100% same, normalize 0 diff (no drift);
  CE half 1 (+0.0051 H2 OOF, public 0.981); CE full (OOF 0.98765). Details in EXPERIMENTS.md.

## Cross-encoder round 2 (CE2, for Soup + Ankit)
- Model `intfloat/multilingual-e5-base` (MIT), seed 43, SAME halves and bands as CE1: Ankit `--half 1`, Soup
  `--half 2`, script `src/gpu/ce_train_predict.py --model intfloat/multilingual-e5-base --seed 43`.
- Deliver the `pred/` parts (`train_h{1,2}_part*.parquet`, `test_h{1,2}_part*.parquet`) + an md5 list
  (`md5sum pred/*`) as `ce2_ankit` / `ce2_soup`. Freeze Sun 12:00: CE2 is used only if ingested + validated by then.
- Ingest (Muaz): verify md5 + rows (train H2 5,394,755 from Ankit, H1 5,388,242 from Soup; test 10,319,014 each),
  then `tools/queue.sh add ce2 "bash $PWD/tools/ingest_ce2.sh ce2_v1 <ce2_ankit_dir> <ce2_soup_dir>"`: tries ce2_p as
  its own feature and the mean of both CEs, keeps the better OOF, writes `output/candidates/ce2_v1/`.
- CE scores live in `src/ce_scores/` (ce_scores_*, ce2_scores_*), gitignored, shipped in the zip.

## Open items
- **CE adoption for the final run**: config `round2.name: lgbm2_ce`, `use_ce: true`, `decode.model: lgbm2_ce`
  (+ `ce_models` / `ce_mix` if CE2 wins). Set when the final config is picked Sun 10:00.
- Pseudo-labelling: DROPPED (sim -0.007 / -0.009). Unseen calibration correction: NOT adopted (+0.001). Blocking
  options: NOT built (best ceiling +0.0022 for 3x pairs). Nothing of these needs disclosure.
- Doc draft done (`student_resource/Documentation_template.md`): fill [TO CONFIRM] / [FINAL] slots. Zip:
  `tools/make_zip.sh <team> [tsv]` (dry run 1.0 GB, exact tree). Backup: `tools/backup.sh <final tsv> <model>`.

## Sunday plan (decided; also in memory `final-day-plan`)
- **12:00 freeze**: everything for the final model merged.
- Before clean room: build `backup/final_2026-09-27.tar.zst` (final matching + candidate_pairs, CE scores,
  EXPERIMENTS.md, OOF preds + reports for the doc) + md5 list; second copy outside `work/` (`../MLC-Zon-backup/`);
  Muaz uploads the tar to Drive in the browser. Then delete main `work/` caches; report free disk; abort if < 45 GB.
- **12:00-17:30 clean room**: `tools/queue.sh add cleanroom_full "bash $PWD/tools/cleanroom.sh full <ref matching.tsv>" 13G`
  (fresh clone + fresh uv env; dataset linked read-only; ~5.5 h, ~40 GB). Close browser/Discord.
- If output differs from ref: report md5, differing rows, clean OOF vs ref OOF. Small (< 1% rows AND OOF within
  +/-0.0005) -> clean file = final candidate, Razan uploads ~18:00. Large -> find cause; not fixed by 19:30 -> upload
  ref, README + doc state the discrepancy.
- **20:00** final upload (best by evidence). Zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt,ce_scores?}`, `Documentation_template.md`.
- Clean-room sample (3%) already PASSED end to end on Sat (see EXPERIMENTS.md).
