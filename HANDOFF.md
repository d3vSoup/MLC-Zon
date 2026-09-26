# HANDOFF — read this first, then EXPERIMENTS.md

State at **Sat 26 Sep 2026 ~11:00 IST**. Amazon ML Challenge 2026, Business Entity Resolution.
Deadline **Sun 27 Sep 23:59 IST**. Muaz = lead/integrator. Razan = only uploader (Google Drive, md5-checked).
Soup + Ankit = cross-encoder (CE) runs on Kaggle GPUs. Repo: origin = `razancodes/MLC-Zon` (private), branch `main`.

## Files that matter

| what | path | md5 | score |
|---|---|---|---|
| fallback (never touch) | `student_resource/output/matching_results.tsv` (copy in `work/fallback/`) | e7dbc8357c3d2035228854ddf8497556 | public 0.965, OOF 0.9773 |
| **current best = upload baseline** | `student_resource/output/candidates/numbers_france_v1/matching_results.tsv` | cd673b353beb3e431f0a23da1ce3bd40 | **public 0.977**, OOF 0.98233 |
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
1. `loco_us2in_x_qn`, `loco_in2us_x_qn`: per-country percentile copies of the top-15 similarity features
   (`ber.loco.variant(..., qnorm=...)`). Baseline to beat (unseen side): **0.91571 (US->IN) / 0.96935 (IN->US)**;
   in-distribution: US 0.98531 / India 0.97808. Rule: if >= +0.005 unseen in BOTH directions and in-distribution not
   hurt -> propose full retrain `numbers_france_v2` with ETA (needs the same percentile features in `r2feats` for
   train + test, computed per country on each split's own rows, no labels).
2. `repro_check` (`tools/repro_check.py`): re-normalize 50k rows per file + re-block 20k records per country with
   today's code vs cached files. Why: blocking probe re-block matched production pairs only 98.5% (US) / 97.8% (IN).
   If the differences are top-10 boundary ties -> harmless-ish (clean room may still differ slightly). If real ->
   find the code drift before Sunday.

## Open items
- **CE (Retrain B)**: not in Drive yet (`ce_outputs/ankit`, `ce_outputs/soup`). When there: verify md5s, then
  `tools/queue.sh add retrain_b "bash $PWD/tools/ingest_ce.sh ce_v1 <ankit_dir> <soup_dir>"`. It keys scores by ids
  into `code/business_entity_resolution/ce_scores/`, trains `lgbm2_ce` with extras, writes `output/candidates/ce_v1/`.
  Report OOF overall / per country / singleton, loss buckets, per-country test stats; recommend upload or not vs 0.977.
  To adopt: config `round2.name: lgbm2_ce`, `use_ce: true`, `decode.model: lgbm2_ce`; ship `ce_scores/` in the zip.
- Pseudo-labelling: DROPPED (sim -0.007 / -0.009). Unseen calibration correction: NOT adopted (+0.001). Blocking
  options: NOT built (best ceiling +0.0022 for 3x pairs). Nothing of these needs disclosure.
- Doc (`student_resource/Documentation_template.md`) not filled yet: disclose dictionaries (`ber/dictionaries.py`),
  indic-transliteration (MIT, installed without extras), sparse_dot_topn (Apache-2.0), xlm-roberta-base (MIT) if CE used.

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
