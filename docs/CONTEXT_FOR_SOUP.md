# Context for Soup: the whole project on one page (Sun 27 Sep, ~22:30 IST)

Read this first, then `student_resource/Documentation_template.md` (the write-up you are editing). Everything below
is true as of commit `5a2d191` on `main`.

---

## 0. TL;DR

- **Our final = "FINAL 2", public 0.986758** (best and last upload). md5 `50c695de7ae6c03d19a70ee75040f983`.
- It is built in three layers:
  1. **our pipeline** (TF-IDF blocking, LightGBM x2, xlm-roberta cross-encoder, expected-F0.5 decoder);
  2. **a pair-level blend** with Razan's independent pipeline (QTpies **M10**, and **M11h** for US/India);
  3. **Razan's `fuse.py`**, a submission-level post-process with per-country thresholds tuned on public probes.
- **Zip ready:** `QTpies_submission.zip` at the repo root = tag `freeze-2026-09-27f`, md5
  `47d9faa6af680c6e5aebc20549a9ec7e`. Its Documentation is the current `Documentation_template.md`.
- **Your job:** make the Documentation clear, correct and complete. Code and numbers are frozen. **If you change the
  Documentation, the zip must be rebuilt from a new tag** (section 9). Keep every disclosure (section 7).

---

## 1. Where things are

| path | what |
|---|---|
| `student_resource/Documentation_template.md` | **the write-up (you edit this)**; it ships in the zip root |
| `student_resource/code/business_entity_resolution/` | the zip's `code/business_entity_resolution/`: `src/`, `README.md` (how to reproduce), `requirements*.txt` |
| `.../src/ber/` | our pipeline (Python package), see section 2 |
| `.../src/configs/default.yaml` | every knob (K, thresholds, model names, blend weight, France k) |
| `.../src/run_all.sh` | single entry point: all stages in order |
| `.../src/gpu/ce_train_predict.py` | our cross-encoder training/scoring script (Kaggle GPU, you and Ankit ran it) |
| `.../src/ce_scores/` | shipped cross-encoder scores (inputs, gitignored parquet) |
| `.../src/razan/` | Razan's pipeline: shipped predictions (inputs), his code (`src/`, `code/`), REPRODUCE files, fuse inputs (`fuse/`) |
| `.../src/tests/` | metric, IO and decoder tests (12 pass) |
| `EXPERIMENTS.md` | **the experiment log: every number in the doc comes from here** (one row per run, dated) |
| `HANDOFF.md`, `HANDOFF_RAZAN.md` | state snapshots (HANDOFF_RAZAN = the current final, all scored files) |
| `docs/MORNING_BRIEF.md` | live log of Sunday (newest first at the top) |
| `docs/NIGHT2.md` | per-country evidence for the final + why no later candidate was uploaded |
| `docs/M8_DISCLOSURE.md` | rules audit of Razan's code: country keys, test-fitted statistics (source of Documentation 4.5) |
| `docs/HAILMARY.md`, `docs/LOCO_ERRORS.md`, `docs/FRANCE_AUDIT.md` | unseen-country studies (sources of 5.3 / 5.5) |
| `docs/AUDIT_razan.md` | Razan's independent audit of our pipeline |
| `tools/` | `make_zip.sh` (builds the zip), `cleanroom.sh`, `queue.sh`, `backup.sh`, ... |
| `TEAM.md` | original team split and shared file contracts |

The dataset, `work/` caches, `output/` and all `.parquet` / `.tsv` files are gitignored, so they are not on GitHub.
The big files are listed with md5s in `~/Downloads/DRIVE_LIST_for_razan.txt` on Muaz's laptop.

---

## 2. Our pipeline, stage by stage (`run_all.sh` -> `python -m ber.cli <stage>`)

| # | stage | module | what it does | key numbers |
|---|---|---|---|---|
| 1 | `normalize` | `ber/normalize.py`, `ber/dictionaries.py` | Unicode folding, Indic scripts -> Latin (`indic-transliteration`), state names -> codes, address short forms, zero-stripping, DBA split, landmark split, sound-alike skeleton | all hand-written maps are in `dictionaries.py` (disclosed in doc 4.3) |
| 2 | `folds` | `ber/folds.py` | 5 folds grouped by S1 (seed 42) + 20% training sample flag | |
| 3 | `block` | `ber/blocking/tfidf.py` | two sparse TF-IDF cosine channels **within each country**: `fw` (all tokens) and `hh` (half name / half address), top-10 S1 per record each | 153.46M train pairs (69.5 per S1), **PC 0.9825** (US 0.9895, India 0.9720) |
| 4 | `blockreport` | `ber/blocking/report.py` | pair completeness, reduction ratio | ceiling F0.5 0.99407 |
| 5 | `features` | `ber/features/strings.py`, `context.py` | 42 pair features: rapidfuzz name/address similarities, IDF-Jaccard, postal/house number, legal form, channel scores/ranks, context margins, frequency shares | |
| 6 | `train` / `predict` | `ber/models/lgbm.py` | LightGBM round 1 (5 folds, OOF on every train pair; test = fold-0 model; 16 threads fixed) | |
| 7 | `graph`, `r2feats` | `ber/features/graph.py`, `numbers.py`, `names2.py` | round-2 features on hard pairs (round-1 p >= 0.01): 13 group/graph features, 15 soft house-number features, 8 v2 name features | 9.84M train hard pairs |
| 8 | `ce` | `ber/ce_ingest.py` | joins the shipped cross-encoder score `ce_p` (xlm-roberta-base, trained in two halves on Kaggle) | CE covers 93% of round-2 rows |
| 9 | `stack` | `ber/models/stack.py` | LightGBM round 2 (3000 rounds, 5 folds) | **OOF 0.98765** (US 0.98971, India 0.98456) |
| 10 | `decode` | `ber/decode.py` | owner rule (each record on its best S1) -> isotonic calibration -> "any match" gate -> exact expected-F0.5 decoder | no hand-set threshold |
| 11 | `report`, `write` | `ber/analysis.py`, `ber/decode.py` | loss buckets; writes both TSVs + runs the official validator | our solo file public 0.980518 |
| 12 | `blend` | `ber/blend.py`, `ber/calibrate.py` | pair-level blend with Razan's p (section 3), label-free France calibration (odds factor k by empty-rate matching) | |

Other modules:
- `loco.py`, `unseen.py`, `france.py`: leave-one-country-out studies.
- `synth.py`: synthetic pairs, tried and dropped. `twins.py`: twin-decoy analysis.
- `night.py`: scripted overnight runs. `eda.py`: phase-0 facts. `metric.py`: the exact macro F0.5.
- `io.py`: paths, TSV IO, validator wrapper.

---

## 3. Razan's pipeline (QTpies) and the blend

Model lineage (all his; code in `src/razan/src/` + `src/razan/code/`, recipes in `src/razan/REPRODUCE*.md`):
- **M5**: char-4-gram + word TF-IDF blocking, 50 features, two-stage LightGBM, two pairwise cross-encoders
  (multilingual-e5-small, MIT, 118M). OOF 0.98820 (his "DROP16" world: 16% of train S1 dropped to match test density).
- **M8** = M5 + **twin-world training** (synthetic decoys from train rows) + a **cluster cross-encoder** (one S1 + all
  its candidates in one sequence). OOF 0.98849.
- **M9** = M8 + a second cluster cross-encoder on multilingual-e5-base (MIT, 278M).
- **M10** = M9 + stage 3: a LightGBM recalibration with label-free "same-name rival row" counts. OOF 0.98898.
- **M11h / M12h** = stage-3 stackers that also take **our** clean-room p as features (via `code/export_for_razan.py`).
  US/India use the stacker, France uses M10. OOF (our decoder) M11 0.98956, M12 0.98963.

The blend (`ber/blend.py`):
- Candidates = union of both pipelines (+634,542 test pairs only his blocking found).
- p = w x ours + (1 - w) x his. **`blend.w` in the config = OUR weight**; "M10 weight" in chat = HIS weight = 1 - w.
- A pair scored by only one side keeps its p ("asis").
- Decoder fitted on his 1,853,766 "live" train S1; France k by empty-rate matching.

| blend | OOF (live S1) | France k | public | md5 |
|---|---|---|---|---|
| M5, w 0.5 | 0.99032 | 0.18386 | 0.985131 | 0a0b1480 (production) / 9c36a54b (clean room) |
| M10, w 0.3 (M10 weight 0.7) | 0.99070 | 0.432 | 0.986456 | d754f06f |
| M11h, w 0.0 | 0.99093 | 0.353 | (not uploaded alone) | e7a6eda3 |
| M12h, w 0.0 | 0.99100 | 0.353 | evaluated, not used | bb1b4474 |

---

## 4. The final file (FINAL 2) and the fuse

`fuse.py` runs on top of a blend file:
- per country, it vetoes team links that Razan's stage-2 p rejects, and adds his confident pairs on unowned records;
- solo rules use our single-pipeline file as a third opinion;
- twin-signature vetoes; France fills for S1 left empty.

The thresholds come from about 10 public probes (`src/razan/LB_LOG_2026-09-27.md`).

| file | recipe | public | md5 |
|---|---|---|---|
| #33 | fuse on M10 blend, US/India rules read M10 p | 0.986655 | c52d413c |
| #34 | fuse on M10 blend, US/India rules read M12h p | 0.986731 | c51d3853 |
| #35 | fuse on M11h blend, US/India rules read M11h p | 0.986666 | 10f5696b |
| **FINAL 2** | **France rows of #34 + US/India rows of #35** (`merge_by_country.py`) | **0.986758** | **50c695de** |

- Exact commands: code README section D.
- All four were re-run on Muaz's laptop with matching md5s, and FINAL 2 also regenerates from the zip's own contents.
- **SOLO caveat:** the fuse uses our reference-run solo file (`3a0d4c77`). The clean room rebuilds it as `677f56f9`,
  which changes 174-176 fused rows (0.01%). So the zip ships the reference file as a fixed input, and that is disclosed.

Full public history, for the progress table: fallback 0.965 -> numbers/v2 features 0.977 -> CE half 0.981 -> CE full
0.980518 -> M5 blend 0.985131 (France k=1 A/B: 0.98488) -> M10 blend 0.986456 -> #33 0.986655 -> #34 0.986731 ->
FINAL 2 0.986758. All 25 scored files with md5s are in `HANDOFF_RAZAN.md`.

---

## 5. Numbers cheat sheet (all traceable in EXPERIMENTS.md)

- Train: 2,206,821 S1 (US 60% / India 40%). Test: 1,732,544 S1 (US 38% / India 47% / **France 15%, unseen in train**).
- Singletons 5.6% of train S1; mean 3.46 true matches per S1; one-owner rule exact; no cross-country pairs.
- Blocking: PC 0.9825, 153.46M train pairs, reduction ratio 0.9999933, ceiling 0.99407.
- Our OOF: v0 0.8891 -> decode v1 0.8932 -> two-channel blocking 0.9305 -> 42 features 0.9715 -> round 2 0.9773 ->
  numbers + v2 names 0.98233 -> + CE 0.98765.
- **Cross-encoder compute** (Kaggle logs): 2x Tesla T4, torch 2.10.0+cu128. CE1 half 1 5.61 h, half 2 5.80 h. CE2
  half 1 5.65 h; CE2 half 2 about 12 h (team-reported). CE2 was not used.
- LOCO (India unseen / US unseen): 0.91571 / 0.96935 with num+names2 features (+0.0099 / +0.0146).
- **Clean room** (fresh clone + fresh env, 27 Sep): PASS in 320 min.
  - Our OOF 0.98766 (ref 0.98765), blend OOF 0.99032 (= ref), candidates byte-identical.
  - 3,451 rows (0.2%) differ from the reference upload.
  - Peak memory 10.4 GB RAM + ~7-8 GB swap at blocking.
  - The first attempt was killed by our 11 GB cap.

---

## 6. What we tried and dropped (with numbers; the doc's 5.5 table already has most)

| idea | result |
|---|---|
| per-country name priors (names2c) | LOCO -0.0007 / -0.0003 |
| second cross-encoder (e5-base) | +0.00015 OOF (gate +0.0003) |
| drop format features | LOCO -0.0052 / -0.0028 |
| hail mary H1 (transfer model) | all mixes worse in both directions (down to -0.0106) |
| hail mary H2 (monotone constraints) | fails |
| pseudo-labels | -0.0072 / -0.0094 |
| synthetic unseen-country pairs | -0.0021 / -0.0044 |
| quantile features | -0.0014 |
| bigger blocking K | ceiling at most +0.0022 for 2-3x the pairs |
| twin decode rule | costs OOF |
| seen-country calibration | fails its null test |
| seed bagging | dropped for time |
| M8 as partner | 0.99044, superseded by M10 |
| M12h blend | +0.00007, below the gate |
| C1 (France at our weight 0.5) | predicted +0.00002 |
| label-free proxies for public score | do not predict |
| Razan's LLM judge (Qwen2.5-7B, local) | AUC 0.511, not used; no API anywhere |

---

## 7. Disclosures that MUST stay in the Documentation

1. **Test-set statistics (label-free):**
   - our France k (empty-rate matching);
   - per-split generic words / IDF;
   - Razan's DROP16 density match;
   - his twin parameters measured on US test (`DBIN` histogram, 89% positive shifts, 94% suffixes, 30% pairs).
2. **Country-keyed settings:** Razan's twin rates (US 0.072 / 0.03, India 0.03) and the fuse's per-country thresholds.
   Also the inactive "France" branch in `calibrate.py:87` and the CV report weights (TEST_MIX). Country is otherwise
   an opaque key and never a model feature; France goes through the same pipeline.
3. **Leaderboard use:** fuse thresholds, the hybrid rule and the FINAL 2 per-country merge were chosen from about 10
   public probes; the France k A/B (0.98488 vs 0.985131).
4. **SOLO caveat** (section 4) and **non-bit-exact GPU steps** (about 1e-4 OOF noise on a full rerun).
5. **Hand-written dictionaries** (doc 4.3).
6. **Licenses and sizes:** xlm-roberta-base (MIT, 278M), multilingual-e5-small (MIT, 118M), multilingual-e5-base
   (MIT, 278M), LightGBM / polars / rapidfuzz (MIT), sparse_dot_topn (Apache-2.0), indic-transliteration (MIT). All
   are far below the 8B limit, and none use external data or APIs.

Sources: `docs/M8_DISCLOSURE.md` (full text + draft paragraph), doc sections 4.5 / 4.6.

---

## 8. The Documentation right now: status and suggested to-dos

The template's required parts are the four points (methodology, blocking, model and features, other). The current
file (~500 lines) covers all of them; every `[FINAL]` / `[TO CONFIRM]` slot is filled.

Suggested work:
- **Section order:** 4.6 (fuse) currently sits before 4.5 (disclosure). Swap them, or renumber.
- **Executive summary:** make the three-layer story (our pipeline -> blend -> fuse) crisp in about 6 lines; keep
  0.986758 and the reproducibility claim.
- **2.2 Solution strategy / pipeline list:** add the blend and fuse steps. It still lists only our 9 stages.
- **4.2 Models:** the blend paragraph grew in pieces (M5, M8, M9/M10 bullets). A small lineage table would read better.
- **5.2 progress table:** check that the rows match section 4 of this page. The public column is right as of now.
- **Consistency:** the same number must appear identically everywhere (e.g. 0.99070, 0.986731, 3,451 rows, 320 min).
  Search the file for each one.
- **Length/clarity:** plain sentences, no internal jargon (explain "live S1", "asis", "DROP16", "OOF" once).
- **Do not change** md5s, scores or disclosures without checking the source (EXPERIMENTS.md / HANDOFF_RAZAN.md).

---

## 9. Rules and gotchas when you edit

- Work on a branch (e.g. `soup/doc`), open a PR, and Muaz merges.
- Commit messages are short and caveman-style (see `git log`). **No AI attribution lines in commits.** No secrets.
- **Changing the Documentation after tag 27f means the zip is stale.** After merge, Muaz or Claude runs:
  ```bash
  git tag -a freeze-2026-09-27g -m "DOC UPDATE"   # new tag on the merged commit
  git push origin main freeze-2026-09-27g
  REF=freeze-2026-09-27g tools/make_zip.sh QTpies ../MLC-Zon-fuse-staged/out_final2/matching_results.tsv
  ```
  Then verify: the zip entries equal the 27f zip except `Documentation_template.md`, `matching_results.tsv` md5 is
  `50c695de…`, and the validator passes. That must run on Muaz's laptop, where the data lives.
- Naming traps:
  - "w" in chat usually means Razan's weight; `blend.w` in the config means ours.
  - "M11h / M12h" = the stacker for US/India, M10 for France.
- Never touch `output/matching_results.tsv` (the protected fallback) or the verified zips.

## 10. Team

Muhammed Razan (leader, QTpies pipeline + fuse), Muaz Ismail Mohammed (our pipeline, integration, packaging), Ankit
Ranjan and Souparno Chakraborty (cross-encoders on Kaggle GPUs; Documentation).
