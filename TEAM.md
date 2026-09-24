# Team split — Amazon ML Challenge 2026 (Business Entity Resolution)

Code root: `student_resource/code/business_entity_resolution/src/` (run every command from there).
Env: `uv venv --python 3.11 .venv && uv pip install -r student_resource/code/business_entity_resolution/requirements.txt` at repo root; then `PY=../../../../.venv/bin/python`.
Caches: `student_resource/work/` (gitignored). Every stage: `$PY -m ber.cli <stage> --split {train,test} [--force]`.
Work on a branch per owner (`razan/blocking`, `soup/features`, `ankit/gpu`); Muaz merges to `main`.
Every run you want kept -> one line in `EXPERIMENTS.md` (CV from `decode`, PC from `blockreport`).

## Shared contracts (do not change a schema without telling Muaz)

IDs are stored as **int64 keys** in every parquet: `S2-123` -> `2*10**10 + 123` (`ber.io.to_key` / `to_id`). This keeps 100M-pair tables inside 16 GB RAM.

| File | Columns |
|---|---|
| `work/{split}/norm_s{1,2,3}.parquet` | `entity_id, source, country, name_raw, name_norm, name_core, legal_class, dba_parts (list), name_skel, addr_raw, addr_norm, addr_nolandmark, landmark, numbers (list), postal` |
| `work/folds.parquet` | `s1_id, fold (0-4), sampled (bool, 20% S1 used for LightGBM fit)` — train only, seed 42 |
| `work/{split}/cands.parquet` | `s1, cand, src (2/3), <channel>_score, <channel>_rank, n_channels, label (train)` |
| `work/{split}/feats.parquet` | `s1, cand` + feature columns, **row-aligned with cands.parquet** |
| `work/preds/{oof,test}_<model>.parquet` | `s1, cand, p`, row-aligned with feats |

Rules for everyone: no `country` feature; missing = NaN, never 0; no class rebalancing; train on all blocked pairs; no network calls in pipeline code; every new dictionary goes in `ber/dictionaries.py`.

## Muaz — integration (`io, folds, metric, decode, analysis, cli`, packaging)
- In: everything above. Out: `student_resource/output/{matching_results,candidate_pairs}.tsv`, zip, README, doc.
- Done when: validator PASS (matching whole + candidate shards), loss report per run, gate + tau_2 decoder beats v0 on OOF, clean-room run passes.
- Deadlines: v0 Fri 04:00; decoder v1 Fri 16:00; clean-room Sun 14:00; final upload Sun 20:00; zip Sun 21:00. Only Muaz uploads.
- Test: `$PY -m pytest -q tests && $PY -m ber.cli decode && $PY -m ber.cli report`

## Razan — normalization + blocking (`normalize, dictionaries, blocking/*`)
- In: `dataset/*/`. Out: `norm_s*.parquet`, `cands.parquet`, `work/reports/blocking_*.md`.
- Deliverables: normalization v1 (US/India/France maps, DBA split, landmarks, number rules, `name_skel`), channels 2-7 from the plan, reverse pass, `candidate_pairs.tsv` correctness, France pseudo-label simulation.
- Done when: PC >= 99% overall and per country (train), mean/p95 cands per S1 reported, each channel's drop-one PC delta reported, test blocking < 30 min.
- Deadlines: normalization v1 + 3 channels Fri 12:00; dense channel merged Sat 00:00.
- Test: `$PY -m ber.cli normalize --split train --force && $PY -m ber.cli block --split train --force && $PY -m ber.cli blockreport --split train`
  (quick timing/PC probe: `$PY -m ber.cli block --split train --limit 50000`)

## Soup — features + models (`features/*, models/lgbm, loco`)
- In: `norm_s*`, `cands`, `folds`. Out: `feats.parquet`, `preds/*`, `work/reports/importance_*.tsv`.
- Deliverables: full Phase 3 feature set (name, address, context, frequency, source), LightGBM round 1 + round 2 (graph/stacked features from OOF p), leave-one-country-out harness + drop-one-group ablation.
- Done when: OOF macro F0.5 beats v0, top-25 gain importance reported, LOCO US->IN and IN->US reported.
- Deadlines: features v1 Fri 14:00; LOCO Sat 00:00; round 2 Sat 22:00.
- Test: `$PY -m ber.cli features --split train --force && $PY -m ber.cli train --force && $PY -m ber.cli decode`

## Ankit — GPU (`gpu/*`)
- In: `work/*/norm_*.parquet` + `folds.parquet` packaged as a Kaggle dataset. Out: `emb_*.npy` + `_ids.parquet`, `ce_p` parquet (s1, cand, ce_p), fine-tuned retriever embeddings.
- Deliverables: `embed.py` (multilingual-e5-base, MIT), `retriever_ft.py`, `cross_encoder.py` (xlm-roberta-base, MIT, 3 folds mapped from the 5), optional Qwen2.5-7B judge only on Muaz's go. Kaggle quota plan.
- Done when: each script runs standalone on Kaggle T4 and locally with fallback paths; runtime estimate from a 2% subsample reported before full runs; dense channel PC gain reported.
- Deadlines: embeddings Fri 20:00; cross-encoder OOF Sat 12:00.
- Test: `$PY gpu/embed.py --limit 10000` (local, 4 GB VRAM RTX 3050: use fp16, batch 64)
