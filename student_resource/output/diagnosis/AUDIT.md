# Mega audit (2026-09-25 ~19:00 IST)

Scope: the fallback `output/matching_results.tsv` (md5 e7dbc835…, OOF 0.9773) and the code that produced it.

## Summary

| # | Item | Risk | Status / fix |
|---|---|---|---|
| A1 | OOF protocol / fold leakage | LOW | Folds are grouped by S1. TF-IDF, IDF, and frequency use no labels. Group features use OOF scores. A leak test found no sign of a leak. |
| A2 | Calibration and threshold tuning out-of-fold | LOW | Isotonic calibration and the gate are cross-fitted by fold. The chosen decoder (expected-F + gate) has no tuned threshold. Picking 1 of 4 decoders on OOF adds a small optimism (≤0.0007). |
| A3 | Plateau width | LOW | gate+tau2 is flat over the whole grid (tau_g 0.35–0.85, tau_2 0.50–0.91). Fold std of the score is 0.00008–0.00012. |
| A4 | France (15% of test, no labels) | **HIGH** | Not tested yet. Test France is more confident than train: its 5th-percentile top score is 0.45, vs about 0.10–0.14 for US and India, and its empty share is 5.0%, vs 5.7–5.9%. This is either real, or over-matching on French chain names. Fix: leave-one-country-out run (plan below, about 3 h). |
| A5 | US/India test drift | LOW | Test stats match OOF and train truth (table below). |
| B1 | Model licenses and size | LOW | The only model family is LightGBM (MIT): gradient-boosted trees, about 0.4M leaf/split values in total, far below 8B. Libraries are MIT, BSD, Apache, or other permissive licenses (list below). There are no neural models. |
| B2 | Network calls in the pipeline | LOW | Code search found none. The `curl` in `tools/aws_sync.sh` is dev tooling, not the pipeline, and not shipped. |
| B3 | External data / dictionaries | **MEDIUM** | There is no external data. The hand-written maps in `ber/dictionaries.py` (LEGAL, NAME_STOP, CHILLU, CANDRA, DBA, PLACE_PHRASES, ADDR_TOKENS, ADDR_FR, ADDR_STOP, SKEL) and the `indic-transliteration` library tables are **not yet disclosed**, because the doc is not written. Fix: disclose them in Documentation_template.md (Phase 10). |
| B4 | Country as a model feature | LOW | It is not in any model (checked in the model files). Country is used only to block within a country, for the France `r → rue` map, and for reports. |
| C1 | candidate_pairs is exactly the model's input | LOW | 148,566,883 IDs in the file = cands rows = final score rows. Round 1 scores all of them. Round 2 re-scores the 8.78M pairs with round-1 p ≥ 0.01, and the rest keep the round-1 score. This must be written in the doc. |
| C2 | Matches ⊆ candidates | LOW | The official validator ran on both files with no subset warning. |
| C3 | Reduction ratio / PC | LOW | Train: PC 0.9825 (US 0.9895, India 0.9720), RR 0.99999326. Test: RR 0.99999140. |
| C4 | 5 empty candidate rows | LOW | These are S1 that no S2/S3 record ranked in its top 10 (blocking searches from records to S1). Example: `S1-501366513 FUO Union SAS, 86 Rue des Maures…`. These S1 are predicted empty, which is correct if they are singletons. |
| D1 | Zip structure | LOW | `code/business_entity_resolution/{src, README.md, requirements.txt}` exists. `make_zip.sh` and the filled doc are still to do (Phase 10). |
| D2 | Pinned requirements | LOW | Every package is pinned with `==`. Python 3.11. |
| D3 | README + run_all rebuild the files | **HIGH → FIXED** | Before: the fallback came from ad-hoc commands (round 2 with 3000 rounds), so `run_all.sh` would have produced the 1000-round file instead. Fixed: the config now sets `round2.name: lgbm1_r3000`, `num_boost_round: 3000`, and `decode.model: lgbm1_r3000`, and the README lists every stage. Estimated full run: about 5.2 h on this laptop. |
| D4 | Determinism | **HIGH → FIXED** | Re-running normalization gives a byte-identical file. LightGBM trained twice gives the same model. **But 16 vs 8 threads gives a different model.** The config used "all cores," so a judge's machine would not reproduce our md5. Fixed: `n_jobs: 16` is now pinned (the fallback was built with 16 threads, so it stays valid). A full clean-room re-run is still to do (Phase 10). |
| E1 | Disk | MEDIUM | 20 GB free. The blocking fix needs about 10 GB of temporary space. |
| E2 | Running shells | LOW | None of our shells are running now (stale ones were killed earlier). The remaining processes belong to the desktop (Hyprland, clipboard, terminal launchers), so nothing was killed. |
| E3 | Uncommitted changes | LOW → FIXED | Committed with this audit. |
| E4 | Fallback backed up twice | LOW → FIXED | `output/matching_results.tsv` + `work/fallback/matching_results.tsv` (same md5), plus Razan's Drive copy. |

## A5: per-country stats

| country | train truth singleton | train truth G per non-empty | OOF predicted empty | OOF k per non-empty | test predicted empty | test k per non-empty |
|---|---|---|---|---|---|---|
| US | 0.0558 | 3.66 | 0.0585 | 3.53 | 0.0569 | 3.58 |
| India | 0.0559 | 3.67 | 0.0598 | 3.49 | 0.0586 | 3.53 |
| France | – | – | – | – | 0.0504 | 3.59 |

Top-1 score per S1 after the owner rule (round 2 with 1000 rounds, raw p): 5th / 10th percentile.
- OOF: US 0.099 / 0.9988, India 0.078 / 0.9976.
- Test: US 0.137 / 0.9988, India 0.116 / 0.9977, **France 0.446 / 0.9993**.

## A4: leave-one-country-out plan (not run yet)

1. Round 1 on US only (5 folds inside US, OOF), then predict India. About 60 min. It reuses the existing features, with no recomputation.
2. Group features, then round 2 on US only, then predict India. About 25 min.
3. Decode, with calibration and gate fitted on US only, then score India. About 5 min.
4. The reverse direction (India → US): about 90 min more.

ETA about 3 h on the laptop, CPU-bound. It is queued after the blocking fix, or it can run overnight.

## B1: libraries and licenses (from package metadata)

- MIT: lightgbm 4.7.0, rapidfuzz 3.14.6, pyyaml 6.0.3, indic-transliteration 2.3.82, typer, toml, six, rich, narwhals, pluggy, iniconfig, markdown-it-py, mdurl, backports-functools-lru-cache.
- Apache-2.0: pyarrow 25.0.1, sparse-dot-topn 1.2.0, regex (Apache-2.0 AND CNRI-Python), packaging (Apache-2.0 OR BSD-2).
- BSD: pandas 3.0.6, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, joblib, threadpoolctl, psutil, cloudpickle, pygments.
- Other permissive: tqdm (MPL-2.0 AND MIT), roman (ZPL-2.1, a transitive dependency of indic-transliteration), shellingham (ISC), python-dateutil (dual BSD/Apache).

Model sizes:

| model | trees | leaves | features |
|---|---|---|---|
| round-1 fold model | 998 | 62,874 | 42 |
| round-2 (r3000) | 3,000 | 189,000 | 57 |
| gate | 183 | 5,673 | 8 |
