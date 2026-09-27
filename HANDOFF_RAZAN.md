# Handoff to Razan (Sun 27 Sep, ~21:30 IST)

## Current final: FINAL 2, public 0.986758 (our best AND last upload)
- **md5 `50c695de7ae6c03d19a70ee75040f983`.** It passes the validator with `--check-ids`, and its matches are a
  subset of the union candidates (`5021548f879c1f8ad6fbf157e0462778`).
- **Recipe:** France rows of fuse #34 + US/India rows of fuse #35 (`merge_by_country.py`, `--a-countries France`).
  - #34 = `fuse.py` on the M10 blend `d754f06f` (`blend.w` 0.3), US/India rules reading M12h p.
  - #35 = the same fuse on the M11h blend `e7a6eda3` (`blend.w` 0.0), US/India rules reading M11h p.
  - France rules read M9 p in both.
  - SOLO = production `ce_full_v1` (`3a0d4c77`, a fixed input).
- **Regenerate from the verified 27e zip** (`QTpies_submission.zip`, md5 `0dd30abc9c6137e53f0a013a9860afdf`): unzip,
  then follow the code README section D (bash, polars 1.44.2). It gives #34 `c51d3853…`, #35 `10f5696b…` and FINAL 2
  `50c695de…`, all verified from the zip's own contents on 27 Sep. The clean-room SOLO (`677f56f9`) instead changes
  174-176 rows (0.01%).

## Every scored file

| file | public | md5 |
|---|---|---|
| fallback lgbm1_r3000 (tag lb-0965) | 0.965 | e7dbc8357c3d2035228854ddf8497556 |
| numbers_france_v1 | 0.977 | cd673b353beb3e431f0a23da1ce3bd40 |
| ce_full_v1 (our solo) | 0.980518 | 3a0d4c771a1203acf207811ee2c8a2fe |
| ce_half_v1 | 0.981 | 7083a967b1c2bd32abbd39f00bad72f5 |
| blend_raw (M5 blend, France k 1) | 0.98488 | da71932fa9a3fb048715ae267d9a23e3 |
| blend_cal (M5 blend, France k 0.18386) | 0.985131 | 0a0b14806bcddf3370258467ae3fddd9 |
| sub19 fuse US/IN veto .3 + add .99 | 0.985391 | fe997cdaacd1d54795c3835aefce9718 |
| sub20 fuse FR veto .1 | 0.985441 | 9092791a2ab7adfc2e4b48829a5f6442 |
| sub21 fuse FR add .99 | 0.985243 | 0d870c45d2d980275f49b81230f9f540 |
| sub22 C0 + FR veto .1-.2 | 0.985955 | fe48ed804c2f972d32f314e1ca3208bb |
| sub23 C0 + FR veto .1-.3 | 0.986057 | 8632c93b074b260ca3ab1a520408d082 |
| sub24 C0 + US/IN veto .3-.5 | 0.985793 | 16f2f583741a9509735dbf0580366331 |
| sub25 C1 + FR veto .3-.5 | 0.986204 | 2b5d9fcf93c8f7616fe5271ab513b174 |
| sub26 C1 + FR veto .3-.7 | 0.986282 | f511fba3359fa96605a0b60c6263182b |
| sub27 C1 + FR veto .3-.9 | 0.986174 | fd2200ada08b0d6e0b03d36f509b600e |
| sub28 C1 + FR add .95-.99 | 0.986082 | 62237ff9bf83e328602f490e3cb0823b |
| sub29 C2 + solo vetoes | 0.986358 | 4c61ed3ff4880537f4550da1df1b7f49 |
| sub30 C2 + solo vetoes + adds | 0.986404 | 16409c785f42e4f4b2780ddc6448a221 |
| sub31 + twin-signature vetoes | 0.986579 | 1d5fe7d603541cb83c0b7995aaf27596 |
| #32 fuse on M5 blend (production inputs) | 0.986611 | dbbc1482af916fdcf250f8425cbe1ab0 |
| M10 blend, M10 weight 0.7 (`blend.w` 0.3) | 0.986456 | d754f06f8800aa5516a2a67d7a1ab2e4 |
| #33 fuse on M10 blend, US/IN rules M10 p | 0.986655 | c52d413c7404e9be4c368aa49d1d4de0 |
| #34 fuse on M10 blend, US/IN rules M12h p | 0.986731 | c51d385383f2fca625aa8ff24eb6d573 |
| #35 fuse on M11h blend, US/IN rules M11h p | 0.986666 | 10f5696b16da21ba9634ad6040110b3f |
| **FINAL 2 = France #34 + US/IN #35** | **0.986758** | **50c695de7ae6c03d19a70ee75040f983** |

Reproduced on this laptop (md5 match): #33, #34, #35, FINAL 2, and the clean-room blends. #32 cannot be the final:
it needs the production M5 `blend_cal`, which the clean room rebuilds as `9c36a54b`, not `0a0b1480`.

## Per-country evidence (pairs of scored files identical outside one country; docs/NIGHT2.md)

| pair | country | public change |
|---|---|---|
| blend_raw (k 1) vs blend_cal (k 0.184) | France | -0.000251 for k 1 |
| #33 -> #34 | US/India | +0.000076 (M12h rules) |
| #34 -> FINAL 2 | US/India | +0.000027 (#35's US/India) |
| #35 -> FINAL 2 | France | +0.000092 (France with our weight 0.3 beats our weight 0) |

FINAL 2 already has the best known component in every country.

## Tried tonight and rejected
- **C1:** France from the fuse on the M10 blend at our weight 0.5, US/India = FINAL 2.
  - md5 `b6f2367e2f8bd613b120ac16462dff47`, in `../MLC-Zon-fuse-staged/n2_c1_final/`.
  - Only 2,335 France rows differ; predicted ~+0.00002, below the +0.0002 gate. **Not uploaded.**
- **M12h as blend partner:** blend OOF 0.99100 vs M11h 0.99093 (+0.00007, below our +0.0001 gate). Its p is used
  inside #34's US/India rules.
- **Label-free proxies:** empty share and links per S1 per country do not predict public changes. For France, fewer
  links won once (+0.00025) and lost once (-0.00009).
- **Earlier today:**
  - M8 blend 0.99044 OOF (superseded);
  - LLM judge (Qwen2.5-7B, local, AUC 0.511);
  - hail mary H1 transfer model / H2 monotone (LOCO worse in both directions);
  - names2c, CE2, the format-feature drop (all failed their gates).

## Zip status
- **Verified:** `QTpies_submission_27e_fuse34.zip` (renamed) = tag `freeze-2026-09-27e` (output #34), 2,581,827,078 bytes, md5
  `0dd30abc9c6137e53f0a013a9860afdf`. `QTpies_submission_w07_27c.zip` = tag 27c (output w0.7), md5
  `9c6798a9f1c929644fbbd11f58b1fe7e`.
- **Built and verified (21:25):** `QTpies_submission.zip` = tag `freeze-2026-09-27f`, output = FINAL 2, 2,581,824,811
  bytes, md5 `47d9faa6af680c6e5aebc20549a9ec7e`. It is CRC-identical to the verified 27e zip except
  `output/matching_results.tsv` and the Documentation. The 27e zip was renamed `QTpies_submission_27e_fuse34.zip`
  (unchanged, md5 `0dd30abc…`).

## Rules
(Muaz mentioned "rules below", but none came with the message; these are the standing ones.)
- Competition: no external data, APIs or lookups; models MIT/Apache and <= 8B parameters; every Source 1 entity in
  the output; matches only S2/S3 ids; the zip must regenerate both output files from its contents.
- Country is an open set: never hard-code or filter the pipeline to known countries. Country-keyed settings must be
  disclosed (the twin rates, the fuse thresholds).
- The final must be our best and our last upload. Every uploaded file must be reproducible from committed code +
  shipped inputs, with an md5 re-run.
- Anti-overfitting: no new thresholds or weights chosen from public scores; no probes of small variants; upload only
  with a predicted gain >= +0.0002 and evidence.
- Never modify the verified zips, `~/Downloads/final2/`, or the existing candidate folders.
