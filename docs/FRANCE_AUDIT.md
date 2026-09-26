# France (unseen country) audit — Sat 26 Sep, ~21:30 IST

Goal: find what makes our pipeline and the blend weaker on a country absent from train, and fix what can be fixed
safely before 09:00. All evidence below is label-free on test (plus train references). Country is only an opaque
grouping key in every proposed fix; no country name in code or rules.

## Evidence

| id | measurement | France (test) | US | India | train reference |
|---|---|---|---|---|---|
| M2 | S1 names containing a word that is generic within the country but missed by the current split-level rule | **27.0%** (131 of 201 in-country generics missed: gestion, residence, danse, patrimoine, musique, communale, ...) | 25.1% (test) | 4.3% (test) | US 14.9%, India 4.0% |
| M4 | records whose best pair p is uncertain (0.05-0.9) | **6.4%** | 2.1% | 1.4% | - |
| M4 | records with best p < 0.01 vs expected distractor share (1 - 3.46 x S1 / records) | 33.0% vs 37.4% | 37.8% vs 39.9% | 41.1% vs 40.6% | - |
| M3 | blend_raw empty-S1 share | 5.25% | 5.75% | 5.80% | singleton share 5.58% |
| M3 | blend_raw assigned records / expected (3.46 true links per S1) | **0.963** | 0.979 | 0.973 | - |
| M1 | decoded links whose record misses the S1's last address part (admin level) | **61.0%** | 30.9% | 48.2% | true pairs: US 32.0%, India 49.2% |
| M1 | mean address token-set similarity of decoded links | 84.3 | 85.5 | 88.7 | true pairs: US 83.9, India 87.1 |

Code reading (fit scopes):
- Blocking TF-IDF: fitted per country (good).
- `names2` (round-2 name features): generic set = tokens in > 0.1% of **all S1 of the split** (countries mixed); IDF of
  `v2_idf_jacc` also over the whole split. Train is 60% US / 40% India; test is 38% US / 47% India / **15% France**.
  So the same country's names get different features in train and test (US 15% -> 25% of names affected), and most
  French generic words look distinctive. This is a train/test feature skew, not only a France issue.
- Round-1 `idf_mats` (IDF-Jaccard name/address) and frequency shares: same split-level scope (same mechanism, smaller
  lever; fixing it needs a full feature + round-1 rebuild).
- Normalization/legal/abbreviations: French legal forms and street types are in the disclosed dictionaries; no French
  admin (region/department) handling, but no measured address-similarity deficit for France links.

## Ranked moves

| rank | move | mechanism | evidence and size | expected gain | LOCO test | build | clean room / CE |
|---|---|---|---|---|---|---|---|
| 1 | **names2 generic set + IDF per country group** (`names2.by_group`) | a country's generic words and IDF weights must not depend on which countries share its split | M2: 27% of France names and 25% of US test names (vs 15% US in train) get skewed rare-part features; M4: France uncertain band 3-4.5x the others | France recall/precision on generic-word variation; also removes US train/test skew. Guess +0.0005 to +0.002 public (unmeasurable before upload) | harness now derives the held-out country's generics from its own records, as test does; LOCO train shares (40/60%) are milder than France's 15%, so LOCO understates the gain | ~1.5 h (features 15 min, round 2 40 min, decode, LOCO 2 x 20 min, re-blend 15 min) | config `names2.by_group: true`; candidate set and hard rows fixed, CE scores valid; changes US/India rows too (public delta not France-only) |
| 2 | **France decode rule chosen by LOCO**: none / empty-rate k / link-count k / two knobs (entity gate from the empty signal, link odds from the link-count signal) | the two label-free signals disagree on France (M3: fewer empties AND fewer links than seen countries); one knob cannot satisfy both | M3 | picks the rule that helps an unseen country with labels in LOCO; informs the A/B uploads | direct: held-out country's labels score each label-free rule | decode-only, ~30 min | config knobs; no feature change |
| 3 | round-1 IDF / frequency shares per country group | same skew at round 1 | same mechanism | unknown | same limits as 1 | 3-4 h (features + round 1 + round 2) | too heavy before the 11:00 freeze: **not tonight** |
| 4 | admin-level address parts (learned per country) | France records drop/replace the S1's admin part (region vs department) | M1: 61% vs 31-49%, but no address-similarity deficit on France links | unclear | possible | 2-3 h | **not tonight** (gap real, mechanism unclear) |
| 5 | France-specific blend weight | partner pipeline learns admin vocabulary | none label-free | unknown | no | - | skip |

Gates for any built move: US/India OOF change >= -0.0002; LOCO unseen side improves; France diagnostics move as the
mechanism predicts; validator PASS + --check-ids. Candidate set fixed (CE scores stay valid).
