# Hail mary: unseen-country gap (Sun 27 Sep, 01:00-03:10 IST)

**Verdict: nothing passed. No build, no candidate, no config flag. The freeze uses the base blend at France k
0.18385635887853335 (blend_cal).**

Setup: LOCO harness (`ber.loco`), round 2 on the production feature set (graph + num + names2, no CE: the CE models saw
both train countries, so ce_p would leak the held-out country). Five S1 folds, trained on the source country only,
1000 rounds. Isotonic, gate and decoder are fitted on the source country; the held-out country's labels only score.
The harness reproduced last night's baselines exactly (US->IN 0.91571, IN->US 0.96935). Code: branch `hailmary`
(local, `ber/hailmary.py`). Outputs: `work/hailmary/`.

## H1: transfer model on unseen rows

The model is a round-2 LightGBM on 58 country-invariant features:
- within-S1 and within-record ranks and gaps
- round-1 p margins (p1 minus the best other S1 / other candidate) instead of raw p levels
- sibling agreement
- exact-match flags
- normalized string similarities
- house-number relations

Dropped (24): TF-IDF scores, IDF-Jaccard, frequency shares, raw round-1 p levels, number-format counts (`n_num_a/b`,
`hn_suffix`, `hn_in_other`, `hn_fmt_eq`, `hn_range`), `v2_idf_jacc` and `rare_*`. Mix on unseen rows only:
p = a * full + (1 - a) * transfer. Seen rows are unchanged, so the seen F is identical.

F0.5 on the unseen country:

| a | US->IN (India unseen) | delta | IN->US (US unseen) | delta |
|---|---|---|---|---|
| 1 (full, baseline) | 0.91571 | - | 0.96935 | - |
| 0.7 | 0.91502 | -0.0007 | 0.96901 | -0.0003 |
| 0.5 | 0.91345 | -0.0023 | 0.96827 | -0.0011 |
| 0.3 | 0.91050 | -0.0052 | 0.96703 | -0.0023 |
| 0 (transfer alone) | 0.90514 | -0.0106 | 0.96492 | -0.0044 |

Singleton-only F and precision fall with the same pattern: India singleton 0.788 -> 0.717, precision 0.952 -> 0.943
at a = 0. **Fails:** every a loses in both directions, and the loss is monotone in (1 - a). The country-specific
features help even on a country the model never saw. This matches the Phase B drop test (-0.0052 / -0.0028).

## H2: monotone constraints (increasing) on 24 similarity features

These are the name and address token/char similarities, IDF-Jaccard and the TF-IDF scores. Method "intermediate".
The change applies to all rows.

| direction | unseen F0.5 | delta | seen F0.5 | delta |
|---|---|---|---|---|
| US->IN | 0.91475 | -0.0010 | 0.98487 | -0.0004 |
| IN->US | 0.96982 | +0.0005 | 0.97725 | -0.0008 |

**Fails:** the unseen side does not improve in both directions, and the seen side loses in both. That is worse than the
-0.0002 OOF gate.

## H3: what the unexplained false positives are (US->IN, analysis only)

137,782 unseen false positives. 58% match none of the known types (core_eq, legal_add, num_shift, num_same,
gen_swap). What that unexplained group looks like:
- **98.7% are decoys:** the record belongs to no S1. 89% sit on an S1 that also has a correct link, and 82% are links
  beyond the S1's true count.
- **The address is near-identical:** median token-set similarity 99.3, and 96% are >= 85.
- **The name differs by one business word:** median name similarity 76. Examples: "Dynamic Ventures" vs "Dynamic
  Construction", "Ew Systems" vs "Ew Manufacturing". Median q 0.994.
- **The record's address starts with a door number the S1 address lacks:** 75% of these false positives vs 12.5% of
  unseen true pairs (6x).
- **Indic-script record names are not enriched:** 19% vs 16% of true pairs.

Is this relevant to France? No labels are needed to check (blend_cal test links vs train true pairs):

| | S1 address without a leading number | link whose record has one and the S1 not |
|---|---|---|
| India | 61% | 12.8% (true pairs 12.5%) |
| US | 14% | 10.4% (true pairs 10.3%) |
| France | 14% | 9.9% |

The decoy recipe that fools the US-trained model on India needs S1 addresses without a house number, and 61% of India's
S1 addresses have none. France looks like the US here, so this failure has little room on France. A fix would need
new name-difference / number-prefix features, a full round-2 rebuild and a clean-room change, and it would target an
India-specific failure. **Not built.**

## What this says

- The unseen-country loss in LOCO is not caused by trusting country-specific features. Removing or constraining them
  loses in both directions (H1, H2, Phase B).
- The biggest LOCO error cell is a generator decoy style that depends on the country's address format. The source
  country's labels cannot teach it.
- The public A/B (blend_cal 0.985131 > blend_raw 0.98488) went the opposite way from LOCO's k prediction. LOCO is a
  weak proxy for France, and only uploads confirm France changes.
