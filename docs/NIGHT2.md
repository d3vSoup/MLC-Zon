# Night 2 (Sun 27 Sep, 20:55-21:20 IST): last push, evidence first

**Verdict: no new candidate passes the upload gate (predicted gain >= +0.0002 with evidence). FINAL 2 (0.986758, md5
`50c695de7ae6c03d19a70ee75040f983`) stays our best and last upload. Recommendation: use no more slots tonight.**

## 1. Country decomposition (pairs of scored files identical outside one country)

| pair | country that differs | rows | public change | reading |
|---|---|---|---|---|
| blend_raw (k 1) vs blend_cal (k 0.184), M5 blend | France | 32,679 | -0.000251 for k 1 | France down-scaling right (France F ~+0.0017) |
| #33 -> #34 (US/India rules: M10 p -> M12h p) | US/India | ~3,200 | +0.000076 | M12h rules better |
| #34 -> FINAL 2 (US/India from #35) | US/India | 5,879 | +0.000027 | noise level |
| #35 -> FINAL 2 (France from #34) | France | 9,458 | +0.000092 | France with our weight 0.3 beats our weight 0 |
| Razan's probes #19-#32 (LB_LOG_2026-09-27.md) | one country each | - | +0.00011 ... +0.00031 each; France veto .7-.9 -0.000108; US/IN veto .3-.5 -0.000020 | fuse thresholds sit at the measured optimum |

**FINAL 2 is already the best known component per country:** France from #34, US/India from #35. No other scored
file has a better component for any country.

## 2. Do label-free diagnostics predict the public deltas?

Per file and country: empty share, links per S1, rows differing (all computed; table in the session log).

France has two separated data points, and they conflict:
- fewer links won (k 0.184 vs 1: 3.197 vs 3.333 links/S1, +0.00025);
- fewer links lost (#35 vs #34 France: 3.172 vs 3.180, -0.00009).

With 2-4 points per country and opposite signs, **no diagnostic predicts reliably**. The uncertain band and the twin
cells were not computed further, for the same reason.

## 3. Candidates

| id | idea | mechanism | evidence | md5 | predicted | gate |
|---|---|---|---|---|---|---|
| C1 | France from the fuse on the M10 blend at OUR weight 0.5 (US/India = FINAL 2) | two independent models help more on an unseen country | our weight 0 -> 0.3: +0.000092 (9,458 France rows) | `b6f2367e2f8bd613b120ac16462dff47` (2,335 France rows differ from FINAL 2) | ~+0.00002 (same per-row rate); sign uncertain | **fails** |
| - | US/India with M12h rules on the M12h blend | M12h stacker +0.00006 CV over M11h | blend OOF +0.00007 (noise) | not built | ~+0.00002 | fails |
| - | more France vetoes / adds | - | already probed: veto .7-.9 negative, adds .95-.99 ~0 | - | <= 0 | fails |
| - | France k away from the rule | - | the rule won its only A/B; no evidence it is off | - | unknown | no mechanism |

Why 0.99 is out of reach tonight:
- The top-50 line (~0.9897) is +0.003 away.
- Today's biggest real steps were +0.0013 (the M10 partner model) and +0.0003 (the fuse).
- Every remaining lever is at the +0.0001 scale, and probe-chosen files add private-board overfit risk.

## 4. Packaging status for tomorrow
- Tag `freeze-2026-09-27f` (`50f0d64`): Documentation [FINAL] = FINAL 2 (public 0.986758), with the per-country
  merge disclosed. The code is identical to 27e.
- Zip from 27f: `tools/make_zip.sh QTpies ../MLC-Zon-fuse-staged/out_final2/matching_results.tsv` (the union
  `candidate_pairs.tsv` is hard-linked next to it). **Blocked tonight by disk** (6.6 GB free, floor 8 GB).
- It differs from the verified 27e zip (`0dd30abc`) only in `output/matching_results.tsv` (#34 -> FINAL 2) and the
  Documentation. The 27e zip already regenerated FINAL 2 from its own contents (md5 match).
- Verified zips kept: 27e `QTpies_submission.zip` (`0dd30abc`, output #34), 27c `QTpies_submission_w07_27c.zip`
  (`9c6798a9`, output w0.7).
