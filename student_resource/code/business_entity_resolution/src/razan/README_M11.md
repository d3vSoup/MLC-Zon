# QTpies M11h bundle: drop-in for src/razan/ (re-blend at partner weight 1.0)

M11 = stage 3 stacker on M10 features + the teammate's clean-room lgbm2_ce p (razan_export_*.parquet from
export_for_razan.py). Train CV on QTpies pairs, live S1s, same folds:
  M10 0.98898 | blend recipe 0.7*M10 + 0.3*teammate + isotonic 0.98933 | **M11 0.98956** (+0.00023 over the blend recipe).
M11h = M11 for US/India test pairs, M10 for France test pairs (the stacker learned no France); train OOF = M11.

## Blend (teammate clean room)
1. Copy `oof_s2ce.parquet` and `pred_test_s2ce.parquet` over `src/razan/` (ids/ and dropped_train_s1.parquet unchanged).
2. default.yaml: **`blend.w: 0.0`** (= partner weight 1.0: M11 already contains his p; pairs only he scored keep his p),
   `mode: asis`, `unseen_k: auto`, `n_jobs: 16`.
3. `python -m ber.cli blend` -> send `output/candidates/blend_cal/matching_results.tsv` (+ md5), `oof_live` overall/US/India
   (M10 blend at partner weight 0.7: 0.9907) and the France k chosen.

## Reproduce
- M11: `BER_WORK=<norm dir> python stage3.py <M9 share> <M11 out> [gt_pairs.parquet] <dir with razan_export_{train,test}.parquet>`
  -> oof a381bdfd1558eb15b2aaa1f28493173d, test 31f05ada3219216d2d72aa4629e134ca (deterministic).
- M11h: `python make_m11h.py <M9 share> <M10 share> <M11 share> <out>` -> oof a381bdfd1558eb15b2aaa1f28493173d,
  test e9ba15537cdc045408f2a4fe7e6594fd.
