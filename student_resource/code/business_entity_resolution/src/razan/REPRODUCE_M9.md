# Reproduce QTpies M9 → `sub18_m9_cal`

**M9 = M8 plus one extra stage-2 feature, `cceb`.** `cceb` is the same cluster cross-encoder as `cce`, on a 3× larger
backbone. Everything in `REPRODUCE_M8.md` still applies (steps 1–19). This adds steps 20–21.

**Code:** MLC-raZON commit `6d6c947` (on `main`, merge `a7e7d69`).
- The `qtpies-cceb-0/1` and `qtpies-m9` kernels embedded snapshot **`3313ebdb1a60`**. That's SHA-256 over the sorted
  `src/*.py`, first 12 hex, the same hash definition as M8's `4594535af76c`.
- It equals `6d6c947` ignoring CR/LF.
- Relative to tag `m8`, it adds only these default-off hooks:
  - `BER_CCE_TAG` in `cce.py`;
  - a `cceb_*.parquet` join in `model.stage2_table`;
  - the `cceb` and `m9` batches in `kaggle_runner.py`.

## Steps 20–21

| Step | Batch | Kernel(s) | Hardware | Settings | Inputs | Outputs | Wall time |
|---|---|---|---|---|---|---|---|
| 20 | `cceb` | `qtpies-cceb-0`, `qtpies-cceb-1`: `cce.py run 0` / `run 1` | GPU T4×2 each, in parallel | `BER_DROP=16 BER_CCE_BASE=intfloat/multilingual-e5-base BER_CCE_TAG=cceb BER_CCE_UNITS=200000` | `qtpies-m7b`, `qtpies-t-gen`, dataset `qtpies-er-data` | `cceb_{train,test}_part{k}.parquet` (ri, si, ce) | 2.0–2.1 h each |
| 21 | `m9` | `qtpies-m9`: `run.py gather1`, `train2ce`, `predictce` | CPU | `BER_DROP=16 BER_POOL=12000000 BER_GENERIC=1 BER_TWINF=1 BER_NUMF=1` | `m8`'s inputs + `cceb-0`, `cceb-1` | `oof_s2ce.parquet`, `pred_test_s2ce.parquet` | 38 min |

**cceb specifics:**
- **Base model:** `intfloat/multilingual-e5-base` (MIT, 278M), downloaded by name on 27 Sep 2026 at 08:44 IST. Pin its
  revision (current `main` of the HF repo) for a rerun.
- **Training:** 200,000 train units per fold, 6,250 steps. Everything else as `cce` (§2.5 of `REPRODUCE_M8.md`): 12 rows
  per unit, raw text, fp16, 2× T4.

As run: `chain cceb m9`. Then the calibration as in M8 §3, into `submissions/sub18_m9_cal`.

## Results

| | Clean CV (original rows) | India | US |
|---|---|---|---|
| M8 | 0.98849 | 0.98754 | 0.98913 |
| **M9** | **0.98852** | 0.98757 | 0.98915 |

- **Stage-2 gain:** `cce` 13.6M, `cceb` 5.7M, `ce2` 3.0M (M8: `cce` 15.4M, `ce2` 6.5M). The larger backbone mostly takes
  over gain the other CEs already had.
- **Twin rows still linked:** N 939 → 864, G 286 → 257. US twin cell 18.4 → 17.5 per 1k S1.

**`sub18_m9_cal`:**
- 1,732,544 rows, 5,766,835 matched IDs, validator PASS.
- France k 0.35; US and India k 1.
- Links per S1: France 3.1556, US 3.3761, India 3.3450.

## Bundle (`runs/qtpies-m9/qtpies_m9_share.zip`)
- **Same layout as the M8 bundle:** `ids/` and `dropped_train_s1.parquet` are byte-identical.
  - `oof_s2ce.parquet` holds real rows only: 8,812,682 pairs.
  - `oof_twin_rows.parquet` holds the 116,106 twin pairs. **Ignore it for the blend.**
  - `pred_test_s2ce.parquet` holds 9,531,458 pairs.
- **Checked:** calibrate from the bundle, with the twin rows concatenated back, reproduces `sub18_m9_cal` **byte for byte**.
- `code_snapshot/` holds the exact kernel code (`3313ebdb1a60`).
