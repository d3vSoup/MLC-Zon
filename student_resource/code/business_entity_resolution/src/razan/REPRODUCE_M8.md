# Reproduce QTpies M8 → `sub17_m8_cal`

**What this reproduces**
- **The model:** M8 (`qtpies-m8`). M5 plus two changes:
  - **twin-world training** (§1);
  - **a cluster cross-encoder** (`cce`, §2.5).
- **Train OOF F0.5:** clean **0.98849** on original rows only (India 0.98754, US 0.98913).
  In the kernel, with the synthetic twin rows included, it reads 0.98835.
- **The submission:** `submissions/sub17_m8_cal/matching_results.tsv`.
  - Public **0.984591**.
  - 1,732,544 rows, 5,765,628 matched IDs.
  - SHA-256 `e980d8b168bcd05ef2bcdc7aee0c5ccaf3a19d68787a2683632b05bc809b94e7`.
- **The blend inputs:** `runs/qtpies-m8/qtpies_m8_share.zip`, a drop-in replacement for the M5 bundle in `src/razan/`
  (§4).

**Scope:** steps 1–12 are identical to M5 and are documented in `REPRODUCE.md` (same pins, datasets and commands).
This file adds steps 13–19.

**Code version: git tag `m8`** (MLC-raZON, branch `er-pipeline`, merged into `main`).
- At that tag, `code/business_entity_resolution/src/` is **byte-identical to the snapshot the `qtpies-m8` kernel ran**.
  Every kernel embeds its exact `src/` as a tarball, so this was extracted from `kaggle/k/qtpies-m8/run.py` and
  diffed: zero differences.
- `kaggle_runner.py` at the tag is the version that pushed M8.
- The next commit on `main` adds only the default-off M9 hooks: `BER_CCE_TAG`, a `cceb` feature read only if
  `cceb_*.parquet` exists, and the `cceb`/`m9` batches.
- **Earlier kernels of the lineage ran older snapshots.** Against the tag, they differ **only additively, behind env
  switches whose defaults are off**:
  - `num_feats` (on only with `BER_NUMF=1`);
  - the CPU-only fp32 path in `ce.py`;
  - `cce.py`, which didn't exist yet.
- So tag `m8` reproduces every step below as run.

**Snapshot IDs** (SHA-256 over the sorted `.py` files, first 12 hex):

| Kernels | Snapshot |
|---|---|
| t-gen, t-feats, t-ce4 | `63575af22e5d` |
| t-ce2, m7b | `30815c4c7e3c` |
| cce-0/1, m8 | `4594535af76c` |

---

## 1. What "twin training" is (the answer to "did we do twin training?")

Yes. M6 onwards train stage 2 in a synthetic **twin world**. Here's why and how.

**The problem.** US test contains about 8× more **near-twin decoys** than train. These are S2/S3 rows copied from a real
business with:
- the house number shifted by a small, mostly positive δ;
- a legal suffix added (LLC/Inc/Co/Corp/Ltd/LP), often in groups of 2.

France has **generic-word-swap** twins ("Graines *École*" → "Graines *Parents*" at the same address). Train has almost
none of either, so the model scored them p ≈ 0.86 and linked them.

**The fix: `src/twins.py`, run in kernel `qtpies-t-gen`.** Synthetic decoys are generated **from train only** and
**appended to train S3**. Test is never modified.

- **N-twins (house number):** 90,665 rows from 72,202 US parent S1s.
  - Rate: `BER_TWIN_N=US:0.072` of eligible parents. India gets none, because its test shows no excess.
  - Each twin is a copy of one of the parent's true rows whose numbers contain the parent's first house number.
  - The number is shifted by δ, with |δ| drawn from the **test** twin histogram bins
    `DBIN = [(1,2,9604), (3,5,12288), (6,10,7809), (11,20,8120), (21,50,4105)]`. It is positive with p 0.89.
  - A legal suffix from `SUF` is appended with p 0.94.
  - With p 0.3 the twin comes as a group of 2 rows sharing the shifted number.
- **G-twins (generic word):** 26,498 rows from 21,739 parents.
  - Rate: `BER_TWIN_G=US:0.03,India:0.03`.
  - One generic core token is swapped for another generic word from the same country's vocabulary: a token is generic
    if it appears in ≥ 0.2% of that country's S1 names (`GEN_DF`, label-free). Same address; raw casing/style copied.
- **Determinism:** `SEED = 42` (numpy `default_rng`). IDs look like `S2-T0000012`.
- **Row placement:** twin rows get ri values 10,320,219–10,437,381, i.e. **after** all real rows, so existing si/ri
  stay valid.
- **Labels:** every twin is a **negative** (y = 0). A twin belongs to no S1.

**How they enter training.** Nothing upstream is retrained: stage 1 and both pairwise CEs are reused.
- `twins.py block` blocks the twin rows against train S1s → `cand_train/{country}_T_{nnn}.parquet`.
- `feats.py train` then `run.py predtwin` give the twin pairs stage-1 **OOF** p from the M2 fold models
  (`oof_m_part9.parquet`). Of 1,658,635 twin candidate pairs, 116,106 pass p ≥ 0.002 into stage 2.
- CE-4 and CE-2 **score** the twin pairs (`ce_train_twin.parquet`, `ce2_train_twin.parquet`), so no CE value is
  missing on twins. A missing value would itself leak "this is a twin".
- **Stage 2 is retrained on real + twin pairs.** It uses twin-aware features (`BER_TWINF=1`):
  - `dn`, `dn_abs`, `dn_pos20`: the number shift;
  - `leg_add`, `leg_drop`: legal-form change.
- M7 onwards add house-number shape features (`BER_NUMF=1`): `hn_lev`, `hn_pre`, `hn_suf`, `hn_trunc`,
  `hn_transpose`, `hn_absdiff_log`, `hn_reldiff`, `num_jacc`.
- The cluster CE (§2.5) also trains on units that contain twin rows, as negatives.

**Effect (label-free, measured on test):**
- The US twin cell (number ≠, name =, p 0.3–0.98) dropped from 71.6 to 18.4 rows per 1k S1.
- The US prior-shift calibration went from k 0.060 to **1.0**: no correction needed any more.
- France raw links per S1 fell from 3.326 to 3.204.

**Public:** M5-era 0.979 → M7b 0.98442 → M8 0.984591.

---

## 2. Kaggle job chain, steps 13–19 (after steps 1–12 of `REPRODUCE.md`)

Run every command from `code/business_entity_resolution/` as `python kaggle_runner.py chain <batch>`.
- `=batch` means wait only.
- Batch settings come from `BATCH_ENV` in `kaggle_runner.py`, and every setting is listed here.
- Kernel inputs come from `BATCHES` (`kernel_sources`), so **the order matters**.
- The kernel template gives files from `/qtpies-t-gen/` **priority** when linking inputs. So the twin-extended
  `n_train_s3` / `train_s3` override the dataset copies in every downstream kernel.

| Step | Batch | Kernel(s) | Hardware | Settings | Inputs (besides `qtpies-er-norm`) | Outputs | Wall time |
|---|---|---|---|---|---|---|---|
| 13 | `t_gen` | `qtpies-t-gen`: `twins.py gen`, `twins.py block` | CPU | `BER_DROP=16 BER_TWIN_G=US:0.03,India:0.03` (`BER_TWIN_N` default `US:0.072`) | dataset `qtpies-er-data` (raw) | `twin/n_train_s3.parquet`, `twin/train_s3.parquet` (extended), `twin/twin_meta.parquet`, `cand_train/*_T_*.parquet` | 6 min |
| 14 | `t_feats` | `qtpies-t-feats`: `feats.py train`, `run.py predtwin` | CPU | `BER_DROP=16` | train blocking, `ftd-train-*`, `m2-fit1`, `t-gen` | twin `F_train` shards, **`oof_m_part9.parquet`** | 2 min |
| 15 | `t_ce4` | `qtpies-t-ce4`: `ce.py score train oof_m_part9.parquet ce_train_twin.parquet ce_fold0 ce_fold1` | GPU T4×2 | `BER_CE_BAND=0.002,1.01` | `ce4-train-{0,1}`, `t-feats`, `t-gen` | `ce_train_twin.parquet` | 3 min |
| 16 | `t_ce2` | `qtpies-t-ce2`: same with `ce2_*` | **CPU** (the GPU queue was saturated; fp32 path) | `BER_CE_BAND=0.002,1.01 BER_CE_RAW=1 BER_CE_TAG=ce2` | `ce2-train-{0,1}`, `t-feats`, `t-gen`, dataset `qtpies-er-data` | `ce2_train_twin.parquet` | 75 min |
| 17 | `m7b` | `qtpies-m7b`: `run.py gather1`, `train2ce`, `predictce` | CPU | `BER_DROP=16 BER_POOL=12000000 BER_GENERIC=1 BER_TWINF=1 BER_NUMF=1` | M5's inputs + `t-gen`, `t-feats`, `t-ce4`, `t-ce2`, dataset `qtpies-er-cand-test` | **`oof_m.parquet` / `pred_test_m.parquet`** (stage 1 incl. twin part 9), M7b `oof_s2ce` / `pred_test_s2ce` | 39 min |
| 18 | `cce` | `qtpies-cce-0`, `qtpies-cce-1`: `cce.py run 0` / `run 1` | GPU T4×2 each, run in parallel | `BER_DROP=16` | `qtpies-m7b`, `qtpies-t-gen`, dataset `qtpies-er-data` | `cce_train_part{k}.parquet`, `cce_test_part{k}.parquet` (ri, si, ce logit) | 81 min each |
| 19 | `m8` | `qtpies-m8`: `run.py gather1`, `train2ce`, `predictce` | CPU | the same as `m7b` | `m7b`'s inputs + `cce-0`, `cce-1` | **`oof_s2ce.parquet`, `pred_test_s2ce.parquet`**, `output/s2ce/{matching_results,candidate_pairs}.tsv` | 31 min |

As run: `chain t_gen t_feats t_ce4` then `chain t_ce2 m7b`, then `chain cce_smoke`, then `chain cce m8`. The smoke run
(`BER_CCE_SMOKE=2000`, CPU, train AUC 0.8996) is optional.

### 2.5 The cluster cross-encoder (`src/cce.py`)

**What it is.** A set-aware matcher: **one sequence = an S1 plus all its stage-2 candidate rows**, laid out as
`[CLS] s1 [SEP] row1 [SEP] row2 [SEP] …`, with one logit per row read at that row's first token. It sees siblings and
twins side by side, which a pairwise CE cannot.

| Setting | Value |
|---|---|
| Base model | `intfloat/multilingual-e5-small`, MIT, rev `614241f622f53c4eeff9890bdc4f31cfecc418b3` (`AutoModel` plus a linear head) |
| Text | **Raw** `business_name \| business_address` |
| Units | Per S1, the stage-2 candidates (stage-1 p ≥ `P_MIN` 0.002, from M7b's `oof_m` / `pred_test_m`), best p first, at most `M_ROWS` 12 |
| Token budget | `S1_TOK` 48, `ROW_TOK` 34 |
| Folds | `hash(si, 42) % 2`, the same as everywhere. The fold-k model trains on 400,000 sampled fold-k train units (`BER_CCE_UNITS`) and scores the **other** fold's train units and the **other** fold's test S1s. Every train score is out-of-fold, and each test S1 gets exactly one model |
| Training | Masked BCE over the valid rows, 1 epoch, batch 32 units, lr 5e-5, AdamW (wd 0.01), linear warm-up 3%, grad clip 1.0, fp16 autocast, `DataParallel` over 2× T4. 12,500 steps |
| Output | `cce_{split}_part{k}.parquet` (ri int32, si int32, ce float32 logit). Fold 0 scored 4,440,349 train and 4,715,522 test pairs |

**How it's used.** `model.stage2_table` left-joins every `cce_{split}*.parquet` as the stage-2 feature **`cce`**, which
is the top-gain feature in M8 (15.4M vs `ce2` 6.5M). Pairs beyond the 12-row cap get `cce` = null, the same on train
and test.

---

## 3. Fetch and calibrate → `sub17_m8_cal`

```
kaggle kernels output razanml/qtpies-m8 -p runs/qtpies-m8 --file-pattern "pred_test_s2ce\.parquet$|oof_s2ce\.parquet$"
cd code/business_entity_resolution
BER_WORK=<repo>/work2 BER_DROP=16 PYTHONUTF8=1 python memguard.py 2.5 python src/calibrate.py \
    ../../runs/qtpies-m8/pred_test_s2ce.parquet ../../submissions/sub17_m8_cal ../../runs/qtpies-m8/oof_s2ce.parquet
cd ../../student_resource && python utils/validate_submission.py --matching ../submissions/sub17_m8_cal/matching_results.tsv --test-dir dataset/test
```

**Expected output:**
```
India: mid-band excess 0.962, prior-shift k=1.000
US: mid-band excess 0.950, prior-shift k=1.000 (anchor)
France: seen-country empty 0.0583; target 0.0598; empty at k=1 0.0571; chosen k=0.35 -> 0.0597
India: empty 0.0587, links/S1 3.3448
US: empty 0.0577, links/S1 3.3759
France: empty 0.0597, links/S1 3.1522
wrote ../../submissions/sub17_m8_cal\matching_results.tsv: 1732544 rows, 5765628 ids
PASS
```

**Important:** the OOF file must **include the twin rows**. Our US difference-in-differences prior shift reads k = 1
with the twin cell present. Without it, it reads a spurious k = 0.159.

---

## 4. The share bundle (`runs/qtpies-m8/qtpies_m8_share.zip`): blend input

**Layout:** same as the M5 bundle. `ids/` and `dropped_train_s1.parquet` are **byte-identical** to M5's, because the
normalised files are the same.

| File | Contents |
|---|---|
| `oof_s2ce.parquet` | 8,812,682 rows (ri, si, y, p), **real rows only**. The 116,106 twin pairs are removed, so every ri maps through `ids/` |
| `oof_twin_rows.parquet` | The 116,106 twin pairs (y = 0). **Ignore it for the blend.** Concatenate it back only to reproduce our own calibration (§3) |
| `pred_test_s2ce.parquet` | 9,531,458 test pairs, raw (before calibration). This set is our `candidate_pairs.tsv`. Test has no twin rows |

**Checked:** `calibrate.py` run from the bundle, with the twin rows concatenated back, reproduces `sub17_m8_cal`
**byte for byte**.

**For the blend:**
- Our p is stage-2 probabilities from a model trained in the **DROP16 + twin world**.
- Your calibration uses k = 1 for seen countries, so the twin rows play no part there.
- Re-derive France k by your empty-rate rule. The frozen 0.18386 was fitted to the M5 blend.

## 5. Determinism

**Bit-exact:** `sub17` from the saved predictions (checked twice). Twin generation is seeded (`SEED 42`) and
deterministic given polars 1.44.2 and numpy.

**Not bit-exact:** LightGBM (multi-threaded, `deterministic` not set) and the CEs (fp16, `DataParallel`, cuDNN) can
differ in the ~1e-4 range of OOF F0.5 on a from-scratch rerun.

**Next model (in progress, 27 Sep):** `cceb` is the same cluster CE on `intfloat/multilingual-e5-base` (MIT), with
200k units per fold, feeding `m9` (M8 + `cceb`). Its bundle will be `runs/qtpies-m9/qtpies_m9_share.zip`, in the same
format, if M9 beats M8 on clean CV.
