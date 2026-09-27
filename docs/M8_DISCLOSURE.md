# M8 inputs: rules audit (Sun 27 Sep, updated 11:15, read-only; no code changed)

Sources:
- `qtpies_m8_share.zip`: the data bundle.
- `qtpies_m8_src.zip`: the kernel source snapshot.
- Razan's repo `MLC-raZON` (main `a7e7d69`, tag `m8` = `051ad18`).
- `REPRODUCE_M8.md` and `AUDIT.md` (from tag `m8`).

## Code identity: PASS
- **Snapshot hash:** SHA-256 over the sorted `src/*.py` files (14 files, contents concatenated), first 12 hex =
  **`4594535af76c`**. This equals the M8 kernel snapshot.
- **Files present:** `twins.py`, `cce.py`, `stage2.py`, `model.py`, `ce.py` in `src/`, and `kaggle_runner.py` next to
  it. The zip and the unzipped folder are identical.
- **Tag `m8` `src/` == the snapshot** (ignoring CR/LF). Tag `m8` `kaggle_runner.py` == the snapshot's.
- **Repo HEAD (`a7e7d69`) vs the snapshot:** only 3 files differ, and every change is additive and off by default:
  - `cce.py`: `TAG = BER_CCE_TAG` (default `"cce"`, so output names are unchanged);
  - `model.py`: also joins `cceb_*.parquet` if such files exist (none for M8);
  - `kaggle_runner.py`: new `cceb` / `m9` batches.

  The `BER_NUMF` and CPU fp32 items that the older REPRODUCE_M8 draft listed are already inside the M8 snapshot
  (M8 ran with `BER_NUMF=1`), so they are not HEAD-only.
- **Data files** match the README SHA-256s. `ids/` and `dropped_train_s1.parquet` are byte-identical to M5's.
- The 4 files of the data bundle's `src/` equal the snapshot's `calibrate.py`, `common.py`, `decide.py`, `run.py`.

## Hard-coded country names (full grep of the snapshot)

| where | what | effect |
|---|---|---|
| `twins.py:29` | `RATE_N = BER_TWIN_N`, default `"US:0.072"` | Number-shift twins only for US parents (India 0: its test shows no excess). **Affects the model.** |
| `twins.py:30` + `kaggle_runner.py:103` | `RATE_G` default `"US:0.012,India:0.012"`; the M8 run set `BER_TWIN_G="US:0.03,India:0.03"` | Generic-word-swap twin rate per country. **Affects the model.** |
| `calibrate.py:87` | `c != "France"` | Unseen-country list when no OOF file is given. The M8 run passes an OOF file, so the set is data-derived. Not used by our blend. |
| `score.py:4,24-25` | `TEST_MIX = {"US": 0.383, "India": 0.468, "France": 0.150}`; France proxied by the mean of the others | Test-mix-weighted CV **report** only (model.py `report`). No training or decode effect. |
| docstrings: `calibrate.py:9-13`, `twins.py:5-6`, `stage2.py:34` | country names in comments | none |

## Statistics fitted on test data (all label-free)

| statistic | value | source | used in |
|---|---|---|---|
| `DBIN` \|δ\| histogram of house-number shifts | `(1,2,9604) (3,5,12288) (6,10,7809) (11,20,8120) (21,50,4105)` | **US test** twin cell on M5 (AUDIT §8.3 table: the test column's counts) | N-twin shift sizes |
| `P_POS` shift positive | 0.89 | **US test** 89.3% (AUDIT §8.3; train OOF 43.2%) | N-twin sign |
| `P_SUF` legal suffix added | 0.94 | **US test** 94.2% (AUDIT §8.3; train OOF 69.5%) | N-twin suffix |
| `P_PAIR` twin groups of 2 | 0.3 | **US test** twin cell ("30% of groups have 2 rows", twins.py docstring, from the §8.3 measurement) | N-twin group size |
| `RATE_N` US 0.072, India 0 | | test excess ~72 rows / 1k S1 in the US cell, ~1 in India | twin volume |
| `RATE_G` US / India 0.03 | | the France "same number, different name" test excess (6-12x) motivated the type; the rate itself is set by hand | twin volume |
| `BER_DROP=16` | 16% of train S1 treated as absent | test distractor share 38-40% vs 26% in train (`common.py`) | training density, CV world |
| `GEN_DF` generic vocabulary for stage-2 features | tokens in >= 0.2% of the split's S1 names, per country | computed per split, so **on test S1 names for test pairs** (`stage2.generic_feats`) | stage-2 features |
| prior-shift k per seen country, France k 0.35 | | test p-band ratios, test empty-S1 rate | Razan's own calibrated submission only; **not used by our blend** |

**Train-only twins: confirmed.** `twins.load()` reads only `n_train_s1/s2/s3` and the train ground truth. The twins
are copies of true rows of live train S1s. The G-twin vocabulary comes from train S1 names. Twins are appended after
all real train rows, and test is never modified. Test data enters the twin generator only through the numbers above
(DBIN, the probabilities, the rates).

**Nothing external: confirmed** for the snapshot.
- **Models:** only `intfloat/multilingual-e5-small` (MIT, 118M), pinned revision in REPRODUCE, used by `ce.py` /
  `cce.py`. `multilingual-e5-base` (MIT) appears only in the post-M8 `cceb` hooks, which M8 does not use.
- **Kaggle inputs:** only the team's own datasets (`razanml/qtpies-er-norm`, `-er-data`, `-er-cand-test`: competition
  data and its derived candidates) and their own upstream kernels. `competition_sources` is empty.
- **Internet** is used for `pip install` and the model download only. No HTTP, API or lookup calls appear in `src/`.
- **Leaderboard use:** the France odds scale was validated with public-LB probes (AUDIT §8.4). That is allowed LB
  feedback, not data.

Our own pipeline, for the same paragraph:
- The France k (empty-rate matching, 0.18385635887853335 on the M5 blend; re-derived for the M8 blends) is fitted
  label-free on test predictions, for countries absent from train (found by a data rule, no names in code).
- Our `names2` generic set and IDF are computed per split, so on test S1 names for test pairs.

## Draft Documentation paragraph

> **Use of test-set statistics and country keys.** No test labels are used anywhere. Some label-free statistics of the
> unlabeled test set shape the final model, and we list them here.
> - In our pipeline:
>   - The odds of countries absent from train are scaled by a factor chosen so that their share of S1 rows with no
>     match equals the seen countries' share plus 0.0015.
>   - Generic-word sets and IDF weights for name features are computed per split.
> - In the partner pipeline (QTpies M8):
>   - Training treats 16% of train S1 as absent. That matches the test distractor share (38-40% vs 26% in train).
>   - Generic-word features use per-split vocabularies.
>   - Training adds synthetic "twin" negatives generated only from train rows: copies of real records with the house
>     number shifted and a legal suffix added, and copies with one generic name word swapped.
>   - The twin parameters were measured label-free on test predictions: the shift-size histogram, the 89% positive
>     shifts, the 94% added suffixes and the 30% two-row groups.
>   - The twin rates are keyed by country name in the configuration: US 0.072 for number twins (none for India), and
>     0.03 for generic-word twins in US and India.
>   - A test-mix weighting (US 0.383, India 0.468, France 0.150) is used only to report cross-validation scores.
>
> Apart from these twin rates, country enters only as an opaque grouping key. The only pretrained model is
> multilingual-e5-small (MIT, 118M parameters), and no external data, APIs or lookups are used.
