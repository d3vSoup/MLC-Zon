# Cross-encoder runs on Kaggle (Ankit + Soup)

We re-score about 16M "unsure" business pairs with a multilingual cross-encoder (`xlm-roberta-base`, MIT license).
Each of you trains on one half of the train entities and scores the other half + the whole test band.
Your scores become one more feature in our round-2 model.

| who | trains on | scores | command flag |
|---|---|---|---|
| **Ankit** | half H1 | H2 band + test band | `--half 1` |
| **Soup** | half H2 | H1 band + test band | `--half 2` |

## 1. Set up the notebook (once)
1. Kaggle → New Notebook.
2. Settings → Accelerator: **GPU T4 x2**. Internet: **On**. The model downloads from Hugging Face.
3. Add Data (all shared with you, all private): `ce-train-pairs`, `ce-test-pairs`, `ber-code`.

## 2. Timing check first (about 10 min)
```
!pip install -q transformers==5.17.0
!python /kaggle/input/ber-code/ce_train_predict.py --half 1 --time_only        # Soup: --half 2
```
Post the last line, `TIMING: ... full run ETA X h`, in Discord. The timing run writes only to `pred_timing/`
and never mixes with the real run.

## 3. Full run
Put the same two lines in one cell, without `--time_only`, then **Save Version → Save & Run All (commit)**.
It keeps running after you close the tab.
- It prints `train step .../...` and `predict ... pairs/s` as it goes.
- It checkpoints training every 30 min and writes predictions in 200k-pair parts.
- It stops cleanly at 11 h, before Kaggle's 12 h limit.
- Our estimate is **about 2–4 h** on T4 x2. Your timing line gives the real number.

## 4. If it stops early (time limit or crash)
1. Open the finished version → Output → "New notebook with this output", or add that version's output as data.
2. Run the same command plus the resume flag:
   `!python /kaggle/input/ber-code/ce_train_predict.py --half 1 --resume_dir /kaggle/input/<that-output-folder>`
3. It restores the checkpoint, the finished model, and the finished parts, and it continues from there.
4. If the log says `loss is NaN`, add `--lr 1e-5` and resume.

## 5. Send back
- When the log says `ALL DONE`: Output → **Download** the `pred/` folder. It holds `train_h?_part*.parquet` and `test_h?_part*.parquet`, about 200 MB.
- Upload it to our Google Drive folder `ce_outputs/<your name>/`. Please do not rename files.
- Also send the timing line and the final `pairs/s` lines.
- Muaz merges both folders with `tools/ingest_ce.sh`.

## Rules
- Only these datasets and the `xlm-roberta-base` weights. No other data, and no internet lookups of businesses.
- Keep the datasets private. Do not share the files outside the team.
- Do not change the random seed or the halves. The merge expects exactly H1 → H2 + test and H2 → H1 + test.
