"""Cross-encoder pair scorer (GPU). Model: intfloat/multilingual-e5-small (MIT, 118M), 1-logit head.

Text of a record = normalised "name | address" (transliterated, junk-stripped; alias wrappers kept so the model
can learn them). 2-fold cross-fitting by S1 hash (same folds as the GBDT) keeps train scores out-of-fold.

python ce.py train <fold>                                   # fold-k pairs: positives + hard negatives (top-5 look-alikes)
python ce.py score <split> <pairs.parquet> <out.parquet> <ce_fold0> <ce_fold1>
    train: each pair scored by the model of the *other* fold (OOF); test: mean logit of both models
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import w, SEED
from model import gt_idx, label

BASE = os.environ.get("BER_CE_BASE", "intfloat/multilingual-e5-small")
MAXLEN, N_POS, N_NEG, BS, LR = 128, 1_200_000, 1_200_000, 256, 5e-5
BAND = tuple(map(float, os.environ.get("BER_CE_BAND", "0.001,0.999").split(",")))  # stage-1 p band that gets a CE score
HARD = os.environ.get("BER_CE_HARD", "0") == "1"  # M4: add stage-1 OOF hard negatives
DEV = "cuda" if torch.cuda.is_available() else "cpu"


RAW = os.environ.get("BER_CE_RAW", "0") == "1"  # CE-2: raw text (casing, native script, source formatting kept)
TAG = os.environ.get("BER_CE_TAG", "ce")          # model dirs {TAG}_fold{k}


def texts(split):
    if RAW:
        cols, pre, n, a = ["business_name", "business_address"], f"{split}_s", "business_name", "business_address"
    else:
        cols, pre, n, a = ["nm", "ad"], f"n_{split}_s", "nm", "ad"
    s1 = pl.read_parquet(w(f"{pre}1.parquet"), columns=cols)
    r = pl.concat([pl.read_parquet(w(f"{pre}{k}.parquet"), columns=cols) for k in (2, 3)])
    t = lambda d: (d[n].fill_null("") + " | " + d[a].fill_null("")).to_list()
    return t(s1), t(r)


def train(fold):
    t0 = time.time()
    torch.manual_seed(SEED)
    gt = gt_idx()
    c = pl.scan_parquet(w("cand_train/*.parquet")).select("ri", "si", "cc") \
          .with_columns(pl.col("cc").rank("ordinal", descending=True).over("ri").alias("rk")).filter(pl.col("rk") <= 5).collect()
    c = label(c, gt)
    tr = c.filter(pl.col("fold") == fold)
    pos, neg = tr.filter(pl.col("y") == 1), tr.filter(pl.col("y") == 0)
    neg = neg.sample(min(N_NEG, neg.height), seed=SEED)
    if HARD and os.path.exists(w("oof_m.parquet")):  # half the negatives = the stage-1 model's own confusions
        h = label(pl.read_parquet(w("oof_m.parquet")).select("ri", "si", "p"), gt) \
              .filter((pl.col("fold") == fold) & (pl.col("y") == 0)).sort("p", descending=True).head(N_NEG // 2)
        print(f"hard negatives: {h.height}, p range {h['p'].min():.3f}-{h['p'].max():.3f}", flush=True)
        neg = pl.concat([neg.head(N_NEG - h.height).select("ri", "si", "y", "fold"), h.select("ri", "si", "y", "fold")])
    tr = pl.concat([pos.sample(min(N_POS, pos.height), seed=SEED).select("ri", "si", "y", "fold"), neg.select("ri", "si", "y", "fold")]) \
           .sample(fraction=1.0, shuffle=True, seed=SEED)
    va = c.filter(pl.col("fold") != fold).sample(100_000, seed=SEED)
    s1t, rt = texts("train")
    print(f"pairs train {tr.height} (pos {tr['y'].sum()}), val {va.height} {time.time() - t0:.0f}s", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEV)
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = (tr.height + BS - 1) // BS
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    scaler = torch.amp.GradScaler()
    a = [s1t[i] for i in tr["si"].to_list()]
    b = [rt[i] for i in tr["ri"].to_list()]
    y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    for step, i in enumerate(range(0, len(a), BS)):
        enc = tok(a[i:i + BS], b[i:i + BS], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            logit = net(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), y[i:i + BS].to(DEV))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        if step % 500 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    out = w(f"{TAG}_fold{fold}")
    model.half().save_pretrained(out)
    tok.save_pretrained(out)
    p = score_pairs(va, s1t, rt, out)
    yv = va["y"].to_numpy()
    from sklearn.metrics import roc_auc_score, log_loss
    print(f"fold {fold} val AUC {roc_auc_score(yv, p):.5f} logloss {log_loss(yv, 1 / (1 + np.exp(-p))):.5f} "
          f"{time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def score_pairs(pairs, s1t, rt, model_dir, bs=1024):
    a = [s1t[i] for i in pairs["si"].to_list()]
    b = [rt[i] for i in pairs["ri"].to_list()]
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])  # length-sorted batches = less padding
    out = np.zeros(len(a), dtype=np.float32)
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
    net = torch.nn.DataParallel(m) if torch.cuda.device_count() > 1 else m
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([a[k] for k in idx], [b[k] for k in idx], truncation=True, max_length=MAXLEN, padding=True,
                  return_tensors="pt").to(DEV)
        out[idx] = net(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out


def score(split, pairs_path, out_path, d0, d1):
    t0 = time.time()
    pairs_path, out_path, d0, d1 = (x if os.path.isabs(x) else w(x) for x in (pairs_path, out_path, d0, d1))
    pairs = pl.read_parquet(pairs_path)
    if "p" in pairs.columns:  # stage-1 predictions given: score only the uncertain band
        pairs = pairs.filter((pl.col("p") >= BAND[0]) & (pl.col("p") <= BAND[1]))
    pairs = pairs.select("ri", "si").with_columns(((pl.col("si").hash(SEED) % 2).cast(pl.Int8)).alias("fold"))
    s1t, rt = texts(split)
    if split == "train":  # OOF: fold-k pairs by the model trained on the other fold
        parts = [pairs.filter(pl.col("fold") == k).pipe(lambda p, k=k: p.with_columns(
            pl.Series("ce", score_pairs(p, s1t, rt, d1 if k == 0 else d0)))) for k in (0, 1)]
        res = pl.concat(parts)
    else:
        res = pairs.with_columns(pl.Series("ce", (score_pairs(pairs, s1t, rt, d0) + score_pairs(pairs, s1t, rt, d1)) / 2))
    res.select("ri", "si", "ce").write_parquet(out_path)
    print(f"scored {res.height} {split} pairs {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    train(int(sys.argv[2])) if sys.argv[1] == "train" else score(*sys.argv[2:])
