"""Cluster (set-aware) cross-encoder: ONE sequence = an S1 record + all its stage-2 candidate rows, one match logit per row.
Pairwise models score each (S1, row) alone; twins and sibling consistency only show up across rows (three rows sharing
6183 + a suffix vs the S1's rows at 6182). This model reads the whole candidate set at once. Raw text, MIT e5 backbone.
Folds as everywhere (hash(si) % 2): the fold-k model trains on fold-k S1s and scores the other fold's train S1s and the
other fold's test S1s, so every score is out-of-fold and test gets one model per S1.

python cce.py run <fold>   -> cce_train_part{fold}.parquet, cce_test_part{fold}.parquet  (ri, si, ce)
Inputs: oof_m.parquet / pred_test_m.parquet (stage-1 p, twin world), raw {split}_s{1,2,3}.parquet, n_train_* ids.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import w, SEED
from model import gt_idx
from stage2 import P_MIN

BASE = os.environ.get("BER_CCE_BASE", "intfloat/multilingual-e5-small")
M_ROWS, S1_TOK, ROW_TOK, BS, LR = 12, 48, 34, 32, 5e-5
N_UNITS = int(os.environ.get("BER_CCE_UNITS", 400_000))   # training S1 units for this fold
SMOKE = int(os.environ.get("BER_CCE_SMOKE", 0))            # >0: tiny run with that many units
TAG = os.environ.get("BER_CCE_TAG", "cce")                   # output prefix = stage-2 feature name (cceb: e5-base run)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def units(split, P):
    """S1 -> its stage-2 candidate rows (p >= P_MIN), best stage-1 p first, at most M_ROWS."""
    return P.filter(pl.col("p") >= P_MIN).sort(["si", "p"], descending=[False, True]) \
            .group_by("si", maintain_order=True).agg(pl.col("ri").head(M_ROWS)) \
            .with_columns((pl.col("si").hash(SEED) % 2).cast(pl.Int8).alias("fold"))


def texts(split, idx, s):
    """Raw 'name | address' for the given row indices of source s (1 = S1, else concat(S2, S3))."""
    cols = ["business_name", "business_address"]
    d = pl.read_parquet(w(f"{split}_s1.parquet"), columns=cols) if s == 1 else \
        pl.concat([pl.read_parquet(w(f"{split}_s{k}.parquet"), columns=cols) for k in (2, 3)])
    d = d.with_row_index("i").join(pl.DataFrame({"i": np.asarray(idx, dtype=np.uint32)}), on="i", how="semi").sort("i")
    return d["i"].to_numpy(), (d["business_name"].fill_null("") + " | " + d["business_address"].fill_null("")).to_list()


def tokenize(tok, ids, txt, n):
    """-> {index: np.int32 token ids}, truncated to n tokens, no special tokens."""
    out = {}
    for a in range(0, len(txt), 200_000):
        enc = tok(txt[a:a + 200_000], add_special_tokens=False, truncation=True, max_length=n)["input_ids"]
        out.update((int(i), np.asarray(e, dtype=np.int32)) for i, e in zip(ids[a:a + 200_000], enc))
    return out


class Enc:
    def __init__(self, tok, split, U):
        rows = np.unique(np.concatenate(U["ri"].to_list()))
        self.s1 = tokenize(tok, *texts(split, U["si"].to_numpy(), 1), S1_TOK)
        self.r = tokenize(tok, *texts(split, rows, 2), ROW_TOK)
        self.cls, self.sep, self.pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    def seq(self, si, ris):
        parts, pos, n = [[self.cls], self.s1[si], [self.sep]], [], 2 + len(self.s1[si])
        for ri in ris:
            pos.append(n)
            parts += [self.r[ri], [self.sep]]
            n += len(self.r[ri]) + 1
        return np.concatenate(parts).astype(np.int64), pos

    def batch(self, sis, riss):
        seqs = [self.seq(s, r) for s, r in zip(sis, riss)]
        L = max(len(x) for x, _ in seqs)
        ids = np.full((len(seqs), L), self.pad, dtype=np.int64)
        pos = np.zeros((len(seqs), M_ROWS), dtype=np.int64)
        valid = np.zeros((len(seqs), M_ROWS), dtype=bool)
        for i, (x, p) in enumerate(seqs):
            ids[i, :len(x)] = x
            pos[i, :len(p)], valid[i, :len(p)] = p, True
        t = lambda a: torch.from_numpy(a).to(DEV)
        return t(ids), t((ids != self.pad).astype(np.int64)), t(pos), t(valid)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BASE)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, mask, pos):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(torch.gather(h, 1, pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))).squeeze(-1)


def run(fold):
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED + fold)
    tok = AutoTokenizer.from_pretrained(BASE)
    gt = gt_idx().with_columns(pl.lit(1, pl.Int8).alias("y"))
    tr = units("train", pl.read_parquet(w("oof_m.parquet"), columns=["ri", "si", "p"]))
    te = units("test", pl.read_parquet(w("pred_test_m.parquet"), columns=["ri", "si", "p"]))
    fit = tr.filter(pl.col("fold") == fold)
    fit = fit.sample(min(SMOKE or N_UNITS, fit.height), seed=SEED)
    tr_s, te_s = tr.filter(pl.col("fold") != fold), te.filter(pl.col("fold") != fold)
    if SMOKE:
        tr_s, te_s = tr_s.head(SMOKE), te_s.head(SMOKE)
    lab = fit.explode("ri").join(gt, on=["si", "ri"], how="left").with_columns(pl.col("y").fill_null(0)) \
             .group_by("si", maintain_order=True).agg("y")
    fit = fit.join(lab, on="si", how="left", maintain_order="left")
    etr = Enc(tok, "train", pl.concat([fit.select("si", "ri"), tr_s.select("si", "ri")]))
    ete = Enc(tok, "test", te_s)
    print(f"fold {fold}: fit {fit.height} units ({fit['y'].list.sum().sum()} pos), score train {tr_s.height} test {te_s.height} "
          f"| tokenised {time.time() - t0:.0f}s", flush=True)
    net = Net().to(DEV)
    par = torch.nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    steps = (fit.height + BS - 1) // BS
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    order = rng.permutation(fit.height)
    si_all, ri_all, y_all = fit["si"].to_list(), fit["ri"].to_list(), fit["y"].to_list()
    par.train()
    for step, a in enumerate(range(0, fit.height, BS)):
        b = order[a:a + BS]
        ids, mask, pos, valid = etr.batch([si_all[i] for i in b], [ri_all[i] for i in b])
        y = torch.zeros(valid.shape, device=DEV)
        for j, i in enumerate(b):
            y[j, :len(y_all[i])] = torch.tensor(y_all[i], dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            logit = par(ids, mask, pos)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float()[valid], y[valid])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        if step % 500 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    par.eval()
    for split, U, enc in (("train", tr_s, etr), ("test", te_s, ete)):
        L = np.array([len(enc.s1[s]) + sum(len(enc.r[r]) for r in rs) for s, rs in zip(U["si"].to_list(), U["ri"].to_list())])
        order, si_l, ri_l = np.argsort(L), U["si"].to_list(), U["ri"].to_list()
        out_si, out_ri, out_s = [], [], []
        with torch.no_grad():
            for a in range(0, U.height, BS * 4):
                b = order[a:a + BS * 4]
                ids, mask, pos, valid = enc.batch([si_l[i] for i in b], [ri_l[i] for i in b])
                with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                    lg = par(ids, mask, pos).float().cpu().numpy()
                for j, i in enumerate(b):
                    n = len(ri_l[i])
                    out_si += [si_l[i]] * n
                    out_ri += ri_l[i]
                    out_s.append(lg[j, :n])
        pl.DataFrame({"ri": np.asarray(out_ri, dtype=np.int32), "si": np.asarray(out_si, dtype=np.int32),
                      "ce": np.concatenate(out_s).astype(np.float32)}).write_parquet(w(f"{TAG}_{split}_part{fold}.parquet"))
        print(f"scored {split}: {len(out_ri)} pairs {time.time() - t0:.0f}s", flush=True)
    if SMOKE:
        s = pl.read_parquet(w(f"{TAG}_train_part{fold}.parquet")).join(gt, on=["si", "ri"], how="left").with_columns(pl.col("y").fill_null(0))
        from sklearn.metrics import roc_auc_score
        print("smoke train OOF AUC", round(roc_auc_score(s["y"].to_numpy(), s["ce"].to_numpy()), 4) if s["y"].n_unique() > 1 else "n/a", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[2]))
