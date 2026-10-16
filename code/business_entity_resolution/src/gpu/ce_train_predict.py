# pip install -q transformers==5.17.0   (torch comes with the Kaggle GPU image; the script prints its version)
# Cross-encoder (xlm-roberta-base, MIT) for business-pair matching. Train on one half of train S1, then score the
# other half's band + the test band. Resumable: checkpoint every 30 min, predictions written in parts.
#
#   python ce_train_predict.py --half 1                     # Ankit: train H1, predict H2 band + test band
#   python ce_train_predict.py --half 2                     # Soup:  train H2, predict H1 band + test band
#   python ce_train_predict.py --half 1 --time_only         # ~1% timing run, prints the full-run ETA
#   python ce_train_predict.py --half 1 --resume_dir /kaggle/input/<previous-run-output>
import argparse
import glob
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

KAGGLE = os.path.exists("/kaggle/input")


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) if KAGGLE else []
    return hits[0] if hits else os.path.join(os.environ.get("CE_DATA", "../../../work/ce"), name)


def args_():
    a = argparse.ArgumentParser()
    a.add_argument("--half", type=int, choices=[1, 2], required=True)
    a.add_argument("--model", default="xlm-roberta-base")
    a.add_argument("--lr", type=float, default=2e-5)
    a.add_argument("--bs", type=int, default=32)
    a.add_argument("--pred_bs", type=int, default=512)
    a.add_argument("--max_len", type=int, default=128)
    a.add_argument("--epochs", type=int, default=1)
    a.add_argument("--out_dir", default="/kaggle/working" if KAGGLE else "../../../work/ce/out")
    a.add_argument("--resume_dir", default=None)
    a.add_argument("--ckpt_min", type=float, default=30)
    a.add_argument("--max_hours", type=float, default=11.0, help="stop cleanly before Kaggle's 12h limit")
    a.add_argument("--part", type=int, default=200_000, help="pairs per prediction part file")
    a.add_argument("--time_only", action="store_true")
    a.add_argument("--seed", type=int, default=42)
    return a.parse_args()


class Clock:
    def __init__(self, hours):
        self.t0, self.limit = time.time(), hours * 3600

    def left(self):
        return self.limit - (time.time() - self.t0)


def enc(tok, a, b, max_len, dev):
    x = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(dev, non_blocking=True) for k, v in x.items()}


def train(a, tok, model, dev, clock):
    ck = os.path.join(a.out_dir, "ckpt", "train.pt")
    final = os.path.join(a.out_dir, "model")
    if os.path.exists(os.path.join(final, "done")):
        return
    d = pd.read_parquet(find("ce_train_pairs.parquet"), columns=["text_a", "text_b", "label", "half"])
    d = d[d.half == a.half].sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    if a.time_only:
        d = d.iloc[: max(len(d) // 100, a.bs * 20)]
    steps = (len(d) + a.bs - 1) // a.bs * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        mod = model.module if hasattr(model, "module") else model
        mod.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sch.load_state_dict(s["sch"])
        scaler.load_state_dict(s["scaler"]); step = s["step"]
        print(f"resumed training at step {step}/{steps}", flush=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    t_last, t0, s0 = time.time(), time.time(), step
    while step < steps:
        i = (step % ((len(d) + a.bs - 1) // a.bs)) * a.bs
        b = d.iloc[i:i + a.bs]
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**enc(tok, b.text_a, b.text_b, a.max_len, dev)).logits.squeeze(-1)
            loss = loss_fn(logit.float(), torch.tensor(b.label.values, dtype=torch.float32, device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        step += 1
        if not torch.isfinite(loss):
            raise SystemExit("loss is NaN/inf: lower --lr (e.g. 1e-5) and resume")
        if step % 200 == 0:
            rate = (step - s0) * a.bs / (time.time() - t0)
            print(f"train step {step}/{steps} loss {loss.item():.4f} {rate:.0f} pairs/s", flush=True)
        if not a.time_only and (time.time() - t_last > a.ckpt_min * 60 or clock.left() < 900 or step == steps):
            os.makedirs(os.path.dirname(ck), exist_ok=True)
            mod = model.module if hasattr(model, "module") else model
            torch.save({"model": mod.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                        "scaler": scaler.state_dict(), "step": step}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            t_last = time.time()
            if clock.left() < 900 and step < steps:
                raise SystemExit(f"time budget reached at step {step}/{steps}; checkpoint saved, re-run with --resume_dir")
    rate = (step - s0) * a.bs / max(time.time() - t0, 1e-9)
    print(f"training done: {steps} steps, {rate:.0f} pairs/s", flush=True)
    if a.time_only:
        return rate
    os.makedirs(final, exist_ok=True)
    (model.module if hasattr(model, "module") else model).save_pretrained(final)
    tok.save_pretrained(final)
    open(os.path.join(final, "done"), "w").write("ok")
    os.remove(ck)
    return rate


@torch.no_grad()
def predict(a, tok, model, dev, clock, which):
    path = find(f"ce_band_{which}.parquet")
    cols = ["pair_id", "text_a", "text_b"] + (["half"] if which == "train" else [])
    d = pd.read_parquet(path, columns=cols)
    if which == "train":
        d = d[d.half != a.half].drop(columns="half")  # score the half we did NOT train on
    d = d.reset_index(drop=True)
    if a.time_only:
        d = d.iloc[: len(d) // 100]
    out = os.path.join(a.out_dir, "pred_timing" if a.time_only else "pred")  # timing parts never mix with real ones
    os.makedirs(out, exist_ok=True)
    model.eval()
    t0, n_done = time.time(), 0
    for pi, lo in enumerate(range(0, len(d), a.part)):
        f = os.path.join(out, f"{which}_h{a.half}_part{pi:04d}.parquet")
        if os.path.exists(f):
            continue
        c = d.iloc[lo:lo + a.part]
        order = np.argsort((c.text_a.str.len() + c.text_b.str.len()).values)  # length-sorted -> less padding
        c = c.iloc[order]
        ps = []
        for j in range(0, len(c), a.pred_bs):
            b = c.iloc[j:j + a.pred_bs]
            with torch.autocast("cuda", dtype=torch.float16):
                ps.append(torch.sigmoid(model(**enc(tok, b.text_a, b.text_b, a.max_len, dev)).logits.squeeze(-1).float()).cpu().numpy())
        pd.DataFrame({"pair_id": c.pair_id.values, "ce_p": np.concatenate(ps).astype(np.float32)}) \
            .to_parquet(f + ".tmp", index=False)
        os.replace(f + ".tmp", f)
        n_done += len(c)
        rate = n_done / (time.time() - t0)
        print(f"predict {which} part {pi} ({lo + len(c)}/{len(d)}) {rate:.0f} pairs/s", flush=True)
        if clock.left() < 600:
            raise SystemExit(f"time budget reached during {which} prediction; parts saved, re-run with --resume_dir")
    return len(d), (n_done / (time.time() - t0) if n_done else None)


def main():
    a = args_()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    clock = Clock(a.max_hours)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.resume_dir:  # bring back checkpoint, finished model and finished prediction parts from the last run
        for sub in ("ckpt", "model", "pred"):
            src = os.path.join(a.resume_dir, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(a.out_dir, sub), dirs_exist_ok=True)
    dev = torch.device("cuda")
    print(f"torch {torch.__version__}, GPUs {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}", flush=True)
    final = os.path.join(a.out_dir, "model")
    src = final if os.path.exists(os.path.join(final, "done")) else a.model
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1).to(dev)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    tr_rate = train(a, tok, model, dev, clock)
    n_tr, r_tr = predict(a, tok, model, dev, clock, "train")
    n_te, r_te = predict(a, tok, model, dev, clock, "test")
    if a.time_only:
        full_train = len(pd.read_parquet(find("ce_train_pairs.parquet"), columns=["half"]).query(f"half == {a.half}"))
        full_pred = n_tr * 100 + n_te * 100
        eta = full_train / tr_rate + full_pred / ((r_tr + r_te) / 2)
        print(f"TIMING: train {tr_rate:.0f} pairs/s on {full_train} pairs; predict ~{(r_tr + r_te) / 2:.0f} pairs/s on "
              f"{full_pred} pairs -> full run ETA {eta / 3600:.2f} h", flush=True)
        return
    parts = sorted(glob.glob(os.path.join(a.out_dir, "pred", f"*_h{a.half}_part*.parquet")))
    print(f"ALL DONE: {len(parts)} prediction parts in {os.path.join(a.out_dir, 'pred')}", flush=True)


if __name__ == "__main__":
    main()
