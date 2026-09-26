import json
import pickle

import numpy as np
import pandas as pd

import ber.decode as D
from ber.io import CFG, read_tsv, source_file, work

# label-free test-time calibration (port of the QTpies calibrate.py, AUDIT 3.9) on calibrated q:
# seen countries: difference-in-differences prior shift vs the least-shifted seen country (anchor, k = 1);
# countries absent from train: odds scaled so their empty-S1 rate = seen countries' rate + DELTA (open-set rule).
EDGES = 1 / (1 + np.exp(-np.linspace(-4.6, 9.2, 47)))  # q 0.01 .. 0.9999, 0.3 logit wide
MID, DEC, DELTA = (0.3, 0.93), (0.9, 0.995), 0.0015


def odds(q, k):
    return q * k / (q * k + 1 - q)


def band(b):
    lo, hi = np.r_[0, EDGES], np.r_[EDGES, 1]
    return (lo >= b[0] - 1e-9) & (hi <= b[1] + 1e-9)


# per band of the records' owner pairs: mean test q, test rows, test/train rows-per-S1 ratio
def profile(q_te, n_te, q_tr, n_tr):
    k = len(EDGES) + 1
    b_te, b_tr = np.digitize(q_te, EDGES), np.digitize(q_tr, EDGES)
    rows_te, rows_tr = np.bincount(b_te, minlength=k), np.bincount(b_tr, minlength=k)
    qbar = np.bincount(b_te, q_te, k) / np.maximum(rows_te, 1)
    return qbar, rows_te, (rows_te / n_te) / np.maximum(rows_tr / n_tr, 1e-12)


def excess(pr):
    m = band(MID) & (pr[1] > 0)
    return float(np.average(pr[2][m], weights=pr[1][m]))


def shift_k(c, ref):
    qbar, rows, r = c
    ok = band(DEC) & (rows >= 200) & (ref[1] >= 200)
    prec = np.clip(qbar[ok] * ref[2][ok] / r[ok], 1e-4, 1 - 1e-4)
    lk = np.minimum(0, np.log(prec / (1 - prec)) - np.log(qbar[ok] / (1 - qbar[ok])))  # a deficit is no evidence
    print("   band k: " + " ".join(f"{a:.3f}:{b:.3f}" for a, b in zip(qbar[ok], np.exp(lk))), flush=True)
    return min(1.0, float(np.exp(np.average(lk, weights=rows[ok])))) if ok.any() else 1.0


# anchor + k per seen country from owner-pair q of "test" vs "train" (either split, or two train folds)
def did(q_te, c_te, n_te, q_tr, c_tr, n_tr):
    prof = {c: profile(q_te[c_te == c], n_te[c], q_tr[c_tr == c], n_tr[c]) for c in n_tr}
    anchor = min(prof, key=lambda c: excess(prof[c]))
    ks = {}
    for c in prof:
        ks[c] = 1.0 if c == anchor else shift_k(prof[c], prof[anchor])
        print(f"{c}: mid-band excess {excess(prof[c]):.3f}, k={ks[c]:.3f}" + (" (anchor)" if c == anchor else ""), flush=True)
    return anchor, ks


# owned pairs + info restricted to S1 in keep (codes renumbered; gate features keep full-split frequencies)
def subset(d, info, keep):
    d = d[keep[d["code"].values]].reset_index(drop=True)
    d["code"] = (np.cumsum(keep) - 1)[d["code"].values]
    return d, {k: v[keep] for k, v in info.items()}


def empty_rate(d, info, k, folds):
    d = d.assign(q=odds(d["q0"].values, k))
    gate = D.gate_test(D.gate_frame(d, info, "test"), folds)
    mask = D.expf_mask(d, len(info["keys"]), gate)
    return 1 - (np.bincount(d["code"].values[mask], minlength=len(info["keys"])) > 0).mean()


# null test (train OOF fold 1 as "test" vs fold 0 as "train": no shift, k must be ~1) + k per country on test
# load(split) -> (owned pairs with p, info); default = this run's decode inputs (the blend passes its own)
def fit(load=None):
    load = load or D.owned
    cfg = json.load(open(work("reports", f"decode_{D.NAME}.json")))
    folds = cfg.get("folds", list(range(CFG["n_folds"])))
    dt, it = load("train")
    f = it["fold"][dt["code"].values]
    q = np.empty(len(dt))
    for k in np.unique(f):  # cross-fit isotonic, as decode does (OOF q honest)
        m = f == k
        q[m] = D.iso().fit(dt["p"].values[~m], dt["label"].values[~m]).predict(dt["p"].values[m])
    c_tr = it["country"][dt["code"].values]
    seen = sorted(set(it["country"]))
    n_fold = {k: {c: int(((it["fold"] == k) & (it["country"] == c)).sum()) for c in seen} for k in (0, 1)}
    print("null test (fold 1 vs fold 0):", flush=True)
    _, null = did(q[f == 1], c_tr[f == 1], n_fold[1], q[f == 0], c_tr[f == 0], n_fold[0])
    n_tr = {c: int((it["country"] == c).sum()) for c in seen}
    q_tr = q
    del dt, it, q
    d, info = load("test")
    d["q0"] = pickle.load(open(work("models", f"iso_{D.NAME}.pkl"), "rb")).predict(d["p"].values)
    c_te = info["country"][d["code"].values]
    n_te = {c: int((info["country"] == c).sum()) for c in set(info["country"])}
    print("test vs train OOF:", flush=True)
    anchor, ks = did(d["q0"].values[np.isin(c_te, seen)], c_te[np.isin(c_te, seen)], n_te, q_tr, c_tr, n_tr)
    emp = {}
    for c in seen:
        dc, ic = subset(d, info, info["country"] == c)
        emp[c] = empty_rate(dc, ic, ks[c], folds)
    ref = sum(emp[c] * n_te[c] for c in seen) / sum(n_te[c] for c in seen)
    for c in sorted(set(info["country"]) - set(seen)):  # unseen: continuous k by bisection on log k
        dc, ic = subset(d, info, info["country"] == c)
        e1, target = empty_rate(dc, ic, 1.0, folds), ref + DELTA
        lo, hi = np.log(0.005), 0.0
        if e1 >= target:
            ks[c] = 1.0
        else:
            for _ in range(18):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if empty_rate(dc, ic, np.exp(mid), folds) > target else (lo, mid)
            ks[c] = float(np.exp(hi))
        emp[c] = empty_rate(dc, ic, ks[c], folds)
        print(f"{c}: seen empty {ref:.4f}, target {target:.4f}, empty at k=1 {e1:.4f}, k={ks[c]:.4f} -> {emp[c]:.4f}", flush=True)
    out = {"anchor": anchor, "k": ks, "null_k": null, "empty": emp, "seen_empty": ref, "delta": DELTA}
    json.dump(out, open(work("reports", f"calibrate_{D.NAME}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out
