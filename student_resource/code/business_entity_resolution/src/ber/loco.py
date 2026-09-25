import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import ber.decode as D
from ber.analysis import BUCKETS, buckets
from ber.features import graph
from ber.io import CFG, read_tsv, source_file, timed, to_key, work
from ber.metric import f05
from ber.models.lgbm import batches, fold_map

ID = ["s1", "cand"]


def params(rounds):
    p = {k: v for k, v in CFG["lgbm"].items() if k not in ("num_boost_round", "early_stopping", "test_models")}
    p.update(seed=CFG["seed"], deterministic=True, force_row_wise=True, num_threads=CFG["n_jobs"], verbose=-1,
             metric="binary_logloss")
    return p


# 5 fold models on rows of the training country; OOF for them, fold-0 model for the other country
def fit_predict(X, y, fold, is_src, rounds, feats):
    p = params(rounds)
    full = lgb.Dataset(X[is_src], y[is_src], feature_name=feats, params=p, free_raw_data=False).construct()
    f_src = fold[is_src]
    out = np.empty(len(X), np.float32)
    src_idx = np.flatnonzero(is_src)
    for k in range(CFG["n_folds"]):
        t, v = full.subset(np.flatnonzero(f_src != k)), full.subset(np.flatnonzero(f_src == k))
        m = lgb.train(p, t, rounds, valid_sets=[v], callbacks=[lgb.early_stopping(100, verbose=False)])
        out[src_idx[f_src == k]] = m.predict(X[src_idx[f_src == k]], num_threads=CFG["n_jobs"])
        if k == 0:
            m0 = m
    out[~is_src] = m0.predict(X[~is_src], num_threads=CFG["n_jobs"])
    return out


# round 1: train on sampled S1 of the source country (as in production), predict every train pair
def round1(src, ctry_of, tag):
    fm = fold_map()
    feats = [c for c in pq.ParquetFile(work("train", "feats.parquet")).schema_arrow.names if c not in ID]
    labels = pq.read_table(work("train", "cands.parquet"), columns=["label"]).column("label").to_numpy()
    s1 = pq.read_table(work("train", "feats.parquet"), columns=["s1"]).column("s1").to_numpy()
    is_src_all = ctry_of.reindex(s1).values == src
    samp = fm["sampled"].reindex(s1).values & is_src_all
    fold_all = fm["fold"].reindex(s1).values
    X = np.empty((int(samp.sum()), len(feats)), np.float32)
    off = row = 0
    for b in batches("train", feats):
        n = b.num_rows
        m = samp[off:off + n]
        X[row:row + m.sum()] = b.to_pandas().to_numpy(np.float32)[m]
        row += int(m.sum())
        off += n
    p = params(CFG["lgbm"]["num_boost_round"])
    full = lgb.Dataset(X, labels[samp], feature_name=feats, params=p, free_raw_data=True).construct()
    del X
    fs = fold_all[samp]
    models = []
    for k in range(CFG["n_folds"]):
        with timed(f"loco {tag} round1 fold {k}"):
            t, v = full.subset(np.flatnonzero(fs != k)), full.subset(np.flatnonzero(fs == k))
            models.append(lgb.train(p, t, CFG["lgbm"]["num_boost_round"], valid_sets=[v],
                                    callbacks=[lgb.early_stopping(100, verbose=False)]))
    del full
    out = np.empty(len(s1), np.float32)
    off = 0
    with timed(f"loco {tag} round1 predict"):
        for b in batches("train", feats):
            n = b.num_rows
            Xb = b.to_pandas()
            src_b, f_b = is_src_all[off:off + n], fold_all[off:off + n]
            pb = np.empty(n, np.float32)
            for k, m in enumerate(models):  # source country: own fold model (OOF); other country: fold-0 model
                w = (src_b & (f_b == k)) | (~src_b & (k == 0))
                if w.any():
                    pb[w] = m.predict(Xb[w], num_threads=CFG["n_jobs"])
            out[off:off + n] = pb
            off += n
    t = pq.read_table(work("train", "cands.parquet"), columns=["s1", "cand"])
    path = work("loco", tag, "oof_r1.parquet")
    pq.write_table(t.append_column("p", pa.array(out)), path)
    return path


# round 2 on hard pairs of the source country; full-length preds for decode
def round2(r1_path, src, ctry_of, tag, rounds):
    r2_path = work("loco", tag, "r2.parquet")
    graph.run("train", force=True, preds=r1_path, out=r2_path)
    d = pq.read_table(r2_path).to_pandas()
    feats = [c for c in d.columns if c not in ("s1", "cand", "row", "label")]
    is_src = ctry_of.reindex(d["s1"].values).values == src
    fold = fold_map()["fold"].reindex(d["s1"].values).values
    with timed(f"loco {tag} round2"):
        p2 = fit_predict(d[feats].to_numpy(np.float32), d["label"].values, fold, is_src, rounds, feats)
    t = pq.read_table(r1_path)
    p = t.column("p").to_numpy().copy()
    p[d["row"].values] = p2
    name = f"loco_{tag}"
    pq.write_table(t.set_column(2, "p", pa.array(p)), work("preds", f"oof_{name}.parquet"))
    return name


# calibration + gate fitted on the source country only; returns owned pairs (with q), info, gate, source mask
def prepare(name, src):
    D.NAME = name
    d, info = D.owned("train")
    n = len(info["keys"])
    is_src = info["country"] == src
    code = d["code"].values
    rs = is_src[code]
    d["q"] = np.nan
    q_src = np.empty(rs.sum())
    f = info["fold"][code][rs]
    for k in range(CFG["n_folds"]):  # cross-fit inside the source country
        m = f == k
        q_src[m] = D.iso().fit(d["p"].values[rs][~m], d["label"].values[rs][~m]).predict(d["p"].values[rs][m])
    iso = D.iso().fit(d["p"].values[rs], d["label"].values[rs])
    d.loc[rs, "q"] = q_src
    d.loc[~rs, "q"] = iso.predict(d["p"].values[~rs])
    g = D.gate_frame(d, info, "train")
    y = (info["G"] > 0).astype(int)
    gate = np.empty(n)
    gm = []
    for k in range(CFG["n_folds"]):
        t, v = is_src & (info["fold"] != k), is_src & (info["fold"] == k)
        m = lgb.train(D.GATE_PARAMS, lgb.Dataset(g.loc[t, D.GATE_FEATS], y[t]), 2000,
                      valid_sets=[lgb.Dataset(g.loc[v, D.GATE_FEATS], y[v])], callbacks=[lgb.early_stopping(100, verbose=False)])
        gate[v] = m.predict(g.loc[v, D.GATE_FEATS])
        gm.append(m)
    gate[~is_src] = np.mean([m.predict(g.loc[~is_src, D.GATE_FEATS]) for m in gm], axis=0)
    return d, info, gate, is_src


# decode with calibration + gate + decoder fitted on the source country only; score the target country
def evaluate(name, src, tag):
    d, info, gate, is_src = prepare(name, src)
    n = len(info["keys"])
    code = d["code"].values
    rs = is_src[code]
    mask = D.expf_mask(d, n, gate)
    k_, t_, f_ = D.score(d, info, mask)
    L = buckets(k_, t_, info["G"], info["g_cand"])
    res = {}
    for c in (src, "tgt"):
        m = is_src if c == src else ~is_src
        rows = rs if c == src else ~rs
        lab, qq, pm = d["label"].values[rows], d["q"].values[rows], mask[rows]
        bins = np.clip((qq * 10).astype(int), 0, 9)
        calib = {f"{b / 10:.1f}": [round(float(qq[bins == b].mean()), 3), round(float(lab[bins == b].mean()), 3), int((bins == b).sum())]
                 for b in range(10) if (bins == b).any()}
        tot = L[m][BUCKETS].sum()
        res[c if c == src else f"{tag.split('2')[1]} (unseen)"] = {
            "F0.5": round(float(f_[m].mean()), 5), "singleton_only": round(float(f_[m & (info["G"] == 0)].mean()), 5),
            "pred_pairs": int(pm.sum()), "true_pairs_in_cands": int(lab.sum()), "true_pairs_all": int(info["G"][m].sum()),
            "precision": round(float(lab[pm].mean()), 4), "pred_empty_share": round(float((k_[m] == 0).mean()), 4),
            "true_singleton_share": round(float((info["G"][m] == 0).mean()), 4),
            "mean_gate": round(float(gate[m].mean()), 4),
            "loss_buckets_%": (tot / tot.sum() * 100).round(1).to_dict(),
            "calibration [mean q, true rate, n] by q decile": calib}
    return res


def run(direction):
    src, tgt = {"us2in": ("US", "India"), "in2us": ("India", "US")}[direction]
    s1 = read_tsv(source_file("train", 1), usecols=["entity_id", "country"])
    ctry_of = pd.Series(s1["country"].values, index=to_key(s1["entity_id"]))
    work("loco", direction, "x").parent.mkdir(parents=True, exist_ok=True)
    r1 = round1(src, ctry_of, direction)
    name = round2(r1, src, ctry_of, direction, CFG["lgbm"]["num_boost_round"])
    res = evaluate(name, src, direction)
    json.dump(res, open(work("reports", f"loco_{direction}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    run(sys.argv[1])


# round-2 variant on an existing LOCO run: extra feature files and/or pseudo-labels on the unseen country
def variant(direction, extra=(), pseudo=False, rounds=None):
    src = {"us2in": "US", "in2us": "India"}[direction]
    rounds = rounds or CFG["lgbm"]["num_boost_round"]
    s1 = read_tsv(source_file("train", 1), usecols=["entity_id", "country"])
    ctry_of = pd.Series(s1["country"].values, index=to_key(s1["entity_id"]))
    d = pq.read_table(work("loco", direction, "r2.parquet")).to_pandas()
    rows = d["row"].values
    if extra:
        from ber.features import names2, numbers
        for e in extra:
            f = work("loco", direction, f"{e}.parquet")
            if not f.exists():
                numbers.run("train", force=True, rows=rows, out=f) if e == "num" else names2.run("train", rows, f)
            d = pd.concat([d, pq.read_table(f).to_pandas()], axis=1)
    feats = [c for c in d.columns if c not in ("s1", "cand", "row", "label")]
    is_src = ctry_of.reindex(d["s1"].values).values == src
    fold = fold_map()["fold"].reindex(d["s1"].values).values
    y = d["label"].values.copy()
    train_rows = is_src.copy()
    if pseudo:  # unseen side: confident round-2 predictions become labels (true labels never used there)
        p_base = pq.read_table(work("preds", f"oof_loco_{direction}.parquet")).column("p").to_numpy()[rows]
        pos, neg = ~is_src & (p_base > 0.95), ~is_src & (p_base < 0.02)
        y[pos], y[neg] = 1, 0
        train_rows |= pos | neg
        print(f"pseudo-labels: {int(pos.sum())} pos, {int(neg.sum())} neg of {int((~is_src).sum())} unseen rows")
    X = d[feats].to_numpy(np.float32)
    p = params(rounds)
    out = np.empty(len(d), np.float32)
    full = lgb.Dataset(X[train_rows], y[train_rows], feature_name=feats, params=p, free_raw_data=False).construct()
    tr_idx, f_tr = np.flatnonzero(train_rows), fold[train_rows]
    models = []
    for k in range(CFG["n_folds"]):  # folds by S1 across both countries: every row is predicted out-of-fold
        t, v = full.subset(np.flatnonzero(f_tr != k)), full.subset(np.flatnonzero(f_tr == k))
        m = lgb.train(p, t, rounds, valid_sets=[v], callbacks=[lgb.early_stopping(100, verbose=False)])
        out[fold == k] = m.predict(X[fold == k], num_threads=CFG["n_jobs"])
    t = pq.read_table(work("loco", direction, "oof_r1.parquet"))
    pp = t.column("p").to_numpy().copy()
    pp[rows] = out
    tag = f"{direction}_{'_'.join(extra) or 'base'}{'_pseudo' if pseudo else ''}"
    pq.write_table(t.set_column(2, "p", pa.array(pp)), work("preds", f"oof_loco_{tag}.parquet"))
    res = evaluate(f"loco_{tag}", src, direction)
    json.dump(res, open(work("reports", f"loco_{tag}.json"), "w"), indent=1)
    print(tag, {k: {x: v[x] for x in ("F0.5", "singleton_only", "precision")} for k, v in res.items()})
    work("preds", f"oof_loco_{tag}.parquet").unlink()
