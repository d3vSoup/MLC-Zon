import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.io import CFG, path, read_truth, read_tsv, source_file, timed, to_key, validate, work, write_id_lists
from ber.metric import f05
from ber.models.lgbm import NAME

TAUS = np.round(np.arange(0.01, 1.0, 0.01), 2)


# keep each S2/S3 record only on its best S1 (one-owner rule is exact in train)
def owner(s1, cand, p):
    o = np.lexsort((-p, cand))
    first = np.r_[True, cand[o][1:] != cand[o][:-1]]
    return o[first]


# all S1 of a split with country, as (keys, country)
def s1_table(split):
    d = read_tsv(source_file(split, 1), usecols=["entity_id", "country"])
    return to_key(d["entity_id"]), d["country"].values, d["entity_id"].values


# OOF pairs after owner rule, with labels and per-S1 truth counts
def oof_frame():
    pr = pq.read_table(work("preds", f"oof_{NAME}.parquet")).to_pandas()
    cf = work("train", "cands.parquet")
    lab = pq.read_table(cf, columns=["label"]).column("label").to_numpy()
    head = pq.ParquetFile(cf).read_row_group(0, columns=["s1", "cand"]).to_pandas()
    assert len(lab) == len(pr) and (pr["s1"].values[:len(head)] == head["s1"].values).all() \
        and (pr["cand"].values[:len(head)] == head["cand"].values).all(), "preds not row-aligned with cands"
    keys, country, _ = s1_table("train")
    idx = pd.Index(keys)
    code_all = idx.get_indexer(pr["s1"].values)
    g_cand = np.bincount(code_all[lab == 1], minlength=len(keys))  # true matches reachable
    keep = owner(pr["s1"].values, pr["cand"].values, pr["p"].values)
    d = pd.DataFrame({"code": code_all[keep], "cand": pr["cand"].values[keep], "p": pr["p"].values[keep],
                      "label": lab[keep]})
    _, long = read_truth()
    G = np.bincount(idx.get_indexer(to_key(long["s1_id"])), minlength=len(keys))
    return d, G, g_cand, country


# per-S1 (k, t) for a threshold
def counts(d, n, tau):
    m = d["p"].values >= tau
    k = np.bincount(d["code"].values[m], minlength=n)
    t = np.bincount(d["code"].values[m & (d["label"].values == 1)], minlength=n)
    return k, t


def summary(f, G, country):
    out = {"overall": f.mean(), "singleton_only": f[G == 0].mean()}
    for c in pd.unique(country):
        out[c] = f[country == c].mean()
    return {k: round(float(v), 5) for k, v in out.items()}


# grid tau on OOF macro F0.5, pick plateau centre
def run(force=False):
    with timed("decode oof"):
        d, G, g_cand, country = oof_frame()
        n = len(G)
        scores = np.array([f05(G, *counts(d, n, t)).mean() for t in TAUS])
        best = scores.max()
        plateau = TAUS[scores >= best - 0.002]
        tau = float(np.round(np.median(plateau), 2))
        k, t = counts(d, n, tau)
        f = f05(G, k, t)
        res = {"tau": tau, "best_tau": float(TAUS[scores.argmax()]), "best": round(float(best), 5),
               "plateau": [float(plateau.min()), float(plateau.max())],
               "all_empty": round(float((G == 0).mean()), 5),
               "pc_after_block": round(float(g_cand.sum() / G.sum()), 5),
               "oof": summary(f, G, country)}
        print(json.dumps(res, indent=1))
        json.dump(res, open(work("reports", f"decode_{NAME}.json"), "w"), indent=1)
        np.savez(work("preds", f"oof_decoded_{NAME}.npz"), k=k, t=t, G=G, g_cand=g_cand, country=country)


# test: owner rule + tau -> both TSVs, then official validator
def write(force=False):
    tau = json.load(open(work("reports", f"decode_{NAME}.json")))["tau"]
    pr = pq.read_table(work("preds", f"test_{NAME}.parquet")).to_pandas()
    _, _, order = s1_table("test")
    valid = np.concatenate([to_key(read_tsv(source_file("test", s), usecols=["entity_id"])["entity_id"])
                            for s in (2, 3)])
    s1, cand, p = pr["s1"].values, pr["cand"].values, pr["p"].values
    keep = owner(s1, cand, p)
    keep = keep[p[keep] >= tau]
    out = path("output_dir")
    with timed("write tsvs"):
        write_id_lists(out / "candidate_pairs.tsv", "candidate_entity_ids", order, s1, cand, valid)
        write_id_lists(out / "matching_results.tsv", "matched_entity_ids", order, s1[keep], cand[keep], valid)
    print(f"test pairs {len(pr)}, matched {len(keep)}, S1 with a match {len(np.unique(s1[keep]))}/{len(order)}")
    validate(out / "matching_results.tsv", out / "candidate_pairs.tsv")
