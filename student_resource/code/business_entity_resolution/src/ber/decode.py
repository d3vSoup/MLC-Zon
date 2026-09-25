import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from ber.io import CFG, path, read_truth, read_tsv, source_file, timed, to_key, validate, work, write_id_lists
from ber.metric import f05
from ber.models.lgbm import fold_map

NAME = CFG["decode"]["model"]

TAUS = np.round(np.arange(0.01, 1.0, 0.01), 2)
TAU_G = np.round(np.arange(0.35, 0.851, 0.01), 2)
TAU_2 = np.round(np.arange(0.45, 0.951, 0.01), 2)
L = 20  # expected-F looks at the top L owned candidates per S1
GATE_FEATS = ["q1", "q2", "q_sum", "n_q50", "n_owned", "n_cand", "name_freq_log", "name_freq_share"]
GATE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                   feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                   seed=CFG["seed"], deterministic=True, force_row_wise=True, num_threads=CFG["n_jobs"])


# keep each S2/S3 record only on its best S1 (one-owner rule is exact in train)
def owner(s1, cand, p):
    o = np.lexsort((-p, cand))
    first = np.r_[True, cand[o][1:] != cand[o][:-1]]
    return o[first]


# all S1 of a split: keys, country, ids
def s1_table(split):
    d = read_tsv(source_file(split, 1), usecols=["entity_id", "country"])
    return to_key(d["entity_id"]), d["country"].values, d["entity_id"].values


# preds -> owned pairs (code, cand, p[, label]) + per-S1 info
def owned(split):
    pr = pq.read_table(work("preds", f"{'oof' if split == 'train' else 'test'}_{NAME}.parquet")).to_pandas()
    keys, country, ids = s1_table(split)
    n = len(keys)
    code_all = pd.Index(keys).get_indexer(pr["s1"].values)
    assert (code_all >= 0).all()
    info = {"keys": keys, "country": country, "ids": ids, "n_cand": np.bincount(code_all, minlength=n)}
    keep = owner(pr["s1"].values, pr["cand"].values, pr["p"].values)
    d = pd.DataFrame({"code": code_all[keep], "cand": pr["cand"].values[keep], "p": pr["p"].values[keep]})
    if split == "train":
        cf = work("train", "cands.parquet")
        lab = pq.read_table(cf, columns=["label"]).column("label").to_numpy()
        head = pq.ParquetFile(cf).read_row_group(0, columns=["s1", "cand"]).to_pandas()
        assert len(lab) == len(pr) and (pr["s1"].values[:len(head)] == head["s1"].values).all() \
            and (pr["cand"].values[:len(head)] == head["cand"].values).all(), "preds not row-aligned with cands"
        info["g_cand"] = np.bincount(code_all[lab == 1], minlength=n)  # true matches reachable
        d["label"] = lab[keep]
        _, long = read_truth()
        info["G"] = np.bincount(pd.Index(keys).get_indexer(to_key(long["s1_id"])), minlength=n)
        fm = fold_map()
        info["fold"] = fm["fold"].reindex(keys).values
    # rank inside each S1 by p (desc)
    d = d.iloc[np.lexsort((-d["p"].values, d["code"].values))].reset_index(drop=True)
    start = np.searchsorted(d["code"].values, d["code"].values, "left")
    d["rank"] = np.arange(len(d)) - start
    return d, info


def iso():
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")


# isotonic calibration, cross-fitted by S1 fold so OOF q stays honest
def calibrate_oof(d, fold):
    f = fold[d["code"].values]
    q = np.empty(len(d))
    for k in range(CFG["n_folds"]):
        m = f == k
        q[m] = iso().fit(d["p"].values[~m], d["label"].values[~m]).predict(d["p"].values[m])
    full = iso().fit(d["p"].values, d["label"].values)
    pickle.dump(full, open(work("models", f"iso_{NAME}.pkl"), "wb"))
    return q


# per-S1 gate features from calibrated q (no labels)
def gate_frame(d, info, split):
    n = len(info["keys"])
    code, q, r = d["code"].values, d["q"].values, d["rank"].values
    g = pd.DataFrame({"q1": np.zeros(n), "q2": np.zeros(n)})
    g.loc[code[r == 0], "q1"] = q[r == 0]
    g.loc[code[r == 1], "q2"] = q[r == 1]
    g["q_sum"] = np.bincount(code, q, n)
    g["n_q50"] = np.bincount(code[q > 0.5], minlength=n)
    g["n_owned"] = np.bincount(code, minlength=n)
    g["n_cand"] = info["n_cand"]
    core = pd.read_parquet(work(split, "norm_s1.parquet"), columns=["name_core"])["name_core"]
    cnt = core.map(core.value_counts()).values
    g["name_freq_log"] = np.log1p(cnt)
    g["name_freq_share"] = cnt / n
    return g


# P(S1 has any true match), OOF by fold; saves fold models for test
def fit_gate(g, y, fold):
    out = np.empty(len(g))
    for k in range(CFG["n_folds"]):
        t, v = fold != k, fold == k
        m = lgb.train(GATE_PARAMS, lgb.Dataset(g.loc[t, GATE_FEATS], y[t]), 2000,
                      valid_sets=[lgb.Dataset(g.loc[v, GATE_FEATS], y[v])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        m.save_model(work("models", f"gate_{NAME}_fold{k}.txt"))
        out[v] = m.predict(g.loc[v, GATE_FEATS])
    return out


def gate_test(g):
    ms = [lgb.Booster(model_file=str(work("models", f"gate_{NAME}_fold{k}.txt"))) for k in range(CFG["n_folds"])]
    return np.mean([m.predict(g[GATE_FEATS]) for m in ms], axis=0)


# decision rule: gate >= tau_g -> top-1, plus rank>=1 while q > tau_2; returns chosen-row mask
def rule_mask(d, gate, tau_g, tau_2):
    ok = gate[d["code"].values] >= tau_g
    return ok & ((d["rank"].values == 0) | (d["q"].values > tau_2))


# exact E[F0.5 | predict top k] under independence, for q sorted desc, shape (N, L) -> (N, L+1)
def expected_f(Q):
    n, l = Q.shape
    A = [np.ones((n, 1))]  # hits inside top-k
    for k in range(l):
        a = A[-1]
        nxt = np.zeros((n, k + 2))
        nxt[:, :-1] += a * (1 - Q[:, k:k + 1])
        nxt[:, 1:] += a * Q[:, k:k + 1]
        A.append(nxt)
    B = [None] * (l + 1)  # hits outside top-k
    B[l] = np.ones((n, 1))
    for k in range(l - 1, -1, -1):
        b = B[k + 1]
        nxt = np.zeros((n, b.shape[1] + 1))
        nxt[:, :-1] += b * (1 - Q[:, k:k + 1])
        nxt[:, 1:] += b * Q[:, k:k + 1]
        B[k] = nxt
    E = np.zeros((n, l + 1))
    E[:, 0] = B[0][:, 0]
    for k in range(1, l + 1):
        t = np.arange(k + 1)[:, None]
        r = np.arange(l - k + 1)[None, :]
        W = np.where(t >= 1, 1.25 * t / (k + 0.25 * (t + r)), 0.0)
        E[:, k] = np.einsum("nt,nr,tr->n", A[k], B[k], W)
    return E


# pick k per S1 by expected F; gate variant swaps E[F|0] for 1 - gate
def expf_mask(d, n, gate=None, chunk=100_000):
    code, r = d["code"].values, d["rank"].values
    top = r < L
    Q = np.zeros((n, L))
    Q[code[top], r[top]] = d["q"].values[top]
    best_k = np.zeros(n, dtype=np.int64)
    for i in range(0, n, chunk):
        E = expected_f(Q[i:i + chunk])
        if gate is not None:
            E[:, 0] = 1 - gate[i:i + chunk]
        best_k[i:i + chunk] = E.argmax(axis=1)
    return r < best_k[code]


def score(d, info, mask):
    n = len(info["keys"])
    code = d["code"].values[mask]
    k = np.bincount(code, minlength=n)
    t = np.bincount(code[d["label"].values[mask] == 1], minlength=n)
    return k, t, f05(info["G"], k, t)


def summary(f, info):
    G, country, fold = info["G"], info["country"], info["fold"]
    out = {"overall": f.mean(), "singleton_only": f[G == 0].mean()}
    for c in pd.unique(country):
        out[c] = f[country == c].mean()
    out["fold_std"] = np.std([f[fold == k].mean() for k in range(CFG["n_folds"])])
    return {k: round(float(v), 5) for k, v in out.items()}


# 2D grid for the gate rule; plateau (within 0.002 of best) point nearest its centroid
def grid_rule(d, info, gate):
    n, fold = len(info["keys"]), info["fold"]
    G = info["G"]
    code, lab, q, r = d["code"].values, d["label"].values, d["q"].values, d["rank"].values
    k1 = np.bincount(code[r == 0], minlength=n)
    t1 = np.bincount(code[(r == 0) & (lab == 1)], minlength=n)
    empty = f05(G, 0, 0)
    S = np.zeros((len(TAU_G), len(TAU_2)))
    SF = np.zeros((CFG["n_folds"], len(TAU_G), len(TAU_2)))
    fsize = np.bincount(fold, minlength=CFG["n_folds"])
    for j, t2 in enumerate(TAU_2):
        ex = (r > 0) & (q > t2)
        k = k1 + np.bincount(code[ex], minlength=n)
        t = t1 + np.bincount(code[ex & (lab == 1)], minlength=n)
        fp = f05(G, k, t)
        for i, tg in enumerate(TAU_G):
            f = np.where(gate >= tg, fp, empty)
            S[i, j] = f.mean()
            SF[:, i, j] = np.bincount(fold, f, CFG["n_folds"]) / fsize
    best = S.max()
    pi, pj = np.nonzero(S >= best - 0.002)
    ci, cj = pi.mean(), pj.mean()
    w = np.argmin((pi - ci) ** 2 + (pj - cj) ** 2)
    i, j = pi[w], pj[w]
    per_fold = [np.unravel_index(SF[k].argmax(), S.shape) for k in range(CFG["n_folds"])]
    return {"tau_g": float(TAU_G[i]), "tau_2": float(TAU_2[j]), "grid_best": round(float(best), 5),
            "plateau_tau_g": [float(TAU_G[pi.min()]), float(TAU_G[pi.max()])],
            "plateau_tau_2": [float(TAU_2[pj.min()]), float(TAU_2[pj.max()])],
            "fold_best_tau_g_std": round(float(np.std([TAU_G[a] for a, _ in per_fold])), 4),
            "fold_best_tau_2_std": round(float(np.std([TAU_2[b] for _, b in per_fold])), 4)}


# v0 single threshold on q (plateau centre)
def grid_single(d, info):
    scores = np.array([score(d, info, d["q"].values >= t)[2].mean() for t in TAUS])
    plateau = TAUS[scores >= scores.max() - 0.002]
    return float(np.round(np.median(plateau), 2))


# phase 5: calibrate, gate, compare decoders on OOF, keep the best
def run(force=False):
    with timed("decode oof"):
        d, info = owned("train")
        n = len(info["keys"])
        d["q"] = calibrate_oof(d, info["fold"])
        g = gate_frame(d, info, "train")
        gate = fit_gate(g, (info["G"] > 0).astype(int), info["fold"])
        res, masks = {}, {}
        tau = grid_single(d, info)
        masks["single_tau"] = d["q"].values >= tau
        res["single_tau"] = {"tau": tau}
        rp = grid_rule(d, info, gate)
        masks["gate_tau2"] = rule_mask(d, gate, rp["tau_g"], rp["tau_2"])
        res["gate_tau2"] = rp
        masks["expf"] = expf_mask(d, n)
        res["expf"] = {}
        masks["expf_gate"] = expf_mask(d, n, gate)
        res["expf_gate"] = {}
        for name, m in masks.items():
            res[name]["oof"] = summary(score(d, info, m)[2], info)
        best = max(res, key=lambda k: res[k]["oof"]["overall"])
        out = {"chosen": best, "decoders": res, "all_empty": round(float((info["G"] == 0).mean()), 5),
               "pc_after_block": round(float(info["g_cand"].sum() / info["G"].sum()), 5)}
        print(json.dumps(out, indent=1))
        json.dump(out, open(work("reports", f"decode_{NAME}.json"), "w"), indent=1)
        k, t, _ = score(d, info, masks[best])
        np.savez(work("preds", f"oof_decoded_{NAME}.npz"), k=k, t=t, G=info["G"], g_cand=info["g_cand"],
                 country=info["country"])


# apply the chosen decoder to test; write both TSVs; official validator
def write(force=False):
    cfg = json.load(open(work("reports", f"decode_{NAME}.json")))
    best, par = cfg["chosen"], cfg["decoders"][cfg["chosen"]]
    d, info = owned("test")
    n = len(info["keys"])
    d["q"] = pickle.load(open(work("models", f"iso_{NAME}.pkl"), "rb")).predict(d["p"].values)
    gate = gate_test(gate_frame(d, info, "test"))
    mask = {"single_tau": lambda: d["q"].values >= par.get("tau", 1),
            "gate_tau2": lambda: rule_mask(d, gate, par.get("tau_g"), par.get("tau_2")),
            "expf": lambda: expf_mask(d, n),
            "expf_gate": lambda: expf_mask(d, n, gate)}[best]()
    pr = pq.read_table(work("preds", f"test_{NAME}.parquet"), columns=["s1", "cand"]).to_pandas()
    valid = np.concatenate([to_key(read_tsv(source_file("test", s), usecols=["entity_id"])["entity_id"])
                            for s in (2, 3)])
    s1m, candm = info["keys"][d["code"].values[mask]], d["cand"].values[mask]
    out = path("output_dir")
    with timed("write tsvs"):
        write_id_lists(out / "candidate_pairs.tsv", "candidate_entity_ids", info["ids"],
                       pr["s1"].values, pr["cand"].values, valid)
        write_id_lists(out / "matching_results.tsv", "matched_entity_ids", info["ids"], s1m, candm, valid)
    has = np.bincount(d["code"].values[mask], minlength=n) > 0
    print(f"decoder {best}; test pairs {len(pr)}, matched {mask.sum()}, S1 with a match {has.sum()}/{n}")
    for c in pd.unique(info["country"]):
        m = info["country"] == c
        print(f"  {c}: empty share {1 - has[m].mean():.4f}, mean gate {gate[m].mean():.4f}, "
              f"mean top-1 q {np.bincount(d['code'].values[d['rank'].values == 0], d['q'].values[d['rank'].values == 0], n)[m].mean():.4f}")
    validate(out / "matching_results.tsv", out / "candidate_pairs.tsv")
