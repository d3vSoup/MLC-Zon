import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.blocking.tfidf import SHIFT, docs, half_half, load, owners, topn, vectorizer
from ber.io import CFG, timed, work

KMAX = 30
KS = (10, 20, 30)
N_SAMPLE = 100_000  # random records per country, for the new-pair cost


def f05(c, g):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(g == 0, 1.0, 1.25 * c / (c + 0.25 * g))


# char 3-gram channel on name + address: typo-tolerant stand-in for a dense channel (no model, no GPU)
def char_channel(r, qq):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=0.01, min_df=2, dtype=np.float32,
                          sublinear_tf=True)
    R = vec.fit_transform(r["name_core"] + " " + r["addr_norm"])
    return topn(vec.transform(qq["name_core"] + " " + qq["addr_norm"]), R, KMAX)


# measure only: blocking misses recovered by bigger K per channel or a char channel, and new pairs each adds
def run():
    rng = np.random.default_rng(CFG["seed"])
    s1 = load("train", 1)
    q = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
    own = owners()
    found = pq.read_table(work("train", "cands.parquet"), columns=["cand"], filters=[("label", "==", 1)]) \
        .column(0).to_numpy()
    missed = own.index[~own.index.isin(found)].values
    g = own.value_counts().reindex(s1["key"].values, fill_value=0).values
    c_now = pd.Series(found).map(own).value_counts().reindex(s1["key"].values, fill_value=0).values
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["key"].values)
    rec = {}  # option -> recovered S1 keys (one per recovered missed record)
    new = {}  # option -> new pairs, extrapolated to all train records
    with Pool(CFG["n_jobs"]) as pool:
        for ctry in s1["country"].unique():
            r = s1[s1.country == ctry].reset_index(drop=True)
            qc = q[q.country == ctry]
            samp = rng.choice(qc["key"].values, min(N_SAMPLE, len(qc)), replace=False)
            qq = qc[qc["key"].isin(missed) | qc["key"].isin(samp)].reset_index(drop=True)
            old_df = pq.read_table(work("train", "cands.parquet"), columns=["s1", "cand"],
                                   filters=[("cand", "in", samp.tolist())]).to_pandas()
            scale = len(qc) / len(samp)
            with timed(f"probe {ctry} Q={len(qq)}"):
                vec = vectorizer()
                R = vec.fit_transform(docs(pool, r))
                hits = {"fw": topn(vec.transform(docs(pool, qq)), R, KMAX)}
                del R
                R, Q = half_half(pool, r, qq)
                hits["hh"] = topn(Q, R, KMAX)
                del R, Q
                hits["char"] = char_channel(r, qq)
            qk, rk = qq["key"].values, r["key"].values
            true_si = pd.Series(np.arange(len(r)), index=rk).reindex(own.reindex(qk).values).fillna(-1).values.astype(np.int64)
            is_miss, is_samp = np.isin(qk, missed), np.isin(qk, samp)
            mq = np.flatnonzero(is_miss & (true_si >= 0))
            truth = (true_si[mq] << SHIFT) | mq
            code = lambda n, kk: ((hits[n][1][hits[n][3] < kk].astype(np.int64) << SHIFT) | hits[n][0][hits[n][3] < kk])
            oi = pd.Series(np.arange(len(qq)), index=qk)
            old = (pd.Series(np.arange(len(r)), index=rk).reindex(old_df["s1"].values).values.astype(np.int64) << SHIFT) \
                | oi.reindex(old_df["cand"].values).values.astype(np.int64)
            base = np.union1d(code("fw", 10), code("hh", 10))
            bs = base[is_samp[base & (2 ** SHIFT - 1)]]
            print(f"{ctry}: probe base pairs on sample {len(bs)}, production {len(old)}, same {np.isin(bs, old).mean():.4f}")
            opts = {f"{n}_K{k}": [(n, k)] for n in ("fw", "hh") for k in KS[1:]}
            opts.update({"fw_hh_K20": [("fw", 20), ("hh", 20)], "fw_hh_K30": [("fw", 30), ("hh", 30)],
                         "char_K10": [("char", 10)], "char_K10+fw_hh_K20": [("char", 10), ("fw", 20), ("hh", 20)]})
            for o, parts in opts.items():
                got = np.union1d(base, np.concatenate([code(n, kk) for n, kk in parts]))
                hit = np.isin(truth, got)
                rec.setdefault(o, []).extend(rk[true_si[mq[hit]]].tolist())
                gq = got & (2 ** SHIFT - 1)
                new[o] = new.get(o, 0) + int((is_samp[gq] & ~np.isin(got, old)).sum()) * scale
    ceil0 = f05(c_now, g).mean()
    res = {"missed_true_pairs": int(len(missed)), "ceiling_now": round(float(ceil0), 5), "options": {}}
    for o in rec:
        c = c_now.copy()
        np.add.at(c, s1_pos.reindex(rec[o]).values, 1)
        res["options"][o] = {"recovered": len(rec[o]), "new_pairs_M": round(new[o] / 1e6, 2),
                             "ceiling_gain": round(float(f05(c, g).mean() - ceil0), 5)}
    json.dump(res, open(work("reports", "blocking_probe.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    run()
