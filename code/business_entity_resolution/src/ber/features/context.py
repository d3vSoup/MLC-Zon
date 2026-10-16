import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.io import CFG, work


# per-record and per-S1 best / second-best channel scores (no labels), one pass per stat
def score_stats(split, ia, ib):
    pf = pq.ParquetFile(work(split, "cands.parquet"))
    chans = [c[:-6] for c in pf.schema_arrow.names if c.endswith("_score")]
    na, nb = len(ia), len(ib)
    st = {f"{c}_{k}": np.full(n, -np.inf, np.float32)
          for c in chans for k, n in (("rec1", nb), ("rec2", nb), ("s1best", na))}
    n_s1, n_rec = np.zeros(na, np.int32), np.zeros(nb, np.int32)
    cols = ["s1", "cand"] + [f"{c}_score" for c in chans]
    for pass_ in (1, 2):
        for b in pf.iter_batches(batch_size=CFG["features"]["chunk"], columns=cols):
            d = b.to_pandas()
            a, r = ia.get_indexer(d["s1"].values), ib.get_indexer(d["cand"].values)
            if pass_ == 1:
                n_s1 += np.bincount(a, minlength=na).astype(np.int32)
                n_rec += np.bincount(r, minlength=nb).astype(np.int32)
            for c in chans:
                s = d[f"{c}_score"].values
                ok = ~np.isnan(s)
                if pass_ == 1:
                    np.maximum.at(st[f"{c}_rec1"], r[ok], s[ok])
                    np.maximum.at(st[f"{c}_s1best"], a[ok], s[ok])
                else:
                    lo = ok & (s < st[f"{c}_rec1"][r])
                    np.maximum.at(st[f"{c}_rec2"], r[lo], s[lo])
    for v in st.values():
        v[np.isinf(v)] = np.nan
    return chans, st, n_s1, n_rec


# context features for one chunk from the precomputed stats
def context(d, a, r, chans, st, n_s1, n_rec):
    out = {"n_cand_s1": n_s1[a], "n_s1_rec": n_rec[r]}
    for c in chans:
        s = d[f"{c}_score"].values
        best, second = st[f"{c}_rec1"][r], st[f"{c}_rec2"][r]
        # >0: this S1 leads the record by that much; <0: behind the record's best S1
        out[f"{c}_margin_rec"] = np.where(s >= best, s - np.nan_to_num(second, nan=0.0), s - best)
        out[f"{c}_gap_s1"] = st[f"{c}_s1best"][a] - s
    return {k: v.astype(np.float32) for k, v in out.items()}


# how common a name / address is inside the split (chains, malls); share keeps train/test comparable
def frequency(A, B):
    an = A.column("name_core").to_pandas()
    bn = B.column("name_core").to_pandas()
    aa = A.column("addr_norm").to_pandas()
    s1_name = an.value_counts()
    rec_name = bn.value_counts()
    f = {
        "a": {"f_s1_name": an.map(s1_name).values, "f_s1_addr": aa.map(aa.value_counts()).values},
        "b": {"f_rec_name": bn.map(rec_name).values,
              "f_rec_name_in_s1": bn.map(s1_name).fillna(0).values},
    }
    na, nb = len(an), len(bn)
    out = {}
    for side, cols in f.items():
        for k, v in cols.items():
            n = na if k != "f_rec_name" else nb
            out[k] = (np.log1p(v).astype(np.float32), (v / n).astype(np.float32), side)
    return out
