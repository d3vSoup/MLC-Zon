import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ber.io import CFG, timed, to_key, work

R2 = CFG["round2"]
R1 = R2["from_model"]


def keys_index(split, sources):
    t = pa.concat_tables([pq.read_table(work(split, f"norm_s{s}.parquet"), columns=["entity_id", "name_core",
                                                                                   "addr_norm"])
                          for s in sources])
    return t, pd.Index(to_key(t.column("entity_id").to_pandas()))


# top-2 of p per group (group ids g), and each row's best competitor in its group
def top2(g, p, n):
    m1 = np.full(n, -1.0, np.float32)
    np.maximum.at(m1, g, p)
    m2 = np.full(n, -1.0, np.float32)
    lo = p < m1[g]
    np.maximum.at(m2, g[lo], p[lo])
    other = np.where(p >= m1[g], m2[g], m1[g])  # best of the rest (-1 if none)
    return np.where(other < 0, np.nan, other).astype(np.float32)


# rank of each row inside its group by p (0 = best)
def rank_in(g, p):
    o = np.lexsort((-p, g))
    start = np.searchsorted(g[o], g[o], "left")
    rk = np.empty(len(g), np.int32)
    rk[o] = np.arange(len(g)) - start
    return rk.astype(np.float32)


# group features for hard pairs (round-1 p >= hard_p): competition + similarity to sure siblings
def run(split, force=False):
    out = work(split, "r2.parquet")
    if out.exists() and not force:
        return
    with timed(f"graph {split} load"):
        pr = pq.read_table(work("preds", f"{'oof' if split == 'train' else 'test'}_{R1}.parquet")).to_pandas()
        p_all = pr["p"].values
        hard = np.flatnonzero(p_all >= R2["hard_p"])
        s1k, ck, p = pr["s1"].values[hard], pr["cand"].values[hard], p_all[hard]
        del pr, p_all
        _, ia = keys_index(split, [1])
        B, ib = keys_index(split, [2, 3])
        a = ia.get_indexer(s1k).astype(np.int64)
        r = ib.get_indexer(ck).astype(np.int64)
        na, nb = len(ia), len(ib)
    with timed(f"graph {split} competition"):
        f = {"row": hard, "p1": p}
        f["p_share_rec"] = (p / np.bincount(r, p, nb)[r]).astype(np.float32)
        f["p_other_rec"] = top2(r, p, nb)          # best other S1 for this record
        f["p_rank_rec"] = rank_in(r, p)
        f["n_s1_c50_rec"] = np.bincount(r[p > 0.5], minlength=nb)[r].astype(np.float32)
        f["p_other_s1"] = top2(a, p, na)           # best other record of this S1
        f["p_rank_s1"] = rank_in(a, p)
        f["p_sum_s1_other"] = (np.bincount(a, p, na)[a] - p).astype(np.float32)
        c80 = p > 0.8
        f["n_s1_c80_other"] = (np.bincount(a[c80], minlength=na)[a] - c80).astype(np.float32)
    with timed(f"graph {split} siblings"):
        # sure rows (p >= sure_p), top max_sib per S1
        sure = np.flatnonzero(p >= R2["sure_p"])
        sure = sure[np.lexsort((-p[sure], a[sure]))]
        rk = np.arange(len(sure)) - np.searchsorted(a[sure], a[sure], "left")
        sure = sure[rk < R2["max_sib"]]
        cnt = np.bincount(a[sure], minlength=na)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        # expand every hard row x its S1's sure rows
        reps = cnt[a]
        t_idx = np.repeat(np.arange(len(p)), reps)
        k_idx = sure[np.repeat(start[a], reps) + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))]
        keep = r[k_idx] != r[t_idx]
        t_idx, k_idx = t_idx[keep], k_idx[keep]
        name_max = np.full(len(p), np.nan, np.float32)
        addr_max = np.full(len(p), np.nan, np.float32)
        wmax = np.full(len(p), np.nan, np.float32)
        addr_eq = np.zeros(len(p), np.float32)
        nsib = np.bincount(t_idx, minlength=len(p)).astype(np.float32)
        for i in range(0, len(t_idx), 5_000_000):
            t, k = t_idx[i:i + 5_000_000], k_idx[i:i + 5_000_000]
            tr, kr = pa.array(r[t]), pa.array(r[k])
            tn, kn = B.column("name_core").take(tr).to_pylist(), B.column("name_core").take(kr).to_pylist()
            ta, ka = B.column("addr_norm").take(tr).to_pylist(), B.column("addr_norm").take(kr).to_pylist()
            sn = cpdist(tn, kn, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            sa = cpdist(ta, ka, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            empty = np.array([not x or not y for x, y in zip(ta, ka)])
            sa[empty] = np.nan
            comb = np.where(empty, sn, (sn + sa) / 2) / 100 * p[k]
            np.fmax.at(name_max, t, sn)
            np.fmax.at(addr_max, t, sa)
            np.fmax.at(wmax, t, comb)
            np.maximum.at(addr_eq, t, np.array([float(bool(x) and x == y) for x, y in zip(ta, ka)], np.float32))
        f.update(sib_name_max=name_max, sib_addr_max=addr_max, sib_wmax=wmax, sib_addr_eq=addr_eq, sib_n=nsib)
        # does this record fit another S1's group better?
        w0 = np.nan_to_num(wmax, nan=0.0)
        f["sib_w_other_rec"] = top2(r, w0, nb)
    with timed(f"graph {split} join round-1 feats + write"):
        pf = pq.ParquetFile(work(split, "feats.parquet"))
        mask = np.zeros(pf.metadata.num_rows, bool)
        mask[hard] = True
        parts, off = [], 0
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            m = mask[off:off + t.num_rows]
            off += t.num_rows
            parts.append(t.filter(pa.array(m)))
        base = pa.concat_tables(parts)
        if split == "train":
            lab = pq.read_table(work("train", "cands.parquet"), columns=["label"]).column("label").to_numpy()
            base = base.append_column("label", pa.array(lab[hard]))
        for k, v in f.items():
            base = base.append_column(k, pa.array(v))
        pq.write_table(base, out, compression="zstd")
    print(f"graph {split}: hard rows {len(hard)}, sibling pairs {len(t_idx)}")
