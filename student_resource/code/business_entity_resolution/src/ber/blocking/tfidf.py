import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.io import CFG, read_truth, timed, to_key, work

B = CFG["blocking"]
COLS = ["entity_id", "country", "name_core", "name_skel", "addr_norm"]
SHIFT = 23  # pair code = s1_idx << 23 | cand_idx (within one country; < 8.4M records)


def load(split, s):
    d = pd.read_parquet(work(split, f"norm_s{s}.parquet"), columns=COLS)
    d["key"] = to_key(d.pop("entity_id"))
    return d


# cand key -> true S1 key (train only; one owner per record is exact)
def owners():
    _, long = read_truth()
    return pd.Series(to_key(long["s1_id"]), index=to_key(long["cand_id"]))


def _pairs(t):
    return ["|".join(sorted(p)) for p in itertools.combinations(t[:6], 2)]


# name tokens: words, word pairs (any order), glued name, sound-alike skeleton words + pairs
def name_toks(n, sk):
    nt, st = n.split(), sk.split()
    out = ["n:" + w for w in nt] + ["np:" + p for p in _pairs(nt)]
    c = [w for w in nt if w != "and"]
    if c:
        out.append("nc:" + "".join(c))
        if len(c) > 2:
            out.append("nc:" + c[0] + c[1])
    return out + ["s:" + w for w in st] + ["sp:" + p for p in _pairs(st)]


def addr_toks(a):
    at = a.split()
    return ["a:" + w for w in at] + ["ab:" + x + "_" + y for x, y in zip(at, at[1:])]


def doc(row):
    return " ".join(name_toks(row[0], row[1]) + addr_toks(row[2]))


def doc_name(row):
    return " ".join(name_toks(row[0], row[1]))


def doc_addr(row):
    return " ".join(addr_toks(row[2]))


def docs(pool, d, fn=doc):
    return pool.imap(fn, zip(d["name_core"], d["name_skel"], d["addr_norm"]), chunksize=20000)


# name and address as separate unit vectors, each half the weight: long addresses stop drowning the name
def half_half(pool, r, qq):
    w = np.float32(0.5 ** 0.5)
    parts = []
    for fn in (doc_name, doc_addr):
        vec = vectorizer()
        parts.append((vec.fit_transform(docs(pool, r, fn)) * w, vec.transform(docs(pool, qq, fn)) * w))
    return sp.hstack([p[0] for p in parts]).tocsr(), sp.hstack([p[1] for p in parts]).tocsr()


def vectorizer():
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", max_df=B["max_df"],
                           dtype=np.float32, sublinear_tf=True)


# top-n index rows per query row -> (query_idx, index_idx, score, rank)
def topn(Q, R, n):
    M = sp_matmul_topn(Q, R.T.tocsr(), top_n=n, n_threads=CFG["n_jobs"]).tocsr()
    row = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    o = np.lexsort((-M.data, row))
    row, col, score = row[o], M.indices[o], M.data[o].astype(np.float32)
    rank = (np.arange(len(row)) - np.repeat(M.indptr[:-1], np.diff(M.indptr))).astype(np.int16)
    return row, col, score, rank


# fw: all tokens, top-K S1 per record; hh: half name + half address; rv: top-K records per S1
def channels(pool, r, qq):
    out = {}
    with timed(f"  fw R={len(r)} Q={len(qq)}"):
        vec = vectorizer()
        R = vec.fit_transform(docs(pool, r))
        Q = vec.transform(docs(pool, qq))
        qi, si, sc, rk = topn(Q, R, B["k_fw"])
        out["fw"] = ((si.astype(np.int64) << SHIFT) | qi, sc, rk)
        del R, Q
    if B["k_hh"]:
        with timed(f"  hh R={len(r)} Q={len(qq)}"):
            R, Q = half_half(pool, r, qq)
            qi, si, sc, rk = topn(Q, R, B["k_hh"])
            out["hh"] = ((si.astype(np.int64) << SHIFT) | qi, sc, rk)
            del R, Q
    if B["k_rv"]:
        with timed(f"  rv S1={len(r)} D={len(qq)}"):
            vec = vectorizer()
            D = vec.fit_transform(docs(pool, qq))
            S = vec.transform(docs(pool, r))
            si, qi, sc, rk = topn(S, D, B["k_rv"])
            out["rv"] = ((si.astype(np.int64) << SHIFT) | qi, sc, rk)
    return out


# union of channels -> one row per pair, NaN score/rank where a channel did not propose it
def merge(ch):
    allp = np.unique(np.concatenate([v[0] for v in ch.values()]))
    cols, n_ch = {}, np.zeros(len(allp), np.int8)
    for name, (p, sc, rk) in ch.items():
        pos = np.searchsorted(allp, p)
        s, r = np.full(len(allp), np.nan, np.float32), np.full(len(allp), np.nan, np.float32)
        s[pos], r[pos] = sc, rk
        cols[f"{name}_score"], cols[f"{name}_rank"] = s, r
        n_ch[pos] += 1
    return allp, cols, n_ch


def run(split, force=False):
    out = work(split, "cands.parquet")
    if out.exists() and not force:
        return
    s1 = load(split, 1)
    q = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    own = owners() if split == "train" else None
    writer, stats = None, []
    with Pool(CFG["n_jobs"]) as pool:
        for c in s1["country"].unique():
            r = s1[s1.country == c].reset_index(drop=True)
            qq = q[q.country == c].reset_index(drop=True)
            assert len(qq) < 2 ** SHIFT
            with timed(f"block {split} {c}"):
                ch = channels(pool, r, qq)
                allp, cols, n_ch = merge(ch)
                del ch
            s1k, ck = r["key"].values[allp >> SHIFT], qq["key"].values[allp & (2 ** SHIFT - 1)]
            t = pd.DataFrame({"s1": s1k, "cand": ck, "src": (ck // 10**10).astype(np.int8), **cols,
                              "n_channels": n_ch})
            if own is not None:
                t["label"] = (own.reindex(ck).values == s1k).astype(np.int8)
                n_true = own.index.isin(qq["key"].values).sum()
                stats.append(f"{c}: pairs {len(t)} PC union {t.label.sum() / n_true:.4f} | " + " ".join(
                    f"{n} {t.label[t[f'{n}_score'].notna()].sum() / n_true:.4f}"
                    for n in ("fw", "hh", "rv") if f"{n}_score" in t))
            tbl = pa.Table.from_pandas(t, preserve_index=False)
            writer = writer or pq.ParquetWriter(out, tbl.schema)
            writer.write_table(tbl)
            del t, tbl
    writer.close()
    if stats:
        print("\n".join(stats))
        with open(work("reports", "blocking_pc_at_k.txt"), "a") as f:
            f.write(f"{split} k_fw={B['k_fw']} k_hh={B['k_hh']} k_rv={B['k_rv']} max_df={B['max_df']} | " + " || ".join(stats) + "\n")
