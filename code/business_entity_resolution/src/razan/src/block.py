"""Candidate generation: every S2/S3 row -> top-K S1 rows of the same country by a combined TF-IDF cosine.

Combined vector = [name char-4grams | address word 1-2grams], each part L2-normalised, scaled by 1/sqrt(2).
Features whose S1 document frequency exceeds a cap are dropped (they cost the most and weigh the least).
Pass 1 (all rows) cap CAPS[0]; pass 2 only for uncertain rows (best pass-1 cosine < UNSURE) with cap CAPS[1].
Measured on India train (50k rows), recall@20: cap 3k 0.974 (0.4 ms/row), 10k 0.979 (1.05 ms), 30k 0.983 (3.4 ms).
Every pair gets both cosines (cc, cc2) plus the name/address parts (cn, ca) of pass 1.
Rows index: si = row of n_{split}_s1, ri = row of concat(n_{split}_s2, n_{split}_s3).
Output work/cand_{split}/{country}_{chunk}.parquet: ri, si, cn, ca, cc, cc2.
Parallel: run shards as separate processes (block.py split country shard nshards), THREADS matmul threads each.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import w, THREADS
from norm import LEGAL, FILLER

K = 15
CAPS = (int(os.environ.get("BER_CAP1", 3000)), int(os.environ.get("BER_CAP2", 10000)))
UNSURE = 0.6
CHUNK = 250_000
HALF = np.float32(np.sqrt(0.5))


def core_of(col):
    return pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(LEGAL + FILLER))) \
        .list.join(" ")


def load(split, s):
    d = pl.read_parquet(w(f"n_{split}_s{s}.parquet"), columns=["id", "country", "core", "al_r", "ad"])
    name = pl.when(pl.col("al_r").is_not_null() & (core_of("al_r") != "")).then(core_of("al_r")).otherwise(pl.col("core"))
    return d.select("id", "country", name.str.replace_all(" ", "").alias("n"), pl.col("ad").alias("a"))


def queries(split):
    return pl.concat([load(split, 2), load(split, 3)]).with_row_index("ri")


def rowdot(X, Y, i, j):
    """cosine of row pairs (X[i], Y[j]) for aligned index arrays."""
    out = np.empty(len(i), dtype=np.float32)
    for a in range(0, len(i), 1_000_000):
        b = a + 1_000_000
        out[a:b] = np.asarray(X[i[a:b]].multiply(Y[j[a:b]]).sum(1)).ravel()
    return out


class View:
    def __init__(self, s1, cap):
        self.vn = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), sublinear_tf=True, dtype=np.float32, max_df=cap)
        self.va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), sublinear_tf=True,
                                  dtype=np.float32, max_df=cap)
        self.Sn = self.vn.fit_transform(s1["n"].to_list()).tocsr()
        self.Sa = self.va.fit_transform(s1["a"].to_list()).tocsr()
        self.S = (sp.hstack([self.Sn, self.Sa]) * HALF).tocsr()
        self.ST = self.S.T.tocsr()

    def q(self, qq):
        Qn, Qa = self.vn.transform(qq["n"].to_list()).tocsr(), self.va.transform(qq["a"].to_list()).tocsr()
        return Qn, Qa, (sp.hstack([Qn, Qa]) * HALF).tocsr()


_G = {}  # fitted views + queries of the current country


def _chunk(n):
    g = _G
    out = w(f"cand_{g['split']}/{g['c']}_{n:03d}.parquet")
    if os.path.exists(out):
        return
    qq = g["q"].slice(n * CHUNK, CHUNK)
    v1, v2 = g["v1"], g["v2"]
    Qn, Qa, Q = v1.q(qq)
    M = sp_matmul_topn(Q, v1.ST, top_n=K, threshold=0.02, n_threads=THREADS).tocsr()
    best = np.asarray(M.max(axis=1).todense()).ravel()
    _, _, Q2 = v2.q(qq)
    unsure = np.where(best < UNSURE)[0]
    M2 = sp_matmul_topn(Q2[unsure], v2.ST, top_n=K, threshold=0.02, n_threads=THREADS).tocoo()
    M = M.tocoo()
    i = np.concatenate([M.row, unsure[M2.row]]).astype(np.int64)
    j = np.concatenate([M.col, M2.col]).astype(np.int64)
    key = np.unique(i * 10_000_000 + j)
    i, j = key // 10_000_000, key % 10_000_000
    pl.DataFrame({"ri": qq["ri"].to_numpy()[i].astype(np.int32), "si": g["si_map"][j].astype(np.int32),
                  "cn": rowdot(Qn, v1.Sn, i, j), "ca": rowdot(Qa, v1.Sa, i, j),
                  "cc": rowdot(Q, v1.S, i, j), "cc2": rowdot(Q2, v2.S, i, j)}).write_parquet(out + ".tmp")
    os.replace(out + ".tmp", out)  # atomic: a crash never leaves a half-written chunk
    print(f"  {g['split']} {g['c']} chunk {n}: unsure {len(unsure)}/{qq.height}, pairs {len(i)}", flush=True)


def run(split, country="all", shard=0, nshards=1):
    """Chunks with index % nshards == shard of the given country (run shards as separate processes)."""
    t0 = time.time()
    os.makedirs(w(f"cand_{split}"), exist_ok=True)
    s1 = load(split, 1).with_row_index("si")
    q = queries(split)
    for c in sorted(s1["country"].unique().to_list()) if country == "all" else [country]:
        a, b = s1.filter(pl.col("country") == c), q.filter(pl.col("country") == c)
        todo = [n for n in range((b.height + CHUNK - 1) // CHUNK) if n % nshards == shard
                and not os.path.exists(w(f"cand_{split}/{c}_{n:03d}.parquet"))]
        if not todo:
            continue
        _G.update(split=split, c=c, q=b, si_map=a["si"].to_numpy(), v1=View(a, CAPS[0]), v2=View(a, CAPS[1]))
        print(f"{split} {c} [{shard}/{nshards}]: fitted {time.time() - t0:.0f}s, {len(todo)} chunks", flush=True)
        for n in todo:
            _chunk(n)
        print(f"{split} {c} [{shard}/{nshards}]: S1 {a.height} Q {b.height} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], *(sys.argv[2:3] or ["all"]), *map(int, sys.argv[3:5]))
