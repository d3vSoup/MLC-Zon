import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.io import CFG, read_truth, timed, to_key, work

B = CFG["blocking"]


def load(split, s, cols=("entity_id", "country", "name_core", "addr_norm")):
    d = pd.read_parquet(work(split, f"norm_s{s}.parquet"), columns=list(cols))
    d["key"] = to_key(d.pop("entity_id"))
    if "addr_norm" in d:
        d["text"] = d.pop("name_core") + " " + d.pop("addr_norm")
    return d


# cand key -> true S1 key (train only; one owner per record is exact)
def owners():
    _, long = read_truth()
    return pd.Series(to_key(long["s1_id"]), index=to_key(long["cand_id"]))


# top-n S1 per query row by char-gram tfidf cosine -> (row, col, score, rank)
def search(vec, r_text, q_text, top_n):
    R = vec.fit_transform(r_text)
    RT = R.T.tocsr()
    Q = vec.transform(q_text)
    M = sp_matmul_topn(Q, RT, top_n=top_n, n_threads=CFG["n_jobs"]).tocsr()
    row = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    order = np.lexsort((-M.data, row))
    row, col, score = row[order], M.indices[order], M.data[order].astype(np.float32)
    start = np.repeat(M.indptr[:-1], np.diff(M.indptr))
    rank = (np.arange(len(row)) - start).astype(np.int8)
    return row, col, score, rank


# channel 1: name_core + addr word 1-2 gram tfidf, S2/S3 -> top-K S1, within country
def run(split, force=False, limit=None):
    out = work(split, "cands.parquet")
    if out.exists() and not force and limit is None:
        return
    s1 = load(split, 1)
    q = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    own = owners() if split == "train" else None
    k, top = B["k"], B["topk_store"]
    groups = [(c, s1.country == c, q.country == c) for c in s1.country.unique()] if B["within_country"] \
        else [("all", np.ones(len(s1), bool), np.ones(len(q), bool))]
    hits = np.zeros(top, np.int64)
    n_true = 0
    writer = None
    for c, ms, mq in groups:
        r, qq = s1[ms].reset_index(drop=True), q[mq].reset_index(drop=True)
        if limit:
            qq = qq.sample(min(limit, len(qq)), random_state=0).reset_index(drop=True)
        vec = TfidfVectorizer(analyzer=B["analyzer"], token_pattern=r"\S+", ngram_range=tuple(B["ngram"]),
                              max_df=B["max_df"], dtype=np.float32, sublinear_tf=True)
        with timed(f"block tfidf {split} {c} R={len(r)} Q={len(qq)}"):
            row, col, score, rank = search(vec, r["text"], qq["text"], top)
        s1k, ck = r["key"].values[col], qq["key"].values[row]
        if own is not None:
            lab = (own.reindex(ck).values == s1k)
            hits += np.bincount(rank[lab], minlength=top)
            n_true += own.index.isin(qq["key"].values).sum()
        keep = rank < k
        t = pd.DataFrame({"s1": s1k[keep], "cand": ck[keep], "src": (ck[keep] // 10**10).astype(np.int8),
                          "tfidf_na_score": score[keep], "tfidf_na_rank": rank[keep], "n_channels": np.int8(1)})
        if own is not None:
            t["label"] = lab[keep].astype(np.int8)
        if limit:
            continue
        tbl = pa.Table.from_pandas(t, preserve_index=False)
        writer = writer or pq.ParquetWriter(out, tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()
    if own is not None:
        pc = np.cumsum(hits) / max(n_true, 1)
        msg = "PC@K " + " ".join(f"{i + 1}:{pc[i]:.4f}" for i in range(top))
        print(msg)
        with open(work("reports", "blocking_pc_at_k.txt"), "a") as f:
            f.write(f"{split} limit={limit} max_df={B['max_df']} {msg}\n")
