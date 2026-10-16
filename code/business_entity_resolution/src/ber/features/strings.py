from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer

from ber.features.context import context, frequency, score_stats
from ber.io import CFG, timed, to_key, work

COLS = ["name_norm", "name_core", "name_skel", "legal_class", "dba_parts", "addr_norm", "addr_nolandmark",
        "landmark", "numbers", "postal"]
NAN = float("nan")


# norm table for sources -> (pyarrow table, key index)
def side(split, sources):
    t = pa.concat_tables([pq.read_table(work(split, f"norm_s{s}.parquet"), columns=["entity_id"] + COLS)
                          for s in sources])
    idx = pd.Index(to_key(t.column("entity_id").to_pandas()))
    return t.drop(["entity_id"]), idx


def pick(t, col, pos):
    return t.column(col).take(pa.array(pos)).to_pylist()


# postal tri-state: 1 same, 0 differ, NaN if either missing
def postal_match(a, b):
    return np.array([float(x == y) if x and y else NAN for x, y in zip(a, b)], dtype=np.float32)


# binary token matrices with sqrt(idf) weights, idf fit on the split (S1 + S2/S3); rows = entities
def idf_mats(A, B, col):
    cv = CountVectorizer(token_pattern=r"\S+", binary=True, dtype=np.float32)
    ta, tb = A.column(col).to_pylist(), B.column(col).to_pylist()
    cv.fit(ta + tb)
    Xa, Xb = cv.transform(ta), cv.transform(tb)
    df = np.asarray(Xa.sum(0)).ravel() + np.asarray(Xb.sum(0)).ravel()
    w = sp.diags(np.sqrt(np.log((len(ta) + len(tb)) / np.maximum(df, 1))).astype(np.float32))
    Xa, Xb = (Xa @ w).tocsr(), (Xb @ w).tocsr()
    return Xa, Xb, np.asarray(Xa.multiply(Xa).sum(1)).ravel(), np.asarray(Xb.multiply(Xb).sum(1)).ravel()


# IDF-weighted Jaccard: sum idf(shared) / sum idf(union); NaN if a side has no tokens
def idf_jaccard(M, pa_, pb):
    Xa, Xb, sa, sb = M
    inter = np.asarray(Xa[pa_].multiply(Xb[pb]).sum(1)).ravel()
    uni = sa[pa_] + sb[pb] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = inter / uni
    out[(sa[pa_] == 0) | (sb[pb] == 0)] = np.nan
    return out.astype(np.float32)


def _initials(t):
    t = [w for w in t if w != "and"]
    return "".join(w[0] for w in t) if len(t) >= 2 else None


# per-pair python features (run in a pool)
def _pair(x):
    an, bn, al, bl, ad, bd, anum, bnum, ba = x
    at, bt = an.split(), bn.split()
    me = NAN
    if at and bt:
        m1 = sum(max(JaroWinkler.similarity(u, v) for v in bt) for u in at) / len(at)
        m2 = sum(max(JaroWinkler.similarity(v, u) for u in at) for v in bt) / len(bt)
        me = (m1 + m2) / 2
    first = float(at[0] == bt[0]) if at and bt else NAN
    ia, ib = _initials(at), _initials(bt)
    acr = float((ia is not None and ia == bn.replace(" ", "")) or (ib is not None and ib == an.replace(" ", "")))
    legal = (float(al == bl) if al and bl else NAN)
    dba = NAN
    if ad or bd:
        dba = max(fuzz.token_set_ratio(p, q) for p in (ad or [an]) for q in (bd or [bn]))
    da, db = {w for w in at if w.isdigit()}, {w for w in bt if w.isdigit()}
    digit = float(bool(da & db)) if da and db else NAN
    nlen = min(len(an), len(bn)) / max(len(an), len(bn)) if an and bn else NAN
    house = float(anum[0] == bnum[0]) if len(anum) and len(bnum) else NAN
    sa, sb = set(anum), set(bnum)
    numj = len(sa & sb) / len(sa | sb) if sa and sb else NAN
    return me, first, acr, legal, dba, digit, nlen, house, numj


# probe (4M train pairs): landmark sim ~100% NaN, acronym ~0% true -> not written (disk)
DROP = {"n_acronym"}
PAIR_COLS = ["n_monge", "n_first", "n_acronym", "legal_eq", "dba_max", "n_digit_eq", "n_len_ratio",
             "house_eq", "num_jacc"]


# v1 pair features, chunked, row-aligned with cands.parquet; limit = only N chunks to feats_probe.parquet
def run(split, force=False, limit=None):
    out = work(split, "feats_probe.parquet" if limit else "feats.parquet")
    if out.exists() and not force and not limit:
        return
    A, ia = side(split, [1])
    B, ib = side(split, [2, 3])
    with timed(f"features {split} prep (idf, stats, frequency)"):
        MN, MA = idf_mats(A, B, "name_core"), idf_mats(A, B, "addr_norm")
        chans, st, n_s1, n_rec = score_stats(split, ia, ib)
        freq = frequency(A, B)
        alen_a = np.array([len(x) for x in A.column("addr_norm").to_pylist()], np.float32)
        alen_b = np.array([len(x) for x in B.column("addr_norm").to_pylist()], np.float32)
    pf = pq.ParquetFile(work(split, "cands.parquet"))
    ctx = [c for c in pf.schema_arrow.names if c.endswith(("_score", "_rank")) or c == "n_channels"]
    sim = lambda a, b, sc: cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32)
    writer = None
    with Pool(CFG["n_jobs"]) as pool:
        for n, batch in enumerate(pf.iter_batches(batch_size=CFG["features"]["chunk"],
                                                  columns=["s1", "cand", "src"] + ctx)):
            with timed(f"features {split} chunk {n}"):
                c = batch.to_pandas()
                pa_, pb = ia.get_indexer(c["s1"]), ib.get_indexer(c["cand"])
                assert (pa_ >= 0).all() and (pb >= 0).all()
                f = c[["s1", "cand", "src"] + ctx].copy()
                g = pick
                ac, bc = g(A, "name_core", pa_), g(B, "name_core", pb)
                f["nc_ratio"] = sim(ac, bc, fuzz.ratio)
                f["nc_tset"] = sim(ac, bc, fuzz.token_set_ratio)
                f["nc_tsort"] = sim(ac, bc, fuzz.token_sort_ratio)
                f["nc_partial"] = sim(ac, bc, fuzz.partial_ratio)
                f["nc_jw"] = sim(ac, bc, JaroWinkler.normalized_similarity)
                f["nc_lev"] = sim(ac, bc, Levenshtein.normalized_similarity)
                f["nn_tset"] = sim(g(A, "name_norm", pa_), g(B, "name_norm", pb), fuzz.token_set_ratio)
                ask, bsk = g(A, "name_skel", pa_), g(B, "name_skel", pb)
                f["sk_tset"] = sim(ask, bsk, fuzz.token_set_ratio)
                f["sk_ratio"] = sim(ask, bsk, fuzz.ratio)
                f["nc_idf_jacc"] = idf_jaccard(MN, pa_, pb)
                aa, ba = g(A, "addr_norm", pa_), g(B, "addr_norm", pb)
                empty_b = np.array([not x for x in ba])
                f["ad_tset"] = sim(aa, ba, fuzz.token_set_ratio)
                f["ad_tsort"] = sim(aa, ba, fuzz.token_sort_ratio)
                f["ad_ratio"] = sim(aa, ba, fuzz.ratio)
                f["adnl_tset"] = sim(g(A, "addr_nolandmark", pa_), g(B, "addr_nolandmark", pb), fuzz.token_set_ratio)
                f["ad_idf_jacc"] = idf_jaccard(MA, pa_, pb)
                f.loc[empty_b, ["ad_tset", "ad_tsort", "ad_ratio", "adnl_tset"]] = np.nan  # missing, not mismatch
                la, lb = alen_a[pa_], alen_b[pb]
                f["ad_len_ratio"] = np.where(empty_b, np.nan, np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1))
                f["b_addr_empty"] = empty_b.astype(np.float32)
                f["postal"] = postal_match(g(A, "postal", pa_), g(B, "postal", pb))
                rows = zip(ac, bc, g(A, "legal_class", pa_), g(B, "legal_class", pb), g(A, "dba_parts", pa_),
                           g(B, "dba_parts", pb), g(A, "numbers", pa_), g(B, "numbers", pb), ba)
                res = np.array(pool.map(_pair, rows, chunksize=20000), dtype=np.float32)
                for i, col in enumerate(PAIR_COLS):
                    if col not in DROP:
                        f[col] = res[:, i]
                for k, v in context(c, pa_, pb, chans, st, n_s1, n_rec).items():
                    f[k] = v
                for k, (lg, share, s) in freq.items():
                    p = pa_ if s == "a" else pb
                    f[k + "_share"] = share[p]
                tbl = pa.Table.from_pandas(f, preserve_index=False)
                writer = writer or pq.ParquetWriter(out, tbl.schema, compression="zstd")
                writer.write_table(tbl)
            if limit and n + 1 >= limit:
                break
    writer.close()
