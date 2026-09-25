import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer

from ber.dictionaries import LEGAL, NAME_STOP, NAME_STOP2
from ber.io import CFG, timed, to_key, work
from ber.normalize import fold

DOTTED = re.compile(r"\b((?:[A-Za-z]\.){2,}[A-Za-z]?)")
DROP = set(LEGAL) | NAME_STOP | NAME_STOP2
COLS = ["v2_tset", "v2_ratio", "v2_jw", "v2_eq", "v2_idf_jacc", "rare_tset", "rare_ratio", "rare_jacc"]


# v2 name: "S.A.S.U." -> "sasu" before folding, then legal forms + multilingual stopwords out
def norm2(raw):
    t = fold(DOTTED.sub(lambda m: m.group(1).replace(".", ""), raw)).split()
    return " ".join(w for w in t if w not in DROP)


def _all_names(split):
    t = pa.concat_tables([pq.read_table(work(split, f"norm_s{s}.parquet"), columns=["entity_id", "name_raw"])
                          for s in (1, 2, 3)])
    return to_key(t.column("entity_id").to_pandas()), t.column("name_raw").to_pylist()


# extra name features for given cands rows (hard pairs); generic = token in > generic_df of S1 names (data-driven)
def run(split, rows, out):
    with timed(f"names2 {split} normalize all names"):
        keys, raw = _all_names(split)
        with Pool(CFG["n_jobs"]) as pool:
            v2 = pool.map(norm2, raw, chunksize=20000)
        s1 = keys // 10**10 == 1
        cv = CountVectorizer(token_pattern=r"\S+", binary=True, dtype=np.float32).fit(v2)
        X = cv.transform(v2).tocsr()
        df = np.asarray(X.sum(0)).ravel()
        w = np.sqrt(np.log(len(v2) / np.maximum(df, 1))).astype(np.float32)
        Xw = (X @ sp.diags(w)).tocsr()
        sq = np.asarray(Xw.multiply(Xw).sum(1)).ravel()
        s1_df = np.asarray(X[s1].sum(0)).ravel() / s1.sum()
        generic = {t for t, i in cv.vocabulary_.items() if s1_df[i] > CFG["names2"]["generic_df"]}
        rare = [" ".join(t for t in n.split() if t not in generic) for n in v2]
        idx = pd.Index(keys)
    c = pq.read_table(work(split, "cands.parquet"), columns=["s1", "cand"]).take(pa.array(rows)).to_pandas()
    ia, ib = idx.get_indexer(c["s1"].values), idx.get_indexer(c["cand"].values)
    va, vb = [v2[i] for i in ia], [v2[i] for i in ib]
    ra, rb = [rare[i] for i in ia], [rare[i] for i in ib]
    sim = lambda a, b, s: cpdist(a, b, scorer=s, workers=-1, dtype=np.float32)
    f = {"v2_tset": sim(va, vb, fuzz.token_set_ratio), "v2_ratio": sim(va, vb, fuzz.ratio),
         "v2_jw": sim(va, vb, JaroWinkler.normalized_similarity),
         "v2_eq": np.array([float(x == y and x != "") for x, y in zip(va, vb)], np.float32)}
    inter = np.asarray(Xw[ia].multiply(Xw[ib]).sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        f["v2_idf_jacc"] = (inter / (sq[ia] + sq[ib] - inter)).astype(np.float32)
    f["v2_idf_jacc"][(sq[ia] == 0) | (sq[ib] == 0)] = np.nan
    empty = np.array([not x or not y for x, y in zip(ra, rb)])
    f["rare_tset"], f["rare_ratio"] = sim(ra, rb, fuzz.token_set_ratio), sim(ra, rb, fuzz.ratio)
    f["rare_jacc"] = np.array([len(set(x.split()) & set(y.split())) / len(set(x.split()) | set(y.split()))
                               if x and y else np.nan for x, y in zip(ra, rb)], np.float32)
    for k in ("rare_tset", "rare_ratio"):
        f[k][empty] = np.nan  # a name made only of generic words has no rare part: missing, not mismatch
    pq.write_table(pa.table(f), out, compression="zstd")
    print(f"names2 {split}: {len(c)} rows, {len(generic)} generic tokens (S1 df > {CFG['names2']['generic_df']})")
