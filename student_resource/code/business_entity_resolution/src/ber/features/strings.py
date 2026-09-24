import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from ber.io import CFG, timed, to_key, work

COLS = ["name_norm", "name_core", "addr_norm", "postal"]


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
    out = np.array([float(x == y) if x and y else np.nan for x, y in zip(a, b)], dtype=np.float32)
    return out


# v0 pair features, chunked, row-aligned with cands.parquet
def run(split, force=False):
    out = work(split, "feats.parquet")
    if out.exists() and not force:
        return
    A, ia = side(split, [1])
    B, ib = side(split, [2, 3])
    pf = pq.ParquetFile(work(split, "cands.parquet"))
    writer = None
    sim = lambda a, b, sc: cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32)
    for n, batch in enumerate(pf.iter_batches(batch_size=CFG["features"]["chunk"],
                                              columns=["s1", "cand", "src", "tfidf_na_score", "tfidf_na_rank"])):
        with timed(f"features {split} chunk {n}"):
            c = batch.to_pandas()
            pa_, pb = ia.get_indexer(c["s1"]), ib.get_indexer(c["cand"])
            assert (pa_ >= 0).all() and (pb >= 0).all()
            f = pd.DataFrame({"s1": c["s1"], "cand": c["cand"], "src": c["src"],
                              "tfidf_na_score": c["tfidf_na_score"], "tfidf_na_rank": c["tfidf_na_rank"]})
            ac, bc = pick(A, "name_core", pa_), pick(B, "name_core", pb)
            f["nc_ratio"] = sim(ac, bc, fuzz.ratio)
            f["nc_tset"] = sim(ac, bc, fuzz.token_set_ratio)
            f["nc_tsort"] = sim(ac, bc, fuzz.token_sort_ratio)
            f["nc_partial"] = sim(ac, bc, fuzz.partial_ratio)
            f["nc_jw"] = sim(ac, bc, JaroWinkler.normalized_similarity)
            f["nn_tset"] = sim(pick(A, "name_norm", pa_), pick(B, "name_norm", pb), fuzz.token_set_ratio)
            aa, ba = pick(A, "addr_norm", pa_), pick(B, "addr_norm", pb)
            f["ad_tset"] = sim(aa, ba, fuzz.token_set_ratio)
            f["ad_ratio"] = sim(aa, ba, fuzz.ratio)
            f.loc[[not x for x in ba], ["ad_tset", "ad_ratio"]] = np.nan  # empty address = missing, not mismatch
            f["postal"] = postal_match(pick(A, "postal", pa_), pick(B, "postal", pb))
            tbl = pa.Table.from_pandas(f, preserve_index=False)
            writer = writer or pq.ParquetWriter(out, tbl.schema)
            writer.write_table(tbl)
    writer.close()
