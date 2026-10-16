from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from ber.io import CFG, timed, to_key, work

NAN = float("nan")
COLS = ["hn_lev", "hn_lev_norm", "hn_transpose", "hn_absdiff_log", "hn_reldiff", "hn_prefix", "hn_suffix",
        "hn_fmt_eq", "hn_range", "hn_in_other", "num_best_lev", "num_jacc_nopostal", "n_num_a", "n_num_b", "postal3"]


# "17-03" split into [17, 3] (zeros stripped) equals "1703"
def _joined(xs, y):
    return len(xs) > 1 and y.startswith(xs[0]) and len(y) > len(xs[0]) and int(y[len(xs[0]):]) == int(xs[1])


# soft house-number agreement for one pair; digits only, country-agnostic (step-4 noise study: dropped digit,
# other number taken as house no, 1-digit typo, cut digits dominate)
def pair(x):
    an, bn, ap, bp = x
    an, bn = list(an), list(bn)
    out = [NAN] * len(COLS)
    out[12], out[13] = len(an), len(bn)
    out[14] = float(ap[:3] == bp[:3]) if ap and bp else NAN
    if not an or not bn:
        return out
    a, b = an[0], bn[0]
    d = Levenshtein.distance(a, b)
    out[0], out[1] = d, d / max(len(a), len(b))
    out[2] = float(a != b and len(a) == len(b) and sorted(a) == sorted(b))
    ia, ib = int(a[:15]), int(b[:15])
    out[3] = np.log1p(abs(ia - ib))
    out[4] = abs(ia - ib) / max(ia, ib, 1)
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) and a[-1 - s] == b[-1 - s]:
        s += 1
    out[5], out[6] = p, s
    out[7] = float(a == b or _joined(an, b) or _joined(bn, a))
    rng = any(u.isdigit() and v.isdigit() and int(u) <= ib <= int(v) and 0 < int(v) - int(u) < 20
              for u, v in zip(an, an[1:])) or \
        any(u.isdigit() and v.isdigit() and int(u) <= ia <= int(v) and 0 < int(v) - int(u) < 20 for u, v in zip(bn, bn[1:]))
    out[8] = float(rng)
    out[9] = float(a in bn or b in an)
    out[10] = min(Levenshtein.distance(u, v) for u in an[:6] for v in bn[:6])
    sa, sb = set(an) - {ap.lstrip("0"), ap}, set(bn) - {bp.lstrip("0"), bp}
    out[11] = len(sa & sb) / len(sa | sb) if sa and sb else NAN
    return out


# features for pairs in cands (or given rows), chunked; writes a row-aligned parquet
def run(split, force=False, rows=None, out=None):
    out = out or work(split, "feats_num.parquet")
    if out.exists() and not force:
        return
    cols = ["entity_id", "numbers", "postal"]
    A = pa.concat_tables([pq.read_table(work(split, "norm_s1.parquet"), columns=cols)])
    B = pa.concat_tables([pq.read_table(work(split, f"norm_s{s}.parquet"), columns=cols) for s in (2, 3)])
    ia, ib = pd.Index(to_key(A.column("entity_id").to_pandas())), pd.Index(to_key(B.column("entity_id").to_pandas()))
    c = pq.read_table(work(split, "cands.parquet"), columns=["s1", "cand"])
    if rows is not None:
        c = c.take(pa.array(rows))
    writer = None
    with Pool(CFG["n_jobs"]) as pool:
        for i in range(0, c.num_rows, CFG["features"]["chunk"]):
            with timed(f"numbers {split} chunk {i // CFG['features']['chunk']}"):
                b = c.slice(i, CFG["features"]["chunk"]).to_pandas()
                pa_, pb = pa.array(ia.get_indexer(b["s1"])), pa.array(ib.get_indexer(b["cand"]))
                g = lambda t, col, p: t.column(col).take(p).to_pylist()
                res = np.array(pool.map(pair, zip(g(A, "numbers", pa_), g(B, "numbers", pb), g(A, "postal", pa_),
                                                  g(B, "postal", pb)), chunksize=20000), dtype=np.float32)
                tbl = pa.table({k: res[:, j] for j, k in enumerate(COLS)})
                writer = writer or pq.ParquetWriter(out, tbl.schema, compression="zstd")
                writer.write_table(tbl)
    writer.close()
