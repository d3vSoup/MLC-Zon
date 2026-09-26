# reproducibility check before the Sunday clean room, read-only on work/:
# 1) re-normalize 50k random rows per source file with today's code, compare with the cached norm parquet
# 2) re-block 20k random train records per country (fw + hh top-10) with today's code, compare with cands.parquet;
#    split differences into "tie at the top-10 boundary" vs "real"; also run the fw search twice (run-to-run check)
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SR = Path(__file__).resolve().parents[1] / "student_resource"
sys.path.insert(0, str(SR / "code/business_entity_resolution/src"))
from ber.blocking.tfidf import docs, half_half, load, topn, vectorizer  # noqa: E402
from ber.io import CFG, work  # noqa: E402
from ber.normalize import _row  # noqa: E402

COLS = ["name_norm", "name_core", "legal_class", "dba_parts", "name_skel", "addr_norm", "addr_nolandmark",
        "landmark", "numbers", "postal"]


def norm_check(pool, rng):
    for split in ("train", "test"):
        for s in (1, 2, 3):
            f = pq.ParquetFile(work(split, f"norm_s{s}.parquet"))
            idx = np.sort(rng.choice(f.metadata.num_rows, 50_000, replace=False))
            t = pq.read_table(work(split, f"norm_s{s}.parquet")).take(pa.array(idx)).to_pandas()
            new = pd.DataFrame(pool.map(_row, zip(t.name_raw, t.addr_raw, t.country), chunksize=2000), columns=COLS)
            as_s = lambda x: str(list(x)) if not isinstance(x, str) and hasattr(x, "__len__") else str(x)  # array vs list print differently
            bad = {c: int((t[c].map(as_s).values != new[c].map(as_s).values).sum()) for c in COLS}
            print(f"normalize {split} s{s}: rows differing per column (of 50k): {bad}", flush=True)


def block_check(pool, rng):
    s1 = load("train", 1)
    q = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
    for c in s1["country"].unique():
        r = s1[s1.country == c].reset_index(drop=True)
        qq = q[q.country == c].sample(20_000, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        vec = vectorizer()
        R = vec.fit_transform(docs(pool, r))
        Q = vec.transform(docs(pool, qq))
        a, b = topn(Q, R, 10), topn(Q, R, 10)
        same_run = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        hits = {"fw": a}
        hits["hh"] = topn(*half_half(pool, r, qq)[::-1], 10)
        rk_, qk_ = r.key.values, qq.key.values
        new = {(int(rk_[j]), int(qk_[i])) for h in hits.values() for i, j in zip(h[0], h[1])}
        kth = {n: pd.Series(h[2]).groupby(h[0]).min() for n, h in hits.items()}  # 10th score per query
        old = pq.read_table(work("train", "cands.parquet"), columns=["s1", "cand", "fw_score", "hh_score"],
                            filters=pc.is_in(pc.field("cand"), value_set=pa.array(qq.key.values))).to_pandas()
        oldset = set(zip(old.s1.values.tolist(), old.cand.values.tolist()))
        only_new, only_old = set(new) - oldset, oldset - set(new)
        qpos = pd.Series(np.arange(len(qq)), index=qq.key.values)
        # production-only pair is a boundary tie if its score equals the new 10th score of that query in its channel
        o = old[[p in only_old for p in zip(old.s1.values.tolist(), old.cand.values.tolist())]]
        tie = np.zeros(len(o), bool)
        for n in ("fw", "hh"):
            k = kth[n].reindex(qpos.reindex(o.cand.values).values).values
            tie |= np.isclose(o[f"{n}_score"].values, k, atol=1e-5)
        print(f"block {c}: new {len(new)}, production {len(oldset)}, only-new {len(only_new)}, only-production "
              f"{len(only_old)} (boundary ties {tie.mean() if len(o) else 1:.3f}); fw run-to-run identical {same_run}",
              flush=True)


if __name__ == "__main__":
    rng = np.random.default_rng(CFG["seed"])
    with Pool(CFG["n_jobs"]) as pool:
        norm_check(pool, rng)
        block_check(pool, rng)
