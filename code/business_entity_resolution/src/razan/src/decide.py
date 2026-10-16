"""Decision layer: calibrated pair probabilities -> final match lists.

1. One owner per S2/S3 row: keep only its best S1, with q = p / max(1, sum_s p) over its candidates.
2. Per S1: choose the top-k prefix (k=0 allowed) maximising exact expected F0.5 under independence
   (Poisson-binomial DP over true positives inside / true matches left outside the prefix).
"""
import numpy as np
import polars as pl
from numba import njit


@njit(cache=True)
def _pb(q):
    """Poisson-binomial pmf of the number of successes."""
    pmf = np.zeros(len(q) + 1)
    pmf[0] = 1.0
    for i in range(len(q)):
        for c in range(i + 1, 0, -1):
            pmf[c] = pmf[c] * (1 - q[i]) + pmf[c - 1] * q[i]
        pmf[0] *= 1 - q[i]
    return pmf


@njit(cache=True)
def best_k(q):
    """q sorted descending. Returns (k, expected F0.5)."""
    n = len(q)
    best, bk = _pb(q)[0], 0  # empty prediction scores 1 iff nothing is true
    for k in range(1, n + 1):
        pin, pout = _pb(q[:k]), _pb(q[k:])
        e = 0.0
        for a in range(1, k + 1):
            for b in range(n - k + 1):
                e += pin[a] * pout[b] * 1.25 * a / (0.25 * (a + b) + k)
        if e > best:
            best, bk = e, k
    return bk, best


@njit(cache=True)
def _decode(starts, q):
    keep = np.zeros(len(q), dtype=np.bool_)
    ef = np.zeros(len(starts) - 1)
    for g in range(len(starts) - 1):
        a, b = starts[g], starts[g + 1]
        k, e = best_k(q[a:b])
        keep[a:a + k] = True
        ef[g] = e
    return keep, ef


def decide(pairs):
    """pairs: id, s1, p. Returns kept (s1, id) pairs and per-S1 expected F."""
    pairs = pairs.filter(pl.col("p") >= 1e-4)  # can never be selected; keeps memory small on ~170M-pair inputs
    d = pairs.with_columns((pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("p").sum().over("id"))).alias("q"))
    d = d.filter(pl.col("p") == pl.col("p").max().over("id")).unique("id", keep="first")  # one owner
    d = d.sort(["s1", "q"], descending=[False, True])
    sizes = d.group_by("s1", maintain_order=True).len()
    starts = np.concatenate([[0], np.cumsum(sizes["len"].to_numpy())]).astype(np.int64)
    keep, ef = _decode(starts, d["q"].to_numpy().astype(np.float64))
    return d.filter(pl.Series(keep)).select("s1", "id"), sizes.select("s1").with_columns(pl.Series("ef", ef))


if __name__ == "__main__":
    k, e = best_k(np.array([0.99, 0.98, 0.97]))
    assert k == 3 and e > 0.95
    assert best_k(np.array([0.05]))[0] == 0  # weak lone candidate -> predict empty
    k, _ = best_k(np.array([0.99, 0.9, 0.3]))
    assert k == 2, k  # F0.5 is precision-heavy: drop the 0.3 row
    p = pl.DataFrame({"id": ["a", "a", "b", "c"], "s1": ["X", "Y", "X", "Y"], "p": [0.9, 0.05, 0.95, 0.1]})
    kept, ef = decide(p)
    assert sorted(kept.rows()) == [("X", "a"), ("X", "b")], kept  # 'a' owned by X only; Y's 0.1 dropped
    print("decide.py self-check OK")
