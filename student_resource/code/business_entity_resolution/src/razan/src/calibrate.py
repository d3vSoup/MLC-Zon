"""Label-free test-time decode calibration (prior shift: test adds distractor rows the model scores as plausible).

Countries seen in training: odds scaled by k_c, estimated from the run's train OOF (BER_DROP = its drop rate).
Per p band of the rows' best pairs, ratio = test rows per S1 / train OOF rows per S1. The generator's true links per
S1 are split-invariant, so excess rows are test-only distractors. Difference-in-differences against the seen country
with the smallest mid-band excess (the anchor, k=1) cancels artifacts common to all (test p = mean of the two fold
models, simulated density): precision = p * ratio_anchor / ratio_c; k_c = row-weighted geometric mean of
min(1, odds(precision) / odds(p)) over bands p 0.9-0.995. Null test (train fold 1 vs fold 0) gives k ~ 1.
M4: India test = train in every band (anchor); US test has ~2x rows at p 0.6-0.97 = near-twin groups (name + legal
suffix, house number +1..20), per-band implied k flat 0.06-0.07 over p 0.9-0.995.

Unseen countries (France): odds scaled by k so the empty-S1 rate = seen countries' rate + DELTA, reproducing the
leaderboard-validated setting (France-only odds x0.15 on M1 raised public 0.969 -> 0.971). Their true links score
lower than in train (top band -10%), so band matching does not apply.
usage: python calibrate.py <pred_test.parquet> <out_dir> [<oof.parquet>]
"""
import os
import sys

import numpy as np
import polars as pl

from common import w, dropped, DROP
from decide import decide
from run import to_ids, write_tsv

DELTA = 0.0015
KS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.13, 0.1, 0.07, 0.05)
EDGES = 1 / (1 + np.exp(-np.linspace(-4.6, 9.2, 47)))  # p 0.01 .. 0.9999, 0.3 logit wide
MID = (0.3, 0.93)   # bands where fold-model sharpness does not move true links (null test: test/train ratio ~1)
DEC = (0.9, 0.995)  # decision-relevant bands the shift is estimated on


def empty_rate(p, n):
    kept, _ = decide(p.select(pl.col("ri").alias("id"), pl.col("si").alias("s1"), "p"))
    return 1 - kept["s1"].n_unique() / n, kept


def scale(p, k):
    return p.with_columns((pl.col("p") * k / (pl.col("p") * k + 1 - pl.col("p"))).alias("p"))


def _mask(band):
    lo, hi = np.r_[0, EDGES], np.r_[EDGES, 1]
    return (lo >= band[0] - 1e-9) & (hi <= band[1] + 1e-9)


def profile(te, tr, n_te, n_tr):
    """Per p band of the rows' best pairs: mean test p, test rows, test/train rows-per-S1 ratio."""
    def bands(d, n):
        d = d.sort("p", descending=True).unique("ri", keep="first")
        b, k = np.digitize(d["p"].to_numpy(), EDGES), len(EDGES) + 1
        return np.bincount(b, minlength=k), np.bincount(b, weights=d["p"].to_numpy(), minlength=k)
    rows_te, psum = bands(te, n_te)
    rows_tr, _ = bands(tr, n_tr)
    return psum / np.maximum(rows_te, 1), rows_te, (rows_te / n_te) / np.maximum(rows_tr / n_tr, 1e-12)


def shift_k(c, ref):
    """Difference-in-differences prior shift of profile c against the least-shifted seen country's profile ref:
    excess e = ratio_c / ratio_ref cancels artifacts common to both (fold-model averaging, simulated density);
    test precision = p / e; k = row-weighted geometric mean over DEC bands of odds(precision) / odds(p), capped at 1."""
    pbar, rows, r = c
    ok = _mask(DEC) & (rows >= 200) & (ref[1] >= 200)
    prec = np.clip(pbar[ok] * ref[2][ok] / r[ok], 1e-4, 1 - 1e-4)
    lk = np.minimum(0, np.log(prec / (1 - prec)) - np.log(pbar[ok] / (1 - pbar[ok])))  # a deficit is no evidence
    print("   band k: " + " ".join(f"{a:.3f}:{b:.3f}" for a, b in zip(pbar[ok], np.exp(lk))), flush=True)
    return min(1.0, float(np.exp(np.average(lk, weights=rows[ok]))))


def excess(prof):
    return np.average(prof[2][_mask(MID)], weights=prof[1][_mask(MID)])


def main(pred, out, oof=None):
    cty = pl.read_parquet(w("n_test_s1.parquet"), columns=["country"]).with_row_index("si").with_columns(pl.col("si").cast(pl.Int32))
    p = pl.read_parquet(pred).select("ri", "si", "p").join(cty, on="si")
    n = dict(cty.group_by("country").len().iter_rows())
    n_tr = {}
    if oof:
        tc = pl.read_parquet(w("n_train_s1.parquet"), columns=["country"]).with_row_index("si").with_columns(pl.col("si").cast(pl.Int32))
        tc = tc.filter(~dropped("si")) if DROP else tc
        tr = pl.read_parquet(oof).select("ri", "si", "p").join(tc, on="si")
        n_tr = dict(tc.group_by("country").len().iter_rows())
        prof = {c: profile(p.filter(pl.col("country") == c), tr.filter(pl.col("country") == c), n[c], n_tr[c]) for c in n if c in n_tr}
        anchor = min(prof, key=lambda c: excess(prof[c]))
    seen = [c for c in n if c in n_tr] if oof else [c for c in n if c != "France"]
    ref = sum(empty_rate(p.filter(pl.col("country") == c), n[c])[0] * n[c] for c in seen) / sum(n[c] for c in seen)
    parts = []
    for c in n:
        pc = p.filter(pl.col("country") == c)
        if c in n_tr:
            k = 1.0 if c == anchor else shift_k(prof[c], prof[anchor])
            print(f"{c}: mid-band excess {excess(prof[c]):.3f}, prior-shift k={k:.3f}" + (" (anchor)" if c == anchor else ""), flush=True)
            pc = scale(pc, k)
        elif c not in seen:  # unseen country: empty-rate matching
            best = min(KS, key=lambda k: abs(empty_rate(scale(pc, k), n[c])[0] - ref - DELTA))
            print(f"{c}: seen-country empty {ref:.4f}; target {ref + DELTA:.4f}; empty at k=1 {empty_rate(pc, n[c])[0]:.4f}; "
                  f"chosen k={best} -> {empty_rate(scale(pc, best), n[c])[0]:.4f}", flush=True)
            pc = scale(pc, best)
        parts.append(pc)
    p = pl.concat(parts)
    for c in n:
        e, k = empty_rate(p.filter(pl.col("country") == c), n[c])
        print(f"{c}: empty {e:.4f}, links/S1 {k.height / n[c]:.4f}", flush=True)
    kept, _ = decide(p.select(pl.col("ri").alias("id"), pl.col("si").alias("s1"), "p"))
    os.makedirs(out, exist_ok=True)
    km, s1ids = to_ids("test", kept.select(pl.col("s1").alias("si"), pl.col("id").alias("ri")))
    write_tsv(os.path.join(out, "matching_results.tsv"), s1ids, km, "matched_entity_ids")


if __name__ == "__main__":
    main(*sys.argv[1:4])
