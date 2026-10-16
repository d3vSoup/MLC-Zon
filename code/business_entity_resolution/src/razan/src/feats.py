"""Pair features for candidate pairs (ri = S2/S3 row, si = S1 row). Language-agnostic by design.

featurize(split): cand_{split}/*.parquet -> F_{split}/*.parquet (same chunking).
"""
import glob
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import w, THREADS, DROP, dropped
from block import core_of

KEEP_RANK = 15  # candidates kept per S2/S3 row (by combined cosine rank)


def live(split):
    """S1 rows present in this split's world (train with BER_DROP treats a hash-slice of S1 as absent)."""
    return ~dropped() if split == "train" and DROP else pl.lit(True)


def load_recs(split):
    s1 = pl.read_parquet(w(f"n_{split}_s1.parquet"), columns=["country", "nm", "core", "ad", "nums"]).with_row_index("si")
    s1 = s1.with_columns(live(split).cast(pl.Int32).alias("_live"))
    n = pl.col("_live").sum()  # frequencies count only S1 rows that exist in this world
    s1 = s1.with_columns(n.over("country", "core").alias("k_core"), n.over("country", "ad").alias("k_addr"),
                         n.over("country", pl.col("nums").list.first()).alias("k_h1"),
                         pl.col("core").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("acr"))
    core_freq = s1.filter(pl.col("_live") == 1).group_by("country", "core").agg(pl.len().alias("k_rcore"))
    s1 = s1.drop("_live")
    cols = ["id", "country", "nm", "core", "al_r", "ad", "nums", "f_handle", "f_domain", "f_native", "f_alias", "f_noaddr"]
    r = pl.concat([pl.read_parquet(w(f"n_{split}_s{s}.parquet"), columns=cols) for s in (2, 3)]).with_row_index("ri")
    r = r.with_columns(pl.when(pl.col("al_r").is_not_null()).then(pl.col("al_r")).otherwise(pl.col("nm")).alias("rnm"),
                       pl.col("id").str.starts_with("S3").cast(pl.Int8).alias("src3"),
                       pl.when(pl.col("al_r").is_not_null()).then(core_of("al_r")).otherwise(pl.col("core")).alias("rcore"))
    r = r.join(core_freq, left_on=["country", "rcore"], right_on=["country", "core"], how="left") \
         .with_columns(pl.col("k_rcore").fill_null(0).cast(pl.Int32))
    return s1.drop("country"), r.select("ri", "rnm", "rcore", "ad", "nums", "src3", "k_rcore",
                                         pl.col("f_handle", "f_domain", "f_native", "f_alias", "f_noaddr").cast(pl.Int8))


def s1_aggs(split):
    """Per-S1 statistics over all candidate rows (needs every chunk)."""
    c = pl.scan_parquet(w(f"cand_{split}/*.parquet")).filter(live(split))
    c = c.with_columns((pl.col("cc") == pl.col("cc").max().over("ri")).alias("top1"))
    return c.group_by("si").agg(pl.len().alias("n_cand_s1"), pl.col("top1").sum().cast(pl.Int32).alias("n_top1_s1"),
                                pl.col("cc").max().alias("ccmax_s1"), pl.col("cn").max().alias("cnmax_s1"),
                                pl.col("ca").max().alias("camax_s1")).collect()


def context(c, agg):
    out = []
    for v in ("cc", "cn", "ca", "cc2"):
        out += [pl.col(v).rank("ordinal", descending=True).over("ri").cast(pl.Int16).alias(f"rk_{v}"),
                (pl.col(v) - pl.col(v).max().over("ri")).alias(f"gap_{v}")]
    c = c.with_columns(out).with_columns(pl.len().over("ri").cast(pl.Int16).alias("n_cand"))
    top2 = c.group_by("ri").agg(pl.col("cc").top_k(2).alias("t2")).with_columns(
        pl.col("t2").list.get(0).alias("_b1"), pl.col("t2").list.get(1, null_on_oob=True).fill_null(0).alias("_b2"))
    c = c.join(top2.drop("t2"), on="ri").with_columns(  # margin to the best *other* S1 of this row
        pl.when(pl.col("rk_cc") == 1).then(pl.col("cc") - pl.col("_b2")).otherwise(pl.col("cc") - pl.col("_b1"))
          .alias("margin_cc")).drop("_b1", "_b2")
    c = c.join(agg, on="si", how="left").with_columns(
        (pl.col("cc") - pl.col("ccmax_s1")).alias("gap_s1_cc"), (pl.col("cn") - pl.col("cnmax_s1")).alias("gap_s1_cn"),
        (pl.col("ca") - pl.col("camax_s1")).alias("gap_s1_ca")).drop("ccmax_s1", "cnmax_s1", "camax_s1")
    return c.filter((pl.col("rk_cc") <= KEEP_RANK) | (pl.col("rk_cc2") <= KEEP_RANK))


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=THREADS, dtype=np.float32)


def pair_feats(c, s1, r):
    d = c.join(r, on="ri").join(s1, on="si", suffix="_1")
    rnm, nm1 = d["rnm"].to_list(), d["nm"].to_list()
    rc, c1 = d["rcore"].to_list(), d["core"].to_list()
    rcat, c1cat = [x.replace(" ", "") for x in rc], [x.replace(" ", "") for x in c1]
    ra, a1 = d["ad"].to_list(), d["ad_1"].to_list()
    h1 = d["nums_1"].list.first().fill_null("").to_list()
    f = {
        "n_ratio": _cp(fuzz.ratio, rnm, nm1),
        "n_tset": _cp(fuzz.token_set_ratio, rnm, nm1),
        "n_tsort": _cp(fuzz.token_sort_ratio, rc, c1),
        "n_part": _cp(fuzz.partial_ratio, rc, c1),
        "n_jw": _cp(JaroWinkler.normalized_similarity, rc, c1),
        "n_cat": _cp(fuzz.ratio, rcat, c1cat),
        "n_catpart": _cp(fuzz.partial_ratio, rcat, c1cat),
        "a_ratio": _cp(fuzz.ratio, ra, a1),
        "a_tset": _cp(fuzz.token_set_ratio, ra, a1),
        "a_tsort": _cp(fuzz.token_sort_ratio, ra, a1),
        "a_part": _cp(fuzz.partial_ratio, ra, a1),
        "h_part": _cp(fuzz.partial_ratio, h1, d["nums"].list.join(" ").to_list()),
        "h_ratio": _cp(fuzz.ratio, h1, d["nums"].list.first().fill_null("").to_list()),
    }
    feats = d.select(
        "ri", "si", "cn", "ca", "cc", "cc2", pl.col("^rk_.*$"), pl.col("^gap_.*$"), "n_cand", "margin_cc",
        "n_cand_s1", "n_top1_s1", "src3", "f_handle", "f_domain", "f_native", "f_alias", "f_noaddr",
        "k_core", "k_addr", "k_rcore", "k_h1",
        (pl.col("rcore") == pl.col("core")).cast(pl.Int8).alias("core_eq"),
        (pl.col("rcore").str.replace_all(" ", "") == pl.col("acr")).cast(pl.Int8).alias("acr_eq"),
        pl.col("rcore").str.len_chars().cast(pl.Int16).alias("len_r"), pl.col("core").str.len_chars().cast(pl.Int16).alias("len_1"),
        pl.col("nums").list.len().cast(pl.Int8).alias("nn_r"), pl.col("nums_1").list.len().cast(pl.Int8).alias("nn_1"),
        pl.col("nums").list.set_intersection("nums_1").list.len().cast(pl.Int8).alias("nn_common"),
        pl.col("nums").list.contains(pl.col("nums_1").list.first()).fill_null(False).cast(pl.Int8).alias("h_exact"),
    )
    return feats.with_columns(**{k: pl.Series(v) for k, v in f.items()})


def featurize(split, shard=0, nshards=1):
    """Chunk files with index % nshards == shard (run shards as separate processes)."""
    t0 = time.time()
    os.makedirs(w(f"F_{split}"), exist_ok=True)
    todo = [f for k, f in enumerate(sorted(glob.glob(w(f"cand_{split}/*.parquet")))) if k % nshards == shard]
    todo = [f for f in todo if not os.path.exists(w(f"F_{split}/{os.path.basename(f)}"))]
    if not todo:
        return
    agg = s1_aggs(split)
    s1, r = load_recs(split)
    print(f"[{shard}/{nshards}] recs loaded {time.time() - t0:.0f}s, {len(todo)} chunks", flush=True)
    for f in todo:
        out = w(f"F_{split}/{os.path.basename(f)}")
        c = context(pl.read_parquet(f).filter(live(split)), agg)
        pl.concat([pair_feats(c.slice(a, 2_000_000), s1, r) for a in range(0, c.height, 2_000_000)]).write_parquet(out + ".tmp")
        os.replace(out + ".tmp", out)
        print(f"[{shard}/{nshards}] {os.path.basename(f)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    featurize(sys.argv[1], *map(int, sys.argv[2:4]))
