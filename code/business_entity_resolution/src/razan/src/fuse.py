"""Decision-level fusion: a blend matching file + QTpies stage-2 pair probabilities -> fused matching file.

Per country c, on the stage-2 p of --pred (M9; --pred-alt replaces it for --alt-countries), with
q = p / max(1, sum of p over the row's candidates):
  veto       drop a blend link (s1, r) with p < veto[c]. Pairs --pred never scored are kept.
  add        add a pair with q >= add[c] on a row r left without an owner after the vetoes (one owner per S2/S3 row;
             add >= 0.5 makes the owner unique, because the q of a row sum to <= 1).
  --solo <tsv> (a second matching file, used as a third opinion):
  veto-solo  also drop a blend link with p < veto_solo[c] that the solo file does not link.
  add-solo   also add a pair with add_solo[c] <= q < add[c] that the solo file links (same ownership rule).
  --norm <dir> (normalised n_test_s{1,2,3}.parquet from norm.py), twin-signature vetoes:
  veto-g     drop a blend link with p < veto_g[c] whose names differ by one swapped core word at the same address.
  veto-n     drop a blend link with p < veto_n[c] with the same core name, a changed legal/filler part and disjoint
             numbers, both sides with an address.
  fill       an S1 left with no links gets its best candidate with q >= fill[c] on a still-unowned row, unless that
             pair was vetoed or carries a twin signature.
A vetoed pair is never re-added. Rows no rule touches are copied byte for byte.
Writes <out>/matching_results.tsv, params.json, stats.json and changes.parquet (every veto and add).

usage: python fuse.py <blend.tsv> <out_dir> --pred <pred_test_s2ce.parquet> --ids <ids_dir> [rule options, see README]
       python fuse.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys

import polars as pl

COUNTRIES = ("France", "US", "India")


def per_country(spec, default):
    d = dict.fromkeys(COUNTRIES, default)
    for kv in filter(None, (spec or "").split(",")):
        k, v = kv.split("=")
        assert k in d, f"unknown country {k}"
        d[k] = float(v)
    return d


def twin_sigs(d):
    """d: pairs with nm/core/ad/nums/f_noaddr suffixed _1 (S1) and _2 (row). Adds boolean columns g_twin, n_twin."""
    tok = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    one_swap = (tok("core_1").list.set_difference(tok("core_2")).list.len() == 1) \
        & (tok("core_2").list.set_difference(tok("core_1")).list.len() == 1) & (pl.col("core_1") != pl.col("core_2"))
    num_disjoint = (pl.col("nums_1").list.len() > 0) & (pl.col("nums_2").list.len() > 0) \
        & (pl.col("nums_1").list.set_intersection(pl.col("nums_2")).list.len() == 0)
    return d.with_columns((one_swap & (pl.col("ad_1") == pl.col("ad_2"))).fill_null(False).alias("g_twin"),
                          ((pl.col("core_1") == pl.col("core_2")) & (pl.col("nm_1") != pl.col("nm_2")) & num_disjoint
                           & ~pl.col("f_noaddr_1") & ~pl.col("f_noaddr_2")).fill_null(False).alias("n_twin"))


def with_norm(d, norm):
    """Attach normalised fields (suffix _1 for the S1 id `s1`, _2 for the row id `id`) to pairs d."""
    f = ["nm", "core", "ad", "nums", "f_noaddr"]
    n1 = pl.read_parquet(os.path.join(norm, "n_test_s1.parquet"), columns=["id"] + f).rename({"id": "s1"} | {c: c + "_1" for c in f})
    n2 = pl.concat([pl.read_parquet(os.path.join(norm, f"n_test_s{k}.parquet"), columns=["id"] + f) for k in (2, 3)]).rename({c: c + "_2" for c in f})
    return d.join(n1, on="s1", how="left").join(n2, on="id", how="left")


def fuse(L, p9, veto, add, solo=None, veto_solo=None, add_solo=None, veto_g=None, veto_n=None):
    """L: blend links (si, ri, country, ...). p9: stage-2 pairs (si, ri, p, country of si). solo: optional (si, ri)
    links of the solo file. Returns (kept, vetoed, added)."""
    th = lambda d: pl.col("country").replace_strict(d, return_dtype=pl.Float64)
    mark = lambda df: df.join(solo.select("si", "ri").with_columns(pl.lit(True).alias("solo")), on=["si", "ri"], how="left") \
                        .with_columns(pl.col("solo").fill_null(False)) if solo is not None else df.with_columns(pl.lit(False).alias("solo"))
    L = mark(L.join(p9.select("si", "ri", "p"), on=["si", "ri"], how="left"))
    bad = pl.col("p").is_not_null() & ((pl.col("p") < th(veto)) | ((pl.col("p") < th(veto_solo or veto)) & ~pl.col("solo")))
    if veto_g is not None:  # L carries g_twin / n_twin (see twin_sigs)
        bad = bad | (pl.col("p").is_not_null() & (((pl.col("p") < th(veto_g)) & pl.col("g_twin")) | ((pl.col("p") < th(veto_n)) & pl.col("n_twin"))))
    L = L.with_columns(bad.alias("veto"))
    kept, vetoed = L.filter(~pl.col("veto")), L.filter(pl.col("veto"))
    q = mark(p9.with_columns((pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("p").sum().over("ri"))).alias("q")))
    added = q.filter((pl.col("q") >= th(add)) | ((pl.col("q") >= th(add_solo or add)) & pl.col("solo"))) \
             .join(kept.select("ri"), on="ri", how="anti").join(vetoed.select("si", "ri"), on=["si", "ri"], how="anti")
    assert added["ri"].is_unique().all(), "add threshold < 0.5 gives a row two owners"
    return kept, vetoed, added


def main(a):
    s1 = pl.read_parquet(os.path.join(a.ids, "n_test_s1.parquet"), columns=["si", "id", "country"]).rename({"id": "s1"})
    r = pl.concat([pl.read_parquet(os.path.join(a.ids, f"n_test_s{k}.parquet"), columns=["ri", "id"]) for k in (2, 3)])
    rows = pl.read_csv(a.team, separator="\t", quote_char=None, infer_schema=False).with_row_index("row")
    head = rows.columns[1:]
    rows = rows.rename({head[0]: "s1", head[1]: "ids"})
    assert rows.height == s1.height and rows["s1"].is_unique().all() and rows.join(s1, on="s1", how="anti").height == 0
    L = rows.filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")) \
            .with_columns(pl.col("ids").str.split(",")).explode("ids").rename({"ids": "id"}) \
            .with_columns(pl.int_range(pl.len()).over("row").alias("pos")) \
            .join(s1, on="s1").join(r, on="id")
    p9 = pl.read_parquet(a.pred, columns=["si", "ri", "p"]).join(s1.select("si", "country"), on="si")
    if a.pred_alt:  # p of the model inside the blend for --alt-countries; --pred elsewhere
        alt = pl.read_parquet(a.pred_alt, columns=["si", "ri", "p"]).rename({"p": "p_alt"})
        cs = a.alt_countries.split(",")
        p9 = p9.join(alt, on=["si", "ri"], how="left") \
               .with_columns(pl.when(pl.col("country").is_in(cs)).then(pl.col("p_alt")).otherwise(pl.col("p")).alias("p")).drop("p_alt")
        assert p9["p"].null_count() == 0, "--pred-alt must cover the same pairs as --pred"
    veto, add = per_country(a.veto, 0.0), per_country(a.add, 2.0)
    solo = veto_solo = add_solo = None
    if a.solo:
        solo = pl.read_csv(a.solo, separator="\t", quote_char=None, infer_schema=False)
        solo = solo.rename(dict(zip(solo.columns, ["s1", "id"]))).filter(pl.col("id").is_not_null() & (pl.col("id") != "")) \
                   .with_columns(pl.col("id").str.split(",")).explode("id").join(s1, on="s1").join(r, on="id").select("si", "ri")
        veto_solo, add_solo = per_country(a.veto_solo, 0.0), per_country(a.add_solo, 2.0)
    veto_g = veto_n = None
    if a.norm:
        veto_g, veto_n = per_country(a.veto_g, 0.0), per_country(a.veto_n, 0.0)
        pmax = max([*veto_g.values(), *veto_n.values()])
        cand = L.select("s1", "id", "si", "ri").join(p9.filter(pl.col("p") < pmax).select("si", "ri"), on=["si", "ri"])
        sig = twin_sigs(with_norm(cand, a.norm)).select("si", "ri", "g_twin", "n_twin")
        L = L.join(sig, on=["si", "ri"], how="left").with_columns(pl.col("g_twin").fill_null(False), pl.col("n_twin").fill_null(False))
    kept, vetoed, added = fuse(L, p9, veto, add, solo, veto_solo, add_solo, veto_g, veto_n)
    fill = per_country(a.fill, 2.0)
    if a.fill:
        assert a.norm, "--fill needs --norm (twin signatures)"
        linked = pl.concat([kept.select("si"), added.select("si")]).unique()
        q = p9.with_columns((pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("p").sum().over("ri"))).alias("q"))
        fc = q.filter(pl.col("q") >= pl.col("country").replace_strict(fill, return_dtype=pl.Float64)).join(linked, on="si", how="anti") \
              .join(pl.concat([kept.select("ri"), added.select("ri")]), on="ri", how="anti").join(vetoed.select("si", "ri"), on=["si", "ri"], how="anti")
        fc = fc.join(s1.select("si", "s1"), on="si").join(r, on="ri")
        fc = twin_sigs(with_norm(fc, a.norm)).filter(~pl.col("g_twin") & ~pl.col("n_twin"))
        fc = fc.sort(["q", "si", "ri"], descending=[True, False, False]).unique("si", keep="first").unique("ri", keep="first")
        added = pl.concat([added, fc.select("si", "ri", "p", "country", "q").with_columns(pl.lit(False).alias("solo")).select(added.columns)])
        assert added["ri"].is_unique().all()
        print(f"fill: {fc.height} S1s filled", flush=True)
    added = added.join(s1.select("si", "s1"), on="si").join(r, on="ri").join(rows.select("row", "s1"), on="s1")

    # rows touched by a veto or an add are rebuilt (kept links in original order, then adds by id); the rest copied
    touched = pl.concat([vetoed.select("row"), added.select("row")]).unique()
    new = pl.concat([kept.join(touched, on="row", how="semi").select("row", "pos", "id"),
                     added.select("row", pl.lit(1 << 30).cast(kept["pos"].dtype).alias("pos"), "id")])
    lists = new.sort(["row", "pos", "id"]).group_by("row").agg(pl.col("id").str.join(",").alias("new"))
    out_rows = rows.join(touched.with_columns(pl.lit(True).alias("t")), on="row", how="left") \
                   .join(lists, on="row", how="left").sort("row") \
                   .with_columns(pl.when(pl.col("t")).then(pl.col("new").fill_null("")).otherwise(pl.col("ids")).alias("ids"))
    os.makedirs(a.out, exist_ok=True)
    tsv = os.path.join(a.out, "matching_results.tsv")
    out_rows.select(pl.col("s1").alias(head[0]), pl.col("ids").alias(head[1])) \
            .write_csv(tsv, separator="\t", quote_style="never", null_value="")

    ba = rows.select("row", "s1", (pl.col("ids").fill_null("") != "").alias("had")) \
             .join(out_rows.select("row", (pl.col("ids").fill_null("") != "").alias("has")), on="row") \
             .join(s1.select("s1", "country"), on="s1")
    stats = {}
    for c in COUNTRIES:
        v, ad, b = (d.filter(pl.col("country") == c) for d in (vetoed, added, ba))
        stats[c] = {"blend_links": L.filter(pl.col("country") == c).height, "veto": v.height, "add": ad.height,
                    "s1_touched": touched.join(b.select("row"), on="row", how="semi").height,
                    "s1_emptied": b.filter(pl.col("had") & ~pl.col("has")).height,
                    "s1_filled": b.filter(~pl.col("had") & pl.col("has")).height}
    md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
    params = {"blend": a.team, "blend_md5": md5(a.team), "pred": a.pred, "pred_md5": md5(a.pred),
              "pred_alt": a.pred_alt or None, "pred_alt_md5": md5(a.pred_alt) if a.pred_alt else None,
              "alt_countries": a.alt_countries if a.pred_alt else None, "veto": veto, "add": add,
              "solo": a.solo or None, "solo_md5": md5(a.solo) if a.solo else None, "veto_solo": veto_solo, "add_solo": add_solo,
              "norm": a.norm or None, "veto_g": veto_g, "veto_n": veto_n, "fill": fill if a.fill else None, "out_md5": md5(tsv)}
    json.dump(params, open(os.path.join(a.out, "params.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    pl.concat([vetoed.select("s1", "id", "si", "ri", "country", "p", pl.lit(None, pl.Float64).alias("q"), pl.lit("veto").alias("kind")),
               added.select("s1", "id", "si", "ri", "country", "p", "q", pl.lit("add").alias("kind"))]) \
      .write_parquet(os.path.join(a.out, "changes.parquet"))
    print(json.dumps(stats), f"\nwrote {tsv} md5 {params['out_md5']}", flush=True)


def selftest():
    i32 = {"si": pl.Int32, "ri": pl.Int32}
    L = pl.DataFrame({"si": [0, 0, 1, 2], "ri": [10, 11, 12, 13], "country": ["US", "US", "US", "France"]}, schema_overrides=i32)
    p9 = pl.DataFrame({"si": [0, 0, 1, 1, 2, 0], "ri": [10, 11, 12, 14, 13, 15], "p": [0.05, 0.9, 0.95, 0.995, 0.05, 0.999],
                       "country": ["US"] * 4 + ["France", "US"]}, schema_overrides=i32)
    veto, add = per_country("US=0.1", 0.0), per_country("US=0.99", 2.0)
    kept, vetoed, added = fuse(L, p9, veto, add)
    assert vetoed["ri"].to_list() == [10], vetoed            # US p 0.05 vetoed; France p 0.05 kept (no France veto)
    assert sorted(added["ri"]) == [14, 15], added            # unowned rows with q >= 0.99
    p9b = p9.vstack(pl.DataFrame({"si": [0], "ri": [12], "p": [0.999], "country": ["US"]}, schema_overrides=i32))
    assert 12 not in fuse(L, p9b, veto, add)[2]["ri"].to_list()   # an owned row is never given a second owner
    p9c = p9.vstack(pl.DataFrame({"si": [2], "ri": [14], "p": [0.9], "country": ["France"]}, schema_overrides=i32))
    assert 14 not in fuse(L, p9c, veto, add)[2]["ri"].to_list()   # competing candidates pull q below 0.99
    # third opinion: US link p 0.9 (ri 11) vetoed under veto_solo 0.95 only when the solo file does not link it
    solo = pl.DataFrame({"si": [1], "ri": [12]}, schema_overrides=i32)
    vs, as_ = per_country("US=0.95", 0.0), per_country("US=0.9", 2.0)
    assert sorted(fuse(L, p9, veto, add, solo, vs, as_)[1]["ri"]) == [10, 11]   # ri 12 (p .95) not < .95
    assert sorted(fuse(L, p9, veto, add, solo.vstack(pl.DataFrame({"si": [0], "ri": [11]}, schema_overrides=i32)), vs, as_)[1]["ri"]) == [10]
    p9d = p9.vstack(pl.DataFrame({"si": [1, 0], "ri": [16, 17], "p": [0.93, 0.93], "country": ["US", "US"]}, schema_overrides=i32))
    solo2 = pl.DataFrame({"si": [1], "ri": [16]}, schema_overrides=i32)
    got = fuse(L, p9d, veto, add, solo2, vs, as_)[2]["ri"].to_list()
    assert 16 in got and 17 not in got, got                     # q .93 added only where the solo file agrees
    # twin signatures: G = one core word swapped at the same address; N = same core, legal part changed, numbers disjoint
    d = pl.DataFrame({"nm_1": ["graines ecole", "acme", "acme", "bo cafe"], "core_1": ["graines ecole", "acme", "acme", "bo cafe"],
                      "ad_1": ["1 rue x", "12 main st", "12 main st", "5 rue y"], "nums_1": [["1"], ["12"], ["12"], ["5"]],
                      "f_noaddr_1": [False] * 4,
                      "nm_2": ["graines parents", "acme llc", "acme llc", "bo cafe sarl"], "core_2": ["graines parents", "acme", "acme", "bo cafe"],
                      "ad_2": ["1 rue x", "19 main st", "12 main st", "5 rue y"], "nums_2": [["1"], ["19"], ["12"], ["5"]],
                      "f_noaddr_2": [False] * 4})
    s = twin_sigs(d)
    assert s["g_twin"].to_list() == [True, False, False, False] and s["n_twin"].to_list() == [False, True, False, False], s
    Lg = L.with_columns(pl.Series("g_twin", [False, True, False, False]), pl.Series("n_twin", [False] * 4))
    assert sorted(fuse(Lg, p9, veto, add, veto_g=per_country("US=0.95", 0.0), veto_n=per_country("", 0.0))[1]["ri"]) == [10, 11]
    # a vetoed pair is never re-added by an add rule (ri 11: p .9 vetoed as a G-twin, solo links it, add_solo .9)
    solo3 = pl.DataFrame({"si": [0], "ri": [11]}, schema_overrides=i32)
    _, v3, a3 = fuse(Lg, p9, veto, add, solo3, None, per_country("US=0.9", 2.0), per_country("US=0.95", 0.0), per_country("", 0.0))
    assert 11 in v3["ri"].to_list() and 11 not in a3["ri"].to_list(), a3
    print("fuse.py self-check OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("team")
    ap.add_argument("out")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--veto", default="")
    ap.add_argument("--add", default="")
    ap.add_argument("--solo", default="")
    ap.add_argument("--veto-solo", default="")
    ap.add_argument("--add-solo", default="")
    ap.add_argument("--norm", default="")
    ap.add_argument("--veto-g", default="")
    ap.add_argument("--veto-n", default="")
    ap.add_argument("--fill", default="")
    ap.add_argument("--pred-alt", default="")
    ap.add_argument("--alt-countries", default="US,India")
    main(ap.parse_args())
