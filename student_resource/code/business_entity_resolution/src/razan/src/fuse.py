"""Submission-level fusion: a team blend TSV + our M9 stage-2 pair probabilities -> fused TSV.

Runs after the team blend (e.g. Muaz's clean-room `blend_cal` output). Per country c, on M9's raw
(pre-calibration) stage-2 p:
  veto  drop a team link (s1, r) that M9 scored with p < veto[c]. Pairs M9 never scored are kept, unless
        --veto-s1 is given: then a team link that our stage 1 scored below 0.002 (i.e. rejected before
        stage 2) is dropped as well. Pairs our blocking never produced are always kept.
  add   add an M9 pair (s1, r) with q = p / max(1, sum_s p over r's M9 candidates) >= add[c] when row r
        has no owner in the team file after vetoes (one owner per S2/S3 row). add >= 0.5 makes the M9
        owner unique, because the q of a row sum to <= 1.
Optional third opinion (--solo <tsv>: the teammate's solo matching file, ce_full_v1). The team blend averages his
model with our M5, so where M9 and his solo file both reject a team link only M5 supported it:
  veto-solo  also drop a team link with M9 p < veto_solo[c] that the solo file does not link.
  add-solo   also add an M9 pair with add_solo[c] <= q < add[c] that the solo file links (same ownership rule).
Optional twin-signature vetoes (--norm <dir>: our normalised n_test_s{1,2,3}.parquet from norm.py). Test holds near-twin
distractor rows that M9 cannot always push below the plain veto threshold (AUDIT §8.3-8.4; label-free excess vs a
twin-free baseline at the same M9 p: France G 2.0k of 2.5k links at p .7-.99, US N 0.37k at p .3-.7, India ~0):
  veto-g  drop a team link with M9 p < veto_g[c] whose names differ by one swapped core word at the same address.
  veto-n  drop a team link with M9 p < veto_n[c] with the same core name, a changed legal/filler part and
          disjoint numbers (house number shifted), both sides with an address.
Optional second model (--pred2 <parquet>, same si/ri/p layout: M10 = M9 recalibrated by a stage-3 LightGBM that adds
same-name rival-row counts; train CV 0.98898 vs M9 0.98852):
  veto2  also drop a team link with p2 < veto2[c].
  add2   also add a pair with q2 >= add2[c] on a row unowned after vetoes (a row both models would give to
         different S1s is left out).
Optional fill (--fill France=0.5, needs --norm): an S1 left with no links gets its best M9 candidate with q >= fill[c]
on a still-unowned row, unless that pair was vetoed or carries a twin signature (France's empty rate after the vetoes,
6.2%, is above the seen countries' 5.8%; a correct fill is +1 for its S1).
An S1 left with no links is written empty (predicted no match). Rows that no rule touches are copied
byte for byte, so identity parameters (no veto, add > 1) reproduce the input file exactly.
Writes <out>/matching_results.tsv, params.json, stats.json and changes.parquet (every veto and add).

usage: python fuse.py <team.tsv> <out_dir> [--veto France=0.1,US=0.1,India=0.1] [--add US=0.99,India=0.99]
                      [--solo solo.tsv --veto-solo US=0.7 --add-solo France=0.9]
                      [--veto-s1] [--pred pred_test_s2ce.parquet] [--ids ids_dir] [--p1 pred_test_m.parquet]
       python fuse.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys

import polars as pl

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
COUNTRIES = ("France", "US", "India")
S1_MIN = 0.002  # our stage-1 threshold for entering stage 2


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
    """Attach our normalised fields (suffix _1 for the S1 id `s1`, _2 for the row id `id`) to pairs d."""
    f = ["nm", "core", "ad", "nums", "f_noaddr"]
    n1 = pl.read_parquet(os.path.join(norm, "n_test_s1.parquet"), columns=["id"] + f).rename({"id": "s1"} | {c: c + "_1" for c in f})
    n2 = pl.concat([pl.read_parquet(os.path.join(norm, f"n_test_s{k}.parquet"), columns=["id"] + f) for k in (2, 3)]).rename({c: c + "_2" for c in f})
    return d.join(n1, on="s1", how="left").join(n2, on="id", how="left")


def fuse(L, p9, veto, add, s1_rejected=None, solo=None, veto_solo=None, add_solo=None, veto_g=None, veto_n=None,
         p2=None, veto2=None, add2=None):
    """L: team links (si, ri, country, ...). p9: M9 stage-2 pairs (si, ri, p, country of si).
    s1_rejected: optional (si, ri) pairs our stage 1 scored below S1_MIN. solo: optional (si, ri) links of the solo
    file, used with the per-country veto_solo / add_solo thresholds. Returns (kept, vetoed, added)."""
    th = lambda d: pl.col("country").replace_strict(d, return_dtype=pl.Float64)
    mark = lambda df: df.join(solo.select("si", "ri").with_columns(pl.lit(True).alias("solo")), on=["si", "ri"], how="left") \
                        .with_columns(pl.col("solo").fill_null(False)) if solo is not None else df.with_columns(pl.lit(False).alias("solo"))
    L = mark(L.join(p9.select("si", "ri", "p"), on=["si", "ri"], how="left"))
    bad = pl.col("p").is_not_null() & ((pl.col("p") < th(veto)) | ((pl.col("p") < th(veto_solo or veto)) & ~pl.col("solo")))
    if veto_g is not None:  # L carries g_twin / n_twin (see twin_sigs)
        bad = bad | (pl.col("p").is_not_null() & (((pl.col("p") < th(veto_g)) & pl.col("g_twin")) | ((pl.col("p") < th(veto_n)) & pl.col("n_twin"))))
    if p2 is not None:
        L = L.join(p2.select("si", "ri", pl.col("p").alias("p2")), on=["si", "ri"], how="left")
        bad = bad | (pl.col("p2") < th(veto2)).fill_null(False)
    if s1_rejected is not None:
        L = L.join(s1_rejected.select("si", "ri").with_columns(pl.lit(True).alias("s1rej")), on=["si", "ri"], how="left")
        bad = bad | (pl.col("p").is_null() & pl.col("s1rej").fill_null(False))
    L = L.with_columns(bad.alias("veto"))
    kept, vetoed = L.filter(~pl.col("veto")), L.filter(pl.col("veto"))
    q = mark(p9.with_columns((pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("p").sum().over("ri"))).alias("q")))
    added = q.filter((pl.col("q") >= th(add)) | ((pl.col("q") >= th(add_solo or add)) & pl.col("solo"))) \
             .join(kept.select("ri"), on="ri", how="anti").join(vetoed.select("si", "ri"), on=["si", "ri"], how="anti")
    if p2 is not None:
        q2 = p2.join(p9.select("si", "ri", "country"), on=["si", "ri"]) \
               .with_columns((pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("p").sum().over("ri"))).alias("q"))
        a2 = q2.filter(pl.col("q") >= th(add2)).join(kept.select("ri"), on="ri", how="anti") \
               .join(vetoed.select("si", "ri"), on=["si", "ri"], how="anti")
        both = pl.concat([added.select("si", "ri"), a2.select("si", "ri")]).unique()
        clash = both.group_by("ri").len().filter(pl.col("len") > 1).select("ri")
        added = pl.concat([added, a2.join(added.select("si", "ri"), on=["si", "ri"], how="anti")
                           .with_columns(pl.lit(False).alias("solo")).select(added.columns)]).join(clash, on="ri", how="anti")
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
    veto, add = per_country(a.veto, 0.0), per_country(a.add, 2.0)
    s1rej = None
    if a.veto_s1:
        unscored = L.join(p9.select("si", "ri"), on=["si", "ri"], how="anti").select("si", "ri")
        s1rej = pl.scan_parquet(a.p1).filter(pl.col("p") < S1_MIN).join(unscored.lazy(), on=["si", "ri"], how="semi") \
                  .select("si", "ri").collect(engine="streaming")
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
    p2 = veto2 = add2 = None
    if a.pred2:
        p2 = pl.read_parquet(a.pred2, columns=["si", "ri", "p"])
        veto2, add2 = per_country(a.veto2, 0.0), per_country(a.add2, 2.0)
    kept, vetoed, added = fuse(L, p9, veto, add, s1rej, solo, veto_solo, add_solo, veto_g, veto_n, p2, veto2, add2)
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
        stats[c] = {"team_links": L.filter(pl.col("country") == c).height,
                    "veto": v.height, "veto_by_stage1": int(v["p"].is_null().sum()), "add": ad.height,
                    "s1_touched": touched.join(b.select("row"), on="row", how="semi").height,
                    "s1_emptied": b.filter(pl.col("had") & ~pl.col("has")).height,
                    "s1_filled": b.filter(~pl.col("had") & pl.col("has")).height}
    md5 = hashlib.md5(open(tsv, "rb").read()).hexdigest()
    params = {"team": a.team, "team_md5": hashlib.md5(open(a.team, "rb").read()).hexdigest(), "pred": a.pred,
              "veto": veto, "add": add, "veto_s1": bool(a.veto_s1), "s1_min": S1_MIN,
              "solo": a.solo, "solo_md5": hashlib.md5(open(a.solo, "rb").read()).hexdigest() if a.solo else None,
              "veto_solo": veto_solo, "add_solo": add_solo, "norm": a.norm or None, "veto_g": veto_g, "veto_n": veto_n,
              "norm_md5": {k: hashlib.md5(open(os.path.join(a.norm, f"n_test_s{k}.parquet"), "rb").read()).hexdigest() for k in (1, 2, 3)} if a.norm else None,
              "pred2": a.pred2 or None, "pred2_md5": hashlib.md5(open(a.pred2, "rb").read()).hexdigest() if a.pred2 else None,
              "veto2": veto2, "add2": add2, "fill": fill if a.fill else None, "out_md5": md5}
    json.dump(params, open(os.path.join(a.out, "params.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    pl.concat([vetoed.select("s1", "id", "si", "ri", "country", "p", pl.lit(None, pl.Float64).alias("q"), pl.lit("veto").alias("kind")),
               added.select("s1", "id", "si", "ri", "country", "p", "q", pl.lit("add").alias("kind"))]) \
      .write_parquet(os.path.join(a.out, "changes.parquet"))
    print(json.dumps(stats), f"\nwrote {tsv} md5 {md5}", flush=True)


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
    assert 12 not in fuse(L, p9b, veto, add)[2]["ri"].to_list()   # a team-owned row is never given a second owner
    p9c = p9.vstack(pl.DataFrame({"si": [2], "ri": [14], "p": [0.9], "country": ["France"]}, schema_overrides=i32))
    assert 14 not in fuse(L, p9c, veto, add)[2]["ri"].to_list()   # competing M9 candidates pull q below 0.99
    L2 = L.vstack(pl.DataFrame({"si": [1], "ri": [99], "country": ["US"]}, schema_overrides=i32))
    rej = pl.DataFrame({"si": [1], "ri": [99]}, schema_overrides=i32)
    assert sorted(fuse(L2, p9, veto, add, rej)[1]["ri"]) == [10, 99]  # --veto-s1 drops a stage-1-rejected team link
    assert sorted(fuse(L2, p9, veto, add)[1]["ri"]) == [10]           # ...and keeps it without the flag
    # third opinion: US team link p 0.9 (ri 11) vetoed under veto_solo 0.95 only when the solo file does not link it
    solo = pl.DataFrame({"si": [1], "ri": [12]}, schema_overrides=i32)
    vs, as_ = per_country("US=0.95", 0.0), per_country("US=0.9", 2.0)
    assert sorted(fuse(L, p9, veto, add, None, solo, vs, as_)[1]["ri"]) == [10, 11]   # ri 12 (p .95) not < .95
    assert sorted(fuse(L, p9, veto, add, None, solo.vstack(pl.DataFrame({"si": [0], "ri": [11]}, schema_overrides=i32)), vs, as_)[1]["ri"]) == [10]
    p9d = p9.vstack(pl.DataFrame({"si": [1, 0], "ri": [16, 17], "p": [0.93, 0.93], "country": ["US", "US"]}, schema_overrides=i32))
    solo2 = pl.DataFrame({"si": [1], "ri": [16]}, schema_overrides=i32)
    got = fuse(L, p9d, veto, add, None, solo2, vs, as_)[2]["ri"].to_list()
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
    _, v3, a3 = fuse(Lg, p9, veto, add, None, solo3, None, per_country("US=0.9", 2.0), per_country("US=0.95", 0.0), per_country("", 0.0))
    assert 11 in v3["ri"].to_list() and 11 not in a3["ri"].to_list(), a3
    # second model: veto2 drops ri 12 (p2 .2); add2 adds ri 16 (q2 .9); ri 14 is proposed by M9 for si 1 and by p2 for si 0 -> left out
    p9e = p9.vstack(pl.DataFrame({"si": [1, 0], "ri": [16, 14], "p": [0.5, 0.001], "country": ["US", "US"]}, schema_overrides=i32))
    p2 = p9e.select("si", "ri", "p").with_columns(pl.when(pl.col("ri") == 12).then(0.2).when(pl.col("ri") == 16).then(0.9)
                                                   .when((pl.col("ri") == 14) & (pl.col("si") == 0)).then(0.99)
                                                   .when((pl.col("ri") == 14) & (pl.col("si") == 1)).then(0.001).otherwise(pl.col("p")).alias("p"))
    _, v4, a4 = fuse(L, p9e, veto, add, p2=p2, veto2=per_country("US=0.5", 0.0), add2=per_country("US=0.85", 2.0))
    assert sorted(v4["ri"]) == [10, 12] and 16 in a4["ri"].to_list() and 14 not in a4["ri"].to_list() and 15 in a4["ri"].to_list(), (v4, a4)
    print("fuse.py self-check OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("team")
    ap.add_argument("out")
    ap.add_argument("--veto", default="")
    ap.add_argument("--add", default="")
    ap.add_argument("--veto-s1", action="store_true")
    ap.add_argument("--solo", default="")
    ap.add_argument("--veto-solo", default="")
    ap.add_argument("--add-solo", default="")
    ap.add_argument("--norm", default="")
    ap.add_argument("--veto-g", default="")
    ap.add_argument("--veto-n", default="")
    ap.add_argument("--pred2", default="")
    ap.add_argument("--veto2", default="")
    ap.add_argument("--add2", default="")
    ap.add_argument("--fill", default="")
    ap.add_argument("--pred", default=os.path.join(ROOT, "runs/qtpies-m9/pred_test_s2ce.parquet"))
    ap.add_argument("--ids", default=os.path.join(ROOT, "runs/qtpies-m9/share/ids"))
    ap.add_argument("--p1", default=os.path.join(ROOT, "runs/qtpies-m7b/pred_test_m.parquet"))
    main(ap.parse_args())
