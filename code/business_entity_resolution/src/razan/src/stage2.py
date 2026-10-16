"""Stage 2 (collective): features from stage-1 probabilities of *other* pairs.

Only pairs with p1 >= P_MIN go to stage 2 (they form candidate_pairs.tsv); the rest are final negatives.
For pair (r, s):
  owner competition   best other S1 of r, r's total mass, rank of s for r
  cluster context     s's confident mass (excluding r), per source, vs caps S2<=5 / S3<=6
  sibling evidence    similarity of r to s's confident rows r' (name, address, numbers);
                      a number shared by r and r' but absent from s = same generator instance
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from common import THREADS
from norm import LEGAL

P_MIN = 0.002
P_FLOOR = 1e-4
OWN_MIN = 0.3
RIV_MIN = 0.01
SIB_BUCKETS = 16


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=THREADS, dtype=np.float32)


GEN_DF = 0.002  # a name token is "generic" if >= 0.2% of the country's S1 names contain it


def generic_feats(split, D, w):
    """Generic-word vs distinctive-token comparison of the two names (vocabulary learned per country from this
    split's S1 names, no labels). Near-twin distractors keep the distinctive token and swap a generic word
    (France: 'Graines Ecole' vs 'Graines Parents'); noise mostly adds/drops fillers or typos tokens."""
    s1 = pl.read_parquet(w(f"n_{split}_s1.parquet"), columns=["core", "country"]).with_row_index("si") \
           .with_columns(pl.col("si").cast(pl.Int32), pl.col("core").str.split(" ").list.unique().alias("t"))
    n = s1.group_by("country").len().rename({"len": "n"})
    vocab = s1.select("country", "t").explode("t").group_by("country", "t").len().join(n, on="country") \
              .filter(pl.col("len") >= GEN_DF * pl.col("n")).select("country", "t", pl.lit(1, pl.Int8).alias("g"))
    r = pl.concat([pl.read_parquet(w(f"n_{split}_s{k}.parquet"), columns=["core", "country"]) for k in (2, 3)]) \
          .with_row_index("ri").with_columns(pl.col("ri").cast(pl.Int32), pl.col("core").str.split(" ").list.unique().alias("t")) \
          .join(D.select("ri").unique(), on="ri", how="semi")
    def parts(df, key):
        x = df.select(key, "country", "t").explode("t").join(vocab, on=["country", "t"], how="left")
        return x.group_by(key).agg(pl.col("t").filter(pl.col("g") == 1).alias(f"g_{key}"),
                                   pl.col("t").filter(pl.col("g").is_null()).alias(f"d_{key}"))
    D = D.join(parts(s1.join(D.select("si").unique(), on="si", how="semi"), "si"), on="si", how="left") \
         .join(parts(r, "ri"), on="ri", how="left")
    inter = pl.col("d_si").list.set_intersection("d_ri").list.len()
    union = pl.col("d_si").list.set_union("d_ri").list.len()
    D = D.with_columns(pl.col("g_si").list.set_difference("g_ri").list.len().alias("gen_drop"),
                       pl.col("g_ri").list.set_difference("g_si").list.len().alias("gen_add"),
                       inter.alias("dist_common"),
                       pl.when(union > 0).then(inter / union).otherwise(0.0).alias("dist_jacc"),
                       ((inter == pl.col("d_si").list.len()) & (inter == pl.col("d_ri").list.len())).cast(pl.Int8).alias("dist_eq"))
    D = D.with_columns(((pl.col("gen_drop") > 0) & (pl.col("gen_add") > 0)).cast(pl.Int8).alias("gen_swap"))
    return D.drop("g_si", "d_si", "g_ri", "d_ri")


def twin_feats(keys, r, s1):
    """Near-twin evidence per pair (BER_TWINF): signed shift dn of the row number closest to the S1's first house number
    (twins sit a few doors away: 6182 vs 6183 looks like a typo to fuzzy ratios) and legal-form tokens added/dropped."""
    leg = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element().is_in(LEGAL))).list.unique()
    s = s1.select("si", pl.col("nums").list.first().cast(pl.Int64, strict=False).alias("h1"), leg("nm").alias("leg1"))
    rr = r.select("ri", pl.col("nums").alias("rn"), leg("rnm").alias("leg"))
    d = keys.join(s, on="si").join(rr, on="ri")
    x = d.select("ri", "si", "h1", "rn").explode("rn").with_columns((pl.col("rn").cast(pl.Int64, strict=False) - pl.col("h1")).alias("dn")) \
         .drop_nulls("dn").group_by("ri", "si").agg(pl.col("dn").sort_by(pl.col("dn").abs()).first())
    return d.join(x, on=["ri", "si"], how="left").select(
        "ri", "si", "dn", pl.col("dn").abs().alias("dn_abs"), pl.col("dn").is_between(1, 20).cast(pl.Int8).alias("dn_pos20"),
        pl.col("leg").list.set_difference("leg1").list.len().alias("leg_add"),
        pl.col("leg1").list.set_difference("leg").list.len().alias("leg_drop"))


def num_feats(keys, r, s1):
    """House-number shape features (BER_NUMF; after the teammate pipeline's numbers.py): edit distance, common
    prefix/suffix, truncation, digit transposition, first-number gap, number-set Jaccard. NaN when a side has no number."""
    from rapidfuzz.distance import Levenshtein, Prefix, Postfix
    d = keys.join(s1.select("si", pl.col("nums").alias("an")), on="si").join(r.select("ri", pl.col("nums").alias("bn")), on="ri")
    d = d.with_columns(pl.col("an").list.first().alias("a"), pl.col("bn").list.first().alias("b"))
    ok = d["a"].is_not_null() & d["b"].is_not_null()
    a, b = d["a"].fill_null("").to_list(), d["b"].fill_null("").to_list()
    nan = lambda v: pl.when(pl.lit(ok)).then(v).otherwise(None)
    d = d.with_columns(pl.Series("hn_lev", _cp(Levenshtein.distance, a, b)), pl.Series("hn_pre", _cp(Prefix.similarity, a, b)),
                       pl.Series("hn_suf", _cp(Postfix.similarity, a, b)))
    ai, bi = pl.col("a").cast(pl.Int64, strict=False), pl.col("b").cast(pl.Int64, strict=False)
    srt = lambda c: pl.col(c).str.split("").list.sort().list.join("")
    return d.select("ri", "si",
        nan(pl.col("hn_lev")).alias("hn_lev"),
        nan(pl.col("hn_lev") / pl.max_horizontal(pl.col("a").str.len_chars(), pl.col("b").str.len_chars())).alias("hn_lev_norm"),
        nan(pl.col("hn_pre")).alias("hn_pre"), nan(pl.col("hn_suf")).alias("hn_suf"),
        nan(((pl.col("a") != pl.col("b")) & (pl.col("a").str.ends_with(pl.col("b")) | pl.col("b").str.ends_with(pl.col("a"))
             | pl.col("a").str.starts_with(pl.col("b")) | pl.col("b").str.starts_with(pl.col("a")))).cast(pl.Int8)).alias("hn_trunc"),
        nan(((pl.col("a") != pl.col("b")) & (srt("a") == srt("b"))).cast(pl.Int8)).alias("hn_transpose"),
        ((ai - bi).abs() + 1).log().alias("hn_absdiff_log"),
        ((ai - bi).abs() / pl.max_horizontal(ai, bi, pl.lit(1))).alias("hn_reldiff"),
        pl.when(pl.col("an").list.len().fill_null(0) + pl.col("bn").list.len().fill_null(0) > 0)
          .then(pl.col("an").list.set_intersection("bn").list.len() / pl.col("an").list.set_union("bn").list.len()).alias("num_jacc"))


def stage2_feats(P, r, s1):
    """P: ri, si, p (all stage-1 pairs of one split). r: ri, rnm, ad, nums, src3. s1: si, nums.
    Returns stage-2 pairs (p >= P_MIN) with new feature columns."""
    P = P.filter(pl.col("p") >= P_FLOOR)  # negligible mass; cuts ~170M train rows to a few 10M
    P = P.with_columns(pl.col("p").rank("ordinal", descending=True).over("ri").cast(pl.Int16).alias("prk"),
                       pl.col("p").sum().over("ri").alias("psum_r"))
    top2 = P.group_by("ri").agg(pl.col("p").top_k(2).alias("t")).select(
        "ri", pl.col("t").list.get(0).alias("p1st"), pl.col("t").list.get(1, null_on_oob=True).fill_null(0).alias("p2nd"))
    P = P.join(top2, on="ri").with_columns(
        pl.when(pl.col("prk") == 1).then(pl.col("p2nd")).otherwise(pl.col("p1st")).alias("p_other"),
        (pl.col("p") / pl.max_horizontal(pl.lit(1.0), pl.col("psum_r"))).alias("q")).drop("p1st", "p2nd")
    own = P.filter((pl.col("prk") == 1) & (pl.col("p") >= OWN_MIN)).join(r.select("ri", "src3"), on="ri") \
           .select("ri", "si", "q", "src3")
    cl = own.group_by("si").agg(pl.col("q").sum().alias("cl_mass"), pl.len().alias("cl_n"),
                                (pl.col("src3") == 0).sum().alias("cl_n2"), (pl.col("src3") == 1).sum().alias("cl_n3"))
    Q = P.filter(pl.col("p") >= P_MIN).join(cl, on="si", how="left").with_columns(pl.col("^cl_.*$").fill_null(0))
    Q = Q.join(own.select("ri", "si", pl.col("q").alias("q_self")), on=["ri", "si"], how="left") \
         .with_columns(pl.col("q_self").fill_null(0)).with_columns(
        (pl.col("cl_mass") - pl.col("q_self")).alias("cl_mass_ex"),
        (pl.col("cl_n") - (pl.col("q_self") > 0).cast(pl.UInt32)).alias("cl_n_ex")).drop("q_self")
    # sibling evidence: every stage-2 pair x every other confident row of the same S1 (bucketed by S1 to bound RAM)
    rr = r.select("ri", "rnm", "ad", "nums", "src3")
    rj = rr.rename({"ri": "rj", "rnm": "rnm_j", "ad": "ad_j", "nums": "nums_j", "src3": "src3_j"})
    ns = s1.select("si", pl.col("nums").alias("nums_s"))
    # rival rows of s: plausible candidates (p >= RIV_MIN) that are NOT its confident owners. Near-twin distractor
    # entities come as groups sharing their own house number / name variant; a true row with a perturbed number rarely has one.
    riv = P.filter((pl.col("p") >= RIV_MIN) & ~((pl.col("prk") == 1) & (pl.col("p") >= OWN_MIN))) \
           .select(pl.col("ri").alias("rj"), "si", pl.col("p").alias("pj"))
    ags, rags = [], []
    for b in range(SIB_BUCKETS):
        rv = Q.select("ri", "si").filter(pl.col("si") % SIB_BUCKETS == b).join(riv, on="si") \
              .filter(pl.col("ri") != pl.col("rj")).join(rr, on="ri").join(rj, on="rj").join(ns, on="si")
        rv = rv.with_columns(
            pl.Series("sn", _cp(fuzz.token_set_ratio, rv["rnm"].to_list(), rv["rnm_j"].to_list())),
            (pl.col("nums").list.set_intersection("nums_j").list.set_difference("nums_s").list.len() > 0).alias("pert"))
        rags.append(rv.group_by("ri", "si").agg(
            pl.len().alias("riv_n"), pl.col("pj").sum().alias("riv_mass"), (pl.col("sn") >= 95).sum().alias("riv_n95"),
            pl.col("pert").sum().alias("riv_pert"), (pl.col("pert") & (pl.col("sn") >= 85)).sum().alias("riv_twin")))
        qb = Q.select("ri", "si").filter(pl.col("si") % SIB_BUCKETS == b)
        sib = qb.join(own.select(pl.col("ri").alias("rj"), "si", pl.col("q").alias("qj")), on="si") \
                .filter(pl.col("ri") != pl.col("rj")).join(rr, on="ri").join(rj, on="rj").join(ns, on="si")
        sib = sib.with_columns(
            pl.Series("sn", _cp(fuzz.token_set_ratio, sib["rnm"].to_list(), sib["rnm_j"].to_list())),
            pl.Series("sa", _cp(fuzz.token_set_ratio, sib["ad"].to_list(), sib["ad_j"].to_list())),
            (pl.col("nums").list.set_intersection("nums_j").list.set_difference("nums_s").list.len() > 0).alias("pert"),
            (pl.col("src3") == pl.col("src3_j")).alias("same_src"))
        ags.append(sib.group_by("ri", "si").agg(
            pl.col("sn").max().alias("sib_n_max"), pl.col("sa").max().alias("sib_a_max"),
            (pl.col("sn") * pl.col("qj")).max().alias("sib_nq_max"), (pl.col("sa") * pl.col("qj")).max().alias("sib_aq_max"),
            (pl.col("sn") >= 95).sum().alias("sib_n95"), (pl.col("sa") >= 95).sum().alias("sib_a95"),
            pl.col("pert").sum().alias("sib_pert"), (pl.col("pert") & pl.col("same_src")).sum().alias("sib_pert_same")))
    Q = Q.join(pl.concat(ags), on=["ri", "si"], how="left").with_columns(pl.col("^sib_.*$").fill_null(-1))
    Q = Q.join(pl.concat(rags), on=["ri", "si"], how="left").with_columns(pl.col("^riv_.*$").fill_null(0))
    # relative evidence across the row's competing S1s: which same-named S1 has the siblings that look like this row?
    return Q.with_columns(
        (pl.col("sib_nq_max") - pl.col("sib_nq_max").max().over("ri")).alias("gap_sib_nq"),
        (pl.col("sib_aq_max") - pl.col("sib_aq_max").max().over("ri")).alias("gap_sib_aq"),
        (pl.col("sib_pert") - pl.col("sib_pert").max().over("ri")).alias("gap_sib_pert"),
        (pl.col("cl_mass_ex") - pl.col("cl_mass_ex").max().over("ri")).alias("gap_cl_mass"),
        pl.col("sib_nq_max").rank("ordinal", descending=True).over("ri").cast(pl.Int16).alias("rk_sib"))
