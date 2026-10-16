"""Normalise every source file -> work/n_{split}_s{k}.parquet.

Learned from train pairs only: a native-script -> Latin token table (names by position
alignment, addresses by co-occurrence). Unknown native tokens fall back to anyascii.
"""
import os
import sys
import time
from collections import Counter, defaultdict

import polars as pl
from anyascii import anyascii

from common import w, raw, gt_pairs

NATIVE = r"[\x{0900}-\x{0DFF}]"
LEGAL = ("inc incorporated llc lc ltd limited pvt private corp corporation co company llp lp pllc pc plc esq "
         "sarl sas sasu sci sa eurl snc selarl cie gmbh ets").split()
FILLER = ("the and of dr mr mrs ms shri sri smt m s center centre services service group associates fils "
          "associes de du des la le les l d et au aux en sur").split()
# "<invented brand> <marker> <real name>" -> real name is the right part
ALIAS = r"\s(?:aka|a k a|nee|doing business as|dba|d b a|formerly known as|formerly|fka|f k a|t a|trading as)\s"
ABBR = {  # address token -> canonical (applied to both sides, so direction does not matter)
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "blvd": "boulevard", "bd": "boulevard", "pl": "place", "trl": "trail",
    "pkwy": "parkway", "hwy": "highway", "sq": "square", "apt": "apartment", "ste": "suite", "fl": "floor",
    "flr": "floor", "bldg": "building", "twp": "township", "mt": "mount", "ft": "fort", "hts": "heights",
    "rte": "route", "expy": "expressway", "fwy": "freeway", "ctr": "center", "centre": "center", "plz": "plaza",
    "ter": "terrace", "cres": "crescent", "pt": "point", "ext": "extension", "extn": "extension",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "r": "rue", "all": "allee", "che": "chemin", "imp": "impasse",
    "fbg": "faubourg", "crs": "cours", "bis": "b", "city": "", "cdp": "", "no": "", "door": "", "hno": "",
    "hn": "", "h": "", "bombay": "mumbai", "gurgaon": "gurugram", "bengaluru": "bangalore",
}
STATES = {  # full name -> code, every country (matching is within-country, so code collisions are harmless)
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "south carolina": "sc", "south dakota": "sd", "rhode island": "ri",
    "west virginia": "wv", "district of columbia": "dc", "puerto rico": "pr",
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "goa", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mnp",
    "meghalaya": "mgl", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
TOKMAP = {**ABBR, **{k: v for k, v in STATES.items() if " " not in k}}


def _is_native(tok):
    return any("ऀ" <= ch <= "෿" for ch in tok)


def _strip(e):
    return e.str.normalize("NFKD").str.replace_all(r"\p{M}", "")


def learn_translit():
    """Native token -> latin token, from train true pairs only."""
    s1 = raw("train", 1).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                                  pl.col("business_address").alias("a1"))
    oth = pl.concat([raw("train", s).select(pl.col("entity_id").alias("id"), pl.col("business_name").alias("n2"),
                                              pl.col("business_address").alias("a2")) for s in (2, 3)])
    oth = oth.filter(pl.col("n2").str.contains(NATIVE) | pl.col("a2").str.contains(NATIVE))
    p = gt_pairs().join(oth, on="id").join(s1, on="s1")
    del s1, oth
    names = defaultdict(Counter)  # position alignment when token counts agree
    for n1, n2 in p.filter(pl.col("n2").str.contains(NATIVE)).select("n1", "n2").iter_rows():
        a, b = n1.lower().split(), n2.split()
        if len(a) == len(b):
            for x, y in zip(a, b):
                if _is_native(y):
                    names[y][x.strip(".,()[]")] += 1
    # addresses: argmax_t P(latin t in a1 | native y in a2), ties -> rarer t; keep if P >= 0.9
    pa = p.filter(pl.col("a2").str.contains(NATIVE)).select("a1", "a2")
    pa = pa.sample(n=min(400_000, pa.height), seed=1)
    df_lat, nat_n, co = Counter(), Counter(), defaultdict(Counter)
    for a1, a2 in pa.iter_rows():
        lat = set(a1.lower().replace(",", " ").split())
        df_lat.update(lat)
        for y in set(a2.replace(",", " ").split()):
            if _is_native(y):
                nat_n[y] += 1
                co[y].update(lat)
    out = {}
    for y, c in co.items():
        if nat_n[y] >= 5:
            t, n = max(c.items(), key=lambda kv: (kv[1], -df_lat[kv[0]]))
            if n / nat_n[y] >= 0.9:
                out[y] = t
    for y, c in names.items():  # names win for shared tokens
        out[y] = c.most_common(1)[0][0]
    t = pl.DataFrame({"tok": list(out), "lat": list(out.values())})
    t.write_parquet(w("translit.parquet"))
    print(f"translit: {len(names)} name tokens, {t.height} total", flush=True)
    return t


def _translit(df, col, table):
    """Replace native-script whitespace tokens of `col` using table, fallback anyascii."""
    sub = df.filter(pl.col(col).str.contains(NATIVE)).select("entity_id", col)
    if sub.height == 0:
        return df
    toks = sub.with_columns(pl.col(col).str.replace_all(",", " , ").str.split(" ")).explode(col).with_row_index("_i")
    uniq = toks.filter(pl.col(col).str.contains(NATIVE)).select(pl.col(col).alias("tok")).unique() \
               .join(table, on="tok", how="left")
    miss = uniq.filter(pl.col("lat").is_null())["tok"].to_list()
    fb = pl.DataFrame({"tok": miss, "fb": [anyascii(x).lower() for x in miss]}, schema={"tok": pl.String, "fb": pl.String})
    uniq = uniq.join(fb, on="tok", how="left").select("tok", pl.coalesce("lat", "fb").alias("lat"))
    toks = toks.join(uniq, left_on=col, right_on="tok", how="left").sort("_i") \
               .with_columns(pl.coalesce("lat", col).alias(col))
    new = toks.group_by("entity_id", maintain_order=True).agg(pl.col(col).str.join(" ").alias("_new"))
    return df.join(new, on="entity_id", how="left").with_columns(pl.coalesce("_new", col).alias(col)).drop("_new")


def norm_names(e):
    e = e.fill_null("").str.to_lowercase()
    e = e.str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ").str.replace_all(r"\|.*$", " ") \
         .str.replace_all(r"https?://|www\.", " ").str.replace_all(r"\d{7,}", " ")
    e = _strip(e).str.replace_all(r"[&+]", " and ")
    e = e.str.replace_all(r"(\p{L})0", "${1}o").str.replace_all(r"0(\p{L})", "o${1}")
    e = e.str.replace_all(r"\.(com|net|org|in|co|fr|biz|info|us|io)\b", " ")
    return e.str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def _dedupe(tok_list):
    return tok_list.list.eval(pl.element().filter(pl.element().ne_missing(pl.element().shift(1))))


def _collapse(s):
    """Join runs of single letters: 's a r l' -> 'sarl', 'l l c' -> 'llc' (dotted legal forms)."""
    out, buf = [], ""
    for t in s.split():
        if len(t) == 1 and t.isalpha():
            buf += t
            continue
        if buf:
            out.append(buf)
            buf = ""
        out.append(t)
    return " ".join(out + ([buf] if buf else []))


def _components(e):
    """Address -> list of normalised comma-separated components."""
    return _strip(e.fill_null("").str.to_lowercase()).str.split(",").list.eval(
        pl.element().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().filter(pl.element() != ""))


def admin_vocab(split, table):
    """Per country, address components at 'state/region' level, learned from this split's own rows:
    S1 components that sit last >=70% of the time (states, regions), plus frequent last components of
    S2/S3 never seen in S1 (departements, state spellings). Removing them neutralises admin-level swaps."""
    p = w(f"admin_{split}.parquet")
    if os.path.exists(p):
        return pl.read_parquet(p)
    def comps(s):
        d = _translit(raw(split, s), "business_address", table)
        return d.select("country", _components(pl.col("business_address")).alias("c")) \
                .with_columns(pl.col("c").list.last().alias("last"))
    s1 = comps(1)
    tot = s1.select("country", "c").explode("c").group_by("country", "c").len()
    last = s1.group_by("country", "last").len().rename({"last": "c", "len": "n_last"})
    a1 = last.join(tot, on=["country", "c"]).filter((pl.col("n_last") >= 200) & (pl.col("n_last") >= 0.7 * pl.col("len")))
    o = pl.concat([comps(2), comps(3)]).group_by("country", "last").len().rename({"last": "c"})
    a2 = o.filter(pl.col("len") >= 200).join(tot.select("country", "c"), on=["country", "c"], how="anti")
    v = pl.concat([a1.select("country", "c"), a2.select("country", "c")]).unique().drop_nulls() \
          .filter(~pl.col("c").str.contains(r"\d") & (pl.col("c").str.len_chars() >= 2))
    v.write_parquet(p)
    print(f"admin vocab {split}: {v.group_by('country').len().rows()}", flush=True)
    return v


def prep(split, s, table):
    t0 = time.time()
    df = raw(split, s)
    df = df.with_columns(pl.col("business_name").fill_null("").str.contains(NATIVE).alias("f_native"))
    df = _translit(_translit(df, "business_name", table), "business_address", table)
    nraw = pl.col("business_name").fill_null("")
    df = df.with_columns(
        nraw.str.contains(r"@|#").alias("f_handle"),
        nraw.str.to_lowercase().str.contains(r"\.(com|net|org|in|co|fr|biz|info|us|io)\b|www\.").alias("f_domain"),
        _dedupe(norm_names(pl.col("business_name")).str.split(" ")).list.join(" ").alias("nm"),
    )
    df = df.with_columns(pl.when(pl.col("nm").str.contains(r"\b\p{L} \p{L}\b"))
                         .then(pl.col("nm").map_elements(_collapse, return_dtype=pl.String)).otherwise(pl.col("nm")).alias("nm"))
    df = df.with_columns(
        pl.col("nm").str.contains(ALIAS).alias("f_alias"),
        pl.col("nm").str.extract(r"^(.*?)" + ALIAS, 1).alias("al_l"),
        pl.col("nm").str.extract(ALIAS + r"(.*)$", 1).alias("al_r"),
        pl.col("nm").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(LEGAL + FILLER)))
          .list.join(" ").alias("core"),
    )
    df = df.with_columns(pl.when(pl.col("core") == "").then(pl.col("nm")).otherwise(pl.col("core")).alias("core"))
    a = pl.col("business_address").fill_null("").str.to_lowercase()
    a = a.str.replace_all(r"<null>|\bnull\b|\bn/a\b|\bnone\b", " ").str.replace_all(r"\bn\s*[°º]", " no ")
    comps = _components(a)
    kept = comps
    for c, v in admin_vocab(split, table).group_by("country"):
        kept = pl.when(pl.col("country") == c[0]).then(comps.list.eval(pl.element().filter(~pl.element().is_in(v["c"].to_list())))) \
                 .otherwise(kept)
    df = df.with_columns(kept.list.join(" ").alias("_a"))
    a = pl.col("_a")
    for full in sorted(STATES, key=len, reverse=True):
        if " " in full:
            a = a.str.replace_all(r"\b" + full + r"\b", STATES[full])
    df = df.with_columns(
        a.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
         .list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True).alias("nums"),
        _dedupe(a.str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
                .list.eval(pl.element().replace(TOKMAP)).list.eval(pl.element().filter(pl.element() != "")))
        .list.join(" ").alias("ad"),
    )
    df = df.with_columns((pl.col("ad") == "").alias("f_noaddr"))
    out = df.select(pl.col("entity_id").alias("id"), "country", "nm", "core", "al_l", "al_r", "ad", "nums",
                    "f_handle", "f_domain", "f_native", "f_alias", "f_noaddr")
    out.write_parquet(w(f"n_{split}_s{s}.parquet"))
    print(f"{split} s{s}: {out.height} rows {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    table = pl.read_parquet(w("translit.parquet")) if os.path.exists(w("translit.parquet")) else learn_translit()
    for job in sys.argv[1:] or ["admin"] + [f"{sp}{s}" for sp in ("train", "test") for s in (1, 2, 3)]:
        if job == "admin":  # learn per-split admin vocab once, so file jobs can then run in parallel
            admin_vocab("train", table), admin_vocab("test", table)
        else:
            prep(job[:-1], int(job[-1]), table)
