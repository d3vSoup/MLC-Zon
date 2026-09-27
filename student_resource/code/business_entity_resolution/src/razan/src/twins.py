"""Test-matched synthetic twin world (train only).

Test S2/S3 hold near-twin distractor entities that train barely contains (measured label-free on test, AUDIT §8.3-8.4):
  N-twins  identical name + a legal suffix added (94%), house number shifted by a small delta (89% positive, |d| median 7),
           30% of groups have 2 rows sharing the shifted number. US test cell excess ~72 rows/1k S1; India ~1 (-> rate 0).
  G-twins  same address, one generic name word swapped for another (France "same number, different name" cell 6-12x).
Twins are copies of REAL true rows of live train S1s (all generator noise kept), relabelled as distractors and appended
at the END of n_train_s3 / train_s3, so every existing ri stays valid; they are then blocked like any other row.

BER_WORK=<work> BER_DROP=16 python twins.py gen     # -> $BER_TWIN_OUT/{n_train_s3,train_s3,twin_meta}.parquet
BER_WORK=<work> python twins.py block               # -> $BER_TWIN_OUT/cand_train/{country}_T_000.parquet
Rates per country: BER_TWIN_N / BER_TWIN_G = "country:fraction of eligible live parents,..." (from the test excess).
"""
import os
import re
import shutil
import sys
import unicodedata

import numpy as np
import polars as pl

from common import w, raw, gt_pairs, dropped, SEED
from norm import LEGAL
from stage2 import GEN_DF

OUT = os.environ.get("BER_TWIN_OUT", w("twin"))
rates = lambda s: {c: float(x) for c, x in (kv.split(":") for kv in s.split(",") if kv)}
RATE_N = rates(os.environ.get("BER_TWIN_N", "US:0.072"))
RATE_G = rates(os.environ.get("BER_TWIN_G", "US:0.012,India:0.012"))
SUF = ["LLC", "Inc", "Co", "Corp", "Ltd", "LP"]                  # raw spelling; normalised = lower()
DBIN = [(1, 2, 9604), (3, 5, 12288), (6, 10, 7809), (11, 20, 8120), (21, 50, 4105)]  # test |delta| histogram (US twins)
P_POS, P_SUF, P_PAIR = 0.89, 0.94, 0.3


def _norm_tok(t):
    t = unicodedata.normalize("NFKD", t.lower())
    return re.sub(r"[^\w]|_", "", "".join(ch for ch in t if not unicodedata.combining(ch)))


def _style(new, like):
    return new.upper() if like.isupper() else new.title() if like[:1].isupper() else new


def load():
    s1 = pl.read_parquet(w("n_train_s1.parquet"), columns=["id", "country", "nm", "core", "nums"]) \
           .with_row_index("si").with_columns(pl.col("si").cast(pl.Int32))
    r = pl.concat([pl.read_parquet(w(f"n_train_s{k}.parquet")) for k in (2, 3)]).with_row_index("ri") \
          .with_columns(pl.col("ri").cast(pl.Int32))
    g = gt_pairs().join(s1.select("si", pl.col("id").alias("s1")), on="s1").join(r.select("ri", "id"), on="id").select("si", "ri")
    live = s1.filter(~dropped("si"))
    return s1, live, r, g


def pick(par, rate, rng):
    """Sample parents per country at the given rate; group size 1 or 2."""
    par = par.sort("si").with_columns(pl.Series("u", rng.random(par.height)),
                                      pl.Series("gs", np.where(rng.random(par.height) < P_PAIR, 2, 1)))
    return par.filter(pl.col("u") < pl.col("country").replace_strict(rate, default=0.0, return_dtype=pl.Float64)).drop("u")


def templates(cand, par, rng):
    """Up to gs random true rows per chosen parent."""
    c = cand.join(par, on="si")
    c = c.with_columns(pl.Series("v", rng.random(c.height)))
    return c.with_columns(pl.col("v").rank("ordinal").over("si").alias("k")).filter(pl.col("k") <= pl.col("gs")).drop("v", "k")


def gen():
    rng = np.random.default_rng(SEED)
    s1, live, r, g = load()
    tr = g.join(live.select("si", "country", "nm", "core", pl.col("nums").list.first().alias("n1")), on="si") \
          .join(r.select("ri", "id", pl.col("nums").alias("rn"), pl.col("core").alias("rcore"), "f_noaddr"), on="ri") \
          .filter(~pl.col("f_noaddr"))
    rows = []  # (template ri, kind, parent si, delta, suffix, t, t2)

    # N-twins: the row's address must contain the parent's first house number
    cn = tr.filter(pl.col("n1").is_not_null() & (pl.col("n1").str.len_chars() <= 6) & pl.col("rn").list.contains(pl.col("n1")))
    par = pick(cn.select("si", "country").unique(), RATE_N, rng)
    b = rng.choice(len(DBIN), par.height, p=np.array([x[2] for x in DBIN]) / sum(x[2] for x in DBIN))
    mag = rng.integers(np.array([x[0] for x in DBIN])[b], np.array([x[1] for x in DBIN])[b] + 1)
    par = par.with_columns(pl.Series("mag", mag), pl.Series("pos", rng.random(par.height) < P_POS),
                           pl.Series("sfx", rng.integers(0, len(SUF), par.height)), pl.Series("add", rng.random(par.height) < P_SUF))
    for t in templates(cn.select("si", "ri", "nm", "n1"), par, rng).iter_rows(named=True):
        n1 = int(t["n1"])
        d = t["mag"] if (t["pos"] or n1 - t["mag"] < 1) else -t["mag"]
        legal1 = set(t["nm"].split()) & set(LEGAL)
        sfx = next((SUF[(t["sfx"] + i) % len(SUF)] for i in range(len(SUF)) if SUF[(t["sfx"] + i) % len(SUF)].lower() not in legal1), None)
        rows.append((t["ri"], "N", t["si"], d, sfx if t["add"] else None, None, None))

    # G-twins: swap one generic word of the row's core (a distinctive word must remain), same address
    voc = {}
    for (c,), d in s1.select("country", pl.col("core").str.split(" ").list.unique().alias("t")).explode("t").group_by("country"):
        n = s1.filter(pl.col("country") == c).height
        df = d.group_by("t").len().filter(pl.col("len") >= GEN_DF * n)
        voc[c] = (df["t"].to_list(), (df["len"] / df["len"].sum()).to_numpy())
    cg = tr.filter(pl.col("rcore").str.contains(" "))
    parg = pick(cg.select("si", "country").unique(), RATE_G, rng)
    for (si,), grp in templates(cg.select("si", "ri", "rcore", "core", "country"), parg, rng).group_by("si"):
        c = grp["country"][0]
        words, wts = voc[c]
        vs = set(words)
        common = set.intersection(*(set(x.split()) for x in grp["rcore"].to_list())) & vs
        if not common or not all(set(x.split()) - vs for x in grp["rcore"].to_list()):
            continue
        t = sorted(common)[rng.integers(len(common))]
        block = set(grp["core"][0].split()) | {t}
        for _ in range(10):
            t2 = words[rng.choice(len(words), p=wts)]
            if t2 not in block:
                break
        else:
            continue
        rows += [(ri, "G", si, None, None, t, t2) for ri in grp["ri"].to_list()]

    meta = pl.DataFrame(rows, schema={"tri": pl.Int32, "kind": pl.String, "si_parent": pl.Int32, "delta": pl.Int32,
                                      "suffix": pl.String, "t": pl.String, "t2": pl.String}, orient="row").sort("kind", "tri", descending=[True, False])
    base = r.join(meta.select("tri").unique(), left_on="ri", right_on="tri", how="semi")
    rawr = pl.concat([raw("train", k) for k in (2, 3)]).join(base.select(pl.col("id").alias("entity_id")), on="entity_id", how="semi")
    tmpl = {x["ri"]: x for x in base.iter_rows(named=True)}
    rawd = {x["entity_id"]: x for x in rawr.iter_rows(named=True)}
    n1s = dict(s1.select("si", pl.col("nums").list.first()).iter_rows())
    norm_rows, raw_rows, keep = [], [], []
    for i, m in enumerate(meta.iter_rows(named=True)):
        t, rw = dict(tmpl[m["tri"]]), dict(rawd[tmpl[m["tri"]]["id"]])
        name, addr = rw["business_name"] or "", rw["business_address"] or ""
        if m["kind"] == "N":
            n1 = n1s[m["si_parent"]]
            new = str(int(n1) + m["delta"])
            pat = re.compile(r"(?<!\d)0*" + n1 + r"(?!\d)")
            if not pat.search(addr) or not pat.search(t["ad"]):
                continue
            addr, t["ad"] = pat.sub(new, addr, count=1), pat.sub(new, t["ad"], count=1)
            t["nums"] = [new if x == n1 else x for x in t["nums"]]
            if m["suffix"]:
                name += " " + (m["suffix"].upper() if name.isupper() else m["suffix"])
                t["nm"] += " " + m["suffix"].lower()
        else:
            ws = name.split(" ")
            j = next((k for k, x in enumerate(ws) if _norm_tok(x) == m["t"]), None)
            if j is None:
                continue
            ws[j] = _style(m["t2"], ws[j])
            name = " ".join(ws)
            sw = lambda s: " ".join(m["t2"] if x == m["t"] else x for x in s.split(" "))
            t["nm"], t["core"] = sw(t["nm"]), sw(t["core"])
        tid = f"{t['id'][:2]}-T{i:07d}"
        t["id"] = tid
        norm_rows.append(t)
        raw_rows.append({**rw, "entity_id": tid, "business_name": name, "business_address": addr})
        keep.append(i)
    meta = meta.with_row_index("i").filter(pl.col("i").is_in(keep)).drop("i")
    n3 = pl.read_parquet(w("n_train_s3.parquet"))
    tw = pl.DataFrame(norm_rows, infer_schema_length=None).select(n3.columns).cast(n3.schema)
    n2h = pl.scan_parquet(w("n_train_s2.parquet")).select(pl.len()).collect().item()
    meta = meta.with_columns(pl.Series("ri", np.arange(tw.height, dtype=np.int32) + n2h + n3.height), pl.Series("id", tw["id"]))
    os.makedirs(OUT, exist_ok=True)
    pl.concat([n3, tw]).write_parquet(os.path.join(OUT, "n_train_s3.parquet"))
    r3 = raw("train", 3)
    pl.concat([r3, pl.DataFrame(raw_rows, infer_schema_length=None).select(r3.columns).cast(r3.schema)]).write_parquet(os.path.join(OUT, "train_s3.parquet"))
    meta.write_parquet(os.path.join(OUT, "twin_meta.parquet"))
    live_n = dict(live.group_by("country").len().iter_rows())
    for (k, c), d in meta.join(s1.select(pl.col("si").alias("si_parent"), "country"), on="si_parent").group_by("kind", "country"):
        print(f"{k}-twins {c}: {d.height} rows, {d['si_parent'].n_unique()} parents, {1000 * d.height / live_n[c]:.1f} rows/1k live S1", flush=True)
    n = meta.filter(pl.col("kind") == "N")
    print("N delta |d|<=20", round((n["delta"].abs() <= 20).mean(), 3), "d>0", round((n["delta"] > 0).mean(), 3),
          "suffix", round(n["suffix"].is_not_null().mean(), 3), "| ri range", meta["ri"].min(), meta["ri"].max(),
          "| G from", meta.filter(pl.col("kind") == "G")["ri"].min(), flush=True)


def block_twins():
    import block
    meta = pl.read_parquet(os.path.join(OUT, "twin_meta.parquet"))
    tw = pl.read_parquet(os.path.join(OUT, "n_train_s3.parquet")).tail(meta.height)
    name = pl.when(pl.col("al_r").is_not_null() & (block.core_of("al_r") != "")).then(block.core_of("al_r")).otherwise(pl.col("core"))
    q = tw.select("country", name.str.replace_all(" ", "").alias("n"), pl.col("ad").alias("a")) \
          .with_columns(pl.Series("ri", meta["ri"]))
    s1 = block.load("train", 1).with_row_index("si")
    os.makedirs(os.path.join(OUT, "cand_train"), exist_ok=True)
    os.makedirs(w("cand_train"), exist_ok=True)
    for (c,), b in q.group_by("country"):
        a = s1.filter(pl.col("country") == c)
        block._G.update(split="train", c=f"{c}_T", q=b, si_map=a["si"].to_numpy(),
                        v1=block.View(a, block.CAPS[0]), v2=block.View(a, block.CAPS[1]))
        for n in range((b.height + block.CHUNK - 1) // block.CHUNK):
            block._chunk(n)
            f = f"{c}_T_{n:03d}.parquet"
            shutil.move(w(f"cand_train/{f}"), os.path.join(OUT, "cand_train", f))
        print(f"blocked {c}: {b.height} twin rows", flush=True)


if __name__ == "__main__":
    {"gen": gen, "block": block_twins}[sys.argv[1]]()
