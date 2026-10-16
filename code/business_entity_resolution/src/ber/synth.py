import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from ber.dictionaries import ADDR_FR, ADDR_TOKENS, LEGAL, PLACE_PHRASES
from ber.io import CFG, read_truth, read_tsv, source_file, work

# self-supervised synthetic pairs: S1 is deduplicated, so two S1 of one country are different businesses (true
# negatives); positives = noisy copies of an S1 made by a noise model whose rates are measured on US+IN true pairs.
SY = CFG.get("synth", {})
WORD = re.compile(r"[^\W_]+", re.U)
LEET = {"l": "1", "i": "1", "o": "0", "s": "5", "e": "3", "a": "4"}
ACC = dict(zip("aeiouAEIOU", "áéíóúÁÉÍÓÚ"))
NUM = re.compile(r"\d+")
# street-type forms grouped by canonical short form (dictionary-driven: blvd/bd/boul, ave/av, rue/r, ...)
FORMS = {}
for k, v in list(ADDR_TOKENS.items()) + [(v, k) for k, v in ADDR_FR.items()]:
    FORMS.setdefault(v, {v}).add(k)
LONG = {max(g, key=len): sorted(g - {max(g, key=len)}) for g in FORMS.values() if len(g) > 1}
SHORT = {s: l for l, ss in LONG.items() for s in ss}
STATE = {**{k: v for k, v in ADDR_TOKENS.items() if len(v) == 2 and len(k) > 3}, **PLACE_PHRASES}
STATE_INV = {v: k for k, v in STATE.items()}


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def words(s):
    return WORD.findall(fold(s))


def latin(s):
    return all(ord(c) < 0x250 for c in s)


def best(t, pool):
    return max((fuzz.ratio(t, p) for p in pool), default=0)


# measure transformation rates (per source) on a fixed sample of US+IN true pairs; also empirical pools
def estimate(n=40_000):
    rng = np.random.default_rng(CFG["seed"])
    _, long = read_truth()
    long = long.iloc[rng.choice(len(long), min(4 * n, len(long)), replace=False)]
    need = set(long.s1_id) | set(long.cand_id)  # read only the sampled rows (low RAM)
    src = pd.concat([pq.read_table(work("train", f"norm_s{s}.parquet"), columns=["entity_id", "name_raw", "addr_raw"],
                                   filters=[("entity_id", "in", list(need))]).to_pandas() for s in (1, 2, 3)])
    src = src.rename(columns={"name_raw": "business_name", "addr_raw": "business_address"}).set_index("entity_id")
    M = {}
    for s in ("S2", "S3"):
        pr = long[long.cand_id.str.startswith(s)].head(n)
        a, b = src.loc[pr.s1_id], src.loc[pr.cand_id]
        c = {k: 0 for k in ("pairs", "latin", "up", "low", "acc", "leet", "brand", "dba", "hash", "dom", "dots", "br",
                            "core", "drop", "typo", "garble", "swap_n", "swap", "leg", "ldrop", "lswap", "lbr",
                            "lfront", "nolegal", "ladd", "suf", "a_up", "a_low", "parts_num", "drop_num", "parts_txt",
                            "drop_txt", "shuf_n", "shuf", "long", "abbr", "short", "expand", "st", "st_sw", "hn", "hn_same",
                            "hn_zpad", "hn_trunc", "hn_drop", "hn_change", "hn_pre", "atok", "atypo")}
        suf, brand, ladd, pre = {}, [], {}, {}
        for n1, a1, n2, a2 in zip(a.business_name, a.business_address, b.business_name, b.business_address):
            c["pairs"] += 1
            for _ in [0] if latin(n2) else []:  # name stats (break = brand-replaced name: stop name stats only)
                c["latin"] += 1
                c["up"] += n2.isupper() and not n1.isupper()
                c["low"] += n2.islower() and not n1.islower()
                c["acc"] += fold(n2) != n2.lower() and fold(n1) == n1.lower()
                c["leet"] += bool(re.search(r"[a-z][0-9]|[0-9][a-z]", n2.lower())) and not re.search(r"[a-z][0-9]|[0-9][a-z]", n1.lower())
                c["hash"] += n2.startswith("#")
                c["dom"] += bool(re.search(r"\.(com|net|org|in|co)\b", n2.lower())) and ".com" not in n1.lower()
                c["dots"] += n2.startswith("...")
                t1, t2 = words(n1), words(n2)
                core, leg = [t for t in t1 if t not in LEGAL], [t for t in t1 if t in LEGAL]
                l2 = [t for t in t2 if t in LEGAL]
                hit = {t: (t in t2) or best(t, t2) >= 80 for t in core}
                if core and not any(hit.values()) and not c["hash"]:
                    c["brand"] += 1
                    if re.search(r"\bd/?b/?a\b", n2.lower()):
                        c["dba"] += 1
                    elif 0 < len(t2) <= 2:
                        brand.append(n2)
                    break
                c["br"] += bool(re.search(r"[\(\[]", n2)) and not re.search(r"[\(\[]", n1)
                used = set()
                for t in core:
                    c["core"] += 1
                    if t in t2:
                        used.add(t)
                    elif best(t, t2) >= 80:
                        c["typo"] += 1
                        used.add(max(t2, key=lambda x: fuzz.ratio(t, x)))
                    else:
                        free = [x for x in t2 if x not in core and x not in LEGAL and abs(len(x) - len(t)) <= 1]
                        c["garble" if free else "drop"] += 1
                        used.update(free[:1])
                kept = [t for t in core if t in t2]
                if len(kept) >= 2:
                    c["swap_n"] += 1
                    c["swap"] += [t2.index(t) for t in kept] != sorted(t2.index(t) for t in kept)
                if leg:
                    c["leg"] += 1
                    if not l2:
                        c["ldrop"] += 1
                    elif not set(leg) & set(l2):
                        c["lswap"] += 1
                    c["lbr"] += bool(re.search(r"[\(\[]\s*(" + "|".join(leg) + r")", fold(n2)))
                    c["lfront"] += bool(t2) and t2[0] in LEGAL and t1[0] not in LEGAL
                else:
                    c["nolegal"] += 1
                    if l2:
                        c["ladd"] += 1
                        ladd[l2[-1]] = ladd.get(l2[-1], 0) + 1
                extra = [x for x in t2 if x not in used and x not in LEGAL and x not in t1 and not x.isdigit()]
                c["suf"] += bool(extra)
                for x in extra[:1]:
                    suf[x] = suf.get(x, 0) + 1
            # address
            c["a_up"] += a2.isupper() and not a1.isupper()
            c["a_low"] += a2.islower() and not a1.islower()
            p1 = [p.strip() for p in a1.split(",") if p.strip()]
            p2 = [fold(p.strip()) for p in a2.split(",") if p.strip()]
            idx = []
            for p in p1:
                num = bool(NUM.search(p))
                c["parts_num" if num else "parts_txt"] += 1
                sc = [fuzz.token_set_ratio(fold(p), q) for q in p2]
                j = int(np.argmax(sc)) if sc else -1
                if j < 0 or sc[j] < 80:
                    c["drop_num" if num else "drop_txt"] += 1
                else:
                    idx.append(j)
            if len(idx) >= 2:
                c["shuf_n"] += 1
                c["shuf"] += idx != sorted(idx)
            w1, w2 = words(a1), set(words(a2))
            for t in w1:
                if t in LONG:
                    c["long"] += 1
                    c["abbr"] += t not in w2 and any(s in w2 for s in LONG[t])
                elif t in SHORT:
                    c["short"] += 1
                    c["expand"] += t not in w2 and SHORT[t] in w2
                elif latin(a2) and len(t) > 3 and not t.isdigit():
                    c["atok"] += 1
                    c["atypo"] += t not in w2 and best(t, w2) >= 80
            f1 = fold(a1)
            for full, code in STATE.items():
                if re.search(rf"\b{re.escape(full)}\b", f1) or re.search(rf"(^|, ){code}(,|$)", f1):
                    c["st"] += 1
                    has_full, has_code = full in fold(a2), bool(re.search(rf"\b{code}\b", fold(a2)))
                    c["st_sw"] += (full in f1 and has_code and not has_full) or (full not in f1 and has_full)
                    break
            m1 = NUM.search(a1)
            if m1:
                h, n2s = m1.group(), NUM.findall(a2)
                c["hn"] += 1
                if h in n2s:
                    c["hn_same"] += 1
                elif any(x.lstrip("0") == h.lstrip("0") for x in n2s):
                    c["hn_zpad"] += 1
                elif any(len(x) < len(h) and (h.endswith(x) or h.startswith(x)) for x in n2s):
                    c["hn_trunc"] += 1
                elif not n2s:
                    c["hn_drop"] += 1
                else:
                    c["hn_change"] += 1
                m2 = re.search(r"(#+|\b[A-Za-z][A-Za-z\. ]{0,9}?)\s*#?\s*" + re.escape(h.lstrip("0") or "0") + r"\b", a2)
                if m2 and fold(m2.group(1)).strip(" .") not in f1:
                    q = m2.group(1).strip()
                    if q and (q.startswith("#") or fold(q).strip(" .") in ("no", "h no", "h.no", "door no", "plot", "unit", "house no", "flat no", "shop no", "sr no", "n", "num")):
                        c["hn_pre"] += 1
                        pre[q] = pre.get(q, 0) + 1
        r = lambda k, d: c[k] / max(c[d], 1)
        M[s] = {"n_pairs": c["pairs"], "up": r("up", "latin"), "low": r("low", "latin"), "acc": r("acc", "latin"),
                "leet": r("leet", "latin"), "brand": r("brand", "latin"), "dba": r("dba", "latin"),
                "hash": r("hash", "latin"), "dom": r("dom", "latin"), "dots": r("dots", "latin"), "br": r("br", "latin"),
                "drop": r("drop", "core"), "typo": r("typo", "core"), "garble": r("garble", "core"),
                "swap": r("swap", "swap_n"), "ldrop": r("ldrop", "leg"), "lswap": r("lswap", "leg"),
                "lbr": r("lbr", "leg"), "lfront": r("lfront", "leg"), "ladd": r("ladd", "nolegal"), "suf": r("suf", "latin"),
                "a_up": r("a_up", "pairs"), "a_low": r("a_low", "pairs"), "drop_num": r("drop_num", "parts_num"),
                "drop_txt": r("drop_txt", "parts_txt"), "shuf": r("shuf", "shuf_n"), "abbr": r("abbr", "long"),
                "expand": r("expand", "short"), "st_sw": r("st_sw", "st"), "atypo": r("atypo", "atok"),
                "hn_zpad": r("hn_zpad", "hn"), "hn_trunc": r("hn_trunc", "hn"), "hn_drop": r("hn_drop", "hn"),
                "hn_change": r("hn_change", "hn"), "hn_pre": r("hn_pre", "hn"),
                "pool_suf": sorted(((k, v) for k, v in suf.items() if len(k) >= 4), key=lambda x: -x[1])[:300],
                "pool_brand": sorted(set(brand))[:5000], "pool_ladd": sorted(ladd.items(), key=lambda x: -x[1])[:30],
                "pool_pre": sorted(pre.items(), key=lambda x: -x[1])[:30]}
    gt = read_tsv(source_file("train", 1).with_name("train_ground_truth.tsv"))
    ids = gt["matched_entity_ids"].str.split(",")
    n2 = ids.map(lambda l: sum(x.startswith("S2") for x in l if x)).values
    n3 = ids.map(lambda l: sum(x.startswith("S3") for x in l if x)).values
    vc = pd.Series(list(zip(n2.tolist(), n3.tolist()))).value_counts(normalize=True).head(80)
    M["counts"] = [[list(k), float(v)] for k, v in vc.items()]
    return M


def pick(rng, pool):
    items, w = zip(*pool)
    w = np.asarray(w, float)
    return items[rng.choice(len(items), p=w / w.sum())]


def typo(t, rng):
    if len(t) < 2:
        return t
    i = int(rng.integers(len(t)))
    op = rng.integers(4)
    ch = chr(ord("a") + int(rng.integers(26)))
    ch = ch.upper() if t[i].isupper() else ch
    if op == 0:
        return t[:i] + ch + t[i + 1:]
    if op == 1:
        return t[:i] + ch + t[i:]
    if op == 2:
        return t[:i] + t[i + 1:]
    j = min(i + 1, len(t) - 1)
    return t[:i] + t[j] + t[i] + t[j + 1:] if j > i else t


def garble(t, rng):
    s = "".join(chr(ord("a") + int(x)) for x in rng.integers(26, size=len(t)))
    return s.upper() if t.isupper() else s.capitalize() if t[:1].isupper() else s


# one noisy copy of (name, address) with source-specific rates m
def generate(name, addr, m, rng):
    u = rng.random
    toks = name.split()
    if u() < m["brand"] and m["pool_brand"]:
        b = m["pool_brand"][int(rng.integers(len(m["pool_brand"])))]
        name = f"{b} d/b/a {name}" if u() < m["dba"] / max(m["brand"], 1e-9) else b
    else:
        leg = [i for i, t in enumerate(toks) if fold(t).strip(".,") in LEGAL]
        if leg:
            if u() < m["ldrop"]:
                toks = [t for i, t in enumerate(toks) if i not in leg] or toks
            elif u() < m["lbr"]:
                toks[leg[-1]] = f"[{toks[leg[-1]]}]"
            elif u() < m["lfront"]:
                toks = [toks[leg[0]]] + [t for i, t in enumerate(toks) if i != leg[0]]
        elif u() < m["ladd"] and m["pool_ladd"]:
            toks.append(pick(rng, m["pool_ladd"]).title())
        out = []
        for t in toks:
            if fold(t).strip(".,[]") in LEGAL:
                out.append(t)
            elif u() < m["drop"] and len(toks) > 1:
                continue
            elif u() < m["garble"]:
                out.append(garble(t, rng))
            elif u() < m["typo"]:
                out.append(typo(t, rng))
            else:
                out.append(t)
        toks = out or toks
        if len(toks) >= 2 and u() < m["swap"]:
            i = int(rng.integers(len(toks) - 1))
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        if u() < m["suf"] and m["pool_suf"]:
            toks.append(pick(rng, m["pool_suf"]).title())
        if u() < m["br"] and len(toks) >= 2:
            toks[-1] = f"({toks[-1]})"
        name = " ".join(toks)
        if u() < m["dots"]:
            name = "... " + name
        if u() < m["hash"]:
            name = "#" + "".join(words(name)[:2])
        elif u() < m["dom"]:
            name = "".join(words(name)[:2]) + ".com"
    if u() < m["leet"]:
        pos = [i for i, ch in enumerate(name) if ch.lower() in LEET]
        if pos:
            i = pos[int(rng.integers(len(pos)))]
            name = name[:i] + LEET[name[i].lower()] + name[i + 1:]
    if u() < m["acc"]:
        pos = [i for i, ch in enumerate(name) if ch in ACC]
        if pos:
            i = pos[int(rng.integers(len(pos)))]
            name = name[:i] + ACC[name[i]] + name[i + 1:]
    r = u()
    name = name.upper() if r < m["up"] else name.lower() if r < m["up"] + m["low"] else name
    # address
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    keep = [p for p in parts if u() >= (m["drop_num"] if NUM.search(p) else m["drop_txt"])]
    parts = keep or parts[:1]
    if len(parts) >= 2 and u() < m["shuf"]:
        parts = [parts[i] for i in rng.permutation(len(parts))]
    out = []
    for p in parts:
        fp = fold(p)
        if fp in STATE and u() < m["st_sw"]:
            p = STATE[fp].upper()
        elif fp in STATE_INV and u() < m["st_sw"]:
            p = STATE_INV[fp].title()
        ws = []
        for w in p.split(" "):
            fw = fold(w).strip(".")
            if fw in LONG and u() < m["abbr"]:
                s = LONG[fw][int(rng.integers(len(LONG[fw])))]
                w = s.upper() if w.isupper() else s.capitalize()
            elif fw in SHORT and u() < m["expand"]:
                w = SHORT[fw].capitalize()
            elif len(fw) > 3 and not fw.isdigit() and u() < m["atypo"]:
                w = typo(w, rng)
            ws.append(w)
        out.append(" ".join(ws))
    addr = ", ".join(out)
    mh = NUM.search(addr)
    if mh:
        h = mh.group()
        r = u()
        cut = [m["hn_zpad"], m["hn_trunc"], m["hn_drop"], m["hn_change"]]
        if r < cut[0]:
            h2 = "0" * int(rng.integers(1, 3)) + h
        elif r < sum(cut[:2]) and len(h) > 1:
            h2 = h[int(rng.integers(1, len(h))):]
        elif r < sum(cut[:3]):
            h2 = ""
        elif r < sum(cut):
            h2 = str(int(rng.integers(1, 10 ** len(h))))
        else:
            h2 = h
        if h2 and u() < m["hn_pre"] and m["pool_pre"]:
            h2 = f"{pick(rng, m['pool_pre'])} {h2}"
        addr = (addr[:mh.start()] + h2 + addr[mh.end():]).strip(" ,")
    r = u()
    addr = addr.upper() if r < m["a_up"] else addr.lower() if r < m["a_up"] + m["a_low"] else addr
    return name, addr


# synthetic dataset dir (train split format) from S1 of <split>: a sampled universe per country, 1 - ghost share of it
# stays as S1, the rest are "ghost" businesses whose copies match nothing (like real unmatched S2/S3 records)
def build(split, out_root, frac=None, seed=None):
    frac, seed = frac or SY.get("frac", 0.5), seed if seed is not None else CFG["seed"]
    mf = work("synth", "noise_model.json")
    if not mf.exists():
        json.dump(estimate(), open(mf, "w"), indent=1)
    M = json.load(open(mf))
    rng = np.random.default_rng(seed)
    s1 = read_tsv(source_file(split, 1))
    s1 = s1[rng.random(len(s1)) < frac].reset_index(drop=True)
    ghost = rng.random(len(s1)) < SY.get("ghost", 0.26)
    # legal forms to add: those used in the country's own S1 names (unlabeled, so France gets its own forms)
    lg = {c: pd.Series([t for n in g for t in words(n) if t in LEGAL]).value_counts().head(30) for c, g in s1.groupby("country").business_name}
    for s in ("S2", "S3"):
        M[s]["by_country_ladd"] = {c: list(v.items()) for c, v in lg.items() if len(v)}
    counts = [tuple(c[0]) for c in M["counts"]]
    cw = np.array([c[1] for c in M["counts"]])
    ci = rng.choice(len(counts), len(s1), p=cw / cw.sum())
    recs = {2: [], 3: []}
    truth = []
    nid = 9_000_000_000
    for i, (eid, nm, ad, ct) in enumerate(zip(s1.entity_id, s1.business_name, s1.business_address, s1.country)):
        mine = []
        for s, k in zip((2, 3), counts[ci[i]]):
            for _ in range(k):
                m = M[f"S{s}"]
                n2, a2 = generate(nm, ad, {**m, "pool_ladd": m["by_country_ladd"].get(ct, m["pool_ladd"])}, rng)
                nid += 1
                recs[s].append((f"S{s}-{nid}", n2.replace("\t", " "), a2.replace("\t", " "), ct))
                mine.append(f"S{s}-{nid}")
        if not ghost[i]:
            truth.append((eid, ",".join(mine)))
    d = Path(out_root) / "dataset" / "train"
    d.mkdir(parents=True, exist_ok=True)
    s1[~ghost].to_csv(d / "train_source1.tsv", sep="\t", index=False, quoting=3)
    for s in (2, 3):
        pd.DataFrame(recs[s], columns=s1.columns).to_csv(d / f"train_source{s}.tsv", sep="\t", index=False, quoting=3)
    pd.DataFrame(truth, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        d / "train_ground_truth.tsv", sep="\t", index=False, quoting=3)
    print(f"synth {split}: S1 {int((~ghost).sum())} (+{int(ghost.sum())} ghosts), S2 {len(recs[2])}, S3 {len(recs[3])} -> {d}")
