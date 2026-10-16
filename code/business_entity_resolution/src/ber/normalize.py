import re
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from indic_transliteration.sanscript import DEVANAGARI, IAST, transliterate

from ber.dictionaries import (ADDR_FR, ADDR_STOP, ADDR_TOKENS, CANDRA, CHILLU, DBA, LEGAL, NAME_STOP,
                              PLACE_PHRASES, SKEL)
from ber.io import CFG, read_tsv, source_file, timed, work

# all 9 Indic blocks share one layout: shift each to Devanagari by offset
TO_DEVA = {cp: 0x0900 + (cp - 0x0900) % 0x80 for cp in range(0x0980, 0x0D80)}
CHILLU_T = {ord(k): v for k, v in CHILLU.items()}
CANDRA_T = {ord(k): v for k, v in CANDRA.items()}
INDIC = re.compile(r"[ऀ-ൿ]")
DEVA_RUN = re.compile(r"[ऀ-ॿ‌‍]+")
CONS_A = re.compile(r"(?<=[bcdfghjklmnpqrstvwxyzḍṭṇṛḷñṅśṣṉṟ])a$")
NONALNUM = re.compile(r"[^a-z0-9]+")
DIGITS = re.compile(r"\d+")
DBA_RE = re.compile(DBA)
PHRASE_RE = re.compile(r"\b(" + "|".join(sorted(PLACE_PHRASES, key=len, reverse=True)) + r")\b")
ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
GLUE = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
LANDMARK = re.compile(r"\b(?:near|opp|behind|beside|next to|adj|adjacent to)\b((?: \S+){1,4})")
SKEL_T = str.maketrans(SKEL)
REPEAT = re.compile(r"(.)\1+")
TRANSLIT = CFG["normalize"]["translit_indic"]
LIST_COLS = ("dba_parts", "numbers")


# one Devanagari word -> latin, drop final inherent 'a' (schwa)
@lru_cache(maxsize=200_000)
def _deva_word(w):
    out = transliterate(w, DEVANAGARI, IAST)
    return CONS_A.sub("", out).replace("ṃ", "n").replace("ḥ", "")


# any Indic script -> latin; latin text passes through
def translit(s):
    if not TRANSLIT or not INDIC.search(s):
        return s
    s = s.translate(CHILLU_T).translate(TO_DEVA).translate(CANDRA_T)
    return DEVA_RUN.sub(lambda m: _deva_word(m.group()), s)


# accents off, lowercase, punctuation to space
def fold(s):
    s = unicodedata.normalize("NFKD", translit(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    return NONALNUM.sub(" ", s).strip()


# name tokens -> core tokens (legal forms and stop words out)
def _core(toks):
    return [t for t in toks if t not in LEGAL and t not in NAME_STOP]


# sound-alike skeleton per word: consonant classes, repeats collapsed
def skeleton(tokens):
    out = [REPEAT.sub(r"\1", t.translate(SKEL_T)) if not t.isdigit() else t for t in tokens]
    return " ".join(x for x in out if x)


# name -> (name_norm, name_core, legal_class, dba_parts, name_skel)
def norm_name(raw):
    n = fold(raw)
    toks = n.split()
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    core = _core(toks)
    parts = [" ".join(_core(p.split())) for p in DBA_RE.split(n)]
    parts = [p for p in parts if p]
    if len(parts) > 1:
        core = " ".join(parts).split()
    else:
        parts = []
    return n, " ".join(core) or n, " ".join(legal), parts, skeleton(core)


# address -> (addr_norm, addr_nolandmark, landmark, numbers, postal)
def norm_addr(raw, country=""):
    a = fold(raw)
    raw_nums = DIGITS.findall(a)
    postal = next((d for d in reversed(raw_nums) if len(d) in (5, 6) and not d.startswith("00")), "")
    a = GLUE.sub(" ", ORDINAL.sub(r"\1", PHRASE_RE.sub(lambda m: PLACE_PHRASES[m.group(1)], a)))
    fr = country == "France"
    toks = []
    for t in a.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_TOKENS.get(t, t)
            if fr:
                t = ADDR_FR.get(t, t)
        if t not in ADDR_STOP:
            toks.append(t)
    a = " ".join(toks)
    lm = " ".join(m.group(1).strip() for m in LANDMARK.finditer(a))
    nolm = LANDMARK.sub(" ", a).split()
    return a, " ".join(nolm), lm, [d.lstrip("0") or "0" for d in raw_nums], postal


def _row(t):
    name, addr, country = t
    return (*norm_name(name), *norm_addr(addr, country))


# one source file -> norm parquet, chunked to keep RAM low
def normalize_file(split, s, pool, chunk=1_000_000):
    out = work(split, f"norm_s{s}.parquet")
    df = read_tsv(source_file(split, s))
    writer = None
    for i in range(0, len(df), chunk):
        d = df.iloc[i:i + chunk]
        res = pool.map(_row, zip(d["business_name"], d["business_address"], d["country"]), chunksize=5000)
        nn, nc, lc, dba, sk, an, anl, lm, nums, pc = zip(*res)
        t = pd.DataFrame({
            "entity_id": d["entity_id"].values, "source": f"S{s}", "country": d["country"].values,
            "name_raw": d["business_name"].values, "name_norm": nn, "name_core": nc, "legal_class": lc,
            "dba_parts": list(dba), "name_skel": sk,
            "addr_raw": d["business_address"].values, "addr_norm": an, "addr_nolandmark": anl,
            "landmark": lm, "numbers": list(nums), "postal": pc,
        })
        schema = pa.schema([(c, pa.list_(pa.string()) if c in LIST_COLS else pa.large_string()) for c in t.columns])
        tbl = pa.Table.from_pandas(t, schema=schema, preserve_index=False)
        writer = writer or pq.ParquetWriter(out, tbl.schema)
        writer.write_table(tbl)
    writer.close()


def run(split, force=False):
    with Pool(CFG["n_jobs"]) as pool:
        for s in (1, 2, 3):
            if work(split, f"norm_s{s}.parquet").exists() and not force:
                continue
            with timed(f"normalize {split} s{s}"):
                normalize_file(split, s, pool)
