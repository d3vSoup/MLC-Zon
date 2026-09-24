import re
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from indic_transliteration.sanscript import DEVANAGARI, IAST, transliterate

from ber.dictionaries import CANDRA, CHILLU, LEGAL, NAME_STOP
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
TRANSLIT = CFG["normalize"]["translit_indic"]


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


# name -> (name_norm, name_core, legal_class)
def norm_name(raw):
    n = fold(raw)
    toks = n.split()
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    return n, " ".join(core) or n, " ".join(legal)


# address -> (addr_norm, numbers, postal)
def norm_addr(raw):
    a = fold(raw)
    raw_nums = DIGITS.findall(a)
    postal = next((d for d in reversed(raw_nums) if len(d) in (5, 6) and not d.startswith("00")), "")
    return a, [d.lstrip("0") or "0" for d in raw_nums], postal


def _row(t):
    name, addr = t
    return (*norm_name(name), *norm_addr(addr))


# one source file -> norm parquet, chunked to keep RAM low
def normalize_file(split, s, pool, chunk=1_000_000):
    out = work(split, f"norm_s{s}.parquet")
    df = read_tsv(source_file(split, s))
    writer = None
    for i in range(0, len(df), chunk):
        d = df.iloc[i:i + chunk]
        res = pool.map(_row, zip(d["business_name"], d["business_address"]), chunksize=5000)
        nn, nc, lc, an, nums, pc = zip(*res)
        t = pd.DataFrame({
            "entity_id": d["entity_id"].values, "source": f"S{s}", "country": d["country"].values,
            "name_raw": d["business_name"].values, "name_norm": nn, "name_core": nc, "legal_class": lc,
            "dba_parts": [[] for _ in nn], "name_skel": "",
            "addr_raw": d["business_address"].values, "addr_norm": an, "addr_nolandmark": an,
            "landmark": "", "numbers": list(nums), "postal": pc,
        })
        tbl = pa.Table.from_pandas(t, preserve_index=False)
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
