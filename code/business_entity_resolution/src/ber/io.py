import csv
import importlib.util
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SRC = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("BER_ROOT", SRC.parents[2]))  # student_resource/
CFG = yaml.safe_load(open(SRC / "configs" / "default.yaml"))
CFG["n_jobs"] = CFG["n_jobs"] or os.cpu_count()


# resolve path under student_resource root
def path(key, *parts):
    return ROOT / CFG[key] / Path(*parts) if parts else ROOT / CFG[key]


def work(*parts):
    p = path("work_dir", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# tsv reader that keeps "NA" and quotes as plain text
def read_tsv(p, usecols=None):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, usecols=usecols)


def source_file(split, s):
    return path("dataset_dir", split, f"{split}_source{s}.tsv")


# ground truth -> long table (s1_id, cand_id)
def read_truth():
    gt = read_tsv(path("dataset_dir", "train", "train_ground_truth.tsv"))
    lst = gt["matched_entity_ids"].str.split(",")
    long = pd.DataFrame({"s1_id": gt["source1_entity_id"].repeat(lst.str.len()).values,
                         "cand_id": [x for l in lst for x in l]})
    return gt, long[long["cand_id"] != ""].reset_index(drop=True)


# write one id-list tsv from (s1 key, cand key) pairs, checking every rule the validator checks
def write_id_lists(p, header, s1_order, s1_keys, cand_keys, valid_keys=None):
    s1_order = list(s1_order)
    order_keys = to_key(s1_order)
    s1_keys, cand_keys = np.asarray(s1_keys, np.int64), np.asarray(cand_keys, np.int64)
    assert len(set(s1_order)) == len(s1_order), "duplicate S1 rows"
    assert (order_keys // 10**10 == 1).all(), "row ids must be S1"
    assert np.isin(s1_keys, order_keys).all(), "rows for S1 not in test set"
    assert np.isin(cand_keys // 10**10, [2, 3]).all(), "only S2/S3 ids allowed"
    assert valid_keys is None or np.isin(cand_keys, valid_keys).all(), "unknown ids"
    o = np.lexsort((cand_keys, s1_keys))
    s1_keys, cand_keys = s1_keys[o], cand_keys[o]
    same = (s1_keys[1:] == s1_keys[:-1]) & (cand_keys[1:] == cand_keys[:-1])
    assert not same.any(), "dup id inside a list"
    lo = np.searchsorted(s1_keys, order_keys, "left")
    hi = np.searchsorted(s1_keys, order_keys, "right")
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for s1, a, b in zip(s1_order, lo, hi):
            f.write(s1 + "\t" + ",".join(to_id(cand_keys[a:b])) + "\n")


# official validator: matching file whole (with --check-ids); candidate file in row shards
# (the whole candidate file would put ~100M ids in python sets and OOM on 16 GB)
def validate(matching, candidate, shard_rows=100_000):
    test_dir = path("dataset_dir", "test")
    cmd = [sys.executable, str(path("validator")), "--matching", str(matching),
           "--candidate", str(Path(matching).with_suffix(".none")), "--test-dir", str(test_dir), "--check-ids"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-2000:])
    assert r.returncode == 0 and "PASS" in r.stdout, "VALIDATOR FAIL (matching)"
    spec = importlib.util.spec_from_file_location("official_validator", path("validator"))
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    matched = {}
    with open(matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if rest:
                matched[s1] = set(rest.split(","))
    valid = v.load_match_targets(str(test_dir), [])
    required_all = v.read_ids(str(test_dir / "test_source1.tsv"))
    seen, errors, not_subset = set(), [], 0
    tmp = Path(candidate).with_suffix(".shard.tsv")
    with open(candidate, encoding="utf-8") as f:
        header = f.readline()
        while True:
            lines = [l for _, l in zip(range(shard_rows), f)]
            if not lines:
                break
            ids = {l.partition("\t")[0] for l in lines}
            assert not (ids & seen), "duplicate S1 row across shards"
            seen |= ids
            tmp.write_text(header + "".join(lines), encoding="utf-8")
            m = v.validate_id_list_file(str(tmp), v.CANDIDATE_HEADER, "candidate_entity_ids", ids, valid, errors)
            not_subset += sum(1 for s1 in ids if matched.get(s1, set()) - m.get(s1, set()))
    tmp.unlink()
    assert seen == required_all, "candidate rows != test S1 set"
    assert not errors, errors[:5]
    assert not_subset == 0, f"{not_subset} S1 have matches outside candidates"
    print("candidate_pairs.tsv: official per-shard checks PASS; matches subset of candidates")


# time a stage and log it
@contextmanager
def timed(name):
    t = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
    yield
    dt = time.time() - t
    print(f"[{time.strftime('%H:%M:%S')}] {name} done {dt:.1f}s", flush=True)
    with open(work("reports", "timings.tsv"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\t{dt:.1f}\n")


# "S2-123" -> 20000000123 (int64 key, RAM-light); ids are <=9 digits, no leading zeros
def to_key(ids):
    s = pd.Series(ids, dtype=str)
    return (s.str[1].astype("int64") * 10**10 + s.str[3:].astype("int64")).values


def to_id(keys):
    return [f"S{k // 10**10}-{k % 10**10}" for k in keys.tolist()]
