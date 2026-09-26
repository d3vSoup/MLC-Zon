import glob
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.io import CFG, ROOT, to_key, work


def parts(dirs, pattern):
    fs = sorted(f for d in dirs for f in glob.glob(f"{d}/**/{pattern}", recursive=True))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None, len(fs)


def scores_file(split, tag="ce"):
    return ROOT / CFG["ce"]["scores_dir"] / f"{tag}_scores_{split}.parquet"


# one time: teammates' prediction parts + band files -> keyed score files (s1, cand, ce_p) shipped with the code.
# test = mean of both halves' models (or one half's model while only one teammate is done).
# keyed by ids, so any later run with any row order can use them.
def collect(dirs, tag="ce"):
    tr, n1 = parts(dirs, "train_h*_part*.parquet")
    te1, n2 = parts(dirs, "test_h1_part*.parquet")
    te2, n3 = parts(dirs, "test_h2_part*.parquet")
    print(f"parts: train {n1}, test_h1 {n2}, test_h2 {n3}")
    assert tr is not None and tr.pair_id.is_unique, "train ce_p missing or duplicated"
    te = pd.concat([x for x in (te1, te2) if x is not None]).groupby("pair_id", as_index=False).ce_p.mean()
    for split, d in (("train", tr), ("test", te)):
        band = pd.read_parquet(work("ce", f"ce_band_{split}.parquet"),
                               columns=["pair_id", "s1_id", "cand_id"] + (["half"] if split == "train" else []))
        if split == "train":  # one teammate only = one half scored; that half must be complete
            cov = sorted(band.half[band.pair_id.isin(d.pair_id)].unique().tolist())
            print(f"train: scored halves {cov}")
            band = band[band.half.isin(cov)]
        assert set(d.pair_id) == set(band.pair_id), f"{split} band coverage {d.pair_id.isin(band.pair_id).sum()}/{len(band)}"
        m = band.merge(d, on="pair_id")
        out = pd.DataFrame({"s1": to_key(m.s1_id), "cand": to_key(m.cand_id), "ce_p": m.ce_p.astype(np.float32)})
        f = scores_file(split, tag)
        f.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(f, index=False, compression="zstd")
        print(f"{split}: ce_p for {len(out)} pairs, mean {out.ce_p.mean():.3f} -> {f}")
    if te1 is not None and te2 is not None:
        m = te1.merge(te2, on="pair_id")
        print(f"test agreement between the two CE models: corr {np.corrcoef(m.ce_p_x, m.ce_p_y)[0, 1]:.3f}")


# pipeline stage: keyed scores -> work/{split}/<tag>_p.parquet (row, <tag>_p) for this run's round-2 rows
def run(split, force=False):
    if not CFG["round2"].get("use_ce"):
        return
    for tag in CFG["round2"].get("ce_models", ["ce"]):
        out = work(split, f"{tag}_p.parquet")
        if out.exists() and not force:
            continue
        s = pd.read_parquet(scores_file(split, tag)).rename(columns={"ce_p": f"{tag}_p"})
        r2 = pq.read_table(work(split, "r2.parquet"), columns=["s1", "cand", "row"]).to_pandas()
        m = r2.merge(s, on=["s1", "cand"])
        print(f"{split} {tag}: {len(m)} of {len(s)} CE-scored pairs found in round-2 rows ({len(m) / len(s):.4f})")
        m[["row", f"{tag}_p"]].astype({"row": np.int64, f"{tag}_p": np.float32}).to_parquet(out, index=False)


# usage: python -m ber.ce_ingest [--tag ce2] <dir>...
if __name__ == "__main__":
    a = sys.argv[1:]
    tag = a[1] if a[:1] == ["--tag"] else "ce"
    collect(a[2:] if a[:1] == ["--tag"] else a, tag)
