import glob
import sys

import numpy as np
import pandas as pd

from ber.io import work


def parts(dirs, pattern):
    fs = sorted(f for d in dirs for f in glob.glob(f"{d}/**/{pattern}", recursive=True))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None, len(fs)


# teammates' prediction parts -> work/{train,test}/ce_p.parquet (row = pair_id, ce_p); test = mean of both halves' models
def run(dirs):
    band_tr = pd.read_parquet(work("ce", "ce_band_train.parquet"), columns=["pair_id"])
    band_te = pd.read_parquet(work("ce", "ce_band_test.parquet"), columns=["pair_id"])
    tr, n1 = parts(dirs, "train_h*_part*.parquet")
    te1, n2 = parts(dirs, "test_h1_part*.parquet")
    te2, n3 = parts(dirs, "test_h2_part*.parquet")
    print(f"parts: train {n1}, test_h1 {n2}, test_h2 {n3}")
    assert tr is not None and tr.pair_id.is_unique, "train ce_p missing or duplicated"
    assert set(tr.pair_id) == set(band_tr.pair_id), f"train band coverage {tr.pair_id.isin(band_tr.pair_id).sum()}/{len(band_tr)}"
    te = pd.concat([x for x in (te1, te2) if x is not None]).groupby("pair_id", as_index=False).ce_p.mean()
    assert set(te.pair_id) == set(band_te.pair_id), f"test band coverage {len(te)}/{len(band_te)}"
    for split, d in (("train", tr), ("test", te)):
        d.rename(columns={"pair_id": "row"}).astype({"row": np.int64, "ce_p": np.float32}) \
            .to_parquet(work(split, "ce_p.parquet"), index=False)
        print(f"{split}: ce_p for {len(d)} pairs, mean {d.ce_p.mean():.3f}")
    if te1 is not None and te2 is not None:
        m = te1.merge(te2, on="pair_id")
        print(f"test agreement between the two CE models: corr {np.corrcoef(m.ce_p_x, m.ce_p_y)[0, 1]:.3f}")


if __name__ == "__main__":
    run(sys.argv[1:])
