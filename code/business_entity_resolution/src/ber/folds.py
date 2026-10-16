import numpy as np
import pandas as pd

from ber.io import CFG, read_tsv, source_file, work


# 5 folds grouped by S1 (one S1 = one group), plus training sample flag
def run(force=False):
    out = work("folds.parquet")
    if out.exists() and not force:
        return
    ids = read_tsv(source_file("train", 1), usecols=["entity_id"])["entity_id"].values
    rng = np.random.default_rng(CFG["seed"])
    fold = np.empty(len(ids), dtype=np.int8)
    fold[rng.permutation(len(ids))] = np.arange(len(ids)) % CFG["n_folds"]
    sampled = rng.random(len(ids)) < CFG["train_s1_frac"]
    pd.DataFrame({"s1_id": ids, "fold": fold, "sampled": sampled}).to_parquet(out, index=False)
