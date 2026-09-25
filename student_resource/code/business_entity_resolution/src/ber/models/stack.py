import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.io import CFG, timed, work
from ber.models.lgbm import fold_map

R2 = CFG["round2"]
DROP = ["s1", "cand", "row", "label"]


def model_path(k):
    return work("models", f"{R2['name']}_fold{k}.txt")


# full-length preds: round-1 p everywhere, round-2 p on hard rows
def write_preds(split, rows, p2):
    tag = "oof" if split == "train" else "test"
    t = pq.read_table(work("preds", f"{tag}_{R2['from_model']}.parquet"))
    p = t.column("p").to_numpy().copy()
    p[rows] = p2
    pq.write_table(t.set_column(2, "p", pa.array(p.astype(np.float32))), work("preds", f"{tag}_{R2['name']}.parquet"))


# round 2: train on ALL hard train pairs (5 folds by S1), OOF + test
def run(force=False):
    if model_path(CFG["n_folds"] - 1).exists() and not force:
        return
    d = pq.read_table(work("train", "r2.parquet")).to_pandas()
    feats = [c for c in d.columns if c not in DROP]
    fold = fold_map()["fold"].reindex(d["s1"].values).values
    y, rows = d["label"].values, d["row"].values
    print(f"round2 rows {len(d)} pos rate {y.mean():.4f} feats {len(feats)}", flush=True)
    p = CFG["lgbm"]
    params = {k: v for k, v in p.items() if k not in ("num_boost_round", "early_stopping", "test_models")}
    params.update(seed=CFG["seed"], deterministic=True, force_row_wise=True, num_threads=CFG["n_jobs"],
                  verbose=-1, metric="binary_logloss")
    full = lgb.Dataset(d[feats].to_numpy(np.float32), y, feature_name=feats, params=params,
                       free_raw_data=True).construct()
    X = d[feats]
    del d
    oof = np.empty(len(y), np.float32)
    imp = []
    for k in range(CFG["n_folds"]):
        with timed(f"round2 fold {k}"):
            t, v = full.subset(np.flatnonzero(fold != k)), full.subset(np.flatnonzero(fold == k))
            m = lgb.train(params, t, p["num_boost_round"], valid_sets=[v],
                          callbacks=[lgb.early_stopping(p["early_stopping"], verbose=False)])
            m.save_model(model_path(k))
            oof[fold == k] = m.predict(X[fold == k], num_threads=CFG["n_jobs"])
            imp.append(pd.Series(m.feature_importance("gain"), index=feats))
            print(f"round2 fold {k} best_iter {m.best_iteration} logloss "
                  f"{m.best_score['valid_0']['binary_logloss']:.5f}", flush=True)
    (pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False) / 1e6).round(3) \
        .to_csv(work("reports", f"importance_{R2['name']}.tsv"), sep="\t")
    write_preds("train", rows, oof)
    predict_test()


def predict_test():
    d = pq.read_table(work("test", "r2.parquet")).to_pandas()
    ms = [lgb.Booster(model_file=str(model_path(k))) for k in range(CFG["lgbm"]["test_models"])]
    X = d[ms[0].feature_name()]
    write_preds("test", d["row"].values, np.mean([m.predict(X, num_threads=CFG["n_jobs"]) for m in ms], axis=0))
