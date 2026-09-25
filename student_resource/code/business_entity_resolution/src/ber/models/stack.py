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


def model_path(k, name=None):
    return work("models", f"{name or R2['name']}_fold{k}.txt")


# full-length preds: round-1 p everywhere, round-2 p on hard rows
def write_preds(split, rows, p2, name=None):
    tag = "oof" if split == "train" else "test"
    t = pq.read_table(work("preds", f"{tag}_{R2['from_model']}.parquet"))
    p = t.column("p").to_numpy().copy()
    p[rows] = p2
    pq.write_table(t.set_column(2, "p", pa.array(p.astype(np.float32))), work("preds", f"{tag}_{name or R2['name']}.parquet"))


# round 2: train on ALL hard train pairs (5 folds by S1), OOF + test
def run(force=False):
    if model_path(CFG["n_folds"] - 1).exists() and not force:
        return
    fit(rounds=R2.get("num_boost_round"))


# frac < 1 trains on that share of S1 (key-hashed) but still predicts every OOF row: learning-curve probe
def fit(name=None, frac=1.0, rounds=None, test=True, use_ce=False, extra=()):
    d = load_r2("train", use_ce, extra)
    feats = [c for c in d.columns if c not in DROP]
    fold = fold_map()["fold"].reindex(d["s1"].values).values
    use = (d["s1"].values % 1000) < frac * 1000
    y, rows = d["label"].values, d["row"].values
    print(f"round2 rows {len(d)} pos rate {y.mean():.4f} feats {len(feats)}", flush=True)
    p = dict(CFG["lgbm"])
    p["num_boost_round"] = rounds or p["num_boost_round"]
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
            t, v = full.subset(np.flatnonzero((fold != k) & use)), full.subset(np.flatnonzero(fold == k))
            m = lgb.train(params, t, p["num_boost_round"], valid_sets=[v],
                          callbacks=[lgb.early_stopping(p["early_stopping"], verbose=False)])
            m.save_model(model_path(k, name))
            oof[fold == k] = m.predict(X[fold == k], num_threads=CFG["n_jobs"])
            imp.append(pd.Series(m.feature_importance("gain"), index=feats))
            print(f"round2 fold {k} best_iter {m.best_iteration} logloss "
                  f"{m.best_score['valid_0']['binary_logloss']:.5f}", flush=True)
    (pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False) / 1e6).round(3) \
        .to_csv(work("reports", f"importance_{name or R2['name']}.tsv"), sep="\t")
    write_preds("train", rows, oof, name)
    if test:
        predict_test(name, use_ce, extra)


# round-2 table + optional row-aligned extra feature files (work/{split}/<name>.parquet) + optional CE score
def load_r2(split, use_ce=False, extra=()):
    d = pq.read_table(work(split, "r2.parquet")).to_pandas()
    for e in extra:
        x = pq.read_table(work(split, f"{e}.parquet")).to_pandas()
        assert len(x) == len(d), f"{e} not row-aligned with r2"
        d = pd.concat([d, x], axis=1)
    if use_ce:
        ce = pd.read_parquet(work(split, "ce_p.parquet"))
        d["ce_p"] = pd.Series(ce["ce_p"].values, index=ce["row"].values).reindex(d["row"].values).values
        print(f"{split}: ce_p present on {d.ce_p.notna().mean():.3f} of round-2 rows", flush=True)
    return d


def predict_test(name=None, use_ce=False, extra=()):
    d = load_r2("test", use_ce, extra)
    ms = [lgb.Booster(model_file=str(model_path(k, name))) for k in range(CFG["lgbm"]["test_models"])]
    X = d[ms[0].feature_name()]
    write_preds("test", d["row"].values, np.mean([m.predict(X, num_threads=CFG["n_jobs"]) for m in ms], axis=0),
                name)
