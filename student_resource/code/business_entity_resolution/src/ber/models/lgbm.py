import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.io import CFG, timed, to_key, work

NAME = "lgbm0"
ID = ["s1", "cand"]


def fold_map():
    f = pd.read_parquet(work("folds.parquet"))
    f.index = to_key(f.pop("s1_id"))
    return f


def batches(split, cols=None):
    return pq.ParquetFile(work(split, "feats.parquet")).iter_batches(batch_size=CFG["features"]["chunk"], columns=cols)


def model_path(k):
    return work("models", f"{NAME}_fold{k}.txt")


# fit 5 fold models on sampled S1 groups; OOF-predict every train pair
def run(force=False):
    if model_path(CFG["n_folds"] - 1).exists() and work("preds", f"oof_{NAME}.parquet").exists() and not force:
        return
    fm = fold_map()
    labels = pq.read_table(work("train", "cands.parquet"), columns=["label"]).column("label").to_numpy()
    feats = [c for c in pq.ParquetFile(work("train", "feats.parquet")).schema_arrow.names if c not in ID]
    masks = [fm["sampled"].reindex(b.column("s1").to_numpy()).values for b in batches("train", ["s1"])]
    # fill one pre-sized float32 matrix (no concat copy: 30M x 45 floats is ~5 GB)
    X = np.empty((sum(int(m.sum()) for m in masks), len(feats)), np.float32)
    y, fold = np.empty(len(X), np.int8), np.empty(len(X), np.int8)
    off = row = 0
    for m, b in zip(masks, batches("train", ["s1"] + feats)):
        k = int(m.sum())
        X[row:row + k] = b.select(feats).to_pandas().to_numpy(np.float32)[m]
        y[row:row + k] = labels[off:off + len(m)][m]
        fold[row:row + k] = fm["fold"].reindex(b.column("s1").to_numpy()[m]).values
        off += len(m)
        row += k
    del labels, masks
    print(f"train rows {len(X)} pos rate {y.mean():.4f} feats {feats}", flush=True)
    p = CFG["lgbm"]
    params = {k: v for k, v in p.items() if k not in ("num_boost_round", "early_stopping", "test_models")}
    params.update(seed=CFG["seed"], deterministic=True, force_row_wise=True, num_threads=CFG["n_jobs"],
                  verbose=-1, metric="binary_logloss")
    # bin once, then per-fold subsets: no raw-data copies (RAM)
    full = lgb.Dataset(X, y, feature_name=feats, params=params, free_raw_data=True).construct()
    del X
    imp = []
    for k in range(CFG["n_folds"]):
        with timed(f"lgbm fold {k}"):
            t, v = full.subset(np.flatnonzero(fold != k)), full.subset(np.flatnonzero(fold == k))
            m = lgb.train(params, t, p["num_boost_round"], valid_sets=[v],
                          callbacks=[lgb.early_stopping(p["early_stopping"], verbose=False), lgb.log_evaluation(200)])
            m.save_model(model_path(k))
            imp.append(pd.Series(m.feature_importance("gain"), index=feats))
            print(f"fold {k} best_iter {m.best_iteration} logloss {m.best_score['valid_0']['binary_logloss']:.5f}",
                  flush=True)
    (pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False) / 1e6).round(2) \
        .to_csv(work("reports", f"importance_{NAME}.tsv"), sep="\t")
    del full
    predict("train")


# train: each pair by its own fold's model (OOF); test: mean of first test_models fold models
def predict(split):
    models = [lgb.Booster(model_file=str(model_path(k))) for k in range(CFG["n_folds"])]
    feats = models[0].feature_name()
    test_models = models[:CFG["lgbm"]["test_models"]]
    fm = fold_map() if split == "train" else None
    out = work("preds", f"{'oof' if split == 'train' else 'test'}_{NAME}.parquet")
    writer = None
    with timed(f"predict {split}"):
        for b in batches(split, ID + feats):
            d = b.to_pandas()
            X = d[feats]
            if fm is None:
                p = np.mean([m.predict(X, num_threads=CFG["n_jobs"]) for m in test_models], axis=0)
            else:
                fold = fm["fold"].reindex(d["s1"]).values
                p = np.empty(len(d))
                for k, m in enumerate(models):
                    w = fold == k
                    p[w] = m.predict(X[w], num_threads=CFG["n_jobs"])
            tbl = pa.table({"s1": d["s1"].values, "cand": d["cand"].values, "p": p.astype(np.float32)})
            writer = writer or pq.ParquetWriter(out, tbl.schema)
            writer.write_table(tbl)
    writer.close()
