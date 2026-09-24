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
    parts, off = [], 0
    for b in batches("train"):
        d = b.to_pandas()
        n = len(d)
        d["label"] = labels[off:off + n]
        off += n
        d["fold"] = fm["fold"].reindex(d["s1"]).values
        parts.append(d[fm["sampled"].reindex(d["s1"]).values])
    tr = pd.concat(parts, ignore_index=True)
    del parts
    feats = [c for c in tr.columns if c not in ID + ["label", "fold"]]
    print(f"train rows {len(tr)} pos rate {tr.label.mean():.4f} feats {feats}")
    p = CFG["lgbm"]
    params = {k: v for k, v in p.items() if k not in ("num_boost_round", "early_stopping")}
    params.update(seed=CFG["seed"], deterministic=True, force_row_wise=True, num_threads=CFG["n_jobs"],
                  verbose=-1, metric="binary_logloss")
    imp = []
    for k in range(CFG["n_folds"]):
        with timed(f"lgbm fold {k}"):
            t, v = tr[tr.fold != k], tr[tr.fold == k]
            m = lgb.train(params, lgb.Dataset(t[feats], t.label), p["num_boost_round"],
                          valid_sets=[lgb.Dataset(v[feats], v.label)],
                          callbacks=[lgb.early_stopping(p["early_stopping"], verbose=False), lgb.log_evaluation(200)])
            m.save_model(model_path(k))
            imp.append(pd.Series(m.feature_importance("gain"), index=feats))
            print(f"fold {k} best_iter {m.best_iteration} logloss {m.best_score['valid_0']['binary_logloss']:.5f}")
    (pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False) / 1e6).round(2) \
        .to_csv(work("reports", f"importance_{NAME}.tsv"), sep="\t")
    del tr
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
