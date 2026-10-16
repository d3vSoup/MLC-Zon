"""LightGBM pair models, 2-fold cross-fitting grouped by S1 (hash of si); OOF decode + score on all train S1.

Stage 1: pair features only.  Stage 2: stage-1 features + stage-1 p + collective features (stage2.py),
trained on stage-1 OOF probabilities. Training rows are uniform samples (keeps calibration).
Test p = mean of the two fold models at each stage.
"""
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import w, gt_pairs, SEED, THREADS, DROP, dropped
from decide import decide
from score import f05, report
from stage2 import stage2_feats, generic_feats, twin_feats, num_feats

GENERIC = os.environ.get("BER_GENERIC", "0") == "1"  # M3: generic-word swap features in stage 2
TWINF = os.environ.get("BER_TWINF", "0") == "1"      # M6: near-twin evidence (number shift, legal forms) in stage 2
NUMF = os.environ.get("BER_NUMF", "0") == "1"      # M7: house-number shape features in stage 2
EXCL = os.environ.get("BER_EXCL_RI", "")             # M6a: "lo:hi" train rows left out of stage 2 (a synthetic twin kind)

KEYS = ["ri", "si"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=THREADS, seed=SEED, verbose=-1)
POOL = int(os.environ.get("BER_POOL", 16_000_000))  # training rows over both folds


def files(split):
    return sorted(glob.glob(w(f"F_{split}/*.parquet")))


def gt_idx():
    """True links as (si, ri) of the train frames."""
    s1 = pl.read_parquet(w("n_train_s1.parquet"), columns=["id"]).with_row_index("si").rename({"id": "s1"})
    r = pl.concat([pl.read_parquet(w(f"n_train_s{s}.parquet"), columns=["id"]) for s in (2, 3)]).with_row_index("ri")
    g = gt_pairs().join(s1, on="s1").join(r, on="id").select(pl.col("si").cast(pl.Int32), pl.col("ri").cast(pl.Int32))
    return g.filter(~dropped()) if DROP else g  # links of absent S1s vanish: their rows are distractors


def label(F, gt):
    return F.join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=KEYS, how="left") \
            .with_columns(pl.col("y").fill_null(0), ((pl.col("si").hash(SEED) % 2).cast(pl.Int8)).alias("fold"))


def fit_folds(pool, tag, params=PARAMS):
    cols = [c for c in pool.columns if c not in KEYS + ["y", "fold"]]
    models = []
    for k in (0, 1):
        tr, va = pool.filter(pl.col("fold") == k), pool.filter(pl.col("fold") != k).head(2_000_000)
        dtr = lgb.Dataset(tr.select(pl.col(cols).cast(pl.Float32)).to_numpy(), tr["y"].to_numpy(), feature_name=cols)
        dva = lgb.Dataset(va.select(pl.col(cols).cast(pl.Float32)).to_numpy(), va["y"].to_numpy(), reference=dtr)
        m = lgb.train(params, dtr, 6000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(params.get("patience", 100)), lgb.log_evaluation(250)])
        m.save_model(w(f"{tag}_fold{k}.txt"))
        models.append(m)
        imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)
        print(f"{tag} fold {k}: iters {m.best_iteration} top gain:", [(c, int(g)) for g, c in imp[:20]], flush=True)
    return models, cols


def oof_predict(models, cols, F):
    p = np.zeros(F.height, dtype=np.float32)
    for k in (0, 1):
        msk = (F["fold"] == k).to_numpy()
        if msk.any():
            p[msk] = models[1 - k].predict(F.filter(pl.col("fold") == k).select(pl.col(cols).cast(pl.Float32)).to_numpy(), num_threads=THREADS)
    return p


def train(tag="m"):
    t0 = time.time()
    gt = gt_idx()
    fs = files("train")
    total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in fs)
    frac = min(1.0, POOL / total)
    pool = pl.concat([label(pl.read_parquet(f), gt).sample(fraction=frac, seed=SEED) for f in fs])
    print(f"pairs {total}, pool {pool.height} (pos {pool['y'].sum()}) {time.time() - t0:.0f}s", flush=True)
    models, cols = fit_folds(pool, tag)
    del pool
    oof = []
    for f in fs:
        F = label(pl.read_parquet(f), gt)
        oof.append(F.select(KEYS + ["y"]).with_columns(pl.Series("p", oof_predict(models, cols, F))))
    oof = pl.concat(oof)
    oof.write_parquet(w(f"oof_{tag}.parquet"))
    print(f"oof done {time.time() - t0:.0f}s", flush=True)
    return evaluate(oof, gt, tag)


def fit1(tag="m"):
    """Stage-1 fold models only (OOF/test prediction is sharded: pred_shard + gather1)."""
    t0 = time.time()
    gt, fs = gt_idx(), files("train")
    total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in fs)
    frac = min(1.0, POOL / total)
    pool = pl.concat([label(pl.read_parquet(f), gt).sample(fraction=frac, seed=SEED) for f in fs])
    print(f"pairs {total}, pool {pool.height} (pos {pool['y'].sum()}) {time.time() - t0:.0f}s", flush=True)
    fit_folds(pool, tag)


def pred_shard(i, n, tag="m"):
    """Files k with k % n == i: stage-1 OOF on train (other fold's model) + mean of both models on test."""
    t0 = time.time()
    models = [lgb.Booster(model_file=w(f"{tag}_fold{k}.txt")) for k in (0, 1)]
    cols, gt = models[0].feature_name(), gt_idx()
    oof = [label(pl.read_parquet(f), gt).pipe(lambda F: F.select(KEYS + ["y"]).with_columns(pl.Series("p", oof_predict(models, cols, F))))
           for k, f in enumerate(files("train")) if k % n == i]
    pl.concat(oof).write_parquet(w(f"oof_{tag}_part{i}.parquet"))
    te = [pl.read_parquet(f).pipe(lambda F: F.select(KEYS).with_columns(pl.Series("p", _mean_pred(tag, F))))
          for k, f in enumerate(files("test")) if k % n == i]
    pl.concat(te).write_parquet(w(f"pred_test_{tag}_part{i}.parquet"))
    print(f"shard {i}/{n}: {sum(x.height for x in oof)} oof, {sum(x.height for x in te)} test pairs {time.time() - t0:.0f}s", flush=True)


def pred_twin(tag="m"):
    """Stage-1 OOF p for the synthetic twin rows only (feature files *_T_*), as an extra part for gather1."""
    models = [lgb.Booster(model_file=w(f"{tag}_fold{k}.txt")) for k in (0, 1)]
    cols, gt = models[0].feature_name(), gt_idx()
    oof = [label(pl.read_parquet(f), gt).pipe(lambda F: F.select(KEYS + ["y"]).with_columns(pl.Series("p", oof_predict(models, cols, F))))
           for f in files("train") if "_T_" in os.path.basename(f)]
    oof = pl.concat(oof)
    oof.write_parquet(w(f"oof_{tag}_part9.parquet"))
    print(f"twin pairs {oof.height}, p>=0.002: {(oof['p'] >= 0.002).sum()}, true links among them {oof['y'].sum()}", flush=True)


def gather1(tag="m"):
    oof = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(w(f"oof_{tag}_part*.parquet")))])
    oof.write_parquet(w(f"oof_{tag}.parquet"))
    pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(w(f"pred_test_{tag}_part*.parquet")))]).write_parquet(w(f"pred_test_{tag}.parquet"))
    return evaluate(oof, None, tag)


def recs2(split):
    s1 = pl.read_parquet(w(f"n_{split}_s1.parquet"), columns=["nm", "nums"]).with_row_index("si").with_columns(pl.col("si").cast(pl.Int32))
    r = pl.concat([pl.read_parquet(w(f"n_{split}_s{s}.parquet"), columns=["id", "nm", "al_r", "ad", "nums"]) for s in (2, 3)]) \
          .with_row_index("ri").with_columns(pl.col("ri").cast(pl.Int32))
    r = r.select("ri", pl.coalesce("al_r", "nm").alias("rnm"), "ad", "nums",
                 pl.col("id").str.starts_with("S3").cast(pl.Int8).alias("src3"))
    return s1, r


def stage2_table(split, P, p1_path=None):
    """Stage-2 design matrix: stage-1 features of pairs with p >= P_MIN, + p, + collective features.
    Stage 3 passes P = stage-2 probabilities and p1_path = stage-1 probabilities (kept as feature p1)."""
    s1, r = recs2(split)
    Q = stage2_feats(P.select(KEYS + ["p"]), r, s1)
    T = twin_feats(Q.select(KEYS), r, s1) if TWINF else None
    N = num_feats(Q.select(KEYS), r, s1) if NUMF else None
    del s1, r
    keys = Q.select(KEYS)
    F = pl.concat([pl.read_parquet(f).join(keys, on=KEYS, how="semi") for f in files(split)])
    D = F.join(Q, on=KEYS)
    if p1_path:
        D = D.join(pl.read_parquet(p1_path).select(KEYS + [pl.col("p").alias("p1")]), on=KEYS, how="left")
    if GENERIC:
        D = generic_feats(split, D, w)
    if T is not None:
        D = D.join(T, on=KEYS, how="left")
    if N is not None:
        D = D.join(N, on=KEYS, how="left")
    for tag, col in (("ce", "ce"), ("ce2", "ce2"), ("cce", "cce"), ("cceb", "cceb")):  # CE logits (+ *_twin files); cce = cluster CE parts
        fs = sorted(glob.glob(w(f"{tag}_{split}*.parquet")))
        if fs:
            D = D.join(pl.concat([pl.read_parquet(f).select(KEYS + [pl.col("ce").alias(col)]) for f in fs]), on=KEYS, how="left")
    return D


def train2(tag="s2", base="m"):
    t0 = time.time()
    gt = gt_idx()
    P = pl.read_parquet(w(f"oof_{base}.parquet"))
    if EXCL:
        lo, hi = map(int, EXCL.split(":"))
        P = P.filter(~pl.col("ri").is_between(lo, hi, closed="left"))
    D = label(stage2_table("train", P, None if base == "m" else w("oof_m.parquet")), gt)
    miss = gt.height - D["y"].sum()
    print(f"stage2 pairs {D.height} (pos {D['y'].sum()}, true links outside stage-2 set {miss}) {time.time() - t0:.0f}s", flush=True)
    frac = min(1.0, POOL / D.height)
    params = dict(PARAMS, **json.loads(os.environ.get("BER_LGB2", "{}")))  # stage-2 hyper-parameter overrides
    models, cols = fit_folds(D.sample(fraction=frac, seed=SEED), tag, params)
    oof = D.select(KEYS + ["y"]).with_columns(pl.Series("p", oof_predict(models, cols, D)))
    oof.write_parquet(w(f"oof_{tag}.parquet"))
    return evaluate(oof, gt, tag)


def evaluate(oof, gt=None, tag="OOF"):
    gt = gt_idx() if gt is None else gt
    ents = pl.read_parquet(w("n_train_s1.parquet"), columns=["country"]).with_row_index("s1") \
             .with_columns(pl.col("s1").cast(pl.Int32))
    if DROP:
        ents = ents.filter(~dropped("s1"))
    kept, _ = decide(oof.select(pl.col("ri").alias("id"), pl.col("si").alias("s1"), "p"))
    return report(f05(ents, kept, gt.rename({"si": "s1", "ri": "id"})), tag)


def _mean_pred(tag, F):
    ms = [lgb.Booster(model_file=w(f"{tag}_fold{k}.txt")) for k in (0, 1)]
    X = F.select(pl.col(ms[0].feature_name()).cast(pl.Float32)).to_numpy()
    return np.mean([m.predict(X, num_threads=THREADS) for m in ms], 0)


def predict(split, tag="m"):
    return pl.concat([pl.read_parquet(f).pipe(lambda F: F.select(KEYS).with_columns(pl.Series("p", _mean_pred(tag, F))))
                      for f in files(split)])


def predict2(split, tag="s2", base="m"):
    """Returns (stage-2 pairs with p2 = final candidate set, stage-1 predictions)."""
    cached = w(f"pred_test_{base}.parquet") if split == "test" else None
    P = pl.read_parquet(cached) if cached and os.path.exists(cached) else predict(split, base)
    D = stage2_table(split, P, None if base == "m" else w(f"pred_{split}_m.parquet"))
    return D.select(KEYS).with_columns(pl.Series("p", _mean_pred(tag, D))), P


if __name__ == "__main__":
    {"train": train, "train2": train2}[sys.argv[1]]()
