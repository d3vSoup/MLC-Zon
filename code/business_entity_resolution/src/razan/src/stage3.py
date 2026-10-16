"""Stage 3 (M10): recalibrate M9 stage-2 pair probabilities with same-name rival-row counts.

New information (label-free, from each split's own M9 p and normalised rows): a no-address row whose name matches a
live S1 is either that S1's row or a row of an absent same-name entity. Addressed rows with the same core that no S1
claims (best p < 0.5) are evidence for the absent entity. Train OOF: P(true) is 0.03-0.05 above M9 p with no unclaimed
rival and 0.05-0.12 below it with 2+.

LightGBM on [M9 p, row/S1 context, rival counts], trained on live train S1s (dropped S1s are simulation artefacts),
cross-fitted by S1 hash (2 folds) for the train OOF; the test model is fit on all live train pairs with the mean best
iteration. Train CV (live S1s): 0.98898 vs M9 0.98852 (India 0.98792, US 0.98969).

usage: BER_WORK=<work with n_{train,test}_s{1,2,3}.parquet> python stage3.py <m9_share_dir> <out_dir> [gt_pairs.parquet] [teammate_dir]
  -> <out_dir>/oof_s2ce.parquet, pred_test_s2ce.parquet (ri, si, p; same pair sets as the M9 share), stage3.txt
M11: with teammate_dir (razan_export_{train,test}.parquet from ../export_for_razan.py: the step-A lgbm2_ce p on these
pairs) the stacker also sees that p, whether that model scored the pair, its row top p / rank and the gap to ours.
"""
import gc
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import w

FEATS = ["p", "noaddr", "src3", "rk", "top_p", "p2", "psum_r", "n_r", "n_s", "mass_s", "conf_s", "k_core_s",
         "rc_rows", "rc_unclaimed", "rc_claimed", "rc_noaddr", "sc_unclaimed", "core_eq"]
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=7)
CLAIM = 0.5
FEATS_M = ["pm", "pm_na", "pm_top", "pm_rk", "pm_gap"]


def teammate_p(tdir, split, share):
    """The step-A model's p on our pairs, keyed back to (si, ri) through our ids (to_key: 'S<k>-<n>' -> k * 10**10 + n)."""
    key = lambda ids: ids.str.slice(1, 1).cast(pl.Int64) * 10**10 + ids.str.slice(3).cast(pl.Int64)
    s1 = pl.read_parquet(os.path.join(share, "ids", f"n_{split}_s1.parquet"), columns=["id"]).with_row_index("si") \
           .select(pl.col("si").cast(pl.Int32), key(pl.col("id")).alias("s1"))
    r = pl.concat([pl.read_parquet(os.path.join(share, "ids", f"n_{split}_s{k}.parquet"), columns=["id"]) for k in (2, 3)]) \
          .with_row_index("ri").select(pl.col("ri").cast(pl.Int32), key(pl.col("id")).alias("cand"))
    e = pl.read_parquet(os.path.join(tdir, f"razan_export_{split}.parquet"))
    return e.join(s1, on="s1").join(r, on="cand").select("si", "ri", pl.col("p_muaz").alias("pm"))


def add_teammate(d, pm):
    d = d.join(pm, on=["si", "ri"], how="left")
    return d.with_columns(pl.col("pm").is_null().cast(pl.Float32).alias("pm_na"), pl.col("pm").max().over("ri").alias("pm_top"),
                          pl.col("pm").rank("ordinal", descending=True).over("ri").alias("pm_rk"), (pl.col("p") - pl.col("pm")).alias("pm_gap")) \
            .with_columns([pl.col(c).cast(pl.Float32) for c in FEATS_M])


def build(split, pairs):
    """pairs: si, ri, p (+y). Returns pairs with FEATS (Float32)."""
    s1 = pl.read_parquet(w(f"n_{split}_s1.parquet"), columns=["country", "core"]).with_row_index("si").with_columns(pl.col("si").cast(pl.Int32))
    rr = pl.concat([pl.read_parquet(w(f"n_{split}_s{k}.parquet"), columns=["country", "core", "f_noaddr"]).with_columns(pl.lit(k == 3).alias("src3"))
                    for k in (2, 3)]).with_row_index("ri").with_columns(pl.col("ri").cast(pl.Int32))
    rows = rr.join(pairs.group_by("ri").agg(pl.col("p").max().alias("bp")), on="ri", how="left").with_columns(pl.col("bp").fill_null(0.0))
    rc = rows.group_by("country", "core").agg(pl.len().alias("rc_rows"),
                                              (~pl.col("f_noaddr") & (pl.col("bp") < CLAIM)).sum().alias("rc_unclaimed"),
                                              (~pl.col("f_noaddr") & (pl.col("bp") >= CLAIM)).sum().alias("rc_claimed"),
                                              pl.col("f_noaddr").sum().alias("rc_noaddr"))
    kc = s1.group_by("country", "core").agg(pl.len().alias("k_core_s"))
    second = pairs.sort("p", descending=True).group_by("ri", maintain_order=True).agg(pl.col("p").slice(1, 1).first().alias("p2"))
    d = pairs.join(rr.rename({"core": "rcore"}), on="ri").join(s1.rename({"core": "score", "country": "c1"}), on="si")
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("ri").alias("rk"), pl.col("p").max().over("ri").alias("top_p"),
                       pl.col("p").sum().over("ri").alias("psum_r"), pl.len().over("ri").alias("n_r"), pl.len().over("si").alias("n_s"),
                       pl.col("p").sum().over("si").alias("mass_s"), (pl.col("p") >= CLAIM).sum().over("si").alias("conf_s"),
                       (pl.col("rcore") == pl.col("score")).alias("core_eq"), pl.col("f_noaddr").alias("noaddr"))
    d = d.join(second, on="ri", how="left").with_columns(pl.col("p2").fill_null(0.0))
    d = d.join(rc, left_on=["country", "rcore"], right_on=["country", "core"], how="left") \
         .join(rc.select("country", "core", pl.col("rc_unclaimed").alias("sc_unclaimed")), left_on=["c1", "score"], right_on=["country", "core"], how="left") \
         .join(kc, left_on=["c1", "score"], right_on=["country", "core"], how="left")
    return d.with_columns([pl.col(c).fill_null(0) for c in FEATS if c != "p"]).with_columns([pl.col(c).cast(pl.Float32) for c in FEATS])


def main(share, out, gt_path=None, tdir=None):
    feats = FEATS + (FEATS_M if tdir else [])
    os.makedirs(out, exist_ok=True)
    ids = pl.read_parquet(os.path.join(share, "ids", "n_train_s1.parquet"), columns=["id"]).with_row_index("si").with_columns(pl.col("si").cast(pl.Int32))
    live = ids.join(pl.read_parquet(os.path.join(share, "dropped_train_s1.parquet"), columns=["id"]), on="id", how="anti").select("si")
    oof = pl.read_parquet(os.path.join(share, "oof_s2ce.parquet"), columns=["si", "ri", "p", "y"])
    D = build("train", oof).with_columns((pl.col("si").hash(11) % 2).alias("fold"))
    if tdir:
        D = add_teammate(D, teammate_p(tdir, "train", share))
        print(f"M11: teammate p on {D['pm'].is_not_null().sum()} of {D.height} train pairs", flush=True)
    D = D.join(live.with_columns(pl.lit(True).alias("live")), on="si", how="left").with_columns(pl.col("live").fill_null(False))
    parts, iters = [], []
    for k in (0, 1):
        tr, va = D.filter((pl.col("fold") != k) & pl.col("live")), D.filter(pl.col("fold") == k)
        vl = va.filter(pl.col("live"))
        ds = lgb.Dataset(tr.select(feats).to_numpy(), tr["y"].to_numpy())
        m = lgb.train(PAR, ds, 2000, valid_sets=[lgb.Dataset(vl.select(feats).to_numpy(), vl["y"].to_numpy(), reference=ds)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        iters.append(m.best_iteration)
        parts.append(va.select("ri", "si", pl.Series("p", m.predict(va.select(feats).to_numpy(), num_iteration=m.best_iteration))))
        del tr, va, vl, ds, m
        gc.collect()
    # same schema as the M9 share OOF (ri, si, y Int8, p Float32): a drop-in for src/razan/ (the blend input)
    P = pl.concat(parts).join(oof.select("ri", "si", "y"), on=["ri", "si"]).select("ri", "si", "y", pl.col("p").cast(pl.Float32)).sort(["ri", "si"])
    assert P.height == oof.height
    P.write_parquet(os.path.join(out, "oof_s2ce.parquet"))
    X, y = D.filter(pl.col("live")).select(feats).to_numpy(), D.filter(pl.col("live"))["y"].to_numpy()
    del D, parts
    gc.collect()
    m = lgb.train(PAR, lgb.Dataset(X, y, free_raw_data=True), int(np.mean(iters)))
    m.save_model(os.path.join(out, "stage3.txt"))
    del X, y
    gc.collect()
    te = pl.read_parquet(os.path.join(share, "pred_test_s2ce.parquet"), columns=["si", "ri", "p"])
    Dt = build("test", te)
    if tdir:
        Dt = add_teammate(Dt, teammate_p(tdir, "test", share))
    Pt = Dt.select("ri", "si", pl.Series("p", m.predict(Dt.select(feats).to_numpy()))).sort(["ri", "si"])
    assert Pt.height == te.height
    Pt.write_parquet(os.path.join(out, "pred_test_s2ce.parquet"))
    print(f"stage3: iters {iters}, oof {P.height} pairs, test {Pt.height} pairs -> {out}", flush=True)
    if gt_path:  # CV on live S1s, same metric as run.py
        from decide import decide
        rid = pl.concat([pl.read_parquet(os.path.join(share, "ids", f"n_train_s{k}.parquet"), columns=["id"]) for k in (2, 3)]) \
                .with_row_index("ri").with_columns(pl.col("ri").cast(pl.Int32))
        gt = pl.read_parquet(gt_path).join(ids.rename({"id": "source1_entity_id"}), on="source1_entity_id") \
               .join(rid.rename({"id": "m"}), on="m").select("si", "ri").join(live, on="si").with_columns(pl.lit(True).alias("true"))
        variants = [("M9", oof), ("M11" if tdir else "M10", P)]
        if tdir:
            from sklearn.isotonic import IsotonicRegression
            A = oof.join(teammate_p(tdir, "train", share), on=["si", "ri"], how="left")                    .with_columns(pl.when(pl.col("pm").is_null()).then(pl.col("p")).otherwise(0.5 * pl.col("p") + 0.5 * pl.col("pm")).alias("pa"),
                                 (pl.col("si").hash(11) % 2).alias("fold")).join(live, on="si")
            parts2 = []
            for k in (0, 1):
                tr, va = A.filter(pl.col("fold") != k), A.filter(pl.col("fold") == k)
                iso = IsotonicRegression(out_of_bounds="clip").fit(tr["pa"].to_numpy(), tr["y"].to_numpy())
                parts2.append(va.select("ri", "si", pl.Series("p", iso.predict(va["pa"].to_numpy()))))
            variants.append(("avg(M9, teammate)+iso", pl.concat(parts2)))
        for name, pr in variants:
            kept, _ = decide(pr.join(live, on="si").select(pl.col("ri").alias("id"), pl.col("si").alias("s1"), "p"))
            A = gt.join(kept.select(pl.col("s1").alias("si"), pl.col("id").alias("ri"), pl.lit(True).alias("kept")), on=["si", "ri"], how="full", coalesce=True) \
                  .with_columns(pl.col("true").fill_null(False), pl.col("kept").fill_null(False))
            per = live.join(A.group_by("si").agg((pl.col("true") & pl.col("kept")).sum().alias("tp"), (~pl.col("true") & pl.col("kept")).sum().alias("fp"),
                                                 (pl.col("true") & ~pl.col("kept")).sum().alias("fn")), on="si", how="left").fill_null(0)
            tp, fp, fn = (per[c].to_numpy() for c in ("tp", "fp", "fn"))
            F = np.where(tp + fp + fn == 0, 1.0, np.where(tp == 0, 0.0, 1.25 * tp / np.maximum(1.25 * tp + 0.25 * fn + fp, 1e-9)))
            print(f"{name} CV (live S1s): {F.mean():.5f}", flush=True)


if __name__ == "__main__":
    main(*[a or None for a in sys.argv[1:5]])
