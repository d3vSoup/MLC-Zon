"""Exact challenge metric: per-S1 F0.5 (singletons included), macro-averaged."""
import polars as pl

TEST_MIX = {"US": 0.383, "India": 0.468, "France": 0.150}


def f05(ents, pred, truth):
    """ents: s1[, country]; pred/truth: (s1, id) pairs. Returns ents with tp, n_pred, n_true, f."""
    tp = pred.join(truth, on=["s1", "id"]).group_by("s1").len().rename({"len": "tp"})
    npred = pred.group_by("s1").len().rename({"len": "n_pred"})
    ntrue = truth.group_by("s1").len().rename({"len": "n_true"})
    e = ents.join(tp, on="s1", how="left").join(npred, on="s1", how="left").join(ntrue, on="s1", how="left") \
            .with_columns(pl.col("tp", "n_pred", "n_true").fill_null(0))
    f = pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0) \
          .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0)).then(0.0) \
          .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
    return e.with_columns(f.alias("f"))


def report(e, tag=""):
    """Overall, per-country, and test-mix-weighted (France proxied by the mean of other countries)."""
    by = dict(e.group_by("country").agg(pl.col("f").mean()).iter_rows())
    known = {c: v for c, v in by.items() if c in TEST_MIX}
    fr = by.get("France", sum(known.values()) / max(len(known), 1))
    mix = sum(TEST_MIX[c] * (by.get(c, fr) if c != "France" else fr) for c in TEST_MIX)
    s = f"{tag} F0.5={e['f'].mean():.5f} mix={mix:.5f} " + " ".join(f"{c}={v:.5f}" for c, v in sorted(by.items()))
    print(s, flush=True)
    return e["f"].mean(), mix


if __name__ == "__main__":
    ents = pl.DataFrame({"s1": ["S1-1", "S1-2", "S1-3", "S1-4"], "country": ["US"] * 4})
    truth = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-2"], "id": ["S2-47", "S3-812", "S3-4"]})
    pred = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-1", "S1-4"], "id": ["S2-47", "S2-193", "S3-812", "S2-9"]})
    e = f05(ents, pred, truth).sort("s1")
    assert abs(e["f"][0] - 0.7142857) < 1e-6, e  # README example
    assert e["f"].to_list()[1:] == [0.0, 1.0, 0.0]  # missed all / correct empty singleton / link on singleton
    print("score.py self-check OK", e["f"].to_list())
