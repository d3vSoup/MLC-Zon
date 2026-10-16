"""End-to-end entry point.

python run.py all            # normalise -> block -> features -> train (2-fold CV) -> predict test -> decide -> write
python run.py <step> [...]   # steps: norm block feats train predict
Outputs: output/matching_results.tsv, output/candidate_pairs.tsv (then run the official validator).
"""
import os
import sys
import time

import polars as pl

from common import w, ROOT

OUT = os.environ.get("BER_OUT", os.path.join(ROOT, "output"))


def write_tsv(path, s1ids, pairs, col):
    """s1ids: all S1 ids in file order; pairs: (s1, id) strings. One row per S1, comma-joined, empty if none."""
    lists = pairs.unique().sort("id").group_by("s1").agg(pl.col("id").str.join(","))
    df = s1ids.join(lists, on="s1", how="left").select(pl.col("s1").alias("source1_entity_id"),
                                                        pl.col("id").fill_null("").alias(col))
    df.write_csv(path, separator="\t", quote_style="never")
    print(f"wrote {path}: {df.height} rows, {pairs.height} ids", flush=True)


def to_ids(split, p):
    s1 = pl.read_parquet(w(f"n_{split}_s1.parquet"), columns=["id"]).with_row_index("si").rename({"id": "s1"})
    r = pl.concat([pl.read_parquet(w(f"n_{split}_s{s}.parquet"), columns=["id"]) for s in (2, 3)]).with_row_index("ri")
    return p.join(s1.with_columns(pl.col("si").cast(pl.Int32)), on="si").join(r.with_columns(pl.col("ri").cast(pl.Int32)), on="ri") \
            .select("s1", "id"), s1.select("s1")


def predict_and_write(tag="s2", base="m"):
    """Final candidate set = stage-2 input pairs (the model actually scored); matches = decoded subset."""
    from model import predict, predict2
    from decide import decide
    t0 = time.time()
    if tag == base:
        p = predict("test", base)
    else:
        p, p1 = predict2("test", tag, base)
        if not os.path.exists(w(f"pred_test_{base}.parquet")):  # may be a read-only upstream input
            p1.write_parquet(w(f"pred_test_{base}.parquet"))
    p.write_parquet(w(f"pred_test_{tag}.parquet"))
    kept, ef = decide(p.select(pl.col("ri").alias("id"), pl.col("si").alias("s1"), "p"))
    kept = kept.select(pl.col("s1").alias("si"), pl.col("id").alias("ri"))
    out = os.path.join(OUT, tag)  # one folder per model tag (s2, s2ce, s3 ...)
    os.makedirs(out, exist_ok=True)
    km, s1ids = to_ids("test", kept)
    cm, _ = to_ids("test", p.select("ri", "si"))
    write_tsv(os.path.join(out, "matching_results.tsv"), s1ids, km, "matched_entity_ids")
    write_tsv(os.path.join(out, "candidate_pairs.tsv"), s1ids, cm, "candidate_entity_ids")
    print(f"expected F (model's own estimate) on test: {ef['ef'].mean():.5f}; {time.time() - t0:.0f}s", flush=True)


def main(steps):
    import norm, block, feats, model
    for s in steps:
        if s == "norm":
            table = norm.learn_translit()
            for sp in ("train", "test"):
                for k in (1, 2, 3):
                    norm.prep(sp, k, table)
        elif s == "block":
            block.run("train"), block.run("test")
        elif s == "feats":
            feats.featurize("train"), feats.featurize("test")
        elif s == "train":
            model.train("m")
        elif s == "train2":
            model.train2("s2", "m")
        elif s == "predict":
            predict_and_write("s2", "m")
        elif s == "predict1":
            predict_and_write("m", "m")
        elif s == "train2ce":  # stage 2 with the cross-encoder feature (needs ce_train/ce_test.parquet)
            model.train2("s2ce", "m")
        elif s == "predictce":
            predict_and_write("s2ce", "m")
        elif s == "fit1":  # sharded stage 1: fit1 -> predshard:i:n (parallel kernels) -> gather1
            model.fit1("m")
        elif s.startswith("predshard:"):
            model.pred_shard(*map(int, s.split(":")[1:]), "m")
        elif s == "gather1":
            model.gather1("m")
        elif s == "predtwin":  # M6: stage-1 OOF for synthetic twin rows (after feats.py on their candidate files)
            model.pred_twin("m")
        elif s == "train2g":  # M3: stage 2 + generic-word swap features (set BER_GENERIC=1)
            model.train2("s2g", "m")
        elif s == "predictg":
            predict_and_write("s2g", "m")
        elif s == "train3":  # stage 3: collective features recomputed from stage-2 (s2ce) probabilities
            model.train2("s3", "s2ce")
        elif s == "predict3":
            predict_and_write("s3", "s2ce")


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    main(["norm", "block", "feats", "train", "train2", "predict"] if args == ["all"] else args)
