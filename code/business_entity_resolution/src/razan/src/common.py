import os
import polars as pl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "student_resource", "dataset"))
SEED = 42
# threads per process (matmul / rapidfuzz / lightgbm); big boxes run several shard processes (see aws/job.py)
THREADS = int(os.environ.get("BER_THREADS", 8))
# density matching: % of TRAIN S1 treated as absent, so their S2/S3 rows become distractors. Test has ~38-40%
# distractor rows vs 26% in train; dropping 16% of train S1 reproduces that (model + CV see test-like density).
DROP = float(os.environ.get("BER_DROP", 0))


def dropped(col="si"):
    """Train S1 rows treated as absent (hash-based, fixed seed). Only ever applied to the train split."""
    return (pl.col(col).hash(SEED + 7) % 100) < DROP


def w(name):
    return os.path.join(WORK, name)


def raw(split, s):
    """Raw source as parquet (converted once from TSV, no quoting)."""
    p = w(f"{split}_s{s}.parquet")
    if not os.path.exists(p):
        pl.read_csv(f"{DATA}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None,
                    infer_schema=False, encoding="utf8-lossy").write_parquet(p)
    return pl.read_parquet(p)


def gt_pairs():
    """One row per true link: s1 id, other id."""
    p = w("gt_pairs.parquet")
    if not os.path.exists(p):
        gt = pl.read_csv(f"{DATA}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
        gt.select("source1_entity_id", pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")) \
          .explode("m").filter(pl.col("m") != "").write_parquet(p)
    return pl.read_parquet(p).rename({"source1_entity_id": "s1", "m": "id"})
