"""M11h bundle = M11 stage-3 p (stacker with the step-A model's p; trained on US/India) for US/India test pairs and M10 p
for France test pairs (the stacker never saw France: train has no France); train OOF = M11.
usage: python make_m11h.py <m9_share> <m10_share> <m11_share> <out_dir>"""
import os
import shutil
import sys

import polars as pl

m9, m10, m11, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
c = pl.read_parquet(os.path.join(m9, "ids", "n_test_s1.parquet"), columns=["si", "country"])
t11 = pl.read_parquet(os.path.join(m11, "pred_test_s2ce.parquet"))
t10 = pl.read_parquet(os.path.join(m10, "pred_test_s2ce.parquet")).rename({"p": "p10"})
h = t11.join(t10, on=["ri", "si"]).join(c, on="si") \
       .with_columns(pl.when(pl.col("country") == "France").then(pl.col("p10")).otherwise(pl.col("p")).alias("p")).select("ri", "si", "p").sort(["ri", "si"])
assert h.height == t11.height == t10.height
h.write_parquet(os.path.join(out, "pred_test_s2ce.parquet"))
shutil.copy(os.path.join(m11, "oof_s2ce.parquet"), os.path.join(out, "oof_s2ce.parquet"))
print("M11h ->", out, h.height)
