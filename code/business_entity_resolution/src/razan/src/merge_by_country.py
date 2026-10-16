"""Country-wise merge of two matching files: rows of S1s in --a-countries from file A, all other S1 rows from file B.
Macro F0.5 is a sum over S1s and every S1 belongs to one country, so the merge scores exactly A's countries' part plus
B's. Row order follows B; asserts one owner per S2/S3 row in the result.
usage: python merge_by_country.py <A.tsv> <B.tsv> <ids n_test_s1.parquet> <out_dir> [--a-countries France]"""
import argparse
import hashlib
import os

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("s1_ids")
ap.add_argument("out")
ap.add_argument("--a-countries", default="France")
a = ap.parse_args()
cty = pl.read_parquet(a.s1_ids, columns=["id", "country"]).rename({"id": "s1"})
read = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False)
A, B = read(a.a), read(a.b)
head = B.columns
A, B = A.rename(dict(zip(A.columns, ["s1", "ids"]))), B.rename(dict(zip(B.columns, ["s1", "ids"])))
assert A.height == B.height == cty.height and A.join(B, on="s1", how="anti").height == 0
cs = a.a_countries.split(",")
out = B.with_row_index("row").join(cty, on="s1").join(A.rename({"ids": "ids_a"}), on="s1") \
       .with_columns(pl.when(pl.col("country").is_in(cs)).then(pl.col("ids_a")).otherwise(pl.col("ids")).alias("ids")).sort("row")
links = out.filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")).select(pl.col("ids").str.split(",")).explode("ids")
assert links["ids"].is_unique().all(), "an S2/S3 row would have two owners"
os.makedirs(a.out, exist_ok=True)
tsv = os.path.join(a.out, "matching_results.tsv")
out.select(pl.col("s1").alias(head[0]), pl.col("ids").alias(head[1])).write_csv(tsv, separator="\t", quote_style="never", null_value="")
print(f"merged: {cs} from {a.a}, rest from {a.b} -> {tsv} md5 {hashlib.md5(open(tsv, 'rb').read()).hexdigest()}")
