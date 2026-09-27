"""Leaderboard attribution probe: copy of a matching file with one country's match lists emptied.

score(full) - score(probe) = share_c * (F_c - singleton_rate_c)  ->  solve for F_c.
usage: python probe.py <matching_results.tsv> <country> <out.tsv>
"""
import sys

import polars as pl

from common import w

src, country, out = sys.argv[1:4]
m = pl.read_csv(src, separator="\t", quote_char=None, infer_schema=False).fill_null("")
s1 = pl.read_parquet(w("n_test_s1.parquet"), columns=["id", "country"])
m = m.join(s1.rename({"id": "source1_entity_id"}), on="source1_entity_id", how="left")
m = m.with_columns(pl.when(pl.col("country") == country).then(pl.lit("")).otherwise(pl.col("matched_entity_ids"))
                   .alias("matched_entity_ids")).drop("country")
m.write_csv(out, separator="\t", quote_style="never")
print(f"{country}: {s1.filter(pl.col('country') == country).height / s1.height:.4f} of S1 emptied -> {out}")
