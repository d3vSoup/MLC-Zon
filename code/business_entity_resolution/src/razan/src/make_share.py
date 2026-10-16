"""M9 share bundle (the input layout of stage3.py and of the blend): si/ri -> entity id maps, the dropped train S1s,
and the stage-2 predictions with the synthetic twin rows removed from the train OOF.

usage: BER_WORK=<dir with n_{train,test}_s{1,2,3}.parquet> BER_DROP=16 python make_share.py <m9_dir> <share_dir>
  m9_dir: oof_s2ce.parquet (twin rows included) and pred_test_s2ce.parquet of the qtpies-m9 kernel
"""
import os
import shutil
import sys

import polars as pl

from common import DROP, dropped, w

src, out = sys.argv[1:3]
assert DROP == 16, "BER_DROP must be 16 (the density world of the M9 OOF)"
os.makedirs(os.path.join(out, "ids"), exist_ok=True)
ids = lambda f: os.path.join(out, "ids", f)
n_real = {}
for split in ("train", "test"):  # si = row of n_{split}_s1; ri = row of concat(n_{split}_s2, n_{split}_s3)
    pl.read_parquet(w(f"n_{split}_s1.parquet"), columns=["id", "country"]).with_row_index("si") \
      .with_columns(pl.col("si").cast(pl.Int32)).write_parquet(ids(f"n_{split}_s1.parquet"))
    off = 0
    for k in (2, 3):
        d = pl.read_parquet(w(f"n_{split}_s{k}.parquet"), columns=["id", "country"]).with_row_index("ri", offset=off) \
              .with_columns(pl.col("ri").cast(pl.Int32))
        d.write_parquet(ids(f"n_{split}_s{k}.parquet"))
        off += d.height
    n_real[split] = off  # ri >= this are synthetic twin rows (train only)
drop = pl.read_parquet(ids("n_train_s1.parquet")).filter(dropped("si"))
drop.write_parquet(os.path.join(out, "dropped_train_s1.parquet"))
oof = pl.read_parquet(os.path.join(src, "oof_s2ce.parquet"))
twin = oof.filter(pl.col("ri") >= n_real["train"])
oof = oof.filter(pl.col("ri") < n_real["train"])
assert oof.join(drop.select("si"), on="si", how="semi").height == 0 and twin["y"].sum() == 0
oof.write_parquet(os.path.join(out, "oof_s2ce.parquet"))
shutil.copyfile(os.path.join(src, "pred_test_s2ce.parquet"), os.path.join(out, "pred_test_s2ce.parquet"))
print(f"share -> {out}: oof {oof.height} pairs ({twin.height} twin pairs removed), dropped S1 {drop.height}")
