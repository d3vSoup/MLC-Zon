import pyarrow.parquet as pq

from ber.features import names2, numbers
from ber.io import work


# round-2 extra features (soft house numbers + v2 names), row-aligned with r2.parquet (hard pairs only)
def run(split, force=False):
    rows = pq.read_table(work(split, "r2.parquet"), columns=["row"]).column(0).to_numpy()
    numbers.run(split, force=force, rows=rows, out=work(split, "r2_num.parquet"))
    if force or not work(split, "r2_names2.parquet").exists():
        names2.run(split, rows, work(split, "r2_names2.parquet"))
