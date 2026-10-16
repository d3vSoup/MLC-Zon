"""Run in the step-A environment after run_all.sh, from code/business_entity_resolution/src:

    PYTHONPATH=. python razan/export_for_razan.py

Writes work/razan_export_{train,test}.parquet with the step-A lgbm2_ce probability for every pair the QTpies bundle in
src/razan/ scored (columns s1, cand = int64 keys from ber.io.to_key; p_muaz = lgbm2_ce p, NaN if it did not score the
pair). Uses blend.union(), so the join is exactly the one the blend uses (train: live S1s only).
~9M rows per split, ~150 MB each. QTpies uses it as a feature of the M11 stacker (cross-fitted on train OOF)."""
import numpy as np
import pandas as pd

from ber.blend import union
from ber.io import work

for split in ("train", "test"):
    u = union(split, "lgbm2_ce")
    u = u[~np.isnan(u["ph"].values)]  # pairs the QTpies bundle scored
    out = pd.DataFrame({"s1": u["s1"].values.astype("int64"), "cand": u["cand"].values.astype("int64"),
                        "p_muaz": u["po"].values.astype("float32")})
    out.to_parquet(work(f"razan_export_{split}.parquet"), index=False)
    print(split, len(out), "pairs;", int(np.isnan(out["p_muaz"].values).sum()), "not scored by lgbm2_ce", flush=True)
