# label-free: share of decoded links (blend_cal test) whose record address starts with a house number the S1 address lacks,
# per country; train reference = true pairs per country. writes work/hailmary/prefix_by_country.json
import json, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from ber.io import read_tsv, read_truth, work, path
lead = lambda v: v.fillna("").str.match(r"^\s*(no\b|no\.|door|h\.?\s?no|plot|#|\d)", case=False).values
def tab(split):
    t = pd.concat([pd.read_parquet(work(split, f"norm_s{s}.parquet"), columns=["entity_id", "country", "addr_raw"]) for s in (1, 2, 3)])
    return t.set_index("entity_id")
res = {}
lk = read_tsv(path("output_dir") / "candidates" / "blend_cal" / "matching_results.tsv")
lk = lk[lk.matched_entity_ids != ""].assign(c=lambda x: x.matched_entity_ids.str.split(",")).explode("c")
t = tab("test")
a, b = t.reindex(lk.source1_entity_id.values), t.reindex(lk.c.values)
f = lead(a["addr_raw"]) 
g = lead(b["addr_raw"]) & ~f
for c in pd.unique(a["country"]):
    m = a["country"].values == c
    res[f"test_links_{c}"] = {"links": int(m.sum()), "S1_addr_no_lead_num": round(float((~f[m]).mean()), 4), "rec_lead_num_S1_not": round(float(g[m].mean()), 4)}
_, tr = read_truth()
t = tab("train")
a, b = t.reindex(tr.s1_id.values), t.reindex(tr.cand_id.values)
f = lead(a["addr_raw"]); g = lead(b["addr_raw"]) & ~f
for c in pd.unique(a["country"]):
    m = a["country"].values == c
    res[f"train_true_{c}"] = {"pairs": int(m.sum()), "S1_addr_no_lead_num": round(float((~f[m]).mean()), 4), "rec_lead_num_S1_not": round(float(g[m].mean()), 4)}
s1 = pd.read_parquet(work("test", "norm_s1.parquet"), columns=["country", "addr_raw"])
for c, gg in s1.groupby("country"):
    res[f"test_S1_{c}_no_lead_num"] = round(float((~lead(gg["addr_raw"])).mean()), 4)
print(json.dumps(res, indent=1))
json.dump(res, open(work("hailmary", "prefix_by_country.json"), "w"), indent=1)
