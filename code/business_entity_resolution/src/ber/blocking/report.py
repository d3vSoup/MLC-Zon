import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.io import read_truth, read_tsv, source_file, to_key, work


# PC by country/source, cands per S1, reduction ratio, sample misses
def run(split="train", force=False):
    names = pq.ParquetFile(work(split, "cands.parquet")).schema_arrow.names
    chans = [n[:-6] for n in names if n.endswith("_score")]
    c = pq.read_table(work(split, "cands.parquet"), columns=["s1", "cand"] + [f"{x}_score" for x in chans]
                      + (["label"] if split == "train" else [])).to_pandas()
    s1 = read_tsv(source_file(split, 1), usecols=["entity_id", "country"])
    n_other = sum(len(read_tsv(source_file(split, s), usecols=["entity_id"])) for s in (2, 3))
    per = c.groupby("s1").size().reindex(to_key(s1["entity_id"]), fill_value=0)
    out = [f"## blocking report ({split})",
           f"pairs {len(c)}; cands per S1 mean {per.mean():.2f} p95 {per.quantile(0.95):.0f} max {per.max()}; "
           f"S1 with zero cands {(per == 0).mean():.4f}",
           f"reduction ratio {1 - len(c) / (len(s1) * n_other):.8f}"]
    if split == "train":
        _, long = read_truth()
        long["s1"], long["cand"] = to_key(long["s1_id"]), to_key(long["cand_id"])
        long["country"] = long["s1"].map(pd.Series(s1["country"].values, index=to_key(s1["entity_id"])))
        found = pd.MultiIndex.from_frame(c.loc[c.label == 1, ["s1", "cand"]])
        long["found"] = pd.MultiIndex.from_frame(long[["s1", "cand"]]).isin(found)
        long["src"] = long["cand_id"].str[:2]
        out.append(f"PC overall **{long.found.mean():.4f}**")
        out.append("PC by country: " + str(long.groupby("country").found.mean().round(4).to_dict()))
        out.append("PC by source: " + str(long.groupby("src").found.mean().round(4).to_dict()))
        pos = c[c.label == 1]
        n_true = len(long)
        for x in chans:
            others = pos[[f"{y}_score" for y in chans if y != x]].notna().any(axis=1) if len(chans) > 1 \
                else pd.Series(False, index=pos.index)
            out.append(f"channel {x}: alone PC {pos[f'{x}_score'].notna().sum() / n_true:.4f}; "
                       f"drop-one PC delta {-(len(pos) - others.sum()) / n_true:+.4f}; "
                       f"pairs {c[f'{x}_score'].notna().sum()}")
        miss = long[~long.found].sample(30, random_state=0)
        info = pd.concat([pd.read_parquet(work(split, f"norm_s{s}.parquet"),
                                          columns=["entity_id", "name_raw", "addr_raw"]) for s in (1, 2, 3)])
        info = info.set_index("entity_id")
        out.append("### 30 missed true pairs\n```")
        for _, r in miss.iterrows():
            a, b = info.loc[r.s1_id], info.loc[r.cand_id]
            out.append(f"{a.name_raw} | {a.addr_raw}  <->  {b.name_raw} | {b.addr_raw}")
        out.append("```")
    txt = "\n".join(out)
    print(txt)
    (work("reports", f"blocking_{split}.md")).write_text(txt)
