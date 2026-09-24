import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ber.io import read_truth, read_tsv, source_file, work


# unicode block names used in a sample of strings
def scripts(strings):
    c = Counter()
    for s in strings:
        for ch in s:
            if ord(ch) > 127:
                c[unicodedata.name(ch, "UNNAMED").split()[0]] += 1
    return c.most_common(15)


# phase 0 facts report
def run():
    out = []
    w = out.append
    rng = np.random.default_rng(42)

    w("# Phase 0 facts\n")
    w("## 1. Row counts and country mix\n")
    src = {}
    rows = []
    for split in ("train", "test"):
        for s in (1, 2, 3):
            df = read_tsv(source_file(split, s))
            rows.append({"file": f"{split}_source{s}", "rows": len(df),
                         **df["country"].value_counts().to_dict()})
            if split == "train":
                src[s] = df
            if split == "test" and s == 1:
                test_s1 = df
            # item 10 + 11 per file
            smp = df.sample(min(300_000, len(df)), random_state=42)
            rows[-1]["_scripts_name"] = scripts(smp["business_name"])[:10]
            rows[-1]["_scripts_addr"] = scripts(smp["business_address"])[:10]
            miss = df.groupby("country").agg(
                n=("entity_id", "size"),
                empty_name=("business_name", lambda x: (x.str.strip() == "").mean()),
                empty_addr=("business_address", lambda x: (x.str.strip() == "").mean()),
                addr_no_digit=("business_address", lambda x: (~x.str.contains(r"\d")).mean()),
                name_nonascii=("business_name", lambda x: x.str.contains(r"[^\x00-\x7f]").mean()),
            )
            rows[-1]["_miss"] = miss
            del df
    cnt = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).fillna(0)
    w("```\n" + cnt.to_string(index=False) + "\n```\n")
    fr = cnt.loc[cnt.file == "test_source1", "France"].iloc[0] / cnt.loc[cnt.file == "test_source1", "rows"].iloc[0]
    w(f"France share of test S1: **{fr:.4f}**\n")

    gt, long = read_truth()
    g = gt["matched_entity_ids"].map(lambda x: len([i for i in x.split(",") if i]))
    w("## 2. Singleton share (all-empty floor)\n")
    w(f"**{(g == 0).mean():.5f}** of {len(g)} train S1\n")

    w("## 3. G distribution\n")
    gb = g.clip(upper=6).value_counts().sort_index()
    w("```\n" + pd.DataFrame({"n": gb, "share": (gb / len(g)).round(4)}).rename(index={6: "6+"}).to_string() + "\n```")
    long["src"] = long["cand_id"].str[:2]
    w(f"mean G={g.mean():.3f}; matches by source: {long['src'].value_counts().to_dict()}\n")

    w("## 4. One-owner rule\n")
    dup = long["cand_id"].duplicated().sum()
    w(f"IDs in more than one GT list: **{dup}** (of {long['cand_id'].nunique()} matched IDs)\n")

    w("## 5. Cross-country matched pairs\n")
    ctry = pd.concat([src[s].set_index("entity_id")["country"] for s in (1, 2, 3)])
    c1 = long["s1_id"].map(ctry)
    c2 = long["cand_id"].map(ctry)
    w(f"cross-country pairs: **{(c1 != c2).sum()}**; unresolved ids: {c2.isna().sum()}\n")

    w("## 6. S2/S3 records matching nothing\n")
    matched = set(long["cand_id"])
    for s in (2, 3):
        m = src[s]["entity_id"].isin(matched)
        w(f"- S{s}: unmatched share **{1 - m.mean():.4f}** "
          f"(by country: {(1 - m.groupby(src[s]['country']).mean()).round(4).to_dict()})")
    w("")

    w("## 7. S1 with 2+ records from the same source\n")
    per = long.groupby(["s1_id", "src"]).size().unstack(fill_value=0)
    w(f"share of all S1 with >=2 S2 matches: {(per.get('S2', 0) >= 2).sum() / len(g):.4f}; "
      f">=2 S3: {(per.get('S3', 0) >= 2).sum() / len(g):.4f}; either: {((per >= 2).any(axis=1)).sum() / len(g):.4f}\n")

    w("## 8. Similarity of true pairs (token_set_ratio, lowercased raw)\n")
    info = pd.concat([src[s].set_index("entity_id")[["business_name", "business_address", "country"]] for s in (1, 2, 3)])
    smp = long.sample(200_000, random_state=42).reset_index(drop=True)
    a = info.loc[smp["s1_id"]].reset_index(drop=True)
    b = info.loc[smp["cand_id"]].reset_index(drop=True)
    lo = lambda s: s.str.lower().tolist()
    smp["name_tsr"] = cpdist(lo(a.business_name), lo(b.business_name), scorer=fuzz.token_set_ratio, workers=-1)
    smp["addr_tsr"] = cpdist(lo(a.business_address), lo(b.business_address), scorer=fuzz.token_set_ratio, workers=-1)
    smp["country"] = a["country"]
    q = smp.groupby(["src", "country"])[["name_tsr", "addr_tsr"]].quantile([0.05, 0.25, 0.5, 0.75]).unstack()
    w("```\n" + q.round(1).to_string() + "\n```\n")

    w("## 9. Hardest true pairs and most similar non-pairs\n")
    w("### 30 true pairs with lowest name similarity\n```")
    for i in smp.nsmallest(30, "name_tsr").index:
        w(f"{smp.name_tsr[i]:5.1f} | {a.business_name[i]} | {a.business_address[i]}  <->  "
          f"{b.business_name[i]} | {b.business_address[i]}")
    w("```\n### 30 random same-country non-pairs with highest name similarity\n```")
    s1 = src[1]
    oth = pd.concat([src[2], src[3]])
    rows_np = []
    for c in s1["country"].unique():
        x = s1[s1.country == c].sample(100_000, replace=True, random_state=1).reset_index(drop=True)
        y = oth[oth.country == c].sample(100_000, replace=True, random_state=2).reset_index(drop=True)
        rows_np.append(pd.DataFrame({"s1": x.entity_id, "c": y.entity_id, "an": x.business_name, "aa": x.business_address,
                                     "bn": y.business_name, "ba": y.business_address}))
    npairs = pd.concat(rows_np, ignore_index=True)
    true = set(zip(long.s1_id, long.cand_id))
    npairs = npairs[[p not in true for p in zip(npairs.s1, npairs.c)]].reset_index(drop=True)
    npairs["sim"] = cpdist(lo(npairs.an), lo(npairs.bn), scorer=fuzz.token_set_ratio, workers=-1)
    for _, r in npairs.nlargest(30, "sim").iterrows():
        w(f"{r.sim:5.1f} | {r.an} | {r.aa}  <->  {r.bn} | {r.ba}")
    w("```\n")

    w("## 10. Scripts (Unicode block of non-ASCII chars, 300k-row sample per file)\n")
    for r in rows:
        w(f"- {r['file']}: name {r['_scripts_name']}\n  addr {r['_scripts_addr']}")
    w("")

    w("## 11. Missing / format rates per file and country\n")
    for r in rows:
        w(f"### {r['file']}\n```\n{r['_miss'].round(4).to_string()}\n```")

    w("## 12. Top 30 most frequent S1 names (lowercased)\n")
    for name, s1df in (("train", s1), ("test", test_s1)):
        vc = s1df["business_name"].str.lower().str.strip().value_counts().head(30)
        w(f"### {name}\n```\n{vc.to_string()}\n```")
    w(f"distinct lowercased S1 names train: {s1['business_name'].str.lower().nunique()} / {len(s1)}\n")

    p = work("reports", "phase0_facts.md")
    p.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    run()
