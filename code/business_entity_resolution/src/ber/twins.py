import json
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import ber.blend as B
import ber.decode as D
from ber.io import CFG, read_tsv, to_key, work
from ber.metric import f05

# twin decoys: test-only near-copies (measured label-free on test). Signatures on decoded links, label-free:
#   num: core name identical, house number shifted 1..20, legal suffix added, and the S1 keeps a sibling with its exact number
#   gen: one generic word swapped (generic = token in > generic_df of the split's S1 names), rest of name + number + address same
# measured as kept links per 1k S1 on test vs train OOF; the cost of dropping them is exact on train (labels).


def hn(nums):
    for x in nums if nums is not None else ():
        if str(x).isdigit():
            return int(x)
    return -1


def table(split, keys):
    keys = np.asarray(sorted(keys))
    fs = []
    for s in (1, 2, 3):  # one source at a time, keep only needed rows (low RAM)
        f = pq.read_table(work(split, f"norm_s{s}.parquet"), columns=["entity_id", "name_core", "legal_class", "numbers", "addr_norm"]).to_pandas()
        f.index = to_key(f.pop("entity_id"))
        fs.append(f.loc[np.isin(f.index.values, keys)])
        del f
    t = pd.concat(fs)
    t["hn"] = t["numbers"].map(hn)
    t["rest"] = t["addr_norm"].str.replace(r"\d+", "", regex=True).str.split().map(lambda x: " ".join(sorted(x)))
    return t.drop(columns=["numbers", "addr_norm"])


def attrs(split, s1, cand):
    t = table(split, set(s1.tolist()) | set(cand.tolist()))
    a, b = t.reindex(s1), t.reindex(cand)
    s1c = pd.read_parquet(work(split, "norm_s1.parquet"), columns=["name_core"])["name_core"]
    df = s1c.str.split().explode().value_counts() / len(s1c)
    generic = set(df.index[df > CFG["names2"]["generic_df"]])
    ta, tb = a["name_core"].str.split().map(set).values, b["name_core"].str.split().map(set).values
    dlt = b["hn"].values - a["hn"].values
    ok = (a["hn"].values >= 0) & (b["hn"].values >= 0)
    x = pd.DataFrame({"s1": s1, "cand": cand})
    x["core_eq"] = a["name_core"].values == b["name_core"].values
    x["legal_add"] = (b["legal_class"].values != "") & (b["legal_class"].values != a["legal_class"].values)
    x["num_shift"] = ok & (np.abs(dlt) >= 1) & (np.abs(dlt) <= 20)
    x["num_same"] = ok & (dlt == 0)
    swap = [len(p - q) == 1 and len(q - p) == 1 and (p - q) <= generic and (q - p) <= generic and len(p & q) > 0
            for p, q in zip(ta, tb)]
    x["gen_swap"] = np.array(swap) & x["num_same"].values & (a["rest"].values == b["rest"].values)
    return x


def flag(x):
    sib = x[x["num_same"]].groupby("s1").size()  # S1 keeps a sibling with its exact house number
    has = x["s1"].map(sib).fillna(0).values > 0
    x["tw_num"] = x["core_eq"] & x["num_shift"] & x["legal_add"] & has
    x["tw_gen"] = x["gen_swap"]
    return x


# decoded links of the blend: train OOF (live S1, with labels) and test (calibrated)
def links(model="lgbm2_ce", w=0.5, mode="asis"):
    D.NAME = f"blend_{w}_{mode}"
    u, live = B.union("train", model), B.live_mask("train")
    d, info = B.owned(u, B.mix(u, w, mode), "train", live)
    del u
    d["q"] = D.calibrate_oof(d, info["fold"])
    gate = D.fit_gate(D.gate_frame(d, info, "train"), (info["G"] > 0).astype(int), info["fold"])
    m = D.expf_mask(d, len(info["keys"]), gate)
    tr = pd.DataFrame({"s1": info["keys"][d["code"].values[m]], "cand": d["cand"].values[m], "label": d["label"].values[m],
                       "q": d["q"].values[m], "code": d["code"].values[m]})
    G, ctry_tr = info["G"], info["country"]
    del d, info
    cal = json.load(open(work("reports", f"calibrate_{D.NAME}.json")))
    u = B.union("test", model)
    d, info = B.owned(u, B.mix(u, w, mode), "test")
    k = pd.Series(info["country"]).map(cal["k_unseen_only"]).fillna(1.0).values[d["code"].values]
    p = d["p"].values
    d["q"] = pickle.load(open(work("models", f"iso_{D.NAME}.pkl"), "rb")).predict(p * k / (p * k + 1 - p))
    m = D.expf_mask(d, len(info["keys"]), D.gate_test(D.gate_frame(d, info, "test"), range(CFG["n_folds"])))
    te = pd.DataFrame({"s1": info["keys"][d["code"].values[m]], "cand": d["cand"].values[m], "q": d["q"].values[m],
                       "code": d["code"].values[m]})
    return tr, G, ctry_tr, te, info["country"]


def run(model="lgbm2_ce", w=0.5, mode="asis"):
    tr, G, ctry_tr, te, ctry_te = links(model, w, mode)
    tr = flag(pd.concat([tr, attrs("train", tr["s1"].values, tr["cand"].values).drop(columns=["s1", "cand"])], axis=1))
    te = flag(pd.concat([te, attrs("test", te["s1"].values, te["cand"].values).drop(columns=["s1", "cand"])], axis=1))
    rows = []
    for sig in ("tw_num", "tw_gen"):
        for c in sorted(set(ctry_te)):
            n_te = (ctry_te == c).sum()
            r = {"sig": sig, "country": c, "test_per_1k": round(1000 * te[sig][ctry_te[te["code"].values] == c].sum() / n_te, 2)}
            if c in set(ctry_tr):
                m = tr[sig].values & (ctry_tr[tr["code"].values] == c)
                r.update(train_per_1k=round(1000 * m.sum() / (ctry_tr == c).sum(), 2),
                         train_precision=round(float(tr["label"].values[m].mean()), 3) if m.any() else None)
            rows.append(r)
    print(pd.DataFrame(rows).to_string())
    n = len(G)
    k0 = np.bincount(tr["code"].values, minlength=n)
    t0 = np.bincount(tr["code"].values[tr["label"].values == 1], minlength=n)
    f0 = f05(G, k0, t0).mean()
    res = {"oof": round(float(f0), 5)}
    for sig in ("tw_num", "tw_gen", "both"):
        drop = (tr["tw_num"] | tr["tw_gen"]).values if sig == "both" else tr[sig].values
        k1 = np.bincount(tr["code"].values[~drop], minlength=n)
        t1 = np.bincount(tr["code"].values[~drop & (tr["label"].values == 1)], minlength=n)
        res[f"oof_drop_{sig}"] = round(float(f05(G, k1, t1).mean()), 5)
        res[f"test_links_dropped_{sig}"] = int(((te["tw_num"] | te["tw_gen"]) if sig == "both" else te[sig]).sum())
    print(json.dumps(res, indent=1))
    json.dump({"cells": rows, **res}, open(work("reports", "twins.json"), "w"), indent=1)
    te[["s1", "cand", "tw_num", "tw_gen"]].to_parquet(work("preds", "test_twin_flags.parquet"), index=False)


# drop flagged twin links from a written candidate (same union candidate file, hard-linked); validate, md5, stats
def apply(src="blend_cal", out="blend_cal_twin", sigs=("tw_num",)):
    import hashlib
    import os
    from ber.io import path, source_file, to_id, validate
    f = pd.read_parquet(work("preds", "test_twin_flags.parquet"))
    drop = f[f[list(sigs)].any(axis=1)]
    bad = set(zip(to_id(drop["s1"].values), to_id(drop["cand"].values)))
    base = path("output_dir") / "candidates"
    (base / out).mkdir(exist_ok=True)
    n = 0
    with open(base / src / "matching_results.tsv", encoding="utf-8") as fi, open(base / out / "matching_results.tsv", "w", encoding="utf-8") as fo:
        fo.write(next(fi))
        for line in fi:
            s1, _, rest = line.rstrip("\n").partition("\t")
            ids = [c for c in rest.split(",") if c and (s1, c) not in bad]
            n += len([c for c in rest.split(",") if c]) - len(ids)
            fo.write(f"{s1}\t{','.join(ids)}\n")
    cf = base / out / "candidate_pairs.tsv"
    cf.unlink(missing_ok=True)
    os.link(base / src / "candidate_pairs.tsv", cf)
    validate((base / out / "matching_results.tsv").resolve(), cf.resolve())
    md5 = hashlib.md5(open(base / out / "matching_results.tsv", "rb").read()).hexdigest()
    c = read_tsv(source_file("test", 1), usecols=["entity_id", "country"]).set_index("entity_id").country
    by = pd.Series([c.get(s) for s, _ in bad]).value_counts().to_dict()
    json.dump({"md5": md5, "src": src, "signatures": list(sigs), "links_dropped": n, "dropped_by_country": by},
              open(base / out / "stats.json", "w"), indent=1)
    print(f"{out}: dropped {n} twin links {by}; md5 {md5}")
