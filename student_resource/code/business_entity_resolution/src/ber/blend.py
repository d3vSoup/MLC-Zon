import json
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import ber.decode as D
from ber.io import CFG, path, read_truth, to_key, validate, work, write_id_lists
from ber.models.lgbm import fold_map

# P2: pair-level blend of our round-2 p with the QTpies M5 stage-2 p (shipped as inputs, keyed by entity ids).
# candidate set = union of both pipelines' final candidates; p = w * ours + (1 - w) * theirs; a pair only one side
# scored keeps its p ("asis") or is shrunk as if the other side said 0 ("shrunk"). Train is scored in their
# density-matched world: dropped S1 (their BER_DROP=16 set) and their pairs are removed, only live S1 evaluated.
BL = CFG.get("blend", {})
MIN_P = 1e-4  # a pair below this on both sides can never be kept (their decide drops it too)


def pkey(s1, cand):
    s, c = s1 % 10**10, cand % 10**10
    assert s.max() < 2**30 and c.max() < 2**30
    return (s << 31) | ((cand // 10**10 - 2) << 30) | c


def theirs(split):
    d = BL.get("dir", "code/business_entity_resolution/src/razan")
    f = path("dataset_dir").parent / d
    t = pq.read_table(f / ("oof_s2ce.parquet" if split == "train" else "pred_test_s2ce.parquet"), columns=["ri", "si", "p"]).to_pandas()
    s1 = pd.read_parquet(f / "ids" / f"n_{split}_s1.parquet", columns=["id"])["id"].values
    r = np.concatenate([pd.read_parquet(f / "ids" / f"n_{split}_s{s}.parquet", columns=["id"])["id"].values for s in (2, 3)])
    s1k, ck = to_key(s1[t["si"].values]), to_key(r[t["ri"].values])
    drop = to_key(pd.read_parquet(f / "dropped_train_s1.parquet", columns=["id"])["id"].values) if split == "train" else None
    return s1k, ck, t["p"].values.astype(np.float32), drop


# union pairs (s1, cand, p_ours, p_theirs; NaN = side did not score it); train: dropped S1 removed
def union(split, model):
    tag = "oof" if split == "train" else "test"
    o = pq.read_table(work("preds", f"{tag}_{model}.parquet"), columns=["s1", "cand", "p"])
    os1, oc, op = (o.column(c).to_numpy() for c in ("s1", "cand", "p"))
    del o
    ts1, tc, tp, drop = theirs(split)
    tk = pkey(ts1, tc)
    order = np.argsort(tk)
    tk, ts1, tc, tp = tk[order], ts1[order], tc[order], tp[order]
    ok = pkey(os1, oc)
    at = np.searchsorted(tk, ok)
    at[at == len(tk)] = 0
    hit = tk[at] == ok
    del ok
    keep = (op >= MIN_P) | hit
    ph = np.where(hit, tp[at], np.nan).astype(np.float32)
    seen = np.zeros(len(tk), bool)
    seen[at[hit]] = True
    s1 = np.r_[os1[keep], ts1[~seen]]
    cand = np.r_[oc[keep], tc[~seen]]
    po = np.r_[op[keep], np.full((~seen).sum(), np.nan, np.float32)]
    ph = np.r_[ph[keep], tp[~seen]]
    print(f"{split}: ours kept {int(keep.sum())} (of {len(op)}), both {int(hit.sum())}, theirs only {int((~seen).sum())}", flush=True)
    u = pd.DataFrame({"s1": s1, "cand": cand, "po": po, "ph": ph})
    recs = np.concatenate([to_key(D.read_tsv(D.source_file(split, s), usecols=["entity_id"])["entity_id"]) for s in (2, 3)])
    u = u[np.isin(u["s1"].values, D.s1_table(split)[0]) & np.isin(u["cand"].values, recs)].reset_index(drop=True)  # this run's ids only (sample runs)
    if drop is not None:
        u = u[~np.isin(u["s1"].values, drop)].reset_index(drop=True)
    return u


def mix(u, w, mode):
    po, ph = u["po"].values, u["ph"].values
    both = ~np.isnan(po) & ~np.isnan(ph)
    if mode == "asis":
        p = np.where(both, w * po + (1 - w) * ph, np.where(np.isnan(ph), po, ph))
    else:
        p = w * np.nan_to_num(po) + (1 - w) * np.nan_to_num(ph)
    return p.astype(np.float32)


# (d, info) like decode.owned, from any pair table; train: labels from truth, live S1 only
def owned(u, p, split, live=None):
    keys, country, ids = D.s1_table(split)
    srt = np.argsort(keys)
    code_all = srt[np.searchsorted(keys, u["s1"].values, sorter=srt)]
    n = len(keys)
    info = {"keys": keys, "country": country, "ids": ids, "n_cand": np.bincount(code_all, minlength=n), "sel": np.arange(n)}
    k = D.owner(u["s1"].values, u["cand"].values, p)
    d = pd.DataFrame({"code": code_all[k], "cand": u["cand"].values[k], "p": p[k]})
    if split == "train":
        _, long = read_truth()
        tk = np.sort(pkey(to_key(long["s1_id"]), to_key(long["cand_id"])))
        pk = pkey(u["s1"].values, u["cand"].values)
        at = np.clip(np.searchsorted(tk, pk), 0, len(tk) - 1)
        lab = (tk[at] == pk).astype(np.int8)
        d["label"] = lab[k]
        info["g_cand"] = np.bincount(code_all[lab == 1], minlength=n)
        info["G"] = np.bincount(pd.Index(keys).get_indexer(to_key(long["s1_id"])), minlength=n)
        info["fold"] = fold_map()["fold"].reindex(keys).values
    d = d.iloc[np.lexsort((-d["p"].values, d["code"].values))].reset_index(drop=True)
    d["rank"] = np.arange(len(d)) - np.searchsorted(d["code"].values, d["code"].values, "left")
    if live is not None:
        from ber.calibrate import subset
        d, info = subset(d, info, live)
    return d, info


def live_mask(split):
    if split != "train":
        return None
    return ~np.isin(D.s1_table("train")[0], theirs("train")[3])


# OOF decode of one blend variant (isotonic cross-fit + gate + expected-F), our decoder on live S1
def score_variant(d, info):
    d = d.copy()
    d["q"] = D.calibrate_oof(d, info["fold"])
    g = D.gate_frame(d, info, "train")
    gate = D.fit_gate(g, (info["G"] > 0).astype(int), info["fold"])
    mask = D.expf_mask(d, len(info["keys"]), gate)
    k, t, f = D.score(d, info, mask)
    out = D.summary(f, info)
    out["precision"] = round(float(t.sum() / max(k.sum(), 1)), 5)  # pooled TP / predicted links
    for c in pd.unique(info["country"]):
        m = info["country"] == c
        out[f"precision_{c}"] = round(float(t[m].sum() / max(k[m].sum(), 1)), 5)
    return out


def search(model="lgbm2_ce", grid=((1.0, "shrunk"), (0.0, "shrunk"), (0.5, "asis"), (0.5, "shrunk"),
                                    (0.3, "asis"), (0.7, "asis"), (0.3, "shrunk"), (0.7, "shrunk"))):
    u = union("train", model)
    live = live_mask("train")
    res = {}
    for w, mode in grid:
        D.NAME = f"blend_{w}_{mode}"
        d, info = owned(u, mix(u, w, mode), "train", live)
        res[f"{w}_{mode}"] = score_variant(d, info)
        print(f"w={w} {mode}: {res[f'{w}_{mode}']}", flush=True)
    json.dump(res, open(work("reports", "blend_search.json"), "w"), indent=1)
    return res


# chosen variant: refit iso + gate on the live-S1 OOF, calibrate (null test + DiD + empty-rate) on the blend in the
# density-matched world, decode test raw (k = 1) and calibrated, write both (validated, md5, per-country stats)
def final(model, w, mode, name, base="ce_full_v1", half=None):
    import ber.calibrate as C
    from ber.ce_export import halves
    D.NAME = f"blend_{w}_{mode}" + (f"_{model}" if model != "lgbm2_ce" else "")
    u_tr, live = union("train", model), live_mask("train")
    if half:  # half-trained model: its OOF exists only on that half's S1
        live &= halves(D.s1_table("train")[0]) == half
    d, info = owned(u_tr, mix(u_tr, w, mode), "train", live)
    oof = score_variant(d, info)
    json.dump({"folds": list(range(CFG["n_folds"])), "oof": oof}, open(work("reports", f"decode_{D.NAME}.json"), "w"))
    del d, info
    u_te = union("test", model)
    cal = C.fit(lambda s: owned(u_tr, mix(u_tr, w, mode), "train", live) if s == "train" else owned(u_te, mix(u_te, w, mode), "test"))
    del u_tr
    d, info = owned(u_te, mix(u_te, w, mode), "test")
    iso = pickle.load(open(work("models", f"iso_{D.NAME}.pkl"), "rb"))
    same = path("output_dir") / "candidates" / "blend_raw"  # same union candidate set (our cands + theirs): hard link
    raw_dir = write_out(d, info, iso, {}, u_te, model, f"{name}_raw", {"oof_live": oof, "w": w, "mode": mode}, base,
                        cand_from=same if name != "blend" and (same / "candidate_pairs.tsv").exists() else None)
    meta = {"oof_live": oof, "w": w, "mode": mode, "calibration": cal}
    write_out(d, info, iso, cal["k_unseen_only"], u_te, model, f"{name}_cal", meta, base, cand_from=raw_dir)
    if "k_full" in cal:  # null test passed: seen-country DiD too (used only if the anchor passes the report's gate)
        write_out(d, info, iso, cal["k_full"], u_te, model, f"{name}_calfull", meta, base, cand_from=raw_dir)


def write_out(d, info, iso, kc, u_te, model, name, meta, base, cand_from=None):
    import hashlib
    import os
    from pathlib import Path
    k = pd.Series(info["country"]).map(kc).fillna(1.0).values[d["code"].values]
    p = d["p"].values
    d = d.assign(q=iso.predict(p * k / (p * k + 1 - p)))
    gate = D.gate_test(D.gate_frame(d, info, "test"), range(CFG["n_folds"]))
    mask = D.expf_mask(d, len(info["keys"]), gate)
    out = Path(work("..", "output", "candidates", name, "x")).parent.resolve()
    valid = np.concatenate([to_key(D.read_tsv(D.source_file("test", s), usecols=["entity_id"])["entity_id"]) for s in (2, 3)])
    cf = out / "candidate_pairs.tsv"
    if cand_from:  # same union candidate set: hard link, no second 2 GB copy
        cf.unlink(missing_ok=True)
        os.link(cand_from / "candidate_pairs.tsv", cf)
    else:  # union of both pipelines' final candidates: all our candidates + their pairs our blocking never proposed
        o = pq.read_table(work("preds", f"test_{model}.parquet"), columns=["s1", "cand"])
        only = np.isnan(u_te["po"].values)
        write_id_lists(cf, "candidate_entity_ids", info["ids"], np.r_[o.column("s1").to_numpy(), u_te["s1"].values[only]],
                       np.r_[o.column("cand").to_numpy(), u_te["cand"].values[only]], valid)
        del o
        print(f"candidates: ours all + theirs-only {int(only.sum())} pairs (union)", flush=True)
    write_id_lists(out / "matching_results.tsv", "matched_entity_ids", info["ids"], info["keys"][d["code"].values[mask]],
                   d["cand"].values[mask], valid)
    validate(out / "matching_results.tsv", cf)
    n = np.bincount(d["code"].values[mask], minlength=len(info["keys"]))
    stats = {"md5": hashlib.md5(open(out / "matching_results.tsv", "rb").read()).hexdigest(), "model": model, **meta}
    bf = path("output_dir") / "candidates" / base / "matching_results.tsv"
    if bf.exists():
        b = D.read_tsv(bf).set_index("source1_entity_id").iloc[:, 0]
        a = D.read_tsv(out / "matching_results.tsv").set_index("source1_entity_id").iloc[:, 0].reindex(b.index)
        srt = lambda x: x.map(lambda v: ",".join(sorted(v.split(","))) if v else "")
        diff = pd.Series((srt(a) != srt(b)).values, index=b.index)
        ctry = pd.Series(info["country"], index=info["ids"]).reindex(b.index)
    for c in pd.unique(info["country"]):
        m = info["country"] == c
        stats[c] = {"S1": int(m.sum()), "k": kc.get(c, 1.0), "empty_share": round(float((n[m] == 0).mean()), 4),
                    "links_per_S1": round(float(n[m].mean()), 4)}
        if bf.exists():
            stats[c][f"rows_changed_vs_{base}"] = int(diff[ctry == c].sum())
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    return out


# singles vs the chosen blend on the live-S1 OOF, with precision (the blend must beat both everywhere)
def check(model, w, mode):
    u, live = union("train", model), live_mask("train")
    res = {}
    for ww, mm in ((1.0, "shrunk"), (0.0, "shrunk"), (w, mode)):
        D.NAME = f"blend_{ww}_{mm}"
        d, info = owned(u, mix(u, ww, mm), "train", live)
        res[f"{ww}_{mm}"] = score_variant(d, info)
        print(f"w={ww} {mm}: {res[f'{ww}_{mm}']}", flush=True)
    json.dump(res, open(work("reports", "blend_check.json"), "w"), indent=1)


# decode-only variants of the written blend: odds k for countries absent from train (derived from data, no names)
def unseen_variants(model="lgbm2_ce", w=0.5, mode="asis", ks=(0.5, 2.0, 3.0), name="blend", base="ce_full_v1"):
    D.NAME = f"blend_{w}_{mode}" + (f"_{model}" if model != "lgbm2_ce" else "")
    u_te = union("test", model)
    d, info = owned(u_te, mix(u_te, w, mode), "test")
    iso = pickle.load(open(work("models", f"iso_{D.NAME}.pkl"), "rb"))
    unseen = sorted(set(info["country"]) - set(D.s1_table("train")[1]))
    raw = path("output_dir") / "candidates" / f"{name}_raw"
    for k in ks:
        write_out(d, info, iso, {c: k for c in unseen}, u_te, model, f"{name}_unseen_k{k:g}", {"w": w, "mode": mode, "unseen_k": k},
                  base, cand_from=raw)


# OOF (live S1) curve of one odds multiplier on seen-country p at decode time: isotonic + gate fitted on unscaled p
# (as on test), k applied before isotonic. Precision ~0.9989, so the loss is recall; k > 1 trades some back.
def seen_k_curve(model="lgbm2_ce", w=0.5, mode="asis", ks=(1.0, 1.25, 1.5, 2.0, 3.0)):
    import lightgbm as lgb
    D.NAME = f"blend_{w}_{mode}_kcurve"
    u, live = union("train", model), live_mask("train")
    d, info = owned(u, mix(u, w, mode), "train", live)
    del u
    fold, p, n = info["fold"], d["p"].values, len(info["keys"])
    f = fold[d["code"].values]
    isos = {k: D.iso().fit(p[f != k], d["label"].values[f != k]) for k in np.unique(f)}
    iso_q = lambda pp: sum(np.where(f == k, isos[k].predict(pp), 0.0) for k in isos)  # each fold's own isotonic
    d["q"] = iso_q(p)
    D.fit_gate(D.gate_frame(d, info, "train"), (info["G"] > 0).astype(int), fold)  # saves fold gate models
    gms = {k: lgb.Booster(model_file=str(work("models", f"gate_{D.NAME}_fold{k}.txt"))) for k in np.unique(fold)}
    res = {}
    for kk in ks:
        dk = d.assign(q=iso_q(p * kk / (p * kk + 1 - p)))
        g = D.gate_frame(dk, info, "train")
        gate = np.zeros(n)
        for k, m in gms.items():
            gate[fold == k] = m.predict(g.loc[fold == k, D.GATE_FEATS])
        kc, t, fs = D.score(dk, info, D.expf_mask(dk, n, gate))
        res[kk] = {**D.summary(fs, info), "precision": round(float(t.sum() / max(kc.sum(), 1)), 5), "links": int(kc.sum())}
        print(f"seen k={kk}: {res[kk]}", flush=True)
    json.dump(res, open(work("reports", "blend_seen_k_curve.json"), "w"), indent=1)
    return res


# pipeline stage (run_all.sh): if blend.enabled, build the blend + calibration from this run's round-2 preds and
# make the chosen file the run's output (clean-room clones only; the main repo never runs run_all)
def run(force=False):
    import shutil
    if not BL.get("enabled"):
        return
    model, w, mode = CFG["round2"]["name"], BL.get("w", 0.5), BL.get("mode", "asis")
    final(model, w, mode, "blend")
    for kk in BL.get("extra_k", []):  # extra France files for the A/B (exact values; each validated with md5)
        unseen_variants(model, w, mode, ks=(float(kk),))
    k = BL.get("unseen_k", "auto")  # "auto" = empty-rate matching; a number = fixed odds k for countries absent from train
    if k == "auto":
        src = path("output_dir") / "candidates" / "blend_cal"
    else:
        src = path("output_dir") / "candidates" / f"blend_unseen_k{float(k):g}"
        if not (src / "matching_results.tsv").exists():
            unseen_variants(model, w, mode, ks=(float(k),))
    import hashlib
    tgt = path("output_dir") / "matching_results.tsv"
    if tgt.exists() and hashlib.md5(open(tgt, "rb").read()).hexdigest() == "e7dbc8357c3d2035228854ddf8497556":
        raise SystemExit("refusing to overwrite the protected fallback output (main repo): use a clean-room clone")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copyfile(src / f, path("output_dir") / f)
    validate((path("output_dir") / "matching_results.tsv").resolve(), (path("output_dir") / "candidate_pairs.tsv").resolve())
    print(f"blend stage: final output = {src.name}")


# OOF of the w-blend of several models on the SAME live S1 set (optionally one half only)
def compare(models, w=0.5, mode="asis", half=None):
    from ber.ce_export import halves
    live = live_mask("train")
    if half:
        live &= halves(D.s1_table("train")[0]) == half
    out = {}
    for m in models:
        u = union("train", m)
        D.NAME = f"cmp_{m}"
        d, info = owned(u, mix(u, w, mode), "train", live)
        out[m] = score_variant(d, info)
        print(f"blend OOF {m} (half {half}): {out[m]}", flush=True)
    return out
