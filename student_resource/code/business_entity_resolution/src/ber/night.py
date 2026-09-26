import hashlib
import json

import ber.blend as B
import ber.decode as D
from ber.io import CFG, ROOT, work

# scripted night chain (runs without a live session): names2c gate -> CE2 half-ingest gate -> re-blend + France files.
# each step reads the previous decision json and appends pass/fail + md5s to docs/NIGHT_RESULTS.md
RES = ROOT.parent / "docs" / "NIGHT_RESULTS.md"
CE2_DIR = "/home/Muaz/Downloads/ce2_ankit"


def log(line):
    with open(RES, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def oof(model):
    j = json.load(open(work("reports", f"decode_{model}.json")))
    return j["decoders"][j["chosen"]]["oof"]


def gate_names2c():
    a, b = oof("lgbm2_ce"), oof("lgbm2_cec")
    l1 = json.load(open(work("reports", "loco_us2in_num_names2c.json")))["in (unseen)"]["F0.5"]
    l2 = json.load(open(work("reports", "loco_in2us_num_names2c.json")))["us (unseen)"]["F0.5"]
    d_us, d_in = b["US"] - a["US"], b["India"] - a["India"]
    ok = d_us >= -0.0002 and d_in >= -0.0002 and l1 > 0.91571 and l2 > 0.96935
    dec = {"pass": ok, "extra": ["r2_num", "r2_names2c" if ok else "r2_names2"], "full": "lgbm2_cec" if ok else "lgbm2_ce"}
    json.dump(dec, open(work("reports", "night_names2c.json"), "w"))
    log(f"- names2c gate: {'PASS' if ok else 'FAIL'} | OOF US {a['US']:.5f}->{b['US']:.5f} ({d_us:+.5f}), India {a['India']:.5f}->"
        f"{b['India']:.5f} ({d_in:+.5f}), overall {a['overall']:.5f}->{b['overall']:.5f} | LOCO unseen US->IN 0.91571->{l1:.5f}, "
        f"IN->US 0.96935->{l2:.5f} | feature set {dec['extra'][1]}")


SOUP_DIR, SOUP_MD5 = "/home/Muaz/Downloads/ce2_soup", "/home/Muaz/Downloads/ce2_md5_soup.txt"


# integrity of one CE2 half: 79 md5s vs the list, 27 train + 52 test parts, rows = band (train: the other half), unique,
# no NaN, 100% key join; plus the signal vs CE1 on the scored half (labels)
def integrity(d, md5file, half_scored, ce1_dir):
    import glob
    import hashlib
    import os
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    fs = sorted(glob.glob(f"{d}/**/*_part*.parquet", recursive=True))
    if not os.path.exists(md5file) or len(fs) != 79:
        log(f"- CE2 integrity {d}: files {len(fs)}, md5 list {'found' if os.path.exists(md5file) else 'MISSING'} -> FAIL")
        return False
    want = {l.split()[1].split("/")[-1]: l.split()[0] for l in open(md5file) if l.strip()}
    got = {os.path.basename(f): hashlib.md5(open(f, "rb").read()).hexdigest() for f in fs}
    tr = pd.concat([pd.read_parquet(f) for f in fs if "/train_" in f or os.path.basename(f).startswith("train_")])
    te = pd.concat([pd.read_parquet(f) for f in fs if os.path.basename(f).startswith("test_")])
    bt = pd.read_parquet(work("ce", "ce_band_train.parquet"), columns=["pair_id", "label", "half"])
    be = pd.read_parquet(work("ce", "ce_band_test.parquet"), columns=["pair_id"])
    band = bt[bt.half == half_scored]
    ok = (want == got and tr.pair_id.is_unique and te.pair_id.is_unique and not tr.ce_p.isna().any() and not te.ce_p.isna().any()
          and set(tr.pair_id) == set(band.pair_id) and set(te.pair_id) == set(be.pair_id))
    c1 = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{ce1_dir}/**/train_h*_part*.parquet", recursive=True)]).rename(columns={"ce_p": "ce1"})
    m = band.merge(c1, on="pair_id").merge(tr, on="pair_id")
    log(f"- CE2 integrity {d}: {'PASS' if ok else 'FAIL'} | md5 {sum(want.get(k) == v for k, v in got.items())}/79, rows train {len(tr)} "
        f"(band H{half_scored} {len(band)}), test {len(te)} | signal on H{half_scored}: AUC CE1 {roc_auc_score(m.label, m.ce1):.5f} "
        f"vs CE2 {roc_auc_score(m.label, m.ce_p):.5f}, corr {np.corrcoef(m.ce1, m.ce_p)[0, 1]:.4f}")
    return ok


# CE2 ingest: FULL (both halves; test = mean of both models, as ce_full_v1) if Soup's half passes integrity, else H2-only
def ce2_half():
    import ber.ce_ingest as C
    from ber.models import stack
    dec = json.load(open(work("reports", "night_names2c.json")))
    ex = tuple(dec["extra"])
    full = integrity(SOUP_DIR, SOUP_MD5, 1, "/home/Muaz/Downloads/ce_scores_soup")
    C.collect([CE2_DIR, SOUP_DIR] if full else [CE2_DIR], "ce2")
    CFG["round2"].update(use_ce=True, ce_models=["ce2"])
    C.run("train", force=True)
    C.run("test", force=True)
    if full:  # baseline = the full no-CE2 model of the passing feature set (same rows: all S1)
        base, name, half = dec["full"], "lgbm2_ce2f", None
        fits = [(name, ["ce", "ce2"])]
    else:
        base, name, half = ("lgbm2_ceh" if ex[1] == "r2_names2" else "lgbm2_cech"), "lgbm2_ce2h", 2
        fits = ([] if base == "lgbm2_ceh" else [(base, ["ce"])]) + [(name, ["ce", "ce2"])]
    for n, tags in fits:
        CFG["round2"].update(ce_models=tags, ce_mix="sep")
        stack.fit(name=n, rounds=3000, test=(n == name), use_ce=True, extra=ex, half=half)
        CFG["decode"]["half"] = half
        D.NAME = n
        D.run()
    g = oof(name)["overall"] - oof(base)["overall"]
    ok = g >= 0.0003
    dec2 = {"pass": ok, "model": name if ok else dec["full"], "half": half if ok else None, "mode": "full" if full else "half"}
    json.dump(dec2, open(work("reports", "night_ce2.json"), "w"))
    log(f"- CE2 {dec2['mode']} gate: {'PASS' if ok else 'FAIL'} | OOF ({'all S1' if full else 'H2 S1'}) {base} {oof(base)['overall']:.5f} vs "
        f"{name} {oof(name)['overall']:.5f} ({g:+.5f}; US {oof(name)['US'] - oof(base)['US']:+.5f}, India "
        f"{oof(name)['India'] - oof(base)['India']:+.5f}) | blend model {dec2['model']}")


def reblend():
    dec2 = json.load(open(work("reports", "night_ce2.json")))
    m, half = dec2["model"], dec2["half"]
    B.final(m, 0.5, "asis", "blend2", half=half)
    B.unseen_variants(m, 0.5, "asis", ks=(0.184,), name="blend2")
    cmp = B.compare(["lgbm2_ce", m], half=half)  # same row set: live S1 (H2 only if the model is half-trained)
    base = ROOT / "output" / "candidates"
    log(f"- blend2 (model {m}, w 0.5): OOF on {'H2 ' if half else ''}live S1 vs blend_cal's model: {cmp['lgbm2_ce']['overall']:.5f} -> "
        f"{cmp[m]['overall']:.5f} ({cmp[m]['overall'] - cmp['lgbm2_ce']['overall']:+.5f})")
    for n in ("blend2_raw", "blend2_cal", "blend2_unseen_k0.184"):
        f = base / n / "matching_results.tsv"
        st = json.load(open(base / n / "stats.json"))
        fr = {c: v for c, v in st.items() if isinstance(v, dict) and "k" in v}
        log(f"  - {n}: md5 {hashlib.md5(open(f, 'rb').read()).hexdigest()} | " +
            "; ".join(f"{c} k {v['k']:.3g} empty {v['empty_share']} links/S1 {v['links_per_S1']}" for c, v in fr.items()))


# Phase A: unseen-country errors vs seen (LOCO held-out preds), error types, shifted features -> docs/LOCO_ERRORS.md
def loco_errors():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from ber import loco
    from ber.twins import attrs
    out = ["# LOCO errors: unseen vs seen (label-scored, same model)\n"]
    imp = pd.read_csv(work("reports", "importance_lgbm2_x.tsv"), sep="\t", index_col=0).iloc[:, 0].sort_values(ascending=False)
    for direction, src in (("us2in", "US"), ("in2us", "India")):
        tag = f"{direction}_num_names2c" if work("preds", f"oof_loco_{direction}_num_names2c.parquet").exists() else f"{direction}_num_names2"
        d, info, gate, is_src = loco.prepare(f"loco_{tag}", src)
        m = D.expf_mask(d, len(info["keys"]), gate)
        code, lab = d["code"].values, d["label"].values
        side = np.where(is_src[code], "seen", "unseen")
        sel = (m & (lab == 0)) | (~m & (lab == 1))
        x = attrs("train", info["keys"][code[sel]], d["cand"].values[sel])
        x["side"], x["err"] = side[sel], np.where(m[sel], "FP", "FN")
        x["q"] = d["q"].values[sel]
        n_s1 = {"seen": int(is_src.sum()), "unseen": int((~is_src).sum())}
        rows = []
        for (sd, er), g in x.groupby(["side", "err"]):
            r = {"side": sd, "error": er, "per_1k_S1": round(1000 * len(g) / n_s1[sd], 2)}
            for c in ("core_eq", "legal_add", "num_shift", "num_same", "gen_swap"):
                r[c] = round(float(g[c].mean()), 3)
            r["q_median"] = round(float(g["q"].median()), 3)
            rows.append(r)
        out.append(f"## {direction} ({tag}): errors per 1k S1 and their type shares\n\n" + "```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```" + "\n")
        # feature shift, hard rows: unseen vs seen, among true pairs and among non-pairs (standardized mean difference)
        r2 = pq.read_table(work("loco", direction, "r2.parquet")).to_pandas()
        ex = [pq.read_table(work("loco", direction, f"{e}.parquet")).to_pandas() for e in tag.split("_")[1:] if work("loco", direction, f"{e}.parquet").exists()]
        r2 = pd.concat([r2] + ex, axis=1)
        s = np.isin(r2["s1"].values, info["keys"][is_src])
        feats = [f for f in imp.index if f in r2.columns][:25]
        sh = []
        for f in feats:
            v = r2[f].astype(float)
            for y in (1, 0):
                a, b = v[s & (r2["label"] == y)], v[~s & (r2["label"] == y)]
                sd_ = np.nanstd(np.r_[a, b]) or 1.0
                sh.append({"feature": f, "label": y, "seen_mean": round(np.nanmean(a), 4), "unseen_mean": round(np.nanmean(b), 4),
                           "smd": round((np.nanmean(b) - np.nanmean(a)) / sd_, 3)})
        sh = pd.DataFrame(sh).sort_values("smd", key=abs, ascending=False).head(20)
        out.append(f"### top shifted high-importance features (hard rows, unseen minus seen, in SD units)\n\n" + "```\n" + sh.to_string(index=False) + "\n```" + "\n")
        del d, r2
    open(ROOT.parent / "docs" / "LOCO_ERRORS.md", "w").write("\n".join(out))
    log("- Phase A: docs/LOCO_ERRORS.md written")


# draft morning brief from the night decisions + candidate stats (a live session refines it)
def brief():
    import os
    base = ROOT / "output" / "candidates"
    rows = []
    for n in ("blend_cal", "blend_raw", "blend_unseen_k0.5", "blend_unseen_k1.5", "blend_unseen_k2", "blend2_raw", "blend2_cal",
              "blend2_unseen_k0.184", "ce_full_cal"):
        f = base / n / "stats.json"
        if f.exists():
            st = json.load(open(f))
            fr = next((v for c, v in st.items() if isinstance(v, dict) and "k" in v and c not in ("US", "India")), {})
            rows.append(f"| {n} | {st['md5']} | {fr.get('k', '-')} | {fr.get('empty_share', '-')} | {fr.get('links_per_S1', '-')} |")
    night = open(RES).read() if RES.exists() else ""
    txt = f"""# Morning brief (auto-draft by the night chain; refine in session)

## 05:30 upload: blend_raw (France k = 1) vs blend_cal (k 0.184, public 0.985131)
- LOCO (India as unseen country, labels): no correction is best; k 0.25 cost -0.006, link-count k 2.3 cost -0.001.
  If France behaves like unseen India, blend_raw wins by roughly +0.001 public.
- QTpies M4 probe: France down-scaling helped (+0.002) on a model that over-linked France. Our blend under-links France
  (assigned ratio 0.963 vs 0.973-0.979), so the LOCO reading is more likely to apply.
- If raw wins: France k 1 stays; no more France points (LOCO: k > 1 loses). If raw loses: try k 0.5 next.

## Night results
{night}
## Candidates (France stats)
| name | md5 | France k | France empty | France links/S1 |
|---|---|---|---|---|
""" + "\n".join(rows) + "\n"
    open(ROOT.parent / "docs" / "MORNING_BRIEF.md", "w").write(txt)
    log("- MORNING_BRIEF.md draft written")
