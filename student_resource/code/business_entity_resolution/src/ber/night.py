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


def ce2_half():
    import ber.ce_ingest as C
    from ber.models import stack
    dec = json.load(open(work("reports", "night_names2c.json")))
    ex = tuple(dec["extra"])
    C.collect([CE2_DIR], "ce2")
    CFG["round2"].update(use_ce=True, ce_models=["ce2"])
    C.run("train", force=True)
    C.run("test", force=True)
    base = "lgbm2_ceh" if ex[1] == "r2_names2" else "lgbm2_cech"
    fits = ([] if base == "lgbm2_ceh" else [(base, ["ce"])]) + [("lgbm2_ce2h", ["ce", "ce2"])]
    for name, tags in fits:
        CFG["round2"].update(ce_models=tags, ce_mix="sep")
        stack.fit(name=name, rounds=3000, test=True, use_ce=True, extra=ex, half=2)
        CFG["decode"]["half"] = 2
        D.NAME = name
        D.run()
    g = oof("lgbm2_ce2h")["overall"] - oof(base)["overall"]
    ok = g >= 0.0003
    dec2 = {"pass": ok, "model": "lgbm2_ce2h" if ok else dec["full"], "half": 2 if ok else None}
    json.dump(dec2, open(work("reports", "night_ce2.json"), "w"))
    log(f"- CE2 half gate: {'PASS' if ok else 'FAIL'} | H2 OOF {base} {oof(base)['overall']:.5f} vs lgbm2_ce2h "
        f"{oof('lgbm2_ce2h')['overall']:.5f} ({g:+.5f}; US {oof('lgbm2_ce2h')['US'] - oof(base)['US']:+.5f}, India "
        f"{oof('lgbm2_ce2h')['India'] - oof(base)['India']:+.5f}) | blend model {dec2['model']}")


def reblend():
    dec2 = json.load(open(work("reports", "night_ce2.json")))
    m, half = dec2["model"], dec2["half"]
    B.final(m, 0.5, "asis", "blend2", half=half)
    B.unseen_variants(m, 0.5, "asis", ks=(0.184,), name="blend2")
    cmp = B.compare(["lgbm2_ce", m], half=2)  # same row set: H2 live S1 (the CE2 half)
    base = ROOT / "output" / "candidates"
    log(f"- blend2 (model {m}, w 0.5): OOF on H2 live S1 vs blend_cal's model: {cmp['lgbm2_ce']['overall']:.5f} -> "
        f"{cmp[m]['overall']:.5f} ({cmp[m]['overall'] - cmp['lgbm2_ce']['overall']:+.5f})")
    for n in ("blend2_raw", "blend2_cal", "blend2_unseen_k0.184"):
        f = base / n / "matching_results.tsv"
        st = json.load(open(base / n / "stats.json"))
        fr = {c: v for c, v in st.items() if isinstance(v, dict) and "k" in v}
        log(f"  - {n}: md5 {hashlib.md5(open(f, 'rb').read()).hexdigest()} | " +
            "; ".join(f"{c} k {v['k']:.3g} empty {v['empty_share']} links/S1 {v['links_per_S1']}" for c, v in fr.items()))
