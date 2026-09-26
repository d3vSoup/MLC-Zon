import json

import numpy as np

import ber.decode as D
from ber import loco
from ber.calibrate import subset
from ber.io import CFG, work
from ber.metric import f05

# decode rules for countries absent from train, tested in LOCO (the held-out country's labels only score them):
#   none   k = 1
#   empty  one odds k on q (gate recomputed from scaled q): unseen empty share = seen empty share + DELTA
#   link   one odds k on q: unseen predicted links per S1 = seen predicted links per S1 (generator's true links/S1
#          is the same in every country: train, and India test = train per the QTpies audit)
#   two    gate odds k_gate (entity level) from the empty signal + q odds k_link (links within entities) from the link
#          signal, solved jointly. The two label-free signals disagree on France, so each gets its own knob.
DELTA = 0.0015


def odds(x, k):
    return x * k / (x * k + 1 - x)


def dec(d, info, gm, k_link=1.0, k_gate=1.0, joint=True):
    q = d["q"].values
    dq = d.assign(q=odds(q, k_link))
    g = D.gate_frame(dq if joint else d, info, "train")
    gate = odds(np.mean([m.predict(g[D.GATE_FEATS]) for m in gm], axis=0), k_gate)
    m = D.expf_mask(dq, len(info["keys"]), gate)
    return dq, m


def stats(d, info, m):
    n = np.bincount(d["code"].values[m], minlength=len(info["keys"]))
    return float((n == 0).mean()), float(n.mean())


def solve(f, target, lo=np.log(0.02), hi=np.log(20), rising=True, it=14):
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (f(np.exp(mid)) < target) == rising else (lo, mid)
    return float(np.exp((lo + hi) / 2))


def score(d, info, m):
    k, t, f = D.score(d, info, m)
    G = info["G"]
    return {"F0.5": round(float(f.mean()), 5), "singleton_only": round(float(f[G == 0].mean()), 5),
            "precision": round(float(t.sum() / max(k.sum(), 1)), 5), "empty": round(float((k == 0).mean()), 4),
            "links_per_S1": round(float(k.mean()), 4)}


def loco_rules(direction, extra="num_names2"):
    src = {"us2in": "US", "in2us": "India"}[direction]
    d, info, gate, is_src = loco.prepare(f"loco_{direction}_{extra}", src)
    gm = loco.prepare.gate_models
    m_all = D.expf_mask(d, len(info["keys"]), gate)
    ds, is_ = subset(d, info, is_src)
    e_src, l_src = stats(ds, is_, m_all[is_src[d["code"].values]])
    g_src, g_tgt = float(info["G"][is_src].mean()), float(info["G"][~is_src].mean())
    du, iu = subset(d, info, ~is_src)
    print(f"{direction}: seen empty {e_src:.4f}, seen links/S1 {l_src:.4f}; true links/S1 seen {g_src:.3f} unseen {g_tgt:.3f}", flush=True)
    res = {"seen_empty": e_src, "seen_links": l_src, "true_links_seen": g_src, "true_links_unseen": g_tgt}
    emp = lambda k: stats(du, iu, dec(du, iu, gm, k)[1])[0]
    lnk = lambda k: stats(du, iu, dec(du, iu, gm, k)[1])[1]
    ks = {"none": (1.0, 1.0, True)}
    ks["empty"] = (solve(emp, e_src + DELTA, rising=False), 1.0, True)
    ks["link"] = (solve(lnk, l_src, rising=True), 1.0, True)
    kl, kg = 1.0, 1.0
    for _ in range(3):  # two knobs: alternate (gate sets the empties, link odds set the link count)
        kg = solve(lambda k: stats(du, iu, dec(du, iu, gm, kl, k, False)[1])[0], e_src + DELTA, rising=False)
        kl = solve(lambda k: stats(du, iu, dec(du, iu, gm, k, kg, False)[1])[1], l_src, rising=True)
    ks["two"] = (kl, kg, False)
    for name, (kl_, kg_, joint) in ks.items():
        dq, m = dec(du, iu, gm, kl_, kg_, joint)
        res[name] = {"k_link": round(kl_, 3), "k_gate": round(kg_, 3), **score(dq, iu, m)}
        print(f"  {name}: {res[name]}", flush=True)
    for k in (0.25, 0.5, 2.0, 4.0):  # reference only (labels): single joint k grid
        dq, m = dec(du, iu, gm, k)
        res[f"grid_k{k}"] = {"k_link": k, **score(dq, iu, m)}
        print(f"  grid k={k}: {res[f'grid_k{k}']}", flush=True)
    json.dump(res, open(work("reports", f"france_rules_{direction}_{extra}.json"), "w"), indent=1)
    return res
