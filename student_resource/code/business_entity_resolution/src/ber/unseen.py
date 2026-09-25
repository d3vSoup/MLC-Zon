import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import ber.decode as D
from ber.io import work
from ber.loco import prepare
from ber.metric import f05

SIMS = {"us2in": "US", "in2us": "India"}
EPS = 1e-6


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


# one LOCO simulation -> cached arrays of the unseen (target) side
def load(tag):
    f = work("loco", tag, "unseen.npz")
    if not f.exists():
        d, info, gate, is_src = prepare(f"loco_{tag}", SIMS[tag])
        np.savez(f, code=d["code"].values, rank=d["rank"].values, q=d["q"].values, label=d["label"].values,
                 cand=d["cand"].values, gate=gate, G=info["G"], is_src=is_src)
    z = dict(np.load(f))
    return z


class Temper:  # q' = sigmoid(a * logit(q) + b), for pairs and for the gate
    def fit(self, x, y):
        self.m = LogisticRegression(C=1e6).fit(logit(x)[:, None], y)
        return self

    def __call__(self, x):
        return self.m.predict_proba(logit(x)[:, None])[:, 1]

    def params(self):
        return [round(float(self.m.coef_[0, 0]), 4), round(float(self.m.intercept_[0]), 4)]


class Iso:
    def fit(self, x, y):
        self.m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)
        return self

    def __call__(self, x):
        return self.m.predict(x)

    def params(self):
        return "isotonic"


# F0.5 on the unseen side with optional corrections of q (pairs) and gate (entities)
def score(z, fq=None, fg=None):
    tgt_rows = ~z["is_src"][z["code"]]
    d = pd.DataFrame({"code": z["code"][tgt_rows], "rank": z["rank"][tgt_rows], "q": z["q"][tgt_rows],
                      "label": z["label"][tgt_rows], "cand": z["cand"][tgt_rows]})
    if fq is not None:
        d["q"] = fq(d["q"].values)
    gate = z["gate"].copy()
    tg = ~z["is_src"]
    if fg is not None:
        gate[tg] = fg(gate[tg])
    n = len(z["G"])
    mask = D.expf_mask(d, n, gate)
    k = np.bincount(d["code"].values[mask], minlength=n)
    t = np.bincount(d["code"].values[mask & (d["label"].values == 1)], minlength=n)
    f = f05(z["G"], k, t)[tg]
    return {"F0.5": round(float(f.mean()), 5), "singleton_only": round(float(f[z["G"][tg] == 0].mean()), 5),
            "precision": round(float(d["label"].values[mask].mean()), 4), "pred_pairs": int(mask.sum())}


# fit on one simulation's unseen side, evaluate on the other's (both ways) + pooled params for France
def run(force=False):
    Z = {t: load(t) for t in SIMS}
    out = {}
    for fit_tag, eval_tag in (("us2in", "in2us"), ("in2us", "us2in")):
        zf, ze = Z[fit_tag], Z[eval_tag]
        tr = ~zf["is_src"][zf["code"]]
        tg = ~zf["is_src"]
        res = {"none": score(ze)}
        for name, C in (("temper", Temper), ("isotonic", Iso)):
            fq = C().fit(zf["q"][tr], zf["label"][tr])
            fg = C().fit(zf["gate"][tg], (zf["G"][tg] > 0).astype(int))
            res[name] = {**score(ze, fq, fg), "q_params": fq.params(), "gate_params": fg.params()}
            res[name + "_q_only"] = score(ze, fq, None)
        out[f"fit {fit_tag} -> eval {eval_tag}"] = res
    # pooled fit (both simulations' unseen sides): what would be applied to countries not in training
    q = np.concatenate([Z[t]["q"][~Z[t]["is_src"][Z[t]["code"]]] for t in SIMS])
    y = np.concatenate([Z[t]["label"][~Z[t]["is_src"][Z[t]["code"]]] for t in SIMS])
    g = np.concatenate([Z[t]["gate"][~Z[t]["is_src"]] for t in SIMS])
    gy = np.concatenate([(Z[t]["G"][~Z[t]["is_src"]] > 0).astype(int) for t in SIMS])
    out["pooled_temper_params"] = {"q": Temper().fit(q, y).params(), "gate": Temper().fit(g, gy).params()}
    json.dump(out, open(work("reports", "unseen_correction.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
