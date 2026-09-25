import numpy as np
import pandas as pd

from ber.io import CFG, work
from ber.metric import f05

NAME = CFG["decode"]["model"]

BUCKETS = ["singleton_fp", "missed_entity", "wrong_top1", "blocking_miss", "extra_fp", "extra_fn"]


# split each S1's loss (1 - F) into buckets; shares of total loss
def buckets(k, t, G, g_cand):
    k, t, G, g_cand = (np.asarray(x, float) for x in (k, t, G, g_cand))
    f = f05(G, k, t)
    L = pd.DataFrame(0.0, index=range(len(G)), columns=BUCKETS)
    L["singleton_fp"] = np.where((G == 0) & (k > 0), 1.0, 0.0)
    no_hit = (G > 0) & (t == 0)
    unreachable = no_hit & (g_cand == 0)
    L["blocking_miss"] = np.where(unreachable, 1.0, 0.0)
    L["missed_entity"] = np.where(no_hit & ~unreachable & (k == 0), 1.0, 0.0)
    L["wrong_top1"] = np.where(no_hit & ~unreachable & (k > 0), 1.0, 0.0)
    hit = t > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_nb = np.where(hit, 1.25 * t / (k + 0.25 * g_cand), 0.0)  # F if blocking had found all
        fp, fn = k - t, g_cand - t
        rest = 1 - f_nb
        denom = fp + 0.25 * fn
        L["blocking_miss"] += np.where(hit, f_nb - f, 0.0)
        L["extra_fp"] = np.where(hit & (denom > 0), rest * fp / denom, 0.0)
        L["extra_fn"] = np.where(hit & (denom > 0), rest * 0.25 * fn / denom, 0.0)
    assert np.allclose(L.sum(axis=1), 1 - f)
    return L


def run(force=False):
    z = np.load(work("preds", f"oof_decoded_{NAME}.npz"), allow_pickle=True)
    L = buckets(z["k"], z["t"], z["G"], z["g_cand"])
    L["country"] = z["country"]
    tot = L.groupby("country")[BUCKETS].sum()
    tot.loc["overall"] = L[BUCKETS].sum()
    share = (tot.div(tot.sum(axis=1), axis=0) * 100).round(1)
    share["loss_per_S1"] = (tot.sum(axis=1) / pd.concat([L.groupby("country").size(),
                                                          pd.Series({"overall": len(L)})])).round(4)
    print(share.to_string())
    share.to_csv(work("reports", f"loss_{NAME}.tsv"), sep="\t")
    return share
