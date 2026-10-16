import numpy as np
import pandas as pd


# per-entity F0.5 from counts: G true, k predicted, t correct
def f05(G, k, t):
    G, k, t = (np.asarray(x, dtype=float) for x in (G, k, t))
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.where(t > 0, 1.25 * t / (k + 0.25 * G), 0.0)
    return np.where(G == 0, (k == 0).astype(float), pos)


# per-S1 score; truth/pred are long frames (s1_id, cand_id); s1_ids = all entities
def per_entity(s1_ids, truth, pred):
    idx = pd.Index(s1_ids, name="s1_id")
    G = truth.groupby("s1_id").size().reindex(idx, fill_value=0)
    k = pred.groupby("s1_id").size().reindex(idx, fill_value=0)
    hit = pred.merge(truth[["s1_id", "cand_id"]], on=["s1_id", "cand_id"])
    t = hit.groupby("s1_id").size().reindex(idx, fill_value=0)
    return pd.DataFrame({"G": G, "k": k, "t": t, "f": f05(G, k, t)})


def macro(s1_ids, truth, pred):
    return per_entity(s1_ids, truth, pred)["f"].mean()
