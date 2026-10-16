import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.io import CFG, read_tsv, source_file, to_id, to_key, work
from ber.models.lgbm import fold_map

CE = CFG["ce"]


# raw "name ; address" per entity key, for one split
def texts(split):
    t = pd.concat([pd.read_parquet(work(split, f"norm_s{s}.parquet"), columns=["entity_id", "name_raw", "addr_raw"])
                   for s in (1, 2, 3)], ignore_index=True)
    return pd.Series((t["name_raw"] + " ; " + t["addr_raw"]).values, index=to_key(t["entity_id"]))


# rank of each pair inside its S2/S3 record by p (0 = best)
def rank_rec(cand, p):
    o = np.lexsort((-p, cand))
    r = np.empty(len(p), np.int32)
    r[o] = np.arange(len(p)) - np.searchsorted(cand[o], cand[o], "left")
    return r


# band = pairs the cross-encoder re-scores: unsure p, or a record's top pairs that are not hopeless
def band_mask(p, r):
    mid = (p > CE["band_lo"]) & (p < CE["band_hi"])
    return mid | ((r < CE["band_rank"]) & (p >= CE["band_floor"]))


# half of train S1: H1 = folds 0,1 + even keys of fold 2; H2 = the rest (fixed, from the fold file)
def halves(s1):
    fm = fold_map()
    f = fm["fold"].reindex(s1).values
    return np.where((f < 2) | ((f == 2) & (s1 % 2 == 0)), 1, 2).astype(np.int8)


def frame(split, rows, s1, cand, tx, extra):
    d = pd.DataFrame({"pair_id": rows.astype(np.int64), "s1_id": to_id(s1), "cand_id": to_id(cand),
                      "text_a": tx.reindex(s1).values, "text_b": tx.reindex(cand).values, **extra})
    assert d.text_a.notna().all() and d.text_b.notna().all()
    return d


def write(d, name):
    out = work("ce", name)
    pq.write_table(pa.Table.from_pandas(d, preserve_index=False), out, compression="zstd", row_group_size=500_000)
    return out


# export cross-encoder train pairs (per half) + band pairs to score (train + test)
def run(force=False):
    rng = np.random.default_rng(CFG["seed"])
    for split, tag in (("train", "oof"), ("test", "test")):
        pr = pq.read_table(work("preds", f"{tag}_{CE['from_model']}.parquet")).to_pandas()
        s1, cand, p = pr["s1"].values, pr["cand"].values, pr["p"].values
        del pr
        r = rank_rec(cand, p)
        band = np.flatnonzero(band_mask(p, r))
        tx = texts(split)
        ctry = read_tsv(source_file(split, 1), usecols=["entity_id", "country"])
        ctry = pd.Series(ctry["country"].values, index=to_key(ctry["entity_id"]))
        extra = {}
        if split == "train":
            lab = pq.read_table(work("train", "cands.parquet"), columns=["label"]).column("label").to_numpy()
            h = halves(s1)
            extra = {"label": lab[band], "half": h[band]}
            # training pairs per half: hard + random positives, hardest + chain-name + easy negatives
            chain = pq.read_table(work("train", "feats.parquet"), columns=["f_rec_name_in_s1_share"]) \
                .column(0).to_numpy() * len(ctry) >= 2
            picks = []
            for half in (1, 2):
                inh = h == half
                pos, neg = np.flatnonzero(inh & (lab == 1)), np.flatnonzero(inh & (lab == 0))
                n_pos, n_neg = int(CE["cap"] * 0.4), int(CE["cap"] * 0.6)
                hard_pos = pos[np.argsort(p[pos])[: n_pos // 2]]
                rand_pos = rng.choice(np.setdiff1d(pos, hard_pos), n_pos - len(hard_pos), replace=False)
                hard_neg = neg[np.argsort(-p[neg])[: n_neg // 2]]
                rest = np.setdiff1d(neg, hard_neg)
                ch = rest[chain[rest]]
                chain_neg = ch[np.argsort(-p[ch])[: n_neg // 4]]
                easy = np.setdiff1d(rest, chain_neg)
                easy = easy[r[easy] < CE["band_rank"]]
                easy_neg = rng.choice(easy, n_neg - len(hard_neg) - len(chain_neg), replace=False)
                picks.append(np.sort(np.concatenate([hard_pos, rand_pos, hard_neg, chain_neg, easy_neg])))
            tr = np.concatenate(picks)
            d = frame(split, tr, s1[tr], cand[tr], tx, {"label": lab[tr], "half": h[tr]})
            print(f"ce train pairs: {len(d)} (H1 {int((d.half == 1).sum())}, H2 {int((d.half == 2).sum())}), "
                  f"pos rate {d.label.mean():.3f}, file {write(d, 'ce_train_pairs.parquet').stat().st_size / 1e6:.0f} MB")
        d = frame(split, band, s1[band], cand[band], tx, extra)
        d["country"] = ctry.reindex(s1[band]).values
        out = write(d.drop(columns="country"), f"ce_band_{split}.parquet")
        print(f"ce band {split}: {len(d)} pairs, " + ", ".join(f"{c} {n}" for c, n in d["country"].value_counts().items())
              + f", file {out.stat().st_size / 1e6:.0f} MB")
