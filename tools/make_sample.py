# clean-room test data: ~3% of S1 per country (fixed seed) + records whose top candidate is one of them
# (+ train: their true matches), raw lines copied byte for byte. Also stand-in CE score files for those pairs (random ce_p:
# tests the CE plumbing only). Reads the main repo's dataset + work/ caches; writes only under <dest_root>.
#   .venv/bin/python tools/make_sample.py <dest student_resource/> [frac]
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SR = Path(__file__).resolve().parents[1] / "student_resource"
sys.path.insert(0, str(SR / "code/business_entity_resolution/src"))
from ber.io import to_id, to_key  # noqa: E402

DS, WORK = SR / "dataset", SR / "work"


# copy header + lines whose first field (entity id) is in keep
def copy_lines(src, dst, keep):
    n = 0
    with open(src, encoding="utf-8", newline="") as f, open(dst, "w", encoding="utf-8", newline="") as g:
        g.write(f.readline())
        for line in f:
            if line.split("\t", 1)[0] in keep:
                g.write(line)
                n += 1
    return n


# records whose top-1 S1 (either blocking channel) is a sampled S1: keeps the sample small (~6% of records)
def cands_of(split, s1_keys):
    t = pq.read_table(WORK / split / "cands.parquet", columns=["s1", "cand", "fw_rank", "hh_rank"],
                      filters=pc.is_in(pc.field("s1"), value_set=pa.array(s1_keys)))
    top = pc.or_kleene(pc.equal(t["fw_rank"], 0), pc.equal(t["hh_rank"], 0)).fill_null(False)
    return t.filter(top).column("cand").to_numpy()


def main(dest, frac=0.03):
    rng = np.random.default_rng(42)
    for split in ("train", "test"):
        out = Path(dest) / "dataset" / split
        out.mkdir(parents=True, exist_ok=True)
        s1 = pd.read_csv(DS / split / f"{split}_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                         quoting=3, usecols=["entity_id", "country"])
        pick = np.concatenate([rng.choice(g.values, max(1, round(len(g) * frac)), replace=False)
                               for _, g in s1.groupby("country")["entity_id"]])
        keep1 = set(pick)
        rec = set(to_id(np.unique(cands_of(split, to_key(pick)))))
        if split == "train":
            gt = pd.read_csv(DS / split / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
            gt = gt[gt.source1_entity_id.isin(keep1)]
            true_ids = {x for l in gt.matched_entity_ids for x in l.split(",") if x}
            copy_lines(DS / split / "train_ground_truth.tsv", out / "train_ground_truth.tsv", keep1)
        else:
            true_ids = set()
        n1 = copy_lines(DS / split / f"{split}_source1.tsv", out / f"{split}_source1.tsv", keep1)
        for s in (2, 3):
            n = copy_lines(DS / split / f"{split}_source{s}.tsv", out / f"{split}_source{s}.tsv", rec | true_ids)
            print(f"{split} S{s}: {n} records")
        print(f"{split} S1: {n1} ({s1.groupby('country').size().to_dict()} full)")
        # stand-in CE scores: CE band pairs of the sampled S1, random ce_p
        band = pd.read_parquet(WORK / "ce" / f"ce_band_{split}.parquet", columns=["s1_id", "cand_id"])
        band = band[band.s1_id.isin(keep1)]
        ce = Path(dest) / "code/business_entity_resolution/src/ce_scores"
        ce.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"s1": to_key(band.s1_id), "cand": to_key(band.cand_id),
                      "ce_p": rng.random(len(band)).astype(np.float32)}) \
            .to_parquet(ce / f"ce_scores_{split}.parquet", index=False)
        print(f"{split} stand-in CE scores: {len(band)} pairs")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.03)
