"""Run pipeline shards as private Kaggle CPU kernels (4 vCPU / 31 GB each), used while AWS quota is pending.

Each kernel embeds the exact src/ snapshot (base64 tarball), pip-installs the pinned requirements, links every
parquet from its inputs (dataset razanml/qtpies-er-norm + optional upstream kernels) into BER_WORK, runs its
commands and exports only newly written parquet files to /kaggle/working (same relative paths).

python kaggle_runner.py push block            # 14 blocking shards
python kaggle_runner.py status block          # status of that batch
python kaggle_runner.py fetch block <dest>    # download outputs into dest (e.g. work2)
"""
import base64
import io
import json
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KDIR = os.path.join(ROOT, "kaggle", "k")
USER, DATASET = "razanml", "razanml/qtpies-er-norm"
REQ = "polars==1.44.2 rapidfuzz==3.14.6 sparse-dot-topn==1.2.0 anyascii==0.3.3 scikit-learn==1.8.0 numba==0.63.1 lightgbm==4.6.0"
BLOCK_SHARDS = {"train": {"India": 3, "US": 4}, "test": {"France": 1, "India": 3, "US": 3}}

BATCHES = {
    "block": {f"qtpies-blk-{sp}-{c.lower()}-{i}of{n}": ([["block.py", sp, c, str(i), str(n)]], [])
              for sp, cs in BLOCK_SHARDS.items() for c, n in cs.items() for i in range(n)},
    # Kaggle allows 5 concurrent batch CPU sessions; GPU sessions (4 vCPU too) count separately
    "block_gpu": {f"qtpies-blk-train-us-{i}of4": ([["block.py", "train", "US", str(i), "4"]], []) for i in (2, 3)},
}
TRAIN_BLOCK = [s for s in BATCHES["block"] if "-train-" in s]
BATCHES["block_test_us"] = {k: v for k, v in BATCHES["block"].items() if "-test-us-" in k}
BATCHES["feats_train"] = {f"qtpies-ft-train-{i}of5": ([["feats.py", "train", str(i), "5"]], TRAIN_BLOCK) for i in range(5)}
BATCHES["feats_test"] = {f"qtpies-ft-test-{i}of4": ([["feats.py", "test", str(i), "4"]], list(BATCHES["block_test_us"]))
                         for i in range(4)}
BATCHES["model"] = {"qtpies-model": ([["run.py", "train"], ["run.py", "train2"], ["run.py", "predict"]],
                                     list(BATCHES["feats_train"]) + list(BATCHES["feats_test"]))}
BATCHES["ce_train"] = {f"qtpies-ce-train-{k}": ([["ce.py", "train", str(k)]], TRAIN_BLOCK) for k in (0, 1)}
BATCHES["ce_score"] = {"qtpies-ce-score": ([["ce.py", "score", "train", "oof_m.parquet", "ce_train.parquet", "ce_fold0", "ce_fold1"],
                                             ["ce.py", "score", "test", "pred_test_m.parquet", "ce_test.parquet", "ce_fold0", "ce_fold1"]],
                                            list(BATCHES["ce_train"]) + ["qtpies-model"])}
BATCHES["model2"] = {"qtpies-model2": ([["run.py", "train2ce"], ["run.py", "predictce"]],
                                       BATCHES["model"]["qtpies-model"][1] + ["qtpies-model", "qtpies-ce-score"])}
# M2: density-matched training (16% of train S1 absent -> test-like ~38% distractors), sharded stage 1, CE + sibling margins
BATCHES["ftd_train"] = {f"qtpies-ftd-train-{i}of5": ([["feats.py", "train", str(i), "5"]], TRAIN_BLOCK) for i in range(5)}
BATCHES["m2_fit1"] = {"qtpies-m2-fit1": ([["run.py", "fit1"]], list(BATCHES["ftd_train"]))}
BATCHES["m2_pred"] = {f"qtpies-m2-pred-{i}of5": ([["run.py", f"predshard:{i}:5"]],
                                                 list(BATCHES["ftd_train"]) + list(BATCHES["feats_test"]) + ["qtpies-m2-fit1"])
                      for i in range(5)}
BATCHES["m2_s2"] = {"qtpies-m2-s2": ([["run.py", "gather1"], ["run.py", "train2ce"], ["run.py", "predictce"]],
                                     list(BATCHES["m2_pred"]) + list(BATCHES["ftd_train"]) + list(BATCHES["feats_test"])
                                     + ["qtpies-ce-score"])}
BATCHES["m3"] = {"qtpies-m3": ([["run.py", "gather1"], ["run.py", "train2g"], ["run.py", "predictg"]],
                               list(BATCHES["m2_pred"]) + list(BATCHES["ftd_train"]) + list(BATCHES["feats_test"])
                               + ["qtpies-ce-score"])}
# M4: cross-encoder retrained with the stage-1 model's own confusions as hard negatives; scores ALL stage-2 pairs
BATCHES["ce4_train"] = {f"qtpies-ce4-train-{k}": ([["ce.py", "train", str(k)]], TRAIN_BLOCK + ["qtpies-m2-s2"]) for k in (0, 1)}
BATCHES["ce4_score"] = {
    "qtpies-ce4-score-train": ([["ce.py", "score", "train", "oof_m.parquet", "ce_train.parquet", "ce_fold0", "ce_fold1"]],
                               list(BATCHES["ce4_train"]) + ["qtpies-m2-s2"]),
    "qtpies-ce4-score-test": ([["ce.py", "score", "test", "pred_test_m.parquet", "ce_test.parquet", "ce_fold0", "ce_fold1"]],
                              list(BATCHES["ce4_train"]) + ["qtpies-m2-s2"])}
BATCHES["m4"] = {"qtpies-m4": ([["run.py", "gather1"], ["run.py", "train2ce"], ["run.py", "predictce"]],
                               list(BATCHES["m2_pred"]) + list(BATCHES["ftd_train"]) + list(BATCHES["feats_test"])
                               + list(BATCHES["ce4_score"]))}
BATCHES["m3t"] = {"qtpies-m3t": ([["run.py", "gather1"], ["run.py", "train2g"], ["run.py", "predictg"]], BATCHES["m3"]["qtpies-m3"][1])}
# CE-2: raw-text cross-encoder (diversity), hard negatives, full stage-2 scoring; m5 = stage 2 with both CEs
BATCHES["ce2_train"] = {f"qtpies-ce2-train-{k}": ([["ce.py", "train", str(k)]], TRAIN_BLOCK + ["qtpies-m2-s2"]) for k in (0, 1)}
BATCHES["ce2_score"] = {
    "qtpies-ce2-score-train": ([["ce.py", "score", "train", "oof_m.parquet", "ce2_train.parquet", "ce2_fold0", "ce2_fold1"]],
                               list(BATCHES["ce2_train"]) + ["qtpies-m2-s2"]),
    "qtpies-ce2-score-test": ([["ce.py", "score", "test", "pred_test_m.parquet", "ce2_test.parquet", "ce2_fold0", "ce2_fold1"]],
                              list(BATCHES["ce2_train"]) + ["qtpies-m2-s2"])}
BATCHES["m5"] = {"qtpies-m5": ([["run.py", "gather1"], ["run.py", "train2ce"], ["run.py", "predictce"]],
                               BATCHES["m4"]["qtpies-m4"][1] + list(BATCHES["ce2_score"]))}
# M6: synthetic twin world (src/twins.py on Kaggle: extended n_train_s3/train_s3 + twin candidates, kernel qtpies-t-gen).
# Stage-1 models and both CEs are reused: twin pairs get stage-1 OOF p and CE scores, then stage 2 is retrained with
# twin features. m6a leaves the generic-swap twins (BER_EXCL_RI = their ri range) out of stage 2; m6b keeps them.
BATCHES["t_gen"] = {"qtpies-t-gen": ([["twins.py", "gen"], ["twins.py", "block"]], [])}
BATCHES["t_feats"] = {"qtpies-t-feats": ([["feats.py", "train"], ["run.py", "predtwin"]],
                                         TRAIN_BLOCK + list(BATCHES["ftd_train"]) + ["qtpies-m2-fit1", "qtpies-t-gen"])}
BATCHES["t_ce4"] = {"qtpies-t-ce4": ([["ce.py", "score", "train", "oof_m_part9.parquet", "ce_train_twin.parquet", "ce_fold0", "ce_fold1"]],
                                     list(BATCHES["ce4_train"]) + ["qtpies-t-feats", "qtpies-t-gen"])}
BATCHES["t_ce2"] = {"qtpies-t-ce2": ([["ce.py", "score", "train", "oof_m_part9.parquet", "ce2_train_twin.parquet", "ce2_fold0", "ce2_fold1"]],
                                     list(BATCHES["ce2_train"]) + ["qtpies-t-feats", "qtpies-t-gen"])}
M6_UP = BATCHES["m5"]["qtpies-m5"][1] + ["qtpies-t-gen", "qtpies-t-feats", "qtpies-t-ce4", "qtpies-t-ce2"]
for v in ("m6a", "m6b", "m7a", "m7b"):  # m7 = m6 + house-number shape features (BER_NUMF)
    BATCHES[v] = {f"qtpies-{v}": ([["run.py", "gather1"], ["run.py", "train2ce"], ["run.py", "predictce"]], M6_UP)}
# M8: cluster (set-aware) cross-encoder over each S1's stage-2 candidate set, on the M7b twin world; score -> stage 2
CCE_UP = ["qtpies-m7b", "qtpies-t-gen"]
BATCHES["cce_smoke"] = {"qtpies-cce-smoke": ([["cce.py", "run", "0"]], CCE_UP)}
BATCHES["cce"] = {f"qtpies-cce-{k}": ([["cce.py", "run", str(k)]], CCE_UP) for k in (0, 1)}
BATCHES["m8"] = {"qtpies-m8": ([["run.py", "gather1"], ["run.py", "train2ce"], ["run.py", "predictce"]], M6_UP + list(BATCHES["cce"]))}
# M9: the same cluster CE on a 3x larger backbone (multilingual-e5-base, MIT), as an extra stage-2 feature on top of M8
BATCHES["cceb"] = {f"qtpies-cceb-{k}": ([["cce.py", "run", str(k)]], CCE_UP) for k in (0, 1)}
BATCHES["m9"] = {"qtpies-m9": ([["run.py", "gather1"], ["run.py", "train2ce"], ["run.py", "predictce"]],
                               M6_UP + list(BATCHES["cce"]) + list(BATCHES["cceb"]))}
M2_ENV = {"BER_DROP": "16"}
BATCH_ENV = {"ftd_train": M2_ENV, "m2_fit1": M2_ENV, "m2_pred": M2_ENV, "m2_s2": dict(M2_ENV, BER_POOL="12000000"),
             "m3": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1"),
             "ce4_train": {"BER_CE_HARD": "1"}, "ce4_score": {"BER_CE_BAND": "0.002,1.01"},
             "m4": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1"),
             "ce2_train": {"BER_CE_HARD": "1", "BER_CE_RAW": "1", "BER_CE_TAG": "ce2"},
             "ce2_score": {"BER_CE_BAND": "0.002,1.01", "BER_CE_RAW": "1", "BER_CE_TAG": "ce2"},
             "m5": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1"),
             "t_gen": dict(M2_ENV, BER_TWIN_G="US:0.03,India:0.03"), "t_feats": M2_ENV, "t_ce4": {"BER_CE_BAND": "0.002,1.01"},
             "t_ce2": {"BER_CE_BAND": "0.002,1.01", "BER_CE_RAW": "1", "BER_CE_TAG": "ce2"},
             "m6b": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1", BER_TWINF="1"),
             "m6a": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1", BER_TWINF="1", BER_EXCL_RI=os.environ.get("BER_EXCL_RI", "")),
             "m7b": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1", BER_TWINF="1", BER_NUMF="1"),
             "m7a": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1", BER_TWINF="1", BER_NUMF="1", BER_EXCL_RI=os.environ.get("BER_EXCL_RI", "")),
             "cce_smoke": dict(M2_ENV, BER_CCE_SMOKE="2000"), "cce": M2_ENV,
             "m8": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1", BER_TWINF="1", BER_NUMF="1"),
             "cceb": dict(M2_ENV, BER_CCE_BASE="intfloat/multilingual-e5-base", BER_CCE_TAG="cceb", BER_CCE_UNITS="200000"),
             "m9": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1", BER_TWINF="1", BER_NUMF="1"),
             "m3t": dict(M2_ENV, BER_POOL="12000000", BER_GENERIC="1",
                         BER_LGB2='{"learning_rate": 0.04, "num_leaves": 511, "min_data_in_leaf": 100, "feature_fraction": 0.7, "patience": 200}')}
GPU_BATCHES = {"block_gpu", "ce_train", "ce_score", "model2", "m2_s2", "ce4_train", "ce4_score", "ce2_train", "ce2_score",
               "t_ce4", "cce", "cceb"}  # single CPU jobs use the separate GPU pool (t_ce2 moved to CPU: T4/P100 queue saturated)
EXTRA_DATASETS = {"feats_test": ["razanml/qtpies-er-cand-test"], "m2_pred": ["razanml/qtpies-er-cand-test"], "m3": ["razanml/qtpies-er-cand-test"], "m4": ["razanml/qtpies-er-cand-test"], "m3t": ["razanml/qtpies-er-cand-test"], "m5": ["razanml/qtpies-er-cand-test"],
                  "ce2_train": ["razanml/qtpies-er-data"], "ce2_score": ["razanml/qtpies-er-data"],
                  "m2_s2": ["razanml/qtpies-er-cand-test"],
                  "t_gen": ["razanml/qtpies-er-data"], "t_ce2": ["razanml/qtpies-er-data"],
                  "m6a": ["razanml/qtpies-er-cand-test"], "m6b": ["razanml/qtpies-er-cand-test"],
                  "m7a": ["razanml/qtpies-er-cand-test"], "m7b": ["razanml/qtpies-er-cand-test"],
                  "cce_smoke": ["razanml/qtpies-er-data"], "cce": ["razanml/qtpies-er-data"], "m8": ["razanml/qtpies-er-cand-test"],
                  "cceb": ["razanml/qtpies-er-data"], "m9": ["razanml/qtpies-er-cand-test"]}

TEMPLATE = r'''
import base64, glob, io, os, shutil, subprocess, sys, tarfile, time
JOBS = __JOBS__
t0 = time.time()
subprocess.run("pip install -q __REQ__ 2>&1 | tail -1", shell=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall("/tmp/code")
W = "/tmp/work"
os.makedirs(W, exist_ok=True)
for p in sorted(glob.glob("/kaggle/input/**/*", recursive=True), key=lambda p: "/qtpies-t-gen/" not in p):  # twin world files win
    if not os.path.isfile(p) or p.endswith((".log", ".ipynb", ".html", ".tsv")):
        continue
    parent, base = os.path.basename(os.path.dirname(p)), os.path.basename(p)
    if "__" in base:  # flattened dataset files: cand_test__US_003.parquet -> cand_test/US_003.parquet
        rel = os.path.join(*base.split("__", 1))
    else:
        rel = os.path.join(parent, base) if parent.startswith(("cand_", "F_", "ce_fold", "ce2_fold")) else base
    dst = os.path.join(W, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        os.symlink(p, dst)
print("inputs linked", len(glob.glob(W + "/**/*.parquet", recursive=True)), flush=True)
env = dict(os.environ, BER_WORK=W, BER_OUT=W + "/output", BER_THREADS="4", POLARS_MAX_THREADS="4", PYTHONUNBUFFERED="1",
           BER_POOL="8000000")
env.update(__ENV__)
print("env", __ENV__, flush=True)
failed = None
for j in JOBS:
    print(">>>", j, flush=True)
    if subprocess.run([sys.executable] + j, cwd="/tmp/code/src", env=env).returncode:
        failed = j  # still export what earlier jobs produced
        break
n = 0
for p in glob.glob(W + "/**/*", recursive=True):
    if os.path.isfile(p) and not os.path.islink(p):
        d = os.path.join("/kaggle/working", os.path.relpath(p, W))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy(p, d)
        n += 1
print(f"exported {n} files, {time.time() - t0:.0f}s", flush=True)
if failed:
    raise SystemExit(f"job failed: {failed}")
'''


def code_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(os.path.join(HERE, "src"), arcname="src", filter=lambda i: None if "__pycache__" in i.name else i)
    return base64.b64encode(buf.getvalue()).decode()


def push(batch, only=None):
    """Push the batch's kernels (or only the given slugs). Returns slugs refused by the concurrent-session cap."""
    assert batch not in ("m6a", "m7a") or BATCH_ENV[batch]["BER_EXCL_RI"], "set BER_EXCL_RI=lo:hi (G-twin ri range) for m6a"
    b64, refused = code_b64(), []
    for slug, (jobs, upstream) in BATCHES[batch].items():
        if only is not None and slug not in only:
            continue
        d = os.path.join(KDIR, slug)
        os.makedirs(d, exist_ok=True)
        src = f'CODE = "{b64}"\n' + TEMPLATE.replace("__JOBS__", json.dumps(jobs)).replace("__REQ__", REQ) \
            .replace("__ENV__", json.dumps(BATCH_ENV.get(batch, {})))
        open(os.path.join(d, "run.py"), "w").write(src)
        json.dump({"id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": batch in GPU_BATCHES, "enable_tpu": False,
                   "enable_internet": True, "dataset_sources": [DATASET] + EXTRA_DATASETS.get(batch, []),
                   "competition_sources": [],
                   "kernel_sources": [f"{USER}/{u}" for u in upstream]}, open(os.path.join(d, "kernel-metadata.json"), "w"))
        acc = ["--accelerator", "NvidiaTeslaT4"] if batch in GPU_BATCHES else []
        r = subprocess.run(["kaggle", "kernels", "push", "-p", d] + acc, capture_output=True, text=True)
        msg = (r.stdout + r.stderr).strip().splitlines()[-1]
        print(slug, "->", msg, flush=True)
        if "Maximum" in msg:
            refused.append(slug)
    return refused


def status(batch):
    for slug in BATCHES[batch]:
        r = subprocess.run(["kaggle", "kernels", "status", f"{USER}/{slug}"], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip().splitlines()[-1])


def fetch(batch, dest):
    for slug in BATCHES[batch]:
        tmp = os.path.join(KDIR, slug, "out")
        subprocess.run(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", tmp, "-o"], capture_output=True, text=True)
        for dp, _, fs in os.walk(tmp):
            for f in fs:
                if f.endswith(".parquet"):
                    rel = os.path.relpath(os.path.join(dp, f), tmp)
                    os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
                    os.replace(os.path.join(dp, f), os.path.join(dest, rel))
        print(slug, "fetched")


def _state(slug):
    r = subprocess.run(["kaggle", "kernels", "status", f"{USER}/{slug}"], capture_output=True, text=True)
    return (r.stdout + r.stderr).rsplit("KernelWorkerStatus.", 1)[-1].strip().strip('"').strip()


def wait(batch, poll=60):
    import time
    while True:
        st = {s: _state(s) for s in BATCHES[batch]}
        bad = [s for s, v in st.items() if v.startswith(("ERROR", "CANCEL")) or "404" in v]  # 404: push was refused
        if bad:
            raise SystemExit(f"{batch}: failed {bad}")
        if all(v.startswith("COMPLETE") for v in st.values()):
            print(time.strftime("%H:%M:%S"), batch, "COMPLETE", flush=True)
            return
        time.sleep(poll)


def chain(*batches):
    """Run batches in order: push (retrying while the 5-session cap is full) then wait for all kernels.
    A batch name prefixed with '=' is only waited on (already pushed)."""
    import time
    for b in batches:
        if not b.startswith("="):
            pending = None
            for attempt in range(120):
                pending = push(b, pending)
                if not pending:
                    break
                time.sleep(60)
            time.sleep(90)  # status right after a push can still show the previous version
        wait(b.lstrip("="))


if __name__ == "__main__":
    cmd, batch, *rest = sys.argv[1:]
    {"push": lambda: push(batch), "status": lambda: status(batch), "fetch": lambda: fetch(batch, rest[0]),
     "wait": lambda: wait(batch), "chain": lambda: chain(batch, *rest)}[cmd]()
