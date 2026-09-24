import argparse
import importlib

# stage -> (module, function)
STAGES = {"eda": ("ber.eda", "run"), "normalize": ("ber.normalize", "run"), "folds": ("ber.folds", "run"),
          "block": ("ber.blocking.tfidf", "run"), "blockreport": ("ber.blocking.report", "run"),
          "features": ("ber.features.strings", "run"), "train": ("ber.models.lgbm", "run"),
          "predict": ("ber.models.lgbm", "predict"), "decode": ("ber.decode", "run"),
          "write": ("ber.decode", "write"), "report": ("ber.analysis", "run")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--split", choices=["train", "test"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, help="block: sample this many queries per country (timing probe)")
    a = ap.parse_args()
    mod, name = STAGES[a.stage]
    fn = getattr(importlib.import_module(mod), name)
    kw = {k: v for k, v in (("split", a.split), ("force", a.force), ("limit", a.limit)) if v is not None}
    fn(**{k: v for k, v in kw.items() if k in fn.__code__.co_varnames})


if __name__ == "__main__":
    main()
