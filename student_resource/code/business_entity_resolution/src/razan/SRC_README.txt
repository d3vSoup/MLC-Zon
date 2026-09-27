QTpies M8 source: the exact code that ran in Kaggle kernel qtpies-m8.

src/ = byte-exact extraction of the tarball embedded in kaggle/k/qtpies-m8/run.py
(SHA-256 over the sorted .py files = 4594535af76c...; the same snapshot ran in qtpies-cce-0/1).
Git: tag m8 on https://github.com/razancodes/MLC-raZON (main a7e7d69). Tag m8 has the same lines in every file;
only line endings differ (git normalises to LF; this Windows checkout had mixed CRLF/LF).
kaggle_runner.py, REPRODUCE.md, REPRODUCE_M8.md and AUDIT.md are from tag m8.
Start with REPRODUCE_M8.md (twin training: section 1; Kaggle DAG steps 13-19: section 2; cluster CE: 2.5; blend bundle: 4).
