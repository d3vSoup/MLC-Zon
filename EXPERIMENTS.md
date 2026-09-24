# Experiments log

One row per run. CV = OOF macro F0.5 on all 2,206,821 train S1 (LightGBM fit on the 20% S1 sample, each pair predicted by the model of its held-out fold). PC = share of true pairs present in candidates.

| time (IST) | git | change | PC | cands/S1 mean | CV overall | US | India | singleton-only | LB |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 03:57 | phase0+1 commit | v0: norm v0 + Indic translit; block = word 1-2gram TF-IDF on name_core+addr_norm (max_df 0.003, top-10 S1 per S2/S3 record, within country); 12 feats (8 rapidfuzz, postal, tfidf score/rank, src); LightGBM 3000 rounds on 20% S1 sample; owner argmax + tau=0.63 (plateau 0.56-0.70); test = fold-0 model only | 0.9654 | 46.65 (p95 127) | 0.8891 | 0.8915 | 0.8854 | 0.7355 | not uploaded yet |
