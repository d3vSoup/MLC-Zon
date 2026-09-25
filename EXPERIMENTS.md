# Experiments log

One row per run. CV = OOF macro F0.5 on all 2,206,821 train S1 (LightGBM fit on the 20% S1 sample, each pair predicted by the model of its held-out fold). PC = share of true pairs present in candidates.

| time (IST) | git | change | PC | cands/S1 mean | CV overall | US | India | singleton-only | LB |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 03:57 | phase0+1 commit | v0: norm v0 + Indic translit; block = word 1-2gram TF-IDF on name_core+addr_norm (max_df 0.003, top-10 S1 per S2/S3 record, within country); 12 feats (8 rapidfuzz, postal, tfidf score/rank, src); LightGBM 3000 rounds on 20% S1 sample; owner argmax + tau=0.63 (plateau 0.56-0.70); test = fold-0 model only | 0.9654 | 46.65 (p95 127) | 0.8891 | 0.8915 | 0.8854 | 0.7355 | not uploaded yet |
| 2026-09-25 05:44 | phase5 commit | decode v1 on same lgbm0 OOF: owner argmax -> isotonic (cross-fit by fold) -> per-S1 gate LightGBM (8 feats) -> decoders: single tau 0.8891 / gate+tau2 (0.56, 0.65) 0.8923 / expected-F 0.8880 / **expected-F with E[F\|0]=1-gate 0.8932 (chosen)**; fold std 0.00025 | 0.9654 | 46.65 | 0.8932 | 0.8954 | 0.8900 | 0.7056 | not uploaded yet |
| 2026-09-25 11:21 | phase2 commit | norm v1 (address canonical short forms, US/India state codes incl. transliterated, zero-strip, glue split, ordinals, DBA split, landmarks, sound-alike skeleton); blocking 2 channels: fw = all tokens (name words, any-order name pairs, glued name, skeleton, address words+bigrams) top-10 + hh = half name / half address cosine top-10, union 153.5M pairs; lgbm lr 0.1, 1000 rounds; decoder expF+gate | 0.9825 (US 0.9895, IN 0.9720) | 69.54 (p95 211) | 0.9305 | 0.9393 | 0.9174 | 0.8124 | not uploaded yet |
