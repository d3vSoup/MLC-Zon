# Blocking fix: sibling-based candidate expansion. STOPPED (ceiling gain < 0.002)

Idea: for each "orphan" S2/S3 record (best OOF p < 0.5), search the "sure" records (p >= 0.5, same country)
by TF-IDF on name + address, and propose the sure record's S1 as a new candidate.
Measured on 60k sampled orphans per country, then extrapolated to all 2.9M train orphans. Uses round-2 (1000 rounds) OOF p.

- Blocking-missed true pairs today: 133,898. 97.3% of them are orphans, so the idea targets the right records.
- Ceiling F0.5 = macro F0.5 if every in-candidate true pair were picked and nothing else.

| sibling top-K | recovered misses (est.) | new pairs (est.) | ceiling before | ceiling after | gain |
|---|---|---|---|---|---|
| 1 | ~15.3k (US 9.2%, India 13.1%) | ~2.8M | 0.99407 | 0.99488 | +0.00080 |
| 3 | ~21.6k (US 12.8%, India 18.6%) | ~5.4M | 0.99407 | 0.99519 | +0.00112 |
| 5 | ~25.5k (US 14.6%, India 22.7%) | ~8.75M | 0.99407 | 0.99541 | **+0.00133** |

Mean candidates per S1: 69.54 now, about 73.5 with K=5.

The ceiling gain is below the 0.002 stop rule, so features, retraining, and new test output were **not** done.
The real OOF gain would be lower than the ceiling, because the recovered pairs are hard ones (weak names or addresses).
Most remaining misses are records with no near-copy among the sure records (for example an empty address plus a chain name).
