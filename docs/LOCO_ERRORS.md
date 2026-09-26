# LOCO errors: unseen vs seen (label-scored, same model)

## us2in (us2in_num_names2): errors per 1k S1 and their type shares

```
  side error  per_1k_S1  core_eq  legal_add  num_shift  num_same  gen_swap  q_median
  seen    FN      52.56    0.467      0.108      0.290     0.153     0.000     0.442
  seen    FP      10.02    0.392      0.336      0.164     0.359     0.001     0.908
unseen    FN     241.21    0.133      0.398      0.116     0.491     0.000     0.312
unseen    FP     156.01    0.083      0.167      0.108     0.164     0.013     0.985
```

### top shifted high-importance features (hard rows, unseen minus seen, in SD units)

```
        feature  label  seen_mean  unseen_mean    smd
      hn_suffix      1     3.1790       1.7158 -1.008
    hn_in_other      0     0.1603       0.5091  0.736
        n_num_b      1     1.1470       1.9021  0.701
        n_num_b      0     0.7384       1.5745  0.670
        n_num_a      1     1.2841       1.9501  0.669
        n_num_a      0     1.3175       1.9788  0.608
       num_jacc      0     0.1690       0.3558  0.584
             p1      0     0.1234       0.2868  0.584
     hn_reldiff      0     0.1366       0.3478  0.581
       sib_wmax      0     0.9129       0.8437 -0.476
   n_s1_c50_rec      0     0.1504       0.3996  0.473
    v2_idf_jacc      0     0.7048       0.5405 -0.457
        n_first      1     0.8510       0.6944 -0.384
sib_w_other_rec      1     0.8376       0.7621 -0.349
        n_first      0     0.8462       0.7009 -0.349
sib_w_other_rec      0     0.9474       0.8972 -0.344
    v2_idf_jacc      1     0.7605       0.6364 -0.337
    p_other_rec      0     0.2779       0.3842  0.323
     hn_reldiff      1     0.0831       0.1773  0.314
    sib_addr_eq      1     0.4830       0.3316 -0.306
```

## in2us (in2us_num_names2c): errors per 1k S1 and their type shares

```
  side error  per_1k_S1  core_eq  legal_add  num_shift  num_same  gen_swap  q_median
  seen    FN      54.41    0.322      0.236      0.161     0.226     0.001     0.488
  seen    FP      13.39    0.331      0.408      0.133     0.323     0.004     0.893
unseen    FN      97.71    0.307      0.102      0.152     0.365     0.003     0.414
unseen    FP      51.41    0.321      0.390      0.302     0.390     0.001     0.955
```

### top shifted high-importance features (hard rows, unseen minus seen, in SD units)

```
        feature  label  seen_mean  unseen_mean    smd
      hn_suffix      1     1.7087       3.1814  1.013
    hn_in_other      0     0.5076       0.1317 -0.866
     hn_reldiff      0     0.3702       0.1176 -0.747
        n_num_b      1     1.8983       1.1470 -0.697
       num_jacc      0     0.3819       0.1587 -0.691
        n_num_a      1     1.9513       1.2838 -0.669
        n_num_a      0     1.9628       1.3116 -0.664
        n_first      1     0.6918       0.8512  0.390
        n_num_b      0     1.2351       0.8085 -0.389
    v2_idf_jacc      1     0.6340       0.7595  0.339
     hn_reldiff      1     0.1801       0.0827 -0.323
       sib_wmax      0     0.8638       0.9056  0.306
    sib_addr_eq      1     0.3376       0.4818  0.292
        n_first      0     0.6831       0.7996  0.272
    v2_idf_jacc      0     0.5992       0.6900  0.270
sib_w_other_rec      1     0.7931       0.8473  0.251
             p1      0     0.1254       0.1793  0.235
      hn_suffix      0     0.6328       0.3991 -0.212
    p_share_rec      0     0.5500       0.6300  0.182
 hn_absdiff_log      1     1.1396       0.7598 -0.169
```

## Reading (Sun 00:15)

**Q1. Which error types grow when the country is unseen?** Errors grow 4-15x (US->IN: FN 53 -> 241, FP 10 -> 156 per
1k S1; IN->US: FN 54 -> 98, FP 13 -> 51). Unseen false positives are confident (median q 0.985 / 0.955): ranking, not
calibration (decode rules cannot fix them; see EXPERIMENTS 21:52). India-unseen FNs are mostly "same house number,
different core name" (num_same 49%, core_eq 13%): transliterated / noisy India names the US-trained model does not
trust. Generic-word twins are rare in every cell (gen_swap <= 1.3%).

**Q2. Which high-importance features shift most?** House-number *format* features: `hn_suffix` (+-1.0 SD),
`hn_in_other` (0.7-0.9), `n_num_a` / `n_num_b` (0.6-0.7: India addresses carry ~1.9 numbers, US ~1.2), `num_jacc`,
`hn_reldiff` (0.6-0.75). The shift flips sign with the direction, i.e. the features encode each country's address
format. Name IDF / generic features shift less (`v2_idf_jacc` 0.3-0.45 SD), consistent with names2c failing LOCO.

**Q3. Country-specific priors vs transferable relations.**
- priors fitted on a split or tied to a format: IDF-Jaccard (`nc_idf_jacc`, `ad_idf_jacc`, `v2_idf_jacc`), generic
  sets (`rare_*`), frequency shares (`f_*`), number counts (`n_num_a`, `n_num_b`), `hn_in_other`, `hn_suffix`.
- transferable relations: within-S1 / within-record ranks, margins and shares (`*_rank`, `*_margin_rec`, `*_gap_s1`,
  `p_share_*`), sibling agreement (`sib_*`), exact-match flags (`house_eq`, `postal`, `v2_eq`, `legal_eq`), string
  similarities of normalized text.

## Ranked fix hypotheses
1. **Drop raw format-count features** (`n_num_a`, `n_num_b`, `hn_in_other`): largest measured shift, no matching
   meaning. Differs from the dropped per-country quantile normalization (-0.0014): that rescaled features within a
   country (and still let the model use format); this removes the format signal. Size: unknown, LOCO test running.
   France addresses are US-like (one house number), so the France effect is expected to be small either way.
2. Per-country fitting of name priors (names2c): **tested, failed LOCO** (-0.0007 / -0.0003).
3. Name-trust for noisy/transliterated names (India-unseen FNs): India-specific (native scripts); low relevance for
   France; not tonight.
