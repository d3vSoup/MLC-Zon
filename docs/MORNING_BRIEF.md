# Morning brief (auto-draft by the night chain; refine in session)

## 05:30 upload: blend_raw (France k = 1) vs blend_cal (k 0.184, public 0.985131)
- LOCO (India as unseen country, labels): no correction is best; k 0.25 cost -0.006, link-count k 2.3 cost -0.001.
  If France behaves like unseen India, blend_raw wins by roughly +0.001 public.
- QTpies M4 probe: France down-scaling helped (+0.002) on a model that over-linked France. Our blend under-links France
  (assigned ratio 0.963 vs 0.973-0.979), so the LOCO reading is more likely to apply.
- If raw wins: France k 1 stays; no more France points (LOCO: k > 1 loses). If raw loses: try k 0.5 next.

## Night results
# Night results (scripted chain, Sat 26 -> Sun 27 Sep)

- names2c gate: FAIL | OOF US 0.98971->0.98972 (+0.00001), India 0.98456->0.98454 (-0.00002), overall 0.98765->0.98765 | LOCO unseen US->IN 0.91571->0.91501, IN->US 0.96935->0.96902 | feature set r2_names2
- CE2 integrity /home/Muaz/Downloads/ce2_soup: files 0, md5 list MISSING -> FAIL
- CE2 half gate: FAIL | OOF (H2 S1) lgbm2_ceh 0.98744 vs lgbm2_ce2h 0.98759 (+0.00015; US +0.00012, India +0.00017) | blend model lgbm2_ce
- blend2 (model lgbm2_ce, w 0.5): OOF on live S1 vs blend_cal's model: 0.99032 -> 0.99032 (+0.00000)
  - blend2_raw: md5 da71932fa9a3fb048715ae267d9a23e3 | US k 1 empty 0.0575 links/S1 3.3877; France k 1 empty 0.0525 links/S1 3.3329; India k 1 empty 0.058 links/S1 3.3678
  - blend2_cal: md5 0a0b14806bcddf3370258467ae3fddd9 | US k 1 empty 0.0575 links/S1 3.3877; France k 0.184 empty 0.0592 links/S1 3.1967; India k 1 empty 0.058 links/S1 3.3678
  - blend2_unseen_k0.184: md5 c838afa6f839a2fbd1e4dc620db1f61e | US k 1 empty 0.0575 links/S1 3.3877; France k 0.184 empty 0.0592 links/S1 3.1967; India k 1 empty 0.058 links/S1 3.3678
- Phase A: docs/LOCO_ERRORS.md written

## Candidates (France stats)
| name | md5 | France k | France empty | France links/S1 |
|---|---|---|---|---|
| blend_cal | 0a0b14806bcddf3370258467ae3fddd9 | 0.18385635887853335 | 0.0592 | 3.1967 |
| blend_raw | da71932fa9a3fb048715ae267d9a23e3 | 1.0 | 0.0525 | 3.3329 |
| blend_unseen_k0.5 | 6e327bf79470ecc4bcbd851bd22aa1fd | 0.5 | 0.0561 | 3.2718 |
| blend_unseen_k1.5 | 2240d33a2fce773d812ba3e339243d41 | 1.5 | 0.0506 | 3.3753 |
| blend_unseen_k2 | 31363753439f13e061b82b3a9749ce64 | 2.0 | 0.0494 | 3.4124 |
| blend2_raw | da71932fa9a3fb048715ae267d9a23e3 | 1.0 | 0.0525 | 3.3329 |
| blend2_cal | 0a0b14806bcddf3370258467ae3fddd9 | 0.18385635887853335 | 0.0592 | 3.1967 |
| blend2_unseen_k0.184 | c838afa6f839a2fbd1e4dc620db1f61e | 0.184 | 0.0592 | 3.1967 |
| ce_full_cal | f3fc911c3beb11f12489a6555c7d4180 | 0.19588304543065221 | 0.0602 | 3.1835 |
