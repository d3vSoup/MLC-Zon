# AUTO RESULTS (Sun 00:03)

# Night results (scripted chain, Sat 26 -> Sun 27 Sep)

- names2c gate: FAIL | OOF US 0.98971->0.98972 (+0.00001), India 0.98456->0.98454 (-0.00002), overall 0.98765->0.98765 | LOCO unseen US->IN 0.91571->0.91501, IN->US 0.96935->0.96902 | feature set r2_names2
- CE2 integrity /home/Muaz/Downloads/ce2_soup: files 0, md5 list MISSING -> FAIL
- CE2 half gate: FAIL | OOF (H2 S1) lgbm2_ceh 0.98744 vs lgbm2_ce2h 0.98759 (+0.00015; US +0.00012, India +0.00017) | blend model lgbm2_ce
- blend2 (model lgbm2_ce, w 0.5): OOF on live S1 vs blend_cal's model: 0.99032 -> 0.99032 (+0.00000)
  - blend2_raw: md5 da71932fa9a3fb048715ae267d9a23e3 | US k 1 empty 0.0575 links/S1 3.3877; France k 1 empty 0.0525 links/S1 3.3329; India k 1 empty 0.058 links/S1 3.3678
  - blend2_cal: md5 0a0b14806bcddf3370258467ae3fddd9 | US k 1 empty 0.0575 links/S1 3.3877; France k 0.184 empty 0.0592 links/S1 3.1967; India k 1 empty 0.058 links/S1 3.3678
  - blend2_unseen_k0.184: md5 c838afa6f839a2fbd1e4dc620db1f61e | US k 1 empty 0.0575 links/S1 3.3877; France k 0.184 empty 0.0592 links/S1 3.1967; India k 1 empty 0.058 links/S1 3.3678
- Phase A: docs/LOCO_ERRORS.md written
- MORNING_BRIEF.md draft written


## LOCO unseen side

- us2in num_names2: unseen F0.5 0.91571, singleton 0.78762, precision 0.952
- us2in num_names2c: unseen F0.5 0.91501, singleton 0.7856, precision 0.9511
- in2us num_names2: unseen F0.5 0.96935, singleton 0.9217, precision 0.9852
- in2us num_names2c: unseen F0.5 0.96902, singleton 0.91947, precision 0.9846

## Candidate folders

| folder | md5 | France k | France empty | France links/S1 |
|---|---|---|---|---|
| blend2_cal | 0a0b14806bcddf3370258467ae3fddd9 | 0.18385635887853335 | 0.0592 | 3.1967 |
| blend2_raw | da71932fa9a3fb048715ae267d9a23e3 | 1.0 | 0.0525 | 3.3329 |
| blend2_unseen_k0.184 | c838afa6f839a2fbd1e4dc620db1f61e | 0.184 | 0.0592 | 3.1967 |
| blend_cal | 0a0b14806bcddf3370258467ae3fddd9 | 0.18385635887853335 | 0.0592 | 3.1967 |
| blend_raw | da71932fa9a3fb048715ae267d9a23e3 | 1.0 | 0.0525 | 3.3329 |
| blend_unseen_k0.5 | 6e327bf79470ecc4bcbd851bd22aa1fd | 0.5 | 0.0561 | 3.2718 |
| blend_unseen_k1.5 | 2240d33a2fce773d812ba3e339243d41 | 1.5 | 0.0506 | 3.3753 |
| blend_unseen_k2 | 31363753439f13e061b82b3a9749ce64 | 2.0 | 0.0494 | 3.4124 |
| blend_unseen_k3 | 861a28ca1cfb3d536968e3b7ef43d8f3 | 3.0 | 0.0481 | 3.4817 |
| ce_full_cal | f3fc911c3beb11f12489a6555c7d4180 | 0.19588304543065221 | 0.0602 | 3.1835 |
| ce_full_v1 | 3a0d4c771a1203acf207811ee2c8a2fe | - | 0.0564 | - |
| ce_half_v1 | 7083a967b1c2bd32abbd39f00bad72f5 | - | 0.0557 | - |
| numbers_france_v1 | cd673b353beb3e431f0a23da1ce3bd40 | - | 0.052 | - |
| probe_unseen_empty | a46ab86c3ec89d34c7e485840583900f | - | - | - |
