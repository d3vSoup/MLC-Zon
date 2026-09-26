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
