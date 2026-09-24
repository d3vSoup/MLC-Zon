# Phase 0 facts

## 1. Row counts and country mix

```
         file    rows      US   India   France
train_source1 2206821 1323633  883188      0.0
train_source2 5034616 3016817 2017799      0.0
train_source3 5285603 3170056 2115547      0.0
 test_source1 1732544  663106  809986 259452.0
 test_source2 4887273 1871330 2312565 703378.0
 test_source3 5082316 1945701 2405000 731615.0
```

France share of test S1: **0.1498**

## 2. Singleton share (all-empty floor)

**0.05585** of 2206821 train S1

## 3. G distribution

```
                         n   share
matched_entity_ids                
0                   123247  0.0558
1                   119157  0.0540
2                   375212  0.1700
3                   530841  0.2405
4                   484115  0.2194
5                   321957  0.1459
6+                  252292  0.1143
```
mean G=3.461; matches by source: {'S3': 3944746, 'S2': 3693619}

## 4. One-owner rule

IDs in more than one GT list: **0** (of 7638365 matched IDs)

## 5. Cross-country matched pairs

cross-country pairs: **0**; unresolved ids: 0

## 6. S2/S3 records matching nothing

- S2: unmatched share **0.2664** (by country: {'India': 0.2663, 'US': 0.2664})
- S3: unmatched share **0.2537** (by country: {'India': 0.2535, 'US': 0.2538})

## 7. S1 with 2+ records from the same source

share of all S1 with >=2 S2 matches: 0.5120; >=2 S3: 0.5547; either: 0.7680

## 8. Similarity of true pairs (token_set_ratio, lowercased raw)

```
            name_tsr                     addr_tsr                   
                0.05  0.25   0.50   0.75     0.05  0.25  0.50   0.75
src country                                                         
S2  India        8.7  43.5   91.3  100.0     60.0  89.1  94.9  100.0
    US          53.6  88.9   98.4  100.0     54.8  85.7  93.1   97.3
S3  India        9.4  74.2   93.1  100.0     45.0  82.5  91.4   95.9
    US          53.3  89.4  100.0  100.0     53.5  78.9  86.2   92.1
```

## 9. Hardest true pairs and most similar non-pairs

### 30 true pairs with lowest name similarity
```
  0.0 | Adda's Management | AZ, 4859 Spoonbill Drive, Tucson  <->  Xylobrix | 004859 Spoonbill Drive, Tucson, Arizona
  0.0 | Next College | Near Khalsa Football Ground Humaynpur, Sirhind, Fatehgarh Sahib, Punjab  <->  Umbraaviaria | PB, Sirhind, Near Khalsa Football Ground Humaynpur, Fatehgarh Sahib
  0.0 | Solstice | OK, 10873 Fox Road, Stratford  <->  5ynumbra | 10873 1/2 FOX RD, STRATFORD, OK
  0.0 | Zanet Corporation | 7333 Chaparral Road, Unit APT 116, Scottsdale, AZ  <->  Fluxlum | 7333 Chaparral Road, Unit APT 116, Scottsdale, Arizona
  0.0 | Big Media | Ec-48, Binodini Road Ghosh Para, Po- Desh Bandhunagar, Kolkata, Howrah, West Bengal  <->  Solonyx | Ec-48, Binodini Road Ghosh Para, Po- Desh Bandhunagar, Kolkata, Howrah, West Bengal
  0.0 | Tech Tech | House No. 262, Brahman Mohalla Katewara, North West Delhi, Delhi, North West, Delhi  <->  टेक टेक | DL, House No. 262, North West, Brahman Mohalla Katewara, North West Delhi, North
  0.0 | Fresh Better Homes | 5099 El Mirage Drive, Topock, AZ  <->  Jaxjax | 099 El Mirage Dr, TOPOCK, AZ
  0.0 | Mesa Ses | 1050 Longmore, Unit 240, Mesa, AZ  <->  Orbixylo | 1050 Longmore, # 240, MSEA, Arizona
  0.0 | Bia Inc. | 1106 S 3rd St, Louisville, KY  <->  Zephdrex | 001106 S 3rd St, Louisvlle CITY, Kentucky
  0.0 | Brown Four | MN, 1424 Arden Oaks Drive, Arden Hills  <->  Avizeph | Minnesota, Arden Oaks Dr, Arden Hills
  0.0 | QRO Golden Pvt Ltd | 18-505/F, 18-505/G, Vass, Arcade, Palakottuvayal, Kozhikode, Kerala  <->  Jaxjax | 18-505/F, Kerala, CALICUT REGION, KOZHIKODE
  0.0 | Cous | 1734 Johnson Avenue, Klamath Falls, OR  <->  Nexzeph | 1734 Johnson Ave, Klamath Falls, Oregon
  0.0 | Gray Financial LLC | 404 Coxs Fork Road, Danville, WV  <->  Veoquo | 404 COXS FORK RD, DANVILLE, WV
  0.0 | Buify | 790 Jacob Circle, Orange, TX  <->  Círanex | 790 Jacob Circle, Orange, Texas
  0.0 | Singhania Sangh | A 342 Manav Apartmentspaschim Vihar, New Delhi, North Delhi, Delhi  <->  Fluxecto | दिल्ली, null, NORTH DELHI, A 342 MANAV APARTMENTSPASCHIM VIHAR, NEW DELHI
  0.0 | Golden & Sons LLP | 3Rd Floor, Kitab Mahal, 192, Dr Dadabhai Naoroji, Mumbai, Maharashtra  <->  Ciraarc | Kitab Mahal, 192, Dr Dadabhai Naoroji, #3Rd Floor, Mumbai, MH
  0.0 | Sv & Co | No-127 8Th Cross 2Nd Main, 1St Block Vv Nagar, Bangalore South, Bangalore, Karnataka  <->  Zetazeph | No-127 8Th Cross 2Nd Main, 1St Block Vv Nagar, Bangalore South, Bangalore, Karnataka
  0.0 | Sinet LLC | 1016 Saber Lane, Herndon Town, VA  <->  Korquo | 1016 1/2 Saber Ln, Herndon, Virginia
  0.0 | Cozy Tattoo Inc | 14 Center Street, Machias, ME  <->  Fluxflux | 14 Center Street, MACHIAS CITY, ME
  3.8 | International Infrastructure | B-106 Sec-Cmahanagar Lucknow, Uttar Pradesh, Lucknow, Uttar Pradesh  <->  इंटरनेशनल इंफ्रास्ट्रक्चर | LUCKNOW, B-106 SEC-CMAHANAGAR LUCKNOW, UTTAR PRADESH
  3.8 | International Hospitality | No 1159 2Nd Floor, 5Th Main Road Hsr Layout 7Th Sector, Bangalore South, Bangalore, Karnataka  <->  ಇಂಟರ್‌ನ್ಯಾಷನಲ್ ಹಾಸ್ಪಿಟಾಲಿಟಿ | NO 1159- 2ND FLOOR, 5TH MAIN ROAD HSR LAYOUT 7TH SECTOR, BANGALORE SOUTH, BANGALORE, Karnataka
  3.9 | Perfect Infrastructure | Chennai, Chennai., Tamil Nadu, D/J-1613Th Main Road Anna Nagar West, Plot No.1493  <->  பர்ஃபெக்ட் இன்ஃப்ராஸ்ட்ரக்சர் | H.no, D/j-1613th Main Road Anna Nagar West, Chennai., Chennai, TN
  4.2 | International Construction | Malapuram, Kerala, 16/298 C, Ernad, C/O Mr. Jashid V K, Malappuram  <->  ഇന്റർനാഷണൽ കൺസ്ട്രക്ഷൻ | Keralam, MALAPURAM, C/O MR. JASHID V K, MALAPPURAM
  4.2 | International Technology | No-283-4, Mariyappa Building, Byramangala, Bidadi Hobli, Ramanagara Taluk, Ramanagara, Bangalore Rural, Karnataka  <->  ಇಂಟರ್‌ನ್ಯಾಷನಲ್ ಟೆಕ್ನಾಲಜಿ | NO-A-283-4, MARIYAPPA BUILDING, BYRAMANGALA, BIDADI HOBLI, RAMANAGARA TALUK, BIDADI, BANGALORE RURAL, Karnataka
  4.2 | International Technology | No-283-4, Mariyappa Building, Byramangala, Bidadi Hobli, Ramanagara Taluk, Ramanagara, Bangalore Rural, Karnataka  <->  ಇಂಟರ್‌ನ್ಯಾಷನಲ್ ಟೆಕ್ನಾಲಜಿ | No-a-283-4, Ramanagara, Bangalore Rural, KA
  4.3 | Premier Infrastructure | 402, Victoria Heights, Jewels Road, Bhavnagar, Gujarat  <->  પ્રીમિયર ઇન્ફ્રાસ્ટ્રક્ચર | 402, VICTORIA HEIGHTS, JEWELS ROAD, BHAVNAGAR, Gujarat
  4.3 | International Technologies | Old No D-6-A New Number D-28 East Vinod Nagar, Delhi, East Delhi, Delhi  <->  इंटरनेशनल टेक्नोलॉजीज | OLD NO D-6-A NEW NUMBER D-28 EAST VINOD NAGAR, DELHI, EAST DELHI, दिल्ली
  4.3 | International Foundation | Karnataka, Bangalore, B403, Amoda Valmark, Doddakammanahalli Off Bannereghatta Road  <->  ಇಂಟರ್‌ನ್ಯಾಷನಲ್ ಫೌಂಡೇಶನ್ | Block F-d/991 B403, Bangalore, Hulimavu, KA
  4.3 | Creative Enterprises | 6, Madurai Road, Trichy, Tamil Nadu  <->  கிரியேட்டிவ் எண்டர்பிரைசஸ் | No 84 6, Trichy, Madurai Road, TN
  4.3 | Tirupati Infrastructure | A-68, Iind Floor, Phase-Ii Mangol Puri Industrial Area, New Delhi, North Delhi, Delhi  <->  तिरुपति इंफ्रास्ट्रक्चर | A-d/68, North Delhi, New Delhi, DL
```
### 30 random same-country non-pairs with highest name similarity
```
100.0 | Physical Therapy Continental Care Group | 3703 Bell Arthur Road, Greenville, NC  <->  Physical Therapy | St Joseph, Indiana, 3010 State Blvd
100.0 | Mount Fellowship | 220 Main Street, Fl 1, New Albany, OH  <->  mount fellowship | 320- Old Hickory Blvd, Nashville, Tennessee
100.0 | Pediatric Dental Center LLC | 12513 Sun Haven Drive, City Of El Paso, TX  <->  Pediatric Dental Center | 12103 RUFFIN DRIVE, FAIRFAX, VA
100.0 | Department of Labor | 13903 Claymont Hill Drive, Cypress, TX  <->  Department of | Winfield, 62 Stanley St, Illinois
100.0 | First Agro Pvt Ltd | House No 300, Sector 4 Urban Estate, Near Blue Bells Model Scho, Ol, Gurgaon, Haryana  <->  First Agro | C110 SONAL LINK RESIDENCY, CHS LTD, LINK ROAD, MALAD WEST, Maharashtra
100.0 | Lp Emporium Private Limited | 8Th Cross, Bhbcs, Uttarahalli, No 11/2, Bangalore, Karnataka  <->  LP | Flat No 11 Tapasya Apartment S No.164 Plot No.14 Shastri Nagar Kothrud, Pune, Maharashtra
100.0 | Food India Ltd | E-323 Greater Kaolash 1, South Delhi, New Delhi, Delhi  <->  India | 51/217/3 West Arjun Nagar, Sindhi Colony, Agra, Uttar Pradesh, Agra
 94.5 | LS Producer Private Limited | 6971/13, First Floor, Multani Dhanda, Pahar Ganj, New Delhi, Central Delhi, Delhi  <->  All Producer Private Limited | NO ##25 MEENA FLORENCE RAJARHAT DASHADRONE BLOCK C, FLAT B 1ST FLOOR, DASHADRONE, KOLKATA, West Bengal
 94.1 | Mangala Services Private Limited | Plot No 27/1, Midc Shiroli, Shiroli, Hatkanangle, Kolhapur, Maharashtra  <->  Hi Services  Private Limited | C-208, II FLOOR, GOPAL HEIGHTS, NETAJI SUBHASH PLACE, PITAMPURA, DELHI, दिल्ली
 94.1 | IE Business Private Limited | S 0-9/471-C Pandeypur, Varanasi, Uttar Pradesh  <->  Balaji Ram Business Private Limited | #52, Gondia, Gondiya, महाराष्ट्र
 93.9 | GT (india) Private Limited | Surat, 1St Floor, Hotel Deepmala, Nr. Amisha Hotel, Una Pani Road, Surat, Gujarat, Room No. 1  <->  NASCENT INTERIOR (INDIA) HOLDINGS PRIVATE LIMITED | #927 Cts No 4756 B Near, Belgaum, Gokak, ಕರ್ನಾಟಕ
 93.9 | Modakpriya (india) Private Limited | Property Id-7E9F9K41, Above Dabar Colony Utkarsh Small Finance Civil Hospital Road Ladwa, Ladwa, Kurukshetra, Haryana  <->  Rs (india) Private Limited | C/o Akbar Hussain Industrial State Rampur Bantara, Ghazipur, UP
 93.9 | Krishna Trading Private Limited | 1/1, Jamalia Nagar, Perambur Purasawalkam, Chennai, Tamil Nadu  <->  Vs Trading  Private Limited | 24 Corporate Parknirmal Life Style Lbs Marg Mulund (W), Mumbai, Mumbai Ii, महाराष्ट्र
 93.1 | Red Engineering Private Limited | P. No. 21, Gopi Nagar, Near Sangam Cinema, Sanganer, Jaipur, Rajasthan  <->  Krishna Private Engineering Limited | C/O ARVIND KUMAR YADAV, KISHINI ROAD NAGAR, KARAHAL, Uttar Pradesh
 93.1 | Tayal Brothers Private Limited | Shop No. 28 Royal Mart, Phirni Road, Fazilka, Firozpur, Punjab  <->  asa brothers private limited | Hn 592 A-26, First Floor, Backside, Bhagwati Garden, Uttam Nagar, New Delhi, West Delhi, DL
 92.9 | Raj Foundation Private Limited | C/O Smt Rashmita Kr. Dubey, Wmahendra Kr. Dubey902-B/557, Daraganj, Allahabad, Uttar Pradesh  <->  Solutions Shubhashish Foundation Group Limited Private | 214 Laxman Place 19 Veer Savarkar Block Shakarpur, New Delhi, East Delhi, DL
 92.9 | New Industries Private Limited | Meerut, Uttar Pradesh, Garh Road, 217-221 Krishna Plaza  <->  Industries Limited Organo Overseas Private Odisha | No.11, Bangalore North, Bangalore, KA
 92.3 | One Ventures Private Limited | Ratan Heights, Medical Square Opposite State Bank Of India, Nagpur, Maharashtra  <->  South Ventures Private Limited | B3/504, 5Th Floor, G-business Park, Plot No. 25/26, Sector-30a, Sanpada, Navimumbai, Thane, MH
 92.3 | One Services Private Limited | Flat No. 203, S. No. 22/2, Sai Tej Apartment, Dhanori, Pune, Maharashtra  <->  Harda Overseas Private Limited Services | 79 Gola Pura, 18222, Harda, Harda, MP
 92.3 | One Services Private Limited | Rz-29, Upper Ground Floor, Khushi Ram Park, New Delhi, West Delhi, Delhi  <->  Ernakulam Private Limited Services | www.ernakulam.com | Flat No : E9-7 Heavenly Plaza, Ernakulam, KL
 92.3 | Red Brothers Private Limited | 5/815 Kuzhuvellil House, Unnichira Thrikkakara P O, Kanayannur, Ernakulam, Kerala  <->  Sahyog & Brothers Overseas Private Limited | D/7-1-617/ 615 And 637, Imperial Towers 7Th Floor Ameerpet, Hyderabad, Andhra Pradesh
 92.3 | Sasco Group of Companies Pvt. Ltd. | D-10, Anand Vihar Jagatpura, Jaipur, Rajasthan  <->  St Group Of Companies | #170 Kesar Symphony Plot 29 Sector 19 Khargar Navi Mumbai, Thane, Raigarh, MH
 92.0 | Sai Trading Private Limited | Basement Shop, Chandravilla Building, Cts No. 810, A/2, V B Phadke Marg, Mulun, D East, Mumbai, Mumbai City, Maharashtra  <->  Sri Parampara  Limited Private Trading | To 1982/1906/1, K R Nagar, Mysore, ಕರ್ನಾಟಕ
 91.4 | Sun Royal Producer Pvt Ltd | Qr No -F, 35/5C, R. P. Colony, Bhubaneswar, Khordha, Orissa  <->  JK  Producer Pvt Ltd | No.29, Miyami Beach, Venpurusham, Mahabalipuram, Kancheepuram, தமிழ்நாடு
 91.3 | Safe Physical Therapy LLC | NC, 1012 Ladera Drive, Waxhaw  <->  PHYSICAL THERAPY SAFE PHYSICIANS | 8950 OKETO AVENUE, MORTON GROVE, IL
 91.3 | As (India) Media Private Limited | Shubh Complex, 02Nd Floor, Room No. 9 Bypass Chowkdi, Timbavadi, Junagadh, Gujarat  <->  Maa Media Private  Limited | #32APOLLO STREET FORT, MUMBAI, Maharashtra
 91.2 | Next & Brothers Private Limited | 7 Deodar Street, Kolkata, Howrah, West Bengal  <->  Dr Alfa Limited Brothers Private & | A-15-2-10 GOWLINDA, HYDERABAD, A.P, Telangana
 91.2 | East Industries Private Limited | T/9 Radha Arcade Nr. Indira Gandhi Statue, Lambhvel Road, Anand, Gujarat  <->  Tech Industries Private Limited | 1125/15, 1St Floor, Service Road, Hampinagar, Bangalore North, ಕರ್ನಾಟಕ
 90.9 | Jay Technologies Limited | Sf.No.16, Coimbatore, 22/12 First Floor, Athipalayam Road Chinnavedampatti, Ganapathy, Tamil Nadu, Coimbatore  <->  Mayfair Technologies (India) Pdiravoote Limited | #241 300, Unnao, UP
 90.9 | Skr Road Private Limited | S No 35 Sanas Estate Vadgaonkh Sinhgad Rd Tal Haveli, Pune, Maharashtra  <->  Limited Basai Road Private Service | HR, Urban, Patel Nagar, Basai Road, 995/5
```

## 10. Scripts (Unicode block of non-ASCII chars, 300k-row sample per file)

- train_source1: name []
  addr [('UNNAMED', 200), ('LATIN', 101), ('INVERTED', 1), ('VULGAR', 1)]
- train_source2: name [('DEVANAGARI', 418918), ('TELUGU', 71135), ('KANNADA', 67811), ('TAMIL', 63493), ('BENGALI', 53044), ('GUJARATI', 48689), ('MALAYALAM', 35369), ('LATIN', 17530), ('ORIYA', 14290), ('GURMUKHI', 9848)]
  addr [('DEVANAGARI', 145609), ('BENGALI', 19070), ('TAMIL', 17901), ('TELUGU', 16569), ('KANNADA', 16100), ('GUJARATI', 11046), ('MALAYALAM', 4810), ('ORIYA', 2262), ('GURMUKHI', 2070), ('UNNAMED', 152)]
- train_source3: name [('DEVANAGARI', 221686), ('TELUGU', 38143), ('KANNADA', 34892), ('TAMIL', 33577), ('BENGALI', 27361), ('GUJARATI', 24510), ('LATIN', 19062), ('MALAYALAM', 18431), ('ORIYA', 6455), ('GURMUKHI', 5236)]
  addr [('DEVANAGARI', 133841), ('BENGALI', 18320), ('TAMIL', 17919), ('TELUGU', 16030), ('KANNADA', 15799), ('GUJARATI', 10674), ('MALAYALAM', 4390), ('GURMUKHI', 2265), ('ORIYA', 2022), ('UNNAMED', 134)]
- test_source1: name [('LATIN', 8299)]
  addr [('LATIN', 14729), ('RIGHT', 372), ('UNNAMED', 235), ('CENT', 1)]
- test_source2: name [('DEVANAGARI', 505120), ('KANNADA', 85412), ('TELUGU', 84449), ('TAMIL', 78077), ('BENGALI', 61271), ('GUJARATI', 58118), ('MALAYALAM', 44832), ('LATIN', 25432), ('ORIYA', 16892), ('GURMUKHI', 12777)]
  addr [('DEVANAGARI', 168819), ('BENGALI', 23120), ('TAMIL', 22581), ('TELUGU', 20557), ('KANNADA', 19341), ('GUJARATI', 13614), ('LATIN', 9292), ('MALAYALAM', 5460), ('ORIYA', 2664), ('GURMUKHI', 2525)]
- test_source3: name [('DEVANAGARI', 267620), ('TELUGU', 45422), ('KANNADA', 45075), ('TAMIL', 41397), ('BENGALI', 32520), ('GUJARATI', 30315), ('LATIN', 26843), ('MALAYALAM', 22478), ('ORIYA', 8822), ('GURMUKHI', 5918)]
  addr [('DEVANAGARI', 162070), ('TAMIL', 21159), ('BENGALI', 21040), ('TELUGU', 19568), ('KANNADA', 18634), ('GUJARATI', 13320), ('LATIN', 9516), ('MALAYALAM', 5445), ('ORIYA', 2562), ('DEGREE', 2105)]

## 11. Missing / format rates per file and country

### train_source1
```
               n  empty_name  empty_addr  addr_no_digit  name_nonascii
country                                                               
India     883188         0.0         0.0         0.0872            0.0
US       1323633         0.0         0.0         0.0000            0.0
```
### train_source2
```
               n  empty_name  empty_addr  addr_no_digit  name_nonascii
country                                                               
India    2017799         0.0      0.0287         0.0868         0.2787
US       3016817         0.0      0.0368         0.0980         0.0670
```
### train_source3
```
               n  empty_name  empty_addr  addr_no_digit  name_nonascii
country                                                               
India    2115547         0.0      0.0307         0.0994         0.1849
US       3170056         0.0      0.0350         0.0870         0.0680
```
### test_source1
```
              n  empty_name  empty_addr  addr_no_digit  name_nonascii
country                                                              
France   259452         0.0         0.0         0.0042         0.1572
India    809986         0.0         0.0         0.0873         0.0000
US       663106         0.0         0.0         0.0000         0.0000
```
### test_source2
```
               n  empty_name  empty_addr  addr_no_digit  name_nonascii
country                                                               
France    703378         0.0      0.0306         0.0685         0.2454
India    2312565         0.0      0.0228         0.0697         0.2761
US       1871330         0.0      0.0294         0.0795         0.0625
```
### test_source3
```
               n  empty_name  empty_addr  addr_no_digit  name_nonascii
country                                                               
France    731615         0.0      0.0294         0.0662         0.2394
India    2405000         0.0      0.0246         0.0812         0.1821
US       1945701         0.0      0.0284         0.0708         0.0639
```
## 12. Top 30 most frequent S1 names (lowercased)

### train
```
business_name
primary care group           253
ear nose & throat group      251
pediatric group              222
womens health group          220
physical therapy group       218
pediatric dental group       216
behavioral health group      215
chiropractic group           209
orthopedic group             208
eye group                    207
dental group                 205
family group                 204
dermatology group            202
vision group                 199
pediatric dentistry group    198
urgent care group            198
internal medicine group      197
foot & ankle group           191
oncology group               188
urology group                185
meridian                     182
cardiology group             175
surgical group               172
meridian llc                 159
family center                130
bureau of transportation     126
stag llc                     125
cedar llc                    121
womens health partners       118
board of public works        115
```
### test
```
business_name
bordeaux club sarl           205
nantes club sarl             157
lille club sarl              147
urgent care group            123
lille club sas               122
bordeaux club sas            118
nantes club sas              117
bordeaux amicale sarl        116
ear nose & throat group      112
internal medicine group      111
eye group                    110
pediatric dental group       110
pediatric group              109
primary care group           107
behavioral health group      107
urology group                106
orthopedic group             105
lille ecole sarl             100
chiropractic group           100
pediatric dentistry group     98
shree & co                    97
bordeaux comite sarl          97
bordeaux ecole sarl           97
bordeaux amicale sas          96
oncology group                96
dermatology group             95
nantes ecole sarl             93
family group                  93
cardiology group              92
dental group                  92
```
distinct lowercased S1 names train: 1538804 / 2206821

## Forks (defaults that change because of these facts)

1. **Singletons are rare (5.6%) and mean G is 3.46.** The all-empty floor is 0.056. The score is decided by getting the *extra* matches right (tau_2 / expected-F), not by the empty-vs-top-1 gate. The gate still matters for the 5.6% singletons plus the 5.4% G=1 entities.
2. **One-owner rule is exact** (0 violations over 7.64M matched IDs). Owner argmax is on by default; no flag needed for correctness.
3. **Zero cross-country pairs.** Blocking stays within country (config `blocking.within_country: true`).
4. **Names are weak, addresses are strong.** Name token_set_ratio 5th percentile is about 9 (India) and 53 (US). Some true pairs carry a random fake name (`Xylobrix`, `Jaxjax`) with the same address (the 30 lowest in the 200k sample have name similarity 0-4; share not measured yet). About 30% of S1 names are not unique (1.54M distinct / 2.21M), with chains like `urology group` and `lille ecole sarl`. **Blocking must use the address**: name-only char TF-IDF gave PC@15 = 0.68 on a probe, and name + address word TF-IDF gave 0.98.
5. **Nine Indic scripts in S2/S3 names** (28% of India S2 names are non-ASCII). S1 is Latin only. Transliteration is needed from v0 (`indic-transliteration`, MIT, with a Unicode-block offset to Devanagari, then IAST, then schwa drop).
6. **S1 has no empty addresses; S2/S3 have about 3% empty.** Treat empty as NaN (missing), never as a mismatch.
7. **Scale: 10.3M S2+S3 records per split.** LightGBM trains on a 20% S1 sample and OOF-predicts every pair; IDs are stored as int64 keys; the candidate file is validated in shards.
8. **S2 and S3 noise differ** (item 8): S3 US addresses are noisier (state names spelled out, median addr tsr 86 vs 93). Keep the `src` feature.
9. **Test mix differs from train:** India 47%, US 38%, France 15% (train: US 60%, India 40%). India is the weakest country in blocking (PC 0.948), so India work pays more on test.
