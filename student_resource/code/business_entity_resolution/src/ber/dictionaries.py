# every hand-written map lives here (domain knowledge, disclosed in the doc)

# legal-form tokens -> canonical class token (v0; phase 2 adds multi-token and France forms)
LEGAL = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "llc": "llc", "llp": "llp", "lp": "lp", "pc": "pc", "plc": "plc", "pllc": "pllc",
    "ltd": "ltd", "limited": "ltd", "limitet": "ltd", "limirrad": "ltd",
    "pvt": "pvt", "private": "pvt", "praivet": "pvt", "piraivet": "pvt", "praivarr": "pvt",
    "co": "co", "company": "co",
    "opc": "opc",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "snc": "snc",
    "sci": "sci", "scop": "scop", "selarl": "selarl",
}

# name tokens with no identity signal (v0)
NAME_STOP = {"the", "com", "www", "net", "org"}

# Malayalam chillu letters -> base consonant + virama, so the offset map works
CHILLU = {"ൺ": "ണ്", "ൻ": "ന്", "ർ": "ര്",
          "ൽ": "ല്", "ൾ": "ള്", "ൿ": "ക്"}

# Devanagari candra vowels (English "o"/"e" loan sounds) -> plain vowels
CANDRA = {"ॉ": "ो", "ऑ": "ओ", "ॅ": "े", "ऍ": "ए"}

# ---- v1 (phase 2) ----

# DBA / trade-name separators (on folded text); left and right both kept in dba_parts
DBA = r"\b(?:dba|d b a|doing business as|trading as|aka|a k a|also known as)\b"

# multi-word place names -> short code (applied before token maps; both sides get the same code)
PLACE_PHRASES = {
    "district of columbia": "dc", "new york": "ny", "new jersey": "nj", "new mexico": "nm",
    "new hampshire": "nh", "north carolina": "nc", "south carolina": "sc", "north dakota": "nd",
    "south dakota": "sd", "west virginia": "wv", "rhode island": "ri",
    "uttar pradesh": "up", "uttar prades": "up", "madhya pradesh": "mp", "madhya prades": "mp",
    "andhra pradesh": "ap", "andhra prades": "ap", "himachal pradesh": "hp", "himachal prades": "hp",
    "arunachal pradesh": "ar", "west bengal": "wb", "paschim bang": "wb", "pascim bang": "wb",
    "tamil nadu": "tn", "tamil natu": "tn", "jammu and kashmir": "jk",
}

# single address tokens -> canonical short form (never expand an ambiguous short form)
ADDR_TOKENS = {
    # US states
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    # India states (English + transliterated forms seen in S2/S3)
    "maharashtra": "mh", "maharastr": "mh", "maharashtr": "mh", "delhi": "dl", "dilli": "dl",
    "karnataka": "ka", "karnatak": "ka", "kerala": "kl", "keralam": "kl", "keral": "kl",
    "gujarat": "gj", "gujrat": "gj", "rajasthan": "rj", "haryana": "hr", "punjab": "pb", "panjab": "pb",
    "bihar": "br", "odisha": "or", "orissa": "or", "odisa": "or", "telangana": "tg", "telangan": "tg",
    "ts": "tg", "jharkhand": "jh", "chhattisgarh": "cg", "chattisgarh": "cg", "uttarakhand": "uk",
    "assam": "as", "tamilnadu": "tn", "tamilnatu": "tn", "pascimabang": "wb", "paschimbang": "wb",
    # street types and units
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "boulevard": "blvd", "boul": "blvd",
    "bd": "blvd", "place": "pl", "parkway": "pkwy", "highway": "hwy", "trail": "trl", "terrace": "ter",
    "square": "sq", "apartment": "apt", "suite": "ste", "floor": "fl", "flr": "fl", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "nr": "near", "opposite": "opp",
    # ordinal words
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    # France (short forms kept; "r" -> rue only for France, see ADDR_FR)
    "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte", "faubourg": "fbg",
}
ADDR_FR = {"r": "rue"}

# address tokens with no location signal
ADDR_STOP = {"null", "cedex", "no", "number", "india", "usa", "us", "france"}

# sound-alike skeleton: letter -> consonant class; vowels, h, w, y dropped (transliteration-robust)
SKEL = {**{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"}, **{c: "3" for c in "dt"},
        "l": "4", "m": "5", "n": "5", "r": "6", **{c: "" for c in "aeiouhwy"}}

# ---- v2 (round-2 name features only; normalization v1 unchanged) ----
# multilingual function words: EN + FR (+ elided l'/d' become "l"/"d" after punctuation folding)
NAME_STOP2 = {"and", "of", "de", "du", "la", "le", "les", "des", "et", "l", "d", "en", "a", "au", "aux"}
