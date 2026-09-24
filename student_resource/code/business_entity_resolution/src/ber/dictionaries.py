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
