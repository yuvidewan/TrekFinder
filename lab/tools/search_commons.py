import json, sys, urllib.parse, urllib.request
UA = {"User-Agent": "TrekFinderCorpusBuilder/0.1 (research; contact via github yuvidewan/TrekFinder)"}
def search(q, n=8):
    p = {"action":"query","format":"json","generator":"search","gsrsearch":q,"gsrnamespace":6,"gsrlimit":n,
         "prop":"imageinfo","iiprop":"url|size|extmetadata|mime"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p)
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))
    for pg in (d.get("query",{}).get("pages",{}) or {}).values():
        ii = pg["imageinfo"][0]; m = ii.get("extmetadata",{})
        lic = m.get("LicenseShortName",{}).get("value","?")
        if ii["mime"] not in ("image/jpeg","image/png","image/tiff"): continue
        print(f'{ii["width"]}x{ii["height"]} | {lic} | {pg["title"]} | {ii["url"]}')
for q in sys.argv[1:]:
    print("==", q); search(q)
