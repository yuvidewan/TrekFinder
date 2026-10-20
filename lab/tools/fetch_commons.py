"""Download Wikimedia Commons maps listed in a JSON file, recording license and author.

Usage: python tools/fetch_commons.py corpus/images corpus/commons_meta.json corpus/commons_list.json

Images larger than 6000 px on the long edge are fetched as 6000 px renditions. Already
downloaded files are skipped, so the script can be re-run after a failure.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "TrekFinderCorpusBuilder/0.1 (research; github yuvidewan/TrekFinder)"}
API = "https://commons.wikimedia.org/w/api.php?"
MAX_EDGE = 6000


def get(url: str) -> bytes:
    for attempt in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 15 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"gave up on {url}")


def info(title: str, width: int | None = None) -> dict:
    p = {"action": "query", "format": "json", "titles": title, "prop": "imageinfo",
         "iiprop": "url|size|extmetadata"}
    if width:
        p["iiurlwidth"] = width
    d = json.loads(get(API + urllib.parse.urlencode(p)))
    return next(iter(d["query"]["pages"].values()))["imageinfo"][0]


def main(out_dir: str, meta_path: str, list_path: str) -> None:
    items = json.load(open(list_path, encoding="utf-8"))
    meta = {m["id"]: m for m in json.load(open(meta_path, encoding="utf-8"))} if os.path.exists(meta_path) else {}
    strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
    for it in items:
        if it["id"] in meta and os.path.exists(os.path.join(os.path.dirname(meta_path), meta[it["id"]]["file"])):
            continue
        ii = info(it["title"])
        w, h = ii["width"], ii["height"]
        url = ii["url"]
        if max(w, h) > MAX_EDGE:
            url = info(it["title"], width=int(MAX_EDGE * min(1, w / h)))["thumburl"]
        ext = ".png" if url.split("?")[0].lower().endswith(".png") else ".jpg"
        fn = it["id"] + ext
        data = get(url)
        with open(os.path.join(out_dir, fn), "wb") as f:
            f.write(data)
        m = ii["extmetadata"]
        meta[it["id"]] = {**it, "file": "images/" + fn, "source": ii["descriptionurl"],
                          "license": m.get("LicenseShortName", {}).get("value"),
                          "author": strip(m.get("Artist", {}).get("value"))[:200],
                          "original_size": [w, h]}
        json.dump(list(meta.values()), open(meta_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(fn, len(data), meta[it["id"]]["license"])
        time.sleep(5)


if __name__ == "__main__":
    main(*sys.argv[1:4])
