"""Merge tool outputs (commons_meta.json, tiles_meta.json) into corpus/manifest.json.

Entries are matched by id; hand-edited fields already in the manifest (e.g. control
points) are kept. Entries whose image file no longer exists are dropped.
Usage: python tools/build_manifest.py corpus
"""

import json
import os
import sys


def main(corpus: str) -> None:
    path = os.path.join(corpus, "manifest.json")
    manifest = json.load(open(path, encoding="utf-8"))
    by_id = {m["id"]: m for m in manifest["maps"]}
    for name in ("commons_meta.json", "tiles_meta.json"):
        p = os.path.join(corpus, name)
        if not os.path.exists(p):
            continue
        for m in json.load(open(p, encoding="utf-8")):
            m.pop("bytes", None)
            entry = by_id.setdefault(m["id"], {"control_points": []})
            entry.update({k: v for k, v in m.items() if k != "control_points"})
    maps = [m for m in by_id.values() if os.path.exists(os.path.join(corpus, m["file"]))]
    maps.sort(key=lambda m: (m.get("category", ""), m["id"]))
    manifest["maps"] = maps
    json.dump(manifest, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{len(maps)} maps in manifest")
    for cat in sorted({m.get("category", "?") for m in maps}):
        print(f"  {cat:22} {sum(m.get('category') == cat for m in maps)}")


if __name__ == "__main__":
    main(sys.argv[1])
