# Phase 0 map corpus

Phase 0 uses a mini corpus of ~12 maps (2 per category). It grows to ~60 for the Phase 1 acceptance bar (plan section 14, 18A.3).

Images go in `images/` (git-ignored: they can be large and some are personal). Each map gets an entry in `manifest.json` so results are reproducible.

## Categories

| # | Category | Phase 0 target | Good sources |
|---|---|---|---|
| 1 | Gridded topographic | 2 | USGS US Topo / historical topo (public domain, has lat/lon grid — ideal for measuring calibration error) |
| 2 | Scale bar + north arrow trail map | 2 | US National Park Service maps (public domain), park brochures |
| 3 | Guidebook PDF page | 2 | Any guidebook/trail PDF you own |
| 4 | Phone photo with glare/folds/perspective | 2 | Photograph a printed/paper map yourself, at an angle, in uneven light |
| 5 | Illustrated / cartoon map | 2 | Theme-park/resort/trail illustrated maps you own or photograph |
| 6 | Hand-drawn sketch map | 2 | Draw one of a walk you know, photograph it |

## Ground truth per map (in `manifest.json`)

- `control_points`: pixel position + real lat/lon for features you can identify (grid intersections, summits, junctions). At least 6, spread across the map. These are used to measure calibration error.
- `trails` (optional in Phase 0): polylines in pixel coordinates for the main trails, used to score trail detection.

Only use maps you have the right to use. Do not commit copyrighted images.
