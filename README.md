# TrekFinder

Offline trekking app that turns a photo of any map into a live navigation map that keeps the original's look.

- Product plan and decision record: [`docs/trek_map_app_product_plan.pdf`](docs/trek_map_app_product_plan.pdf) (source: `docs/trek_map_app_product_plan.html`)
- Current stage: **Phase 0 — feasibility prototype** (iPhone first, Android later)

## Layout

| Folder | What |
|---|---|
| `docs/` | Product plan, decision record, Phase 0 reports |
| `lab/` | Python + OpenCV algorithm lab (calibration, image prep, feature extraction experiments) |
| `app/` | Flutter app (Phase 0 device test app) |
| `scripts/` | Setup and build helpers (incl. one-command Mac setup, later) |

## Lab quick start (Windows)

```bash
cd lab
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
```
