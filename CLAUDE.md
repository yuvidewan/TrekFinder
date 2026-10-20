# TrekFinder — notes for Claude

- **Git: never commit or push** unless the user explicitly says so at that moment. Give a ready-to-use commit message instead; the user runs git. Read-only git commands are fine.

- Source of truth for product decisions: `docs/trek_map_app_product_plan.html` (section 18A, "Decision record"). Regenerate the PDF with headless Chrome `--print-to-pdf` after editing; keep `_v1.pdf` untouched.
- Guiding rule: never confuse visual resemblance with geographic accuracy. No generative models in the map pipeline.
- Platform priority: **iPhone first** (user's main device; builds happen on a friend's Mac). Android later. User develops on Windows.
- Phase 0 approach: algorithms are prototyped in Python + OpenCV under `lab/`, winners ported to the C++ core later.
- Calibration maths lives in `lab/trek_lab/calibration.py`: fits in a local east/north plane (metres), never in lat/lon; pixel y points down, so it is flipped before similarity fits.
- **When iOS device work starts:** write `scripts/setup_mac.sh` — a one-command setup for the friend's Mac (check Xcode, install Homebrew/Flutter/CocoaPods, clone/pull repo, `pod install`, `flutter doctor`, open Xcode) plus a one-command update/run script. The friend is not a developer; keep it zero-effort.
