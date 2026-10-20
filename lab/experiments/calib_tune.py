"""Sweep 'Good' rule settings for the residual estimator and report honesty.

For each (confidence, min points, min coverage): % of runs rated Good, % of those Goods
whose true map-wide RMS error exceeded 25 m, and how bad the bad ones were.
Usage: python experiments/calib_tune.py [runs_per_case]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import calib_sim as cs  # noqa: E402
import trek_lab.calibration as cal  # noqa: E402
from trek_lab.calibration import Status  # noqa: E402

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
print(f"{'conf':>5} {'minN':>5} {'cov':>5} | {'Good% (all runs)':>16} {'bad Goods':>10} {'bad-Good true RMS p50/p90/max':>31}")
for conf in (0.9, 0.95):
    for min_n in (4, 5, 6):
        for cov in (0.15,):
            cal.RESID_CONF, cal.GOOD_MIN_POINTS, cal.GOOD_MIN_HULL_COVERAGE = conf, min_n, cov
            _, rows = cs.run(runs, seed=11)
            good = [r for r in rows if r.status_resid is Status.GOOD]
            bad = np.array([r.true_rms for r in good if r.true_rms > 25])
            dist = f"{np.median(bad):6.1f} /{np.percentile(bad, 90):6.1f} /{bad.max():6.1f}" if len(bad) else "-"
            print(f"{conf:5} {min_n:5} {cov:5} | {100 * len(good) / len(rows):16.1f} "
                  f"{len(bad):4}/{len(good):<5} {dist:>31}")
