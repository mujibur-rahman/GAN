#!/usr/bin/env python3
"""
Per-seed test of spatial fairness: is forgetting spread more evenly?

    python3 fairness_test.py r_abl/ablation_cells.csv --a geo_cvae_input --b no_geo_input
    python3 fairness_test.py r_main/cells.csv --a "Geo-CVAE (input)" --b "Lightweight AC-GAN"

analyze.py's `partition` command pools every seed before taking the standard
deviation across partitions, which gives one number per method and no way to
test it. This computes the dispersion WITHIN each seed, giving five paired
observations, then runs a paired test on them.

Three statistics per seed:
  std      spread of per-partition BWT -- the fairness measure
  worst    the single worst-hit partition
  mean     average per-partition BWT, for reference

A method can match another on mean BWT and still concentrate all its
forgetting in a few regions. For spatial crowdsourcing that distinction
matters: workers in those regions get a model that has forgotten their area.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy import stats


def per_seed_dispersion(path, method):
    """Return {seed: (std, worst, mean)} of per-partition BWT."""
    rows = defaultdict(lambda: defaultdict(list))   # seed -> partition -> [bwt]
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["method"].strip() != method:
                continue
            bwt = (float(r["acc_final"]) - float(r["acc_first"]))
            rows[r["seed"]][r["partition"]].append(bwt)
    out = {}
    for seed, parts in rows.items():
        means = np.array([np.mean(v) for v in parts.values()])
        if len(means) >= 2:
            out[seed] = (float(means.std(ddof=1)), float(means.min()),
                         float(means.mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--a", required=True, help="method under test")
    ap.add_argument("--b", required=True, help="comparison method")
    args = ap.parse_args()

    A = per_seed_dispersion(args.path, args.a)
    B = per_seed_dispersion(args.path, args.b)
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 2:
        raise SystemExit(f"only {len(seeds)} matched seeds; check the method names")

    print(f"{args.a}  vs  {args.b}      ({len(seeds)} matched seeds)\n")
    labels = ["dispersion (std)", "worst partition", "mean BWT"]
    better = ["lower", "higher", "higher"]
    for i, (lab, dirn) in enumerate(zip(labels, better)):
        a = np.array([A[s][i] for s in seeds])
        b = np.array([B[s][i] for s in seeds])
        t, p = stats.ttest_rel(a, b)
        d = a - b
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
        win = (a.mean() < b.mean()) if dirn == "lower" else (a.mean() > b.mean())
        mark = "*" if p < 0.05 else " "
        print(f"  {lab:18s} {a.mean():+.4f} vs {b.mean():+.4f}   "
              f"delta {a.mean()-b.mean():+.4f}  p={p:.4f}{mark}  dz={dz:+.2f}"
              f"   {'FAVOURS ' + args.a if win else 'favours ' + args.b}")
        print(f"  {'':18s} per-seed: {np.round(a,3)}")
        print(f"  {'':18s}           {np.round(b,3)}\n")

    print("Lower dispersion means forgetting is spread evenly across regions")
    print("rather than concentrated in a few. Report it beside BWT: two methods")
    print("with the same mean BWT can differ a lot in who bears the loss.")


if __name__ == "__main__":
    main()
