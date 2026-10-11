#!/usr/bin/env python3
"""
Result-side analysis for the Geo-CVAE VTC paper.

Covers four of the Tier-1 experiments. Each sub-command reads a CSV your
training loop writes and emits both a console summary and a LaTeX fragment
ready to \\input{} into the paper.

    seeds      mean +/- std and paired tests across seeds        -> Table I
    cells      sparse vs dense partition-class breakdown         -> Table III
    partition  per-partition BWT                                 -> Table IV
    ablation   ablation arms vs full method                      -> Table V

Expected input schemas
----------------------
seeds.csv      method,seed,aa,bwt
cells.csv      method,seed,partition,cls,n_users,acc_first,acc_final
ablation.csv   arm,seed,aa,bwt

`cells.csv` is the important one: one row per (partition, class) cell per seed,
with accuracy at the round the cell was first learned and at the final round.
Everything else (per-partition BWT, sparse/dense split, cell-level BWT) is
derived from it, so if your loop writes only this file you can produce three of
the four tables.

Usage
    python3 analyze.py seeds     results/seeds.csv     --out tab_main.tex
    python3 analyze.py cells     results/cells.csv     --out tab_sparse.tex --threshold 3
    python3 analyze.py partition results/cells.csv     --out tab_partition.tex
    python3 analyze.py ablation  results/ablation.csv  --out tab_ablation.tex --full geo_cvae
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- io helpers

def read_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path}: no rows")
    return rows


def as_float(row, key, path):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        sys.exit(f"{path}: row missing or malformed column '{key}': {row}")


def fmt_pm(mean, std, places=1):
    return f"{mean:.{places}f} $\\pm$ {std:.{places}f}"


def write_out(text, out):
    if out:
        with open(out, "w") as fh:
            fh.write(text)
        print(f"\n[written] {out}")
    else:
        print("\n" + text)


# ------------------------------------------------------------ paired testing

def paired_test(a, b):
    """Paired comparison on seed-matched vectors.

    Returns (test_name, statistic, p). Wilcoxon needs n>=6 to reach p<0.05 at
    all, so below that we report the paired t-test and say so rather than
    quoting a test that cannot resolve the question.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2:
        return ("none", float("nan"), float("nan"))
    if n >= 6:
        try:
            s, p = stats.wilcoxon(a, b)
            return ("Wilcoxon signed-rank", s, p)
        except ValueError:
            pass
    s, p = stats.ttest_rel(a, b)
    return ("paired t", s, p)


def cohens_dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float("nan") if sd == 0 else d.mean() / sd


# ----------------------------------------------------------------- seed stats

def cmd_seeds(args):
    rows = read_csv(args.path)
    by = defaultdict(lambda: {"seed": [], "aa": [], "bwt": []})
    for r in rows:
        m = r["method"].strip()
        by[m]["seed"].append(r["seed"])
        by[m]["aa"].append(as_float(r, "aa", args.path))
        by[m]["bwt"].append(as_float(r, "bwt", args.path))

    counts = {m: len(v["aa"]) for m, v in by.items()}
    n = min(counts.values())
    print(f"methods: {len(by)}   seeds per method: {counts}")
    if len(set(counts.values())) != 1:
        print("! unequal seed counts; paired tests use seed-matched rows only")
    if n < 5:
        print(f"! n={n} seeds. 5-10 recommended before claiming a margin.")

    lines = []
    for m, v in by.items():
        aa, bwt = np.array(v["aa"]), np.array(v["bwt"])
        lines.append((m, aa.mean(), aa.std(ddof=1) if len(aa) > 1 else 0.0,
                      bwt.mean(), bwt.std(ddof=1) if len(bwt) > 1 else 0.0))
        print(f"  {m:22s} AA {aa.mean():6.2f} +/- {aa.std(ddof=1) if len(aa)>1 else 0:4.2f}"
              f"   BWT {bwt.mean():+.4f} +/- {bwt.std(ddof=1) if len(bwt)>1 else 0:.4f}")

    # paired comparisons against the reference arm
    ref = args.reference
    if ref and ref in by:
        print(f"\npaired comparisons vs '{ref}' (seed-matched):")
        ref_by_seed = dict(zip(by[ref]["seed"], zip(by[ref]["aa"], by[ref]["bwt"])))
        for m, v in by.items():
            if m == ref:
                continue
            pairs = [(a, b, ref_by_seed[s])
                     for s, a, b in zip(v["seed"], v["aa"], v["bwt"]) if s in ref_by_seed]
            if len(pairs) < 2:
                print(f"  {m:22s} too few matched seeds")
                continue
            aa_m = [p[0] for p in pairs]
            aa_r = [p[2][0] for p in pairs]
            bwt_m = [p[1] for p in pairs]
            bwt_r = [p[2][1] for p in pairs]
            for label, x, y in (("AA", aa_m, aa_r), ("BWT", bwt_m, bwt_r)):
                name, _, p = paired_test(x, y)
                d = cohens_dz(x, y)
                delta = np.mean(x) - np.mean(y)
                verdict = "" if np.isnan(p) else ("  <-- not separated" if p >= 0.05 else "")
                print(f"  {m:22s} {label:3s} delta {delta:+7.3f}  {name} p={p:.4f}"
                      f"  dz={d:+.2f}{verdict}")

    tex = [
        "% generated by analyze.py seeds",
        "\\begin{tabular}{lcc}", "\\toprule",
        "\\textbf{Method} & \\textbf{AA (\\%)} & \\textbf{BWT} \\\\", "\\midrule",
    ]
    for m, am, asd, bm, bsd in lines:
        tex.append(f"{m} & {fmt_pm(am, asd, 1)} & {fmt_pm(bm, bsd, 3)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            f"% n={n} seeds per arm."]
    write_out("\n".join(tex), args.out)


# ------------------------------------------------- cell-level / sparse vs dense

def load_cells(path):
    """Load per-cell results, normalising accuracy to percent exactly once.

    Accuracies may be written as fractions (0-1) or percent (0-100). Detect
    which from the global maximum and scale here, so every downstream
    computation works in the same units. Scaling at each use site instead is
    how the two different numbers for the same quantity arise.
    """
    rows = read_csv(path)
    out = []
    for r in rows:
        out.append({
            "method": r["method"].strip(),
            "seed": r["seed"],
            "partition": r["partition"].strip(),
            "cls": r["cls"].strip(),
            "n_users": int(float(r["n_users"])),
            "first": as_float(r, "acc_first", path),
            "final": as_float(r, "acc_final", path),
        })
    hi = max(max(c["first"], c["final"]) for c in out)
    if hi <= 1.0:
        for c in out:
            c["first"] *= 100.0
            c["final"] *= 100.0
    return out


def cell_bwt(c):
    """Cell-level backward transfer, as a fraction.

    load_cells normalises accuracy to percent, but the paper reports BWT on
    the fractional scale (-0.041, not -4.1), so divide back here to keep the
    harness and the manuscript in the same units.
    """
    return (c["final"] - c["first"]) / 100.0


def cmd_cells(args):
    cells = load_cells(args.path)
    thr = args.threshold

    groups = defaultdict(lambda: defaultdict(list))   # method -> group -> [(aa, bwt)]
    for c in cells:
        g = "sparse" if c["n_users"] < thr else "dense"
        groups[c["method"]][g].append((c["final"], cell_bwt(c)))

    n_sparse = sum(1 for c in cells if c["n_users"] < thr) / max(len(set(c["seed"] for c in cells)), 1)
    n_dense = sum(1 for c in cells if c["n_users"] >= thr) / max(len(set(c["seed"] for c in cells)), 1)
    total = n_sparse + n_dense
    print(f"cell split at n_users < {thr}: "
          f"{n_sparse:.0f} sparse / {n_dense:.0f} dense ({100*n_sparse/total:.0f}% sparse)")

    tex = [
        "% generated by analyze.py cells",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Sparse cells}} & \\multicolumn{2}{c}{\\textbf{Dense cells}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "\\textbf{Method} & \\textbf{AA (\\%)} & \\textbf{BWT} & \\textbf{AA (\\%)} & \\textbf{BWT} \\\\",
        "\\midrule",
    ]
    print()
    for m, gs in groups.items():
        row = [m]
        for g in ("sparse", "dense"):
            vals = gs.get(g, [])
            if not vals:
                row += ["--", "--"]
                continue
            aa = np.array([v[0] for v in vals])
            bw = np.array([v[1] for v in vals])
            row += [f"{aa.mean():.1f}", f"{bw.mean():+.3f}"]
            print(f"  {m:22s} {g:6s} AA {aa.mean():6.2f}  BWT {bw.mean():+.4f}  (n={len(vals)})")
        tex.append(" & ".join(row) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            f"% sparse = fewer than {thr} contributing users."]

    # the claim under test: is the margin larger on sparse cells?
    if args.compare and args.reference:
        a, b = args.compare, args.reference
        if a in groups and b in groups:
            print(f"\nmargin of '{a}' over '{b}':")
            for g in ("sparse", "dense"):
                va, vb = groups[a].get(g, []), groups[b].get(g, [])
                if va and vb:
                    da = np.mean([v[0] for v in va]) - np.mean([v[0] for v in vb])
                    print(f"  {g:6s} AA margin {da:+.2f} pts")
            print("  (a larger sparse margin supports the paper's explanation;"
                  " a flat or reversed one refutes it)")

    write_out("\n".join(tex), args.out)


def cmd_partition(args):
    cells = load_cells(args.path)
    per = defaultdict(lambda: defaultdict(list))   # method -> partition -> [bwt]
    for c in cells:
        per[c["method"]][c["partition"]].append(cell_bwt(c))

    parts = sorted({c["partition"] for c in cells})
    print(f"{len(parts)} partitions, {len(per)} methods")

    tex = [
        "% generated by analyze.py partition",
        "\\begin{tabular}{l" + "c" * len(per) + "}", "\\toprule",
        "\\textbf{Partition} & " + " & ".join(f"\\textbf{{{m}}}" for m in per) + " \\\\",
        "\\midrule",
    ]
    for p in parts:
        tex.append(f"{p} & " + " & ".join(
            f"{np.mean(per[m][p]):+.3f}" if per[m].get(p) else "--" for m in per) + " \\\\")

    # dispersion across partitions is the fairness signal
    tex.append("\\midrule")
    tex.append("Std.\\ across partitions & " + " & ".join(
        f"{np.std([np.mean(v) for v in per[m].values()], ddof=1):.3f}" for m in per) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]

    print("\nBWT dispersion across partitions (lower = forgetting spread more evenly):")
    for m in per:
        means = [np.mean(v) for v in per[m].values()]
        print(f"  {m:22s} mean {np.mean(means):+.4f}  std {np.std(means, ddof=1):.4f}"
              f"  worst {min(means):+.4f}")
    write_out("\n".join(tex), args.out)


# -------------------------------------------------------------------- ablation

def cmd_ablation(args):
    rows = read_csv(args.path)
    by = defaultdict(lambda: {"seed": [], "aa": [], "bwt": []})
    for r in rows:
        a = r["arm"].strip()
        by[a]["seed"].append(r["seed"])
        by[a]["aa"].append(as_float(r, "aa", args.path))
        by[a]["bwt"].append(as_float(r, "bwt", args.path))

    full = args.full
    if full not in by:
        sys.exit(f"full arm '{full}' not in {sorted(by)}")

    ref_seed = dict(zip(by[full]["seed"], zip(by[full]["aa"], by[full]["bwt"])))
    f_aa, f_bwt = np.mean(by[full]["aa"]), np.mean(by[full]["bwt"])
    print(f"full method '{full}': AA {f_aa:.2f}  BWT {f_bwt:+.4f}\n")

    tex = [
        "% generated by analyze.py ablation",
        "\\begin{tabular}{lccc}", "\\toprule",
        "\\textbf{Variant} & \\textbf{AA (\\%)} & \\textbf{$\\Delta$AA} & \\textbf{BWT} \\\\",
        "\\midrule",
    ]
    tex.append(f"{full} (full) & {f_aa:.1f} & -- & {f_bwt:+.3f} \\\\")
    for a, v in by.items():
        if a == full:
            continue
        aa, bwt = np.mean(v["aa"]), np.mean(v["bwt"])
        d = aa - f_aa
        pairs = [(x, ref_seed[s][0]) for s, x in zip(v["seed"], v["aa"]) if s in ref_seed]
        if len(pairs) >= 2:
            name, _, p = paired_test([x for x, _ in pairs], [y for _, y in pairs])
            sig = f"{name} p={p:.4f}"
            flag = "  <-- component not shown to contribute" if p >= 0.05 else ""
        else:
            sig, flag = "n/a", ""
        print(f"  {a:28s} AA {aa:6.2f}  dAA {d:+6.2f}  BWT {bwt:+.4f}  {sig}{flag}")
        tex.append(f"{a} & {aa:.1f} & {d:+.1f} & {bwt:+.3f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_out("\n".join(tex), args.out)


# ------------------------------------------------------------------------ main

def cmd_retention(args):
    """Old-class vs new-class accuracy -- where continual learning lives.

    AA averages over every cell, so a method that recovers old classes while
    giving up a little on new ones can look flat in AA while being far better
    at the thing the paper is about. Report this beside AA and BWT; a
    forgetting claim that only cites AA is not evidence about forgetting.
    """
    cells = load_cells(args.path)
    old = {int(c) for c in args.old_classes.split(",") if c != ""}
    by = defaultdict(lambda: {"old": [], "new": []})
    for c in cells:
        by[c["method"]]["old" if int(c["cls"]) in old else "new"].append(c["final"])

    print(f"old classes = {sorted(old)}   (everything else counts as new)\n")
    print(f"{'Method':26s} {'Old %':>8} {'New %':>8} {'All %':>8}")
    print("-" * 54)
    tex = ["% generated by analyze.py retention",
           "\\begin{tabular}{lccc}", "\\toprule",
           "\\textbf{Method} & \\textbf{Old-class AA} & \\textbf{New-class AA} "
           "& \\textbf{Overall AA} \\\\", "\\midrule"]
    for m, g in by.items():
        o = np.mean(g["old"]) if g["old"] else float("nan")
        n = np.mean(g["new"]) if g["new"] else float("nan")
        a = np.mean(g["old"] + g["new"])
        print(f"{m:26s} {o:8.2f} {n:8.2f} {a:8.2f}")
        tex.append(f"{m} & {o:.1f} & {n:.1f} & {a:.1f} \\\\")
        if o < 1.0:
            print(f"{'':26s}   ^ old-class accuracy is ~0: this method is not "
                  f"retaining anything, whatever its BWT says")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_out("\n".join(tex), args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seeds", help="mean/std + paired tests across seeds")
    s.add_argument("path"); s.add_argument("--out")
    s.add_argument("--reference", default="Lightweight AC-GAN",
                   help="arm to run paired comparisons against")
    s.set_defaults(func=cmd_seeds)

    c = sub.add_parser("cells", help="sparse vs dense cell breakdown")
    c.add_argument("path"); c.add_argument("--out")
    c.add_argument("--threshold", type=int, default=3,
                   help="n_users below this counts as a sparse cell")
    c.add_argument("--compare", default="Geo-CVAE")
    c.add_argument("--reference", default="Lightweight AC-GAN")
    c.set_defaults(func=cmd_cells)

    q = sub.add_parser("partition", help="per-partition BWT and dispersion")
    q.add_argument("path"); q.add_argument("--out")
    q.set_defaults(func=cmd_partition)

    t = sub.add_parser("retention", help="old-class vs new-class accuracy")
    t.add_argument("path"); t.add_argument("--out")
    t.add_argument("--old-classes", default="0,1,2,3",
                   help="comma-separated classes that arrived in earlier stages")
    t.set_defaults(func=cmd_retention)

    a = sub.add_parser("ablation", help="ablation arms vs full method")
    a.add_argument("path"); a.add_argument("--out")
    a.add_argument("--full", default="geo_cvae_full")
    a.set_defaults(func=cmd_ablation)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
