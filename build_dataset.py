#!/usr/bin/env python3
"""
Build and validate a real dataset before spending days of compute on it.

    python3 build_dataset.py --dataset geolife --data-path ~/data/Geolife_Trajectories_1.3
    python3 build_dataset.py --dataset nyc_taxi --data-path ~/data/tlc/

Extracts features once, caches to an .npz, and prints a validation report.
Re-running reads the cache, so later runs start instantly.

The report answers four questions, in order of how much they can cost you:

  1. Is the task learnable at all?            (centralised upper bound)
  2. How many cells, and how many are sparse? (does the sparse-cell claim hold)
  3. Do partitions actually differ?           (CAN geo-conditioning help?)
  4. Is forgetting measurable?                (do the stages carry real shifts)

Question 3 is the one worth running before anything else. Geo-conditioning can
only help if the class-conditional feature distribution genuinely varies by
partition. If it does not, E1 is decided before you run it, and the honest move
is to reframe rather than to keep tuning.
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
import data as D


# --------------------------------------------------------------------- cache

def cache_path(cfg):
    return os.path.join(cfg.out_dir, f"{cfg.data.dataset}_cache.npz")


def save_cache(ds, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, X=ds.X, y=ds.y, part=ds.part, client=ds.client,
                        n_partitions=ds.n_partitions, n_classes=ds.n_classes,
                        name=ds.name)
    print(f"[cache] wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    return D.Dataset(z["X"], z["y"], z["part"], z["client"],
                     int(z["n_partitions"]), int(z["n_classes"]), str(z["name"]))


# ---------------------------------------------------------------- diagnostics

def report_basic(ds):
    print("\n=== 1. shape ===")
    print(f"samples      {len(ds):,}")
    print(f"features     {ds.X.shape[1]}")
    print(f"classes      {ds.n_classes}   counts {np.bincount(ds.y, minlength=ds.n_classes)}")
    print(f"partitions   {ds.n_partitions}")
    print(f"clients      {len(np.unique(ds.client))}")
    bal = np.bincount(ds.y, minlength=ds.n_classes) / len(ds)
    if bal.max() > 0.5:
        print(f"! one class holds {bal.max():.0%} of samples. Report per-class "
              f"metrics, not just accuracy, or the majority class will carry it.")
    if not np.isfinite(ds.X).all():
        print("! non-finite feature values present -- fix before training")


def report_learnable(ds, seed=0):
    """Centralised upper bound. Everything federated sits below this."""
    print("\n=== 2. is it learnable ===")
    try:
        import torch
        import torch.nn.functional as F
        from models import TaskModel
    except ImportError:
        print("(torch not available, skipped)")
        return None
    cfg = Config()
    cfg.model.in_dim = ds.X.shape[1]
    tr, te = ds.split(0.25, seed)
    torch.manual_seed(seed)
    m = TaskModel(cfg, ds.n_classes)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    X, y = torch.tensor(tr.X), torch.tensor(tr.y)
    for _ in range(30):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(torch.tensor(te.X)).argmax(-1).numpy() == te.y).mean())
    chance = 1.0 / ds.n_classes
    print(f"centralised 30-epoch test accuracy  {acc*100:.1f}%   (chance {chance*100:.1f}%)")
    if acc < chance * 1.5:
        print("! barely above chance. The features do not separate the classes; "
              "fix feature extraction before any FL experiment.")
    else:
        print(f"  federated runs will land below this. Treat {acc*100:.0f}% as the ceiling.")
    return acc


def report_cells(ds, threshold=3):
    """Cell occupancy -- the sparse-cell claim in Section VII-A."""
    print("\n=== 3. cell occupancy ===")
    counts = ds.cell_user_counts()
    occupied = {k: v for k, v in counts.items() if v > 0}
    sparse = {k: v for k, v in occupied.items() if v < threshold}
    total_cells = ds.n_partitions * ds.n_classes
    print(f"cells in the grid        {total_cells}")
    print(f"cells with any data      {len(occupied)}")
    print(f"sparse (<{threshold} users)        {len(sparse)}  "
          f"({100*len(sparse)/max(len(occupied),1):.0f}% of occupied)")
    if occupied:
        v = np.array(list(occupied.values()))
        print(f"users per occupied cell  median {np.median(v):.0f}  max {v.max()}")
    frac = len(sparse) / max(len(occupied), 1)
    if frac < 0.15:
        print(f"! only {frac:.0%} sparse. The paper attributes its margin to sparse "
              f"cells; with this few, that explanation has little room to show up.")
    elif frac > 0.75:
        print(f"! {frac:.0%} sparse. Most cells have almost no data, so per-cell "
              f"accuracy will be dominated by noise. Consider merging partitions.")
    return frac


def report_partition_shift(ds):
    """Do partitions actually change the class-conditional distribution?

    This bounds what geo-conditioning can possibly buy. For each class, compare
    the spread of per-partition feature means (between-partition) against the
    within-partition spread. A ratio near zero means every partition looks the
    same given the class, so conditioning on partition adds no information and
    the E1 ablation is decided in advance.

    Reported as eta^2-like: between / (between + within), in [0, 1].
    """
    print("\n=== 4. do partitions differ (bounds what geo-conditioning can buy) ===")
    ratios = []
    for c in range(ds.n_classes):
        mc = ds.y == c
        if mc.sum() < 30:
            continue
        Xc = ds.X[mc]
        pc = ds.part[mc]
        parts = [p for p in np.unique(pc) if (pc == p).sum() >= 10]
        if len(parts) < 2:
            continue
        grand = Xc.mean(0)
        between, within, n_tot = 0.0, 0.0, 0
        for p in parts:
            Xp = Xc[pc == p]
            between += len(Xp) * float(((Xp.mean(0) - grand) ** 2).sum())
            within += float(((Xp - Xp.mean(0)) ** 2).sum())
            n_tot += len(Xp)
        if between + within > 0:
            ratios.append(between / (between + within))
    if not ratios:
        print("(not enough data per class/partition to measure)")
        return None
    eta = float(np.mean(ratios))
    print(f"between-partition share of variance, averaged over classes: {eta:.3f}")
    print(f"  per class: {' '.join(f'{r:.2f}' for r in ratios)}")
    if eta < 0.02:
        print("! NEAR ZERO. Given the class, partitions are statistically "
              "indistinguishable in feature space.")
        print("  Geo-conditioning has no signal to exploit here, so E1 will come "
              "back flat for reasons that are about the DATA, not the method.")
        print("  Options: use a finer spatial partition, add spatial features "
              "(zone id, density, road class), or reframe the paper on the "
              "privacy result and drop the geo-accuracy claim.")
    elif eta < 0.08:
        print("  Weak but present. Geo-conditioning has limited headroom; expect "
              "a small effect and size the seed count accordingly.")
    else:
        print("  Partitions carry real signal. Geo-conditioning has something to "
              "work with, and E1 is a fair test.")
    return eta


def report_stages(ds, scenario="cgil", seed=0):
    print(f"\n=== 5. incremental staging ({scenario}) ===")
    sts = D.stages(ds, scenario, seed)
    prev = set()
    for i, (classes, partitions) in enumerate(sts):
        m = D.stage_mask(ds, classes, partitions)
        cells = {(int(p), int(c)) for p, c in zip(ds.part[m], ds.y[m])}
        new = cells - prev
        print(f"  stage {i+1}: classes {classes}  partitions {len(partitions):2d}  "
              f"samples {m.sum():6,}  cells {len(cells):3d}  (+{len(new)} new)")
        prev |= cells
        if m.sum() < 200:
            print(f"    ! only {m.sum()} samples in this stage; too few to learn from")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["geolife", "nyc_taxi", "synthetic"],
                    required=True)
    ap.add_argument("--data-path", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--partitions", type=int, default=18)
    ap.add_argument("--scenario", default="cgil")
    ap.add_argument("--sparse-threshold", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    cfg = Config()
    cfg.data.dataset = args.dataset
    cfg.data.path = args.data_path
    cfg.data.n_partitions = args.partitions
    cfg.out_dir = args.out
    os.makedirs(args.out, exist_ok=True)

    cp = cache_path(cfg)
    if os.path.exists(cp) and not args.rebuild:
        print(f"[cache] reading {cp}  (--rebuild to redo extraction)")
        ds = load_cache(cp)
    else:
        print(f"[build] extracting {args.dataset} from {args.data_path or '(generated)'} ...")
        try:
            ds = D.load(cfg)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n[cannot build] {e}\n")
            print("See DATASETS.md for where to download each set and the "
                  "directory layout the loader expects.")
            sys.exit(1)
        save_cache(ds, cp)

    report_basic(ds)
    acc = report_learnable(ds)
    frac = report_cells(ds, args.sparse_threshold)
    eta = report_partition_shift(ds)
    report_stages(ds, args.scenario)

    print("\n=== verdict ===")
    stop = []
    if acc is not None and acc < 1.5 / ds.n_classes:
        stop.append("features do not separate the classes")
    if eta is not None and eta < 0.02:
        stop.append("partitions are indistinguishable, so geo-conditioning "
                    "cannot help on this partitioning")
    if stop:
        print("Fix before running experiments:")
        for s in stop:
            print(f"  - {s}")
    else:
        print("Dataset looks usable. Next:")
        print(f"  python3 run.py --experiment main     --dataset {args.dataset} "
              f"--data-path {args.data_path} --seeds 5")
        print(f"  python3 run.py --experiment ablation --dataset {args.dataset} "
              f"--data-path {args.data_path} --seeds 5")


if __name__ == "__main__":
    main()
