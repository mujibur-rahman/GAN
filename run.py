#!/usr/bin/env python3
"""
Entry point. One command per experiment in EXPERIMENTS.md.

Quick check that everything works (seconds):
    python3 run.py --smoke

E1/E2 ablations + E3 seeds, on synthetic data:
    python3 run.py --experiment ablation --seeds 5

E3 baselines across seeds (feeds Table I):
    python3 run.py --experiment main --seeds 5

E6 MIA with the no-replay control:
    python3 run.py --experiment mia --seeds 3

Everything at once:
    python3 run.py --experiment all --seeds 5

On real data, add:  --dataset geolife --data-path /path/to/GeoLife
Scenario:           --scenario cil | gil | cgil     (default cgil)

Results land in --out (default results/) as seeds.csv, cells.csv,
ablation.csv and scores.csv, which is exactly what exp/analyze.py and
exp/mia_eval.py read.
"""

import argparse
import csv
import os
import sys
import time

from config import Config, ARMS
import train as trainer
import mia as mia_mod

BASELINES = ["ewc", "er", "dgr", "ac_gan", "geo_cvae"]
ABLATION_ARMS = ["geo_cvae_full", "no_geo_conditioning",
                 "no_drift_selective", "no_distillation"]
MIA_ARMS = ["no_replay", "ewc", "er", "ac_gan", "geo_cvae"]

PRETTY = {"ewc": "EWC", "er": "Exp. Replay (ER)", "dgr": "DGR (Full GAN)",
          "ac_gan": "Lightweight AC-GAN", "geo_cvae": "Geo-CVAE"}


def write_csv(path, rows, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def make_cfg(args, seed):
    cfg = Config()
    cfg.seed = seed
    cfg.scenario = args.scenario
    cfg.data.dataset = args.dataset
    cfg.data.path = args.data_path
    cfg.out_dir = args.out
    if args.quick:
        cfg.train.rounds_per_stage = 3
        cfg.train.local_epochs = 1
        cfg.data.n_clients = 24
        cfg.data.samples_per_user = 120
    return cfg


def run_one(args, method, seed, pretty=None):
    cfg = make_cfg(args, seed)
    label = pretty or method
    print(f"[{label}] seed {seed} scenario {cfg.scenario} ...")
    res = trainer.run(cfg, method, verbose=True)
    res["label"] = label
    return res, cfg


def exp_main(args):
    """E3: every baseline across seeds -> seeds.csv + cells.csv (Table I)."""
    sp, cp = os.path.join(args.out, "seeds.csv"), os.path.join(args.out, "cells.csv")
    for seed in range(args.seeds):
        for m in BASELINES:
            res, _ = run_one(args, m, seed, PRETTY[m])
            write_csv(sp, [[res["label"], seed, res["aa"], res["bwt"]]],
                      ["method", "seed", "aa", "bwt"])
            write_csv(cp, [[res["label"], seed, r["partition"], r["cls"],
                            r["n_users"], r["acc_first"], r["acc_final"]]
                           for r in res["cells"]],
                      ["method", "seed", "partition", "cls", "n_users",
                       "acc_first", "acc_final"])
            print(f"    time/round {res['time_per_round']}s")


def exp_ablation(args):
    """E1 + E2: isolate geo-conditioning and drift-selective replay."""
    ap = os.path.join(args.out, "ablation.csv")
    for seed in range(args.seeds):
        for arm in ABLATION_ARMS:
            res, _ = run_one(args, arm, seed)
            write_csv(ap, [[arm, seed, res["aa"], res["bwt"]]],
                      ["arm", "seed", "aa", "bwt"])


def exp_mia(args):
    """E6: one common attack across all arms, plus generator-specific ones."""
    path = os.path.join(args.out, "scores.csv")
    for seed in range(args.seeds):
        for m in MIA_ARMS:
            res, _ = run_one(args, m, seed, PRETTY.get(m, m))
            rows = mia_mod.evaluate(res, n=args.mia_n, seed=seed)
            write_csv(path, rows,
                      ["arm", "attack", "family", "is_member", "score"])
            print(f"    wrote {len(rows)} MIA score rows")


def smoke(args):
    """Fast end-to-end check: does every method run and log correctly."""
    args.quick = True
    args.seeds = 1
    cfg = make_cfg(args, 0)
    print(f"config: {cfg.summary()}\n")

    from models import report_sizes
    sizes = report_sizes(cfg, cfg.data.n_partitions, cfg.data.n_classes)
    print("parameter accounting")
    for k, v in sizes.items():
        print(f"  {k:24s} {v:>10,}")
    ok = sizes["geo_cvae_train"] == 3464
    print(f"  Geo-CVAE vs Table I (3,464): {'MATCH' if ok else 'MISMATCH'}\n")

    for m in ["no_replay", "ewc", "er", "ac_gan", "geo_cvae",
              "no_geo_conditioning"]:
        t0 = time.time()
        res, _ = run_one(args, m, 0, PRETTY.get(m, m))
        print(f"    ok in {time.time()-t0:.1f}s\n")
    print("smoke test passed -- pipeline runs end to end")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["main", "ablation", "mia", "all"],
                    default="all")
    ap.add_argument("--scenario", choices=["cil", "gil", "cgil"], default="cgil")
    ap.add_argument("--dataset", choices=["synthetic", "geolife", "nyc_taxi"],
                    default="synthetic")
    ap.add_argument("--data-path", default="")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--mia-n", type=int, default=1500)
    ap.add_argument("--quick", action="store_true",
                    help="fewer rounds/clients, for checking the plumbing")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.smoke:
        smoke(args)
        return

    if args.dataset == "synthetic":
        print("\n*** SYNTHETIC DATA: verifies the pipeline, not a result. "
              "Nothing from this run belongs in the paper. ***\n")

    t0 = time.time()
    if args.experiment in ("main", "all"):
        exp_main(args)
    if args.experiment in ("ablation", "all"):
        exp_ablation(args)
    if args.experiment in ("mia", "all"):
        exp_mia(args)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}/")
    print("next:")
    print(f"  python3 ../exp/analyze.py seeds     {args.out}/seeds.csv --out tab_main.tex")
    print(f"  python3 ../exp/analyze.py cells     {args.out}/cells.csv --out tab_sparse.tex")
    print(f"  python3 ../exp/analyze.py partition {args.out}/cells.csv --out tab_part.tex")
    print(f"  python3 ../exp/analyze.py ablation  {args.out}/ablation.csv --out tab_abl.tex")
    print(f"  python3 ../exp/mia_eval.py          {args.out}/scores.csv --out tab_mia.tex")


if __name__ == "__main__":
    main()
