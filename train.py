"""
The experiment driver: stages -> federated rounds -> per-cell evaluation.

Writes the three CSV schemas exp/analyze.py already reads, so every table in
the paper is one command once a run finishes. The per-cell file is the
important one: sparse-vs-dense (E4), per-partition BWT (E5) and cell-level
BWT all derive from it.
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F

from data import load, stages, stage_mask
from models import TaskModel
from fl import build_method, fedavg, ExperienceReplay, EWC


def _eval_cells(model, ds, idx):
    """Accuracy for every (partition, class) cell present in `idx`."""
    model.eval()
    out = {}
    with torch.no_grad():
        X = torch.tensor(ds.X[idx])
        logits = model(X)
        pred = logits.argmax(-1).numpy()
    y, part = ds.y[idx], ds.part[idx]
    for p in np.unique(part):
        for c in np.unique(y):
            m = (part == p) & (y == c)
            if m.sum() >= 3:
                out[(int(p), int(c))] = float((pred[m] == y[m]).mean())
    model.train()
    return out


def local_train(model, method, xb_all, yb_all, pb_all, cfg, hist_cells):
    """One client's local update.

    Replay enters at the classifier head for feature-space methods, so the
    backbone sees no replay gradient. Raw-space DGR is the exception: its
    samples are inputs and do pass through the backbone.
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr_task)
    n = len(xb_all)
    bs = cfg.train.batch_size

    for _ in range(cfg.train.local_epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.tensor(xb_all[idx])
            yb = torch.tensor(yb_all[idx])
            pb = torch.tensor(pb_all[idx])

            loss = F.cross_entropy(model(xb), yb)

            # --- replay -------------------------------------------------
            rep = method.replay_batch(model, None, hist_cells)
            if rep is not None:
                if isinstance(rep, tuple) and rep and rep[0] == "raw":
                    _, x_rep, y_rep = rep
                    loss = loss + F.cross_entropy(model(x_rep), y_rep)
                else:
                    h_rep, y_rep, _ = rep
                    loss = loss + F.cross_entropy(
                        model.forward_features(h_rep), y_rep)

            # --- raw exemplar buffer (ER only) ---------------------------
            if isinstance(method, ExperienceReplay):
                rb = method.raw_replay(bs)
                if rb is not None:
                    loss = loss + F.cross_entropy(model(rb[0]), rb[1])

            # --- extra terms: EWC penalty or feature distillation --------
            loss = loss + method.local_extra_loss(model, xb, yb, pb, None)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if hasattr(method, "train_generator"):
                method.train_generator(model, xb, yb, pb)


def run(cfg, method_name, log_dir=None, verbose=True):
    """One (method, scenario, seed) run. Returns a result dict."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds = load(cfg)
    train_ds, test_ds = ds.split(0.25, cfg.seed)
    sts = stages(ds, cfg.scenario, cfg.seed)
    model = TaskModel(cfg, ds.n_classes)
    method = build_method(method_name, cfg, ds)

    user_counts = ds.cell_user_counts()
    acc_first, acc_last = {}, {}
    tau_first = {}
    seen_cells = set()
    round_times = []
    round_idx = 0

    for si, (classes, partitions) in enumerate(sts):
        m = stage_mask(train_ds, classes, partitions)
        stage_idx = np.flatnonzero(m)
        if len(stage_idx) == 0:
            continue
        clients = np.unique(train_ds.client[stage_idx])

        for r in range(cfg.train.rounds_per_stage):
            t0 = time.time()
            states, weights = [], []
            hist = sorted(seen_cells)

            for ci in clients:
                ci_idx = stage_idx[train_ds.client[stage_idx] == ci]
                if len(ci_idx) < 8:
                    continue
                local = copy.deepcopy(model)
                local_train(local, method,
                            train_ds.X[ci_idx], train_ds.y[ci_idx],
                            train_ds.part[ci_idx], cfg, hist)
                states.append(local.state_dict())
                weights.append(len(ci_idx))

            if states:
                model.load_state_dict(fedavg(states, weights))
            method.after_round(model, round_idx, hist)
            round_times.append(time.time() - t0)
            round_idx += 1

        # end of stage: record what was learned, and consolidate
        te_mask = stage_mask(test_ds, classes, partitions)
        cur = _eval_cells(model, test_ds, np.flatnonzero(te_mask))
        for cell, a in cur.items():
            if cell not in acc_first:
                acc_first[cell] = a
                tau_first[cell] = round_idx
            seen_cells.add(cell)

        # drift histograms are recorded at tau_last, from real data
        if hasattr(method, "drift"):
            tr_mask = stage_mask(train_ds, classes, partitions)
            ti = np.flatnonzero(tr_mask)
            model.eval()
            with torch.no_grad():
                for cell in cur:
                    p, c = cell
                    sel = ti[(train_ds.part[ti] == p) & (train_ds.y[ti] == c)]
                    if len(sel) >= 3:
                        probs = F.softmax(
                            model(torch.tensor(train_ds.X[sel])), -1).mean(0).numpy()
                        method.drift.record(cell, probs, round_idx)
            model.train()

        if isinstance(method, EWC):
            method.consolidate(model, train_ds.X[stage_idx], train_ds.y[stage_idx])
        if isinstance(method, ExperienceReplay):
            method.add(train_ds.X[stage_idx], train_ds.y[stage_idx])
        # freeze the generator so the next stage can rehearse against it
        method.end_stage(sorted(seen_cells))

        if verbose:
            print(f"  stage {si+1}/{len(sts)}  cells seen {len(seen_cells):3d}  "
                  f"classes {classes}  partitions {len(partitions)}")

    # final evaluation over every cell ever seen
    final = _eval_cells(model, test_ds, np.arange(len(test_ds)))
    for cell in seen_cells:
        acc_last[cell] = final.get(cell, acc_first.get(cell, 0.0))

    cells = sorted(seen_cells)
    aa = 100.0 * float(np.mean([acc_last[c] for c in cells])) if cells else 0.0
    bwt = float(np.mean([acc_last[c] - acc_first[c] for c in cells])) if cells else 0.0

    rows = [{
        "method": method_name, "seed": cfg.seed,
        "partition": f"P{p:02d}", "cls": c,
        "n_users": user_counts.get((p, c), 0),
        "acc_first": round(acc_first[(p, c)], 6),
        "acc_final": round(acc_last[(p, c)], 6),
    } for (p, c) in cells]

    res = {
        "method": method_name, "scenario": cfg.scenario, "seed": cfg.seed,
        "aa": round(aa, 3), "bwt": round(bwt, 5),
        "time_per_round": round(float(np.mean(round_times)), 3) if round_times else 0.0,
        "n_cells": len(cells), "cells": rows,
        "model": model, "method_obj": method, "data": (train_ds, test_ds),
    }
    if verbose:
        print(f"  -> AA {aa:.2f}%  BWT {bwt:+.4f}  "
              f"{res['time_per_round']:.2f}s/round  ({len(cells)} cells)")
    return res
