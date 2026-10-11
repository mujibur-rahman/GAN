# Experiment plan — Geo-CVAE VTC paper

Ordered by what the paper currently claims but does not show. Tier 1 makes the
existing claims defensible; nothing below it changes the paper's scope.

Everything in Tier 1 runs on the GeoLife CG-IL configuration you already have.
No new dataset, no new scenario, no architecture work.

---

## Why this order

The paper lists three contributions. Contribution 1 is geo-conditioning,
contribution 2 is drift-selective replay plus feature distillation. Only the
distillation half of contribution 2 is ablated. Nothing isolates
geo-conditioning — the mechanism in the title — and nothing isolates drift
selection. That is the first thing a reviewer will look for and the first
thing to fix.

---

## Tier 1 — required

### E1. Geo-conditioning ablation
**Arm:** `no_geo_conditioning`. Same CVAE, conditioning vector is
`c(p,c) = e_c^cls` only; drop `e_p^geo`. Keep latent dim, hidden width, loss
weights, replay budget and schedule identical, so the only difference is the
geo term. Keep total conditioning width matched if you can (widen `d_k` to
`d_g + d_k`), otherwise a capacity confound enters and a reviewer will say so.

**Answers:** does the mechanism in the title do anything.
**Cost:** one arm × N seeds, same pipeline.
**Risk:** if ΔAA is near zero, the architectural claim is unsupported. Better
found now. See "If results go against you" below.

### E2. Drift-selective replay ablation
**Arm:** `no_drift_selective`. Replace `r_{p,c}` with uniform allocation over
historical cells, same total budget `R`.

Also log, for the full method, the realised distribution of `r_{p,c}` across
cells (write it to the per-cell CSV). If allocation is near-uniform in
practice, the mechanism is inert regardless of the ablation result, and the
honest move is to cut it from the contribution list rather than defend it.

**Answers:** whether contribution 2's first half earns its place.

### E3. Seed variance
Re-run every arm in Table I at **5–10 seeds**. Nothing else changes.

**Answers:** whether the 2.3-point AA margin over AC-GAN exceeds seed noise.
At n=3 it cannot be assessed, and the margin is small enough that this is a
real risk. `analyze.py seeds` runs the paired test and prints
`<-- not separated` when p ≥ 0.05.

### E4. Sparse vs dense cell breakdown
**No new training.** Reanalysis of per-cell logs you already have. You need
one row per (partition, class, seed) with `n_users`, `acc_first`, `acc_final`.

**Answers:** the paper says the Geo-CVAE margin comes from cells with fewer
than three contributing users. Currently asserted, not measured. This is the
cheapest item on the list and it converts your main explanation into evidence.

### E5. Per-partition BWT
**No new training.** Same per-cell CSV as E4.

**Answers:** whether forgetting is geographically concentrated. §III-B
originally promised this; it also supplies the fairness angle that makes the
spatial framing matter rather than being a relabelling of class-incremental FL.

### E6. MIA control, single attack across all arms
Run **one** attack — loss or confidence threshold on the shared task model —
identically on: `no_replay`, `ewc`, `er`, `ac_gan`, `geo_cvae`. Report AUC with
bootstrap CI and TPR at 1% FPR.

Keep the generator-specific attacks (recon-gap, D-score) as a **separate**
table, labelled as available surface, never cross-compared.

**Answers:** two problems at once. (a) Table II currently applies a different
attack to each method, so its columns are not a ranking. (b) There is no
no-replay baseline, so the paper cannot show replay *adds* exposure. `er` is
the natural high-leakage anchor since it stores raw segments.

**Note:** the recon-gap score is lower-for-members, so pass `--flip recon_gap`
or negate when writing the CSV.

---

## Tier 2 — scope

### E7. GeoLife G-IL
If you add exactly one scenario, add this one. Your contribution is the geo
axis and G-IL is the only scenario that isolates it. It also exercises the
untrained-embedding limitation (§IV-A), so it tests the mechanism where it is
weakest — which reads better than another favourable number.

### E8. Partition-count scaling
`param_scaling.py` computes the curve. Run it with your real dimensions and
confirm `--check` reproduces 3,464 before quoting it.

**Answers:** the O(1)-backbone claim, currently argued but never measured.

---

## Tier 3 — if time permits

- **E9. NYC Taxi CG-IL** — closes the two-dataset claim; the more
  vehicle-relevant dataset for this venue.
- **E10. DGR trained to convergence off-budget** — separates "no
  geo-conditioning" from "did not converge", currently confounded in §VII-A.
- **E11. Gradient-step or FLOP counts** — so the 2.2× timing result cannot be
  dismissed as tuning.

---

## Logging schema

Write these three CSVs and every table in the paper is one command.

**`seeds.csv`** — one row per (method, seed)
```
method,seed,aa,bwt
```

**`cells.csv`** — one row per (method, seed, partition, class). The important
one: E4, E5 and cell-level BWT all derive from it.
```
method,seed,partition,cls,n_users,acc_first,acc_final
```
`acc_first` is accuracy at the round the cell was first learned
(`tau_first`), `acc_final` at round T. Accuracy may be 0–1 or 0–100; the
loader detects which and normalises once.

**`ablation.csv`** — one row per (arm, seed)
```
arm,seed,aa,bwt
```
Arms: `geo_cvae_full`, `no_geo_conditioning`, `no_drift_selective`,
`no_distillation`.

**`scores.csv`** — one row per candidate record per attack
```
arm,attack,family,is_member,score
```
`family` is `common` (comparable across arms) or `specific` (generator-only).
Higher score must mean more member-like.

---

## Commands

```bash
python3 analyze.py seeds     results/seeds.csv    --out tab_main.tex \
                                                  --reference "Lightweight AC-GAN"
python3 analyze.py cells     results/cells.csv    --out tab_sparse.tex --threshold 3
python3 analyze.py partition results/cells.csv    --out tab_partition.tex
python3 analyze.py ablation  results/ablation.csv --out tab_ablation.tex \
                                                  --full geo_cvae_full
python3 mia_eval.py          results/scores.csv   --out tab_mia.tex --flip recon_gap
python3 param_scaling.py --d-hidden 20 --d-z 10 --d-g 16 --d-k 12 --check 3464 \
                         --out tab_scaling.tex
```

Each writes a LaTeX fragment that drops straight into the scaffolded tables in
the manuscript.

---

## Hyperparameters still unreported

The manuscript has TODOs for `beta`, `d_z`, `d_g`, `d_k`, `lambda_r`,
`lambda_d`, the divergence `D`, `epsilon`, and the MIA threshold. Reviewers
treat this as a reproducibility failure, and it is the single most likely
cause of rejection independent of results.

Searching the parameter-count space, **exactly one** configuration reproduces
your reported 3,464 at P=18, K=7, d_feat=32:

```
d_hidden=20  d_z=10  d_g=16  d_k=12
```

This is inferred from the parameter count, not read from your code. Verify it
against the implementation before putting it in the paper. If it matches, four
TODOs close immediately.

---

## If results go against you

E1 and E3 can undercut the paper, so decide the response before running them
rather than after.

- **Geo-conditioning shows no effect (E1).** The accuracy framing is dead, but
  the paper is not. Reframe around the privacy-surface analysis, which is the
  more distinctive contribution and does not depend on the margin: generative
  replay is widely treated as privacy-safe because it stores no raw data, and
  a measured characterisation of what it does leak stands on its own. Retitle
  toward the MIA result and demote the accuracy table to context.

- **Margin inside seed noise (E3).** Report it as parity, not superiority: the
  Geo-CVAE matches AC-GAN accuracy at half the per-round training time and
  with one fewer attack surface. That is a publishable claim and an honest one.
  "Comparable accuracy, lower cost, smaller surface" survives review;
  an unsupported 2.3-point claim does not.

- **Drift selection inert (E2).** Cut it from the contributions, keep it as a
  negative result in one sentence. A paper claiming two mechanisms that work
  beats one claiming three where the third does not.
