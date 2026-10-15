# Geo-CVAE experiment pipeline

Runnable implementation of the VTC paper's method and baselines, wired to the
analysis scripts so every table is one command.

## Start here

```bash
pip install --break-system-packages torch          # numpy/scipy/sklearn assumed
clone-repo
cd geocvae
python3 run.py --smoke                             # ~30 s, checks everything runs
python3 run.py --experiment all --seeds 5          # the Tier-1 set
```

`--smoke` also prints the parameter accounting and asserts the Geo-CVAE comes
to 3,464 parameters, the figure already in Table I.

Synthetic data is the default, so the pipeline runs with no download. **Numbers
from synthetic data verify the plumbing and belong nowhere near the paper.**
For real runs:

```bash
python3 run.py --experiment all --seeds 5 \
    --dataset geolife --data-path /path/to/Geolife\ Trajectories\ 1.3
```

## What maps to which experiment

| Command | Experiment | Produces |
|---|---|---|
| `--experiment main` | E3 baselines across seeds | `seeds.csv`, `cells.csv` → Table I |
| `--experiment ablation` | E1 geo-conditioning, E2 drift-selective | `ablation.csv` → ablation table |
| `--experiment mia` | E6 common attack + no-replay control | `scores.csv` → MIA table |
| `cells.csv` reanalysis | E4 sparse/dense, E5 per-partition | no extra training |

Then:

```bash
python3 analyze.py seeds     results/seeds.csv    --out tab_main.tex
python3 analyze.py cells     results/cells.csv    --out tab_sparse.tex
python3 analyze.py partition results/cells.csv    --out tab_part.tex
python3 analyze.py ablation  results/ablation.csv --out tab_abl.tex
python3 mia_eval.py          results/scores.csv   --out tab_mia.tex --flip recon_gap
```

## Hyperparameters

`config.py` is the single source of truth and closes the manuscript's open
TODOs: `beta`, `d_z`, `d_g`, `d_k`, `lambda_r`, `lambda_d`, the divergence `D`,
`epsilon` and the replay budget `R`. Fill Section IV from this file, not the
other way round. `cfg.summary()` prints the line to paste.

The CVAE dimensions (`d_hidden=20, d_z=10, d_g=16, d_k=12`) are the one
configuration that reproduces 3,464 parameters at P=18, K=7, d_feat=32.

## Four implementation points that differ from the submitted draft

These came out of the manuscript review and are worth carrying back into the
text, because the draft as written describes something that cannot run.

**1. `L_distil` is computed on the current real batch.** The draft summed it
over the replay batch, which holds features and no inputs, so `f_phi(x)` was
not computable. See `GeoCVAEReplay.local_extra_loss`.

**2. The penultimate feature is pre-activation.** With a trailing ReLU on the
backbone, real features on this data are ~94% exact zeros. A Gaussian decoder
under MSE cannot represent that point mass, so sampled features land
off-manifold and actively corrupt the classifier head they are replayed into.
Measured: with the ReLU, replay *reduced* accuracy from 34.4% to 19.1% and the
model collapsed to one class. Removing it flipped replay to a gain. If the real
implementation has a post-activation penultimate layer, this is probably the
first thing to check.

**3. The generator needs self-replay.** Training the CVAE only on current-stage
features makes it forget historical cells as fast as the task model does, so
replay injects wrong features. Each generator is now snapshotted at stage end
and rehearsed against (`end_stage`), which is DGR's scholar–student scheme. The
AC-GAN baseline gets the same treatment so the comparison is between generator
families, not between one that rehearses and one that does not.

**4. The no-geo ablation zeroes the embedding; it does not remove it.**
`GeoCVAE.cond` substitutes a zero vector of the same width when
`geo_cond=False`, so parameter count and input dimensions are unchanged.
Dropping the columns would confound "no geo information" with "smaller model".

## Honest limitation in the drift signal

`DriftTracker.update` estimates the current posterior for a historical cell by
running the model over features **generated for that cell**, since no real data
for it exists at round *t*. The signal therefore tracks the classifier relative
to the generator and is blind to the decoder degrading on its own. That is
Limitation 2 in the paper, and it is real, not a modelling convenience.

## Layout

```
config.py    every hyperparameter; ARMS defines the ablation variants
data.py      synthetic generator, GeoLife + NYC loaders, C-IL/G-IL/CG-IL splits
models.py    TaskModel, GeoCVAE, FeatureACGAN, RawDGR, report_sizes()
fl.py        FedAvg, DriftTracker, and the five methods
train.py     stage loop, per-cell evaluation, tau_first / tau_last
mia.py       common-family and generator-specific attacks
run.py       CLI
```

## Cost

On 2 CPU cores, one CG-IL run at 73 clients is roughly 1.5 s/round × 45 rounds
≈ 1 min. The full Tier-1 set at 5 seeds is a few hours. Real GeoLife will be
slower in feature extraction (cached after the first load) but similar to training.
