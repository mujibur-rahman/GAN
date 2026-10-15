# Representation drift in federated spatial crowdsourcing

Code for *"Catastrophic Forgetting Survives Feature Replay: Representation
Drift in Federated Spatial Crowdsourcing."*

The paper's claim is that **generative replay in the task model's penultimate
feature space retains nothing under federated continual learning** — not
degraded retention, zero — because the backbone keeps training and a feature
synthesised at round *t−1* describes a space the round-*t* classifier has left.
Moving the same conditional VAE into input space fixes it on two of three
datasets. This repository reproduces every number in the paper.

Everything runs on CPU. The full set of experiments is a few hours; the
30-second check below needs no data at all.

---

## 1. Clone and check it runs

```bash
git clone <this-repo> && cd <this-repo>
pip install -r requirements.txt

cd geocvae
python3 run.py --smoke
```

`--smoke` trains every method on generated data for a few rounds and prints the
parameter accounting, asserting that the Geo-CVAE comes to 3,464 parameters —
the figure in the paper. If that line says MISMATCH, stop: the model
dimensions in `config.py` have drifted from the manuscript.

> **Synthetic data verifies the plumbing and nothing else.** It is the default
> so the repo runs with no download, and every run on it prints a warning. No
> number from a synthetic run belongs in a paper.

---

## 2. Layout

```
geocvae/            the experiment
  config.py         every hyperparameter, single source of truth
  data.py           dataset loaders, caching, per-cell subsampling
  models.py         task model, Geo-CVAE, AC-GAN, raw DGR
  fl.py             FedAvg, drift tracking, all methods and ablation arms
  train.py          stages -> federated rounds -> per-cell evaluation
  run.py            CLI; one command per experiment
  build_dataset.py  extract + validate a dataset before you train on it
  mia.py            membership-inference attacks
  DATASETS.md       where to download each dataset and the layout expected

exp/                analysis; reads the CSVs run.py writes
  analyze.py        seeds / cells / partition / retention / ablation
  retention_sd.py   retention with the spread across seeds
  fairness_test.py  per-seed spatial dispersion test
  mia_eval.py       MIA AUC with bootstrap intervals
  param_scaling.py  generator size against P and K
```

Results are CSVs under whatever `--out` you pass. **Use a fresh `--out` per
experiment** — the writers append, so reusing a directory silently mixes runs
together. This bit us once: a crashed NYC run left rows that made a later
GeoLife analysis report 11 seeds and 52 partitions for an 18-partition dataset.

---

## 3. Get the data

Full download instructions and the directory layout each loader expects are in
`geocvae/DATASETS.md`. Always run `build_dataset.py` before `run.py`: it caches
the extraction and tells you whether the set can answer the question you are
asking, which is cheaper than finding out after five seeds.

```bash
cd geocvae

# GeoLife — the primary dataset (~300 MB)
python3 build_dataset.py --dataset geolife \
        --data-path "~/data/Geolife Trajectories 1.3" --partitions 18

# NYC Yellow Taxi — one month of trips, parquet or csv (~350 MB)
python3 build_dataset.py --dataset nyc_taxi --data-path ~/data/tlc/ \
        --partitions 18 --max-per-cell 1388 --min-per-cell 20

# Yelp Open Dataset — business + review JSON (~5 GB extracted)
python3 build_dataset.py --dataset yelp --data-path ~/data/yelp/ \
        --yelp-max-reviews 7000000 --feat-dim 8 \
        --partitions 18 --min-per-cell 20
```

Three things to check in that output before committing to a sweep:

| Dataset | Check | Expected |
|---|---|---|
| GeoLife | samples | ~9,250 over 73 clients |
| NYC | schema line | `zone ids` or `coordinates` — both are handled |
| Yelp | samples, cells | **≥15,000** samples, **≥60** occupied cells |

The Yelp thresholds matter. Streaming only part of the review file starves the
per-worker counts: 400k reviews gives 4,388 samples over 12–17 cells, which is
too few for any statistic to resolve. The whole file (`7000000`) gives 63,370
samples over ~52 cells per seed.

---

## 4. Reproduce the paper

Run these from `geocvae/`. Times are for a 4-core laptop CPU.

### Table II — GeoLife main comparison (~25 min)

```bash
python3 run.py --experiment main --dataset geolife \
        --data-path ~/data/geolife/ --seeds 5 --out r_main
python3 ../exp/analyze.py seeds     r_main/seeds.csv
python3 ../exp/analyze.py retention r_main/cells.csv --old-classes 0,1,2,3
```

### Table III — retention on all three datasets

GeoLife as above, plus:

```bash
# Yelp at d_h = 8, where the input is 3.9x the feature space (~20 min)
python3 run.py --experiment main --dataset yelp --data-path ~/data/yelp/ \
        --yelp-max-reviews 7000000 --feat-dim 8 \
        --partitions 18 --min-per-cell 20 --seeds 5 --out r_yelp_d8

# NYC (~6 h; the 12.7M-row parse is cached after the first seed)
python3 run.py --experiment main --dataset nyc_taxi --data-path ~/data/tlc/ \
        --partitions 18 --max-per-cell 1388 --min-per-cell 20 \
        --seeds 5 --out r_nyc

python3 ../exp/analyze.py retention r_yelp_d8/cells.csv --old-classes 0,1,2,3
python3 ../exp/analyze.py retention r_nyc/cells.csv     --old-classes 0,1
```

### The no-replay control

The headline result is that feature-space replay lands on plain FedAvg, so the
control is not optional:

```bash
python3 run.py --experiment main --methods no_replay --dataset geolife \
        --data-path ~/data/geolife/ --seeds 5 --out r_main_ctl
python3 ../exp/analyze.py retention r_main_ctl/cells.csv --old-classes 0,1,2,3
```

### Feature-width control — rules out cost and capacity (~20 min)

At `--feat-dim 4` the feature-space generator (2,316 params) is *cheaper* than
the input-space one (2,439) and still retains nothing:

```bash
for D in 32 8 4; do
  python3 run.py --experiment main --dataset geolife \
          --data-path ~/data/geolife/ \
          --methods geo_cvae,geo_cvae_input,ac_gan,no_replay \
          --feat-dim $D --seeds 5 --out r_gl_d$D
  python3 ../exp/analyze.py retention r_gl_d$D/cells.csv --old-classes 0,1,2,3
done
```

### Figure 1 — replay-budget sweep on NYC (~1 h)

```bash
for R in 32 64 128; do
  python3 run.py --experiment main --dataset nyc_taxi --data-path ~/data/tlc/ \
          --methods geo_cvae_input --replay-budget $R \
          --partitions 18 --max-per-cell 1388 --min-per-cell 20 \
          --seeds 3 --out r_nyc_R$R
  python3 ../exp/analyze.py retention r_nyc_R$R/cells.csv --old-classes 0,1
done
cd .. && python3 fig_budget.py        # edit DATA at the top if budgets differ
```

### Table IV — component ablation (~50 min for both)

```bash
python3 run.py --experiment ablation --dataset geolife \
        --data-path ~/data/geolife/ --seeds 5 --out r_abl_gl
python3 run.py --experiment ablation --dataset yelp --data-path ~/data/yelp/ \
        --yelp-max-reviews 7000000 --feat-dim 8 \
        --partitions 18 --min-per-cell 20 --seeds 5 --out r_abl_yelp

python3 ../exp/analyze.py ablation     r_abl_gl/ablation.csv --full abl_full
python3 ../exp/retention_sd.py r_abl_gl/ablation_cells.csv \
        --old-classes 0,1,2,3 --vs abl_full
```

The six arms switch off one component each: `abl_full`, `abl_no_geo`,
`abl_no_drift`, `abl_no_distil`, `abl_no_selfreplay`, `abl_no_balance`. They
run on the **input-space** method deliberately — ablating the feature-space arm
is uninformative, since it retains 0.00% whichever component is active.

### Spatial fairness and membership inference

```bash
python3 ../exp/fairness_test.py r_yelp_d8/cells.csv \
        --a "Geo-CVAE (input)" --b "Lightweight AC-GAN"

python3 run.py --experiment mia --dataset geolife \
        --data-path ~/data/geolife/ --seeds 3 --out r_mia
python3 ../exp/mia_eval.py r_mia/scores.csv
```

Use `fairness_test.py`, not `analyze.py partition`, for any dispersion claim:
`partition` pools seeds before taking the standard deviation, which understates
the variance and inflates apparent differences. Two claims in an early draft
did not survive the per-seed test.

---

## 5. Flags you will actually use

| Flag | What it does |
|---|---|
| `--methods a,b` | run specific arms instead of all baselines |
| `--feat-dim N` | penultimate width; 8 for Yelp, 32 elsewhere |
| `--replay-budget R` | generated samples per client per round (default 192) |
| `--max-per-cell N` | cap per (partition, class) cell — never a global sample, which destroys sparse cells |
| `--min-per-cell N` | drop cells below N samples |
| `--partitions P` | override the geo-partition count |
| `--yelp-max-reviews N` | reviews streamed; use the whole file |
| `--quick` | fewer rounds and clients, for checking plumbing |

Method names: `no_replay`, `ewc`, `er`, `dgr`, `ac_gan`, `geo_cvae`
(feature space), `geo_cvae_input` (input space), plus the `abl_*` arms.

---

## 6. Things that will bite you

**Read `retention`, not just `seeds`.** Average accuracy conceals the result
the paper is about. On all three datasets AA is close to a cell-count-weighted
average of old- and new-class accuracy, so a model that abandons the old label
space entirely and learns the new one well scores respectably. Backward
transfer has the same problem in a different way: when forgetting is total it
saturates, and three methods can report BWT identical to five decimal places
while differing in nothing because all three are at the floor.

**Caches are keyed, but not on everything.** The raw extraction is cached per
dataset (plus input width and review cap for Yelp) and the subsampled view
under its own key. Changing `--max-per-cell` does not force a re-extract;
changing the data itself does, so pass `--rebuild` to `build_dataset.py` if you
replace the source files.

**Label leakage is easy here.** NYC classes are binned trip distance, so
including distance, dropoff or duration as features makes the label a
deterministic function of the input: 98.6% centralised accuracy instead of
45.3%. The loader uses pre-trip observables only. Yelp's business attributes
include category-naming keys, which are filtered for the same reason. If you
add features, check the centralised ceiling first.

**Use a fresh `--out`.** See §2.

---

## 7. Environment

Tested on Python 3.11 and 3.13 with the versions in `requirements.txt`. CPU
only; nothing here needs a GPU and nothing is written to use one. GeoLife's
first extraction reads ~18k trajectory files and takes 10–30 minutes; NYC's
12.7M-row parse takes a few minutes and is cached thereafter.
