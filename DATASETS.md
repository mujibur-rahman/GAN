# Building the real datasets

Neither dataset is reachable from a sandbox, so both downloads happen on your
machine. Budget ~1.7 GB and one evening.

Always run `build_dataset.py` before `run.py`. It caches the extraction and
tells you whether the set can answer the question you are asking, which is
cheaper than finding out after five seeds.

---

## GeoLife (primary — the paper's main results)

**Download.** "GeoLife GPS Trajectories 1.3", Microsoft Research Open Data.
Search for `GeoLife GPS Trajectories` — Microsoft has moved the direct link
more than once, so take it from the current MSR Open Data page rather than a
cached URL. ~300 MB zipped.

**Extract** so the layout is:

```
Geolife Trajectories 1.3/
  Data/
    000/
      Trajectory/
        20081023025304.plt      ...
    010/
      Trajectory/ ...
      labels.txt                <- only some users have this
```

**Build and validate:**

```bash
python3 build_dataset.py --dataset geolife \
        --data-path "~/data/Geolife Trajectories 1.3" --partitions 18
```

First run takes 10–30 min (182 users, ~18k trajectory files). It caches to
`results/geolife_cache.npz`; later runs load in seconds.

**What the loader does.** Keeps only users with `labels.txt`, splits each
trajectory at the labelled mode intervals, drops segments under 10 points, and
computes the 7-d motion vector the paper describes: mean speed, speed variance,
mean |acceleration|, heading-change rate, stop rate, segment length, duration.
Partitions come from a 5×5 ring-road grid over central Beijing, modulo
`--partitions`.

**Expect ~69 labelled users, not 73.** The labelled subset is usually quoted as
69; the paper says 73. Whatever the builder reports is the number to put in the
paper — check it and correct Section VI-A if needed.

---

## NYC Yellow Taxi (secondary — the more VTC-relevant set)

**Download** `yellow_tripdata_2016-01.parquet`, `-02`, `-03` from the NYC TLC
trip record page (CloudFront-hosted parquet). ~400 MB each.

```bash
python3 build_dataset.py --dataset nyc_taxi --data-path ~/data/tlc/ --partitions 52
```

**Subsample first.** 33M trips will not fit in memory and you do not need them:

```python
import pandas as pd
df = pd.read_parquet("yellow_tripdata_2016-01.parquet")
df.sample(300_000, random_state=0).to_parquet("tlc/jan_sample.parquet")
```

300k trips per month is ample for 52 partitions × 4 classes.

**Note the client definition.** The paper simulates workers by "sampling
disjoint sets of taxi zone combinations". Taxi records have no driver id, so
clients are a modelling choice, not a fact in the data — say so explicitly in
the paper or a reviewer will ask what a client is.

---

## The decisions that are yours, not the code's

**Spatial granularity is the real experimental variable.** It sets both
quantities the paper depends on, and they trade against each other:

| Partitions | Geo signal (η²) | Sparse cells | Consequence |
|---|---|---|---|
| few (6–12) | lower | few | geo-conditioning has little to exploit |
| paper's 18 | ? | ? | **measure it** |
| many (40+) | higher | many | strong geo signal, but cells too thin to learn |

Run the builder at `--partitions 12`, `18`, `36` and read §4 and §3 of the
report. Pick the setting where η² is meaningful *and* sparse cells stay under
roughly half. That choice is a legitimate methodological decision — report it,
with the numbers behind it.

**Train/test split.** The pipeline splits randomly. For incremental learning a
*temporal* split is more honest (train on earlier trajectories, test on later),
and harder. If you use the random split, say so; if a reviewer asks whether
your forgetting is temporal or just sampling, the random split cannot answer.

**Segment length.** Minimum 10 points is arbitrary. Too low and the motion
features are noise; too high and sparse cells empty out. Worth one sensitivity
check.

---

## What the report tells you, and what to do about it

**§2 learnable** — centralised accuracy is the ceiling. Everything federated
lands below it. If this is near chance, the features are wrong and nothing
downstream matters.

**§3 cell occupancy** — the paper says sparse cells are "roughly 40% of the
partition-class vocabulary". Check that. If it comes out at 10%, the
sparse-cell explanation for your margin has little room to be true.

**§4 partition shift — run this before E1.** It measures whether partitions are
distinguishable given the class, which bounds what geo-conditioning can
possibly buy. Near zero means E1 is decided before you run it, and the answer
is about your partitioning, not your method.

On my synthetic data this came out at **0.74** — very strong geo signal — and
geo-conditioning *still* showed no effect, because backbone drift defeated
replay before conditioning could matter. So a healthy η² is necessary, not
sufficient. Check η² first, then check the old-vs-new class split described in
the README.

**§5 staging** — confirms each stage has enough data to learn from. A stage
with a few hundred samples produces forgetting curves that are mostly noise.
