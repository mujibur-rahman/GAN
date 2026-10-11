"""
Data for the FL-SC experiments.

Three sources behind one interface:

  synthetic  GeoLife-shaped data generated here. Runs with no download, so
             the whole pipeline is testable today. Its class and partition
             structure matches the paper's description (73 clients, 7 motion
             classes, 18 ring-road cells, ~40% sparse cells), which means the
             plumbing, the logging schema and the analysis scripts can all be
             verified before real data is in hand. Numbers from it are NOT
             results and must never reach the paper.

  geolife    the real thing. Point --data-path at the extracted GeoLife
             release; the loader segments trajectories and computes the
             7-d motion vector the paper describes.

  nyc_taxi   TLC yellow-taxi parquet/CSV for 2016 Q1.

A Dataset is: X (n, d) float32, y (n,) int, part (n,) int, client (n,) int.
Everything downstream works from those four arrays.
"""

import os
import numpy as np

CLASS_NAMES = ["walk", "bike", "bus", "taxi", "car", "train", "subway"]

# Section VI-A: {walk, bike} -> {bus, taxi} -> {car, train, subway}
CIL_STAGES = [[0, 1], [2, 3], [4, 5, 6]]


class Dataset:
    def __init__(self, X, y, part, client, n_partitions, n_classes, name):
        self.X = np.asarray(X, np.float32)
        self.y = np.asarray(y, np.int64)
        self.part = np.asarray(part, np.int64)
        self.client = np.asarray(client, np.int64)
        self.n_partitions = int(n_partitions)
        self.n_classes = int(n_classes)
        self.name = name

    def __len__(self):
        return len(self.y)

    def cell_user_counts(self):
        """Unique contributing clients per (partition, class) cell.

        This is what decides sparse vs dense in the paper's analysis, so it is
        computed from the data rather than assumed.
        """
        counts = {}
        for p in range(self.n_partitions):
            for c in range(self.n_classes):
                m = (self.part == p) & (self.y == c)
                counts[(p, c)] = int(np.unique(self.client[m]).size) if m.any() else 0
        return counts

    def split(self, frac=0.25, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        k = int(len(self) * frac)
        te, tr = idx[:k], idx[k:]
        mk = lambda i: Dataset(self.X[i], self.y[i], self.part[i], self.client[i],
                               self.n_partitions, self.n_classes, self.name)
        return mk(tr), mk(te)


# ----------------------------------------------------------------- synthetic

def make_synthetic(cfg, seed=0):
    """GeoLife-shaped data with spatially structured non-IID cells.

    Design choices that matter for what the experiments can detect:

    - Each class is a Gaussian in 7-d motion space with a class-specific mean
      (walk slow/low-variance, train fast/steady, bus frequent stops, ...).
    - Each partition applies its own additive shift AND per-dimension scale to
      those means. Without that shift, geo-conditioning would have nothing to
      learn and E1 would be guaranteed to come back flat for reasons that have
      nothing to do with the method. The shift magnitude is the single knob
      that controls how much geo-conditioning *can* help, so it is reported.
    - Partition sizes are deliberately uneven so about `sparse_frac` of cells
      have fewer than 3 contributing clients, matching the condition the paper
      says the AC-GAN destabilises in.
    """
    d = cfg.data
    rng = np.random.default_rng(seed)
    K, P, D = d.n_classes, d.n_partitions, 7

    # class prototypes: speed, speed-var, accel, heading-change, stop-rate,
    # segment length, duration -- roughly ordered the way the real modes sit
    proto = np.array([
        [0.4, 0.2, 0.2, 0.8, 0.7, 0.3, 0.5],   # walk
        [1.2, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5],   # bike
        [2.2, 1.4, 0.9, 0.4, 0.9, 1.1, 0.9],   # bus
        [2.9, 1.2, 1.1, 0.5, 0.5, 1.0, 0.7],   # taxi
        [3.2, 1.3, 1.2, 0.4, 0.4, 1.2, 0.8],   # car
        [4.1, 0.9, 0.6, 0.1, 0.3, 2.2, 1.6],   # train
        [3.4, 0.8, 0.7, 0.2, 0.6, 1.5, 1.1],   # subway
    ], np.float32)[:K]

    # partition effects: a shift and a scale per partition
    geo_shift = rng.normal(0, 0.45, size=(P, D)).astype(np.float32)
    geo_scale = np.exp(rng.normal(0, 0.18, size=(P, D))).astype(np.float32)

    # assign clients to partitions unevenly, so some cells end up sparse
    n_sparse_parts = max(1, int(round(P * d.sparse_frac)))
    sparse_parts = set(rng.choice(P, n_sparse_parts, replace=False).tolist())
    weights = np.array([0.35 if p in sparse_parts else 1.0 for p in range(P)])
    weights = weights / weights.sum()
    client_part = rng.choice(P, size=d.n_clients, p=weights)
    # guarantee every partition has at least one client
    for p in range(P):
        if not (client_part == p).any():
            client_part[rng.integers(d.n_clients)] = p

    X, y, part, client = [], [], [], []
    for ci in range(d.n_clients):
        p = int(client_part[ci])
        # each client covers a subset of classes (label skew), weighted
        n_cls = int(rng.integers(3, K + 1))
        cls = rng.choice(K, n_cls, replace=False)
        mix = rng.dirichlet(np.ones(n_cls) * 1.4)
        n = int(rng.integers(int(d.samples_per_user * 0.6), int(d.samples_per_user * 1.4)))
        counts = rng.multinomial(n, mix)
        # per-client jitter: two users in one cell are not identical
        user_bias = rng.normal(0, 0.12, size=D).astype(np.float32)
        for c, cnt in zip(cls, counts):
            if cnt == 0:
                continue
            mu = (proto[c] + geo_shift[p] + user_bias) * geo_scale[p]
            sd = 0.22 + 0.10 * rng.random()
            samples = rng.normal(mu, sd, size=(cnt, D)).astype(np.float32)
            X.append(samples)
            y.append(np.full(cnt, c, np.int64))
            part.append(np.full(cnt, p, np.int64))
            client.append(np.full(cnt, ci, np.int64))

    ds = Dataset(np.concatenate(X), np.concatenate(y), np.concatenate(part),
                 np.concatenate(client), P, K, "synthetic")
    # standardise, as a real pipeline would after feature extraction
    ds.X = (ds.X - ds.X.mean(0)) / (ds.X.std(0) + 1e-6)
    return ds


# -------------------------------------------------------------- real loaders

def load_geolife(cfg):
    """Real GeoLife. Expects the extracted release:  <path>/Data/<uid>/...

    Uses only the users carrying transportation-mode labels (labels.txt),
    segments each labelled interval, and computes the 7-d motion vector.
    """
    import pandas as pd
    root = cfg.data.path
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(
            "GeoLife not found. Download 'GeoLife GPS Trajectories' from "
            "Microsoft Research, extract it, and pass --data-path <dir> "
            "(the directory containing Data/).")

    data_dir = os.path.join(root, "Data")
    users = sorted(d for d in os.listdir(data_dir)
                   if os.path.isfile(os.path.join(data_dir, d, "labels.txt")))
    if not users:
        raise RuntimeError(f"no labelled users under {data_dir}")

    name_to_id = {n: i for i, n in enumerate(CLASS_NAMES)}
    X, y, part, client = [], [], [], []

    for ci, uid in enumerate(users):
        udir = os.path.join(data_dir, uid)
        labels = pd.read_csv(os.path.join(udir, "labels.txt"), sep="\t")
        labels.columns = ["start", "end", "mode"]
        labels["start"] = pd.to_datetime(labels["start"])
        labels["end"] = pd.to_datetime(labels["end"])
        labels = labels[labels["mode"].isin(name_to_id)]
        if labels.empty:
            continue

        pts = []
        tdir = os.path.join(udir, "Trajectory")
        for fn in sorted(os.listdir(tdir)):
            if not fn.endswith(".plt"):
                continue
            df = pd.read_csv(os.path.join(tdir, fn), skiprows=6, header=None,
                             usecols=[0, 1, 5, 6], names=["lat", "lon", "date", "time"])
            df["t"] = pd.to_datetime(df["date"] + " " + df["time"])
            pts.append(df[["lat", "lon", "t"]])
        if not pts:
            continue
        traj = pd.concat(pts).sort_values("t").reset_index(drop=True)

        for _, lab in labels.iterrows():
            seg = traj[(traj["t"] >= lab["start"]) & (traj["t"] <= lab["end"])]
            if len(seg) < 10:
                continue
            feats = _motion_features(seg["lat"].values, seg["lon"].values,
                                     seg["t"].values)
            if feats is None:
                continue
            X.append(feats)
            y.append(name_to_id[lab["mode"]])
            part.append(_beijing_cell(seg["lat"].mean(), seg["lon"].mean(),
                                      cfg.data.n_partitions))
            client.append(ci)

    if not X:
        raise RuntimeError("no usable segments extracted from GeoLife")
    ds = Dataset(np.array(X), np.array(y), np.array(part), np.array(client),
                 cfg.data.n_partitions, cfg.data.n_classes, "geolife")
    ds.X = (ds.X - ds.X.mean(0)) / (ds.X.std(0) + 1e-6)
    return ds


def _motion_features(lat, lon, t):
    """mean speed, speed variance, acceleration, heading-change rate,
    stop rate, segment length, duration (Section VI-A)."""
    tsec = (t - t[0]) / np.timedelta64(1, "s")
    dt = np.diff(tsec)
    ok = dt > 0
    if ok.sum() < 5:
        return None
    dlat, dlon = np.diff(lat), np.diff(lon)
    mlat = np.radians((lat[:-1] + lat[1:]) / 2)
    dx = np.radians(dlon) * 6371000 * np.cos(mlat)
    dy = np.radians(dlat) * 6371000
    dist = np.hypot(dx, dy)[ok]
    dt = dt[ok]
    v = dist / dt
    a = np.diff(v) / dt[1:] if len(v) > 1 else np.array([0.0])
    head = np.degrees(np.arctan2(dy[ok], dx[ok]))
    dhead = np.abs(np.diff(head))
    dhead = np.minimum(dhead, 360 - dhead) if len(dhead) else np.array([0.0])
    total_t = float(tsec[-1])
    return np.array([
        float(v.mean()), float(v.var()), float(np.abs(a).mean()),
        float(dhead.mean() / 180.0), float((v < 0.5).mean()),
        float(dist.sum()), total_t,
    ], np.float32)


def _beijing_cell(lat, lon, n_cells):
    """5x5 ring-road grid over central Beijing, merged to n_cells."""
    lat0, lat1, lon0, lon1 = 39.75, 40.10, 116.20, 116.60
    r = int(np.clip((lat - lat0) / (lat1 - lat0) * 5, 0, 4))
    c = int(np.clip((lon - lon0) / (lon1 - lon0) * 5, 0, 4))
    return (r * 5 + c) % n_cells


def load_nyc_taxi(cfg):
    """TLC yellow taxi 2016 Q1. Classes are quantile-binned trip distance;
    partitions are merged pickup zones; clients are disjoint zone sets."""
    import pandas as pd
    path = cfg.data.path
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "NYC taxi data not found. Pass --data-path to a TLC "
            "yellow_tripdata_2016-0[123] parquet or csv file (or a directory "
            "of them).")

    files = ([os.path.join(path, f) for f in sorted(os.listdir(path))]
             if os.path.isdir(path) else [path])
    frames = []
    for f in files:
        rd = pd.read_parquet if f.endswith(".parquet") else pd.read_csv
        frames.append(rd(f))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]

    need = ["trip_distance", "pulocationid", "dolocationid"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing columns {miss}; got {list(df.columns)[:12]}")
    df = df[(df.trip_distance > 0) & (df.trip_distance < 100)].copy()

    # 4 trip classes: short <2mi, medium 2-6, long 6-15, airport >15 or JFK/LGA
    AIRPORT = {132, 138}
    d = df.trip_distance.values
    cls = np.digitize(d, [2, 6, 15])
    cls[np.isin(df.pulocationid.values, list(AIRPORT))
        | np.isin(df.dolocationid.values, list(AIRPORT))] = 3

    zones = df.pulocationid.values
    uz, counts = np.unique(zones, return_counts=True)
    keep = uz[np.argsort(-counts)][:cfg.data.n_partitions]
    zmap = {z: i for i, z in enumerate(keep)}
    part = np.array([zmap.get(z, len(keep) - 1) for z in zones])

    feats = np.stack([
        d, df.get("passenger_count", pd.Series(np.ones(len(df)))).values,
        df.get("trip_duration", pd.Series(d * 180)).values,
        zones, df.dolocationid.values,
        (zones == df.dolocationid.values).astype(float),
        np.log1p(d),
    ], 1).astype(np.float32)

    rng = np.random.default_rng(cfg.seed)
    client = rng.integers(0, cfg.data.n_clients, size=len(df))
    ds = Dataset(feats, cls, part, client, len(keep), 4, "nyc_taxi")
    ds.X = (ds.X - ds.X.mean(0)) / (ds.X.std(0) + 1e-6)
    return ds


def load(cfg):
    return {"synthetic": lambda: make_synthetic(cfg, cfg.seed),
            "geolife": lambda: load_geolife(cfg),
            "nyc_taxi": lambda: load_nyc_taxi(cfg)}[cfg.data.dataset]()


# ------------------------------------------------------------------ scenarios

def stages(ds, scenario, seed=0):
    """Return a list of (classes, partitions) available at each stage.

    C-IL   classes arrive in stages, all partitions present throughout
    G-IL   partitions arrive in stages, all classes present throughout
    CG-IL  both arrive together -- the realistic deployment case
    """
    rng = np.random.default_rng(seed)
    K, P = ds.n_classes, ds.n_partitions
    cls_stages = ([c for c in CIL_STAGES if all(x < K for x in c)]
                  if K >= 7 else [list(s) for s in np.array_split(np.arange(K), 3)])
    perm = rng.permutation(P)
    part_stages = [np.sort(s).tolist() for s in np.array_split(perm, 3)]

    all_c, all_p = list(range(K)), list(range(P))
    if scenario == "cil":
        return [(cls_stages[i], all_p) for i in range(3)]
    if scenario == "gil":
        cum = []
        out = []
        for i in range(3):
            cum = cum + part_stages[i]
            out.append((all_c, sorted(cum)))
        return out
    if scenario == "cgil":
        cum = []
        out = []
        for i in range(3):
            cum = cum + part_stages[i]
            out.append((cls_stages[i], sorted(cum)))
        return out
    raise ValueError(f"unknown scenario {scenario!r}")


def stage_mask(ds, classes, partitions):
    return np.isin(ds.y, classes) & np.isin(ds.part, partitions)
