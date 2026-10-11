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


# NYC bounding box, used to drop the (0,0) and out-of-region rows that make up
# a few percent of every TLC coordinate-era file.
NYC_BBOX = (40.50, 40.92, -74.27, -73.68)   # lat_min, lat_max, lon_min, lon_max
JFK_LGA_ZONES = {132, 138}


def load_nyc_taxi(cfg):
    """TLC yellow taxi. Handles BOTH published schemas.

    The TLC stopped publishing coordinates in July 2016. Files before that
    (2015, and all of 2016 Q1) carry pickup_longitude / pickup_latitude;
    files from 2016-07 onward carry PULocationID / DOLocationID instead.
    The loader detects which and builds partitions accordingly:

      coordinates -> an N x N grid over the NYC bounding box, merged to
                     n_partitions. Granularity is yours to choose, which is
                     what makes the partition sweep possible.
      zone ids    -> the n_partitions busiest zones, rest folded into one.

    Note for Section VI-A: the paper says "263 zones in 6 boroughs" for 2016
    Q1, but 2016 Q1 predates the zone-id switch, so those files have no zone
    column. Either the description belongs to a spatial join against the TLC
    taxi-zone shapefile (which should then be stated), or it should say the
    partitions come from a coordinate grid.
    """
    import pandas as pd
    path = cfg.data.path
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "NYC taxi data not found. Pass --data-path to a TLC "
            "yellow_tripdata_*.parquet or .csv file, or a directory of them.")

    if os.path.isdir(path):
        names = sorted(os.listdir(path))
        # a directory often holds BOTH .csv and .parquet of the same month;
        # loading both would silently double every trip
        stems = {}
        for f in names:
            if not f.endswith((".csv", ".parquet")):
                continue
            stem = f.rsplit(".", 1)[0]
            # prefer parquet: smaller, typed, much faster
            if stem not in stems or f.endswith(".parquet"):
                stems[stem] = f
        files = [os.path.join(path, f) for f in stems.values()]
        skipped = len([f for f in names if f.endswith((".csv", ".parquet"))]) - len(files)
        if skipped:
            print(f"[nyc] {skipped} duplicate csv/parquet pair(s) skipped; "
                  f"using parquet")
    else:
        files = [path]

    frames = []
    for f in files:
        rd = pd.read_parquet if f.endswith(".parquet") else pd.read_csv
        frames.append(rd(f))
        print(f"[nyc] read {os.path.basename(f)}  ({len(frames[-1]):,} rows)")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower().strip() for c in df.columns]

    if "trip_distance" not in df.columns:
        raise RuntimeError(f"no trip_distance column; got {list(df.columns)[:14]}")

    has_coords = {"pickup_latitude", "pickup_longitude"} <= set(df.columns)
    has_zones = {"pulocationid", "dolocationid"} <= set(df.columns)
    if not (has_coords or has_zones):
        raise RuntimeError(
            "file has neither coordinates nor zone ids. Columns: "
            f"{list(df.columns)[:14]}")
    print(f"[nyc] schema: {'coordinates (pre-2016-07)' if has_coords else 'zone ids (2016-07+)'}")

    n0 = len(df)
    df = df[(df.trip_distance > 0) & (df.trip_distance < 100)]

    # trip duration, if the timestamps are there
    tp = next((c for c in df.columns if "pickup_datetime" in c), None)
    td = next((c for c in df.columns if "dropoff_datetime" in c), None)
    if tp and td:
        dur = (pd.to_datetime(df[td]) - pd.to_datetime(df[tp])).dt.total_seconds()
        df = df.assign(_dur=dur)
        df = df[(df._dur > 30) & (df._dur < 4 * 3600)]
    else:
        df = df.assign(_dur=df.trip_distance * 180)

    if has_coords:
        la0, la1, lo0, lo1 = NYC_BBOX
        df = df[(df.pickup_latitude.between(la0, la1))
                & (df.pickup_longitude.between(lo0, lo1))
                & (df.dropoff_latitude.between(la0, la1))
                & (df.dropoff_longitude.between(lo0, lo1))]
    df = df.reset_index(drop=True)
    print(f"[nyc] {n0:,} -> {len(df):,} rows after cleaning "
          f"({100*(n0-len(df))/max(n0,1):.1f}% dropped)")
    if len(df) == 0:
        raise RuntimeError("all rows filtered out; check the file")

    dist = df.trip_distance.values.astype(np.float64)

    # --- partitions -----------------------------------------------------
    P = cfg.data.n_partitions
    if has_coords:
        la0, la1, lo0, lo1 = NYC_BBOX
        side = int(np.ceil(np.sqrt(P)))
        r = np.clip(((df.pickup_latitude.values - la0) / (la1 - la0) * side).astype(int), 0, side - 1)
        c = np.clip(((df.pickup_longitude.values - lo0) / (lo1 - lo0) * side).astype(int), 0, side - 1)
        raw = r * side + c
        # keep the P busiest grid cells, fold the long tail into the last one
        uz, counts = np.unique(raw, return_counts=True)
        keep = uz[np.argsort(-counts)][:P]
        zmap = {z: i for i, z in enumerate(keep)}
        part = np.array([zmap.get(z, len(keep) - 1) for z in raw])
        n_parts = len(keep)
        pu_id, do_id = raw.astype(np.float64), np.zeros(len(df))
        is_airport = np.zeros(len(df), bool)   # no zone ids to test against
    else:
        zones = df.pulocationid.values
        uz, counts = np.unique(zones, return_counts=True)
        keep = uz[np.argsort(-counts)][:P]
        zmap = {z: i for i, z in enumerate(keep)}
        part = np.array([zmap.get(z, len(keep) - 1) for z in zones])
        n_parts = len(keep)
        pu_id = zones.astype(np.float64)
        do_id = df.dolocationid.values.astype(np.float64)
        is_airport = (np.isin(zones, list(JFK_LGA_ZONES))
                      | np.isin(df.dolocationid.values, list(JFK_LGA_ZONES)))

    # --- classes: short / medium / long / airport ------------------------
    cls = np.digitize(dist, [2, 6, 15])
    if is_airport.any():
        cls[is_airport] = 3
    else:
        # no zone ids: airport trips are the long ones, already class 3
        pass

    # --- features --------------------------------------------------------
    # LABEL LEAKAGE: the classes are quantile bins of trip_distance, so
    # trip_distance, log(distance) and speed (= distance / duration) each
    # determine the label outright. Including any of them makes the task
    # trivial -- measured at 99% centralised accuracy, which leaves nothing
    # for forgetting to destroy and no room for any method to differ.
    # The features below are the ones an allocator would actually have BEFORE
    # the trip: where it starts, where it is going, when, and how many riders.
    dur = df._dur.values.astype(np.float64)
    pax = df.get("passenger_count", pd.Series(np.ones(len(df)))).values.astype(np.float64)
    # Time is cyclical: hour 23 sits next to hour 0, and a raw integer tells
    # the model the opposite. Sin/cos puts them adjacent.
    if tp:
        t = pd.to_datetime(df[tp])
        h = t.dt.hour.values.astype(np.float64)
        d = t.dt.dayofweek.values.astype(np.float64)
    else:
        h = np.zeros(len(df))
        d = np.zeros(len(df))
    time_feats = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                           np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], 1)
    # DROPOFF AND DURATION ARE ALSO LEAKS. Trip distance is very nearly the
    # straight-line distance between pickup and dropoff, and duration is
    # distance divided by a fairly narrow speed band, so either one recovers
    # the label. Keeping them gave 98.6% centralised accuracy -- a task with
    # no headroom, where no continual-learning method can show anything.
    #
    # Only PRE-TRIP observables are kept: where the request comes from, when,
    # and how many riders. Predicting the trip-distance class from those is a
    # genuine allocation problem ("what kind of job is this request?") and is
    # the SC framing the paper actually argues for -- the platform assigns the
    # task before the destination is known.
    if has_coords:
        geo = np.stack([df.pickup_latitude.values, df.pickup_longitude.values], 1)
    else:
        # Zone id is categorical, so a raw integer is a poor input: zone 140 is
        # not "between" 139 and 141 in any useful sense. Give the model the
        # zone's centroid-like statistics instead -- its share of pickups and
        # its mean trip distance -- which are ordinal and carry real signal.
        # (A learned zone embedding would be better still; that is a change to
        # the task model, not the loader.)
        share = pd.Series(pu_id).map(pd.Series(pu_id).value_counts(normalize=True)).values
        geo = np.stack([pu_id, share], 1)
    feats = np.concatenate(
        [geo, time_feats, pax.reshape(-1, 1)], 1).astype(np.float32)

    # --- clients ---------------------------------------------------------
    # Taxi records carry no driver id, so a "client" is a modelling choice.
    # One client per partition would make every (partition, class) cell have
    # exactly one contributing user, which makes the sparse-vs-dense analysis
    # vacuous -- every cell is "sparse" by construction. Instead each
    # partition is worked by a VARIABLE number of drivers, so cell occupancy
    # genuinely varies and the sparse-cell claim becomes testable.
    # Say in the paper how clients are defined; a reviewer will ask.
    rng = np.random.default_rng(cfg.seed)
    cells = np.unique(part)
    n_cl = cfg.data.n_clients
    drivers_per = {z: int(rng.integers(1, max(2, n_cl // max(len(cells), 1) * 2 + 1)))
                   for z in cells}
    pool, nxt = {}, 0
    for z in cells:
        k = drivers_per[z]
        pool[z] = [(nxt + i) % n_cl for i in range(k)]
        nxt += k
    client = np.array([pool[z][rng.integers(len(pool[z]))] for z in part])

    ds = Dataset(feats, cls, part, client, n_parts, 4, "nyc_taxi")
    ds.X = (ds.X - ds.X.mean(0)) / (ds.X.std(0) + 1e-6)
    return ds


def subsample_by_cell(ds, max_per_cell=None, min_per_cell=0, seed=0, verbose=True):
    """Shrink a dataset by CAPPING each (partition, class) cell, never by
    sampling the whole set at random.

    This distinction decides whether the paper's sparse-cell analysis survives.
    Global random sampling scales every cell by the same factor, so a cell with
    4 samples becomes 1 and a cell with 2 contributing users becomes 1 or 0 --
    exactly the cells Section VII-A attributes the method's margin to. Capping
    leaves small cells untouched and only trims the large ones, so the sparse
    structure is preserved while total size comes down.

    max_per_cell  keep at most this many samples per cell (None = no cap)
    min_per_cell  drop cells below this many samples; they cannot support a
                  per-cell accuracy estimate and only add noise to AA and BWT
    """
    rng = np.random.default_rng(seed)
    keep = []
    dropped_cells = 0
    # Group once by cell id instead of scanning the whole array per cell.
    # At 12.6M rows and ~1000 cells the naive form is ~1000 full passes.
    cell_id = ds.part.astype(np.int64) * ds.n_classes + ds.y.astype(np.int64)
    order = np.argsort(cell_id, kind="stable")
    sorted_ids = cell_id[order]
    bounds = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
            idx = order[b0:b1]
            if len(idx) == 0:
                continue
            if len(idx) < min_per_cell:
                dropped_cells += 1
                continue
            if max_per_cell is not None and len(idx) > max_per_cell:
                # keep every contributing client represented before trimming,
                # so capping cannot silently remove a user from a cell
                chosen = []
                for u in np.unique(ds.client[idx]):
                    ui = idx[ds.client[idx] == u]
                    take = max(1, int(round(max_per_cell * len(ui) / len(idx))))
                    chosen.append(rng.choice(ui, min(take, len(ui)), replace=False))
                idx = np.concatenate(chosen)[:max_per_cell]
            keep.append(idx)
    if not keep:
        raise RuntimeError("subsampling removed everything; lower --min-per-cell")
    keep = np.sort(np.concatenate(keep))
    out = Dataset(ds.X[keep], ds.y[keep], ds.part[keep], ds.client[keep],
                  ds.n_partitions, ds.n_classes, ds.name)
    if verbose:
        print(f"[subsample] {len(ds):,} -> {len(out):,} samples "
              f"({100*len(out)/len(ds):.0f}%)"
              + (f", dropped {dropped_cells} cells under {min_per_cell} samples"
                 if dropped_cells else ""))
    return out


def load(cfg):
    return {"synthetic": lambda: make_synthetic(cfg, cfg.seed),
            "geolife": lambda: load_geolife(cfg),
            "nyc_taxi": lambda: load_nyc_taxi(cfg)}[cfg.data.dataset]()


def cache_file(cfg):
    return os.path.join(cfg.out_dir, f"{cfg.data.dataset}_cache.npz")


def load_cached(cfg, verbose=False):
    """Load via an on-disk cache, then apply subsampling.

    Extraction is per-dataset, but a sweep runs the same dataset 70+ times.
    Re-parsing a 12M-row parquet on every run costs more than the training
    does, so the raw extraction is cached once and every later run reads the
    .npz. Subsampling is applied AFTER the cache, so changing --max-per-cell
    does not force a re-extract.
    """
    cp = cache_file(cfg)
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        ds = Dataset(z["X"], z["y"], z["part"], z["client"],
                     int(z["n_partitions"]), int(z["n_classes"]), str(z["name"]))
        if verbose:
            print(f"[cache] {cp}  ({len(ds):,} samples)")
    else:
        ds = load(cfg)
        os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
        np.savez_compressed(cp, X=ds.X, y=ds.y, part=ds.part, client=ds.client,
                            n_partitions=ds.n_partitions,
                            n_classes=ds.n_classes, name=ds.name)
        print(f"[cache] wrote {cp}  ({len(ds):,} samples) -- later runs reuse this")

    mx = getattr(cfg.data, "max_per_cell", None)
    mn = getattr(cfg.data, "min_per_cell", 0)
    if mx or mn:
        ds = subsample_by_cell(ds, mx, mn, seed=0, verbose=verbose)
    return ds


# ------------------------------------------------------------------ scenarios

def stages(ds, scenario, seed=0):
    """Return a list of (classes, partitions) available at each stage.

    C-IL   classes arrive in stages, all partitions present throughout
    G-IL   partitions arrive in stages, all classes present throughout
    CG-IL  both arrive together -- the realistic deployment case
    """
    rng = np.random.default_rng(seed)
    K, P = ds.n_classes, ds.n_partitions
    # At least two classes per stage. A single-class stage is degenerate:
    # during it the model can reach 100% by predicting that one class, so the
    # forgetting curve measures the collapse rather than the method. With
    # K=4 (NYC) that means 2 stages, not 3; with K=7 (GeoLife), 3.
    n_stages = max(2, min(3, K // 2))
    if K >= 7:
        cls_stages = [c for c in CIL_STAGES if all(x < K for x in c)]
    else:
        cls_stages = [sorted(s.tolist())
                      for s in np.array_split(np.arange(K), n_stages)]
    n_stages = len(cls_stages)
    perm = rng.permutation(P)
    part_stages = [np.sort(s).tolist() for s in np.array_split(perm, n_stages)]

    all_c, all_p = list(range(K)), list(range(P))
    if scenario == "cil":
        return [(cls_stages[i], all_p) for i in range(n_stages)]
    if scenario == "gil":
        cum, out = [], []
        for i in range(n_stages):
            cum = cum + part_stages[i]
            out.append((all_c, sorted(cum)))
        return out
    if scenario == "cgil":
        cum, out = [], []
        for i in range(n_stages):
            cum = cum + part_stages[i]
            out.append((cls_stages[i], sorted(cum)))
        return out
    raise ValueError(f"unknown scenario {scenario!r}")


def stage_mask(ds, classes, partitions):
    return np.isin(ds.y, classes) & np.isin(ds.part, partitions)
