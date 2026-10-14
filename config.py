"""
Every hyperparameter the paper reports, in one place.

The manuscript currently carries TODOs for beta, d_z, d_g, d_k, lambda_r,
lambda_d, the divergence D and epsilon. Those are the fields below. Whatever
you change here is what the paper should state, so treat this file as the
source of truth for Section IV and fill the manuscript from it rather than
the other way round.

The CVAE dimensions default to the one configuration that reproduces the
3,464-parameter figure already in Table I at P=18, K=7, d_feat=32. Change
them only if the real implementation differs, and expect Table I to move.
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class ModelConfig:
    # Task model. "3-layer MLP, output dimension 32" (Section VI-B).
    in_dim: int = 7                 # GeoLife motion features
    hidden: tuple = (64, 64)
    feat_dim: int = 32              # penultimate feature h

    # Geo-CVAE. These four reproduce 3,464 params at P=18, K=7.
    d_hidden: int = 20
    d_z: int = 10
    d_g: int = 16                   # geo-partition embedding
    d_k: int = 12                   # task-class embedding
    beta: float = 0.25              # KL weight; < 1 to favour reconstruction


@dataclass
class TrainConfig:
    rounds_per_stage: int = 15
    local_epochs: int = 3
    batch_size: int = 32
    lr_task: float = 5e-3
    lr_gen: float = 2e-3

    # Loss weights in L_w = L_task + lambda_r*L_CVAE + lambda_d*L_distil
    lambda_r: float = 1.0
    lambda_d: float = 1.0

    # Replay
    replay_budget: int = 192        # R, synthetic features per client per round
    drift_divergence: str = "js"    # D(.,.): "js" or "kl"
    drift_eps: float = 1e-3         # epsilon, prevents zero allocation
    drift_update_every: int = 1     # rounds between drift-score refreshes

    # Freeze the backbone after this stage index (0 = after the first stage).
    # REMIND, which this method's feature replay comes from, keeps the feature
    # extractor fixed after base initialisation ("theta_G is no longer
    # plastic"). Feature replay is only valid while the feature space it was
    # fitted to still exists; if the backbone keeps moving, replayed features
    # describe a space the classifier has left. None = never freeze.
    freeze_backbone_after: int = None

    # AC-GAN baseline
    acgan_d_steps: int = 1
    acgan_lr_ratio: float = 0.5     # D lr = lr_gen * ratio

    # EWC / ER baselines
    ewc_lambda: float = 100.0
    er_buffer_frac: float = 0.05    # 5% raw exemplars per class per client


@dataclass
class DataConfig:
    dataset: str = "synthetic"      # synthetic | geolife | nyc_taxi | yelp
    path: str = ""                  # required for geolife / nyc_taxi / yelp
    n_clients: int = 73             # GeoLife labelled users
    n_partitions: int = 18          # merged ring-road cells
    n_classes: int = 7
    sparse_frac: float = 0.40       # share of cells with < 3 contributing users
    samples_per_user: int = 220
    # Subsampling, applied after the cache. Cap per (partition, class) cell --
    # never a global random sample, which destroys sparse cells.
    max_per_cell: int = None
    min_per_cell: int = 0

    # Yelp only. yelp_dims sets the target input width: the loader keeps the
    # (yelp_dims - 6) most frequent non-leaky business attributes and adds six
    # location/rating/time features. 96 puts the input well above feat_dim=32,
    # which is the condition the mobility datasets cannot test -- there,
    # input-space replay is both cheaper and more stable, so the two effects
    # are confounded. yelp_max_reviews caps the review stream; the full file is
    # ~7M lines and parsing all of it is not needed to fill 73 workers.
    yelp_dims: int = 96
    yelp_max_reviews: int = 400_000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    scenario: str = "cgil"          # cil | gil | cgil
    seed: int = 0
    out_dir: str = "results"

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    def summary(self):
        m, t = self.model, self.train
        return (f"beta={m.beta} d_z={m.d_z} d_g={m.d_g} d_k={m.d_k} "
                f"d_hidden={m.d_hidden} | lambda_r={t.lambda_r} "
                f"lambda_d={t.lambda_d} | D={t.drift_divergence} "
                f"eps={t.drift_eps} R={t.replay_budget}")


# Ablation arms. Each flips one mechanism off; everything else is held fixed,
# which is what makes the comparison an ablation rather than two models.
ARMS = {
    "geo_cvae_full":       dict(geo_cond=True,  drift_sel=True,  distil=True),
    "no_geo_conditioning": dict(geo_cond=False, drift_sel=True,  distil=True),
    "no_drift_selective":  dict(geo_cond=True,  drift_sel=False, distil=True),
    "no_distillation":     dict(geo_cond=True,  drift_sel=True,  distil=False),
}
