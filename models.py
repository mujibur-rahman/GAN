"""
Task model, Geo-CVAE, and the two GAN baselines.

Two implementation details here are the ones the manuscript review flagged,
so they are enforced in code rather than left to convention:

1. Replay features enter at the classifier head. `TaskModel.head` is called
   directly on synthetic features; they never pass through the backbone. That
   is why the backbone gets no replay gradient and needs the distillation
   anchor, and why removing distillation costs accuracy.

2. The CVAE conditioning vector is [e_geo ; e_cls]. With geo_cond=False the
   geo half is replaced by a zero vector of the same width, so the no-geo
   ablation keeps the parameter count and input dimensions identical. Dropping
   the columns instead would confound "no geo information" with "smaller
   model", and a reviewer would be right to reject that comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskModel(nn.Module):
    """3-layer MLP backbone -> 32-d feature h, then a linear classifier."""

    def __init__(self, cfg, n_classes):
        super().__init__()
        m = cfg.model
        h1, h2 = m.hidden
        # The penultimate layer is deliberately PRE-activation (no trailing
        # ReLU). The CVAE reconstructs h under a Gaussian likelihood (MSE), and
        # ReLU features are ~94% exact zeros on this data -- a point mass at 0
        # that a Gaussian decoder cannot represent, so generated features land
        # off-manifold and corrupt the classifier head they are replayed into.
        # A real-valued h is also better conditioned for the KL term.
        self.backbone = nn.Sequential(
            nn.Linear(m.in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, m.feat_dim),
        )
        self.head = nn.Linear(m.feat_dim, n_classes)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))

    def forward_features(self, h):
        """Classify a feature directly. This is the replay path."""
        return self.head(h)


class GeoCVAE(nn.Module):
    """Conditional VAE over penultimate features, conditioned on (p, c).

    Parameter budget at the defaults is 3,464, matching Table I:
        encoder  (32+28)*20+20 + 2*(20*10+10)      = 1,640
        decoder  (10+28)*20+20 + 20*32+32          = 1,452
        geo table 18*16 = 288,  class table 7*12 = 84
    """

    def __init__(self, cfg, n_partitions, n_classes, geo_cond=True):
        super().__init__()
        m = cfg.model
        self.geo_cond = geo_cond
        self.d_g, self.d_k, self.d_z = m.d_g, m.d_k, m.d_z
        self.beta = m.beta

        self.e_geo = nn.Embedding(n_partitions, m.d_g)
        self.e_cls = nn.Embedding(n_classes, m.d_k)
        d_cond = m.d_g + m.d_k

        self.enc = nn.Linear(m.feat_dim + d_cond, m.d_hidden)
        self.mu = nn.Linear(m.d_hidden, m.d_z)
        self.logvar = nn.Linear(m.d_hidden, m.d_z)
        self.dec1 = nn.Linear(m.d_z + d_cond, m.d_hidden)
        self.dec2 = nn.Linear(m.d_hidden, m.feat_dim)

    def cond(self, p, c):
        eg = self.e_geo(p)
        if not self.geo_cond:
            eg = torch.zeros_like(eg)      # same width, no geo information
        return torch.cat([eg, self.e_cls(c)], -1)

    def encode(self, h, p, c):
        z = F.relu(self.enc(torch.cat([h, self.cond(p, c)], -1)))
        return self.mu(z), self.logvar(z)

    def decode(self, z, p, c):
        return self.dec2(F.relu(self.dec1(torch.cat([z, self.cond(p, c)], -1))))

    def forward(self, h, p, c):
        mu, logvar = self.encode(h, p, c)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, p, c), mu, logvar

    def loss(self, h, p, c):
        """Geo-conditioned ELBO (Eq. 5): reconstruction + beta * KL."""
        hhat, mu, logvar = self(h, p, c)
        rec = F.mse_loss(hhat, h, reduction="none").sum(-1).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
        return rec + self.beta * kl, rec.detach(), kl.detach()

    @torch.no_grad()
    def sample(self, p, c):
        """Synthetic features for prior cells (Eq. 6): z ~ N(0, I)."""
        z = torch.randn(len(p), self.d_z)
        return self.decode(z, p, c)

    @torch.no_grad()
    def recon_error(self, h, p, c, draws=8):
        """Per-sample reconstruction gap, averaged over latent draws.

        z is sampled, so a single draw is a noisy membership score. The
        manuscript now says this is averaged; `draws` is that number.
        """
        errs = []
        for _ in range(draws):
            hhat, _, _ = self(h, p, c)
            errs.append((hhat - h).pow(2).sum(-1))
        return torch.stack(errs).mean(0)

    def n_params(self):
        return sum(q.numel() for q in self.parameters())


class FeatureACGAN(nn.Module):
    """Lightweight feature-space AC-GAN baseline: generator + discriminator
    with auxiliary geo and class heads, spectral norm on D."""

    def __init__(self, cfg, n_partitions, n_classes):
        super().__init__()
        m = cfg.model
        self.d_z = m.d_z
        d_cond = m.d_g + m.d_k
        self.e_geo = nn.Embedding(n_partitions, m.d_g)
        self.e_cls = nn.Embedding(n_classes, m.d_k)

        self.g1 = nn.Linear(m.d_z + d_cond, m.d_hidden)
        self.g2 = nn.Linear(m.d_hidden, m.feat_dim)

        sn = nn.utils.parametrizations.spectral_norm
        self.d1 = sn(nn.Linear(m.feat_dim, m.d_hidden))
        self.d_adv = sn(nn.Linear(m.d_hidden, 1))
        self.d_cls = nn.Linear(m.d_hidden, n_classes)
        self.d_geo = nn.Linear(m.d_hidden, n_partitions)

    def cond(self, p, c):
        return torch.cat([self.e_geo(p), self.e_cls(c)], -1)

    def generate(self, p, c):
        z = torch.randn(len(p), self.d_z)
        return self.g2(F.relu(self.g1(torch.cat([z, self.cond(p, c)], -1))))

    def discriminate(self, h):
        f = F.relu(self.d1(h))
        return self.d_adv(f).squeeze(-1), self.d_cls(f), self.d_geo(f)

    @torch.no_grad()
    def d_score(self, h):
        """Discriminator adversarial score: the membership signal of Sec. V-B."""
        return self.discriminate(h)[0]

    def gen_params(self):
        return (list(self.g1.parameters()) + list(self.g2.parameters())
                + list(self.e_geo.parameters()) + list(self.e_cls.parameters()))

    def disc_params(self):
        return (list(self.d1.parameters()) + list(self.d_adv.parameters())
                + list(self.d_cls.parameters()) + list(self.d_geo.parameters()))

    def n_params_generator(self):
        return sum(q.numel() for q in self.gen_params())

    def n_params_total(self):
        return sum(q.numel() for q in self.parameters())


class RawDGR(nn.Module):
    """Deep Generative Replay in raw input space, no geo-conditioning.

    Deliberately sized the way the paper's baseline is (~2.1M parameters) so
    the cost comparison is against something representative rather than a
    strawman scaled down to be cheap.
    """

    def __init__(self, cfg, n_classes, width=820):
        super().__init__()
        m = cfg.model
        self.d_z = 64
        self.e_cls = nn.Embedding(n_classes, 32)
        self.g = nn.Sequential(
            nn.Linear(self.d_z + 32, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, m.in_dim),
        )
        self.d = nn.Sequential(
            nn.Linear(m.in_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def generate(self, c):
        z = torch.randn(len(c), self.d_z)
        return self.g(torch.cat([z, self.e_cls(c)], -1))

    def n_params_total(self):
        return sum(q.numel() for q in self.parameters())


def report_sizes(cfg, n_partitions, n_classes):
    """Parameter accounting on one consistent basis, for Table I.

    Section VI-B now states that generator-only counts are not comparable
    because the GANs also train a discriminator. Both bases are returned so
    the table can say which it reports.
    """
    cvae = GeoCVAE(cfg, n_partitions, n_classes)
    acgan = FeatureACGAN(cfg, n_partitions, n_classes)
    dgr = RawDGR(cfg, n_classes)
    return {
        "geo_cvae_train": cvae.n_params(),
        "geo_cvae_replay_only": sum(
            q.numel() for q in list(cvae.dec1.parameters())
            + list(cvae.dec2.parameters()) + list(cvae.e_geo.parameters())
            + list(cvae.e_cls.parameters())),
        "acgan_generator_only": acgan.n_params_generator(),
        "acgan_train_total": acgan.n_params_total(),
        "dgr_train_total": dgr.n_params_total(),
    }
