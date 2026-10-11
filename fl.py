"""
Federated loop, continual-learning methods, and per-cell metric logging.

Three things the manuscript review changed are implemented here as described,
not as originally written:

  L_distil  is evaluated on the client's CURRENT REAL batch, anchoring the
            backbone between consecutive rounds. The original equation summed
            over the replay batch, which contains features and no inputs, so
            f_phi(x) was not computable. See `local_train`.

  drift     q^(t) for a historical cell is estimated by running the current
            model over features REPLAYED for that cell, because no real data
            for it exists at round t. `DriftTracker.update` does exactly that
            and the limitation is real: the signal cannot see decoder
            degradation. Recorded in Limitation 2.

  tau       tau_first (for BWT) and tau_last (for drift) are tracked
            separately. They were one symbol in the original draft.
"""

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

from models import TaskModel, GeoCVAE, FeatureACGAN, RawDGR


# --------------------------------------------------------------------- utils

def fedavg(states, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            out[k] = sum(s[k] * float(wi) for s, wi in zip(states, w))
        else:
            out[k] = states[0][k]
    return out


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    return float(np.sum((p / p.sum()) * np.log((p / p.sum()) / (q / q.sum()))))


# ------------------------------------------------------------- drift tracker

class DriftTracker:
    """Per-cell drift score (Eq. 9) and replay allocation (Eq. 10).

    Retains only a K-bin class-posterior histogram per cell, recorded at
    tau_last. That is aggregate state, not raw data, which is what lets the
    mechanism run without violating the no-retention claim.
    """

    def __init__(self, n_partitions, n_classes, cfg):
        self.K = n_classes
        self.hist = {}          # (p, c) -> histogram at tau_last
        self.tau_last = {}
        self.score = {}
        self.div = (js_divergence if cfg.train.drift_divergence == "js"
                    else kl_divergence)
        self.eps = cfg.train.drift_eps

    def record(self, cell, posterior, round_idx):
        self.hist[cell] = np.asarray(posterior, float)
        self.tau_last[cell] = round_idx

    @torch.no_grad()
    def update(self, model, generator, cells, round_idx):
        """Refresh delta_{p,c} for historical cells.

        The current posterior is computed on GENERATED features, since the
        cell supplies no real data now. The drift signal therefore tracks the
        classifier relative to the generator, and is blind to the decoder
        drifting on its own -- the honest limitation stated in the paper.
        """
        if generator is None:
            return
        model.eval()
        for (p, c) in cells:
            if (p, c) not in self.hist:
                continue
            pt = torch.full((48,), p, dtype=torch.long)
            ct = torch.full((48,), c, dtype=torch.long)
            h = generator.sample(pt, ct) if hasattr(generator, "sample") \
                else generator.generate(pt, ct)
            probs = F.softmax(model.forward_features(h), -1).mean(0).numpy()
            self.score[(p, c)] = self.div(probs, self.hist[(p, c)])
        model.train()

    def allocate(self, cells, budget, selective=True):
        """Split the replay budget R across historical cells."""
        cells = list(cells)
        if not cells:
            return {}
        if not selective:
            n = budget // len(cells)
            return {cell: n for cell in cells if n > 0}
        w = np.array([self.score.get(cell, 0.0) + self.eps for cell in cells])
        w = w / w.sum()
        counts = np.floor(w * budget).astype(int)
        return {cell: int(n) for cell, n in zip(cells, counts) if n > 0}


# ------------------------------------------------------------------- methods

class Method:
    """Base: plain federated training with no continual-learning component."""
    name = "none"
    has_generator = False

    def __init__(self, cfg, ds):
        self.cfg = cfg
        self.ds = ds
        self.generator = None

    def local_extra_loss(self, model, xb, yb, pb, state):
        return torch.zeros(())

    def replay_batch(self, model, client_cells, hist_cells):
        return None

    def after_round(self, model, round_idx, hist_cells):
        pass

    def end_stage(self, hist_cells):
        pass

    def generator_state(self):
        return None

    def load_generator_state(self, st):
        pass


class EWC(Method):
    name = "ewc"

    def __init__(self, cfg, ds):
        super().__init__(cfg, ds)
        self.fisher = None
        self.anchor = None

    def local_extra_loss(self, model, xb, yb, pb, state):
        if self.fisher is None:
            return torch.zeros(())
        pen = 0.0
        for (n, q) in model.named_parameters():
            if n in self.fisher:
                pen = pen + (self.fisher[n] * (q - self.anchor[n]).pow(2)).sum()
        return self.cfg.train.ewc_lambda * pen

    def after_round(self, model, round_idx, hist_cells):
        pass

    def consolidate(self, model, X, y):
        """Diagonal Fisher at the end of a stage."""
        model.eval()
        fisher = {n: torch.zeros_like(q) for n, q in model.named_parameters()}
        xb = torch.tensor(X[:512])
        yb = torch.tensor(y[:512])
        for i in range(0, len(xb), 32):
            model.zero_grad()
            loss = F.cross_entropy(model(xb[i:i + 32]), yb[i:i + 32])
            loss.backward()
            for n, q in model.named_parameters():
                if q.grad is not None:
                    fisher[n] += q.grad.detach().pow(2)
        n_b = max(1, len(xb) // 32)
        self.fisher = {n: f / n_b for n, f in fisher.items()}
        self.anchor = {n: q.detach().clone() for n, q in model.named_parameters()}
        model.zero_grad()
        model.train()


class ExperienceReplay(Method):
    """5% raw exemplar buffer per class per client. Stores raw trajectory
    features, which is exactly what the SC privacy model objects to."""
    name = "er"

    def __init__(self, cfg, ds):
        super().__init__(cfg, ds)
        self.buf_x, self.buf_y = [], []

    def add(self, X, y):
        n = max(1, int(len(X) * self.cfg.train.er_buffer_frac))
        idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
        self.buf_x.append(X[idx])
        self.buf_y.append(y[idx])

    def raw_replay(self, bs):
        if not self.buf_x:
            return None
        X = np.concatenate(self.buf_x)
        y = np.concatenate(self.buf_y)
        if len(X) == 0:
            return None
        idx = np.random.default_rng().choice(len(X), min(bs, len(X)), replace=False)
        return torch.tensor(X[idx]), torch.tensor(y[idx])


class GeoCVAEReplay(Method):
    """The proposed method."""
    name = "geo_cvae"
    has_generator = True

    def __init__(self, cfg, ds, geo_cond=True, drift_sel=True, distil=True):
        super().__init__(cfg, ds)
        self.geo_cond, self.drift_sel, self.distil = geo_cond, drift_sel, distil
        self.generator = GeoCVAE(cfg, ds.n_partitions, ds.n_classes, geo_cond)
        self.opt = torch.optim.Adam(self.generator.parameters(), lr=cfg.train.lr_gen)
        self.drift = DriftTracker(ds.n_partitions, ds.n_classes, cfg)
        self.prev_backbone = None
        self.prev_gen = None         # frozen generator from the previous stage
        self.hist_cells = []

    def local_extra_loss(self, model, xb, yb, pb, state):
        """Feature distillation on the CURRENT REAL batch (corrected Eq. 8)."""
        if not self.distil or self.prev_backbone is None:
            return torch.zeros(())
        with torch.no_grad():
            old = self.prev_backbone(xb)
        new = model.features(xb)
        return self.cfg.train.lambda_d * F.mse_loss(new, old, reduction="none").sum(-1).mean()

    def train_generator(self, model, xb, yb, pb):
        """Train the CVAE on current real features PLUS its own prior-stage
        output (scholar-student self-replay, as in DGR).

        Without the second term the generator is only ever fitted to the
        current stage, so it forgets historical cells as fast as the task
        model does and replay actively injects wrong features. Self-replay is
        what makes a generator a memory rather than a model of the present.
        """
        with torch.no_grad():
            h = model.features(xb)
        p_all, c_all, h_all = pb, yb, h

        if self.prev_gen is not None and self.hist_cells:
            n = min(len(xb), self.cfg.train.replay_budget)
            sel = [self.hist_cells[i] for i in
                   np.random.randint(0, len(self.hist_cells), n)]
            pp = torch.tensor([p for p, _ in sel], dtype=torch.long)
            cc = torch.tensor([c for _, c in sel], dtype=torch.long)
            with torch.no_grad():
                h_old = self.prev_gen.sample(pp, cc)
            h_all = torch.cat([h, h_old], 0)
            p_all = torch.cat([pb, pp], 0)
            c_all = torch.cat([yb, cc], 0)

        loss, rec, kl = self.generator.loss(h_all, p_all, c_all)
        self.opt.zero_grad()
        (self.cfg.train.lambda_r * loss).backward()
        self.opt.step()
        return float(rec)

    def end_stage(self, hist_cells):
        """Freeze the generator so the next stage can rehearse against it."""
        self.hist_cells = list(hist_cells)
        self.prev_gen = copy.deepcopy(self.generator).eval()
        for q in self.prev_gen.parameters():
            q.requires_grad_(False)

    def replay_batch(self, model, client_cells, hist_cells):
        """Synthetic features for prior cells, injected at the classifier head."""
        alloc = self.drift.allocate(hist_cells, self.cfg.train.replay_budget,
                                    selective=self.drift_sel)
        if not alloc:
            return None
        ps, cs = [], []
        for (p, c), n in alloc.items():
            ps += [p] * n
            cs += [c] * n
        if not ps:
            return None
        pt = torch.tensor(ps, dtype=torch.long)
        ct = torch.tensor(cs, dtype=torch.long)
        h = self.generator.sample(pt, ct)
        return h, ct, pt

    def after_round(self, model, round_idx, hist_cells):
        self.drift.update(model, self.generator, hist_cells, round_idx)
        self.prev_backbone = copy.deepcopy(model.backbone).eval()
        for q in self.prev_backbone.parameters():
            q.requires_grad_(False)

    def generator_state(self):
        return copy.deepcopy(self.generator.state_dict())

    def load_generator_state(self, st):
        self.generator.load_state_dict(st)


class ACGANReplay(Method):
    name = "ac_gan"
    has_generator = True

    def __init__(self, cfg, ds):
        super().__init__(cfg, ds)
        self.generator = FeatureACGAN(cfg, ds.n_partitions, ds.n_classes)
        self.og = torch.optim.Adam(self.generator.gen_params(), lr=cfg.train.lr_gen,
                                   betas=(0.5, 0.999))
        self.od = torch.optim.Adam(self.generator.disc_params(),
                                   lr=cfg.train.lr_gen * cfg.train.acgan_lr_ratio,
                                   betas=(0.5, 0.999))
        self.drift = DriftTracker(ds.n_partitions, ds.n_classes, cfg)
        self.prev_gen = None
        self.hist_cells = []

    def end_stage(self, hist_cells):
        """Same self-replay snapshot as the CVAE arm, so the comparison is
        between generator families and not between one that rehearses and one
        that does not."""
        self.hist_cells = list(hist_cells)
        self.prev_gen = copy.deepcopy(self.generator).eval()
        for q in self.prev_gen.parameters():
            q.requires_grad_(False)

    def train_generator(self, model, xb, yb, pb):
        """Alternating D / G steps -- the structural reason this is slower."""
        with torch.no_grad():
            h_real = model.features(xb)
        if self.prev_gen is not None and self.hist_cells:
            n = min(len(xb), self.cfg.train.replay_budget)
            sel = [self.hist_cells[i] for i in
                   np.random.randint(0, len(self.hist_cells), n)]
            pp = torch.tensor([p for p, _ in sel], dtype=torch.long)
            cc = torch.tensor([c for _, c in sel], dtype=torch.long)
            with torch.no_grad():
                h_old = self.prev_gen.generate(pp, cc)
            h_real = torch.cat([h_real, h_old], 0)
            yb = torch.cat([yb, cc], 0)
            pb = torch.cat([pb, pp], 0)
        g = self.generator

        for _ in range(self.cfg.train.acgan_d_steps):
            h_fake = g.generate(pb, yb).detach()
            adv_r, cls_r, geo_r = g.discriminate(h_real)
            adv_f, _, _ = g.discriminate(h_fake)
            d_loss = (F.binary_cross_entropy_with_logits(adv_r, torch.ones_like(adv_r))
                      + F.binary_cross_entropy_with_logits(adv_f, torch.zeros_like(adv_f))
                      + F.cross_entropy(cls_r, yb) + F.cross_entropy(geo_r, pb))
            self.od.zero_grad(); d_loss.backward(); self.od.step()

        h_fake = g.generate(pb, yb)
        adv_f, cls_f, geo_f = g.discriminate(h_fake)
        g_loss = (F.binary_cross_entropy_with_logits(adv_f, torch.ones_like(adv_f))
                  + F.cross_entropy(cls_f, yb) + F.cross_entropy(geo_f, pb))
        self.og.zero_grad(); g_loss.backward(); self.og.step()
        return float(g_loss)

    def replay_batch(self, model, client_cells, hist_cells):
        alloc = self.drift.allocate(hist_cells, self.cfg.train.replay_budget, True)
        if not alloc:
            return None
        ps, cs = [], []
        for (p, c), n in alloc.items():
            ps += [p] * n
            cs += [c] * n
        if not ps:
            return None
        pt = torch.tensor(ps, dtype=torch.long)
        ct = torch.tensor(cs, dtype=torch.long)
        with torch.no_grad():
            h = self.generator.generate(pt, ct)
        return h, ct, pt

    def after_round(self, model, round_idx, hist_cells):
        self.drift.update(model, self.generator, hist_cells, round_idx)

    def generator_state(self):
        return copy.deepcopy(self.generator.state_dict())

    def load_generator_state(self, st):
        self.generator.load_state_dict(st)


class DGRReplay(Method):
    """Raw-space GAN replay, no geo-conditioning. Runs under the same
    per-round budget as everything else, which is the comparison the paper
    makes and should state."""
    name = "dgr"
    has_generator = True

    def __init__(self, cfg, ds):
        super().__init__(cfg, ds)
        self.generator = RawDGR(cfg, ds.n_classes)
        self.og = torch.optim.Adam(
            list(self.generator.g.parameters()) + list(self.generator.e_cls.parameters()),
            lr=cfg.train.lr_gen, betas=(0.5, 0.999))
        self.od = torch.optim.Adam(self.generator.d.parameters(),
                                   lr=cfg.train.lr_gen * 0.5, betas=(0.5, 0.999))
        self.seen_classes = set()

    def train_generator(self, model, xb, yb, pb):
        g = self.generator
        x_fake = g.generate(yb).detach()
        dr, df = g.d(xb).squeeze(-1), g.d(x_fake).squeeze(-1)
        d_loss = (F.binary_cross_entropy_with_logits(dr, torch.ones_like(dr))
                  + F.binary_cross_entropy_with_logits(df, torch.zeros_like(df)))
        self.od.zero_grad(); d_loss.backward(); self.od.step()

        x_fake = g.generate(yb)
        adv = g.d(x_fake).squeeze(-1)
        g_loss = F.binary_cross_entropy_with_logits(adv, torch.ones_like(adv))
        self.og.zero_grad(); g_loss.backward(); self.og.step()
        self.seen_classes.update(yb.tolist())
        return float(g_loss)

    def replay_batch(self, model, client_cells, hist_cells):
        """Raw-space replay produces INPUTS, so it goes through the backbone.
        No geo-conditioning: the generator cannot be asked for a partition."""
        hist_classes = sorted({c for (_, c) in hist_cells})
        if not hist_classes:
            return None
        n = self.cfg.train.replay_budget
        ct = torch.tensor(np.random.choice(hist_classes, n), dtype=torch.long)
        with torch.no_grad():
            x = self.generator.generate(ct)
        return ("raw", x, ct)

    def generator_state(self):
        return copy.deepcopy(self.generator.state_dict())

    def load_generator_state(self, st):
        self.generator.load_state_dict(st)


def build_method(name, cfg, ds):
    if name == "ewc":
        return EWC(cfg, ds)
    if name == "er":
        return ExperienceReplay(cfg, ds)
    if name == "dgr":
        return DGRReplay(cfg, ds)
    if name == "ac_gan":
        return ACGANReplay(cfg, ds)
    if name in ("geo_cvae", "geo_cvae_full"):
        return GeoCVAEReplay(cfg, ds)
    if name == "no_geo_conditioning":
        return GeoCVAEReplay(cfg, ds, geo_cond=False)
    if name == "no_drift_selective":
        return GeoCVAEReplay(cfg, ds, drift_sel=False)
    if name == "no_distillation":
        return GeoCVAEReplay(cfg, ds, distil=False)
    if name == "no_replay":
        return Method(cfg, ds)
    raise ValueError(f"unknown method {name!r}")
