"""
Membership inference, in the two families the manuscript now distinguishes.

common    One attack applied identically to every arm: a loss / confidence
          threshold on the shared task model. This is the only family whose
          numbers may be compared across methods, and it is the family that
          contains the `no_replay` control. Without that control the paper
          cannot claim replay ADDS exposure, which is what the abstract
          implies -- the gap E6 closes.

specific  Generator-specific attacks: reconstruction gap against the CVAE
          decoder, discriminator score against the AC-GAN. These describe
          which surface each generator exposes. They use different attacks
          per method, so they are never cross-compared.

Writes the scores.csv schema exp/mia_eval.py reads, which computes AUC with
bootstrap CIs and TPR at low FPR.
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def task_model_scores(model, X, y):
    """Common attack: negative per-sample loss. Members fit better, so a
    higher score means more member-like, which is what mia_eval expects."""
    model.eval()
    out = []
    for i in range(0, len(X), 512):
        xb = torch.tensor(X[i:i + 512])
        yb = torch.tensor(y[i:i + 512])
        loss = F.cross_entropy(model(xb), yb, reduction="none")
        out.append((-loss).numpy())
    model.train()
    return np.concatenate(out)


@torch.no_grad()
def cvae_recon_scores(model, generator, X, y, part, draws=8, raw_space=False):
    """Reconstruction gap against the CVAE decoder.

    Needs the feature extractor, both CVAE paths, and the candidate's (p, c)
    label -- a conditional decoder cannot reconstruct without its condition.
    Averaged over `draws` latent samples, since z is stochastic.
    Members reconstruct better (lower error), so the score is negated.

    `raw_space` selects what the decoder is asked to reconstruct. A
    feature-space CVAE was fitted to the 32-d penultimate feature, so the
    candidate must be passed through the backbone first. An input-space CVAE
    was fitted to the raw observation itself, so x goes in unchanged -- and
    the attack it exposes is correspondingly different in kind, since what
    leaks is the observation rather than an embedding of it.
    """
    model.eval()
    generator.eval()
    out = []
    for i in range(0, len(X), 512):
        xb = torch.tensor(X[i:i + 512])
        yb = torch.tensor(y[i:i + 512])
        pb = torch.tensor(part[i:i + 512])
        h = xb if raw_space else model.features(xb)
        err = generator.recon_error(h, pb, yb, draws=draws)
        out.append((-err).numpy())
    model.train()
    generator.train()
    return np.concatenate(out)


@torch.no_grad()
def acgan_d_scores(model, generator, X):
    """Discriminator adversarial score. D is trained to separate real from
    synthetic, so it is directly usable as a membership scorer whenever D is
    shared or aggregated -- the asymmetry Section V-C describes."""
    model.eval()
    generator.eval()
    out = []
    for i in range(0, len(X), 512):
        h = model.features(torch.tensor(X[i:i + 512]))
        out.append(generator.d_score(h).numpy())
    model.train()
    generator.train()
    return np.concatenate(out)


def balanced_sample(train_ds, test_ds, n=1500, seed=0):
    """1:1 member / non-member split at the record level.

    Record-level, not user-level: given a candidate segment and its (p, c)
    label, was it in some client's training set. The threat model paragraph
    in Section V states this; keep the two in agreement.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(train_ds), len(test_ds))
    mi = rng.choice(len(train_ds), n, replace=False)
    ni = rng.choice(len(test_ds), n, replace=False)
    return (
        (train_ds.X[mi], train_ds.y[mi], train_ds.part[mi]),
        (test_ds.X[ni], test_ds.y[ni], test_ds.part[ni]),
    )


def evaluate(result, n=1500, seed=0):
    """Score rows for one finished run, both families."""
    model = result["model"]
    method = result["method_obj"]
    train_ds, test_ds = result["data"]
    arm = result["method"]

    (mx, my, mp), (nx, ny, np_) = balanced_sample(train_ds, test_ds, n, seed)
    rows = []

    def emit(attack, family, s_mem, s_non):
        for s in s_mem:
            rows.append((arm, attack, family, 1, float(s)))
        for s in s_non:
            rows.append((arm, attack, family, 0, float(s)))

    # --- family: common -----------------------------------------------
    emit("loss_threshold", "common",
         task_model_scores(model, mx, my), task_model_scores(model, nx, ny))

    # --- family: specific ----------------------------------------------
    gen = getattr(method, "generator", None)
    if gen is not None and hasattr(gen, "recon_error"):
        rs = bool(getattr(method, "raw_space", False))
        emit("recon_gap", "specific",
             cvae_recon_scores(model, gen, mx, my, mp, raw_space=rs),
             cvae_recon_scores(model, gen, nx, ny, np_, raw_space=rs))
    elif gen is not None and hasattr(gen, "d_score"):
        emit("d_score", "specific",
             acgan_d_scores(model, gen, mx), acgan_d_scores(model, gen, nx))

    return rows
