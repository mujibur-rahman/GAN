#!/usr/bin/env python3
"""Figure 1: replay budget against old- and new-class accuracy, NYC Taxi.

The one chart the paper needs. It carries an argument a table does not: the
two curves cross, so R is not a hyperparameter to tune but a dial between two
failure modes -- and the horizontal line marks where feature-space replay sits,
unable to move along the curve at all because its gradient never reaches the
backbone.

    python3 fig_budget.py            -> fig_budget.pdf (+ .png to eyeball)

Numbers are from analyze.py retention on r_nyc_R{32,64,128} (3 seeds) and
r_nyc (5 seeds, R=192). Edit DATA if you rerun with different budgets.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# R : (old-class accuracy %, new-class accuracy %)
DATA = {32: (39.08, 23.42),
        64: (46.78, 13.48),
        128: (49.86, 6.26),
        192: (50.36, 4.40)}

# feature-space replay, same dataset: retains nothing at any budget
FEATURE_OLD = 0.00

R = sorted(DATA)
old = [DATA[r][0] for r in R]
new = [DATA[r][1] for r in R]

# IEEE single column is 3.5in; keep the font at 8pt so it stays legible
# after LaTeX scales nothing.
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(3.5, 1.95))

ax.plot(R, old, marker="o", ms=4, lw=1.6, color="#1b6ca8",
        label="old classes (retention)")
ax.plot(R, new, marker="s", ms=4, lw=1.6, color="#c0392b",
        label="new classes (plasticity)")
ax.axhline(FEATURE_OLD, ls=":", lw=1.2, color="#555555")
ax.text(33, 1.6, "feature-space replay, old classes: 0.0% at every $R$",
        fontsize=6.8, color="#555555", va="bottom")

ax.set_xscale("log", base=2)
ax.set_xticks(R)
ax.set_xticklabels([str(r) for r in R])
ax.set_xlabel("replay budget $R$ (generated samples / client / round)")
ax.set_ylabel("final accuracy (%)")
ax.set_ylim(-3, 58)
ax.grid(axis="y", lw=0.4, alpha=0.35)
ax.legend(frameon=False, loc="center right")

fig.tight_layout(pad=0.3)
fig.savefig("fig_budget.pdf", bbox_inches="tight")
fig.savefig("fig_budget.png", bbox_inches="tight")
print("wrote fig_budget.pdf and fig_budget.png")
