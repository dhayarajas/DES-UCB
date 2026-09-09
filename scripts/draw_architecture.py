#!/usr/bin/env python3
"""Draw paper/figures/fig00_architecture.png (block diagram of DES-UCB) with matplotlib.

The figure stands in for a hand-drawn diagram.  Replace the PNG with the hand-drawn version at
the same path and the LaTeX build picks it up unchanged.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "fig00_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

GOLD, GREY, PURPLE = "#b8860b", "#555555", "#6a3d9a"
fig, ax = plt.subplots(figsize=(14, 6.4), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(-0.9, 6.4)
ax.axis("off")


def box(x, y, w, h, title, sub="", fc="#f4f4f4", ec="black", lw=1.4, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
    ax.text(
        x + w / 2, y + h / 2 + (0.17 if sub else 0), title, ha="center", va="center", fontsize=fs, fontweight="bold"
    )
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center", fontsize=8.5, color="#333333")


def arrow(p, q, color="black", ls="-", lw=1.5, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def label(x, y, s, color="black", ha="center", va="center", fs=8.5):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
    )


# ---- row 1 (top): suppression --------------------------------------------------------------
box(
    3.2,
    5.05,
    10.5,
    1.05,
    "Suppression path (runs once per detected drift)",
    r"$w\leftarrow\max(w_{\min},\lfloor w/2\rfloor)$    purge F rows with $s<t-w+1$    "
    r"reset P credit in the switch    force F for $B$ rounds    discard open BOB block",
    fc="#fff3cd",
    ec=GOLD,
    lw=1.8,
)

# ---- row 2: prior -> P source -----------------------------------------------------------------
box(0.3, 3.6, 2.1, 0.95, "Prior rows", r"$D_{\mathrm{prior}}=(X_P,r_P)$", fc="#e8eef8")
box(3.2, 3.6, 2.7, 0.95, "P source", r"ridge fit on $D_{\mathrm{prior}}$, frozen", fc="#dbe6f7")

# ---- row 3: candidates -> switch -> argmax -----------------------------------------------------
box(0.3, 2.3, 2.1, 0.95, "Candidates", r"$X_t\in\mathbb{R}^{K\times d}$", fc="#ffffff")
box(6.9, 2.3, 2.9, 0.95, "Evidence switch", r"EXP3 over $\{P,F\}$, clipped gain $g_t$", fc="#fbe8c8")
box(
    10.8,
    2.3,
    2.9,
    0.95,
    "UCB score, argmax",
    r"$a_t=\arg\max_a\ \hat\theta_{m_t}^\top x_a+\beta\|x_a\|_{V_{m_t}^{-1}}$",
    fc="#ffffff",
)

# ---- row 4: feedback -> F source -------------------------------------------------------------------
box(0.3, 1.0, 2.1, 0.95, "Live feedback", r"$(t,x_t,r_t)$", fc="#e8f6e8")
box(3.2, 1.0, 2.7, 0.95, "F source", r"sliding window $w$, bounded history, purge", fc="#dbeedb")

# ---- row 5 (bottom): detector, BOB -----------------------------------------------------------------
box(6.9, 0.2, 2.9, 0.95, "Drift detector", r"Page-Hinkley or ADWIN on $(r_t-\hat r_t)^2$", fc="#f7d9d9")
box(10.8, 0.2, 2.9, 0.95, "BOB window selector", r"EXP3 over $\{w_j\}$, one block of $H$ rounds", fc="#e9dff5")

# ---- data flow (solid black) ---------------------------------------------------------------------
arrow((2.4, 4.075), (3.2, 4.075))
arrow((2.4, 1.475), (3.2, 1.475))
arrow((2.4, 2.775), (6.9, 2.775))
label(4.65, 2.95, r"$X_t$")
arrow((5.9, 4.075), (6.9, 2.95), rad=0.0)
label(6.55, 3.65, "P scores", ha="center")
arrow((5.9, 1.475), (6.9, 2.6), rad=0.0)
label(6.55, 1.95, "F scores", ha="center")
arrow((9.8, 2.775), (10.8, 2.775))
label(10.3, 2.95, r"source $m_t$")
arrow((13.7, 2.775), (13.95, 2.775))
label(13.95, 3.05, r"$a_t$", ha="right")

# ---- statistics (dashed grey) ---------------------------------------------------------------------
arrow((8.35, 2.3), (8.35, 1.15), color=GREY, ls="--", lw=1.1)
label(8.35, 1.72, "residual of active source", color=GREY)
arrow((12.25, 2.3), (12.25, 1.15), color=GREY, ls="--", lw=1.1)
label(12.25, 1.72, "block mean reward", color=GREY)

# ---- window choice (dash-dot purple) --------------------------------------------------------------
arrow((12.25, 0.2), (12.25, -0.3), color=PURPLE, ls="-.", lw=1.3)
arrow((12.25, -0.3), (4.55, -0.3), color=PURPLE, ls="-.", lw=1.3)
arrow((4.55, -0.3), (4.55, 1.0), color=PURPLE, ls="-.", lw=1.3)
label(8.35, -0.3, r"$w\leftarrow w_j$ at the start of each block", color=PURPLE)

# ---- suppression path (gold) --------------------------------------------------------------------
arrow((6.9, 0.85), (6.4, 0.85), color=GOLD, lw=1.8)
arrow((6.4, 0.85), (6.4, 5.05), color=GOLD, lw=1.8)
label(6.4, 4.5, "drift fired", color=GOLD)
arrow((4.55, 5.05), (4.55, 4.55), color=GOLD, ls="--", lw=1.4)
label(4.55, 4.8, "(unchanged)", color=GOLD, fs=7.5)
arrow((3.5, 5.05), (3.5, 1.95), color=GOLD, ls="--", lw=1.4)
label(3.05, 3.05, "halve $w$,\npurge", color=GOLD)
arrow((8.35, 5.05), (8.35, 3.25), color=GOLD, ls="--", lw=1.4)
label(8.35, 4.15, "reset P credit,\nforce F", color=GOLD)
arrow((13.4, 5.05), (13.4, 1.15), color=GOLD, ls="--", lw=1.4)
label(13.4, 3.9, "discard\nopen\nblock", color=GOLD, fs=8)

# ---- legend ---------------------------------------------------------------------------------------
ax.text(
    0.3,
    -0.85,
    "black solid: contexts, rows, scores\ngrey dashed: statistics fed to detector and BOB\n"
    "purple dash-dot: window choice\ngold: suppression path",
    fontsize=8.5,
    color="#333333",
    va="bottom",
)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
