"""Render reports/figures/biology_overview.png from the probe tables.

    python scripts/biology_probes.py && python scripts/plot_biology.py

Reads reports/allele_pair_concordance.csv and reports/position_eta2.csv so the
figure and the tables can never disagree.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

GREY = "#888888"
DARK = "#1B3A6B"
MID = "#8FA7C4"
PALE = "#C9CDD3"
ALARM = "#B2432F"
POCKET = {1: "A", 2: "B", 3: "D", 4: "-", 5: "-", 6: "C", 7: "E", 8: "-", 9: "F"}

RC = {
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 150,
}


def main() -> int:
    conc = pd.read_csv(REPORT_DIR / "allele_pair_concordance.csv")
    eta = pd.read_csv(REPORT_DIR / "position_eta2.csv")
    pos = eta.groupby("position").eta2_adj.median()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context(RC):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.1))

        jit = np.random.default_rng(1).normal(0, 0.16, len(conc))
        axL.axhline(0, color=GREY, lw=0.8, zorder=1)
        axL.scatter(conc.hamming + jit, conc.rho, s=14, color=MID, alpha=.55,
                    edgecolor="none", zorder=2,
                    label="allele pair (>=50 shared peptides)")
        bins = pd.cut(conc.hamming, [-1, 0, 4, 9, 14, 19, 34])
        grp = conc.groupby(bins, observed=True)
        axL.plot([g.hamming.mean() for _, g in grp], [g.rho.median() for _, g in grp],
                 "-D", color=DARK, lw=2.0, ms=6, zorder=5, label="binned median")

        top = conc.nlargest(1, "rho").iloc[0]
        axL.annotate(f"{top.a.replace('HLA-', '')} / {top.b.replace('HLA-', '')}\n"
                     f"rho = {top.rho:.2f}", xy=(top.hamming, top.rho),
                     xytext=(3.2, 1.02), fontsize=7,
                     arrowprops=dict(arrowstyle="-", lw=.7, color=GREY))
        c67 = conc[conc.hamming == 0].iloc[0]
        axL.annotate("identical contact residues\n(C67S pair, 89%/75% at assay floor)",
                     xy=(0, c67.rho), xytext=(5.0, 0.52), fontsize=7, color=ALARM,
                     arrowprops=dict(arrowstyle="-", lw=.7, color=ALARM))
        axL.scatter([0], [c67.rho], s=42, facecolor="none", edgecolor=ALARM,
                    lw=1.4, zorder=6)
        axL.set_xlabel("contact residues differing between the two grooves (of 34)")
        axL.set_ylabel("Spearman rho between the two alleles' half-lives")
        axL.set_title("Grooves stop agreeing after ~5 contact substitutions", loc="left")
        axL.legend(frameon=False, loc="lower left", fontsize=7)
        axL.margins(0.04)
        axL.set_ylim(-0.45, 1.15)
        axL.yaxis.labelpad = 6

        colours = [DARK if p in (1, 2, 9) else MID if POCKET[p] != "-" else PALE
                   for p in range(1, 10)]
        axR.bar(pos.index, pos.to_numpy(), color=colours, width=.72, zorder=3)
        for p in pos.index:
            axR.text(p, pos[p] + 0.004, POCKET[p], ha="center", va="bottom",
                     fontsize=7, color=GREY)
        axR.text(1.0, 0.093, "anchor pockets", fontsize=7, color=DARK)
        axR.text(4.3, 0.030, "solvent-facing", fontsize=7, color=GREY)
        axR.set_xticks(range(1, 10))
        axR.set_xlabel("peptide position (pocket label above bar)")
        axR.set_ylabel("variance in log half-life explained\n"
                       "(within allele, permutation-adjusted)")
        axR.set_title("P1, P2 and P9 carry the signal; P4/P5 carry none", loc="left")
        axR.margins(0.04)
        axR.set_ylim(0, 0.105)
        axR.yaxis.labelpad = 6

        fig.text(0.5, -0.03,
                 f"n = {len(conc)} allele pairs (left) and "
                 f"{eta.allele.nunique()} alleles with >=100 measurements (right), "
                 "Rasmussen et al. 2016 stability panel; rho is rank correlation "
                 "of log1p half-life.", ha="center", fontsize=7, color=GREY)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "biology_overview.png", bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'biology_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
