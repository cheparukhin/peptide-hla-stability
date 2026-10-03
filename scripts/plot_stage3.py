"""Stage 3 headline figure: per-allele accuracy by arm, and accuracy against cost.

    python scripts/plot_stage3.py

Reads the prediction files in preds/ and the cost JSONs in reports/, so the
figure cannot drift from the tables. Writes reports/figures/stage3_overview.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import eligible_alleles, score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

GREY, DARK, MID, PALE, ALARM = "#888888", "#1B3A6B", "#8FA7C4", "#C9CDD3", "#B2432F"
RC = {"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
      "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 6,
      "ytick.labelsize": 6, "axes.linewidth": 0.6,
      "xtick.direction": "out", "ytick.direction": "out",
      "axes.spines.top": False, "axes.spines.right": False,
      "savefig.dpi": 300, "figure.dpi": 150}

#: display name -> (prediction file, colour)
ARMS = {
    "sequence baseline\n(one-hot + pseudoseq)": ("seq_ensemble_pep_pseudo.csv", DARK),
    "ESM stacked on\nbaseline": ("esm_stacked_esm2_35M.csv", MID),
    "ESM alone": ("esm_alone_esm2_35M.csv", PALE),
    "ESM + cross-attention\n(coupled)": ("xattn_esm2_35M.csv", ALARM),
    "ESM + mean-pooled groove\n(ablation)": ("meanpool_esm2_35M.csv", GREY),
    "BLOSUM cross-features\nstacked": ("blosum_cross_stacked.csv", "#5B8C5A"),
    "positive control\n(baseline features, same pipeline)": ("control_onehot_pca_alone_esm2_35M.csv", "#6B4C8A"),
    "negative control\n(random block, stacked)": ("control_random_stacked_esm2_35M.csv", PALE),
    "negative control\n(shuffled ESM, stacked)": ("control_shuffled_stacked_esm2_35M.csv", GREY),
}

#: Rank agreement between HLA alleles differing at one contact residue, from
#: reports/allele_pair_concordance.csv (companion branch). Two distinct molecules
#: cannot agree better than the assay resolves, so this bounds assay
#: reproducibility from below -- and therefore bounds what any model can reach.
NOISE_CEILING = 0.90


def main() -> int:
    df = load_with_splits()
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y = val.y_log1p.to_numpy()

    panels, medians = {}, {}
    for name, (fname, _) in ARMS.items():
        path = PRED_DIR / fname
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        pred = pd.read_csv(path).set_index("pair_id").loc[val.pair_id].y_pred.to_numpy()
        s = score(name, val.allele, y, pred, alleles)
        panels[name] = s.panel.to_numpy()
        medians[name] = s.median_spearman

    cost = {}
    for tag, key in (("stage3_cost_esm2_35M.json", "esm"),
                     ("stage3b_cost_esm2_35M.json", "coupling")):
        path = REPORT_DIR / tag
        if path.exists():
            cost[key] = json.loads(path.read_text())
    seq_members = pd.read_csv(REPORT_DIR / "stage2_ensemble_pep_pseudo.csv")

    # CPU-seconds to produce one 30-network ensemble, embedding extraction included.
    emb = cost.get("esm", {}).get("embedding_seconds", 0.0)
    seconds = {
        "sequence baseline\n(one-hot + pseudoseq)": seq_members.fit_seconds.sum(),
        "ESM stacked on\nbaseline": emb + cost.get("esm", {}).get("total_fit_seconds", np.nan) / 2,
        "ESM alone": emb + cost.get("esm", {}).get("total_fit_seconds", np.nan) / 2,
        "ESM + cross-attention\n(coupled)": emb + cost.get("coupling", {}).get("total_fit_seconds", np.nan) / 2,
        "ESM + mean-pooled groove\n(ablation)": emb + cost.get("coupling", {}).get("total_fit_seconds", np.nan) / 2,
    }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(RC):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                       gridspec_kw={"width_ratios": [1.45, 1]})
        names = [n for n in ARMS if n in panels]
        rng = np.random.default_rng(0)
        for i, name in enumerate(names):
            vals = panels[name]
            axL.scatter(np.full(len(vals), i) + rng.normal(0, 0.07, len(vals)), vals,
                        s=9, color=ARMS[name][1], alpha=.45, edgecolor="none", zorder=2)
            axL.plot([i - .28, i + .28], [medians[name]] * 2, color=ARMS[name][1],
                     lw=2.4, zorder=4, solid_capstyle="butt")
            axL.text(i, 0.955, f"{medians[name]:.3f}", ha="center", fontsize=7,
                     color=ARMS[name][1])
        base_med = medians[names[0]]
        axL.axhline(base_med, color=DARK, lw=.7, ls=(0, (4, 3)), zorder=1)
        axL.axhline(NOISE_CEILING, color=ALARM, lw=1.0, zorder=1)
        axL.text(len(names) - 0.45, NOISE_CEILING + 0.015,
                 "noise ceiling 0.90 — one-substitution allele pairs",
                 ha="right", fontsize=6.5, color=ALARM)
        axL.annotate("", xy=(-0.42, NOISE_CEILING), xytext=(-0.42, base_med),
                     arrowprops=dict(arrowstyle="<->", lw=.8, color=ALARM))
        axL.text(-0.34, (NOISE_CEILING + base_med) / 2,
                 f"{NOISE_CEILING - base_med:.2f}\nheadroom", fontsize=6.5,
                 color=ALARM, va="center")
        axL.set_xticks(range(len(names)))
        axL.set_xticklabels(names, fontsize=6.5)
        axL.set_ylabel("per-allele Spearman rho (validation)")
        axL.set_title("Frozen ESM-2 features do not reach the sequence baseline", loc="left")
        axL.set_ylim(-0.35, 1.02)
        axL.margins(x=0.04)

        for name in names:
            if not np.isfinite(seconds.get(name, np.nan)):
                continue
            axR.scatter(seconds[name], medians[name], s=48, color=ARMS[name][1],
                        zorder=3, edgecolor="white", lw=.8)
            axR.annotate(name.replace("\n", " "), (seconds[name], medians[name]),
                         textcoords="offset points", xytext=(7, -2), fontsize=6.5,
                         color=ARMS[name][1], va="center")
        axR.set_xscale("log")
        axR.set_xlabel("CPU-seconds to build one 30-network ensemble\n"
                       "(embedding extraction included)")
        axR.set_ylabel("median per-allele Spearman rho")
        axR.set_title("More compute, less accuracy", loc="left")
        axR.text(0.02, 0.04, "higher and further left = better", transform=axR.transAxes,
                 fontsize=6.5, color=GREY)
        axR.margins(0.18)

        fig.text(0.5, -0.04, f"Validation split, {len(alleles)} eligible alleles "
                 "(>=20 rows, >=2 distinct labels); each point is one allele. "
                 "Arms share folds, seeds, ensemble size and tuning budget; "
                 "ESM-2 35M, layers 6 and 12.", ha="center", fontsize=7, color=GREY)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "stage3_overview.png", bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'stage3_overview.png'}")
    print(pd.Series(medians).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
