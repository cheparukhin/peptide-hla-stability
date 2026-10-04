"""Stage 6 submission figures. CPU only, matplotlib only, no network.

    .venv/bin/python reports/figures/make_figures.py

Writes into this directory:

    fig1_per_allele_spearman.png   per-allele Spearman distribution per arm
    fig2_cost_vs_accuracy.png      accuracy against $ per 1,000 new predictions
    figure_data.csv                the numbers behind both panels

Every arm listed in ``ARMS`` is drawn if its prediction file exists and is
skipped (with a note printed) if it does not. That is deliberate: the ESM-2 and
Boltz-2 structural arms are in flight at the time of writing, so the figures are
regenerated unchanged once their prediction files land in ``preds/``.

``--split`` defaults to ``val``. **Only stage 6 may pass ``--split test``**, and
only once, per EVALUATION.md.

Cost figures are per 1,000 *new* predictions at inference, and every one of them
is sourced in reports/compute_ledger.md. They are not invented here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    MIN_ROWS_BY_SPLIT,
    eligible_alleles,
    score,
)

HERE = Path(__file__).resolve().parent

#: (label, prediction file stem, $ per 1,000 new predictions, cost status).
#:
#: ``cost`` is dollars of *inference* compute per 1,000 new peptide-HLA pairs.
#: ``cost_status`` is "measured", "derived" (measured time x a published rate)
#: or "forecast" (not yet run). Sources: reports/compute_ledger.md.
ARMS = [
    # Constant baselines and ridge have no measured inference timing in
    # reports/stage2_baselines.md, so they carry no cost: they appear on the
    # accuracy figure and are omitted from the cost figure rather than
    # assigned a guess.
    ("training allele mean", "allele_mean", np.nan, "not measured"),
    ("ridge, peptide+pseudoseq", "ridge_onehot_pep_pseudo", np.nan, "not measured"),
    ("MLP, peptide only", "mlp_onehot_pep", 2.6e-9, "derived"),
    ("MLP, peptide+pseudoseq (single)", "seq_baseline", 6.6e-9, "derived"),
    ("MLP, peptide+domain (single)", "mlp_onehot_pep_domain", 2.6e-8, "derived"),
    ("sequence ensemble, 30 nets", "seq_ensemble_pep_pseudo", 2.0e-7, "derived"),
    ("sequence ensemble, full domain", "seq_ensemble_pep_domain", 7.9e-7, "derived"),
    # --- in flight: drawn automatically once the prediction file exists ---
    # ESM-2's cost stays NaN until hole E1b lands: extraction is measured
    # (reports/stage3_embedding_cost.csv) but the head's forward pass is not,
    # so a total would be a guess. The band below shows the measured half.
    # Two ESM-2 arms, because they answer different questions and the contrast
    # between them is itself a result. "ESM-only" asks whether frozen ESM-2 can
    # *replace* sequence features; "sequence+ESM-2" asks whether it *adds* to
    # them, which is the question the challenge brief actually poses. Drawing
    # only the first would understate the foundation-model case; drawing only
    # the second would hide that it cannot stand alone.
    ("ESM-2 ensemble (ESM only)", "esm_ensemble", np.nan, "pending"),
    ("sequence + ESM-2 ensemble", "esm_plus_seq_ensemble", np.nan, "pending"),
    ("Boltz-2 structural ensemble", "boltz_structural", 6.90, "measured"),
]

#: Arms whose compute cost is known but whose accuracy is not yet. Drawn on
#: fig. 2 as a vertical band so the reader sees the open question rather than a
#: guess. Remove an entry here once its prediction file exists.
PENDING_COST_ONLY = [
    ("ESM-2", 4.4e-4, "extraction only, 35M\n($4.4e-4 / 1,000 measured;\nhead cost = hole E1b)"),
    ("Boltz-2 structural", 6.90, "$6.90 / 1,000 measured\n(stage 4c pilot, arm B)"),
]

MIN_WORTHWHILE_GAIN = 0.05  # predeclared, EVALUATION.md


def collect(split: str) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Score every available arm on ``split`` over the common eligible panel."""
    df = load_with_splits()
    part = df[df.split == split]
    if part.empty:
        raise SystemExit(f"no rows in split {split!r}")
    panel = eligible_alleles(part.allele, part.y_log1p.values,
                             MIN_ROWS_BY_SPLIT[split])

    rows, per_allele = [], {}
    for label, stem, cost, cost_status in ARMS:
        path = ROOT / "preds" / f"{stem}.csv"
        if not path.exists():
            print(f"  skip {label}: {path.relative_to(ROOT)} not written yet")
            continue
        preds = pd.read_csv(path)
        merged = part.merge(preds[["pair_id", "y_pred"]], on="pair_id", how="left")
        if merged.y_pred.isna().any():
            print(f"  skip {label}: {int(merged.y_pred.isna().sum())} rows unscored")
            continue
        s = score(label, merged.allele, merged.y_log1p.values,
                  merged.y_pred.values, alleles=panel, split=split)
        row = s.as_row()
        row["file"] = stem
        row["usd_per_1k"] = cost
        row["cost_status"] = cost_status
        rows.append(row)
        # The scored panel, not the raw Spearman: constants score 0 by the
        # predeclared rule, and dropping them would flatter a constant arm.
        per_allele[label] = s.panel.dropna()

    return pd.DataFrame(rows), per_allele


def fig_per_allele(per_allele: dict[str, pd.Series], split: str) -> Path:
    """Per-allele Spearman distribution: box + every allele as a point."""
    labels = list(per_allele)
    data = [per_allele[k].values for k in labels]
    height = 1.0 + 0.62 * len(labels)
    fig, ax = plt.subplots(figsize=(9.5, height))

    ax.boxplot(data, orientation="horizontal", widths=0.55, showfliers=False,
               medianprops=dict(color="#c0392b", lw=2),
               boxprops=dict(color="#4a4a4a"),
               whiskerprops=dict(color="#4a4a4a"),
               capprops=dict(color="#4a4a4a"))
    rng = np.random.default_rng(20261003)
    for i, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        ax.plot(values, i + jitter, "o", ms=3.4, alpha=0.45,
                color="#2c6fbb", mec="none", zorder=3)

    ax.axvline(0.0, color="#999", lw=1, ls="--")
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spearman rho within allele (one point per allele)")
    n_alleles = len(data[0]) if data else 0
    ax.set_title(
        f"Per-allele ranking accuracy, {split} split  "
        f"({n_alleles} eligible alleles)\n"
        "box = IQR, red line = median (the primary metric)",
        fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = HERE / "fig1_per_allele_spearman.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def fig_cost_vs_accuracy(table: pd.DataFrame, split: str) -> Path:
    """Median per-allele Spearman against inference cost per 1,000 predictions."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    known = table[table.usd_per_1k.notna()]

    for _, r in known.iterrows():
        marker = "o" if r.cost_status in {"measured", "derived"} else "s"
        ax.plot(r.usd_per_1k, r.median_per_allele_spearman, marker, ms=9,
                color="#2c6fbb", mec="white", mew=1.2, zorder=4)
        ax.annotate(f"{r.model}\n({r.median_per_allele_spearman:.3f})",
                    (r.usd_per_1k, r.median_per_allele_spearman),
                    textcoords="offset points", xytext=(9, -4),
                    fontsize=8, va="center")

    for label, cost, note in PENDING_COST_ONLY:
        if label.split()[0].lower() in " ".join(known.model).lower():
            continue  # it landed; it is a real point now
        ax.axvline(cost, color="#c0392b", lw=1.6, ls="--", zorder=2)
        ax.annotate(f"{label}\n{note}\naccuracy: not yet measured",
                    (cost, ax.get_ylim()[0]), textcoords="offset points",
                    xytext=(-6, 14), fontsize=8, color="#c0392b",
                    ha="right", va="bottom")

    best = known.median_per_allele_spearman.max() if len(known) else 0.0
    ax.axhline(best, color="#888", lw=1, ls=":")
    ax.axhline(best + MIN_WORTHWHILE_GAIN, color="#888", lw=1.2, ls="-.")
    ax.text(0.012, best, " best sequence arm", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#555", va="top")
    ax.text(0.012, best + MIN_WORTHWHILE_GAIN,
            f" + the predeclared {MIN_WORTHWHILE_GAIN:.2f} minimum worthwhile gain",
            transform=ax.get_yaxis_transform(), fontsize=8, color="#555",
            va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("US$ of compute per 1,000 new predictions (log scale) "
                  "— see reports/compute_ledger.md")
    ax.set_ylabel("median per-allele Spearman rho")
    ax.set_title(
        f"Is the compute worth it? {split} split.\n"
        "A foundation-model arm has to clear the dashed line to earn its cost.",
        fontsize=10, loc="left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = HERE / "fig2_cost_vs_accuracy.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="test is the single stage 6 scoring run")
    args = ap.parse_args()

    print(f"scoring arms on split={args.split}")
    table, per_allele = collect(args.split)
    if table.empty:
        raise SystemExit("no prediction files found under preds/")

    table = table.sort_values("median_per_allele_spearman")
    table.to_csv(HERE / "figure_data.csv", index=False)
    print(table[["model", "median_per_allele_spearman", "mae_log1p",
                 "usd_per_1k", "cost_status"]].to_string(index=False))

    print("wrote", fig_per_allele(per_allele, args.split).name)
    print("wrote", fig_cost_vs_accuracy(table, args.split).name)
    print("wrote figure_data.csv")


if __name__ == "__main__":
    main()
