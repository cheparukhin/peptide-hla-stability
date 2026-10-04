"""Plot already-published test comparisons; never loads labels or scores models.

Run from the repository root:
    .venv/bin/python reports/figures/make_report_comparison.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
ARMS = {
    "seq_baseline": "Single sequence network",
    "seq_ensemble_pep_domain": "Sequence ensemble: full domain",
    "esm_ensemble": "ESM-2 35M ensemble",
    "esm_plus_seq_ensemble": "Sequence + ESM-2",
    "boltz_structural": "Sequence + Boltz-2 features",
}


def main():
    intervals = pd.read_csv(REPORTS / "stage6_test_paired_ci.csv")
    intervals = intervals[
        intervals.statistic.eq("median_per_allele_spearman")
    ].set_index("model").loc[list(ARMS)]
    assert len(intervals) == 5
    assert intervals.baseline.eq("seq_ensemble_pep_pseudo").all()
    assert intervals.n_boot.eq(2000).all()
    summary = pd.read_csv(REPORTS / "stage6_test_summary.csv").set_index("model")
    baseline = summary.loc["seq_ensemble_pep_pseudo", "median_per_allele_spearman"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (arm, row) in enumerate(intervals.iterrows()):
        color = "#b44b37" if row.ci_high < 0 else "#256b88"
        ax.errorbar(
            row.delta, i,
            xerr=[[row.delta - row.ci_low], [row.ci_high - row.delta]],
            fmt="o", color=color, capsize=5, markersize=8, linewidth=2,
        )
        ax.text(
            0.077, i,
            f"{row.delta:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]",
            va="center", fontsize=10,
        )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.axvline(0.05, color="#7b8590", linestyle="--", linewidth=1.4)
    ax.text(0, -0.6, "Sequence reference", ha="right", fontsize=10)
    ax.text(0.05, -0.6, "+0.05 bar", ha="left", fontsize=10)
    ax.set_yticks(range(len(ARMS)), list(ARMS.values()))
    ax.set_xlim(-0.15, 0.22)
    ax.set_xticks([-0.15, -0.10, -0.05, 0, 0.05])
    ax.set_ylim(len(ARMS) - 0.5, -1)
    ax.set_xlabel("Change in median per-allele Spearman vs. sequence ensemble")
    ax.set_title("ESM does not establish a worthwhile gain; the structural arm is worse",
                 loc="left", fontsize=13, pad=22)
    ax.grid(axis="x", color="#e7ebee")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    fig.text(
        0.03, 0.015,
        f"Test split · 67 eligible alleles · reference Spearman {baseline:.4f} · "
        "paired 95% cluster-bootstrap intervals, 2,000 draws",
        fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output = HERE / "report_test_comparisons.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
