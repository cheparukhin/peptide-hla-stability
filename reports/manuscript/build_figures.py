"""Render manuscript figures from saved comparisons; no fitting or resampling.

Requires matplotlib==3.10.8. Run from any directory with Python 3.11+.
Outputs vector SVG/PDF, preview PNG, and an exact source-linked data table.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = "reports/further_research/round2/results/main/paired_comparisons.csv"
TRANSFER = "reports/finalization/transfer_cloud_recovery/results/comparisons.csv"
BLUE, GREY, INK = "#206b92", "#77828a", "#26343c"


def save_figure(fig, stem):
    for ext in ("svg", "pdf", "png"):
        metadata = {"Creator": "peptide-hla-stability manuscript figures"}
        if ext == "svg":
            metadata["Date"] = None
        elif ext == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        else:
            metadata = {"Software": "matplotlib " + matplotlib.__version__}
        target = OUT / "figures" / f"{stem}.{ext}"
        fig.savefig(target, dpi=180, metadata=metadata)
        if ext == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def comparison(source, label, role, **filters):
    raw = (ROOT / source).read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    selected = [r for r in rows if all(r.get(k) == str(v) for k, v in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one row: {source}, {filters}; found {len(selected)}")
    row = selected[0]
    values = {k: float(row[k]) for k in ("baseline_score", "candidate_score", "delta", "ci_low", "ci_high")}
    if not all(math.isfinite(x) for x in values.values()):
        raise ValueError("Nonfinite saved estimate")
    if not math.isclose(values["candidate_score"] - values["baseline_score"], values["delta"], abs_tol=1e-12):
        raise ValueError("Saved scores and paired contrast disagree")
    if values["ci_low"] > values["ci_high"]:
        raise ValueError("Reversed saved interval")
    return dict(label=label, role=role, source=source,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                row_filter=json.dumps(filters, sort_keys=True),
                baseline=row["baseline"], candidate=row["candidate"],
                interval_level=0.95, resampling="crossed HLA groups and peptide clusters" if row.get("mode") == "crossed" else "whole peptide clusters",
                n_alleles=int(row["n_alleles"]), n_rows_eligible=int(row["n_rows_eligible"]),
                uncertainty="conditional on saved fitted predictions and selection decisions",
                **values)


def render(rows, stem, title, limits):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 1.05 * len(rows) + 1.65),
                                     gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=0.265, right=0.98, top=0.80, bottom=0.22 if len(rows) == 3 else 0.15, wspace=0.25)
    fig.suptitle(title, x=0.03, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    for ax in (left, right):
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e4e8eb", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=INK, labelsize=9)
    positions = list(reversed(range(len(rows))))
    left.set_yticks(positions, [r["label"] for r in rows], fontsize=10, color=INK)
    right.set_yticks([])
    left.set_xlim(0, 0.87)
    left.set_xticks([0, .2, .4, .6, .8])
    right.set_xlim(*limits)
    right.axvline(0, color=GREY, linewidth=1)
    left.set_title("A  Absolute ranking performance", loc="left", fontsize=11, pad=32)
    right.set_title("B  Paired improvement", loc="left", fontsize=11, pad=32)
    left.set_xlabel("Median within-allele Spearman correlation", fontsize=10, labelpad=10)
    right.set_xlabel("Difference in panel medians", fontsize=10, labelpad=10)
    for y, r in zip(positions, rows):
        b, c = r["baseline_score"], r["candidate_score"]
        left.plot([b, c], [y, y], color="#b8c0c5", linewidth=1.4, zorder=2)
        left.plot(b, y, "s", mfc="white", mec=GREY, ms=6, zorder=3)
        left.plot(c, y, "o", color=BLUE, ms=5, zorder=4)
        left.annotate(f"{b:.4f}", (b, y), xytext=(0, -16), textcoords="offset points", ha="center", color=GREY, fontsize=9)
        left.annotate(f"{c:.4f}", (c, y), xytext=(0, 9), textcoords="offset points", ha="center", color=BLUE, fontsize=9)
        right.plot([r["ci_low"], r["ci_high"]], [y, y], color=BLUE, linewidth=1.8)
        right.plot(r["delta"], y, "D" if r["role"] == "primary" else "o", color=BLUE, ms=5)
        right.annotate(f"{r['delta']:+.4f}", (r["delta"], y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK)
    left.legend(handles=[Line2D([], [], marker="s", mfc="white", mec=GREY, ls="", label="Reference"),
                         Line2D([], [], marker="o", color=BLUE, ls="", label="Candidate")],
                loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9, borderaxespad=0)
    fig.text(.03, .028, "Points are saved estimates; bars are conditional 95% paired bootstrap intervals. Estimates are paired within each assessment.", fontsize=9, color=GREY)
    save_figure(fig, stem)


def render_adaptation():
    source = "reports/followup3/figures/fewshot_curves.json"
    raw = (ROOT / source).read_bytes()
    data = json.loads(raw)
    for path, digest in data["input_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved adaptation figure input changed: {path}")
    with (ROOT / "reports/followup3/fewshot/absolute_curves.csv").open() as handle:
        curves = [r for r in csv.DictReader(handle) if r["panel_min_rows"] == "20"]
    if curves != data["displayed_curve_rows"]:
        raise ValueError("Adaptation curve rows disagree with the saved figure record")
    family = data["primary_family_interval"]
    r5 = {int(r["budget"]): float(r["score"]) for r in curves if r["arm"] == "R5"}
    if not math.isclose(r5[20] - r5[0], family["point"], abs_tol=1e-12):
        raise ValueError("Adaptation curve and primary contrast disagree")
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    fig.subplots_adjust(left=.09, right=.97, top=.77, bottom=.23)
    fig.suptitle("Retrospective adaptation with a small target-allele support set", x=.03, y=.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    for arm, label, color, marker, style in [
        ("R5", "Boltz pair ridge (R5)", BLUE, "o", "-"),
        ("sequence_anchor", "Reduced sequence MLP\n(4 members)", GREY, "s", "-"),
        ("R0", "Additive sequence ridge\n(sensitivity)", "#a0a8ae", "^", "--"),
    ]:
        rows = sorted((r for r in curves if r["arm"] == arm), key=lambda r: int(r["budget"]))
        xs, ys = [int(r["budget"]) for r in rows], [float(r["score"]) for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, linestyle=style, linewidth=1.8, markersize=5, label=label)
        ax.annotate(f"{ys[-1]:.4f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=color)
    ax.set(xlim=(-.5, 21.8), ylim=(0, .67), xticks=[0, 5, 10, 20])
    ax.set_xlabel("Target-allele support labels (in addition to source training)", fontsize=10, labelpad=10)
    ax.set_ylabel("Mean of design-specific\nmedian within-allele Spearman", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)
    ax.grid(axis="y", color="#e4e8eb", linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=9, borderaxespad=0)
    ax.text(.98, .045, f"R5 adaptation: {family['point']:+.4f}\nConditional 98.33% family interval [{family['ci_low']:+.4f}, {family['ci_high']:+.4f}]", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)
    fig.text(.03, .028, "68 target alleles; three fixed support designs; joint HLA/peptide holdout. Curve uncertainty bands were not estimated.", fontsize=9, color=GREY)
    save_figure(fig, "figure3_adaptation")
    (OUT / "figure3_data.json").write_text(json.dumps({
        "source": source, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "aggregation": data["aggregation"], "curve_rows": curves,
        "primary_family_interval": family,
    }, indent=2) + "\n")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "pmhc-manuscript", "pdf.fonttype": 42})
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    generalization = [comparison(MAIN, "Peptide-cluster holdout\nSelected procedures (primary)", "primary", baseline="sequence_selector", candidate="embedding_selector", panel_min_rows=20)]
    for regime, label in [("hla", "HLA-group holdout"), ("joint", "Joint peptide/HLA-group holdout")]:
        generalization.append(comparison(TRANSFER, label + "\nMatched MLPs", "secondary", baseline="R0_mlp", candidate="R5_mlp", mode="crossed", regime=regime, panel_min_rows=20))
        generalization.append(comparison(TRANSFER, label + "\nSelected procedures", "secondary", baseline="expanded_sequence_selector", candidate="embedding_selector", mode="crossed", regime=regime, panel_min_rows=20))
    ablations = [
        comparison(MAIN, "Positional token indexing\nvs same-residue pooled tokens", "secondary", baseline="R3_pooled_mlp", candidate="R3_mlp", panel_min_rows=20),
        comparison(MAIN, "Full 9x34 groove pairs\nvs contact-restricted pooled pairs", "secondary", baseline="R5_position9x34_pooled_mlp", candidate="R5_position9x34_mlp", panel_min_rows=20),
        comparison(MAIN, "R5 crystal contact mask\nvs shuffled contact mask", "secondary", baseline="R5_shuffled_mlp", candidate="R5_mlp", panel_min_rows=20),
    ]
    render(generalization, "figure1_generalization", "Stability ranking within and across HLA sequence groups", (-0.035, 0.53))
    render(ablations, "figure2_representation_controls", "Token indexing, pair coverage and resolution, and R5 contact-mask comparisons", (-0.035, 0.53))
    render_adaptation()
    with (OUT / "figure_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", *generalization[0].keys()], lineterminator="\n")
        writer.writeheader()
        for figure, rows in [(1, generalization), (2, ablations)]:
            for row in rows:
                writer.writerow(dict(figure=figure, **row))
    print("Rendered three figures from eight saved comparisons and the verified adaptation curves; no estimates or intervals recomputed.")


if __name__ == "__main__":
    main()
