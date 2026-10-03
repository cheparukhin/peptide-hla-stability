"""Stage 2b: characterise the two negative pools. Why the arms behave as they do.

    .venv/bin/python scripts/stage2b_negatives.py

The arm comparison in ``scripts/baseline_augmented.py`` says whether each source
helped. This says *what each source actually contains*, which is what makes a
null result interpretable rather than just disappointing. Two measurements, both
on the manifests alone -- no validation or test label is read.

**Anchor composition.** HLA class I pins a peptide by its P2 and PΩ side chains,
so a peptide's residues at those two positions largely decide whether it can
bind at all. If one source's negatives violate the anchor preferences and the
other's respect them, the two arms are supplying different *kinds* of negative:
easy ones a model can reject on sight, and hard ones that look like binders and
are not.

**What the baseline already predicts.** The sharper version of the same
question. The stage 2 measured-only model is fitted exactly as stage 2 fitted
it, then asked to predict the augmented rows. A negative the model already
places at the assay floor carries almost no information -- training on it mostly
restates what the model knows. The genuine floor rows of the validation split
give the calibration point: that is what "the model thinks this is a zero" looks
like in this metric.

Writes ``reports/stage2b_negatives.csv`` (per-pool summary) and
``reports/stage2b_composition.csv`` (per-residue composition).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.augment import attach_hla  # noqa: E402
from pepstab.data import AMINO_ACIDS, load_with_splits  # noqa: E402
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_augmented import ENCODING, HIDDEN, INPUT_SET, L2  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS, inner_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
AUG_DIR = REPO_ROOT / "data" / "augmentation"
REPORT_DIR = REPO_ROOT / "reports"

POOLS = ("measured_affinity", "predicted_affinity", "predicted_affinity_full")

#: Residues the C-terminal (F) pocket of most class I alleles prefers:
#: hydrophobic or aromatic. A charged residue at PΩ is the classic
#: anchor violation.
HYDROPHOBIC_AROMATIC = set("AVLIMFWY")
CHARGED = set("DEKR")

#: 0-based positions of the two dominant anchors in a 9-mer: P2 and P9 (PΩ).
ANCHOR_P2 = 1
ANCHOR_POMEGA = 8


def composition(peptides: list[str]) -> pd.Series:
    """Residue frequencies over a peptide set, as shares."""
    counts = np.zeros(len(AMINO_ACIDS))
    for pep in peptides:
        for aa in pep:
            counts[AMINO_ACIDS.index(aa)] += 1
    return pd.Series(counts / counts.sum(), index=list(AMINO_ACIDS))


def anchor_summary(peptides: list[str]) -> dict:
    """Anchor-position statistics for one peptide set."""
    p2 = [p[ANCHOR_P2] for p in peptides]
    pw = [p[ANCHOR_POMEGA] for p in peptides]
    return {
        "n_peptides": len(peptides),
        "pomega_hydrophobic_aromatic": float(
            np.mean([a in HYDROPHOBIC_AROMATIC for a in pw])),
        "pomega_charged": float(np.mean([a in CHARGED for a in pw])),
        "p2_charged": float(np.mean([a in CHARGED for a in p2])),
        "pomega_top": max(set(pw), key=pw.count),
        "p2_top": max(set(p2), key=p2.count),
    }


def main() -> int:
    df = load_with_splits()
    train = df[df.split == "train"]
    fold = inner_folds(train)
    fit, dev = train[(fold == "fit").to_numpy()], train[(fold == "dev").to_numpy()]
    val = df[df.split == "val"]

    missing = [p for p in POOLS if not (AUG_DIR / f"{p}.csv").exists()]
    if missing:
        raise SystemExit(f"manifests not built: {missing}. Run "
                         "scripts/augment_affinity.py first.")
    manifests = {p: pd.read_csv(AUG_DIR / f"{p}.csv") for p in POOLS}

    # The stage 2 baseline, refitted exactly as stage 2 fitted it. Seed 0 only:
    # this is a characterisation of the pools, not a scored comparison.
    model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=L2, seed=SEEDS[0],
                                   max_epochs=MAX_EPOCHS, patience=PATIENCE))
    model.fit(build_features(fit, INPUT_SET, ENCODING),
              fit.y_log1p.to_numpy(dtype=np.float32),
              build_features(dev, INPUT_SET, ENCODING),
              dev.y_log1p.to_numpy(dtype=np.float32))
    pred_val = model.predict(build_features(val, INPUT_SET, ENCODING))
    at_floor = val.thalf_hours.to_numpy() == 0

    rows = []
    for name, manifest in manifests.items():
        peptides = sorted(manifest.peptide.unique())
        pred = model.predict(build_features(
            attach_hla(manifest, df), INPUT_SET, ENCODING))
        rows.append({
            "pool": name,
            "n_rows": len(manifest),
            "n_alleles": int(manifest.allele.nunique()),
            **anchor_summary(peptides),
            # The assumed label is 0.0, so the prediction *is* the residual.
            "baseline_pred_mean": float(pred.mean()),
            "baseline_pred_median": float(np.median(pred)),
            "baseline_share_above_half": float(np.mean(pred > 0.5)),
            "baseline_mae_vs_assumed_zero": float(np.abs(pred).mean()),
        })

    # Calibration rows: what the baseline does on held-out rows whose label is
    # known. Not a pool, so the anchor columns stay empty.
    for label, mask in [("val rows at the assay floor", at_floor),
                        ("val rows above the floor", ~at_floor)]:
        peptides = sorted(val.loc[mask, "peptide"].unique())
        rows.append({
            "pool": label,
            "n_rows": int(mask.sum()),
            "n_alleles": int(val.loc[mask, "allele"].nunique()),
            **anchor_summary(peptides),
            "baseline_pred_mean": float(pred_val[mask].mean()),
            "baseline_pred_median": float(np.median(pred_val[mask])),
            "baseline_share_above_half": float(np.mean(pred_val[mask] > 0.5)),
            "baseline_mae_vs_assumed_zero": float("nan"),
        })

    summary = pd.DataFrame(rows)
    comp = pd.DataFrame({
        **{name: composition(sorted(m.peptide.unique()))
           for name, m in manifests.items()},
        "stability_train": composition(sorted(train.peptide.unique())),
    })
    comp["predicted_minus_stability_train"] = (
        comp["predicted_affinity"] - comp["stability_train"])
    comp["measured_minus_stability_train"] = (
        comp["measured_affinity"] - comp["stability_train"])
    comp = comp.sort_values("predicted_minus_stability_train")

    pd.set_option("display.width", 200)
    print("negative-pool summary (baseline = the stage 2 measured-only model, "
          "seed 0):")
    print(summary[["pool", "n_rows", "n_peptides", "pomega_hydrophobic_aromatic",
                   "pomega_charged", "p2_charged", "baseline_pred_median",
                   "baseline_share_above_half"]].round(3).to_string(index=False))
    print("\nresidue composition, most depleted in the predicted pool first "
          "(% of residues):")
    print((comp * 100).round(2).to_string())

    REPORT_DIR.mkdir(exist_ok=True)
    summary.to_csv(REPORT_DIR / "stage2b_negatives.csv", index=False)
    comp.to_csv(REPORT_DIR / "stage2b_composition.csv")
    print("\nwrote reports/stage2b_negatives.csv and "
          "reports/stage2b_composition.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
