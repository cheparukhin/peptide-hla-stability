"""Stage 2b: build the two weak-binder augmentation manifests.

    .venv/bin/python scripts/augment_affinity.py

Writes three candidate manifests to ``data/augmentation/`` plus the provenance
and leakage report the stage 2b deliverable asks for. Trains nothing that gets
scored -- the only model here is the affinity predictor that labels the
predicted-affinity arm.

**Measured-affinity arm.** Pairs from
``data/data_augmentation_iedb/affinity_reference_75alleles.csv`` whose
measurement or lower bound puts affinity at or above 20,000 nM. Inequalities are
respected: ``>`` at or above the threshold is a lower bound that establishes
weak binding, ``=`` is a measurement, and no row in the reference carries ``<``
at or above the threshold, so nothing is flagged weak on an upper bound. The
IEDB convention of recording a competitive-assay non-binder as ``>20000``
accounts for most of the pool; that is a quantitative bound at the threshold,
not a generic negative assay result, and the counts are reported split out.
C67S constructs are excluded -- position 67 is one of the 34 contact positions,
so a wild-type affinity describes a different groove.

**Predicted-affinity arm.** Random 9-mers from the reviewed human proteome whose
*predicted* affinity on that allele is weaker than 20,000 nM, following
Rasmussen et al.'s recipe of padding with random natural peptides.

The predictor is built here rather than downloaded. NetMHCpan needs a licensed
DTU download and MHCflurry pulls TensorFlow, either of which can eat the stage
2b time box; more importantly both were trained on affinity corpora that
include our validation and test peptides, so their knowledge of those peptides
would be an undocumented path into the augmentation labels. An in-house
predictor can be held to the *same* exclusion rule as the augmented rows
themselves: it never sees any peptide within 3 substitutions of an inner-dev,
validation or test peptide. Its architecture, data and held-out accuracy are all
recorded in the provenance file.

The cost of that choice is stated plainly in the report: the predictor is
trained on the same measured corpus that supplies the measured arm, so the two
arms do not differ in the *evidence* behind the weak call. What they differ in
is the **peptide universe** -- assay-selected peptides against random natural
ones -- which is the selection-bias question the random-negative literature
actually motivates.

**Matched counts.** Each allele gets ``min(25% of its fit rows, measured
availability, predicted availability)`` rows in *both* arms, so the arms differ
only in where the negatives came from. A third manifest,
``predicted_affinity_full``, lifts the measured-availability term and so covers
all 75 alleles; it is a superset of the matched predicted manifest, drawn in one
pass, and is reported as a secondary arm answering "does the predicted source's
broader allele coverage buy anything?"
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.augment import (  # noqa: E402
    CAP_FRACTION,
    MIN_HAMMING,
    WEAK_NM,
    annotate_candidates,
    attach_hla,
    finalise_manifest,
    holdout_peptides,
    matched_counts,
    min_hamming,
    per_allele_cap,
    sample_per_allele,
)
from pepstab.data import (  # noqa: E402
    AMINO_ACIDS,
    PEPTIDE_LENGTH,
    load_with_splits,
)
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from pepstab.splits import single_linkage_clusters, assign_clusters  # noqa: E402
from scripts.baseline_sequence import inner_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
AFFINITY_CSV = (REPO_ROOT / "data" / "data_augmentation_iedb"
                / "affinity_reference_75alleles.csv")
OUT_DIR = REPO_ROOT / "data" / "augmentation"
CACHE_DIR = REPO_ROOT / "external"
REPORT_DIR = REPO_ROOT / "reports"

#: Reviewed human proteins, the natural-peptide source for the predicted arm.
#: Reviewed (SwissProt) rather than the full proteome: curated, one canonical
#: sequence per gene, and no unreviewed fragments to sample junk 9-mers out of.
PROTEOME_URL = ("https://rest.uniprot.org/uniprotkb/stream"
                "?query=reviewed%3Atrue+AND+organism_id%3A9606"
                "&format=fasta&compressed=true")
PROTEOME_GZ = CACHE_DIR / "uniprot_human_reviewed.fasta.gz"

#: Seeds, fixed here so a manifest is reproducible from this file alone.
PEPTIDE_SAMPLE_SEED = 20262103     # which 9-mers are drawn from the proteome
MEASURED_DRAW_SEED = 20262104      # which measured pairs enter the manifest
PREDICTED_DRAW_SEED = 20262105     # which predicted pairs enter the manifest

#: How many distinct natural 9-mers to draw before predicting affinity. Needs to
#: leave every allele more predicted-weak candidates than its cap after the
#: distance filter; 40,000 clears the largest cap (165) by orders of magnitude
#: and still scores in seconds.
N_NATURAL_PEPTIDES = 40_000

#: The affinity predictor. Same input set and encoding as the stability model
#: stage 2 selected, so no new feature machinery enters the project; the network
#: is wider only because it fits ~25x the rows.
AFFINITY_INPUT_SET = "pep_pseudo"
AFFINITY_ENCODING = "onehot"
AFFINITY_HIDDEN = (256, 64)
AFFINITY_L2 = 1e-5
AFFINITY_SEED = 0
AFFINITY_MAX_EPOCHS = 150
AFFINITY_PATIENCE = 10
AFFINITY_DEV_FRACTION = 0.10

#: The field-standard affinity transform: ``1 - log(nM)/log(50000)``, so 50,000
#: nM maps to 0 and 1 nM to 1. Used because it spreads the strong-binder end
#: where the data is dense, unlike raw nM.
AFFINITY_SCALE_NM = 50_000.0


def affinity_to_unit(nm: np.ndarray) -> np.ndarray:
    """``1 - log(nM)/log(50000)``, clipped to [0, 1]."""
    nm = np.clip(np.asarray(nm, dtype=float), 1e-4, None)
    return np.clip(1.0 - np.log(nm) / np.log(AFFINITY_SCALE_NM), 0.0, 1.0)


def unit_to_affinity(unit: np.ndarray) -> np.ndarray:
    """Inverse of :func:`affinity_to_unit`, in nM."""
    return AFFINITY_SCALE_NM ** (1.0 - np.asarray(unit, dtype=float))


#: Predicted unit score at or below which a pair counts as weak. Derived from
#: the same 20,000 nM threshold the measured arm uses, not tuned.
WEAK_UNIT = float(affinity_to_unit(np.array([WEAK_NM]))[0])


# --- measured arm ---------------------------------------------------------

def measured_candidates(df: pd.DataFrame, fold: pd.Series,
                        ) -> tuple[pd.DataFrame, dict]:
    """Admissible measured weak-affinity pairs, with the exclusion ledger.

    The ledger is cumulative and in the order the filters are applied, so the
    report can say where the pool went rather than only how big it ended up.
    """
    ref = pd.read_csv(AFFINITY_CSV)
    ledger = {"reference_rows": len(ref)}

    weak = ref[ref["affinity_nM"] >= WEAK_NM].copy()
    ledger["at_or_above_threshold"] = len(weak)
    ledger["bound_at_threshold"] = int(
        ((weak["inequality"] == ">") & (weak["affinity_nM"] == WEAK_NM)).sum())
    ledger["measured_above_threshold"] = int((weak["inequality"] == "=").sum())
    ledger["bound_above_threshold"] = int(
        ((weak["inequality"] == ">") & (weak["affinity_nM"] > WEAK_NM)).sum())
    # An upper bound below the threshold can never establish weak binding. The
    # reference carries none at or above it, but the check is cheap and the
    # claim "inequalities are respected" should be enforced, not asserted.
    upper_bounds = int((weak["inequality"] == "<").sum())
    if upper_bounds:
        raise ValueError(f"{upper_bounds} row(s) at or above {WEAK_NM:g} nM "
                         "carry '<', which does not establish weak binding")

    weak = weak[~weak["engineered_construct_mismatch"]]
    ledger["after_dropping_c67s_constructs"] = len(weak)

    # A pair with its own measured half-life keeps that label; an assumed zero
    # must never overwrite a measurement.
    weak = weak[~weak["in_stability_dataset"]]
    ledger["after_dropping_pairs_with_measured_stability"] = len(weak)

    weak = annotate_candidates(weak, df, fold)
    ledger["peptides_before_distance_filter"] = int(weak["peptide"].nunique())
    ledger["after_inner_folds_distance_filter"] = int(weak["ok_inner_folds"].sum())
    ledger["after_cv_folds_distance_filter"] = int(weak["ok_cv_folds"].sum())
    ledger["excluded_by_distance"] = int((~weak["ok_inner_folds"]).sum())
    ledger["excluded_peptides_by_distance"] = int(
        weak.loc[~weak["ok_inner_folds"], "peptide"].nunique())
    for d in range(MIN_HAMMING):
        ledger[f"excluded_at_hamming_{d}"] = int(
            (weak["min_dist_holdout"] == d).sum())

    pool = weak[weak["ok_inner_folds"]].copy()
    # dataset_allele carries the (C67S) suffix, which is how the stability CSV
    # names its alleles; joining on the stripped name would merge a construct
    # into its wild type.
    pool["allele"] = pool["dataset_allele"]
    pool["affinity_kind"] = "measured"
    pool["protein_source"] = ""
    pool["protein_offset"] = -1
    ledger["pool_rows"] = len(pool)
    ledger["pool_alleles"] = int(pool["allele"].nunique())
    ledger["pool_peptides"] = int(pool["peptide"].nunique())
    ledger["pool_corroborated_by_bd2013"] = int(pool["bd2013_agrees_weak"].sum())
    ledger["pool_contradicted_by_bd2013"] = int(
        (pool["bd2013_affinity_nM"].notna() & ~pool["bd2013_agrees_weak"]).sum())
    return pool, ledger


# --- the affinity predictor -----------------------------------------------

def fetch_proteome() -> tuple[str, dict]:
    """The reviewed human proteome FASTA, cached under ``external/``."""
    CACHE_DIR.mkdir(exist_ok=True)
    if not PROTEOME_GZ.exists():
        print(f"  downloading {PROTEOME_URL.split('?')[0]} ...")
        urllib.request.urlretrieve(PROTEOME_URL, PROTEOME_GZ)
    text = gzip.decompress(PROTEOME_GZ.read_bytes()).decode()
    # The *decompressed* digest, because gzip output is not byte-reproducible
    # across zlib versions while the FASTA content is.
    return text, {
        "url": PROTEOME_URL,
        "cached_at": str(PROTEOME_GZ.relative_to(REPO_ROOT)),
        "fasta_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "n_proteins": text.count(">"),
    }


def sample_natural_peptides(fasta: str, n: int, seed: int,
                            ) -> tuple[pd.DataFrame, dict]:
    """Draw ``n`` distinct 9-mers uniformly over all 9-mer *positions*.

    Uniform over positions, not over proteins, so a 2,000-residue protein
    contributes proportionally more windows than a 100-residue one -- which is
    what "a random natural 9-mer" means. Windows containing a non-standard
    residue (``X``, ``U``, ``B``, ``Z``, ``O``) are rejected rather than
    repaired: the encoders raise on them, and a substituted residue would be a
    peptide that does not occur in nature.
    """
    records = []
    for block in fasta.split("\n>"):
        if not block.strip():
            continue
        header, _, body = block.lstrip(">").partition("\n")
        seq = body.replace("\n", "")
        if len(seq) >= PEPTIDE_LENGTH:
            acc = header.split("|")[1] if header.count("|") >= 2 else header.split()[0]
            records.append((acc, seq))

    accs = [a for a, _ in records]
    lengths = np.array([len(s) - PEPTIDE_LENGTH + 1 for _, s in records])
    starts = np.concatenate([[0], np.cumsum(lengths)])
    total = int(starts[-1])

    rng = np.random.default_rng(seed)
    allowed = set(AMINO_ACIDS)
    seen: dict[str, tuple[str, int]] = {}
    rejected = 0
    tries = 0
    # Draw in blocks: a rejection-sampling loop with one draw per iteration
    # spends all its time in Python, and the non-standard-residue rate is low
    # enough that a block of 4n almost always finishes in one or two passes.
    while len(seen) < n and tries < 40:
        tries += 1
        picks = rng.integers(0, total, size=4 * n)
        protein = np.searchsorted(starts, picks, side="right") - 1
        offset = picks - starts[protein]
        for pi, off in zip(protein.tolist(), offset.tolist()):
            acc, seq = records[pi]
            pep = seq[off:off + PEPTIDE_LENGTH]
            if not allowed.issuperset(pep):
                rejected += 1
                continue
            seen.setdefault(pep, (acc, off))
            if len(seen) >= n:
                break
    if len(seen) < n:
        raise RuntimeError(f"drew only {len(seen):,} distinct clean 9-mers of {n:,}")

    frame = pd.DataFrame(
        [(pep, acc, off) for pep, (acc, off) in sorted(seen.items())],
        columns=["peptide", "protein_source", "protein_offset"])
    stats_ = {
        "n_proteins_used": len(records),
        "n_9mer_positions": total,
        "n_sampled": len(frame),
        "n_rejected_nonstandard_residue": rejected,
        "sampling": "uniform over 9-mer start positions, distinct peptides kept",
        "seed": seed,
    }
    return frame, stats_


def affinity_training_frame(df: pd.DataFrame, fold: pd.Series,
                            ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray,
                                       np.ndarray, dict]:
    """The affinity predictor's training data, under stage 2b's exclusion rule.

    Held out from the predictor's *training* data, not just from the augmented
    rows: every reference peptide within 3 substitutions of an inner-dev,
    validation or test peptide is dropped before fitting. That is what lets the
    provenance record say the predictor never saw the neighbourhood of a
    held-out stability peptide.

    The fit/dev cut is by whole single-linkage clusters at Hamming <= 3 over the
    *reference* peptides, the same rule the frozen splits use, so reported
    accuracy is accuracy on distant peptides rather than on near-duplicates of
    the training set.

    Returns ``(frame, X, y, is_fit, exclusion_counts)``.
    """
    ref = pd.read_csv(AFFINITY_CSV)
    ref = ref[~ref["engineered_construct_mismatch"]].copy()
    ref["allele"] = ref["dataset_allele"]

    holdout = holdout_peptides(df, fold)
    peptides = sorted(ref["peptide"].unique())
    d_holdout = pd.Series(min_hamming(peptides, holdout), index=peptides)
    keep = ref["peptide"].map(d_holdout) >= MIN_HAMMING
    dropped_rows, dropped_peptides = int((~keep).sum()), int(
        ref.loc[~keep, "peptide"].nunique())
    ref = ref[keep].copy()

    ref = attach_hla(ref[["allele", "peptide", "affinity_nM", "inequality"]], df)
    ref["y"] = affinity_to_unit(ref["affinity_nM"].to_numpy())

    kept_peptides = sorted(ref["peptide"].unique())
    labels = single_linkage_clusters(
        np.asarray([[AMINO_ACIDS.index(a) for a in p] for p in kept_peptides],
                   dtype=np.int8))
    cluster = pd.Series(labels, index=kept_peptides)
    ref["cluster_id"] = ref["peptide"].map(cluster).astype(np.int32)
    placement = assign_clusters(
        ref.groupby("cluster_id").size(),
        fractions=(1 - AFFINITY_DEV_FRACTION, AFFINITY_DEV_FRACTION),
        names=("fit", "dev"))
    is_fit = (ref["cluster_id"].map(placement) == "fit").to_numpy()

    X = build_features(ref, AFFINITY_INPUT_SET, AFFINITY_ENCODING)
    y = ref["y"].to_numpy(dtype=np.float32)
    return ref, X, y, is_fit, {"excluded_rows_near_holdout": dropped_rows,
                               "excluded_peptides_near_holdout": dropped_peptides}


def train_affinity_predictor(df: pd.DataFrame, fold: pd.Series,
                             ) -> tuple[MLPRegressor, dict, list[str]]:
    """Fit the predictor that labels the predicted-affinity arm."""
    ref, X, y, is_fit, excluded = affinity_training_frame(df, fold)
    model = MLPRegressor(MLPConfig(hidden=AFFINITY_HIDDEN, l2=AFFINITY_L2,
                                   seed=AFFINITY_SEED,
                                   max_epochs=AFFINITY_MAX_EPOCHS,
                                   patience=AFFINITY_PATIENCE))
    model.fit(X[is_fit], y[is_fit], X[~is_fit], y[~is_fit])

    pred = model.predict(X[~is_fit])
    truth = y[~is_fit]
    is_weak = truth <= WEAK_UNIT
    called_weak = pred <= WEAK_UNIT
    exact = (ref.loc[~is_fit, "inequality"] == "=").to_numpy()
    tp = int((called_weak & is_weak).sum())
    provenance = {
        "role": "labels the predicted-affinity arm; never scored on stability",
        "architecture": f"MLP {AFFINITY_INPUT_SET}/{AFFINITY_ENCODING} "
                        f"hidden={'x'.join(map(str, AFFINITY_HIDDEN))} "
                        f"l2={AFFINITY_L2:g} seed={AFFINITY_SEED}",
        "target": "1 - log(affinity_nM)/log(50000), clipped to [0, 1]",
        "inequality_handling": ("the recorded bound is used as the point value; "
                                "'>' rows are therefore trained toward a value "
                                "stronger than the truth, which under-calls "
                                "weakness and is the conservative direction for "
                                "a weak-binder flag"),
        "training_data": str(AFFINITY_CSV.relative_to(REPO_ROOT)),
        "training_snapshot": "MHCflurry curated 2023-10-23 + BD2013 2013-02-22",
        **excluded,
        "n_fit_rows": int(is_fit.sum()),
        "n_dev_rows": int((~is_fit).sum()),
        "dev_fold": "whole single-linkage clusters at Hamming <= 3 over the "
                    "reference peptides",
        "best_epoch": model.best_epoch_,
        "epochs_run": model.n_epochs_,
        "fit_seconds": round(model.fit_seconds_, 1),
        "dev_spearman": round(float(stats.spearmanr(pred, truth).statistic), 4),
        "dev_pearson": round(float(stats.pearsonr(pred, truth).statistic), 4),
        "dev_mae_unit": round(float(np.abs(pred - truth).mean()), 4),
        "dev_weak_base_rate": round(float(is_weak.mean()), 4),
        "dev_weak_precision": round(tp / max(1, int(called_weak.sum())), 4),
        "dev_weak_recall": round(tp / max(1, int(is_weak.sum())), 4),
        "dev_weak_precision_on_exact_rows": round(
            float((called_weak & is_weak)[exact].sum()
                  / max(1, int(called_weak[exact].sum()))), 4),
        "weak_threshold_nM": WEAK_NM,
        "weak_threshold_unit": round(WEAK_UNIT, 4),
    }
    provenance["dev_peptides"] = int(ref.loc[~is_fit, "peptide"].nunique())
    return model, provenance, sorted(ref.loc[~is_fit, "peptide"].unique())


#: Configs compared by ``--sweep-predictor``. The first is what the manifests
#: actually used; the rest exist to show that the weak-call recall below is a
#: property of the target, not of this architecture.
PREDICTOR_SWEEP = (
    ((256, 64), 1e-5, 10, 150),      # the shipped config
    ((256, 64), 1e-4, 30, 300),
    ((512, 128), 1e-5, 30, 300),
    ((256, 128, 64), 1e-4, 30, 300),
)


def sweep_predictor(df: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """Why the weak-call recall is low, measured rather than asserted.

    Two things come out of this, and the second is the explanation:

    - **Architecture barely matters.** Wider, deeper and more regularised
      networks move held-out Spearman by a hundredth and never lift recall.
    - **A third of the target sits exactly on the decision threshold.** The IEDB
      convention records a competitive-assay non-binder as ``>20000``, so those
      rows land on the ``20,000 nM`` boundary itself. An MSE regressor shrinks
      toward the conditional mean, so most of that mass is predicted on the
      *non-weak* side of a boundary it sits exactly on. The per-band call rates
      in the output show it directly: even rows *below* the threshold are
      called weak only about a quarter of the time.

    Writes ``reports/stage2b_predictor_sweep.csv``.
    """
    ref, X, y, is_fit, _ = affinity_training_frame(df, fold)
    truth = y[~is_fit]
    at_threshold = np.abs(truth - WEAK_UNIT) < 1e-6
    below = truth < WEAK_UNIT - 1e-6
    above = ~at_threshold & ~below
    is_weak = truth <= WEAK_UNIT

    rows = []
    for hidden, l2, patience, max_epochs in PREDICTOR_SWEEP:
        model = MLPRegressor(MLPConfig(hidden=hidden, l2=l2, seed=AFFINITY_SEED,
                                       max_epochs=max_epochs, patience=patience))
        model.fit(X[is_fit], y[is_fit], X[~is_fit], y[~is_fit])
        pred = model.predict(X[~is_fit])
        called = pred <= WEAK_UNIT
        tp = int((called & is_weak).sum())
        rows.append({
            "hidden": "x".join(map(str, hidden)), "l2": l2,
            "patience": patience, "max_epochs": max_epochs,
            "shipped": hidden == AFFINITY_HIDDEN and l2 == AFFINITY_L2
                       and patience == AFFINITY_PATIENCE,
            "best_epoch": model.best_epoch_, "epochs": model.n_epochs_,
            "fit_seconds": round(model.fit_seconds_, 1),
            "dev_spearman": round(float(stats.spearmanr(pred, truth).statistic), 4),
            "dev_mae_unit": round(float(np.abs(pred - truth).mean()), 4),
            "weak_precision": round(tp / max(1, int(called.sum())), 4),
            "weak_recall": round(tp / max(1, int(is_weak.sum())), 4),
            "weak_call_rate": round(float(called.mean()), 4),
            # The diagnosis: what fraction of each target band gets called weak.
            "share_of_dev_below_threshold": round(float(below.mean()), 4),
            "share_of_dev_at_threshold": round(float(at_threshold.mean()), 4),
            "called_weak_given_below": round(float(called[below].mean()), 4),
            "called_weak_given_at_threshold": round(float(called[at_threshold].mean()), 4),
            "called_weak_given_above": round(float(called[above].mean()), 4),
        })
    frame = pd.DataFrame(rows)
    REPORT_DIR.mkdir(exist_ok=True)
    frame.to_csv(REPORT_DIR / "stage2b_predictor_sweep.csv", index=False)
    return frame


def label_fidelity(model: MLPRegressor, dev_peptides: list[str],
                   df: pd.DataFrame) -> dict:
    """How often is the assumed zero actually right, per source?

    The only place a weak-binder call can be checked against a half-life is the
    set of pairs carrying *both* measurements. This compares the two sources
    there: among pairs each one would label assumed-zero, what share really sit
    at the assay floor, and what share exceed 2 hours?

    Two restrictions keep it honest:

    - **Training rows only.** The comparison consults measured stability, so it
      is confined to ``split == "train"``. Reading validation or test labels
      here would spend the split that selects the arms.
    - **The predictor's own dev peptides only.** The predictor was fitted on
      affinity for peptides that include training-split peptides, so scoring it
      on all of them would report in-sample accuracy and flatter the predicted
      arm. Restricting to the peptides held out of its fit makes the two
      sources' error rates comparable.

    Neither source measures a half-life, so neither number is expected to be
    perfect. The point is which one is *less* wrong, and by how much.
    """
    ref = pd.read_csv(AFFINITY_CSV)
    ref = ref[~ref["engineered_construct_mismatch"]].copy()
    ref["allele"] = ref["dataset_allele"]
    train = df[df.split == "train"]
    dual = ref.merge(train[["allele", "peptide", "thalf_hours"]],
                     on=["allele", "peptide"], how="inner")
    dual = dual[dual["peptide"].isin(set(dev_peptides))]
    if not len(dual):
        return {"n_dual_measured_train_pairs": 0}

    scored = attach_hla(dual[["allele", "peptide"]], df)
    unit = model.predict(build_features(scored, AFFINITY_INPUT_SET, AFFINITY_ENCODING))

    def stats_for(mask: np.ndarray, label: str) -> dict:
        hours = dual.loc[mask, "thalf_hours"]
        if not len(hours):
            return {f"{label}_n": 0}
        return {
            f"{label}_n": int(len(hours)),
            f"{label}_share_at_floor": round(float((hours == 0).mean()), 4),
            f"{label}_share_above_2h": round(float((hours > 2.0).mean()), 4),
            f"{label}_median_thalf_hours": round(float(hours.median()), 3),
        }

    out = {"n_dual_measured_train_pairs": int(len(dual)),
           "restriction": ("training split only, and only peptides held out of "
                           "the affinity predictor's own fit")}
    out.update(stats_for((dual["affinity_nM"] >= WEAK_NM).to_numpy(), "measured_weak"))
    out.update(stats_for(unit <= WEAK_UNIT, "predicted_weak"))
    out.update(stats_for(np.ones(len(dual), dtype=bool), "all_dual_measured"))
    return out


def predicted_candidates(model: MLPRegressor, natural: pd.DataFrame,
                         df: pd.DataFrame, fold: pd.Series,
                         ) -> tuple[pd.DataFrame, dict]:
    """Score every allele against the natural 9-mers and keep the weak ones."""
    ref = pd.read_csv(AFFINITY_CSV)
    ref["allele"] = ref["dataset_allele"]
    # Any measured affinity for this exact pair overrides a prediction: a pair
    # measured stronger than the threshold is not a weak binder whatever the
    # model says.
    contradicts = set(zip(ref.loc[ref["affinity_nM"] < WEAK_NM, "allele"],
                          ref.loc[ref["affinity_nM"] < WEAK_NM, "peptide"]))
    measured_pairs = set(zip(df["allele"], df["peptide"]))

    natural = annotate_candidates(natural, df, fold)
    ledger = {
        "natural_peptides_sampled": len(natural),
        "natural_peptides_after_inner_folds_filter": int(natural["ok_inner_folds"].sum()),
        "natural_peptides_after_cv_folds_filter": int(natural["ok_cv_folds"].sum()),
    }
    for d in range(MIN_HAMMING):
        ledger[f"natural_peptides_at_hamming_{d}"] = int(
            (natural["min_dist_holdout"] == d).sum())
    natural = natural[natural["ok_inner_folds"]].reset_index(drop=True)

    alleles = sorted(df["allele"].unique())
    hla = df[["allele", "hla_seq", "hla_pseudoseq"]].drop_duplicates().set_index("allele")
    chunks = []
    n_scored = n_called_weak = n_contradicted = n_already_measured = 0
    for allele in alleles:
        frame = natural[["peptide", "protein_source", "protein_offset",
                         "min_dist_holdout", "min_dist_train",
                         "ok_inner_folds", "ok_cv_folds"]].copy()
        frame["allele"] = allele
        frame["hla_seq"] = hla.loc[allele, "hla_seq"]
        frame["hla_pseudoseq"] = hla.loc[allele, "hla_pseudoseq"]
        X = build_features(frame, AFFINITY_INPUT_SET, AFFINITY_ENCODING)
        unit = model.predict(X)
        n_scored += len(frame)

        frame["affinity_nM"] = unit_to_affinity(unit)
        weak = frame[unit <= WEAK_UNIT].copy()
        n_called_weak += len(weak)
        before = len(weak)
        pairs = list(zip(weak["allele"], weak["peptide"]))
        weak = weak[[p not in contradicts for p in pairs]]
        n_contradicted += before - len(weak)
        before = len(weak)
        pairs = list(zip(weak["allele"], weak["peptide"]))
        weak = weak[[p not in measured_pairs for p in pairs]]
        n_already_measured += before - len(weak)
        chunks.append(weak)

    pool = pd.concat(chunks, ignore_index=True)
    pool["affinity_kind"] = "predicted"
    pool["inequality"] = "="
    ledger.update({
        "pairs_scored": n_scored,
        "pairs_predicted_weak": n_called_weak,
        "rejected_contradicted_by_measured_affinity": n_contradicted,
        "rejected_pair_already_has_measured_stability": n_already_measured,
        "pool_rows": len(pool),
        "pool_alleles": int(pool["allele"].nunique()),
        "pool_peptides": int(pool["peptide"].nunique()),
    })
    return pool, ledger


# --- orchestration --------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-natural", type=int, default=N_NATURAL_PEPTIDES,
                    help=f"natural 9-mers to draw (default {N_NATURAL_PEPTIDES:,})")
    ap.add_argument("--sweep-predictor", action="store_true",
                    help="measure why the weak-call recall is low, then exit. "
                         "Writes reports/stage2b_predictor_sweep.csv.")
    ap.add_argument("--skip-predicted", action="store_true",
                    help="build only the measured arm -- the documented fallback "
                         "if the predictor threatens the time box")
    args = ap.parse_args()

    started = time.perf_counter()
    df = load_with_splits()
    train = df[df.split == "train"]
    fold = inner_folds(train)
    fit = train[(fold == "fit").to_numpy()]

    if args.sweep_predictor:
        frame = sweep_predictor(df, fold)
        pd.set_option("display.width", 220)
        print("affinity-predictor sweep (dev fold = whole Hamming <= 3 clusters):")
        print(frame[["hidden", "l2", "shipped", "best_epoch", "dev_spearman",
                     "weak_precision", "weak_recall", "weak_call_rate"]
                    ].to_string(index=False))
        row = frame.iloc[0]
        print(f"\nwhy recall is capped: {row.share_of_dev_at_threshold:.1%} of the "
              f"dev target sits *exactly* on the {WEAK_NM:g} nM threshold "
              "(the IEDB '>20000' convention).")
        print(f"  called weak | below threshold       "
              f"{row.called_weak_given_below:.3f}")
        print(f"  called weak | exactly at threshold  "
              f"{row.called_weak_given_at_threshold:.3f}")
        print(f"  called weak | above threshold       "
              f"{row.called_weak_given_above:.3f}")
        print("\nMSE shrinkage puts most of the at-threshold mass on the "
              "non-weak side of a boundary it sits on.")
        print("\nwrote reports/stage2b_predictor_sweep.csv")
        return 0

    print(f"train={len(train):,} -> fit={len(fit):,} / dev={len(train) - len(fit):,}")
    print(f"held-out peptides (inner-dev + val + test): "
          f"{len(holdout_peptides(df, fold)):,}")
    print(f"weak-binder threshold: {WEAK_NM:g} nM  "
          f"(unit score <= {WEAK_UNIT:.4f})\n")

    print("measured-affinity arm")
    measured_pool, measured_ledger = measured_candidates(df, fold)
    for k, v in measured_ledger.items():
        print(f"  {k:48s} {v:>10,}")

    predicted_pool = pd.DataFrame()
    predicted_ledger: dict = {"skipped": True}
    predictor_provenance: dict = {"skipped": True}
    fidelity: dict = {"skipped": True}
    proteome_info: dict = {"skipped": True}
    sample_info: dict = {"skipped": True}
    if not args.skip_predicted:
        print("\npredicted-affinity arm")
        fasta, proteome_info = fetch_proteome()
        print(f"  proteome: {proteome_info['n_proteins']:,} reviewed human proteins")
        natural, sample_info = sample_natural_peptides(
            fasta, args.n_natural, PEPTIDE_SAMPLE_SEED)
        print(f"  sampled {len(natural):,} distinct natural 9-mers from "
              f"{sample_info['n_9mer_positions']:,} positions "
              f"({sample_info['n_rejected_nonstandard_residue']:,} windows "
              f"rejected for non-standard residues)")
        model, predictor_provenance, dev_peptides = train_affinity_predictor(df, fold)
        print(f"  affinity predictor: {predictor_provenance['n_fit_rows']:,} fit / "
              f"{predictor_provenance['n_dev_rows']:,} dev rows, "
              f"{predictor_provenance['fit_seconds']}s")
        print(f"    held-out Spearman {predictor_provenance['dev_spearman']:+.3f}, "
              f"weak-call precision {predictor_provenance['dev_weak_precision']:.3f} "
              f"at base rate {predictor_provenance['dev_weak_base_rate']:.3f}")
        predicted_pool, predicted_ledger = predicted_candidates(
            model, natural, df, fold)
        for k, v in predicted_ledger.items():
            print(f"  {k:48s} {v:>10,}")

        fidelity = label_fidelity(model, dev_peptides, df)
        print(f"\nassumed-zero label fidelity on {fidelity['n_dual_measured_train_pairs']:,} "
              "dual-measured training pairs")
        print("  (training rows only, and only peptides held out of the "
              "predictor's own fit)")
        for label in ("all_dual_measured", "measured_weak", "predicted_weak"):
            if not fidelity.get(f"{label}_n"):
                continue
            print(f"  {label:20s} n={fidelity[f'{label}_n']:>5,}  "
                  f"at floor {fidelity[f'{label}_share_at_floor']:.1%}  "
                  f">2h {fidelity[f'{label}_share_above_2h']:.1%}  "
                  f"median {fidelity[f'{label}_median_thalf_hours']:.2f} h")

    # --- matched counts ---------------------------------------------------
    cap = per_allele_cap(fit, CAP_FRACTION)
    measured_avail = measured_pool.groupby("allele").size()
    predicted_avail = (predicted_pool.groupby("allele").size()
                       if len(predicted_pool) else pd.Series(dtype=int))

    sources = [measured_avail] if args.skip_predicted else [measured_avail,
                                                            predicted_avail]
    n_matched = matched_counts(cap, *sources)
    n_full = matched_counts(cap, predicted_avail) if len(predicted_pool) else n_matched

    coverage = pd.DataFrame({
        "fit_rows": fit.groupby("allele").size(),
        "cap": cap,
        "measured_avail": measured_avail.reindex(cap.index).fillna(0).astype(int),
        "predicted_avail": predicted_avail.reindex(cap.index).fillna(0).astype(int),
        "n_matched": n_matched,
        "n_predicted_full": n_full,
    }).sort_values("n_matched", ascending=False)

    print(f"\nmatched counts: {int(n_matched.sum()):,} rows per arm "
          f"({int(n_matched.sum()) / len(fit):.1%} of fit rows), "
          f"{int((n_matched > 0).sum())} of {len(cap)} alleles")
    if len(predicted_pool):
        print(f"full-coverage predicted arm: {int(n_full.sum()):,} rows, "
              f"{int((n_full > 0).sum())} of {len(cap)} alleles")

    # --- draw and write ---------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    measured_rows = sample_per_allele(measured_pool, n_matched, MEASURED_DRAW_SEED)
    measured_manifest = finalise_manifest(
        measured_rows, "measured_affinity",
        f"assumed thalf=0: measured affinity >= {WEAK_NM:g} nM "
        f"(MHCflurry curated 2023-10-23)")
    written["measured_affinity"] = measured_manifest

    if len(predicted_pool):
        # One draw serves both predicted arms: the full sample is taken first
        # and the matched manifest is its per-allele head, so the only
        # difference between the two is the extra rows.
        full_rows = sample_per_allele(predicted_pool, n_full, PREDICTED_DRAW_SEED)
        written["predicted_affinity_full"] = finalise_manifest(
            full_rows, "predicted_affinity",
            f"assumed thalf=0: predicted affinity > {WEAK_NM:g} nM "
            f"(in-house MLP, see provenance)")
        keep = full_rows["draw_rank"] < full_rows["allele"].map(n_matched).fillna(0)
        written["predicted_affinity"] = finalise_manifest(
            full_rows[keep.to_numpy()], "predicted_affinity",
            f"assumed thalf=0: predicted affinity > {WEAK_NM:g} nM "
            f"(in-house MLP, see provenance)")
        nested = set(zip(written["predicted_affinity"]["allele"],
                         written["predicted_affinity"]["peptide"]))
        superset = set(zip(written["predicted_affinity_full"]["allele"],
                           written["predicted_affinity_full"]["peptide"]))
        if not nested <= superset:
            raise AssertionError("the matched predicted manifest is not a subset "
                                 "of the full-coverage one")

    for name, manifest in written.items():
        path = OUT_DIR / f"{name}.csv"
        manifest.to_csv(path, index=False)
        print(f"\nwrote {path.relative_to(REPO_ROOT)}: {len(manifest):,} rows, "
              f"{manifest['allele'].nunique()} alleles, "
              f"{manifest['peptide'].nunique():,} peptides")
        print(f"  min Hamming to a held-out peptide: "
              f"{int(manifest['min_dist_holdout'].min())} "
              f"(rule: >= {MIN_HAMMING})")
        print(f"  rows also admissible under cv_folds(): "
              f"{int(manifest['ok_cv_folds'].sum()):,} of {len(manifest):,}")

    coverage.to_csv(REPORT_DIR / "stage2b_coverage.csv")
    provenance = {
        "built_by": "scripts/augment_affinity.py",
        "weak_binder_threshold_nM": WEAK_NM,
        "cap_fraction_of_fit_rows": CAP_FRACTION,
        "min_hamming_to_holdout": MIN_HAMMING,
        "exclusion_rule": ("no augmented peptide within Hamming "
                           f"{MIN_HAMMING - 1} of any inner-dev, validation or "
                           "test peptide, across all alleles"),
        "assumed_label": {"thalf_hours": 0.0, "y_log1p": 0.0,
                          "note": "assumed, never measured; measured stability "
                                  "is never overwritten"},
        "seeds": {"natural_peptide_sample": PEPTIDE_SAMPLE_SEED,
                  "measured_draw": MEASURED_DRAW_SEED,
                  "predicted_draw": PREDICTED_DRAW_SEED},
        "measured_arm": measured_ledger,
        "predicted_arm": {"proteome": proteome_info, "sampling": sample_info,
                          "predictor": predictor_provenance,
                          "selection": predicted_ledger},
        "assumed_label_fidelity": fidelity,
        # The digests make a committed manifest verifiable without re-running
        # the build; data/SHA256SUMS stays reserved for the read-only raw CSV.
        "manifests": {name: {"rows": len(m), "alleles": int(m["allele"].nunique()),
                             "peptides": int(m["peptide"].nunique()),
                             "ok_cv_folds_rows": int(m["ok_cv_folds"].sum()),
                             "sha256": hashlib.sha256(
                                 (OUT_DIR / f"{name}.csv").read_bytes()).hexdigest()}
                      for name, m in written.items()},
        "matched_rows_per_arm": int(n_matched.sum()),
        "matched_alleles": int((n_matched > 0).sum()),
        "total_alleles": int(len(cap)),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n")
    print(f"\nwrote {(OUT_DIR / 'provenance.json').relative_to(REPO_ROOT)} and "
          f"reports/stage2b_coverage.csv")
    print(f"build time: {provenance['build_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
