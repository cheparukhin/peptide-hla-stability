"""Weak-binder augmentation: candidate filtering, leakage checks, matched counts.

Stage 2b tests whether adding assumed-zero *negative* rows helps, and whether
the **source** of those negatives matters. Two sources, one schema:

- **measured affinity** -- pairs from ``data/data_augmentation_iedb`` whose
  measurement or lower bound establishes affinity >= 20,000 nM.
- **predicted affinity** -- random natural 9-mers whose *predicted* affinity on
  that allele is weaker than 20,000 nM, following Rasmussen et al.'s recipe.

Neither source measures a half-life. Both are assigned ``thalf_hours = 0``, so
the augmented label is an **assumption**, recorded as such in the manifest's
``label_provenance`` column and never written back over a measured stability
value.

The one thing this module exists to get right is leakage. A candidate is
admissible only if its peptide is at least :data:`MIN_HAMMING` substitutions
from every held-out peptide, **across all alleles** -- the frozen splits group
peptides globally, so a peptide held out under one allele is held out under all
of them. The reference table's own ``padding_eligible`` flag is not enough: it
matches on the ``(allele, peptide)`` pair, so a validation peptide reappears as
"absent from the stability dataset" whenever it was measured on some other
allele.

Which peptides count as held out depends on the stopping-fold scheme of the arm
under test, and the two schemes are very different:

- ``inner_folds()`` (single networks, stage 2b's pilot) keeps one permanent dev
  fold, so the held-out set is dev + val + test.
- ``cv_folds()`` (the 30-network ensemble) makes *every* training peptide some
  member's stopping peptide, so a candidate must clear the threshold against
  every training peptide too.

:func:`annotate_candidates` measures both distances and flags both rules, so one
manifest serves either protocol and the stricter count is always visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PEPTIDE_LENGTH, encode_sequences

#: Affinity at or above this, in nM, is a weak binder. The threshold stage 2b
#: predeclares for both sources; it is also the convention the affinity
#: reference's ``is_weak_binder`` column uses.
WEAK_NM = 20000.0

#: A candidate peptide must be at least this many substitutions from every
#: held-out peptide. Same threshold as the frozen split's clustering, so an
#: augmented peptide is no closer to a held-out peptide than a training peptide
#: is.
MIN_HAMMING = 4

#: Per-allele cap on added rows, as a fraction of that allele's real fit rows.
#: Keeps augmentation from swamping a thinly measured allele, and keeps the
#: added block small enough that the comparison is about the labels rather than
#: about training-set size.
CAP_FRACTION = 0.25

#: Sample weights tried for the assumed-zero rows. Measured rows are always 1.
ASSUMED_WEIGHTS = (0.1, 0.25)

#: The label every augmented row carries, on both scales.
ASSUMED_THALF_HOURS = 0.0
ASSUMED_Y_LOG1P = 0.0

#: Columns every candidate manifest carries, in order. Both sources write the
#: same schema so a model arm can swap one file for the other.
MANIFEST_COLUMNS = (
    "allele",              # dataset_allele form, joins to the stability CSV
    "peptide",
    "source",              # measured_affinity | predicted_affinity
    "thalf_hours",         # always 0.0 -- assumed, not measured
    "y_log1p",             # always 0.0
    "label_provenance",    # free text: what justifies the assumed zero
    "affinity_kind",       # measured | predicted
    "affinity_nM",
    "inequality",          # '=' or '>' for measured; '=' for predicted
    "protein_source",      # UniProt accession for predicted; '' for measured
    "protein_offset",      # 0-based start of the 9-mer; -1 for measured
    "min_dist_holdout",    # Hamming to nearest dev/val/test peptide
    "min_dist_train",      # Hamming to nearest training peptide
    "ok_inner_folds",      # admissible under the single-network protocol
    "ok_cv_folds",         # admissible under the 30-network ensemble protocol
)


def min_hamming(candidates: list[str], reference: list[str],
                chunk: int = 2048) -> np.ndarray:
    """Hamming distance from each candidate to its nearest reference peptide.

    Equal-length 9-mers throughout, so this is exact and needs no alignment.
    Returns :data:`PEPTIDE_LENGTH` for every candidate when ``reference`` is
    empty -- "as far away as a 9-mer can be", which is the right identity for a
    minimum over nothing.
    """
    if not len(candidates):
        return np.zeros(0, dtype=np.int8)
    if not len(reference):
        return np.full(len(candidates), PEPTIDE_LENGTH, dtype=np.int8)
    ref_codes = encode_sequences(reference, PEPTIDE_LENGTH)
    cand_codes = encode_sequences(candidates, PEPTIDE_LENGTH)
    out = np.empty(len(cand_codes), dtype=np.int8)
    for start in range(0, len(cand_codes), chunk):
        block = cand_codes[start:start + chunk]
        dist = (block[:, None, :] != ref_codes[None, :, :]).sum(axis=2)
        out[start:start + len(block)] = dist.min(axis=1)
    return out


def holdout_peptides(df: pd.DataFrame, fold: pd.Series) -> list[str]:
    """Peptides no augmented row may come near: inner-dev, validation, test.

    ``fold`` is the ``"fit"``/``"dev"`` series that
    :func:`scripts.baseline_sequence.inner_folds` returns for the training
    split, aligned to the training rows of ``df``.

    Validation and test come straight from the frozen splits. The inner dev fold
    joins them because it chooses the stopping epoch: a candidate sitting next
    to a dev peptide would tune that epoch on a near-duplicate of a row the
    model also fits.
    """
    train = df[df.split == "train"]
    if len(fold) != len(train):
        raise ValueError(f"fold has {len(fold)} entries but the training split "
                         f"has {len(train)} rows")
    dev = train.loc[np.asarray(fold) == "dev", "peptide"]
    held = df.loc[df.split.isin(["val", "test"]), "peptide"]
    return sorted(set(dev) | set(held))


def annotate_candidates(candidates: pd.DataFrame, df: pd.DataFrame,
                        fold: pd.Series) -> pd.DataFrame:
    """Attach both distance columns and both admissibility flags.

    ``candidates`` needs a ``peptide`` column; everything else is passed
    through. Distances are computed per *unique peptide*, not per row -- a
    candidate peptide typically appears on several alleles.
    """
    holdout = holdout_peptides(df, fold)
    train_peptides = sorted(df.loc[df.split == "train", "peptide"].unique())
    peptides = sorted(candidates["peptide"].unique())

    d_holdout = pd.Series(min_hamming(peptides, holdout), index=peptides)
    d_train = pd.Series(min_hamming(peptides, train_peptides), index=peptides)

    out = candidates.copy()
    out["min_dist_holdout"] = out["peptide"].map(d_holdout).astype(np.int8)
    out["min_dist_train"] = out["peptide"].map(d_train).astype(np.int8)
    out["ok_inner_folds"] = out["min_dist_holdout"] >= MIN_HAMMING
    # Under cv_folds() every training peptide is some member's stopping
    # peptide, so clearing the dev fold is not enough.
    out["ok_cv_folds"] = out["ok_inner_folds"] & (out["min_dist_train"] >= MIN_HAMMING)
    return out


def per_allele_cap(fit: pd.DataFrame, fraction: float = CAP_FRACTION) -> pd.Series:
    """Added-row ceiling per allele: ``fraction`` of its real fit rows, floored.

    Indexed by allele and covering every allele present in ``fit``, including
    the ones no source can supply -- those get a cap and then nothing, which is
    what keeps them in the evaluation panel with zero added rows.
    """
    return (fit.groupby("allele").size() * fraction).astype(int).rename("cap")


def matched_counts(cap: pd.Series, *available: pd.Series) -> pd.Series:
    """How many rows each allele gets, equal across every source.

    The per-allele count is ``min(cap, availability in each source)``, so the
    arms differ in *where the negatives came from* and in nothing else. An
    allele that one source cannot supply gets 0 from **all** of them rather than
    being dropped: it still carries measured rows, so it stays in the
    evaluation panel and the comparison keeps a fixed allele set.
    """
    counts = cap.copy()
    for series in available:
        counts = np.minimum(counts, series.reindex(counts.index).fillna(0).astype(int))
    return counts.astype(int).rename("n_add")


def sample_per_allele(candidates: pd.DataFrame, counts: pd.Series,
                      seed: int) -> pd.DataFrame:
    """Take ``counts[allele]`` rows per allele, deterministically.

    One generator seeded once, then alleles drawn in sorted order, so the sample
    is reproducible from ``seed`` alone and does not shift when an unrelated
    allele's count changes.

    Rows come back in **draw order** within each allele, which is what lets one
    draw serve two nested arms: taking the first *k* rows of an allele's block
    is a uniform random subsample of that allele's sample, so the matched-count
    manifest can be derived from the full-coverage one instead of drawn
    separately. Two independent draws would differ in *which* peptides they hold
    as well as how many, and the coverage comparison would confound the two.
    """
    rng = np.random.default_rng(seed)
    picked = []
    for allele in sorted(counts.index):
        want = int(counts[allele])
        if want <= 0:
            continue
        pool = candidates[candidates["allele"] == allele]
        if len(pool) < want:
            raise ValueError(f"{allele}: asked for {want} candidates but the "
                             f"pool holds {len(pool)}")
        # Sort first so the pool order does not depend on how the frame was
        # built, then draw without replacement.
        pool = pool.sort_values("peptide")
        take = rng.choice(len(pool), size=want, replace=False)
        picked.append(pool.iloc[take].assign(draw_rank=np.arange(want)))
    if not picked:
        return candidates.iloc[:0].copy()
    return pd.concat(picked, ignore_index=True)


def finalise_manifest(rows: pd.DataFrame, source: str,
                      provenance: str) -> pd.DataFrame:
    """Stamp the assumed label and provenance, and order the shared columns."""
    out = rows.copy()
    out["source"] = source
    out["thalf_hours"] = ASSUMED_THALF_HOURS
    out["y_log1p"] = ASSUMED_Y_LOG1P
    out["label_provenance"] = provenance
    for col in MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in ("protein_source", "inequality") else -1
    return out[list(MANIFEST_COLUMNS)].sort_values(
        ["allele", "peptide"]).reset_index(drop=True)


def attach_hla(manifest: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Join each augmented row to its allele's HLA sequences.

    The manifests store ``(allele, peptide)`` only; the 34-residue
    pseudosequence and the 182-residue domain come from the stability CSV at use
    time, so a manifest cannot drift out of step with the dataset. Alleles are
    1:1 with both sequences, asserted here rather than assumed.
    """
    key = df[["allele", "hla_seq", "hla_pseudoseq"]].drop_duplicates()
    if key["allele"].duplicated().any():
        raise ValueError("an allele carries more than one HLA sequence")
    merged = manifest.merge(key, on="allele", how="left", validate="many_to_one")
    missing = merged["hla_pseudoseq"].isna()
    if missing.any():
        names = sorted(merged.loc[missing, "allele"].unique())
        raise ValueError(f"alleles absent from the stability dataset: {names}")
    return merged


def verify_manifest(manifest: pd.DataFrame, df: pd.DataFrame, fold: pd.Series,
                    protocol: str = "inner_folds") -> dict:
    """Re-check a manifest from disk against the splits. Raises on a leak.

    Called by the training script before any augmented row reaches a model, so a
    hand-edited or stale manifest cannot quietly contaminate a run. Checks the
    assumed label really is zero, the distance rule really holds under the
    requested protocol, and no augmented ``(allele, peptide)`` pair duplicates a
    measured one.
    """
    if protocol not in ("inner_folds", "cv_folds"):
        raise ValueError(f"unknown protocol {protocol!r}")
    if not len(manifest):
        return {"n_rows": 0, "n_peptides": 0, "n_alleles": 0,
                "min_dist_holdout": PEPTIDE_LENGTH, "min_dist_train": PEPTIDE_LENGTH}

    if not (manifest["thalf_hours"] == ASSUMED_THALF_HOURS).all():
        raise ValueError("an augmented row carries a non-zero half-life")
    if not np.allclose(manifest["y_log1p"], ASSUMED_Y_LOG1P):
        raise ValueError("an augmented row carries a non-zero y_log1p")

    recomputed = annotate_candidates(manifest[["allele", "peptide"]], df, fold)
    column = "ok_cv_folds" if protocol == "cv_folds" else "ok_inner_folds"
    bad = ~recomputed[column].to_numpy()
    if bad.any():
        worst = int(recomputed.loc[bad, "min_dist_holdout"].min())
        raise ValueError(
            f"{int(bad.sum()):,} augmented row(s) violate the {protocol} "
            f"distance rule (nearest held-out peptide at Hamming {worst}, "
            f"need >= {MIN_HAMMING})")

    measured = set(zip(df["allele"], df["peptide"]))
    clash = [p for p in zip(manifest["allele"], manifest["peptide"]) if p in measured]
    if clash:
        raise ValueError(f"{len(clash):,} augmented pair(s) already carry a "
                         f"measured half-life, e.g. {clash[0]}")
    if manifest.duplicated(["allele", "peptide"]).any():
        raise ValueError("the manifest repeats an (allele, peptide) pair")

    return {
        "n_rows": len(manifest),
        "n_peptides": int(manifest["peptide"].nunique()),
        "n_alleles": int(manifest["allele"].nunique()),
        "min_dist_holdout": int(recomputed["min_dist_holdout"].min()),
        "min_dist_train": int(recomputed["min_dist_train"].min()),
    }
