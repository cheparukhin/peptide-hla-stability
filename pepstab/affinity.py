"""Auxiliary binding-affinity labels for stage 2c, and the leakage rules on them.

Affinity is how strongly a peptide binds; stability -- our target -- is how long
it stays bound once it has. They are different measurements of different events,
which is why an auxiliary affinity head is a real experiment and not a
restatement of the target. On the 7,281 pairs carrying both, they correlate at
Spearman -0.491 (see docs/AFFINITY_REFERENCE.md): same direction, plenty of
disagreement in detail.

Two label sets come out of the same reference table:

- :func:`dual_labelled` -- affinity for pairs that *already* have a measured
  half-life. These rows are inside the frozen splits, so attaching a second
  label to a training row adds no peptide and cannot move a peptide across a
  split boundary. This is the probe.
- :func:`auxiliary_only` -- affinity for peptides absent from the stability set
  entirely. These are new peptides, so they must clear the distance rule in
  :func:`filter_by_distance` before any of them is trained on. This is the
  expansion.

The expansion is where leakage lives, and ``padding_eligible`` in the reference
table does **not** protect against it: it tests the exact ``(allele, peptide)``
pair, while the frozen splits group peptides across every allele. A peptide held
out in val or test reappears as "not in the stability dataset" the moment it was
measured on some other allele. Hence :func:`held_out_peptides`, which works on
``peptide`` alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DATA_DIR, PEPTIDE_LENGTH, encode_sequences

AFFINITY_CSV = (DATA_DIR / "data_augmentation_iedb"
                / "affinity_reference_75alleles.csv")

#: The NetMHC-family affinity transform: ``1 - log10(nM) / log10(50000)``,
#: clipped to [0, 1]. 50,000 nM maps to 0 (no useful binding), 1 nM to 1.0.
#:
#: Why this and not ``log10`` directly: it puts the auxiliary target on the same
#: rough [0, 1] scale as the centred ``log1p`` stability target, so one
#: ``lambda_aff`` value means something comparable across the two losses. It is
#: also the convention the NetMHCpan family trains on, which keeps the auxiliary
#: task recognisable to anyone reading the result.
AFFINITY_SCALE_NM = 50_000.0

#: Candidate auxiliary peptides must sit strictly further than this from every
#: peptide the model is scored or stopped on. Same threshold as the frozen
#: splits' clustering, so an auxiliary peptide is no closer to a held-out
#: peptide than a training peptide is.
MAX_HAMMING_TO_HELD_OUT = 3


def affinity_to_target(affinity_nM) -> np.ndarray:
    """Transform nM affinities to the [0, 1] auxiliary target.

    Entries that are NaN stay NaN -- the multi-task loss reads NaN as "this row
    has no auxiliary label" and masks it out.
    """
    nm = np.asarray(affinity_nM, dtype=float)
    out = np.full(nm.shape, np.nan, dtype=float)
    ok = np.isfinite(nm) & (nm > 0)
    out[ok] = 1.0 - np.log10(nm[ok]) / np.log10(AFFINITY_SCALE_NM)
    return np.clip(out, 0.0, 1.0)


def load_reference(drop_censored: bool = False) -> pd.DataFrame:
    """Load the committed affinity reference, C67S mismatches already dropped.

    ``engineered_construct_mismatch`` rows carry a *wild-type* affinity against a
    C67S stability construct. C67S substitutes position 67, one of the 34
    peptide-contact positions, so the affinity describes a different groove from
    the one whose half-life was measured. Never join them; see
    docs/AFFINITY_REFERENCE.md.

    ``drop_censored`` additionally drops the rows whose measurement carries a
    ``<`` or ``>`` inequality (6.0% of the dual-labelled set). The default keeps
    them -- the stability target is itself left-censored and stage 1 chose to
    keep those rows rather than discard a fifth of the data -- but stage 2c runs
    the comparison both ways, because an auxiliary label is cheap to drop.
    """
    if not AFFINITY_CSV.exists():
        raise FileNotFoundError(
            f"{AFFINITY_CSV} is missing. It is a committed artifact; restore it "
            "from git or regenerate with scripts/fetch_affinity_reference.py."
        )
    ref = pd.read_csv(AFFINITY_CSV, dtype={
        "dataset_allele": "string", "allele": "string", "peptide": "string",
        "inequality": "string", "bd2013_inequality": "string",
    })
    ref = ref[~ref["engineered_construct_mismatch"].astype(bool)]
    if drop_censored:
        ref = ref[ref["inequality"].fillna("=") == "="]
    return ref.reset_index(drop=True)


def dual_labelled(df: pd.DataFrame, drop_censored: bool = False) -> pd.Series:
    """The auxiliary target for ``df``'s rows: NaN where no affinity exists.

    Joined on ``(allele, peptide)`` -- the pair, not the peptide -- because an
    affinity measurement belongs to one allele. The reference is already one row
    per pair (deduplicated to the strongest measurement), so the join is
    many-to-one and validated as such.

    Returns a Series aligned to ``df``'s index. Attaching the label to *every*
    split's rows is deliberate and safe: the caller trains on the training rows
    only, and having val and test affinities to hand is what lets the auxiliary
    head be diagnosed at all.
    """
    ref = load_reference(drop_censored=drop_censored)
    ref = ref[ref["in_stability_dataset"].astype(bool)]
    merged = df[["allele", "peptide"]].merge(
        ref[["dataset_allele", "peptide", "affinity_nM"]],
        left_on=["allele", "peptide"], right_on=["dataset_allele", "peptide"],
        how="left", validate="many_to_one")
    return pd.Series(affinity_to_target(merged["affinity_nM"].to_numpy()),
                     index=df.index, name="y_affinity")


def auxiliary_only(df: pd.DataFrame, drop_censored: bool = False) -> pd.DataFrame:
    """Affinity rows whose peptide is absent from the stability dataset entirely.

    Absence is tested on ``peptide`` across all alleles, not on the
    ``(allele, peptide)`` pair, so a peptide held out in val or test cannot
    re-enter under a different allele. Still needs
    :func:`filter_by_distance` -- absence is distance 0; near neighbours are a
    separate problem.

    ``df`` must be the full dataset (every split), since the exclusion is
    against every peptide the stability set contains.
    """
    ref = load_reference(drop_censored=drop_censored)
    in_stability = set(df["peptide"].astype(str))
    out = ref[~ref["peptide"].astype(str).isin(in_stability)].copy()
    out["y_affinity"] = affinity_to_target(out["affinity_nM"].to_numpy())
    return out.reset_index(drop=True)


def held_out_peptides(df: pd.DataFrame, dev_peptides=None) -> list[str]:
    """Every peptide an auxiliary candidate must stay away from.

    That is the union of the validation peptides, the test peptides, and the
    inner stopping fold's peptides. The stopping fold belongs in this set:
    under ``cv_folds()`` every training peptide is some ensemble member's
    stopping peptide, so an auxiliary peptide close to one of them would tune
    that member's epoch count on a near-duplicate.
    """
    held = set(df.loc[df["split"].isin(["val", "test"]), "peptide"].astype(str))
    if dev_peptides is not None:
        held |= set(pd.Series(dev_peptides).astype(str))
    return sorted(held)


def min_hamming_to(peptides, reference, chunk: int = 512) -> np.ndarray:
    """Minimum Hamming distance from each of ``peptides`` to any of ``reference``.

    All peptides are 9 residues, so Hamming is exact and needs no alignment --
    the same reason the frozen splits use it.
    """
    peptides = [str(p) for p in peptides]
    reference = [str(p) for p in reference]
    if not peptides:
        return np.empty(0, dtype=np.int8)
    if not reference:
        return np.full(len(peptides), PEPTIDE_LENGTH, dtype=np.int8)
    ref_codes = encode_sequences(reference, PEPTIDE_LENGTH)
    codes = encode_sequences(peptides, PEPTIDE_LENGTH)
    out = np.empty(len(codes), dtype=np.int8)
    for start in range(0, len(codes), chunk):
        block = codes[start:start + chunk]
        dist = (block[:, None, :] != ref_codes[None, :, :]).sum(axis=2)
        out[start:start + len(block)] = dist.min(axis=1)
    return out


def filter_by_distance(candidates: pd.DataFrame, held_out,
                       threshold: int = MAX_HAMMING_TO_HELD_OUT
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop candidates within ``threshold`` substitutions of any held-out peptide.

    Returns ``(kept, audit)``. ``audit`` is one row per distance value with the
    candidate peptide and row counts at that distance, so the exclusion is
    reported as a table rather than asserted in prose.

    The distance is computed once per *unique candidate peptide*, then fanned
    out to rows -- the same peptide appears under many alleles.
    """
    peptides = pd.Index(sorted(candidates["peptide"].astype(str).unique()))
    dist = pd.Series(min_hamming_to(peptides, held_out), index=peptides,
                     name="min_hamming_to_held_out")
    per_row = candidates["peptide"].astype(str).map(dist)

    audit = (pd.DataFrame({"min_hamming_to_held_out": dist.to_numpy()},
                          index=peptides)
             .groupby("min_hamming_to_held_out").size()
             .rename("n_peptides").reset_index())
    audit["n_rows"] = audit["min_hamming_to_held_out"].map(
        per_row.value_counts()).fillna(0).astype(int)
    audit["excluded"] = audit["min_hamming_to_held_out"] <= threshold

    kept = candidates.loc[(per_row > threshold).to_numpy()].copy()
    kept["min_hamming_to_held_out"] = per_row.loc[kept.index].to_numpy()
    return kept.reset_index(drop=True), audit
