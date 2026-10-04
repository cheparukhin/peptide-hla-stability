"""Stage 3c: the atlas is a scoring target, never a fit set.

These tests guard the three things that would silently invalidate the external
validation: a decoy that is really a ligand, an asymmetric filter that the
model could score instead of the biology, and a scoring set that moves between
runs so a second arm is not measured on the same rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import elution as E  # noqa: E402

PSEUDO = {"HLA-A*01:01": "A" * 34, "HLA-B*07:02": "C" * 34,
          "HLA-B*08:01": "A" * 33 + "C"}


def make_atlas(n: int = 200) -> pd.DataFrame:
    """A synthetic atlas: three alleles, disjoint ligand sets, plus noise."""
    rng = np.random.default_rng(0)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    rows = []
    for atlas_allele in ("A0101", "B0702", "B0801"):
        peps = {"".join(rng.choice(aa, 9)) for _ in range(n)}
        rows += [{"Allele": atlas_allele, "Peptide": p} for p in peps]
    rows += [{"Allele": "H2-Kb", "Peptide": "SIINFEKL"},          # non-human
             {"Allele": "A0101", "Peptide": "ACDEFGHIKLM"},       # not a 9-mer
             {"Allele": "A0101", "Peptide": "ACDEFGHIs"}]         # phospho
    atlas = pd.DataFrame(rows)
    return atlas.rename(columns={"Allele": "atlas_allele", "Peptide": "peptide"}
                        ).assign(allele=lambda d: d.atlas_allele.map(
                            E.atlas_allele_to_standard))


def fake_proteome(k: int = 60_000) -> np.ndarray:
    rng = np.random.default_rng(1)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    return np.unique(np.array(
        ["".join(s) for s in rng.choice(aa, (k, 9))], dtype="S9"))


# --- allele parsing --------------------------------------------------------


@pytest.mark.parametrize("raw,want", [
    ("A0201", "HLA-A*02:01"), ("B5701", "HLA-B*57:01"), ("C0702", "HLA-C*07:02"),
    ("A2402", "HLA-A*24:02"), ("B15101", "HLA-B*15:101"),
    ("H2-Kb", None), ("G0101", None), ("E0103", None), ("DRB1_0101", None),
])
def test_allele_parsing(raw, want):
    assert E.atlas_allele_to_standard(raw) == want


def test_c67s_is_stripped_but_not_a_panel_allele():
    s = pd.Series(["HLA-B*14:01(C67S)", "HLA-A*02:01"])
    assert list(E.strip_c67s(s)) == ["HLA-B*14:01", "HLA-A*02:01"]


def test_modified_residues_are_not_standard_9mers():
    s = pd.Series(["ACDEFGHIK", "ACDEFGHIs", "ACDEFGHI", "ACDEFGHIX"])
    assert list(E.is_standard_9mer(s)) == [True, False, False, False]


# --- the decoy protocol ----------------------------------------------------


def test_no_decoy_is_an_atlas_ligand_or_a_measured_peptide():
    atlas = make_atlas()
    measured = {"QQQQQQQQQ", "WWWWWWWWW"}
    prot = np.union1d(fake_proteome(), E.atlas_nonamers(atlas))
    prot = np.union1d(prot, np.array(sorted(measured), dtype="S9"))
    scoring, audit = E.build_scoring_set(
        atlas, measured, PSEUDO, min_ligands=10, max_ligands=50,
        decoy_ratio=3, proteome=prot)

    decoys = set(scoring.loc[scoring.label == 0, "peptide"])
    ligands_any_allele = set(atlas.peptide[atlas.peptide.str.len() == 9])
    assert not decoys & ligands_any_allele
    assert not decoys & measured
    assert audit["decoy_universe_removed"] >= len(ligands_any_allele)


def test_exclusion_is_symmetric_across_both_classes():
    """A measured peptide may not sit in either class.

    If it were filtered out of the decoys only, "absent from the stability CSV"
    would become a property of the negative class and a model could score the
    filter instead of the peptide.
    """
    atlas = make_atlas()
    measured = set(atlas.peptide[atlas.peptide.str.len() == 9].iloc[:40])
    prot = np.union1d(fake_proteome(), np.array(sorted(measured), dtype="S9"))
    scoring, audit = E.build_scoring_set(
        atlas, measured, PSEUDO, min_ligands=10, max_ligands=500,
        decoy_ratio=3, proteome=prot)
    assert not set(scoring.peptide) & measured
    assert audit["ligands_dropped_measured_in_stability_csv"] == len(measured)


def test_ratio_and_lengths_hold_per_allele():
    atlas = make_atlas()
    scoring, _ = E.build_scoring_set(
        atlas, set(), PSEUDO, min_ligands=10, max_ligands=50,
        decoy_ratio=7, proteome=fake_proteome())
    assert (scoring.peptide.str.len() == 9).all()
    for _, g in scoring.groupby("allele"):
        n_lig = int((g.label == 1).sum())
        assert n_lig <= 50
        assert int((g.label == 0).sum()) == 7 * n_lig
        assert g.peptide[g.label == 0].is_unique
        assert not set(g.peptide[g.label == 0]) & set(g.peptide[g.label == 1])


def test_scoring_set_is_reproducible_and_seed_dependent():
    atlas = make_atlas()
    prot = fake_proteome()
    kw = dict(min_ligands=10, max_ligands=50, decoy_ratio=3, proteome=prot)
    a, _ = E.build_scoring_set(atlas, set(), PSEUDO, **kw)
    b, _ = E.build_scoring_set(atlas, set(), PSEUDO, **kw)
    c, _ = E.build_scoring_set(atlas, set(), PSEUDO, seed=E.SEED + 1, **kw)
    pd.testing.assert_frame_equal(a, b)
    assert not a.peptide.equals(c.peptide)


def test_min_ligands_drops_only_small_alleles():
    atlas = make_atlas()
    atlas = atlas[~((atlas.allele == "HLA-B*08:01")
                    & (atlas.groupby("allele").cumcount() >= 5))]
    _, audit = E.build_scoring_set(
        atlas, set(), PSEUDO, min_ligands=10, max_ligands=50,
        decoy_ratio=3, proteome=fake_proteome())
    assert audit["alleles_dropped_too_few_ligands"] == ["HLA-B*08:01"]
    assert audit["alleles_scored"] == 2


def test_pseudoseq_is_the_allele_the_row_belongs_to():
    atlas = make_atlas()
    scoring, _ = E.build_scoring_set(
        atlas, set(), PSEUDO, min_ligands=10, max_ligands=50,
        decoy_ratio=3, proteome=fake_proteome())
    assert (scoring.hla_pseudoseq == scoring.allele.map(PSEUDO)).all()


# --- the allele-swapped control -------------------------------------------


def test_swapped_decoys_are_other_alleles_ligands_only():
    atlas = make_atlas()
    primary, _ = E.build_scoring_set(
        atlas, set(), PSEUDO, min_ligands=10, max_ligands=30,
        decoy_ratio=2, proteome=fake_proteome())
    ligands = E.eligible_atlas_ligands(atlas, set(), PSEUDO, min_ligands=10)
    swapped, audit = E.build_swapped_scoring_set(
        ligands, primary, PSEUDO, decoy_ratio=2)

    for allele, g in swapped.groupby("allele"):
        own = set(ligands.peptide[ligands.allele == allele])
        decoys = set(g.peptide[g.label == 0])
        assert decoys <= set(ligands.peptide)   # every decoy is a real ligand
        assert not decoys & own                 # but never of the target allele
    # the positives are exactly the primary pass's, so only negatives differ
    for label_set in (primary, swapped):
        pass
    assert (swapped[swapped.label == 1].reset_index(drop=True).peptide
            .equals(primary[primary.label == 1].reset_index(drop=True).peptide))
    assert audit["seed"] == E.SEED


def test_donor_distance_is_the_nearest_presenting_allele():
    atlas = make_atlas()
    primary, _ = E.build_scoring_set(
        atlas, set(), PSEUDO, min_ligands=10, max_ligands=30,
        decoy_ratio=2, proteome=fake_proteome())
    ligands = E.eligible_atlas_ligands(atlas, set(), PSEUDO, min_ligands=10)
    swapped, _ = E.build_swapped_scoring_set(ligands, primary, PSEUDO, decoy_ratio=2)
    dec = swapped[swapped.label == 0]
    # A*01:01 and B*08:01 differ at one pseudosequence position, B*07:02 at 34.
    a_decoys = dec[dec.allele == "HLA-A*01:01"]
    presented_by = ligands.groupby("peptide").allele.apply(set)
    for pep, d in zip(a_decoys.peptide, a_decoys.donor_distance):
        want = min(1 if a == "HLA-B*08:01" else 34 for a in presented_by[pep])
        assert d == want


# --- metrics ---------------------------------------------------------------


def test_roc_auc_matches_known_cases():
    assert E.roc_auc(np.array([1, 1, 0, 0]), np.array([4.0, 3, 2, 1])) == 1.0
    assert E.roc_auc(np.array([1, 1, 0, 0]), np.array([1.0, 2, 3, 4])) == 0.0
    # all tied -> no information -> 0.5
    assert E.roc_auc(np.array([1, 1, 0, 0]), np.zeros(4)) == 0.5
    assert E.roc_auc(np.array([1, 0, 1, 0]), np.array([4.0, 3, 2, 1])) == 0.75


def test_roc_auc_agrees_with_sklearn_on_random_data():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    for _ in range(5):
        label = rng.integers(0, 2, 500)
        score = rng.normal(label * 0.8, 1.0)
        assert E.roc_auc(label, score) == pytest.approx(
            sklearn_metrics.roc_auc_score(label, score))
        assert E.average_precision(label, score) == pytest.approx(
            sklearn_metrics.average_precision_score(label, score))


def test_auprc_of_a_random_ranking_approaches_the_positive_rate():
    """Chance AUPRC is the positive rate, 1/(1 + decoy ratio) here.

    It converges from above: with few positives the first hit lands early
    often enough to lift the mean, which is why the reported baseline is the
    positive rate itself and not a simulated one. At the panel's smallest
    allele (101 positives against 1,010 decoys) the bias is already under a
    point.
    """
    label = np.array([1] * 100 + [0] * 1000)
    rng = np.random.default_rng(4)
    vals = [E.average_precision(label, rng.normal(size=1100)) for _ in range(100)]
    assert np.mean(vals) == pytest.approx(1 / 11, abs=0.01)
    assert E.score_allele("A", label, rng.normal(size=1100)).auprc_baseline == \
        pytest.approx(1 / 11)


def test_enrichment_of_a_perfect_ranking_is_the_ceiling():
    label = np.array([1] * 10 + [0] * 90)
    perfect = -np.arange(100.0)
    precision, ef = E.enrichment_at(label, perfect, 0.10)
    assert precision == 1.0
    assert ef == pytest.approx(10.0)


def test_score_panel_rejects_non_finite_predictions():
    scoring = pd.DataFrame({"allele": ["A"] * 4, "peptide": list("abcd"),
                            "label": [1, 1, 0, 0]})
    with pytest.raises(ValueError):
        E.score_panel(scoring, np.array([1.0, np.nan, 0.0, 0.0]))


def test_derangement_moves_every_allele():
    items = [f"HLA-A*{i:02d}:01" for i in range(1, 12)]
    swap = E.derangement(items)
    assert set(swap) == set(items) and set(swap.values()) == set(items)
    assert all(k != v for k, v in swap.items())
