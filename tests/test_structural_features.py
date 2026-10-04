"""Stage 4c.5 feature extraction: mapping verification, exclusion, joins."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pepstab import structural_features as sf

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "structures/ectodomain_pilot/ectodomain-20261004"
ARM_B = PILOT / "boltz2/seed_0/A0201_LLWNGPMAV_arm_B"
ARM_A = PILOT / "boltz2/seed_0/A0201_LLWNGPMAV_arm_A"

pytestmark = pytest.mark.skipif(not PILOT.exists(), reason="pilot structures not on disk")


@pytest.fixture(scope="module")
def fold_b():
    return sf.load_fold(ARM_B)


# --- Discovery and the production sibling trees -----------------------------


def _fake_tree(tmp_path: Path) -> Path:
    """A production-shaped tree: cohort output plus the harness siblings."""
    root = tmp_path / "stage4c/ectodomain-20261004/boltz2"
    layout = [
        "production/HLA-A_02_01/A0201_LLWNGPMAV",
        "production/HLA-A_02_01/A0201_SLLMWITQV",
        "production/HLA-B_07_02/B0702_IPRRNVATL",
        "_smoke/HLA-A_02_01/A0201_SMOKE0001",
        "_smoke/HLA-B_07_02/B0702_SMOKE0002",
        "_shards/shard_000/HLA-A_02_01/A0201_SHARDED",
        "_failed/HLA-A_02_01/A0201_BROKEN",
    ]
    for rel in layout:
        d = root / rel
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"model": "boltz2", "seed": 0}))
    return root


def test_discovery_skips_smoke_shards_and_failed(tmp_path):
    root = _fake_tree(tmp_path)
    found = sorted(p.name for p in sf.discover_folds(root))
    assert found == ["A0201_LLWNGPMAV", "A0201_SLLMWITQV", "B0702_IPRRNVATL"]
    # And nothing from the excluded trees slipped through under any name.
    assert not any(
        part in sf.EXCLUDED_DIR_NAMES
        for p in sf.discover_folds(root)
        for part in p.relative_to(root).parts
    )


def test_discovery_finds_seven_without_the_exclusion(tmp_path):
    """Guards the test itself: the fake tree really does contain the traps."""
    root = _fake_tree(tmp_path)
    assert len(list(root.rglob("metadata.json"))) == 7


@pytest.mark.parametrize("name", sorted(sf.EXCLUDED_DIR_NAMES))
def test_is_excluded_at_any_depth(tmp_path, name):
    root = tmp_path / "r"
    deep = root / "boltz2" / name / "a" / "b" / "c"
    assert sf.is_excluded(deep, root)
    assert not sf.is_excluded(root / "boltz2" / "production" / "a" / "b", root)


def test_excluded_names_are_the_documented_three():
    assert sf.EXCLUDED_DIR_NAMES == frozenset({"_smoke", "_shards", "_failed"})


# --- The verified mapping ---------------------------------------------------


def test_chain_layout_is_derived_not_assumed(fold_b):
    assert [(c.chain_id, c.length) for c in fold_b.chains] == [("A", 275), ("B", 99), ("C", 9)]
    assert fold_b.n_tokens == 383
    assert fold_b.pae.shape == (383, 383)
    assert (fold_b.groove.start, fold_b.groove.stop) == (0, 182)
    assert (fold_b.peptide.start, fold_b.peptide.stop) == (374, 383)
    assert (fold_b.beta2m.start, fold_b.beta2m.stop) == (275, 374)
    assert (fold_b.alpha3.start, fold_b.alpha3.stop) == (182, 275)


def test_plddt_token_order_matches_cif_ca_order(fold_b):
    ca = fold_b.atoms[fold_b.atoms.atom_name == "CA"]
    assert np.abs(ca.b_factor - fold_b.plddt * 100).max() <= sf.BFACTOR_TOLERANCE


def test_pae_is_not_symmetric(fold_b):
    """Both directions are kept because they are genuinely different numbers."""
    assert not np.allclose(fold_b.pae, fold_b.pae.T)


def test_wrong_pae_shape_fails_loudly(fold_b):
    with pytest.raises(sf.MappingError, match="pae"):
        sf.verify_fold(
            fold_b.folder,
            fold_b.metadata,
            fold_b.atoms,
            np.zeros((382, 382)),
            fold_b.plddt,
            fold_b.confidence,
        )


def test_shuffled_plddt_fails_loudly(fold_b):
    bad = fold_b.plddt.copy()
    bad[0], bad[-1] = bad[-1], bad[0]
    with pytest.raises(sf.MappingError, match="token order"):
        sf.verify_fold(
            fold_b.folder, fold_b.metadata, fold_b.atoms, fold_b.pae, bad, fold_b.confidence
        )


def test_declared_chains_must_match_the_cif(fold_b):
    meta = json.loads(json.dumps(fold_b.metadata))
    meta["inputs"]["chains"][1]["sequence"] = meta["inputs"]["chains"][1]["sequence"][:-1]
    with pytest.raises(sf.MappingError, match="do not match the declared"):
        sf.verify_fold(
            fold_b.folder, meta, fold_b.atoms, fold_b.pae, fold_b.plddt, fold_b.confidence
        )


def test_non_finite_arrays_fail_loudly(fold_b):
    bad = fold_b.pae.copy()
    bad[0, 0] = np.nan
    with pytest.raises(sf.MappingError, match="non-finite"):
        sf.verify_fold(
            fold_b.folder, fold_b.metadata, fold_b.atoms, bad, fold_b.plddt, fold_b.confidence
        )


def test_two_chain_arm_is_handled_without_beta2m():
    fold = sf.load_fold(ARM_A)
    assert fold.beta2m is None and fold.alpha3 is None
    assert fold.n_tokens == 191
    row = sf.extract_features(fold)
    assert row["pae_pep_rows_b2m_cols_mean"] is None
    assert row["beta2m_plddt_mean"] is None
    assert row["pae_pep_rows_groove_cols_mean"] > 0


# --- Feature semantics ------------------------------------------------------


def test_core_features_reproduce_the_pose_checker(fold_b):
    """The existing scorer computed these independently; they must agree."""
    pose = json.loads((ROOT / "reports/ectodomain-20261004/pilot_pose_scores.json").read_text())
    ref = next(
        r
        for r in pose
        if r.get("model") == "boltz2"
        and r.get("seed") == 0
        and r.get("arm") == "B"
        and r.get("complex_id") == "A0201_LLWNGPMAV"
    )
    row = sf.extract_features(fold_b)
    assert row["pae_pep_rows_groove_cols_mean"] == pytest.approx(
        ref["peptide_to_groove_pae_mean"], abs=1e-5
    )
    assert row["pae_groove_rows_pep_cols_mean"] == pytest.approx(
        ref["groove_to_peptide_pae_mean"], abs=1e-5
    )
    assert [row[f"pep_plddt_p{i + 1}"] for i in range(9)] == pytest.approx(ref["peptide_plddt"])
    assert [row[f"groove_contacts_4p5A_p{i + 1}"] for i in range(9)] == ref["groove_contacts_4p5A"]


def test_global_iptm_is_named_global_and_pair_scores_name_their_direction(fold_b):
    row = sf.extract_features(fold_b)
    assert "global_iptm" in row and "iptm" not in row
    assert row["global_iptm"] == pytest.approx(fold_b.confidence["iptm"])
    # No column may claim to be interface confidence without saying which pair.
    for key in row:
        if "iptm" in key and not key.startswith("global_"):
            assert "_rows_" in key and "_cols" in key, key


def test_pair_iptm_orientation_follows_the_pae_convention(fold_b):
    """``pair_chains_iptm[a][b]`` scores PAE rows in b against columns in a."""
    row = sf.extract_features(fold_b)
    raw = fold_b.confidence["pair_chains_iptm"]
    idx = {c.chain_id: i for i, c in enumerate(fold_b.chains)}
    hla, pep = str(idx[fold_b.hla.chain_id]), str(idx[fold_b.peptide.chain_id])
    assert row["iptm_pair_peptide_rows_hla_chain_cols"] == pytest.approx(raw[hla][pep])
    assert row["iptm_pair_hla_chain_rows_peptide_cols"] == pytest.approx(raw[pep][hla])
    assert row["iptm_pair_peptide_rows_hla_chain_cols"] != row["iptm_pair_hla_chain_rows_peptide_cols"]


def test_pair_iptm_is_withheld_for_unverified_models():
    fold = sf.load_fold(PILOT / "esmfold2/seed_0/A0201_LLWNGPMAV_arm_B")
    row = sf.extract_features(fold)
    assert "esmfold2" not in sf.PAIR_IPTM_VERIFIED_MODELS
    assert row["iptm_pair_peptide_rows_hla_chain_cols"] is None


def test_burial_is_between_zero_and_one_and_contacts_exist(fold_b):
    row = sf.extract_features(fold_b)
    assert 0.0 < row["pep_buried_sasa_frac"] <= 1.0
    assert row["pep_buried_sasa_A2"] > 0
    assert row["groove_contacts_4p5A_total"] > 0
    assert row["pep_min_dist_to_groove_A"] < sf.CONTACT_CUTOFF_A


def test_every_feature_value_is_finite_or_none(fold_b):
    for key, value in sf.extract_features(fold_b).items():
        if isinstance(value, float):
            assert np.isfinite(value), key


def test_both_metadata_schemas_are_read(fold_b):
    """The production runner records ``seed`` under ``settings`` and ``shard``
    at the top level; the pilot does the opposite. The schemas look alike and
    differ enough to raise ``KeyError`` -- the first real production fold did
    exactly that, so both are pinned here."""
    pilot = json.loads(json.dumps(fold_b.metadata))
    assert "seed" in pilot and "seed" not in pilot.get("settings", {})
    assert sf._seed(pilot) == 0

    production = json.loads(json.dumps(pilot))
    production.pop("seed")
    production["settings"]["seed"] = 0
    production["shard"] = 7
    production["is_shard_first"] = False
    production["settings"]["actual_prediction_order"] = [
        production["inputs"]["complex_id"]
    ]
    assert sf._seed(production) == 0

    fold = sf.verify_fold(
        fold_b.folder, production, fold_b.atoms, fold_b.pae, fold_b.plddt, fold_b.confidence
    )
    row = sf.extract_features(fold)
    assert row["seed"] == 0
    assert row["shard"] == 7
    assert row["is_shard_first"] is False
    assert row["prediction_order_index"] == 0


def test_a_seed_is_never_invented(fold_b):
    meta = json.loads(json.dumps(fold_b.metadata))
    meta.pop("seed")
    meta["settings"].pop("seed", None)
    with pytest.raises(sf.MappingError, match="no seed"):
        sf._seed(meta)


def test_provenance_travels_with_the_features(fold_b):
    row = sf.extract_features(fold_b)
    assert row["boltz_package"] == "boltz==2.1.1"
    assert row["weight_revision"] == "6fdef46d763fee7fbb83ca5501ccceff43b85607"
    assert len(row["cif_sha256"]) == 64
    assert len(row["msa_csv_sha256_hla_chain"]) == 64
    assert row["msa_depth_hla_chain"] == 1024
    # The authoritative pair_id comes from the (allele, peptide) join, so the
    # one in metadata.json is carried under a name that cannot be mistaken.
    assert "metadata_pair_id" in row


def test_failures_become_rows_not_crashes(tmp_path):
    d = tmp_path / "A0201_BROKEN"
    d.mkdir()
    (d / "metadata.json").write_text(
        json.dumps({"model": "boltz2", "seed": 0,
                    "inputs": {"arm": "B", "allele": "HLA-A*02:01", "peptide": "LLWNGPMAV",
                               "complex_id": "A0201_LLWNGPMAV"}})
    )
    row = sf.extract_path(d)
    assert row["status"] != "ok"
    assert row["allele"] == "HLA-A*02:01"


def test_a_low_confidence_fold_is_still_extracted(fold_b):
    """Confidence ranks error but does not isolate the pilot's one genuine
    failure, so it is a feature and never a per-prediction row filter."""
    bad_plddt = np.full_like(fold_b.plddt, 0.2)
    ca = fold_b.atoms[fold_b.atoms.atom_name == "CA"]
    atoms = fold_b.atoms.copy()
    atoms.b_factor[atoms.atom_name == "CA"] = bad_plddt * 100
    conf = dict(fold_b.confidence, iptm=0.05, ptm=0.05, confidence_score=0.05)
    fold = sf.verify_fold(
        fold_b.folder, fold_b.metadata, atoms, np.full_like(fold_b.pae, 25.0), bad_plddt, conf
    )
    row = sf.extract_features(fold)
    assert row["status"] == "ok"
    assert row["pep_plddt_mean"] == pytest.approx(0.2)
    assert row["global_iptm"] == pytest.approx(0.05)
    assert ca is not None


# --- Joins ------------------------------------------------------------------


def test_pair_ids_are_attached_by_allele_peptide_not_positionally():
    from pepstab.data import load_with_splits

    splits = load_with_splits()
    frame = pd.DataFrame(
        [
            {"model": "boltz2", "allele": "HLA-A*02:01", "peptide": "LLWNGPMAV", "arm": "B", "seed": 0},
            {"model": "boltz2", "allele": "HLA-B*07:02", "peptide": "IPRRNVATL", "arm": "B", "seed": 0},
        ]
    )
    joined = sf.attach_pair_ids(frame, splits)
    assert joined["pair_id"].notna().all()
    for _, r in joined.iterrows():
        ref = splits[(splits.allele == r["allele"]) & (splits.peptide == r["peptide"])].iloc[0]
        assert r["pair_id"] == ref["pair_id"]
        assert r["split"] == ref["split"]


def test_alpha3_provenance_flags_exactly_the_three_borrowed_alleles():
    """``a3_source`` keeps the ``HLA-`` prefix only on the borrowed rows, so a
    naive ``a3_source != allele`` flags all 75. Both sides must be normalised."""
    prov = sf.alpha3_provenance()
    assert len(prov) == 75
    assert prov["alpha3_provenance"].value_counts().to_dict() == {
        "exact": 69,
        "borrowed_relative": 3,
        "wildtype_of_engineered": 3,
    }
    borrowed = prov[prov["alpha3_borrowed"]].set_index("allele")["alpha3_source_allele"]
    assert borrowed.to_dict() == {
        "HLA-A*02:50": "A*02:01",
        "HLA-A*24:19": "A*24:07",
        "HLA-B*08:03": "B*08:01",
    }
    # The engineered C67S alleles take alpha3 from the wild type of the *same*
    # allele, which is a different situation and keeps its own label.
    assert not prov.loc[prov["allele"].str.contains(r"\("), "alpha3_borrowed"].any()


def test_provenance_covers_every_cohort_allele():
    cohort = pd.read_csv(ROOT / "data/structural_cohort.csv")
    merged = cohort.merge(sf.alpha3_provenance(), on="allele", how="left")
    assert merged["alpha3_provenance"].notna().all()
    assert int(merged["alpha3_borrowed"].sum()) == 1103


def test_cohort_is_loaded_not_recomputed():
    cohort = pd.read_csv(ROOT / "data/structural_cohort.csv")
    assert len(cohort) == 28166
    assert cohort.groupby("profile").size().to_dict() == {
        "a-cheparukhin": 14083,
        "colleague": 14083,
    }
    # Disjoint halves: a pair belongs to exactly one profile.
    assert not cohort.duplicated(["allele", "peptide"]).any()


# --- Stage 5 structural arm: parity and gating -----------------------------


def _arm():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stage5_structural_arm", ROOT / "scripts/stage5_structural_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ensemble_size_matches_the_sequence_comparator():
    """5 folds x 2 encodings x 3 seeds = 30 networks for the comparator, so the
    structural arm fits 30 too. Ensembling alone is worth ~0.090 median SCC
    here, so a smaller ensemble would read as a feature verdict."""
    arm = _arm()
    assert arm.N_FOLDS * len(arm.SEEDS) == arm.PARITY_NETWORKS == 30
    from scripts.baseline_sequence import HIDDEN_GRID, L2_GRID, SEEDS as SEQ_SEEDS
    assert arm.N_FOLDS * len(HIDDEN_GRID) * len(SEQ_SEEDS) == arm.PARITY_NETWORKS


def test_tuning_budget_matches_the_comparator():
    """Equal budget, scale-appropriate values -- not transplanted values."""
    arm = _arm()
    from scripts.baseline_sequence import ALPHA_GRID, HIDDEN_GRID, L2_GRID

    assert len(arm.STRUCT_ALPHA_GRID) == len(ALPHA_GRID)
    assert len(arm.STRUCT_L2_GRID) == len(HIDDEN_GRID) * len(L2_GRID)
    # Standardised dense features need far stronger shrinkage than stage 2's
    # sparse one-hot columns; transplanting its values would handicap the arm.
    # The ladder starts at the comparator's ceiling and extends three decades
    # above it, so it covers strictly stronger shrinkage.
    assert min(arm.STRUCT_L2_GRID) >= max(L2_GRID)
    assert max(arm.STRUCT_L2_GRID) > max(L2_GRID) * 100
    # An interior check is unsatisfiable on a two-point ladder.
    assert len(arm.STRUCT_L2_GRID) >= 3


def test_check_interior_refuses_an_edge_selection():
    arm = _arm()
    ladder = (1e-3, 1e-2, 1e-1, 1.0)
    arm.check_interior("x", {"g": (1e-2, ladder, "l2")})  # interior: fine
    for edge in (min(ladder), max(ladder)):
        with pytest.raises(SystemExit, match="edge of ladder"):
            arm.check_interior("x", {"g": (edge, ladder, "l2")})


def test_unclassified_feature_is_an_error_not_a_silent_drop():
    arm = _arm()
    frame = pd.DataFrame({"groove_contacts_4p5A_total": [1.0],
                          "pep_plddt_mean": [0.9],
                          "some_new_feature": [2.0]})
    with pytest.raises(SystemExit, match="unclassified numeric columns"):
        arm.feature_columns(frame)


def test_groups_are_disjoint_and_confidence_is_not_mixed_into_geometry():
    arm = _arm()
    frame = pd.read_csv(ROOT / "reports/stage4c5_pilot_features.csv")
    groups = arm.feature_columns(frame)
    assert set(groups["geometry"]).isdisjoint(groups["confidence"])
    assert groups["geometry+confidence"] == groups["geometry"] + groups["confidence"]
    for col in groups["geometry"]:
        assert not any(t in col for t in ("plddt", "pae", "iptm", "ptm", "pde")), col
    assert any("iptm" in c for c in groups["confidence"])


def test_prediction_path_is_the_frozen_name():
    """The submission figure auto-draws arms by this stem."""
    arm = _arm()
    assert arm.PRED_PATH == ROOT / "preds/boltz_structural.csv"


def test_only_validation_can_be_written():
    """The test split is scored once, at stage 6, by the orchestrator."""
    import argparse

    arm = _arm()
    ns = argparse.Namespace(mode="ensemble", table=ROOT / "nope.csv",
                            split="test", select="geometry", write_preds=False)
    with pytest.raises(SystemExit, match="validation predictions only"):
        arm.mode_ensemble(ns)
