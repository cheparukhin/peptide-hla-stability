"""Guards on the ESM component of Phase A arm A5 (`scripts/phase_a_esm_oof.py`).

Four kinds of check:

- **Split discipline** -- the script must load the training split and nothing
  else. Phase A scores neither frozen split, and reading one here is the single
  unrecoverable mistake available to this arm, so it is asserted on the source
  rather than left to review.
- **Arity parity with A0** -- the axis product must reach A0's 30 members per
  outer fold. Ensembling is worth roughly +0.07 to +0.09 median Spearman here,
  so an arity mismatch between A5's two blend components would land as the
  blend's main effect rather than as a feature effect.
- **Selection provenance** -- the ESM configuration must be stage 3's recorded
  selection, not a re-tuning. A truncated L2 ladder already cost this arm 0.093
  median Spearman once; re-opening it inside a blend would also add a second
  difference between the blend's components.
- **Contract shape** -- the member loop mirrors
  ``pepstab.nested.fit_mlp_ensemble`` (which cannot be called, because its
  ``input_set`` routes to one-hot/BLOSUM features), so the two arms' member
  records, ``inner_oof`` accumulation and hold-out counts must agree.

The one fitting test runs 2 inner folds x 2 layers x 1 seed on 1,200 rows with a
3-epoch cap and an 8-component peptide PCA. Everything else is metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import nested  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.features import ENCODINGS  # noqa: E402
from pepstab.mlp import MLPConfig  # noqa: E402
from modal_app import phase_a_esm as esm_app  # noqa: E402
from scripts import phase_a_esm_oof as arm  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402
from scripts.esm_arm import assemble  # noqa: E402

SCRIPT = Path(arm.__file__)


@pytest.fixture(scope="module")
def train():
    df = load_with_splits()
    return df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)


# --- split discipline -----------------------------------------------------

def test_script_selects_only_the_training_split():
    """Phase A scores neither frozen split; this arm must not load one.

    ``load_with_splits`` returns all three, so the filter is the whole
    safeguard. Asserted on the source because a single stray ``== "val"`` would
    spend a development resource that stages 2-7 have already consulted
    repeatedly, and the test split is spent outright.
    """
    source = SCRIPT.read_text()
    assert 'df.split == "train"' in source
    for forbidden in ('== "val"', "== 'val'", '== "test"', "== 'test'",
                      'split == "val"', 'split == "test"'):
        assert forbidden not in source, (
            f"{SCRIPT.name} selects {forbidden}; Phase A scores neither split")


def test_script_does_not_import_the_peptide_fold_scheme():
    """``stage6.nested_folds`` is the wrong partition for this arm.

    It folds by *peptide* and deliberately places near-neighbour mutants in
    different folds, which is why ``preds/esm_ensemble_oof.csv`` cannot be
    reused. Importing it here would silently reintroduce exactly that.
    """
    source = SCRIPT.read_text()
    assert "from pepstab.stage6" not in source
    assert "import stage6" not in source
    assert "nested_folds(" not in source
    assert "from pepstab import nested" in source


def test_fold_structure_is_inherited_not_recomputed():
    """The fold primitives come from ``pepstab.nested``, not a local copy."""
    source = SCRIPT.read_text()
    assert "nested.outer_folds(" in source
    assert "nested.inner_folds(" in source
    assert "assign_clusters" not in source, (
        "the water-fill must be reached through pepstab.nested, so the outer "
        "and inner levels cannot drift from A0's")


# --- arity parity with A0 -------------------------------------------------

def test_axis_product_reaches_a0_arity():
    """5 inner folds x 2 ESM layers x 3 seeds = 30, exactly A0's member count.

    A0 multiplies 5 inner folds x 2 encodings x 3 seeds. The layer axis is the
    two-way axis stage 3 declared as the stand-in for the encoding axis, so the
    products match term by term, not only in total.
    """
    esm = nested.ASSESS_INNER_FOLDS * len(arm.LAYERS) * len(SEEDS)
    a0 = nested.ASSESS_INNER_FOLDS * len(ENCODINGS) * len(SEEDS)
    assert esm == a0 == 30
    assert len(arm.LAYERS) == len(ENCODINGS) == 2
    assert arm.LAYERS == ("mid", "final")


def test_row_count_identities_are_the_a0_contract():
    """1 outer_held + 4 inner_oof_6 + 4 inner_oof_30 per training row."""
    assert arm.N_TRAIN_ROWS == 19_716
    assert arm.N_TRAIN_ROWS * 9 == 177_444


def test_sentinels_match_a0(train):
    """A0 and this arm must mark 'no sub-fold' and 'outer-held' the same way."""
    assert arm.NO_SUB_FOLD == -1
    assert arm.OUTER_HELD_INNER_FOLD == -1
    assert arm.SENSITIVITY_MIN_ROWS == 50
    assert nested.MIN_ROWS_PER_ALLELE_FOLD == 20
    assert len(train) == arm.N_TRAIN_ROWS


# --- selection provenance -------------------------------------------------

def test_selected_config_is_stage_3s_recorded_selection():
    """The configuration is read back from the artifact that recorded it.

    ``reconcile_selection`` raises rather than returns on any drift between
    ``scripts/esm_arm.py``'s constants and ``reports/stage3_headline.json``, so
    calling it is the check.
    """
    config = arm.reconcile_selection()
    headline = json.loads(arm.STAGE3_HEADLINE.read_text())
    assert config["checkpoint"] == headline["checkpoint_selected"]
    assert config["checkpoint"] == "esm2_t12_35M_UR50D"
    assert (config["pep_rep"], config["hla_rep"]) == ("pos", "contact")
    assert (config["pep_pca"], config["hla_pca"]) == (256, 74)
    assert config["hidden"] == [256, 64]
    assert config["l2"] == 1e-2
    assert config["layers"] == ["mid", "final"]


def test_selection_is_not_re_tuned_here():
    """No L2 or hidden-size ladder appears in this script.

    Stage 3's ladder selection is interior and reported; A5 blends two arms at
    their own selected configurations, and re-tuning one component would add a
    second difference between them.
    """
    source = SCRIPT.read_text()
    for ladder in ("ESM_L2_GRID", "L2_GRID", "HIDDEN_GRID", "ALPHA_GRID"):
        assert ladder not in source


def test_reconcile_selection_rejects_a_drifted_headline(tmp_path, monkeypatch):
    """The reconciliation must fail loudly, not fall back to a default."""
    headline = json.loads(arm.STAGE3_HEADLINE.read_text())
    headline["head"]["l2"] = 0.5
    drifted = tmp_path / "stage3_headline.json"
    drifted.write_text(json.dumps(headline))
    monkeypatch.setattr(arm, "STAGE3_HEADLINE", drifted)
    with pytest.raises(AssertionError, match="SWEEP_CONFIG"):
        arm.reconcile_selection()


def test_member_config_inherits_the_shared_reference_member():
    """Only hidden and l2 differ from ``nested.REFERENCE_MLP``.

    lr, batch size, the 300-epoch cap and the 15-epoch patience are the shared
    values stage 3's head also used, so the ESM and sequence members differ in
    their features and their regularisation, not in their optimiser.
    """
    assert (nested.REFERENCE_MLP.max_epochs, nested.REFERENCE_MLP.patience) == (
        MAX_EPOCHS, PATIENCE)
    headline = json.loads(arm.STAGE3_HEADLINE.read_text())["head"]
    assert headline["lr"] == nested.REFERENCE_MLP.lr
    assert headline["max_epochs"] == nested.REFERENCE_MLP.max_epochs
    assert headline["patience"] == nested.REFERENCE_MLP.patience


# --- embedding cache ------------------------------------------------------

def test_cache_digests_cover_every_file_the_arm_reads():
    """Both layers, both sequence kinds, plus the index content hashes.

    A stale cache keyed to the right sequence set would otherwise pass
    unnoticed, so the embeddings are digested as well as the index.
    """
    config = arm.reconcile_selection()
    digests = arm.cache_digests(config["checkpoint"])
    layer_ids = {"mid": 6, "final": 12}
    for kind in ("peptide", "hla"):
        assert f"{kind}.index.json::content_hash" in digests
        for layer, ident in layer_ids.items():
            key = f"features/esm/{config['checkpoint']}/{kind}_L{ident}.npz"
            assert key in digests, f"{key} not digested"
            assert len(digests[key]) == 64
    assert len(digests) == 2 * (1 + 1 + len(arm.LAYERS))


def test_no_embedding_extraction_happens_here():
    """The cache is on disk; re-extracting it would be wasted spend."""
    source = SCRIPT.read_text()
    for forbidden in ("pesm.extract", "build_cache", "import torch", "esm_features"):
        assert forbidden not in source


# --- outer fold agreement with A0 -----------------------------------------

def test_outer_folds_equal_a0s_published_file(train):
    """The file A5 joins on, compared rather than recomputed into agreement."""
    path = arm.OUT_DIR / "phase_a_outer_folds.csv"
    if not path.exists():
        pytest.skip("A0 has not written phase_a_outer_folds.csv yet")
    published = pd.read_csv(path)
    np.testing.assert_array_equal(published.pair_id.to_numpy(),
                                  train.pair_id.to_numpy())
    np.testing.assert_array_equal(published.outer_fold.to_numpy(),
                                  nested.outer_folds(train))


# --- contract shape -------------------------------------------------------

def test_assert_no_overlap_catches_a_fitted_row_being_predicted():
    arm.assert_no_overlap("clean", np.array([1, 2, 3]), np.array([4, 5]))
    with pytest.raises(AssertionError, match="both fitted and predicted"):
        arm.assert_no_overlap("dirty", np.array([1, 2, 3]), np.array([3, 4]))


def test_fit_esm_ensemble_arity_and_holdout(train):
    """Mirrors ``test_fit_mlp_ensemble_arity_and_holdout`` for the ESM axis.

    Every fit row must be held out by exactly ``len(layers) * len(seeds)``
    members, the member records must carry the same keys A0's runs table does
    (with ``block`` where A0 has ``encoding``), and ``inner_oof`` must be
    finite everywhere. Small on purpose: 1,200 rows, 3 epochs, an 8-component
    peptide PCA.
    """
    split = next(iter(nested.iter_outer(train, 2)))
    fit_rows = train[split.fit_mask].reset_index(drop=True).head(1200)
    inner = nested.inner_folds(fit_rows, 2)
    held = train[split.held_mask].reset_index(drop=True).head(300)

    config = {**arm.reconcile_selection(), "hidden": [16], "pep_pca": 8}
    result = arm.fit_esm_ensemble(
        fit_rows, [held], inner, arm.LAYERS, (0,), config,
        base_config=MLPConfig(max_epochs=3, patience=2))

    assert result["n_members"] == 4            # 2 inner folds x 2 layers x 1 seed
    assert len(result["predictions"]) == 1
    assert result["predictions"][0].shape == (len(held),)
    assert result["inner_oof"].shape == (len(fit_rows),)
    assert np.isfinite(result["inner_oof"]).all()
    assert np.isfinite(result["predictions"][0]).all()
    assert {m["block"] for m in result["members"]} == set(arm.LAYERS)
    assert {m["inner_fold"] for m in result["members"]} == {0, 1}
    assert all(set(m) == {"inner_fold", "block", "seed", "n_fit_rows",
                          "best_epoch", "dev_mse", "fit_seconds"}
               for m in result["members"])
    # 8 peptide components + the HLA block's own rank-limited reduction.
    assert result["n_features"] > 8


def test_fit_esm_ensemble_rejects_an_incomplete_inner_cover(train):
    """A fit row held out by the wrong number of members must fail loudly.

    ``inner_oof`` divides by the declared arity, so a row that landed in no
    inner fold would come back as a 0/`per_inner` mean rather than as an error
    -- and A5 fits its blend weights on exactly that column.
    """
    split = next(iter(nested.iter_outer(train, 2)))
    fit_rows = train[split.fit_mask].reset_index(drop=True).head(600)
    inner = np.array(nested.inner_folds(fit_rows, 2), copy=True)
    inner[:10] = -1                      # ten rows in no inner fold at all
    held = train[split.held_mask].reset_index(drop=True).head(100)

    config = {**arm.reconcile_selection(), "hidden": [8], "pep_pca": 4}
    with pytest.raises(AssertionError, match="held out by exactly"):
        arm.fit_esm_ensemble(fit_rows, [held], inner, ("mid",), (0,), config,
                             base_config=MLPConfig(max_epochs=2, patience=1))


# --- the Modal payload ----------------------------------------------------

@pytest.fixture(scope="module")
def payload(train):
    return arm.build_payload(train, arm.reconcile_selection())


def test_payload_holds_training_sequences_only(train, payload):
    """A container cannot be handed a frozen-split embedding.

    The one unrecoverable mistake available to this arm is reading val or
    test. Subsetting the artifact to the training sequences makes that
    impossible in the cloud rather than merely avoided: 3,942 of the cache's
    5,633 peptides.
    """
    full = load_with_splits()
    assert payload["n_sequences"]["peptide"] == train.peptide.nunique() == 3942
    assert payload["n_sequences"]["hla"] == train.hla_seq.nunique() == 75
    held_out = set(full[full.split != "train"].peptide.astype(str)) - set(
        train.peptide.astype(str))
    assert held_out, "no peptide is exclusive to val/test; the check is vacuous"
    assert not held_out & set(payload["index"]["peptide"]), (
        "a val/test-only peptide is in the payload")


def test_payload_hla_block_is_pre_sliced_to_the_contact_positions(payload):
    """34 of 182 residues, which is what ``hla_rep='contact'`` selects anyway.

    Recorded in the payload so a reader cannot slice it a second time and
    silently take 34 positions out of 34.
    """
    assert payload["hla_rep_applied"] == "contact"
    for layer in arm.LAYERS:
        block = payload["blocks"][f"hla_{layer}"]
        assert block.shape[1] == len(arm.pesm.CONTACT_POSITIONS_1BASED) == 34
        assert payload["blocks"][f"peptide_{layer}"].shape[1:] == (9, 480)


def test_payload_size_is_measured_not_assumed(payload):
    """146 MB, not the 15 MB a mean-pooled payload would be.

    The selected peptide representation is per-position (9 x 480) and the PCA
    above it is refitted per fitting partition, so the raw 4,320 columns have
    to travel. This is the number the transport decision rests on, so it is
    asserted rather than quoted.
    """
    mb = payload["bytes"] / 1e6
    assert 140 < mb < 155, mb
    mean_pooled = 3942 * 480 * 4 * len(arm.LAYERS) / 1e6
    assert mean_pooled < 20 < mb


@pytest.mark.parametrize("layer", ["mid", "final"])
def test_payload_assembly_is_bit_identical_to_the_cache_path(train, payload, layer):
    """The cloud path and the local pilot must not drift.

    ``build_payload`` keeps the cache's row order precisely so the
    standardisation sums in the same order; ``randomized_svd``'s projection is
    drawn over columns, which are unchanged. Together those make the 3,942-row
    payload give the *same* transform as the 5,633-row cache, not a close one --
    so this asserts exact equality rather than a tolerance.
    """
    split = next(iter(nested.iter_outer(train, nested.ASSESS_INNER_FOLDS)))
    fit_rows = train[split.fit_mask].reset_index(drop=True).head(2000)
    held = train[split.held_mask].reset_index(drop=True).head(400)
    config = arm.reconcile_selection()

    X_a, (P_a,), info_a = assemble(fit_rows, [held], config["checkpoint"], layer,
                                   config["pep_rep"], config["hla_rep"], None,
                                   pep_pca=config["pep_pca"])
    X_b, (P_b,), info_b = arm.assemble_from_payload(fit_rows, [held], payload,
                                                    layer, config)
    assert info_a["n_features"] == info_b["n_features"]
    assert info_a["peptide_explained"] == info_b["peptide_explained"]
    assert info_a["hla_explained"] == info_b["hla_explained"]
    np.testing.assert_array_equal(X_a, X_b)
    np.testing.assert_array_equal(P_a, P_b)


def test_payload_round_trips_through_disk(tmp_path, payload):
    written = arm.write_payload(tmp_path / "esm_payload.npz", payload)
    back = arm.read_payload(tmp_path / "esm_payload.npz")
    assert back["digest"] == payload["digest"]
    assert back["checkpoint"] == payload["checkpoint"]
    assert back["index"] == payload["index"]
    for name, block in payload["blocks"].items():
        np.testing.assert_array_equal(back["blocks"][name], block)
    assert written["disk_bytes"] > 0


def test_read_payload_refuses_a_foreign_schema(tmp_path, payload, monkeypatch):
    """A container must not read an artifact whose shape its reader predates."""
    arm.write_payload(tmp_path / "p.npz", payload)
    # Patched on the app module, which is where the reader reads it: the
    # constant has one definition and the script re-exports it.
    monkeypatch.setattr(esm_app, "PAYLOAD_SCHEMA_VERSION",
                        esm_app.PAYLOAD_SCHEMA_VERSION + 1)
    with pytest.raises(AssertionError, match="payload schema"):
        arm.read_payload(tmp_path / "p.npz")


def test_fit_esm_ensemble_accepts_a_payload(train, payload):
    """The same member loop serves both transports."""
    split = next(iter(nested.iter_outer(train, 2)))
    fit_rows = train[split.fit_mask].reset_index(drop=True).head(900)
    inner = nested.inner_folds(fit_rows, 2)
    held = train[split.held_mask].reset_index(drop=True).head(200)
    config = {**arm.reconcile_selection(), "hidden": [16], "pep_pca": 8}

    kwargs = dict(base_config=MLPConfig(max_epochs=3, patience=2))
    local = arm.fit_esm_ensemble(fit_rows, [held], inner, ("mid",), (0,),
                                 config, **kwargs)
    cloud = arm.fit_esm_ensemble(fit_rows, [held], inner, ("mid",), (0,),
                                 config, payload=payload, **kwargs)
    assert local["n_members"] == cloud["n_members"] == 2
    np.testing.assert_allclose(local["predictions"][0], cloud["predictions"][0],
                               rtol=0, atol=0)
    np.testing.assert_allclose(local["inner_oof"], cloud["inner_oof"],
                               rtol=0, atol=0)


def test_read_payload_rejects_a_corrupted_artifact(tmp_path, payload):
    """A half-uploaded 146 MB file that merely *exists* is worse than none.

    It would hand a container a different PCA basis and so a quietly different
    arm. The digest is recomputed from the loaded bytes, not read back from the
    file's own header.
    """
    path = tmp_path / "p.npz"
    arm.write_payload(path, payload)
    assert arm.read_payload(path)["digest"] == payload["digest"]

    tampered = dict(payload["blocks"])
    first = sorted(tampered)[0]
    block = tampered[first].copy()
    block.flat[0] = block.flat[0] + np.float32(1.0)
    tampered[first] = block
    assert arm.compute_payload_digest(tampered) != payload["digest"]


def test_payload_digest_is_the_one_the_run_was_priced_against(payload):
    """The artifact on the Volume must be the one the forecast described."""
    assert payload["digest"].startswith("6f5200968670ea7d")


def test_a0_comparator_reads_a0s_own_file(tmp_path):
    """Transcribed numbers drift; the comparator is read from A0's provenance."""
    absent = arm.a0_comparator(tmp_path)
    assert absent["available"] is False

    real = arm.a0_comparator(arm.OUT_DIR)
    if not real.get("available"):
        pytest.skip("A0 provenance not present")
    assert len(real["per_fold_median_spearman"]) == 5
    assert real["fold_spread"] == pytest.approx(
        max(real["per_fold_median_spearman"])
        - min(real["per_fold_median_spearman"]))
    # The spread with the model held fixed is ~the whole worthwhile bar, which
    # is why a single-fold difference cannot carry a verdict.
    assert real["fold_spread"] > 0.04
    assert "per-fold" in real["caveat"]


def test_inner_oof_residual_scale_reports_both_roles_without_a_verdict(train):
    """The ratio is measured and reported; which role stacks better is open.

    On A0 the measurement came out *against* the reasoning that motivated the
    third nesting level, so this function must not encode a preference.
    """
    rng = np.random.default_rng(0)
    pairs = train.pair_id.to_numpy()[:400]
    y = train.set_index("pair_id").y_log1p.reindex(pairs).to_numpy()
    frames = [pd.DataFrame({"pair_id": pairs, "outer_fold": 0,
                            "role": "outer_held",
                            "y_pred": y + rng.normal(scale=0.4, size=len(y))})]
    for role, scale in (("inner_oof_6", 0.44), ("inner_oof_30", 0.48)):
        for fold in range(4):
            frames.append(pd.DataFrame({
                "pair_id": pairs, "outer_fold": fold + 1, "role": role,
                "y_pred": y + rng.normal(scale=scale, size=len(y))}))
    oof = pd.concat(frames, ignore_index=True)

    out = arm.inner_oof_residual_scale(oof, train)
    assert out["outer_held"]["n_rows"] == len(pairs)
    for role in ("inner_oof_6", "inner_oof_30"):
        assert out[role]["n_rows"] == 4 * len(pairs)
        assert out[role]["rmse_ratio_measured_rows"] > 0
        assert out[role]["rmse_ratio_all_rows"] > 0
        # The replicate average is recorded as a hazard, never as a ratio.
        assert "rmse_replicate_averaged_DO_NOT_USE" in out[role]
        assert not any(k.endswith("_paired") for k in out[role])
    assert "OPEN" in out["interpretation"]
    assert "ONE SIDE ONLY" in out["interpretation"]
    assert "un-averaged" in out["method"]
    assert "do not bracket it" in out["interpretation"]


def test_replicate_averaging_can_reverse_the_residual_comparison(train):
    """The hazard the un-averaged rule exists to avoid, demonstrated.

    Four noisy replicates average to something tighter than any one of them --
    an ensemble-of-ensembles no stacker ever sees. On A0 that moved
    ``inner_oof_6`` from 2.0% above the target to below it, which would have
    fabricated a bracket around the out-of-sample residual scale.
    """
    rng = np.random.default_rng(7)
    pairs = train.pair_id.to_numpy()[:600]
    y = train.set_index("pair_id").y_log1p.reindex(pairs).to_numpy()
    frames = [pd.DataFrame({"pair_id": pairs, "outer_fold": 0,
                            "role": "outer_held",
                            "y_pred": y + rng.normal(scale=0.40, size=len(y))})]
    for fold in range(4):
        frames.append(pd.DataFrame({
            "pair_id": pairs, "outer_fold": fold + 1, "role": "inner_oof_6",
            "y_pred": y + rng.normal(scale=0.44, size=len(y))}))
        frames.append(pd.DataFrame({
            "pair_id": pairs, "outer_fold": fold + 1, "role": "inner_oof_30",
            "y_pred": y + rng.normal(scale=0.48, size=len(y))}))
    out = arm.inner_oof_residual_scale(pd.concat(frames, ignore_index=True),
                                       train)
    # Un-averaged, the inner role is correctly the worse one.
    assert out["inner_oof_6"]["rmse_ratio_measured_rows"] > 1.0
    # Averaged, it looks better than the target -- the sign reversal.
    assert (out["inner_oof_6"]["rmse_replicate_averaged_DO_NOT_USE"]
            < out["outer_held"]["rmse_measured_rows"])


def test_collect_from_journal_refuses_an_incomplete_run(tmp_path, train):
    """A contract file missing a fold is worse than no contract file."""
    journal = tmp_path / "runs.partial.jsonl"
    journal.write_text(json.dumps({
        "key": "L2_o0", "ok": True, "level": 2, "outer_fold": 0,
        "payload_digest": "x", "n_members": 30, "thread_pin_ok": True,
        "members": [], "pair_id": [], "y_pred": [], "inner_pair_id": [],
        "inner_oof_6": [], "n_features": 330, "n_predicted": 0,
        "n_alleles_eligible": 1, "eligible_row_coverage": 1.0,
        "median_per_allele_spearman": 0.5, "iqr_low": 0.1, "iqr_high": 0.9,
        "mae_log1p": 0.5, "n_alleles_unranked": 0,
        "median_per_allele_spearman_min50": 0.5,
        "n_alleles_eligible_min50": 1,
        "eligible_row_coverage_min50": 1.0}) + "\n")
    with pytest.raises(AssertionError, match="journal holds 1 units"):
        arm.collect_from_journal(journal, train, nested.ASSESS_INNER_FOLDS,
                                 nested.ASSESS_INNER_FOLDS, 30)


def test_collect_from_journal_refuses_a_failed_unit(tmp_path, train):
    journal = tmp_path / "runs.partial.jsonl"
    journal.write_text(json.dumps({"key": "L2_o0", "ok": False,
                                   "error": "boom"}) + "\n")
    with pytest.raises(AssertionError, match="still failed"):
        arm.collect_from_journal(journal, train, nested.ASSESS_INNER_FOLDS,
                                 nested.ASSESS_INNER_FOLDS, 30)


def test_collect_from_journal_takes_the_last_attempt_per_unit(tmp_path, train):
    """A resumed journal holds a unit's failure *then* its success.

    That is exactly what ``--resume`` writes, so judging every line rather than
    the final state per key would reject the runs this path exists to rescue.
    The real run hit this: 4 units failed on a bad assertion, were resumed, and
    left 34 lines for 30 units.
    """
    journal = tmp_path / "runs.partial.jsonl"
    journal.write_text(
        json.dumps({"key": "L2_o0", "ok": False, "error": "boom"}) + "\n"
        + json.dumps({"key": "L2_o0", "ok": True, "level": 2, "outer_fold": 0,
                      "payload_digest": "x", "n_members": 30,
                      "thread_pin_ok": True, "members": [], "pair_id": [],
                      "y_pred": [], "inner_pair_id": [], "inner_oof_6": [],
                      "n_features": 330, "n_predicted": 0}) + "\n")
    # It gets past the ok check and fails later, on the unit count -- which is
    # the point: the superseded failure no longer stops it.
    with pytest.raises(AssertionError, match="journal holds 1 units"):
        arm.collect_from_journal(journal, train, nested.ASSESS_INNER_FOLDS,
                                 nested.ASSESS_INNER_FOLDS, 30)


def test_quick_mode_writes_partial_filenames():
    """``--quick`` must not overwrite the canonical contract files.

    A5 blends against the canonical ones; a partial run replacing them would
    leave the blend averaging predictions that are no longer on disk.
    """
    source = SCRIPT.read_text()
    assert '"_partial" if args.quick else ""' in source
    for stem in ("phase_a_esm_oof_cluster", "phase_a_esm_runs",
                 "phase_a_esm_provenance"):
        assert f'f"{stem}{{suffix}}' in source
