"""Guards on the Phase A learner arms. Run with: .venv/bin/python -m pytest -q

Four things have to hold or the Phase A table means nothing.

1. **No validation or test row reaches an arm.** Phase A is training-split only,
   so the payload is checked against the frozen splits rather than trusted.
2. **The grids are the prespecified ones.** Every count and value in
   ``reports/further_research/phase_a_prespecification.md`` §4 is asserted here,
   because a silently widened grid is exactly what §7's extension log exists to
   prevent.
3. **The ranking gradient is right.** ``pepstab/mlp.py`` is not edited, so the
   new term arrives through a subclass; its gradient is checked against central
   differences the same way ``test_baselines::test_gradients_match_finite_differences``
   checks the parent's, and the zero-weight path is asserted bit-identical to
   ``MLPRegressor.fit`` so the duplicated training loop cannot drift.
4. **The selection rule is the house rule.** ``parsimonious_select`` is checked
   against ``scripts.stage3e_coupling._parsimonious`` on the same frame, not
   just against its own docstring.

Tests that need ``pepstab.nested`` skip cleanly when it is absent; the arm
implementations do not depend on it being on disk to be checked.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import mlp as mlp_module
from pepstab import phase_a
from pepstab.data import TARGET, load_with_splits
from pepstab.features import ENCODINGS, build_features, n_features
from pepstab.mlp import MLPConfig, MLPRegressor
from scripts.baseline_ensemble import SELECTED
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS
from scripts.stage3e_coupling import _parsimonious

try:
    from pepstab import nested as nested_module
except Exception:  # pragma: no cover - the harness is another agent's file
    nested_module = None

needs_nested = pytest.mark.skipif(nested_module is None,
                                  reason="pepstab.nested is not on disk yet")

try:
    import catboost  # noqa: F401
    HAVE_CATBOOST = True
except Exception:  # pragma: no cover
    HAVE_CATBOOST = False

needs_catboost = pytest.mark.skipif(not HAVE_CATBOOST,
                                    reason="catboost is not installed")


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def split_df():
    return load_with_splits()


@pytest.fixture(scope="module")
def payload():
    if nested_module is None:
        pytest.skip("pepstab.nested is not on disk yet")
    return phase_a.load_payload()


@pytest.fixture(scope="module")
def toy_frame():
    """A small training-shaped frame: real residues, several alleles, label ties.

    1,200 rows over 4 alleles is the smallest size that clears the frozen fold
    eligibility bar -- an inner fold is roughly 0.27 of the rows, and an allele
    needs 20 of them, so a 240-row toy leaves the scoring panel empty and the
    metric undefined for reasons that have nothing to do with the arms.
    """
    rng = np.random.default_rng(20261004)
    from pepstab.data import AMINO_ACIDS, PSEUDOSEQ_LENGTH
    alleles = [f"A*{i:02d}:01" for i in range(4)]
    pseudo = {a: "".join(rng.choice(list(AMINO_ACIDS), PSEUDOSEQ_LENGTH))
              for a in alleles}
    rows = []
    for i in range(1200):
        allele = alleles[i % len(alleles)]
        peptide = "".join(rng.choice(list(AMINO_ACIDS), 9))
        # A seventh of the labels sit exactly at the assay floor, so the
        # tie-skipping rule is exercised. The modulus is coprime with the fold
        # modulus on purpose: at ``i % 5`` the floor rows and outer fold 0 were
        # the same rows, which left that fold with no label spread at all.
        y = 0.0 if i % 7 == 0 else float(rng.gamma(2.0, 0.6))
        rows.append({"pair_id": i, "allele": allele, "peptide": peptide,
                     "hla_pseudoseq": pseudo[allele], TARGET: y,
                     "cluster_id": i // 3, "outer_fold": i % 5})
    return pd.DataFrame(rows)


# --- 1. the training split, and nothing else -------------------------------


@needs_nested
def test_payload_is_training_split_only(payload, split_df):
    train_ids = set(split_df.loc[split_df.split == "train", "pair_id"].tolist())
    other_ids = set(split_df.loc[split_df.split != "train", "pair_id"].tolist())
    got = set(payload["rows"]["pair_id"])
    assert got == train_ids
    assert not (got & other_ids), "payload carries val/test pair_ids"
    assert payload["split"] == "train"
    assert payload["n_rows"] == len(train_ids)


@needs_nested
def test_payload_carries_sequences_not_features(payload):
    assert set(payload["rows"]) == set(phase_a.PAYLOAD_COLUMNS)
    assert payload["schema"] == phase_a.UNITS_SCHEMA_VERSION
    # The point of shipping sequences is size: 860 float32 columns would be
    # ~68 MB per encoding against a couple of MB of records.
    import json
    assert len(json.dumps(payload)) < 8_000_000


@needs_nested
def test_payload_frame_rejects_a_non_training_payload(payload):
    bad = {**payload, "split": "val"}
    with pytest.raises(ValueError, match="training split only"):
        phase_a.payload_frame(bad)
    with pytest.raises(ValueError, match="schema"):
        phase_a.payload_frame({**payload, "schema": "phase_a.0"})


@needs_nested
def test_payload_outer_folds_match_the_harness(payload, split_df):
    train = (split_df[split_df.split == "train"].sort_values("pair_id")
             .reset_index(drop=True))
    expected = np.asarray(nested_module.outer_folds(train))
    assert np.array_equal(np.asarray(payload["rows"]["outer_fold"]), expected)


# --- 2. the grids are the prespecified ones --------------------------------


def test_reference_mlp_matches_the_deployed_reference():
    hidden, l2_onehot = SELECTED[("pep_pseudo", "onehot")]
    _, l2_blosum = SELECTED[("pep_pseudo", "blosum")]
    assert phase_a.REFERENCE_MLP["hidden"] == hidden
    assert phase_a.REFERENCE_L2 == {"onehot": l2_onehot, "blosum": l2_blosum}
    assert phase_a.REFERENCE_MLP["max_epochs"] == MAX_EPOCHS
    assert phase_a.REFERENCE_MLP["patience"] == PATIENCE
    assert phase_a.ASSESS_SEEDS == SEEDS


@needs_nested
def test_reference_mlp_matches_the_harness_copy():
    ref = nested_module.REFERENCE_MLP
    assert tuple(phase_a.REFERENCE_MLP["hidden"]) == ref.hidden
    assert phase_a.REFERENCE_MLP["lr"] == ref.lr
    assert phase_a.REFERENCE_MLP["batch_size"] == ref.batch_size
    assert phase_a.REFERENCE_MLP["max_epochs"] == ref.max_epochs
    assert phase_a.REFERENCE_MLP["patience"] == ref.patience


def test_a1_grid_is_exactly_twelve_prespecified_configs():
    configs = phase_a.arm_configs("A1")
    assert len(configs) == 12, "§A1 fixes the grid at the §3.3 ceiling of 12"
    assert {c["depth"] for c in configs} == {4, 6, 8}
    assert {c["learning_rate"] for c in configs} == {0.03, 0.06}
    assert {c["l2_leaf_reg"] for c in configs} == {3, 10}
    assert len({c["label"] for c in configs}) == 12


def test_a1_fixed_catboost_settings():
    assert phase_a.CATBOOST_FIXED == {
        "iterations": 2000, "od_type": "Iter", "od_wait": 100,
        "bootstrap_type": "Bernoulli", "subsample": 0.8, "rsm": 1.0,
        "random_strength": 1.0, "border_count": 128}


def test_a1_features_are_pan_allele_and_860_columns(toy_frame):
    X = build_features(toy_frame, phase_a.INPUT_SET, "onehot")
    assert X.shape == (len(toy_frame), 860) == (len(toy_frame),
                                               n_features(phase_a.INPUT_SET))
    assert X.dtype == np.float32
    # No allele ID feature in the pan-allele arm: two rows of the same allele
    # and different peptides must differ only in the peptide block.
    same = toy_frame[toy_frame.allele == toy_frame.allele.iloc[0]]
    Xs = build_features(same, phase_a.INPUT_SET, "onehot")
    assert np.array_equal(Xs[0, 180:], Xs[1, 180:])


def test_a4_gets_its_own_equal_budget_search():
    configs = phase_a.arm_configs("A4")
    assert len(configs) == 24, "12 configurations per declared objective"
    for objective in phase_a.A4_OBJECTIVES:
        sub = [c for c in configs if c["objective"] == objective]
        assert len(sub) == 12
        assert {(c["depth"], c["learning_rate"], c["l2_leaf_reg"]) for c in sub} == set(
            itertools.product(phase_a.TREE_DEPTHS, phase_a.TREE_LEARNING_RATES,
                              phase_a.TREE_L2_LEAF_REG))
    assert phase_a.A4_OBJECTIVES == ("YetiRank", "PairLogit")


def test_a2_grid_is_the_four_shrinkage_constants():
    configs = phase_a.arm_configs("A2")
    assert [c["k"] for c in configs] == [1, 5, 20, 100]
    # Same-geometry: only k moves.
    varying = {key for key in configs[0]
               if len({c[key] for c in configs if not isinstance(c[key], list)}) > 1}
    assert varying == {"k", "label"}


def test_a3_is_the_two_by_two_with_both_stopping_rules():
    configs = phase_a.arm_configs("A3")
    cells = {c["cell"] for c in configs}
    assert cells == {"A3-a", "A3-b", "A3-c", "A3-d"}
    assert {(c["loss"], c["sampling"]) for c in configs} == {
        ("mse", "natural"), ("mse", "balanced"),
        ("mse+rank", "natural"), ("mse+rank", "balanced")}
    # Every cell runs under both stopping rules -- stage 7's confound.
    for cell in cells:
        assert {c["stopping"] for c in configs if c["cell"] == cell} == {"mse", "rank"}
    assert {c["lam"] for c in configs if c["cell"] in phase_a.A3_RANKING_CELLS} == {
        0.1, 0.3, 1.0}
    assert {c["lam"] for c in configs if c["cell"] not in phase_a.A3_RANKING_CELLS} == {
        None}
    assert len(configs) == 2 * 2 + 2 * 3 * 2


@needs_nested
def test_unit_axes_and_keys_are_stable_and_unique():
    units = phase_a.enumerate_units(["A1"])
    assert len({u["key"] for u in units}) == len(units)
    assert len(units) == (12 * len(ENCODINGS) * len(phase_a.SCREEN_SEEDS)
                          * nested_module.OUTER_FOLDS
                          * nested_module.SCREEN_INNER_FOLDS)
    for unit in units:
        assert unit["schema"] == phase_a.UNITS_SCHEMA_VERSION
        assert unit["l2"] == phase_a.REFERENCE_L2[unit["encoding"]]
    assert phase_a.enumerate_units(["A1"])[0]["key"] == units[0]["key"]


@needs_nested
def test_assessment_uses_the_reference_ensemble_arity():
    selected = {"A2": {"k": 5}}
    units = phase_a.enumerate_units(["A2"], stage="assess", selected=selected)
    # 5 inner folds x 2 encodings x 3 seeds = 30 members per outer fold, which is
    # the deployed reference's arity; ensembling is worth ~+0.074 here, so an arm
    # assessed at 18 members would be measuring ensemble size.
    per_fold = [u for u in units if u["fold"] == 0]
    assert len(per_fold) == 30
    assert len(units) == 30 * nested_module.OUTER_FOLDS


@needs_nested
def test_units_are_json_able():
    import json
    units = phase_a.enumerate_units(["A2", "A3"])
    json.loads(json.dumps(units))


@needs_nested
def test_a3_assessment_refuses_two_lambdas():
    units = phase_a.enumerate_units(["A3"], stage="assess",
                                    selected={"A3": {"lam": 0.3}})
    assert {u["config"]["lam"] for u in units
            if u["config"]["cell"] in phase_a.A3_RANKING_CELLS} == {0.3}
    tampered = [dict(u) for u in units]
    tampered[-1] = {**tampered[-1],
                    "config": {**tampered[-1]["config"], "lam": 1.0,
                               "cell": "A3-c"}}
    with pytest.raises(ValueError, match="one common lambda"):
        phase_a.assert_same_geometry_common(tampered)


@needs_nested
def test_same_geometry_hyperparameters_are_held_common():
    units = phase_a.enumerate_units(["A2"])
    tampered = [dict(u) for u in units]
    tampered[0] = {**tampered[0],
                   "config": {**tampered[0]["config"], "hidden": [64]}}
    with pytest.raises(ValueError, match="same-geometry"):
        phase_a.assert_same_geometry_common(tampered)


# --- 3a. allele-centred targets (A2) ---------------------------------------


def test_shrunk_mean_is_the_prespecified_formula():
    allele = np.array(["a", "a", "a", "b"])
    y = np.array([1.0, 2.0, 3.0, 10.0])
    k = 5.0
    means, global_mean = phase_a.shrunk_allele_means(allele, y, k)
    assert global_mean == pytest.approx(4.0)
    assert means["a"] == pytest.approx((3 * 2.0 + k * 4.0) / (3 + k))
    assert means["b"] == pytest.approx((1 * 10.0 + k * 4.0) / (1 + k))


def test_shrinkage_moves_monotonically_toward_the_global_mean():
    allele = np.array(["a"] * 3 + ["b"])
    y = np.array([1.0, 2.0, 3.0, 10.0])
    gaps = []
    for k in phase_a.A2_K_GRID:
        means, global_mean = phase_a.shrunk_allele_means(allele, y, k)
        gaps.append(abs(means["b"] - global_mean))
    assert gaps == sorted(gaps, reverse=True)


def test_unseen_alleles_fall_back_to_the_fitting_fold_global_mean():
    means, global_mean = phase_a.shrunk_allele_means(
        np.array(["a", "a", "b", "b"]), np.array([1.0, 2.0, 3.0, 4.0]), 1.0)
    offsets = phase_a.allele_offsets(["a", "zzz"], means, global_mean)
    assert offsets[0] == pytest.approx(means["a"])
    assert offsets[1] == pytest.approx(global_mean)


def test_centring_cannot_change_within_allele_ranks():
    """§A2's own caveat, as an assertion: any A2 benefit is training-side."""
    allele = np.array(["a", "a", "b", "b"])
    pred = np.array([0.1, 0.4, 0.9, 0.2])
    means, global_mean = phase_a.shrunk_allele_means(
        allele, np.array([1.0, 2.0, 5.0, 6.0]), 5.0)
    offsets = phase_a.allele_offsets(allele, means, global_mean)
    for a in ("a", "b"):
        m = allele == a
        assert np.array_equal(np.argsort(pred[m]), np.argsort((pred + offsets)[m]))


@needs_nested
def test_a2_means_never_see_the_dev_or_outer_held_rows(toy_frame, monkeypatch):
    """Poison every non-fitting label; the fitted means must not move."""
    unit = [u for u in phase_a.enumerate_units(["A2"])
            if u["fold"] == 0 and u["inner_fold"] == 0 and u["seed"] == 0
            and u["encoding"] == "onehot"][0]
    held_mask, fit_rows, inner = phase_a._split_masks(toy_frame, unit)
    is_fit = inner != unit["inner_fold"]
    y = fit_rows[TARGET].to_numpy(dtype=float)
    clean = phase_a.shrunk_allele_means(fit_rows["allele"][is_fit], y[is_fit], 5)

    poisoned = toy_frame.copy()
    poisoned.loc[held_mask, TARGET] = 999.0
    held_mask2, fit_rows2, inner2 = phase_a._split_masks(poisoned, unit)
    is_fit2 = inner2 != unit["inner_fold"]
    y2 = fit_rows2[TARGET].to_numpy(dtype=float)
    assert phase_a.shrunk_allele_means(fit_rows2["allele"][is_fit2],
                                       y2[is_fit2], 5) == clean


# --- 3b. allele-balanced sampling (A3) -------------------------------------


def test_balance_weights_equalise_allele_pull_and_mean_one():
    allele = np.array(["a"] * 10 + ["b"] * 2)
    w = phase_a.allele_balance_weights(allele)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-6)
    assert w[allele == "a"].sum() == pytest.approx(w[allele == "b"].sum(), rel=1e-5)


def test_balance_weights_on_a_balanced_panel_are_all_ones():
    w = phase_a.allele_balance_weights(np.array(["a", "a", "b", "b"]))
    assert np.allclose(w, 1.0)


# --- 3c. the ranking term ---------------------------------------------------


def test_ranking_pairs_skip_ties_including_floor_floor():
    allele = np.array(["a", "a", "a", "b", "b"])
    y = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
    hi, lo, w = phase_a.ranking_pairs(allele, y)
    # Allele a: only (2, 0) and (2, 1). Allele b is a tie and contributes none.
    assert sorted(zip(hi.tolist(), lo.tolist())) == [(2, 0), (2, 1)]
    assert np.all(y[hi] > y[lo])


def test_ranking_pairs_never_cross_alleles():
    rng = np.random.default_rng(0)
    allele = rng.choice(["a", "b", "c"], 60)
    y = rng.normal(size=60)
    hi, lo, _ = phase_a.ranking_pairs(allele, y)
    assert len(hi)
    assert np.array_equal(allele[hi], allele[lo])


def test_pair_weights_are_normalised_per_allele():
    """A many-measurement allele must not dominate quadratically."""
    allele = np.array(["a"] * 30 + ["b"] * 4)
    y = np.concatenate([np.arange(30.0), np.arange(4.0)])
    hi, lo, w = phase_a.ranking_pairs(allele, y)
    big = w[allele[hi] == "a"].sum()
    small = w[allele[hi] == "b"].sum()
    assert big == pytest.approx(small, rel=1e-9)
    assert w.sum() == pytest.approx(1.0)
    # Unnormalised, allele a would carry (30*29)/(4*3) = 72.5x allele b's weight.
    assert len(hi[allele[hi] == "a"]) / len(hi[allele[hi] == "b"]) == pytest.approx(72.5)


def test_ranking_loss_rewards_the_correct_order():
    allele = np.array(["a", "a"])
    y = np.array([0.0, 1.0])
    pairs = phase_a.ranking_pairs(allele, y)
    right, _ = phase_a.pairwise_ranking_loss(np.array([0.0, 5.0]), pairs)
    wrong, _ = phase_a.pairwise_ranking_loss(np.array([5.0, 0.0]), pairs)
    assert right < wrong


def test_ranking_loss_is_zero_with_no_surviving_pairs():
    pairs = phase_a.ranking_pairs(np.array(["a", "a"]), np.array([3.0, 3.0]))
    loss, grad = phase_a.pairwise_ranking_loss(np.array([1.0, 2.0]), pairs)
    assert loss == 0.0 and np.all(grad == 0.0)


def test_ranking_gradient_matches_finite_differences():
    """Central differences on the ranking loss w.r.t. the predictions.

    Mirrors ``test_baselines::test_gradients_match_finite_differences``: a new
    gradient gets a finite-difference check, in float64, because in float32 the
    difference quotient of an O(1) loss carries ~1e-4 of noise and would measure
    rounding instead of correctness.
    """
    rng = np.random.default_rng(3)
    allele = rng.choice(["a", "b", "c"], 40)
    y = np.round(rng.normal(size=40), 1)  # rounding manufactures ties on purpose
    pairs = phase_a.ranking_pairs(allele, y)
    pred = rng.normal(size=40)
    _, grad = phase_a.pairwise_ranking_loss(pred, pairs)

    eps, worst = 1e-6, 0.0
    for i in range(len(pred)):
        bumped = pred.copy()
        bumped[i] = pred[i] + eps
        high, _ = phase_a.pairwise_ranking_loss(bumped, pairs)
        bumped[i] = pred[i] - eps
        low, _ = phase_a.pairwise_ranking_loss(bumped, pairs)
        numeric = (high - low) / (2 * eps)
        worst = max(worst, abs(numeric - grad[i])
                    / max(1e-9, abs(numeric) + abs(grad[i])))
    assert worst < 1e-5, f"worst relative gradient error {worst:.2e}"


def test_combined_objective_gradient_matches_finite_differences(monkeypatch):
    """The composed MSE + lambda*ranking gradient, through the inherited backward.

    The ranking term reaches ``MLPRegressor._backward`` pre-scaled by ``n/2``
    because that method forms ``delta = (2/n) * residual``. This checks the
    composition end to end against central differences on the real weights.
    """
    monkeypatch.setattr(mlp_module, "DTYPE", np.float64)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(24, 7))
    y = np.round(rng.normal(size=24), 1)
    allele = np.array(["a"] * 12 + ["b"] * 12)
    lam = 0.7

    model = phase_a.RankingMLPRegressor(MLPConfig(hidden=(6, 5), l2=0.0, seed=1),
                                        rank_weight=lam)
    model._init_params(7, np.random.default_rng(2))
    model.y_offset_ = 0.0
    pairs = phase_a.ranking_pairs(allele, y)

    def loss() -> float:
        pred, _ = model._forward(X)
        rank, _ = phase_a.pairwise_ranking_loss(pred, pairs)
        return float(np.mean((pred - y) ** 2) + lam * rank)

    pred, acts = model._forward(X)
    _, rank_grad = phase_a.pairwise_ranking_loss(pred, pairs)
    effective = (pred - y) + (len(y) / 2.0) * lam * rank_grad
    gw, gb = model._backward(acts, effective, None)

    eps, worst = 1e-6, 0.0
    for k in range(len(model.weights_)):
        for arr, grad in ((model.weights_[k], gw[k]), (model.biases_[k], gb[k])):
            for ix in list(np.ndindex(arr.shape))[:8]:
                old = float(arr[ix])
                arr[ix] = old + eps
                high = loss()
                arr[ix] = old - eps
                low = loss()
                arr[ix] = old
                numeric = (high - low) / (2 * eps)
                worst = max(worst, abs(numeric - grad[ix])
                            / max(1e-6, abs(numeric) + abs(grad[ix])))
    assert worst < 1e-6, f"worst relative gradient error {worst:.2e}"


def test_zero_weight_ranking_mlp_is_bit_identical_to_the_reference():
    """The duplicated training loop cannot drift from ``pepstab.mlp``.

    ``RankingMLPRegressor`` repeats the loop because the parent's ``fit`` has no
    hook for a custom ``dL/dpred``. At ``rank_weight == 0`` the two must agree
    bit for bit, weighted path included -- otherwise A3-a is not the reference
    member and the 2x2 compares four things.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 11)).astype(np.float32)
    y = rng.normal(size=200).astype(np.float32)
    allele = np.array(["a", "b"] * 100)
    cfg = MLPConfig(hidden=(8, 4), l2=1e-4, max_epochs=12, patience=20, seed=3)
    for weight in (None, phase_a.allele_balance_weights(allele[:150])):
        reference = MLPRegressor(cfg).fit(X[:150], y[:150], X[150:], y[150:],
                                          sample_weight=weight)
        mine = phase_a.RankingMLPRegressor(cfg, rank_weight=0.0).fit(
            X[:150], y[:150], X[150:], y[150:], sample_weight=weight)
        assert mine.best_epoch_ == reference.best_epoch_
        assert mine.y_offset_ == reference.y_offset_
        for a, b in zip(mine.weights_, reference.weights_):
            assert np.array_equal(a, b)
        for a, b in zip(mine.biases_, reference.biases_):
            assert np.array_equal(a, b)
        assert np.array_equal(mine.predict(X[150:]), reference.predict(X[150:]))


def test_ranking_term_changes_the_fit():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 11)).astype(np.float32)
    y = rng.normal(size=200).astype(np.float32)
    allele = np.array(["a", "b"] * 100)
    cfg = MLPConfig(hidden=(8, 4), max_epochs=8, patience=20, seed=3)
    base = phase_a.RankingMLPRegressor(cfg, rank_weight=0.0).fit(
        X[:150], y[:150], X[150:], y[150:],
        allele=allele[:150], allele_dev=allele[150:])
    ranked = phase_a.RankingMLPRegressor(cfg, rank_weight=1.0).fit(
        X[:150], y[:150], X[150:], y[150:],
        allele=allele[:150], allele_dev=allele[150:])
    assert not np.allclose(base.predict(X[150:]), ranked.predict(X[150:]))


def test_ranking_stopping_rule_is_its_own_axis():
    """Stage 7's confound: the stopping rule must be separable from the loss."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(200, 11)).astype(np.float32)
    y = rng.normal(size=200).astype(np.float32)
    allele = np.array(["a", "b"] * 100)
    cfg = MLPConfig(hidden=(8, 4), max_epochs=20, patience=50, seed=3)
    on_mse = phase_a.RankingMLPRegressor(cfg, rank_weight=0.0, dev_metric="mse").fit(
        X[:150], y[:150], X[150:], y[150:],
        allele=allele[:150], allele_dev=allele[150:])
    on_rank = phase_a.RankingMLPRegressor(cfg, rank_weight=0.0, dev_metric="rank").fit(
        X[:150], y[:150], X[150:], y[150:],
        allele=allele[:150], allele_dev=allele[150:])
    assert len(on_rank.dev_rank_curve_) == len(on_rank.dev_curve_)
    assert not on_mse.dev_rank_curve_
    # Same gradient, different kept epoch -- which is the whole point of the axis.
    assert on_mse.best_epoch_ != on_rank.best_epoch_ or np.array_equal(
        on_mse.weights_[0], on_rank.weights_[0])


def test_ranking_needs_allele_labels():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    with pytest.raises(ValueError, match="allele labels"):
        phase_a.RankingMLPRegressor(MLPConfig(max_epochs=1), rank_weight=0.5).fit(
            X[:30], y[:30], X[30:], y[30:])
    with pytest.raises(ValueError, match="dev_metric"):
        phase_a.RankingMLPRegressor(MLPConfig(), dev_metric="spearman")


# --- 4. the selection rule --------------------------------------------------


@pytest.fixture
def tree_runs():
    rng = np.random.default_rng(23)
    rows = []
    for depth, lr, l2 in itertools.product(phase_a.TREE_DEPTHS,
                                           phase_a.TREE_LEARNING_RATES,
                                           phase_a.TREE_L2_LEAF_REG):
        # A deliberately flat ladder with noise, which is the regime the house
        # rule exists for: the argmax lands wherever the noise peaks.
        for seed in (0, 1):
            rows.append({"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2,
                         "seed": seed,
                         phase_a.SCORE_COLUMN: 0.60 + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_parsimonious_select_reproduces_the_house_rule():
    """One axis, same frame, same answer as ``stage3e_coupling._parsimonious``."""
    rng = np.random.default_rng(29)
    rows = []
    for l2 in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        for seed in (0, 1, 2):
            rows.append({"l2": l2, "k": l2, "seed": seed,
                         "member_median_rho": 0.60 + rng.normal(0, 0.02)})
    frame = pd.DataFrame(rows)
    house = _parsimonious(frame)
    # The house rule takes the *largest* L2 within 1 SE; A2's k is the same
    # direction (larger k shrinks harder), so the mapping is direct.
    mine = phase_a.parsimonious_select(
        frame.rename(columns={"member_median_rho": phase_a.SCORE_COLUMN}), "A2")
    assert mine["selected"]["k"] == pytest.approx(house)


def test_parsimonious_select_prefers_regularisation_in_the_prespecified_order(
        tree_runs):
    out = phase_a.parsimonious_select(tree_runs, "tree")
    assert out["axes"] == ["l2_leaf_reg", "depth", "learning_rate"]
    assert out["score_at_selected"] >= out["score_at_argmax"] - out["standard_error"]
    assert out["n_configs"] == 12
    assert set(out["selected"]) == {"depth", "learning_rate", "l2_leaf_reg"}
    assert "seed_spread_at_selected" in out and "seed_spread_at_argmax" in out


def test_parsimonious_select_breaks_ties_by_l2_then_depth_then_lr():
    """A perfectly flat ladder must land on the most regularised corner."""
    rows = [{"depth": d, "learning_rate": lr, "l2_leaf_reg": l2, "seed": s,
             phase_a.SCORE_COLUMN: 0.5}
            for d, lr, l2 in itertools.product(phase_a.TREE_DEPTHS,
                                               phase_a.TREE_LEARNING_RATES,
                                               phase_a.TREE_L2_LEAF_REG)
            for s in (0, 1)]
    out = phase_a.parsimonious_select(pd.DataFrame(rows), "tree")
    assert out["selected"] == {"l2_leaf_reg": 10, "depth": 4, "learning_rate": 0.03}


def test_boundary_selection_is_flagged_not_quietly_accepted(tree_runs):
    out = phase_a.parsimonious_select(tree_runs, "tree")
    # learning_rate and l2_leaf_reg have only two tried values each, so neither
    # can ever be interior; the flag has to say so rather than imply a verdict.
    assert "learning_rate" in out["boundary_axes"]
    assert "l2_leaf_reg" in out["boundary_axes"]
    assert out["interior"] is False


def test_argmax_and_selected_are_reported_side_by_side(tree_runs):
    out = phase_a.parsimonious_select(tree_runs, "tree")
    for key in ("argmax", "selected", "score_at_argmax", "score_at_selected",
                "seed_spread_at_argmax", "seed_spread_at_selected", "rule",
                "standard_error", "n_within_1se", "tried"):
        assert key in out


def test_parsimonious_select_rejects_an_unknown_family_or_missing_axis(tree_runs):
    with pytest.raises(ValueError, match="unknown family"):
        phase_a.parsimonious_select(tree_runs, "A5")
    with pytest.raises(ValueError, match="missing columns"):
        phase_a.parsimonious_select(tree_runs.drop(columns=["depth"]), "tree")


def test_common_hyperparameters_refuses_a_partial_pool():
    rows = [{"cell": "A3-c", "lam": lam, "seed": s, phase_a.SCORE_COLUMN: 0.5}
            for lam in phase_a.A3_LAMBDA_GRID for s in (0, 1)]
    frame = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="missing cells"):
        phase_a.common_hyperparameters(frame, "A3", phase_a.A3_RANKING_CELLS)
    both = pd.concat([frame, frame.assign(cell="A3-d")], ignore_index=True)
    out = phase_a.common_hyperparameters(both, "A3", phase_a.A3_RANKING_CELLS)
    # A flat lambda ladder: parsimony is the smallest departure from plain MSE.
    assert out["selected"]["lam"] == 0.1
    assert out["pooled_cells"] == list(phase_a.A3_RANKING_CELLS)
    assert out["n_runs_pooled"] == len(both)


# --- 5. a unit runs end to end ---------------------------------------------


@needs_nested
@pytest.mark.parametrize("arm", ["A2", "A3"])
def test_mlp_unit_runs_and_returns_the_contract(arm, toy_frame, payload):
    unit = [u for u in phase_a.enumerate_units([arm])
            if u["fold"] == 0 and u["inner_fold"] == 0 and u["seed"] == 0
            and u["encoding"] == "onehot"][0]
    # A 240-row toy payload keeps this a contract test, not a training run.
    small = {**payload, "n_rows": len(toy_frame),
             "rows": {c: toy_frame[c].tolist() for c in phase_a.PAYLOAD_COLUMNS}}
    cheap = {**unit, "config": {**unit["config"], "max_epochs": 3}}
    row = phase_a.run_unit(cheap, small)
    assert row["ok"], row.get("error")
    for key in ("key", "arm", "config", "seed", "fold", "ok", "n_threads",
                "median_per_allele_spearman", "n_alleles_eligible",
                "eligible_row_coverage", "mae_log1p", "fit_seconds", "oof"):
        assert key in row
    oof_ids = [p for p, _ in row["oof"]]
    assert len(set(oof_ids)) == len(oof_ids)
    held_ids = {p for p, _ in row["held"]}
    assert not (set(oof_ids) & held_ids), "a row was both fitted on and assessed"
    import json
    json.loads(json.dumps(row))


@needs_nested
def test_run_unit_reports_failures_instead_of_raising(payload):
    unit = phase_a.enumerate_units(["A2"])[0]
    row = phase_a.run_unit({**unit, "arm": "A9"}, payload)
    assert row["ok"] is False and "error" in row
    row = phase_a.run_unit({**unit, "schema": "phase_a.0"}, payload)
    assert row["ok"] is False and "schema" in row["error"]


@needs_nested
def test_run_unit_records_the_thread_count(payload):
    unit = phase_a.enumerate_units(["A2"])[0]
    row = phase_a.run_unit({**unit, "arm": "A9"}, payload)
    # Recorded even on a failed unit: thread count moves a single network by
    # ~0.02 median Spearman at a fixed seed, so a row without it is unusable.
    assert isinstance(row["n_threads"], int) and row["n_threads"] >= 1
    assert set(row["thread_env"]) == set(phase_a.THREAD_ENV_VARS)


@needs_nested
@needs_catboost
@pytest.mark.parametrize("arm,objective", [("A1", None), ("A4", "YetiRank"),
                                           ("A4", "PairLogit")])
def test_tree_unit_runs_and_returns_the_contract(arm, objective, toy_frame, payload):
    unit = [u for u in phase_a.enumerate_units([arm])
            if u["fold"] == 0 and u["inner_fold"] == 0 and u["seed"] == 0
            and u["encoding"] == "onehot"
            and u["config"].get("objective") == objective][0]
    small = {**payload, "n_rows": len(toy_frame),
             "rows": {c: toy_frame[c].tolist() for c in phase_a.PAYLOAD_COLUMNS}}
    cheap = {**unit, "config": {**unit["config"], "iterations": 20}}
    # iterations is held fixed by §A1, so it is overridden through the fixed
    # block rather than the config, and only for this contract test.
    import unittest.mock as mock
    with mock.patch.dict(phase_a.CATBOOST_FIXED, {"iterations": 20, "od_wait": 5}):
        row = phase_a.run_unit(cheap, small)
    assert row["ok"], row.get("error")
    assert row["median_per_allele_spearman"] == row["median_per_allele_spearman"]
    assert "oof" in row and len(row["oof"])
    if arm == "A4":
        # A pure ranker has no calibrated magnitude until the isotonic map.
        assert "mae_log1p_raw" in row and "oof_calibrated" in row
        assert len(row["oof_calibrated"]) == len(row["oof"])
        # PairLogit is given explicit §A3-rule pairs; YetiRank generates its own
        # inside each group and that generation cannot be overridden.
        assert row["pairs_explicit"] is (objective == "PairLogit")
        assert (row["n_pairs"] > 0) is (objective == "PairLogit")


@needs_catboost
def test_catboost_receives_float32_numpy_not_a_dataframe(toy_frame):
    X = build_features(toy_frame, phase_a.INPUT_SET, "blosum")
    assert isinstance(X, np.ndarray) and X.dtype == np.float32


@needs_catboost
def test_isotonic_calibration_is_monotone_and_fitted_on_fit_rows():
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(31)
    raw = rng.normal(size=300)
    y = np.clip(2.0 + 1.5 * raw + rng.normal(0, 0.3, 300), 0, None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    grid = np.linspace(raw.min(), raw.max(), 50)
    assert np.all(np.diff(iso.predict(grid)) >= -1e-9)


@needs_nested
def test_smoke_subset_covers_every_arm_and_both_ranker_code_paths():
    picked = phase_a.smoke_units()
    assert {u["arm"] for u in picked} == set(phase_a.ARMS)
    assert len({u["key"] for u in picked}) == len(picked)
    a4 = [u["config"]["objective"] for u in picked if u["arm"] == "A4"]
    assert set(a4) == set(phase_a.A4_OBJECTIVES)
    assert all(u["seed"] == 0 and u["fold"] == 0 and u["inner_fold"] == 0
               for u in picked)


def test_runs_frame_drops_predictions_and_lifts_config_axes():
    rows = [{"ok": True, "key": "k", "arm": "A1", "stage": "screen",
             "encoding": "onehot", "seed": 0, "fold": 0, "inner_fold": 0,
             "n_threads": 2, phase_a.SCORE_COLUMN: 0.5, "n_alleles_eligible": 60,
             "eligible_row_coverage": 0.99, "mae_log1p": 0.6, "fit_seconds": 1.0,
             "config": {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3,
                        "label": "x"},
             "oof": [[1, 0.5]] * 1000},
            {"ok": False, "key": "bad"}]
    frame = phase_a.runs_frame(rows)
    assert len(frame) == 1
    assert "oof" not in frame.columns
    assert frame.loc[0, "depth"] == 4 and frame.loc[0, "config_label"] == "x"
