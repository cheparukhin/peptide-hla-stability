"""Guards on the Phase A screen's post-processing layer (`scripts/phase_a_select.py`).

Four kinds of check:

- **Arity** -- an assembled prediction must average exactly the screen's 12
  members (3 inner folds x 2 seeds x 2 encodings), and an assembled inner
  out-of-fold row exactly 4. Ensembling is worth roughly +0.07 to +0.09 median
  Spearman in this project, so a silently short mean would read as a worse arm
  rather than as a bug, and ``n_members`` has to reach every output row so a
  12-member screen number is never mistaken for a 30-member assessment number.
- **Schema** -- the journal's ``outer_fold`` field is null on every row and the
  outer fold lives in ``fold``. Reading the null field would collapse five folds
  into one bucket and return a single plausible-looking number that is wrong, so
  the fold is asserted non-null and agreed against the ``of=`` field in the key.
- **Delegation** -- the parsimony rule, the metric and the bootstrap are the
  project's frozen implementations. The tests assert this layer reproduces
  ``phase_a.parsimonious_select`` rather than re-deriving a rule, and that the
  §3 grid-boundary trigger is *reported* and not run.
- **Blend machinery** -- weights non-negative and summing to one, fitted on
  inner out-of-fold rows only.

Everything runs on synthetic journal rows except the two slower cases marked
with the real training split. No test touches ``val`` or ``test``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import nested, phase_a  # noqa: E402
from scripts import phase_a_select as sel  # noqa: E402

N_ROWS = 60
N_FOLDS = nested.OUTER_FOLDS


@pytest.fixture(scope="module")
def tiny_train() -> pd.DataFrame:
    """A synthetic stand-in for the training frame: pair ids, folds, labels.

    Only the columns the assembly path reads. The real split is used in the two
    tests that need the real folds.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "pair_id": np.arange(N_ROWS),
        "outer_fold": np.arange(N_ROWS) % N_FOLDS,
        "allele": ["HLA-A*02:01" if i % 2 else "HLA-B*07:02" for i in range(N_ROWS)],
        "y_log1p": rng.normal(size=N_ROWS),
        "cluster_id": np.arange(N_ROWS) // 2,
        "thalf_hours": rng.gamma(2.0, size=N_ROWS),
    })


def _unit_rows(train: pd.DataFrame, arm: str = "A1", drop: set | None = None,
               value_for=None, **overrides) -> list[dict]:
    """Journal rows for one configuration over every member axis and outer fold.

    Shaped like ``phase_a.run_unit``'s output, including the null ``outer_fold``
    the real screen writes.
    """
    config = phase_a.arm_configs(arm)[0]
    label = config["label"]
    rows = []
    for unit in phase_a.enumerate_units([arm]):
        if unit["config"]["label"] != label:
            continue
        axis = (unit["encoding"], unit["seed"], unit["inner_fold"], unit["fold"])
        if drop and axis in drop:
            continue
        on_fold = train["outer_fold"].to_numpy() == unit["fold"]
        fitting = train.loc[~on_fold, "pair_id"]
        # A unit's ``oof`` block holds **only** the rows of its own inner fold,
        # which is what makes an assembled inner row a 4-member mean rather than
        # a 12-member one. A deterministic partition stands in for the real
        # cluster water-fill; only the shape matters here.
        inner_rows = fitting[fitting % unit["n_inner"] == unit["inner_fold"]].tolist()
        held_rows = train.loc[on_fold, "pair_id"].tolist()
        value = (1.0 if value_for is None else value_for(unit))
        rows.append({
            "schema": phase_a.UNITS_SCHEMA_VERSION, "key": unit["key"],
            "arm": arm, "stage": "screen", "config": unit["config"],
            "encoding": unit["encoding"], "seed": unit["seed"],
            "fold": unit["fold"], "outer_fold": None,
            "inner_fold": unit["inner_fold"], "n_inner": unit["n_inner"],
            "n_threads": 2, "threads_observed": 2, "thread_pin_ok": True,
            "eval_metric": "RMSE", "payload_digest": "digest0", "ok": True,
            phase_a.SCORE_COLUMN: 0.5, "n_alleles_eligible": 2,
            "eligible_row_coverage": 1.0, "mae_log1p": 0.6, "fit_seconds": 1.0,
            "peak_rss_mb": 100.0,
            "oof": [[int(p), value] for p in inner_rows],
            "held": [[int(p), value] for p in held_rows],
            **overrides,
        })
    return rows


def _write_journal(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "journal.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _ingest(tmp_path: Path, train: pd.DataFrame, rows: list[dict]):
    by_key, axes = sel.expected_screen()
    journal = _write_journal(tmp_path, rows)
    assemblies, lite, stats = sel.read_journal(journal, by_key, train)
    return sel.assemble(assemblies, axes, train), lite, stats


# --- arity ------------------------------------------------------------------


def test_screen_arity_is_twelve_members_not_thirty():
    """The screen's arity is derived from the harness, and is not the assessment's."""
    assert sel.SCREEN_HELD_MEMBERS == 12
    assert sel.SCREEN_OOF_MEMBERS == 4
    assert sel.ASSESS_MEMBERS == 30
    assert sel.SCREEN_HELD_MEMBERS != sel.ASSESS_MEMBERS
    assert sel.INNER_OOF == "inner_oof_4"


def test_assembly_averages_every_member_and_records_the_count(tmp_path, tiny_train):
    """The assembled value is the member mean and ``n_members`` says how many."""
    # A distinct value per member axis, so a mean over the wrong subset cannot
    # coincidentally equal the mean over the right one.
    values = {}

    def value_for(unit):
        v = 1.0 + len(values)
        values[(unit["encoding"], unit["seed"], unit["inner_fold"], unit["fold"])] = v
        return v

    rows = _unit_rows(tiny_train, value_for=value_for)
    (assembled, coverage), _, stats = _ingest(tmp_path, tiny_train, rows)

    assert stats["units_present"] == len(rows)
    assert coverage["n_members_outer_held"].eq(sel.SCREEN_HELD_MEMBERS).all()
    for fold in range(N_FOLDS):
        key = next(k for k in assembled if k[2] == fold)
        members = [v for (enc, seed, inner, f), v in values.items() if f == fold]
        assert len(members) == sel.SCREEN_HELD_MEMBERS
        np.testing.assert_allclose(assembled[key][sel.OUTER_HELD],
                                   np.mean(members))

    table = sel.assembled_table(assembled, tiny_train,
                               inner_for={("A1", next(iter(assembled))[1])})
    assert set(table["n_members"].unique()) == {sel.SCREEN_HELD_MEMBERS,
                                                sel.SCREEN_OOF_MEMBERS}
    held = table[table.role == sel.OUTER_HELD]
    assert len(held) == len(tiny_train)
    assert held["n_members"].eq(sel.SCREEN_HELD_MEMBERS).all()
    assert table[table.role == sel.INNER_OOF]["n_members"].eq(
        sel.SCREEN_OOF_MEMBERS).all()


def test_inner_oof_row_averages_four_members(tmp_path, tiny_train):
    """A row sits in one inner fold, so its inner mean has 2 encodings x 2 seeds."""
    rows = _unit_rows(tiny_train,
                      value_for=lambda u: 10.0 if u["inner_fold"] == 0 else 1.0)
    (assembled, _), _, _ = _ingest(tmp_path, tiny_train, rows)
    block = assembled[("A1", rows[0]["config"]["label"], 0)]
    # Every inner out-of-fold value must be one of the per-inner-fold constants,
    # never a blend of them: a blend would mean a row was averaged across the
    # inner folds, which is 12 members and a different object.
    assert set(np.round(block[sel.INNER_OOF], 6)) <= {10.0, 1.0}


def test_a_short_member_set_is_not_assembled(tmp_path, tiny_train):
    """A fold missing one member is left incomplete rather than averaged over 11."""
    rows = _unit_rows(tiny_train, drop={("onehot", 0, 0, 2)})
    (assembled, coverage), _, _ = _ingest(tmp_path, tiny_train, rows)
    assert 2 not in {fold for _, _, fold in assembled}
    assert len({fold for _, _, fold in assembled}) == N_FOLDS - 1
    assert coverage["n_folds_complete"].iloc[0] == N_FOLDS - 1


def test_a_duplicated_unit_cannot_inflate_the_member_count(tmp_path, tiny_train):
    """Re-appending a unit is ignored: arity is axis identity, not a count to 12."""
    rows = _unit_rows(tiny_train)
    (assembled, _), _, stats = _ingest(tmp_path, tiny_train, rows + [rows[0]])
    assert stats["duplicate_keys"] == 1
    assert stats["units_present"] == len(rows)
    assert len(assembled) == N_FOLDS


# --- journal schema ---------------------------------------------------------


def test_outer_fold_comes_from_fold_not_from_the_null_field(tiny_train):
    """``outer_fold`` is null on every real row; ``fold`` is authoritative."""
    row = _unit_rows(tiny_train)[7]
    assert row["outer_fold"] is None
    assert sel.unit_outer_fold(row) == row["fold"]

    with pytest.raises(AssertionError, match="null fold"):
        sel.unit_outer_fold({**row, "fold": None})
    # A non-null outer_fold would mean the schema moved under this script.
    with pytest.raises(AssertionError, match="non-null 'outer_fold'"):
        sel.unit_outer_fold({**row, "outer_fold": 3})


def test_fold_is_cross_checked_against_the_unit_key(tiny_train):
    row = _unit_rows(tiny_train)[0]
    moved = {**row, "fold": (row["fold"] + 1) % N_FOLDS}
    with pytest.raises(AssertionError, match="key encodes"):
        sel.unit_outer_fold(moved)


def test_an_unpinned_thread_count_is_refused(tmp_path, tiny_train):
    """thread_pin_ok=False means that member's float32 results are not poolable."""
    rows = _unit_rows(tiny_train)
    rows[3] = {**rows[3], "thread_pin_ok": False}
    with pytest.raises(AssertionError, match="thread_pin_ok=False"):
        _ingest(tmp_path, tiny_train, rows)


def test_mixed_thread_counts_are_not_averaged(tmp_path, tiny_train):
    """Thread count moves a single network by ~0.02 median Spearman here."""
    rows = _unit_rows(tiny_train)
    rows[3] = {**rows[3], "n_threads": 8, "threads_observed": 8}
    with pytest.raises(AssertionError, match="different thread counts"):
        _ingest(tmp_path, tiny_train, rows)


def test_mixed_payload_digests_are_refused(tmp_path, tiny_train):
    """Two digests mean the units were not fitted on the same training rows."""
    rows = _unit_rows(tiny_train)
    rows[5] = {**rows[5], "payload_digest": "digest1"}
    with pytest.raises(AssertionError, match="payload digests"):
        _ingest(tmp_path, tiny_train, rows)


def test_a_truncated_tail_line_is_tolerated(tmp_path, tiny_train):
    """The screen appends while this runs, so a half-written last line is normal."""
    rows = _unit_rows(tiny_train)
    path = _write_journal(tmp_path, rows)
    with path.open("a") as handle:
        handle.write('{"schema": "phase_a.1", "key": "A1|dep')
    by_key, _ = sel.expected_screen()
    _, _, stats = sel.read_journal(path, by_key, tiny_train)
    assert stats["truncated_tail"] == 1
    assert stats["units_present"] == len(rows)
    assert not stats["complete"]


def test_a_failed_unit_is_skipped_not_assembled(tmp_path, tiny_train):
    rows = _unit_rows(tiny_train)
    rows[2] = {**rows[2], "ok": False, "error": "ValueError: synthetic"}
    (assembled, _), lite, stats = _ingest(tmp_path, tiny_train, rows)
    assert stats["not_ok"] == 1
    assert len(lite) == len(rows) - 1
    assert len(assembled) == N_FOLDS - 1


def test_the_screen_expects_1920_units_over_a1_a2_a3():
    """A4 is withdrawn and deferred, so it is simply absent -- never special-cased."""
    by_key, axes = sel.expected_screen()
    assert len(by_key) == 1920
    assert {u["arm"] for u in by_key.values()} == set(sel.SCREEN_ARMS)
    assert "A4" not in sel.SCREEN_ARMS
    assert all(len(members) == sel.SCREEN_HELD_MEMBERS for members in axes.values())


# --- selection --------------------------------------------------------------


def _ladder(scores: dict, seeds=(0, 1)) -> pd.DataFrame:
    """A tree-family runs frame with one mean score per ``l2_leaf_reg``."""
    rows = []
    for (depth, lr, l2), score in scores.items():
        for seed in seeds:
            for fold in range(N_FOLDS):
                rows.append({"arm": "A1", "depth": depth, "learning_rate": lr,
                             "l2_leaf_reg": l2, "seed": seed, "fold": fold,
                             "config_label": f"depth={depth},l2_leaf_reg={l2},"
                                             f"learning_rate={lr}",
                             phase_a.SCORE_COLUMN: score})
    return pd.DataFrame(rows)


def test_selection_delegates_to_the_house_parsimony_rule():
    """This layer reports ``parsimonious_select``; it does not re-derive the rule."""
    runs = _ladder({(4, 0.03, 3): 0.50, (4, 0.03, 10): 0.50,
                    (6, 0.03, 3): 0.51, (6, 0.03, 10): 0.51})
    direct = phase_a.parsimonious_select(runs, "tree")
    reported = next(s for s in sel.select_arms(runs) if s["ladder"] == "declared")
    assert reported["selected"] == sel._jsonable(direct["selected"])
    assert reported["argmax"] == sel._jsonable(direct["argmax"])
    assert reported["standard_error"] == pytest.approx(direct["standard_error"])
    assert reported["boundary_axes"] == direct["boundary_axes"]


def test_the_boundary_trigger_fires_at_the_weak_end_and_is_not_run():
    """A selection at ``l2_leaf_reg == 3`` fires §3's trigger, which is reported only."""
    # A flat ladder: every configuration is within one standard error, so the
    # parsimony rule takes the most regularised -- and a deliberately reversed
    # ladder puts that at the weak end instead.
    weak = _ladder({(4, 0.03, 3): 0.60, (4, 0.03, 10): 0.40})
    report = sel.boundary_report("tree", phase_a.parsimonious_select(weak, "tree"))
    assert report["trigger_fired"] is True
    assert report["trigger_axis"] == "l2_leaf_reg"
    assert report["declared_extension"] == "l2_leaf_reg -> {1, 3, 10, 30}"
    assert report["extension_run"] is False

    strong = _ladder({(4, 0.03, 3): 0.40, (4, 0.03, 10): 0.60})
    other = sel.boundary_report("tree", phase_a.parsimonious_select(strong, "tree"))
    assert other["trigger_fired"] is False
    assert other["strong_boundary"] is True


def test_boundary_axes_not_interior_is_the_quantity_reported():
    """§6a: ``interior`` is structurally unreachable for the two-value tree axes."""
    runs = _ladder({(4, 0.03, 3): 0.50, (4, 0.03, 10): 0.51})
    selection = phase_a.parsimonious_select(runs, "tree")
    assert selection["interior"] is False
    report = sel.boundary_report("tree", selection)
    assert set(report["unextendable_by_construction"]) >= {"l2_leaf_reg",
                                                           "learning_rate"}


def test_a3_lambda_is_chosen_on_the_pooled_ranking_cells():
    """§3's same-geometry rule: one common lambda, refused on a partial pool."""
    rows = []
    for cell in phase_a.A3_RANKING_CELLS:
        for lam in phase_a.A3_LAMBDA_GRID:
            for stopping in phase_a.A3_STOPPING:
                for seed in (0, 1):
                    rows.append({"arm": "A3", "cell": cell, "lam": lam,
                                 "stopping": stopping, "seed": seed,
                                 "config_label": f"cell={cell},lam={lam},"
                                                 f"stopping={stopping}",
                                 # Flat in lambda, with a seed spread so the
                                 # standard error is positive: §6a records in
                                 # advance that a flat lambda ladder always
                                 # selects 0.1 and is always flagged at a
                                 # boundary.
                                 phase_a.SCORE_COLUMN: 0.5 + (0.002 if seed
                                                              else -0.002)})
    runs = pd.DataFrame(rows)
    selections = sel.select_arms(runs)
    declared = next(s for s in selections if s["ladder"] == "declared")
    assert declared["pooled_cells"] == list(phase_a.A3_RANKING_CELLS)
    # Parsimony for lambda is the smallest departure from the reference MSE
    # objective (§6a), so a flat ladder selects 0.1 and is flagged at a boundary.
    assert declared["selected"]["lam"] == pytest.approx(0.1)
    assert "lam" in declared["boundary_axes"]
    assert [s["ladder"] for s in selections if s["ladder"].startswith("diagnostic")]

    partial = runs[runs.cell == phase_a.A3_RANKING_CELLS[0]]
    assert next(s for s in sel.select_arms(partial)
                if s["ladder"] == "declared")["error"].startswith("ValueError")


# --- blend machinery --------------------------------------------------------


def test_simplex_grid_is_the_simplex():
    for n in (2, 3):
        grid = sel.simplex_grid(n, step=0.1)
        assert grid.shape[1] == n
        assert (grid >= 0).all()
        np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    # The enumeration starts at the reference-heavy corner, which fixes the
    # tie-break on a flat objective.
    np.testing.assert_allclose(sel.simplex_grid(2, step=0.1)[0], [1.0, 0.0])


def test_blend_weights_are_non_negative_and_sum_to_one():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    good = y + 0.1 * rng.normal(size=200)
    noise = rng.normal(size=200)
    weights, mse = sel.fit_blend_weights([good, noise], y)
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    # The informative component must take essentially all the weight, and the
    # blend must beat the noise component alone.
    assert weights[0] > 0.9
    assert mse < float(np.mean((noise - y) ** 2))


def test_blend_weights_split_between_two_equally_good_components():
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    a = y + rng.normal(size=400)
    b = y + rng.normal(size=400)
    weights, _ = sel.fit_blend_weights([a, b], y)
    assert weights.sum() == pytest.approx(1.0)
    assert 0.2 < weights[0] < 0.8


def test_blend_refuses_components_that_do_not_match_the_rows():
    with pytest.raises(ValueError, match="rows but y has"):
        sel.fit_blend_weights([np.zeros(5), np.zeros(5)], np.zeros(4))


# --- the real split ---------------------------------------------------------


def test_the_assessment_frame_is_training_rows_only():
    """The one unrecoverable mistake available: never read val or test."""
    train = sel.train_frame()
    assert set(train["split"].unique()) == {"train"}
    assert len(train) == 19716
    assert sorted(train["outer_fold"].unique()) == list(range(N_FOLDS))


def test_outer_folds_match_the_file_a0_wrote():
    """Fold identity against A0's own output, not an assumption of determinism."""
    checks = sel.check_outer_folds(sel.train_frame())
    assert checks and "identical" in checks[0]


def test_a0_reference_reproduces_its_published_per_fold_medians():
    """A0 is the reference *and* the end-to-end check on this metric path."""
    train = sel.train_frame()
    scores = sel.score_reference("A0", sel.a0_predictions().set_index(
        "pair_id")["y_pred"], train, sel.ASSESS_MEMBERS)
    assert sel.check_a0_reference(scores)
    five = scores[(scores.scope == "five_fold")
                  & (scores.panel_min_rows == nested.MIN_ROWS_PER_ALLELE_FOLD)]
    assert float(five["median_per_allele_spearman"].iloc[0]) == pytest.approx(
        sel.A0_DECLARED_MEDIAN_OF_MEDIANS, abs=sel.A0_DECLARED_TOLERANCE)
    # A0's own fold-to-fold span with the model held fixed, which is the bar a
    # difference has to clear before it is a result at all.
    assert float(five["fold_span"].iloc[0]) == pytest.approx(0.0484, abs=5e-4)


def test_both_panels_are_scored_and_the_fifty_row_panel_is_much_narrower():
    """§2 requires the >=50 panel for every arm; per fold it is a weaker check.

    §2 quotes >=50 as 68/75 alleles at 99.51% coverage, which is the *pooled*
    training split. On a ~3,943-row outer fold the same bar keeps far fewer
    alleles and a much smaller share of the rows, so the sensitivity check is not
    the mild variation the pooled figures suggest.
    """
    train = sel.train_frame()
    scores = sel.score_reference("A0", sel.a0_predictions().set_index(
        "pair_id")["y_pred"], train, sel.ASSESS_MEMBERS)
    folds = scores[scores.scope == "fold"]
    assert set(folds["panel_min_rows"]) == set(sel.PANELS)
    wide = folds[folds.panel_min_rows == 20]
    narrow = folds[folds.panel_min_rows == 50]
    assert wide["eligible_row_coverage"].min() > 0.99
    assert narrow["eligible_row_coverage"].max() < 0.85
    assert narrow["n_alleles_eligible"].max() < wide["n_alleles_eligible"].min()


def test_the_inner_surrogates_are_one_sided_and_replicate_averaging_flips_the_sign():
    """Scored per replicate row, as a stacker sees them -- never pre-averaged.

    Both inner surrogates are pessimistic: they bound the out-of-sample residual
    scale from one side and do not bracket it, with ``inner_oof_6`` the tighter.
    Averaging each row's four replicates first puts ``inner_oof_6`` *below*
    ``outer_held`` and fabricates the bracket, so the wrong aggregation is
    computed and labelled rather than left available to re-derive.
    """
    table = sel.surrogate_scale_table(sel.train_frame())
    per_row = table[table.aggregation == "per replicate row"].set_index("role")
    assert per_row.loc["inner_oof_6", "ratio_measured_to_outer_held"] > 1.0
    assert per_row.loc["inner_oof_30", "ratio_measured_to_outer_held"] > 1.0
    assert (per_row.loc["inner_oof_6", "ratio_measured_to_outer_held"]
            < per_row.loc["inner_oof_30", "ratio_measured_to_outer_held"])
    assert per_row.loc["inner_oof_6", "n_rows"] == 4 * 19716

    averaged = table[table.aggregation.str.startswith("replicate-averaged")]
    assert averaged["aggregation"].str.contains("WRONG").all()
    flipped = averaged.set_index("role").loc[
        "inner_oof_6", "ratio_measured_to_outer_held"]
    assert flipped < 1.0


def test_the_declared_extension_is_checked_for_direction_not_trusted():
    """§3 pairs each trigger with a remedy; for two families they point apart.

    The remedy is declared as "the axis extends once, by one declared step **in
    that direction**". For the tree ladder the declared set ``{1, 3, 10, 30}``
    reaches both ends. For A3's lambda the trigger is ``lam == 1.0`` while the
    declared set ``{0.03, 0.1, 0.3, 1.0}`` adds a value at the *other* end, so a
    fired trigger has no step past it; A2's ``k`` has the mirror gap at the
    strong end. This layer measures that rather than asserting the remedy fits.
    """
    tree = sel.boundary_report(
        "tree", phase_a.parsimonious_select(
            _ladder({(4, 0.03, 3): 0.60, (4, 0.03, 10): 0.40}), "tree"))
    assert tree["trigger_fired"] is True
    assert tree["extension_reaches_past_selection"] is True
    assert tree["extension_values_past_selection"] == [1]

    lam_rows = []
    for lam in phase_a.A3_LAMBDA_GRID:
        for cell in phase_a.A3_RANKING_CELLS:
            for seed in (0, 1):
                lam_rows.append({"arm": "A3", "cell": cell, "lam": lam,
                                 "seed": seed, "config_label": f"lam={lam}",
                                 # Rising in lambda, so the parsimony rule
                                 # cannot pull the selection back off 1.0.
                                 phase_a.SCORE_COLUMN: 0.5 + 0.05 * lam})
    a3 = sel.boundary_report("A3", phase_a.parsimonious_select(
        pd.DataFrame(lam_rows), "A3"))
    assert a3["trigger_fired"] is True
    assert a3["extension_reaches_past_selection"] is False
    assert a3["extension_values_past_selection"] == []


def test_weak_end_axes_are_derived_and_name_axes_the_trigger_misses():
    """§3 names only ``l2_leaf_reg`` for the tree family; depth can also edge out."""
    runs = _ladder({(4, 0.03, 10): 0.40, (8, 0.06, 10): 0.60})
    report = sel.boundary_report("tree", phase_a.parsimonious_select(runs, "tree"))
    # depth 8 and learning_rate 0.06 are the least-regularised ends of their
    # axes under phase_a.REGULARISATION_ORDER, and neither carries a trigger.
    assert set(report["weak_end_axes"]) >= {"depth", "learning_rate"}
    assert "l2_leaf_reg" not in report["weak_end_axes_without_a_declared_trigger"]
    assert set(report["weak_end_axes_without_a_declared_trigger"]) == {
        "depth", "learning_rate"}
