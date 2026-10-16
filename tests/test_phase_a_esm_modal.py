"""Guards on the A5/ESM Modal runner (`modal_app/phase_a_esm.py`).

Mirrors ``tests/test_phase_a_modal.py``: everything here runs **locally**,
starts no container and costs nothing. Five kinds of check:

- **Shape of the batch** -- the unit set, the axis product that reaches A0's
  arity, and the network count. An arity mismatch between A5's two blend
  components would land as the blend's main effect, so the product is asserted
  term by term rather than in total.
- **CPU only, by reading the decorator**, not the prose. Per
  ``modal_app/boltz_common.py:3-8`` Modal validates every function in an app at
  creation time, so one ``gpu=`` would make the app undeployable in a workspace
  without GPU access and take the CPU work with it.
- **Container hygiene** -- no repo file read at module scope, BLAS threads
  pinned in the image ``.env`` (a worker cannot pin them late enough), the
  Volume mounted, and torch/catboost absent.
- **Cost model** -- the forecast refuses to invent a per-unit time, and the
  rates come from the repo rather than a pricing page.
- **The journal** -- resume must reconstruct a run without refitting, and a
  journal present without ``--resume`` must be a hard stop.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modal_app import phase_a_esm as app  # noqa: E402
from pepstab import nested  # noqa: E402
from scripts.baseline_sequence import SEEDS  # noqa: E402

SOURCE = Path(app.__file__).read_text()
MODULE = ast.parse(SOURCE)


def code_of(name: str) -> str:
    """One top-level function's **code**, with its docstring and comments gone.

    These tests assert properties of the runner, and this file's own prose
    quotes the hazards it guards against -- "no ``gpu=``", "no isotonic map".
    Matching raw source would let a comment satisfy a test, which is the
    opposite of the point.
    """
    for node in MODULE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == name:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body = body[1:]
            return "\n".join(ast.unparse(stmt) for stmt in body)
    raise AssertionError(f"{name} is not a top-level function of the runner")


def decorator_of(name: str) -> str:
    """The decorator call on one top-level function, as source."""
    for node in MODULE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == name:
            return "\n".join(ast.unparse(d) for d in node.decorator_list)
    raise AssertionError(f"{name} is not a top-level function of the runner")


# --- shape of the batch ---------------------------------------------------

def test_units_are_fitting_partitions_not_fits():
    """30 units: 5 level-2 partitions and 25 level-3 cells.

    The unit is a *fitting partition* because the PCA is refitted per
    partition; a per-network unit would do the same feature assembly 30 times.
    """
    units = app.enumerate_units()
    assert len(units) == 30
    assert sum(1 for u in units if u["level"] == 2) == nested.OUTER_FOLDS == 5
    assert sum(1 for u in units if u["level"] == 3) == 25
    assert len({u["key"] for u in units}) == 30
    for unit in units:
        assert unit["schema"] == app.UNITS_SCHEMA_VERSION
        if unit["level"] == 2:
            assert unit["inner_fold"] == -1
        else:
            assert 0 <= unit["inner_fold"] < nested.ASSESS_INNER_FOLDS


def test_axis_product_reaches_a0_arity_term_by_term():
    """5 inner folds x 2 ESM layers x 3 seeds = 30, exactly A0's product.

    A0 multiplies 5 inner folds x 2 encodings x 3 seeds. The layer axis is the
    two-way axis stage 3 declared as the stand-in for the encoding axis, so the
    two products match factor for factor, not only in total.
    """
    assert len(app.LAYERS) == 2
    assert nested.ASSESS_INNER_FOLDS * len(app.LAYERS) * len(SEEDS) == 30
    assert len(app.enumerate_units()) * 30 == 900


def test_smoke_subset_is_predeclared_and_covers_both_levels():
    """3-5 units, per the project invariant, naming both nesting levels.

    Named rather than "the first n the enumerator emitted", which would be
    three level-2 units and no level-3 one.
    """
    picked = app.smoke_units()
    assert 3 <= len(picked) <= 5
    assert {u["level"] for u in picked} == {2, 3}
    assert [u["key"] for u in picked] == ["L2_o0", "L3_o0_i0", "L3_o0_i1"]


def test_sentinels_and_panels_match_the_shared_protocol():
    assert app.NO_SUB_FOLD == -1
    assert app.SENSITIVITY_MIN_ROWS == 50
    assert nested.MIN_ROWS_PER_ALLELE_FOLD == 20
    assert app.N_TRAIN_ROWS == 19_716
    assert app.PEP_REP == "pos" and app.HLA_REP == "contact"


# --- CPU only, read from the decorator ------------------------------------

def test_no_function_declares_a_gpu():
    """Asserted by reading every decorator, not by trusting the docstring."""
    decorated = [n.name for n in MODULE.body
                 if isinstance(n, ast.FunctionDef) and n.decorator_list]
    assert "run_chunk" in decorated
    for name in decorated:
        assert "gpu" not in decorator_of(name), name
    # And no GPU rate is consulted anywhere in the cost model.
    assert "gpu" not in code_of("forecast_numbers")
    assert "gpu" not in code_of("load_rates")


def test_run_chunk_mounts_the_payload_volume_and_reserves_cpu():
    decorator = decorator_of("run_chunk")
    assert "volumes={PAYLOAD_MOUNT: payload_volume}" in decorator
    assert "cpu=WORKER_CPU" in decorator and "memory=WORKER_MEMORY" in decorator
    assert "retries=1" in decorator
    assert app.WORKER_CPU == 2.0 and app.WORKER_MEMORY == 4096
    assert app.payload_volume is not None


def test_image_pins_blas_threads_and_omits_torch_and_catboost():
    """A worker cannot pin threads late enough, so the image must.

    The BLAS pool is sized when numpy is imported and importing the science
    layer imports numpy, so the image environment -- in place before the
    interpreter starts -- is the only point early enough. torch and catboost
    are absent because nothing here uses them: the learner is this project's
    numpy MLP and the payload arrives pre-sliced, so no container reads the ESM
    extractor.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert f'"{var}": str(WORKER_THREADS)' in SOURCE
    assert '"torch' not in SOURCE
    assert '"catboost' not in SOURCE
    # randomized_svd is the peptide PCA, and a different version would change
    # the basis and so the features -- the exact local/cloud drift this app
    # exists to avoid.
    assert '"scikit-learn==1.9.1"' in SOURCE
    assert app.WORKER_THREADS == int(app.WORKER_CPU)


def test_pepstab_modules_cover_the_import_graph():
    """``pepstab/__init__.py`` imports augment, so omitting it breaks every
    ``import pepstab.*`` -- as silent a failure as a module-scope repo read.
    ``esm.py`` is deliberately absent: no container reads the cache."""
    assert "augment.py" in app.PEPSTAB_MODULES
    assert "nested.py" in app.PEPSTAB_MODULES
    assert "esm.py" not in app.PEPSTAB_MODULES
    repo = Path(app.REPO)
    for module in app.PEPSTAB_MODULES:
        assert (repo / "pepstab" / module).exists(), module


# --- container hygiene ----------------------------------------------------

def test_no_repo_file_is_read_at_module_scope():
    """A module-scope repo read kills the container on import.

    It surfaces as "0 tasks" and nothing else -- the defect documented at
    ``modal_app/proteinmpnn_scoring.py:97-105``, which neither a local forecast
    nor ``--dry-run`` can catch because both run where the file is present.
    """
    head = SOURCE.split("def load_rates")[0]
    for pattern in ("read_text()", "json.load", "pd.read_csv", "np.load",
                    "open("):
        assert pattern not in head, f"module scope reads a file: {pattern}"
    # The rates path is a Path object at module scope, which is fine; reading
    # it is what must be lazy.
    assert "RATES_PATH = REPO" in SOURCE
    assert "RATES_PATH.read_text()" in code_of("load_rates")


def test_self_check_is_above_the_modal_import():
    """``--self-check`` must not import the SDK at all.

    ``modal run --dry-run`` still builds and validates the image, so "imports
    modal" and "free" are not the same claim, and this project has already
    described a ``--dry-run`` as free while it quietly started a container.
    """
    assert SOURCE.index("def self_check") < SOURCE.index("\nimport modal")
    assert SOURCE.index('"--self-check" in sys.argv') < SOURCE.index("\nimport modal")


def test_every_local_entrypoint_checks_the_profile_first():
    """The forecast is priced against one workspace's rates, and the payload
    Volume lives in one workspace, so a run landing in the other both spends
    the wrong credit and finds no artifact."""
    for name in ("smoke", "forecast", "run"):
        first = code_of(name).splitlines()[0]
        assert first == "check_profile(profile)", \
            f"{name} starts with {first!r}, not check_profile(profile)"


def test_profiles_and_volume_names_are_the_house_ones():
    assert app.PROFILES == ("a-cheparukhin", "colleague")
    assert app.PAYLOAD_VOLUME == "pepstab-phasea-esm-payload"
    assert app.PAYLOAD_MOUNT == "/payload"
    assert app.app.name == "pepstab-phasea-esm"


def test_rows_payload_sends_an_allele_map_not_an_hla_column():
    """182 characters x 19,716 rows is 3.6 MB of repeated strings for 75
    distinct values, so ``hla_seq`` travels as a 75-entry map."""
    rows = app.load_rows()
    assert rows["n_rows"] == app.N_TRAIN_ROWS
    assert set(rows) == {"pair_id", "peptide", "allele", "cluster_id",
                         "y_log1p", "allele_hla", "n_rows"}
    assert len(rows["allele_hla"]) == 75
    frame = app.rows_to_frame(rows)
    assert len(frame) == app.N_TRAIN_ROWS
    assert frame.hla_seq.notna().all()
    assert frame.hla_seq.str.len().unique().tolist() == [182]


def test_rows_payload_is_training_only():
    """Phase A scores neither frozen split, and a container cannot score a row
    it was never handed."""
    from pepstab.data import load_with_splits

    rows = app.load_rows()
    df = load_with_splits()
    held = set(df[df.split != "train"].pair_id.astype(int))
    assert held
    assert not held & set(rows["pair_id"])
    assert "df.split == 'train'" in code_of("load_rows")


# --- cost model -----------------------------------------------------------

def test_forecast_refuses_to_invent_a_per_unit_time(tmp_path):
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"all_ok": True}))
    with pytest.raises(SystemExit, match="no seconds_per_unit"):
        app.seconds_from_smoke(smoke)


def test_forecast_refuses_a_failed_smoke(tmp_path):
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"seconds_per_unit": 200.0, "all_ok": False}))
    with pytest.raises(SystemExit, match="failed smoke"):
        app.seconds_from_smoke(smoke)


def test_forecast_accepts_a_passing_smoke(tmp_path):
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"seconds_per_unit": 212.5, "all_ok": True}))
    assert app.seconds_from_smoke(smoke) == 212.5


def test_cost_model_bills_reserved_cores_for_the_whole_wall():
    """Modal bills what a container reserves for as long as it lives, so an
    unfilled container is a real expense rather than something the arithmetic
    hides."""
    f = app.forecast_numbers(30, 215.0, containers=30, chunk=1)
    assert f["containers_active"] == 30
    assert f["n_chunks"] == 30
    assert f["chunk_fill"] == 1.0
    expected_wall = 215.0 + app.CONTAINER_STARTUP_S
    assert f["wall_h"] == pytest.approx(expected_wall / 3600)
    assert f["core_hours"] == pytest.approx(
        30 * expected_wall / 3600 * app.WORKER_CPU)
    assert f["usd"] == pytest.approx(
        f["core_hours"] * f["cpu_hour_rate_usd"]
        + f["mem_gib_hours"] * f["mem_gib_hour_rate_usd"])
    assert f["usd_with_margin"] == pytest.approx(f["usd"] * 1.25)
    assert f["margin"] == app.COST_MARGIN == 1.25
    # Priced under a dollar; a surprise here should be caught by reading.
    assert 0.2 < f["usd_with_margin"] < 0.5


def test_rates_come_from_the_repo_not_a_pricing_page():
    rates = app.load_rates()
    raw = json.loads((Path(app.REPO) / "reports" / "ectodomain_rates.json")
                     .read_text())
    assert rates["cpu_hour"] == float(raw["cpu_hour_cost"]) == 0.0473
    assert rates["mem_gib_hour"] == float(raw["mem_gib_hour_cost"]) == 0.008


def test_volume_cost_is_reported_beside_the_run_not_folded_into_it():
    f = app.forecast_numbers(30, 215.0)
    assert 0.005 < f["volume_usd_per_month"] < 0.05
    assert "volume_usd_per_month" in f and f["usd"] < 1.0


def test_check_profile_rejects_a_mismatch(monkeypatch):
    monkeypatch.setenv("MODAL_PROFILE", "colleague")
    with pytest.raises(AssertionError, match="MODAL_PROFILE"):
        app.check_profile("a-cheparukhin")
    assert app.check_profile("colleague") == "colleague"
    with pytest.raises(AssertionError, match="--profile must be one of"):
        app.check_profile("not-a-profile")


# --- the journal ----------------------------------------------------------

def test_journal_is_fsynced_before_anything_else():
    """Rows accumulate on the client, and the A1-A4 screen lost its gRPC
    connection mid-``.map()`` twice -- once capturing 8 units of 3,360.
    ``--detach`` keeps containers alive but nothing collects their results."""
    body = code_of("run")
    assert "journal.write(" in body
    assert "journal.flush()" in body
    assert "os.fsync(journal.fileno())" in body
    assert body.index("journal.write(") < body.index("results.extend(part)")
    assert ".partial.jsonl" in body


def test_an_existing_journal_without_resume_is_a_hard_stop():
    body = code_of("run")
    assert "--resume to continue it" in body
    assert "not a safe default" in body
    assert "if resume and partial.exists()" in body


def test_run_journals_the_full_return_including_predictions():
    """A unit's 30 fits are paid for by the time its numbers exist, so the
    predictions are journalled, not just a metric."""
    body = code_of("run_unit")
    for key in ("pair_id", "y_pred", "inner_oof_6", "members"):
        assert key in body
    # And the CSV deliberately leaves the bulk arrays to the journal.
    assert "y_pred" in code_of("_write_runs_csv")


def test_unit_runs_csv_uses_restval_for_a_ragged_header():
    """A failed unit has ``error`` and no metrics; a level-2 unit has
    ``inner_oof_6`` and a level-3 one does not. ``extrasaction="ignore"`` would
    silently drop a column only some units emit."""
    body = code_of("_write_runs_csv")
    assert "restval=''" in body or 'restval=""' in body
    assert "extrasaction" not in body


def test_every_row_carries_its_fold_under_an_unambiguous_name():
    """The A1-A3 screen's ``outer_fold`` is null on every row, the fold hiding
    in ``fold`` and in the ``key`` string. A consumer grouping on the null
    field collapses all five folds into one bucket and gets a single
    plausible-looking number that is wrong. Both the success and the failure
    path must name it, and nothing here may be called just ``fold``."""
    chunk = code_of("run_chunk")
    assert "row.setdefault('outer_fold', unit.get('outer_fold'))" in chunk
    assert "row.setdefault('inner_fold', unit.get('inner_fold'))" in chunk
    assert "if row.get('outer_fold') is None" in chunk
    assert "'outer_fold'" in code_of("run_unit")
    # A bare `fold` key would be the very ambiguity this guards against.
    for body in (chunk, code_of("run_unit")):
        assert "'fold'" not in body


def test_units_carry_their_fold_so_a_failed_row_still_has_one():
    for unit in app.enumerate_units():
        assert unit["outer_fold"] is not None
        assert unit["inner_fold"] is not None
        assert str(unit["outer_fold"]) in unit["key"]


def test_unit_refuses_a_pooled_fit_by_row_count():
    """The giveaway for a pooled fit is a row count, not a metric.

    Phase B fitted one calibration map on pooled ``inner_oof_30`` rows across
    all five outer folds and applied it to pooled outer-held predictions. A row
    is outer-held in exactly one fold and inner-OOF in the other four, so that
    fit used every scored row's own label -- and the leaked and correct
    versions agreed on Brier to four decimals. What exposed it was a log line
    saying 19,716 where a per-fold fit can only see 15,772. So the count is
    asserted directly, alongside the set-disjointness.
    """
    body = code_of("run_unit")
    assert "N_TRAIN_ROWS" in body and "N_HELD_ROWS_PER_OUTER_FOLD" in body
    assert "pooled split" in body
    assert "both fitted and predicted" in body
    # Checked as an identity, not against a constant: a constant (15,772) was
    # tried and failed four of five level-2 units, because the water-filled
    # folds are 3,944 / 3,943 x 4 rather than equal.
    assert "must partition it" in body
    assert sum(app.N_HELD_ROWS_PER_OUTER_FOLD) == app.N_TRAIN_ROWS


def test_declared_fold_sizes_match_the_real_folds_including_the_odd_one():
    """The folds are water-filled by cluster, so they are NOT equal.

    This is the test that would have caught the constant: fold 0 holds 3,944
    rows and folds 1-4 hold 3,943, so any single expected fit-row count is
    wrong for four of the five units. Asserted per fold, not as a set.
    """
    from pepstab import nested
    from pepstab.data import load_with_splits

    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    outer = nested.outer_folds(train)
    held = tuple(int((outer == f).sum()) for f in range(nested.OUTER_FOLDS))
    assert held == app.N_HELD_ROWS_PER_OUTER_FOLD == (3944, 3943, 3943, 3943, 3943)
    assert len(set(held)) == 2, "fold sizes are unequal; one constant cannot fit"
    assert sum(held) == app.N_TRAIN_ROWS


def test_required_metric_keys_include_coverage():
    """Prespecification §2 requires coverage next to every score; a score whose
    coverage is missing cannot be read against that rule."""
    assert "median_per_allele_spearman" in app.REQUIRED_METRIC_KEYS
    assert "n_alleles_eligible" in app.REQUIRED_METRIC_KEYS
    assert "eligible_row_coverage" in app.REQUIRED_METRIC_KEYS


def test_no_calibration_map_precedes_the_ranking_metric():
    """Isotonic regression preserves rank order but destroys rank resolution:
    it is a step function, and the frozen contract gives ties average ranks.
    Median per-allele Spearman is scored on raw predictions, always."""
    assert "Isotonic" not in SOURCE          # the sklearn class, not the prose
    assert "fit_transform" not in SOURCE
    body = code_of("run_unit")
    assert "nested.score_fold(unit['key'], predicted.allele, y_true, y_pred)" \
        in body, "the primary metric must be scored on raw predictions"


def test_retries_are_safe_because_a_unit_is_pure():
    """``run_unit`` reads only its arguments and the read-only Volume and
    writes nothing, so a retry recomputes rather than double-counting."""
    assert "retries=1" in decorator_of("run_chunk")
    body = code_of("run_unit")
    for mutator in ("to_csv", "write_text", "savez", "open("):
        assert mutator not in body, f"run_unit is not pure: {mutator}"
