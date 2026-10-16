"""Guards on the A0 Linux runner (`modal_app/phase_a_a0_modal.py`).

Everything here runs locally, starts no container and costs nothing. The
runner exists because A0 is the only arm in the study fitted on macOS while
every other arm -- the A1-A3 screen, the A5/ESM arm, Phase B's arms, and the
forthcoming assessment -- runs in Linux containers, and Phase B measured
macOS/Accelerate arm64 against Linux/OpenBLAS x86-64 at max|diff| 1.82e-01 on
identical code.

Five kinds of check:

- **Contract parity with the macOS A0** -- same unit set, same axis product,
  same sentinels, and ``*_linux`` output names that cannot overwrite it.
- **The budget is inherited, not retyped.** ``run_unit`` must call
  ``pepstab.nested.fit_mlp_ensemble`` rather than mirror it; unlike the ESM
  arm, A0's features are exactly what that function's ``input_set`` routes to.
  Asserted *by execution* against a measured tolerance, not by reading source.
- **CPU only and thread-pinned**, read from the decorators and the images.
- **Container hygiene** -- no repo read at module scope, folds named
  unambiguously, the pooled-fit guard present.
- **The free paths stay free** -- ``--self-check`` above ``import modal``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modal_app import phase_a_a0_modal as app  # noqa: E402
from pepstab import nested  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.features import ENCODINGS  # noqa: E402
from pepstab.mlp import MLPConfig  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

SOURCE = Path(app.__file__).read_text()
MODULE = ast.parse(SOURCE)


def code_of(name: str) -> str:
    """One top-level function's code, docstring and comments stripped.

    This file's own prose quotes the hazards it guards against, so matching raw
    source would let a comment satisfy a test.
    """
    for node in MODULE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body = body[1:]
            return "\n".join(ast.unparse(s) for s in body)
    raise AssertionError(f"{name} is not a top-level function of the runner")


def decorator_of(name: str) -> str:
    for node in MODULE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(ast.unparse(d) for d in node.decorator_list)
    raise AssertionError(f"{name} is not a top-level function")


@pytest.fixture(scope="module")
def train():
    df = load_with_splits()
    return df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)


# --- contract parity with the macOS A0 ------------------------------------

def test_unit_set_and_arity_match_the_macos_a0():
    units = app.enumerate_units()
    assert len(units) == 30
    assert sum(1 for u in units if u["level"] == 2) == 5
    assert sum(1 for u in units if u["level"] == 3) == 25
    cfg = app.member_config()
    per_outer = (nested.ASSESS_INNER_FOLDS * len(cfg["encodings"])
                 * len(cfg["seeds"]))
    assert per_outer == 30
    assert len(units) * per_outer == 900
    assert tuple(cfg["encodings"]) == ENCODINGS == ("onehot", "blosum")
    assert tuple(cfg["seeds"]) == tuple(SEEDS) == (0, 1, 2)


def test_member_recipe_is_the_stage_2_selection_not_a_retype():
    """``REFERENCE_MLP`` and ``SELECTED`` are the authorities, and
    ``member_config`` raises on drift between them rather than reporting it."""
    cfg = app.member_config()
    assert cfg["selected"] == {"onehot": [[256, 64], 1e-05],
                               "blosum": [[256, 64], 0.001]}
    assert (cfg["max_epochs"], cfg["patience"]) == (MAX_EPOCHS, PATIENCE)
    assert cfg["lr"] == nested.REFERENCE_MLP.lr
    assert cfg["batch_size"] == nested.REFERENCE_MLP.batch_size
    assert cfg["input_set"] == app.INPUT_SET == "pep_pseudo"


def test_output_names_cannot_overwrite_the_macos_a0():
    """The macOS files are the study's shipped reference and stay untouched."""
    body = code_of("assemble")
    assert "phase_a_seq_oof_cluster{suffix}.csv" in body
    assert "phase_a_a0_runs{suffix}.csv" in body
    assert "phase_a_a0_provenance{suffix}.json" in body
    assert app.assemble.__defaults__[1] == "_linux"
    # The exact macOS names must never be written by this file.
    for shipped in ('"phase_a_seq_oof_cluster.csv"',
                    '"phase_a_a0_runs.csv"',
                    '"phase_a_a0_provenance.json"'):
        assert shipped not in SOURCE


def test_provenance_records_platform_blas_and_threads_as_first_class():
    """The run exists because of these, so they are not footnotes."""
    body = code_of("assemble")
    assert "'platform'" in body
    for field in ("'blas'", "'threads'", "'where'", "'contrast'"):
        assert field in body
    assert "does_not_replace" in body


def test_sentinels_match_the_shipped_a0():
    assert app.NO_SUB_FOLD == -1
    assert app.SENSITIVITY_MIN_ROWS == 50
    assert nested.MIN_ROWS_PER_ALLELE_FOLD == 20
    assert app.N_TRAIN_ROWS == 19_716


# --- the budget is inherited, by execution --------------------------------

def test_run_unit_calls_fit_mlp_ensemble_rather_than_mirroring_it():
    body = code_of("run_unit")
    assert "nested.fit_mlp_ensemble(" in body
    assert body.count("nested.fit_mlp_ensemble(") == 2  # one per level


def test_run_unit_reproduces_fit_mlp_ensemble_within_a_measured_tolerance(train):
    """Parity asserted against a **measured** tolerance, not an optimistic one.

    Phase B's own parity check used 1e-9 and would have failed on a 1.18e-06
    difference from round-tripping member predictions through Python lists --
    same networks, same seeds, different float32 summation order -- and would
    have reported "the budget is not inherited", which is false. This runner
    has the same list round-trip, so the bound here is set from what the
    difference actually is.

    Measured on this pair at the reduced config below: ``max|diff|`` is 0.0,
    because the member mean is taken by ``fit_mlp_ensemble`` *before* the
    conversion and ``float()`` on a float64 is exact. The bound is set to
    1e-9 only after observing that, and the observation is asserted alongside
    it so a future change that reintroduces a re-average is caught rather than
    absorbed.
    """
    split = next(iter(nested.iter_outer(train, 2)))
    fit_rows = train[split.fit_mask].reset_index(drop=True).head(900)
    inner = nested.inner_folds(fit_rows, 2)
    held = train[split.held_mask].reset_index(drop=True).head(200)
    selected = {"onehot": ((16,), 1e-3)}
    base = MLPConfig(hidden=(16,), l2=1e-3, max_epochs=3, patience=2)

    direct = nested.fit_mlp_ensemble(fit_rows, [held], inner, ("onehot",), (0,),
                                     selected, input_set="pep_pseudo",
                                     base_config=base)
    # The runner's boundary crossing, in isolation: the member mean, then the
    # list conversion the journal round-trips through.
    crossed = json.loads(json.dumps([float(v) for v in direct["predictions"][0]]))

    diff = np.abs(np.asarray(crossed) - direct["predictions"][0]).max()
    assert diff == 0.0, (
        f"the list round trip now moves predictions by {diff:.3e}; it used to "
        f"be exact. If a re-average has been introduced, raise the bound from "
        f"a measurement rather than deleting this assertion.")
    assert diff < 1e-9


def test_a_reaverage_after_the_round_trip_would_be_caught():
    """Shows the failure mode the measured bound is protecting against.

    Averaging members *after* crossing the boundary rather than before changes
    the summation order, which is exactly the 1.18e-06 Phase B saw.
    """
    rng = np.random.default_rng(0)
    members = [rng.normal(size=500).astype(np.float32) for _ in range(30)]
    before = np.mean(members, axis=0)
    after = np.mean([np.asarray(json.loads(json.dumps(m.tolist())))
                     for m in members], axis=0)
    moved = np.abs(before - after).max()
    # Small, but far above an optimistic 1e-9 threshold.
    assert moved > 1e-9, moved


# --- CPU only, thread pinning ---------------------------------------------

def test_no_function_declares_a_gpu():
    decorated = [n.name for n in MODULE.body
                 if isinstance(n, ast.FunctionDef) and n.decorator_list]
    assert "run_chunk_t1" in decorated and "run_chunk_t2" in decorated
    for name in decorated:
        assert "gpu" not in decorator_of(name), name


def test_one_image_per_thread_count_pins_in_the_image_env():
    """A worker cannot pin threads late enough: the BLAS pool is sized when
    numpy is imported and importing the science layer imports numpy. So the
    pin is baked per image, which is why there are two."""
    assert sorted(app.IMAGES) == [1, 2] == sorted(app.THREAD_CHOICES)
    assert app.DEFAULT_THREADS == 2
    assert app.THREAD_ENV_VARS == (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    assert "THREAD_ENV_VARS" in code_of("_with_threads")
    assert "IMAGES[1]" in decorator_of("run_chunk_t1")
    assert "IMAGES[2]" in decorator_of("run_chunk_t2")


def test_the_worker_also_pins_threads_before_numpy_is_imported():
    """The image ``.env`` alone is not trusted: a ``--threads 1`` run of this
    app dispatched correctly to ``run_chunk_t1`` and still reported
    ``OMP_NUM_THREADS=2`` in all five containers.

    The worker-side pin is only valid because nothing imports numpy at module
    scope, so the BLAS pool has not been sized when it runs. That property is
    asserted here on the module, and in the worker at run time -- otherwise
    the assignment would be a no-op that looked like a fix.
    """
    import ast as _ast

    top = [n for n in MODULE.body if isinstance(n, (_ast.Import, _ast.ImportFrom))]
    names = [a.name if isinstance(n, _ast.Import) else (n.module or "")
             for n in top for a in n.names]
    assert "numpy" not in names and "pandas" not in names, names
    body = code_of("_run_chunk_body")
    assert "os.environ[var] = str(threads)" in body
    assert "numpy_preimported" in body
    assert "already sized" in body
    # And the pin must be set before anything heavy is touched.
    assert body.index("os.environ[var] = str(threads)") < body.index("run_unit(")


def test_cpu_fingerprint_is_recorded_per_unit():
    """Two runs of the same five units at the same nominal settings came back
    bit-identical on three folds and different on two, by up to 1.9e-01. The
    host is therefore a run variable and belongs in the record."""
    body = code_of("_run_chunk_body")
    assert "fingerprint = _cpu_fingerprint()" in body
    assert "row.update(fingerprint)" in body
    fp = code_of("_cpu_fingerprint")
    assert "cpu_model" in fp and "numpy_blas" in fp


def test_thread_count_is_recorded_and_mismatch_is_visible():
    body = code_of("_run_chunk_body")
    assert "threads_observed" in body and "thread_pin_ok" in body
    assert "OMP_NUM_THREADS" in body


def test_shape_refuses_an_unsupported_thread_count():
    with pytest.raises(AssertionError, match="--threads must be one of"):
        app._shape(4, 2.0, 2.0, 1)


def test_journal_and_smoke_names_carry_the_thread_count():
    """The threads=1 decomposition arm must not resume from or overwrite the
    threads=2 batch."""
    assert "_t{threads}.partial.jsonl" in code_of("run")
    assert "_t{threads}.json" in code_of("smoke")


def test_image_omits_sklearn_torch_and_catboost():
    """A0's features are one-hot and BLOSUM from ``build_features``."""
    for absent in ('"scikit-learn', '"torch', '"catboost'):
        assert absent not in SOURCE
    assert '"numpy==2.5.3"' in SOURCE and '"scipy==1.18.1"' in SOURCE
    assert "esm.py" not in app.PEPSTAB_MODULES


# --- container hygiene ----------------------------------------------------

def test_no_repo_file_is_read_at_module_scope():
    head = SOURCE.split("def load_rates")[0]
    for pattern in ("read_text()", "json.load", "pd.read_csv", "open("):
        assert pattern not in head, f"module scope reads a file: {pattern}"
    assert "RATES_PATH.read_text()" in code_of("load_rates")


def test_self_check_is_above_the_modal_import():
    assert SOURCE.index("def self_check") < SOURCE.index("\nimport modal")
    assert SOURCE.index('"--self-check" in sys.argv') < SOURCE.index("\nimport modal")


def test_every_local_entrypoint_checks_the_profile_first():
    for name in ("smoke", "forecast", "run"):
        first = code_of(name).splitlines()[0]
        assert first == "check_profile(profile)", f"{name}: {first!r}"


def test_every_row_carries_its_fold_under_an_unambiguous_name():
    """The A1-A3 screen's ``outer_fold`` is null on every row with the fold
    hiding in ``fold``; a consumer grouping on it merges all five folds into
    one bucket and gets a single plausible-looking number that is wrong."""
    body = code_of("_run_chunk_body")
    assert "row.setdefault('outer_fold', unit.get('outer_fold'))" in body
    assert "if row.get('outer_fold') is None" in body
    assert "'fold'" not in body and "'fold'" not in code_of("run_unit")


def test_pooled_fit_guard_is_an_identity_not_a_literal_count():
    """A literal count was tried on the A5/ESM arm and failed four of five
    level-2 units: the water-filled folds are 3,944 / 3,943 x 4."""
    body = code_of("run_unit")
    assert "must partition it" in body
    assert "pooled split" in body
    assert "both fitted and\\npredicted" in body or "both fitted and " in body
    assert sum(app.N_HELD_ROWS_PER_OUTER_FOLD) == app.N_TRAIN_ROWS


def test_declared_fold_sizes_match_the_real_folds(train):
    outer = nested.outer_folds(train)
    held = tuple(int((outer == f).sum()) for f in range(nested.OUTER_FOLDS))
    assert held == app.N_HELD_ROWS_PER_OUTER_FOLD == (3944, 3943, 3943, 3943, 3943)
    assert len(set(held)) == 2, "unequal by design; one constant cannot fit"


def test_rows_payload_is_training_only_and_sends_an_allele_map():
    rows = app.load_rows()
    assert rows["n_rows"] == app.N_TRAIN_ROWS
    assert set(rows["allele_maps"]) == {"hla_pseudoseq"}
    assert len(rows["allele_maps"]["hla_pseudoseq"]) == 75
    df = load_with_splits()
    held = set(df[df.split != "train"].pair_id.astype(int))
    assert held and not held & set(rows["pair_id"])
    frame = app.rows_to_frame(rows)
    assert len(frame) == app.N_TRAIN_ROWS
    assert frame.hla_pseudoseq.str.len().unique().tolist() == [34]


def test_row_payload_carries_every_column_build_features_needs():
    """The test that would have caught the smoke's failure.

    The first row payload sent ``hla_seq`` (the 182-residue domain) when
    ``pep_pseudo`` needs ``hla_pseudoseq`` (the 34 contact residues), and every
    unit died on a bare ``KeyError`` two seconds in. The columns are now taken
    from ``INPUT_SETS`` rather than chosen by hand, and this builds the real
    feature matrix instead of trusting that they are present.
    """
    from pepstab.features import INPUT_SETS, build_features, n_features

    frame = app.rows_to_frame(app.load_rows())
    assert set(INPUT_SETS[app.INPUT_SET]).issubset(frame.columns)
    X = build_features(frame.head(64), app.INPUT_SET, "onehot")
    assert X.shape == (64, n_features(app.INPUT_SET)) == (64, 860)
    for encoding in ENCODINGS:
        assert build_features(frame.head(8), app.INPUT_SET, encoding).shape[0] == 8


def test_rows_to_frame_names_a_missing_column_rather_than_keyerroring():
    rows = app.load_rows()
    stripped = {**rows, "allele_maps": {}}
    with pytest.raises(AssertionError, match="hla_pseudoseq"):
        app.rows_to_frame(stripped)


def test_no_volume_is_used():
    """Unlike the A5/ESM arm there is nothing bulk to stage: ``build_features``
    is row-local, so the rows travel as a function argument."""
    assert "modal.Volume" not in SOURCE
    assert "volumes=" not in SOURCE
    for name in ("run_chunk_t1", "run_chunk_t2"):
        assert "volume" not in decorator_of(name).lower()


def test_no_calibration_map_precedes_the_ranking_metric():
    assert "Isotonic" not in SOURCE and "fit_transform" not in SOURCE
    body = code_of("run_unit")
    assert "nested.score_fold(unit['key'], predicted.allele, y_true, y_pred)" \
        in body


# --- cost model -----------------------------------------------------------

def test_forecast_refuses_to_invent_a_per_unit_time(tmp_path):
    smoke = tmp_path / "s.json"
    smoke.write_text(json.dumps({"all_ok": True}))
    with pytest.raises(SystemExit, match="no seconds_per_unit"):
        app.seconds_from_smoke(smoke)
    smoke.write_text(json.dumps({"seconds_per_unit": 1.0, "all_ok": False}))
    with pytest.raises(SystemExit, match="failed smoke"):
        app.seconds_from_smoke(smoke)


def test_cost_model_bills_reserved_cores_for_the_whole_wall():
    f = app.forecast_numbers(30, 300.0, containers=30, chunk=1)
    expected_wall = 300.0 + app.CONTAINER_STARTUP_S
    assert f["core_hours"] == pytest.approx(
        30 * expected_wall / 3600 * app.WORKER_CPU)
    assert f["usd_with_margin"] == pytest.approx(f["usd"] * 1.25)
    assert 0.2 < f["usd_with_margin"] < 1.0


def test_levels_filter_selects_the_decomposition_arm():
    """Level-3 units feed only ``inner_oof_30`` and never a fold score, so
    ``--levels 2`` reproduces every per-fold median for a sixth of the batch."""
    only2 = app.enumerate_units((2,))
    assert len(only2) == 5
    assert {u["level"] for u in only2} == {2}
    assert len(app.enumerate_units((3,))) == 25


def test_check_profile_rejects_a_mismatch(monkeypatch):
    monkeypatch.setenv("MODAL_PROFILE", "a-cheparukhin")
    with pytest.raises(AssertionError, match="MODAL_PROFILE"):
        app.check_profile("colleague")
    assert app.check_profile("a-cheparukhin") == "a-cheparukhin"
