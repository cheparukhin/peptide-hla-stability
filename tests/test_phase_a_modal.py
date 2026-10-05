"""Guards for the Phase A Modal execution layer.

Two groups, matching the two ways this layer could cost money or invalidate a
result without anything crashing:

1. **Runner discipline.** CPU-only by construction, no module-scope read of a
   repo file, a ``--profile`` that must match ``MODAL_PROFILE``, a free path
   that really is free, and a journal that survives a client death.
2. **Cost arithmetic.** Modal bills reserved cores and memory for a
   container's whole lifetime. The model is pinned against hand-computed
   values here because this project has already paid for an unmeasured cost
   assumption once (a 4.6x error) and for a forecaster that assumed perfect
   packing (an 88% overspend).

None of these tests imports ``pepstab.phase_a``: the science layer is written
concurrently, and a guard that cannot run until the thing it guards exists is
not a guard.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

APP = REPO / "modal_app" / "phase_a_experiments.py"
SCRIPT = REPO / "scripts" / "phase_a_forecast.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load(APP, "phase_a_experiments_app")


@pytest.fixture(scope="module")
def fc():
    return _load(SCRIPT, "phase_a_forecast")


# ---------------------------------------------------------------------------
# 1. runner discipline
# ---------------------------------------------------------------------------


def test_no_gpu_anywhere_in_the_app():
    """CPU only, asserted on the decorators rather than on the prose.

    A substring search would be satisfied by a docstring saying "no ``gpu=``",
    so this reads every ``@app.function`` and every ``with_options`` call and
    checks that none of them requests a GPU. The cost of getting this wrong is
    not only a wasted GPU-hour: Modal validates every function in an app at
    creation time (``modal_app/boltz_common.py:3-8``), so one ``gpu=`` here
    would make the whole CPU app undeployable in a workspace without GPU
    access.
    """
    tree = ast.parse(APP.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "gpu" and not (
                    isinstance(kw.value, ast.Constant) and kw.value.value is None
                ):
                    offenders.append(node.lineno)
    assert not offenders, f"a GPU is requested at line(s) {offenders}"


#: Calls that read a file's contents or enumerate a directory. ``exists()`` is
#: deliberately absent: it does not read anything and is safe at module scope.
_READ_CALLS = {
    "read_text", "read_bytes", "readlines", "read", "open",
    "read_csv", "read_json", "read_parquet", "load", "loads",
    "iterdir", "glob", "rglob",
}


def test_no_module_scope_read_of_a_repo_file():
    """A module-scope read kills every container on import, with no error.

    ``modal_app/proteinmpnn_scoring.py:97-105`` documents the incident: Modal
    imports the app module inside every container, where the repo does not
    exist, so a module-scope read of a repo file makes the container die before
    it executes a line of its own. It surfaces as "0 tasks" and nothing else,
    and neither a local forecast nor ``--dry-run`` can catch it, because both
    run where the file is present.

    ``if __name__ == "__main__"`` blocks are exempt: Modal imports the module
    under its own name, never as ``__main__``, so those statements cannot run
    in a container.
    """
    tree = ast.parse(APP.read_text())
    offenders = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If) and "__main__" in ast.dump(stmt.test):
            continue
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _READ_CALLS:
                offenders.append((node.lineno, name))
    assert not offenders, (
        f"module-scope file read(s) {offenders}: every container would die on "
        f"import. Read it lazily inside a function instead."
    )


def test_the_rates_are_read_lazily_inside_a_function():
    """The premise of the test above, asserted rather than assumed.

    The app does need the rates. The guard is only meaningful if the read
    exists somewhere, and that somewhere is a function body.
    """
    tree = ast.parse(APP.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "load_rates")
    assert "read_text" in ast.dump(fn)


def test_check_profile_is_the_first_statement_of_every_local_entrypoint():
    """Every entrypoint bills a workspace, so every entrypoint names it first.

    Phase A has no disjoint halves to protect -- what the guard protects is the
    budget, because the forecast is priced at one workspace's metered rates
    against one workspace's reserved credit.
    """
    tree = ast.parse(APP.read_text())
    entrypoints = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any("local_entrypoint" in ast.dump(d) for d in n.decorator_list)
    ]
    assert len(entrypoints) == 3, (
        f"expected smoke/forecast/run, found {[n.name for n in entrypoints]}"
    )
    assert {n.name for n in entrypoints} == {"smoke", "forecast", "run"}
    for fn in entrypoints:
        body = fn.body
        if (isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        first = body[0]
        assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), (
            f"{fn.name}() does not open with a call"
        )
        assert getattr(first.value.func, "id", None) == "check_profile", (
            f"{fn.name}() opens with {ast.dump(first.value.func)}, not "
            f"check_profile(profile)"
        )


def test_profile_must_match_modal_profile(app, monkeypatch):
    monkeypatch.setenv("MODAL_PROFILE", "colleague")
    with pytest.raises(AssertionError, match="MODAL_PROFILE"):
        app.check_profile("a-cheparukhin")
    assert app.check_profile("colleague") == "colleague"
    with pytest.raises(AssertionError, match="must be one of"):
        app.check_profile("nonsense")


def test_the_forecast_script_never_imports_modal():
    """Free by construction, like ``scripts/foldx_forecast.py``.

    ``modal run --dry-run`` still builds and validates the image, so "makes no
    remote call" and "free" are different claims. This path cannot start
    anything because it has no SDK to start it with.
    """
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "modal" not in imported, f"the free path imports modal: {sorted(imported)}"


def test_the_self_check_runs_before_the_modal_import():
    """``--self-check`` must not import the SDK at all.

    Checked on line numbers, because the guarantee is entirely about where the
    ``import modal`` statement sits relative to the ``--self-check`` exit.
    """
    tree = ast.parse(APP.read_text())
    imports = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Import)
               and any(a.name == "modal" for a in n.names)]
    assert imports, "the app does not import modal at all"
    guards = [
        n.lineno for n in tree.body
        if isinstance(n, ast.If) and "self-check" in ast.dump(n.test)
    ]
    assert guards, "no module-scope `--self-check` guard"
    assert max(guards) < min(imports), (
        f"the --self-check exit at line {max(guards)} sits below `import modal` "
        f"at line {min(imports)}, so the free path imports the SDK"
    )


def test_the_self_check_really_does_not_import_modal():
    """The line-number guard above, confirmed by running it."""
    import runpy

    for stale in ("modal", "modal.app"):
        sys.modules.pop(stale, None)
    argv = sys.argv
    sys.argv = ["phase_a_experiments.py", "--self-check"]
    try:
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(APP), run_name="__main__")
    finally:
        sys.argv = argv
    assert exit_info.value.code == 0
    assert "modal" not in sys.modules


def test_the_worker_is_cpu_and_memory_bounded_with_one_retry(app):
    """The shape the forecast is priced at, and the retry policy it assumes.

    ``retries=1`` is only safe because a unit is idempotent: ``run_unit`` is
    pure, reads nothing and writes nothing, so a retry recomputes rather than
    double-counting.
    """
    assert app.WORKER_CPU == 2.0
    assert app.WORKER_MEMORY == 4096
    assert app.WORKER_THREADS == int(app.WORKER_CPU)
    tree = ast.parse(APP.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_chunk")
    decorator = next(d for d in fn.decorator_list if isinstance(d, ast.Call))
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in decorator.keywords}
    assert kwargs["cpu"] == "WORKER_CPU"
    assert kwargs["memory"] == "WORKER_MEMORY"
    assert kwargs["max_containers"] == "MAX_CONTAINERS"
    assert kwargs["retries"] == "1"
    assert kwargs["timeout"] == "60 * 60"


def test_container_shape_is_a_parameter_not_a_constant():
    """Decorator values are defaults; the run sets the shape.

    ``modal_app/foldx_scoring.py:532-534`` is the pattern: ``with_options`` at
    call time, so a measured shape can be launched without editing the source.
    """
    source = APP.read_text()
    assert "with_options(" in source
    for flag in ("cpu: float", "memory_gib: float", "containers: int", "chunk: int"):
        assert flag in source, f"{flag} is not an entrypoint parameter"


def test_the_thread_count_is_pinned_in_the_image_and_recorded_on_every_row(app):
    """Thread count is part of the run record, not a performance knob.

    This project has measured that the same seed at a different BLAS thread
    count moves a single network by ~0.02 median Spearman. A row that does not
    say how many threads produced it cannot be reproduced.
    """
    source = APP.read_text()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert var in source, f"{var} is not pinned in the image"
    body = source.split("def run_chunk(")[1].split("\ndef ")[0]
    assert 'row["threads"] = WORKER_THREADS' in body, (
        "the thread count is pinned but not recorded on the returned rows"
    )
    assert app.WORKER_THREADS == 2


#: Distribution name in the image -> import name in the venv.
_PINNED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "catboost": "catboost",
}


def _image_pins() -> dict[str, str]:
    """The ``pip_install`` pins, read off the image declaration's AST."""
    tree = ast.parse(APP.read_text())
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "pip_install"
    )
    pins = {}
    for arg in call.args:
        assert isinstance(arg, ast.Constant), f"non-literal pin {ast.dump(arg)}"
        assert "==" in arg.value, f"{arg.value!r} is not pinned exactly"
        name, version = arg.value.split("==")
        pins[name] = version
    return pins


def test_the_image_pins_every_version_exactly():
    """A floating version would make a Phase A result unreproducible."""
    pins = _image_pins()
    assert set(pins) == set(_PINNED), f"image installs {sorted(pins)}"
    assert "download.pytorch.org/whl/cpu" in APP.read_text(), (
        "torch 2.14.1 from PyPI is the CUDA build and drags ~2.5 GB of nvidia "
        "wheels into a CPU-only image"
    )


def test_every_pin_matches_the_version_installed_locally():
    """Cloud and local must be the same learner, not merely the same code.

    The pepstab modules are copied into the image so that "the local pilot run
    and the cloud run cannot drift apart". A version skew in a learner or an
    array library is that same drift one layer down, and it would stay
    invisible until two runs disagreed -- so the pins are checked against what
    is installed rather than against a string written here, which is what let
    them drift in the first place.
    """
    import importlib

    pins = _image_pins()
    skew, absent = [], []
    for dist, module in _PINNED.items():
        try:
            installed = importlib.import_module(module).__version__
        except ImportError:
            absent.append(dist)
            continue
        if pins[dist] != installed:
            skew.append(f"{dist}: image {pins[dist]}, venv {installed}")
    assert not skew, (
        f"the image and the venv would run different code -- {skew}. The cloud "
        f"and local arms must be the same learner, not merely the same scripts."
    )
    if absent:
        pytest.skip(f"not installed locally, so unchecked: {absent}")


def test_the_image_carries_the_pepstab_modules_the_science_layer_needs(app):
    """Including ``augment``, which ``pepstab/__init__.py`` imports.

    Omitting a module ``__init__`` imports makes every ``import pepstab.*``
    fail inside the container -- the same silent class of failure as a
    module-scope repo read.
    """
    for module in ("__init__.py", "data.py", "features.py", "mlp.py",
                   "evaluation.py", "splits.py", "nested.py", "phase_a.py"):
        assert module in app.PEPSTAB_MODULES, f"{module} is not copied into the image"
    init = (REPO / "pepstab" / "__init__.py").read_text()
    tree = ast.parse(init)
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
        for alias in node.names
    }
    missing = sorted(m for m in imported if f"{m}.py" not in app.PEPSTAB_MODULES)
    assert not missing, (
        f"pepstab/__init__.py imports {missing}, which the image does not carry, "
        f"so `import pepstab.phase_a` would fail in every container"
    )
    assert "/root/pepstab/" in APP.read_text()
    assert 'sys.path.insert(0, "/root")' in APP.read_text()


def test_no_volumes_and_no_secrets():
    """Nothing is cached between containers, so a unit can be retried freely."""
    source = APP.read_text()
    assert "modal.Volume" not in source
    assert "modal.Secret" not in source


def test_smoke_is_three_to_five_units_and_calls_the_batch_function():
    """The invariant, and the thing it is supposed to be exercising.

    "No batch GPU job without a passing end-to-end pilot on 3-5 examples"
    applies to a new runner as well as a new model, and a smoke that runs a
    different function has not piloted the runner.
    """
    body = APP.read_text().split("def smoke(")[1].split("\n@app")[0]
    assert "3 <= n <= 5" in body
    assert "the invariant asks for 3-5 examples" in body
    assert "_shape(cpu, memory_gib, 1).remote(" in body, (
        "smoke must call the same run_chunk the batch does"
    )
    # Priced before anything remote happens, not after.
    assert body.index("at most $") < body.index(".remote(")
    assert "do not launch the batch" in body
    assert "Next: ::forecast" in body
    assert "phase_a_smoke_" in body
    assert "seconds_per_unit" in body and "peak_gib" in body


def test_smoke_refuses_a_null_metric(app):
    """A unit that returns ``ok=True`` and a null score is a failed smoke.

    The two coverage keys are required alongside the correlation because
    prespecification Sec. 2 says coverage is reported for every fold next to
    every score, and a score without it cannot be read against that rule.
    """
    assert app.REQUIRED_METRIC_KEYS == (
        "median_per_allele_spearman", "n_alleles_eligible", "eligible_row_coverage"
    )
    body = APP.read_text().split("def smoke(")[1].split("\n@app")[0]
    assert "REQUIRED_METRIC_KEYS" in body
    assert "nulls" in body and "all_ok" in body


def test_the_required_metric_keys_are_the_ones_run_unit_emits(app):
    """The contract, checked against the science layer rather than remembered.

    A key name that ``run_unit`` does not emit would make every smoke fail on a
    missing metric, which reads as a broken arm rather than as a typo here.
    """
    phase_a = pytest.importorskip("pepstab.phase_a")
    source = Path(phase_a.__file__).read_text()
    block = source.split("def _score_block(")[1].split("\ndef ")[0]
    for key in app.REQUIRED_METRIC_KEYS:
        assert f'"{key}"' in block, f"_score_block does not emit {key}"
    assert phase_a.SCORE_COLUMN == app.REQUIRED_METRIC_KEYS[0]


def test_smoke_uses_the_predeclared_subset_not_the_first_n_enumerated(app):
    """"The first n units" is several cells of one arm, not one per arm.

    A smoke that never touches A4 has not piloted the CatBoost ranker, which is
    the arm most likely to fail on a missing dependency.
    """
    body = APP.read_text().split("def smoke(")[1].split("\n@app")[0]
    assert "phase_a.smoke_units(wanted)" in body
    assert "units[:n]" not in body
    phase_a = pytest.importorskip("pepstab.phase_a")
    picked = phase_a.smoke_units(list(app.ARMS))
    assert 3 <= len(picked) <= 5
    assert {u["arm"] for u in picked} == set(app.ARMS)


def test_smoke_fails_on_a_thread_count_the_image_did_not_pin():
    """A dropped environment variable changes every float32 result silently.

    The image's ``.env`` is the only thing pinning the count, and nothing
    downstream would fail if it went missing -- the batch would simply produce
    results that cannot be reproduced against each other. The smoke is the
    place that refuses, because it is the gate in front of the batch.
    """
    chunk = APP.read_text().split("def run_chunk(")[1].split("\n\n\n")[0]
    assert "phase_a.n_threads()" in chunk
    assert 'row["thread_pin_ok"]' in chunk
    body = APP.read_text().split("def smoke(")[1].split("\n@app")[0]
    assert "unpinned" in body
    assert "not unpinned" in body, (
        "thread_pin_ok is recorded but does not gate the smoke verdict"
    )


def test_assessment_is_reachable_and_needs_its_selection_passed_in(app):
    """Finalists are chosen on the screen's inner folds, never by this runner.

    Assessment runs at 3 seeds x 5 inner folds so an arm's per-outer-fold
    ensemble is 30 members, the reference's arity. Running it at the screening
    arity would be measuring ensemble size, which is worth about +0.074 median
    Spearman here.
    """
    source = APP.read_text()
    assert 'stage: str = "screen"' in source
    with pytest.raises(SystemExit, match="--selected-from"):
        app._enumerate(None, ["A1"], "assess", "")
    with pytest.raises(SystemExit, match="screen' or 'assess'"):
        app._enumerate(None, ["A1"], "nonsense", "")
    with pytest.raises(AssertionError, match="assess only"):
        app._enumerate(None, ["A1"], "screen", "somewhere.json")


def test_the_stage_is_in_the_output_name_not_only_in_a_column():
    """Assessment must not be overwritten by a later screen.

    ``modal_app/foldx_scoring.py`` puts the repair arm in its output filename
    for the same reason: one name for a cheap arm and an expensive one means
    the cheap one silently wins.
    """
    body = APP.read_text().split("def run(")[1].split("\ndef ")[0]
    assert 'suffix = "" if stage == "screen"' in body
    assert "runs_csv" in body and "_write_csv(rows, runs_csv)" in body
    assert 'f"phase_a_runs{suffix}_{profile}.partial.jsonl"' in body


def test_forecast_refuses_an_unmeasured_default():
    """No default per-unit time, in either forecaster."""
    app_source = APP.read_text()
    assert "seconds_per_unit: float = 0.0" in app_source
    assert "--seconds-per-unit is required" in app_source
    assert "no default, deliberately" in app_source

    script_source = SCRIPT.read_text()
    assert '"--seconds-per-unit", type=float, default=0.0' in script_source
    assert "--seconds-per-unit or --measured-from is required" in script_source


def test_forecast_labels_measured_against_assumed():
    source = APP.read_text()
    assert "MEASURED" in source and "ASSUMED -- run ::smoke" in source
    assert "MEASURED" in SCRIPT.read_text()


def test_dry_run_cannot_reach_modal():
    """``run`` must return before any remote call when ``dry_run`` is set.

    The last ``--dry-run`` in this project was described as free and quietly
    started a container. Checked structurally, so it stays true if the body is
    edited.
    """
    tree = ast.parse(APP.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    remote_lines = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and n.attr in {"remote", "map", "starmap"}
    ]
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.If) and "dry_run" in ast.dump(n.test)]
    assert guards, "run() has no `if dry_run` guard"
    guard = guards[0]
    assert any(isinstance(n, ast.Return) for n in ast.walk(guard)), (
        "the `if dry_run` guard does not return, so execution falls through to "
        "the remote call below it"
    )
    guard_end = max(n.lineno for n in ast.walk(guard) if hasattr(n, "lineno"))
    assert remote_lines and min(remote_lines) > guard_end, (
        f"a remote call at line {min(remote_lines)} precedes the dry-run guard "
        f"ending at line {guard_end}: --dry-run would reach Modal"
    )


def test_run_journals_each_chunk_before_continuing():
    """A client death must not discard work that has already been billed.

    ``run`` accumulates rows on the client, so without a journal written as
    chunks land, a dropped connection loses completed units -- and ``--detach``
    does not help: it keeps the containers alive, but nothing is collecting
    their results.
    """
    body = APP.read_text().split("def run(")[1].split("\ndef ")[0]
    assert "partial.jsonl" in body, "no journal path in run()"
    # The journal write must precede the in-memory accumulation, so a crash
    # between the two cannot lose a chunk that was reported as done.
    assert body.index("journal.write(") < body.index("rows.extend(part)")
    assert "journal.flush()" in body
    assert "os.fsync(journal.fileno())" in body, (
        "a buffered journal that is never fsynced can lose its tail in exactly "
        "the crash it exists to survive"
    )


def test_run_refuses_to_silently_recompute_over_a_journal():
    body = APP.read_text().split("def run(")[1].split("\ndef ")[0]
    assert "--resume to continue" in body
    guard = body.split("elif partial.exists():")[1].split("\n\n")[0]
    assert "raise SystemExit" in guard


def test_run_writes_the_csv_over_the_union_of_keys(app, tmp_path):
    """Arms return different metric sets, so the header is the union.

    ``extrasaction="ignore"`` over the first row's keys would silently drop a
    column that only a later arm emits.
    """
    assert "DictWriter" in APP.read_text()
    out = tmp_path / "phase_a_runs.csv"
    app._write_csv(
        [{"key": "A0/f0", "median_per_allele_spearman": 0.69},
         {"key": "A4/f0", "mae_log1p_calibrated": 0.41}],
        out,
    )
    header = out.read_text().splitlines()[0].split(",")
    assert header == ["key", "median_per_allele_spearman", "mae_log1p_calibrated"]
    assert out.read_text().splitlines()[1] == "A0/f0,0.69,"


def test_the_app_is_named_for_the_stage(app):
    assert app.app.name == "pepstab-phasea-experiments"


def test_arms_match_the_prespecification(app):
    """``--arms`` rejects a typo on the client, not in a container.

    A0 and A5 are prespecified arms but not container units: A0 is a local
    reference reproduction and A5 is arithmetic over the rows this batch
    produces, so neither is dispatched here.
    """
    assert app.ARMS == ("A1", "A2", "A3", "A4")
    assert app._parse_arms("A1,A4") == ["A1", "A4"]
    assert app._parse_arms("") == list(app.ARMS)
    with pytest.raises(AssertionError, match="unknown arm"):
        app._parse_arms("A1,A9")
    prespec = (REPO / "reports" / "further_research"
               / "phase_a_prespecification.md").read_text()
    for arm in app.ARMS:
        assert f"### {arm} " in prespec, f"{arm} is not prespecified"
    for arm in ("A0", "A5"):
        assert arm not in app.ARMS, f"{arm} is not a container unit"


def test_the_arm_list_does_not_drift_from_the_science_layer(app):
    """Two lists of arm ids is two chances for them to disagree.

    The disagreement would surface as a container raising ``unknown arm`` after
    the batch had been launched and paid for, so ``_parse_arms`` cross-checks.
    """
    phase_a = pytest.importorskip("pepstab.phase_a")
    assert tuple(app.ARMS) == tuple(phase_a.ARMS)
    app._parse_arms("A1", phase_a)

    class Drifted:
        ARMS = ("A1", "A2")

    with pytest.raises(AssertionError, match="drifted apart"):
        app._parse_arms("A1", Drifted)


# ---------------------------------------------------------------------------
# 2. cost arithmetic
# ---------------------------------------------------------------------------
#
# Hand-computed, at the metered rates in reports/ectodomain_rates.json:
# $0.04730 per core-hour and $0.00800 per GiB-hour. Modal bills the cores and
# memory a container RESERVES for the whole of its life, so the cost is
# containers x cpu x wall, with the per-container startup charged explicitly.


def test_rates_are_the_metered_ones_from_the_repo(app):
    rates = app.load_rates()
    assert rates["cpu_hour"] == pytest.approx(0.04730)
    assert rates["mem_gib_hour"] == pytest.approx(0.00800)
    assert "ectodomain_rates.json" in str(app.RATES_PATH)


def test_forecast_matches_a_hand_computed_screening_budget(app):
    """12 configs x 2 seeds x 3 folds x 4 arms = 288 units at 120 s each.

    36 chunks of 8 over 40 requested containers, so 36 run one chunk each:
    busy 8 x 120 = 960 s, + 45 s startup = 1,005 s of wall.
    container-hours 36 x 1005 / 3600 = 10.05
    core-hours      10.05 x 2 = 20.10        GiB-hours 10.05 x 4 = 40.20
    usd             20.10 x 0.04730 + 40.20 x 0.00800 = 0.95073 + 0.32160
    """
    f = app.forecast_numbers(288, 120.0)
    assert f["n_chunks"] == 36
    assert f["containers_active"] == 36
    assert f["chunks_per_container"] == 1
    assert f["chunk_fill"] == pytest.approx(1.0)
    assert f["container_fill"] == pytest.approx(36 / 40)
    assert f["wall_h"] == pytest.approx(1005 / 3600)
    assert f["container_hours"] == pytest.approx(10.05)
    assert f["core_hours"] == pytest.approx(20.10)
    assert f["mem_gib_hours"] == pytest.approx(40.20)
    assert f["usd"] == pytest.approx(1.27233)
    assert f["usd_with_margin"] == pytest.approx(1.5904125)
    assert f["margin"] == 1.25


def test_cost_scales_with_reserved_cores_not_busy_ones(app):
    """The billing model, asserted as an identity rather than described."""
    f = app.forecast_numbers(288, 120.0)
    assert f["core_hours"] == pytest.approx(
        f["containers_active"] * f["cpu_per_container"] * f["wall_h"]
    )
    assert f["mem_gib_hours"] == pytest.approx(
        f["containers_active"] * f["memory_gib_per_container"] * f["wall_h"]
    )
    assert f["usd"] == pytest.approx(
        f["core_hours"] * f["cpu_hour_rate_usd"]
        + f["mem_gib_hours"] * f["mem_gib_hour_rate_usd"]
    )
    assert f["usd_with_margin"] == pytest.approx(f["usd"] * 1.25)


def test_forecast_reports_a_partial_tail_chunk_rather_than_dividing_it_away(app):
    """10 units at --chunk 4 is 3 chunks, the last of them half empty.

    3 containers x (4 x 60 s + 45 s) = 3 x 285 s = 0.2375 container-hours
    core-hours 0.475, GiB-hours 0.95
    usd        0.475 x 0.04730 + 0.95 x 0.00800 = 0.0224675 + 0.0076
    """
    f = app.forecast_numbers(10, 60.0, containers=3, chunk=4)
    assert f["n_chunks"] == 3
    assert f["chunk_fill"] == pytest.approx(10 / 12)
    assert f["container_fill"] == pytest.approx(1.0)
    assert f["container_hours"] == pytest.approx(0.2375)
    assert f["usd"] == pytest.approx(0.0300675)


def test_requesting_more_containers_than_chunks_buys_nothing(app):
    """Modal never starts them, so the forecast must not imply a speedup.

    The wall time and the bill are identical at 3 and at 10 requested
    containers, because there are only 3 chunks.
    """
    three = app.forecast_numbers(10, 60.0, containers=3, chunk=4)
    ten = app.forecast_numbers(10, 60.0, containers=10, chunk=4)
    assert ten["containers_active"] == 3
    assert ten["container_fill"] == pytest.approx(0.3)
    assert ten["wall_h"] == pytest.approx(three["wall_h"])
    assert ten["usd"] == pytest.approx(three["usd"])


def test_startup_is_charged_per_container_not_once(app):
    """Three containers pay three startups; one container pays one.

    1 container x (3 chunks x 4 x 60 s + 45 s) = 765 s = 0.2125 container-hours
    usd 0.425 x 0.04730 + 0.85 x 0.00800 = 0.0201025 + 0.0068
    """
    one = app.forecast_numbers(10, 60.0, containers=1, chunk=4)
    three = app.forecast_numbers(10, 60.0, containers=3, chunk=4)
    assert one["chunks_per_container"] == 3
    assert one["container_hours"] == pytest.approx(0.2125)
    assert one["usd"] == pytest.approx(0.0269025)
    # Fanning out costs more in total because every container pays startup,
    # and buys a 3x shorter wall.
    assert three["usd"] > one["usd"]
    assert three["wall_h"] < one["wall_h"]


def test_forecast_refuses_a_chunk_that_cannot_hold_a_unit(app):
    with pytest.raises(AssertionError, match="at least 1 unit"):
        app.forecast_numbers(288, 120.0, chunk=0)
    with pytest.raises(AssertionError, match="at least 1"):
        app.forecast_numbers(288, 120.0, containers=0)
    with pytest.raises(AssertionError, match="nothing to forecast"):
        app.forecast_numbers(0, 120.0)


def test_the_free_script_and_the_app_agree_to_the_cent(app, fc):
    """Two implementations of one bill is two chances to be wrong.

    They are separate files because one must not import ``modal``, so the
    arithmetic is pinned against each other on several shapes rather than
    trusted to stay in step.
    """
    for n, spu, cpu, mem, containers, chunk in (
        (288, 120.0, 2.0, 4.0, 40, 8),
        (288, 37.5, 1.0, 2.0, 10, 16),
        (720, 240.0, 4.0, 8.0, 80, 4),
        (10, 60.0, 2.0, 4.0, 3, 4),
    ):
        a = app.forecast_numbers(n, spu, cpu, mem, containers, chunk)
        b = fc.forecast(n, spu, cpu, mem, containers, chunk)
        assert a["usd"] == pytest.approx(b["usd"]), (n, spu, cpu, containers, chunk)
        assert a["wall_h"] == pytest.approx(b["wall_h"])
        assert a["container_hours"] == pytest.approx(b["container_hours"])
        assert a["usd_with_margin"] == pytest.approx(b["usd_with_margin"])


def test_the_free_script_defaults_to_the_enumerated_screen(fc):
    """3,360 units, and no smaller sketch of it offered as an alternative.

    Encoding and outer fold are unit axes, not overhead, so a forecast that
    leaves them out is not a cheaper plan -- it is the wrong number.
    """
    assert (fc.SCREEN_SEEDS, fc.SCREEN_INNER_FOLDS) == (2, 3)
    assert (fc.SCREEN_ENCODINGS, fc.OUTER_FOLDS) == (2, 5)
    assert fc.enumerated_screen_units() == 3360
    assert not hasattr(fc, "SCREEN_ARMS"), (
        "the 288-unit sketch was a superseded orchestration estimate, not a "
        "plan figure; it must not be reachable as a default"
    )
    prespec = (REPO / "reports" / "further_research"
               / "phase_a_prespecification.md").read_text()
    assert "**three** inner folds × two paired seeds" in prespec
    assert "12 configurations" in prespec


def test_no_learner_or_objective_family_exceeds_twelve_configurations(fc):
    """The Sec. 3.3 ceiling holds per family, not per arm.

    56 configurations in total reads as a breach until the families are
    counted, so the decomposition in the app's module docstring is pinned here
    against ``arm_configs``. A family above 12 is a real extension and would
    owe a Sec. 7 log entry.
    """
    phase_a = pytest.importorskip("pepstab.phase_a")
    families: dict[str, int] = {}
    for arm in phase_a.ARMS:
        for config in phase_a.arm_configs(arm):
            if arm == "A3":
                family = f"A3:{config['loss']}"
            elif arm == "A4":
                family = f"A4:{config['objective']}"
            else:
                family = arm
            families[family] = families.get(family, 0) + 1
    assert families == {
        "A1": 12,
        "A2": 4,
        "A3:mse": 4,
        "A3:mse+rank": 12,
        "A4:YetiRank": 12,
        "A4:PairLogit": 12,
    }
    over = {k: v for k, v in families.items() if v > 12}
    assert not over, f"families above the Sec. 3.3 ceiling of 12: {over}"
    assert sum(families.values()) == sum(fc.ARM_CONFIGS.values()) == 56


def test_the_module_docstring_records_why_the_outer_fold_axis_is_not_optional():
    """The 5x is a leakage guard, so the reason has to travel with the code.

    Selecting hyperparameters once on outer fold 0's fitting partition and
    reusing them across all five folds leaks: fold 1's held-out rows sit inside
    fold 0's fitting partition.
    """
    doc = ast.get_docstring(ast.parse(APP.read_text())) or ""
    assert "independently" in doc and "outer fitting partition" in doc
    assert "leak" in doc
    for family in ("YetiRank", "PairLogit"):
        assert family in doc, "the per-family config accounting is not recorded"
    assert "Sec. 7" in doc or "Sec. 3.3" in doc


def test_the_free_scripts_unit_count_matches_what_is_enumerated(fc):
    """Arithmetic in a standalone script against the enumerator that runs."""
    phase_a = pytest.importorskip("pepstab.phase_a")
    assert fc.enumerated_screen_units() == len(
        phase_a.enumerate_units(list(phase_a.ARMS))
    )
    assert fc.ARM_CONFIGS == {
        arm: len(phase_a.arm_configs(arm)) for arm in phase_a.ARMS
    }


def test_the_shared_defaults_do_not_drift(app, fc):
    """The two files price the same shape, so the shape has to match."""
    assert fc.WORKER_CPU == app.WORKER_CPU
    assert fc.WORKER_MEMORY_GIB == app.WORKER_MEMORY / 1024
    assert fc.MAX_CONTAINERS == app.MAX_CONTAINERS
    assert fc.DEFAULT_CHUNK == app.DEFAULT_CHUNK
    assert fc.CONTAINER_STARTUP_S == app.CONTAINER_STARTUP_S
    assert fc.COST_MARGIN == app.COST_MARGIN == 1.25
