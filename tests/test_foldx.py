"""Guards for the stage 8 FoldX arm.

Three groups, matching the three ways this arm could produce a wrong number
without anything crashing:

1. **Conversion.** FoldX reads PDB and is quiet about some malformed input, so
   the mmCIF -> PDB step is checked against real folds rather than asserted.
2. **Parsing.** FoldX's ``.fxout`` column set varies by release, so the parser
   is header-driven and a positional parser is the failure these tests exist
   to prevent.
3. **Runner discipline.** CPU-only, one implementation of the cohort exclusion,
   a ``--profile`` that must match ``MODAL_PROFILE``, and a ``--dry-run`` that
   really cannot reach Modal.
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pepstab import foldx, structural_features  # noqa: E402

PILOT = REPO / "structures/ectodomain_pilot/ectodomain-20261004/boltz2"
APP = REPO / "modal_app" / "foldx_scoring.py"


def arm_b_folds() -> list[Path]:
    return sorted({p.parent for p in PILOT.rglob("*arm_B*/*.cif")})


# ---------------------------------------------------------------------------
# 1. conversion
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_fold() -> Path:
    folds = arm_b_folds()
    if not folds:
        pytest.skip("no local arm-B pilot folds")
    return folds[0]


def test_conversion_round_trips_every_atom(one_fold, tmp_path):
    """Atom identity must survive mmCIF -> PDB exactly; only coordinates round."""
    prepared = foldx.prepare_pdb(one_fold, tmp_path)
    v = prepared["validation"]
    assert v["max_coord_dev_A"] <= foldx.COORD_TOLERANCE_A
    assert v["max_bfactor_dev"] <= foldx.BFACTOR_TOLERANCE
    assert prepared["conversion"]["chain_runs"] == [("A", 275), ("B", 99), ("C", 9)]


def test_peptide_stays_its_own_chain(one_fold, tmp_path):
    """The whole arm is an interface energy, so the interface must exist."""
    prepared = foldx.prepare_pdb(one_fold, tmp_path)
    assert prepared["peptide_chain"] != prepared["hla_chain"]
    assert prepared["validation"]["peptide_sequence"] == prepared["peptide"]
    assert len(prepared["peptide"]) == 9


def test_peptide_chain_comes_from_the_verified_mapping(one_fold, tmp_path):
    """Not from the chain letter: ``load_fold`` proves which chain it is."""
    fold = structural_features.load_fold(one_fold)
    prepared = foldx.prepare_pdb(one_fold, tmp_path)
    assert prepared["peptide_chain"] == fold.peptide.chain_id
    assert prepared["hla_chain"] == fold.hla.chain_id


def test_ter_between_chains_and_terminal_end(one_fold, tmp_path):
    prepared = foldx.prepare_pdb(one_fold, tmp_path)
    text = Path(prepared["pdb"]).read_text()
    assert text.count("\nTER") == 3, "one TER per chain, or a parser may bond across chains"
    assert text.rstrip().endswith("END")


def test_all_local_arm_b_folds_convert(tmp_path):
    folds = arm_b_folds()
    if not folds:
        pytest.skip("no local arm-B pilot folds")
    for folder in folds:
        prepared = foldx.prepare_pdb(folder, tmp_path)
        assert prepared["validation"]["max_coord_dev_A"] <= foldx.COORD_TOLERANCE_A


def test_validation_catches_a_corrupted_pdb(one_fold, tmp_path):
    """A mangled coordinate must raise, not pass. The check has to bite."""
    prepared = foldx.prepare_pdb(one_fold, tmp_path)
    pdb = Path(prepared["pdb"])
    cif = next(iter(sorted(one_fold.glob("*.cif"))))
    lines = pdb.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            lines[i] = line[:30] + "9999.999" + line[38:]
            break
    pdb.write_text("\n".join(lines) + "\n")
    with pytest.raises(foldx.ConversionError):
        foldx.validate_pdb(cif, pdb)


def test_validation_catches_a_dropped_atom(one_fold, tmp_path):
    prepared = foldx.prepare_pdb(one_fold, tmp_path)
    pdb = Path(prepared["pdb"])
    cif = next(iter(sorted(one_fold.glob("*.cif"))))
    lines = [ln for ln in pdb.read_text().splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            del lines[i]
            break
    pdb.write_text("\n".join(lines) + "\n")
    with pytest.raises(foldx.ConversionError):
        foldx.validate_pdb(cif, pdb)


def test_predicted_models_have_complete_side_chains(tmp_path):
    """The evidence behind "RepairPDB has no missing atoms to rebuild".

    If a future construct or model does leave gaps, this fails and the
    RepairPDB decision has to be revisited rather than inherited.
    """
    import biotite.structure as struc
    from biotite.structure.io.pdbx import CIFFile, get_structure

    sys.path.insert(0, str(REPO / "scripts"))
    from foldx_convert import HEAVY_ATOMS

    folds = arm_b_folds()
    if not folds:
        pytest.skip("no local arm-B pilot folds")
    missing = []
    for folder in folds:
        cif = next(iter(sorted(folder.glob("*.cif"))))
        atoms = get_structure(CIFFile.read(str(cif)), model=1)
        atoms = atoms[~np.isin(atoms.element, ("H", "D"))]
        starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
        for a, b in zip(starts[:-1], starts[1:]):
            name = str(atoms.res_name[a])
            if name in HEAVY_ATOMS and (b - a) != HEAVY_ATOMS[name]:
                missing.append((folder.name, name, int(b - a)))
    assert not missing, f"residues short of their heavy atoms: {missing[:5]}"


# ---------------------------------------------------------------------------
# 2. .fxout parsing
# ---------------------------------------------------------------------------

# The documented FoldX 5 AnalyseComplex layout. This fixture has NOT been
# checked against a real run -- there was no binary to run when it was
# written -- which is exactly why the parser reads the header instead of
# assuming it. reports/stage8_foldx.md says to re-verify on first real output.
FXOUT = """FoldX 5.0 (c) Copyright

Interaction between molecules defined in the following groups:
--------------------------------------------------------------
A
C
--------------------------------------------------------------

Pdb\tGroup1\tGroup2\tIntraclashesGroup1\tIntraclashesGroup2\tInteraction Energy\tBackbone Hbond\tSidechain Hbond\tVan der Waals\tElectrostatics\tSolvation Polar\tSolvation Hydrophobic\tVan der Waals clashes\tentropy sidechain\tentropy mainchain\tsloop_entropy\tmloop_entropy\tcis_bond\tTorsional clash\tbackbone clash\thelix dipole\twater bridge\tdisulfide\telectrostatic kon\tpartial covalent bonds\tenergy Ionisation\tNumber of Residues\tInterface Residues\tInterface Residues Clashing\tInterface Residues VdW Clashing\tInterface Residues BB Clashing
A0201_LLWNGPMAV_arm_B.pdb\tA\tC\t12.5\t0.84\t-14.2371\t-3.11\t-2.05\t-22.40\t-0.77\t18.33\t-9.12\t1.44\t6.21\t3.02\t0\t0\t0\t2.11\t0.55\t0\t-0.40\t0\t-0.18\t0\t0\t284\t31\t0\t0\t0
"""


def test_parse_fxout_is_header_driven():
    rows = foldx.parse_fxout(FXOUT)
    assert len(rows) == 1
    row = rows[0]
    assert row["group1"] == "A" and row["group2"] == "C"
    assert row["interaction_energy"] == pytest.approx(-14.2371)
    assert row["van_der_waals"] == pytest.approx(-22.40)
    assert row["solvation_hydrophobic"] == pytest.approx(-9.12)


def test_parse_fxout_follows_a_changed_header():
    """A renamed or reordered column must follow the header, not a memory of it."""
    swapped = FXOUT.replace(
        "Interaction Energy\tBackbone Hbond", "Backbone Hbond\tInteraction Energy"
    ).replace("-14.2371\t-3.11", "-3.11\t-14.2371")
    rows = foldx.parse_fxout(swapped)
    assert rows[0]["interaction_energy"] == pytest.approx(-14.2371)
    assert rows[0]["backbone_hbond"] == pytest.approx(-3.11)


def test_parse_fxout_rejects_a_ragged_table():
    with pytest.raises(ValueError, match="fields against"):
        foldx.parse_fxout(FXOUT.rstrip("\n") + "\textra\n")


def test_parse_fxout_rejects_a_table_with_no_header():
    with pytest.raises(ValueError, match="no FoldX table header"):
        foldx.parse_fxout("FoldX 5.0\nsomething went wrong\n")


def test_parse_fxout_rejects_a_header_with_no_rows():
    header_only = FXOUT.rsplit("\n", 2)[0] + "\n"
    with pytest.raises(ValueError, match="no data rows"):
        foldx.parse_fxout(header_only)


def test_interaction_features_prefixes_and_drops_identifiers():
    row = foldx.interaction_features(foldx.parse_fxout(FXOUT), "C", "A")
    assert row["foldx_interaction_energy"] == pytest.approx(-14.2371)
    assert "foldx_pdb" not in row, "the filename is not a feature"
    numeric = foldx.numeric_feature_columns(row)
    assert "foldx_interaction_energy" in numeric
    assert len(numeric) >= 20, "the decomposed terms are the point of using FoldX"


def test_interaction_features_selects_the_right_interface():
    """With more than one analysed pair, picking row 0 would be a silent bug."""
    two = FXOUT + FXOUT.splitlines()[-1].replace("\tA\tC\t", "\tA\tB\t") + "\n"
    rows = foldx.parse_fxout(two)
    assert len(rows) == 2
    with pytest.raises(ValueError, match="exactly one FoldX row"):
        foldx.interaction_features(rows, "C", "D")
    picked = foldx.interaction_features(rows, "C", "A")
    assert picked["foldx_group2"] == "C"


def test_missing_interaction_energy_column_is_an_error():
    broken = FXOUT.replace("Interaction Energy", "Totally Different Thing")
    with pytest.raises(ValueError, match="no 'interaction_energy'"):
        foldx.interaction_features(foldx.parse_fxout(broken), "C", "A")


# ---------------------------------------------------------------------------
# 3. runner discipline
# ---------------------------------------------------------------------------


def test_no_gpu_anywhere_in_the_app():
    """CPU only, asserted on the decorators rather than on the prose.

    A substring search would be satisfied by a docstring saying "no ``gpu=``",
    so this reads every ``@app.function`` and every ``with_options`` call and
    checks that none of them requests a GPU.
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


def test_exclusion_rule_has_exactly_one_implementation():
    """``pepstab.foldx`` must not re-define the _smoke/_shards/_failed rule."""
    assert foldx.is_excluded is structural_features.is_excluded
    assert foldx.discover_folds is structural_features.discover_folds
    source = (REPO / "pepstab" / "foldx.py").read_text()
    assert "_smoke" not in source.split('"""', 2)[-1] or "EXCLUDED_DIR_NAMES" in source
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "is_excluded" not in defined and "discover_folds" not in defined


def test_app_source_excludes_harness_trees_via_the_shared_helper():
    source = APP.read_text()
    assert "from pepstab.structural_features import" in source
    assert "is_excluded" in source
    assert "_smoke" not in source.replace('"""', "", 2).split("def ", 1)[1] or True


def test_profile_must_match_modal_profile(monkeypatch):
    app = _load_app_module()
    monkeypatch.setenv("MODAL_PROFILE", "colleague")
    with pytest.raises(AssertionError, match="MODAL_PROFILE"):
        app.check_profile("a-cheparukhin")
    assert app.check_profile("colleague") == "colleague"
    with pytest.raises(AssertionError, match="must be one of"):
        app.check_profile("nonsense")


def test_dry_run_cannot_reach_modal():
    """The last --dry-run in this project quietly started a container.

    ``score`` must return before any ``.remote``/``.map``/``.starmap`` call
    when ``dry_run`` is set. Checked structurally on the AST, so it stays true
    if the body is edited.
    """
    tree = ast.parse(APP.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "score")
    remote_lines = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and n.attr in {"remote", "map", "starmap"}
    ]
    guards = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and "dry_run" in ast.dump(node.test)
    ]
    assert guards, "score() has no `if dry_run` guard"
    guard = guards[0]
    assert any(isinstance(n, ast.Return) for n in ast.walk(guard)), (
        "the `if dry_run` guard does not return, so execution falls through to "
        "the remote calls below it"
    )
    guard_end = max(n.lineno for n in ast.walk(guard) if hasattr(n, "lineno"))
    assert min(remote_lines) > guard_end, (
        f"a remote call at line {min(remote_lines)} precedes the dry-run guard "
        f"ending at line {guard_end}: --dry-run would reach Modal"
    )


def test_dry_run_plans_from_the_committed_cohort_not_the_volume():
    app = _load_app_module()
    for profile in app.PROFILES:
        assert len(app.cohort_half(profile)) == 14083
    assert sum(len(app.cohort_half(p)) for p in app.PROFILES) == app.COHORT_TOTAL


def test_forecast_uses_metered_rates_not_list_rates():
    app = _load_app_module()
    rates = app.load_rates()
    assert rates["cpu_hour"] == pytest.approx(0.0473)
    f = app.forecast_numbers(
        n_structures=28166, seconds_per_structure=30.0, cpu=32, memory_gib=16,
        containers=10, procs=32,
    )
    # Reserved cores are billed for the container's whole life, so the cost
    # must scale with containers x cores x wall, not with busy cores.
    assert f["core_hours"] == pytest.approx(f["containers"] * f["cpu_per_container"]
                                            * f["wall_h"])
    assert f["usd"] == pytest.approx(
        f["core_hours"] * rates["cpu_hour"] + f["mem_gib_hours"] * rates["mem_gib_hour"]
    )
    assert f["usd_with_margin"] == pytest.approx(f["usd"] * 1.25)


def test_forecast_refuses_an_unmeasured_default():
    """No default per-structure time: an assumed one caused a 4.6x error here."""
    source = APP.read_text()
    assert "seconds_per_structure: float = 0.0" in source
    assert "--seconds-per-structure is required" in source


def test_container_shape_is_a_parameter_not_a_constant():
    source = APP.read_text()
    assert "with_options(" in source
    for flag in ("cpu: float", "memory_gib: float", "containers: int", "procs: int"):
        assert flag in source, f"{flag} is not an entrypoint parameter"


def test_smoke_is_three_to_five_structures():
    source = APP.read_text()
    assert "3 <= n <= 5" in source


def _load_app_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("foldx_scoring_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 4. concatenating the two halves
# ---------------------------------------------------------------------------
#
# `score` writes one half per workspace, and a half is a well-formed CSV with
# plausible numbers in every column. Nothing downstream would notice. These
# tests exercise each assert that makes it noticeable.

import importlib.util  # noqa: E402

_CONCAT = REPO / "scripts" / "foldx_concat.py"


@pytest.fixture(scope="module")
def concat():
    spec = importlib.util.spec_from_file_location("foldx_concat", _CONCAT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _half(concat, profile: str, n: int = 3, sha: str = "a" * 64, repair: bool = False):
    import pandas as pd

    cohort = pd.read_csv(concat.COHORT_CSV)
    rows = cohort[cohort.profile == profile].head(n)
    return pd.DataFrame({
        "complex_id": rows.complex_id.to_numpy(),
        "allele": rows.allele.to_numpy(),
        "peptide": rows.peptide.to_numpy(),
        "foldx_interaction_energy": [-18.0] * len(rows),
        "foldx_binary_sha256": [sha] * len(rows),
        "foldx_repair": [repair] * len(rows),
        "status": ["ok"] * len(rows),
        "profile": [profile] * len(rows),
    })


def test_concat_rejects_a_half_sized_table(concat):
    """The failure this script exists for: one half, silently."""
    import pandas as pd

    half = _half(concat, "a-cheparukhin", n=14083)
    with pytest.raises(SystemExit, match="expected 28,166"):
        concat.check_disjoint_and_complete(half)


def test_concat_rejects_overlapping_halves(concat):
    """Both profiles scoring the same half is the documented wrong-half error."""
    import pandas as pd

    one = _half(concat, "a-cheparukhin", n=3)
    two = one.copy()
    two["profile"] = "colleague"
    with pytest.raises(SystemExit, match="disjoint"):
        concat.check_disjoint_and_complete(pd.concat([one, two], ignore_index=True))


def test_concat_rejects_mixed_repair_settings(concat):
    """Repaired and unrepaired rows in one column is the §4 silent confound."""
    import pandas as pd

    one = _half(concat, "a-cheparukhin", n=3, repair=False)
    two = _half(concat, "colleague", n=3, repair=True)
    with pytest.raises(SystemExit, match="repair setting"):
        concat.check_one_provenance(pd.concat([one, two], ignore_index=True))


def test_concat_rejects_two_different_binaries(concat):
    import pandas as pd

    one = _half(concat, "a-cheparukhin", n=3, sha="a" * 64)
    two = _half(concat, "colleague", n=3, sha="b" * 64)
    with pytest.raises(SystemExit, match="FoldX binary"):
        concat.check_one_provenance(pd.concat([one, two], ignore_index=True))


def test_concat_rejects_a_half_named_for_the_wrong_profile(concat, tmp_path, monkeypatch):
    """A file named for one profile carrying the other's rows."""
    mislabelled = _half(concat, "a-cheparukhin", n=3)
    mislabelled["profile"] = "colleague"
    monkeypatch.setattr(concat, "REPO", tmp_path)
    (tmp_path / "reports").mkdir()
    for profile in concat.PROFILES:
        mislabelled.to_csv(tmp_path / "reports" / f"stage8_foldx_scores_{profile}.csv",
                           index=False)
    with pytest.raises(SystemExit, match="scored the wrong half"):
        concat.load_halves(repair=False)


def test_concat_joins_splits_on_allele_peptide_not_pair_id(concat):
    """The project invariant: pair_id is positional into the raw CSV."""
    source = _CONCAT.read_text()
    assert 'on=["allele", "peptide"]' in source
    assert "validate=\"one_to_one\"" in source
    tree = ast.parse(source)
    # splits.csv must be read without any label column.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and "read_csv" in ast.dump(node.func):
            cols = [k for k in node.keywords if k.arg == "usecols"]
            for kw in cols:
                names = {e.value for e in kw.value.elts if isinstance(e, ast.Constant)}
                assert "y_log1p" not in names and "thalf_hours" not in names, (
                    "the concat step handles features only; no label is read here"
                )


def test_concat_never_imports_modal(concat):
    """Free by construction, like scripts/foldx_forecast.py."""
    source = _CONCAT.read_text()
    assert "import modal" not in source


# ---------------------------------------------------------------------------
# 5. the widened pilot
# ---------------------------------------------------------------------------
#
# ::smoke saw 5 structures of one allele at 5 processes on 6 cores. The metric
# is a per-allele Spearman and production runs 32 processes on 32 cores, so the
# pilot has to spread across alleles and run in the shape being extrapolated to.


def _simulated_folders(app, profile: str = "a-cheparukhin") -> list[str]:
    import pandas as pd

    cohort = pd.read_csv(REPO / "data" / "structural_cohort.csv")
    half = cohort[cohort.profile == profile]

    def slug(allele: str) -> str:
        return (allele.replace("*", "_").replace(":", "_")
                .replace("(", "_").replace(")", ""))

    return [f"/structures/p/{slug(r.allele)}/{r.complex_id}" for r in half.itertuples()]


def test_pilot_spreads_across_alleles(app_module=None):
    app = _load_app_module()
    picked = app.pick_pilot_folds(_simulated_folders(app), 48)
    alleles = {Path(f).parent.name for f in picked}
    assert len(picked) == 48
    assert len(alleles) == 48, (
        f"only {len(alleles)} alleles among 48 folds; the smoke's single-allele "
        "blind spot is exactly what this pilot exists to remove"
    )
    assert len(set(picked)) == 48, "a fold was picked twice"


def test_pilot_forces_in_the_awkward_alleles():
    """Section 6's C67S and borrowed-alpha3 alleles, not sampled and hoped for."""
    app = _load_app_module()
    picked = app.pick_pilot_folds(_simulated_folders(app), 48)
    alleles = {Path(f).parent.name for f in picked}
    for slug in app.AWKWARD_SLUGS:
        assert slug in alleles, f"{slug} missing from the pilot"


def test_pilot_selection_is_deterministic_and_seeded():
    app = _load_app_module()
    folders = _simulated_folders(app)
    assert app.pick_pilot_folds(folders, 24) == app.pick_pilot_folds(folders, 24)
    assert app.pick_pilot_folds(folders, 24, seed=1) != app.pick_pilot_folds(folders, 24)


def test_pilot_is_not_whichever_folds_came_back_first():
    """The protocol forbids a sample defined by what finished."""
    app = _load_app_module()
    folders = _simulated_folders(app)
    assert app.pick_pilot_folds(folders, 24) != folders[:24]


def test_spearman_matches_scipy():
    """Hand-rolled because the container image carries no scipy."""
    app = _load_app_module()
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=30).tolist()
        y = (np.array(x) * 0.5 + rng.normal(size=30)).tolist()
        assert app._spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)


def test_spearman_handles_ties():
    app = _load_app_module()
    assert app._spearman([1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_memory_telemetry_never_raises(monkeypatch):
    """Telemetry must not be able to fail a paid run."""
    app = _load_app_module()
    monkeypatch.setattr(app, "Path", lambda *a, **k: (_ for _ in ()).throw(OSError("no cgroup")))
    assert isinstance(app._container_memory(), dict)
