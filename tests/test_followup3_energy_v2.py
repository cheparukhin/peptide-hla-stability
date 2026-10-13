import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_image_libraries_and_no_mounted_dependency_path():
    source = (ROOT / "modal_app/followup3_energy_v2.py").read_text()
    assert '.pip_install("numpy==2.5.3", "pandas==3.0.6", "scipy==1.18.1", "biotite==1.7.1")' in source
    assert "runs/followup3/dependencies" not in source
    assert "package.__file__" in source
    assert "outputs.reload()" in source and "outputs.commit()" in source


def test_uncached_production_path_gate_precedes_fanout_without_repeat():
    source = (ROOT / "modal_app/followup3_energy_v2.py").read_text()
    assert source.index("first_result = score_one.remote") < source.index("for row in score_one.map")
    assert 'if first_result["status"] != "ok" or first_result["cached"]:' in source
    assert '!= (first_identity["allele"], first_identity["peptide"])' in source
    assert "result = [first_result]" in source


def test_resource_and_scientific_limits_unchanged():
    tree = ast.parse((ROOT / "modal_app/followup3_energy_v2.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "score_one")
    keys = ("cpu", "memory", "max_containers", "timeout", "retries", "startup_timeout", "scaledown_window")
    values = {k.arg:ast.literal_eval(k.value) for k in function.decorator_list[0].keywords if k.arg in keys}
    assert values == dict(cpu=(1., 1.), memory=(2048, 2048), max_containers=100, timeout=660, retries=0, startup_timeout=120, scaledown_window=2)
