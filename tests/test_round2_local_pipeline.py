import json
from pathlib import Path

from scripts.round2_local_pipeline import input_state


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_pipeline_requires_full_cache_and_all_sequence_repeats(tmp_path):
    assert input_state(tmp_path)[0] == "waiting"
    write(tmp_path / "extractor/full_training/feature_bundles.json",
          {"status": "complete", "expected_rows": 19716, "completed_rows": 19715})
    for fold in range(5):
        write(tmp_path / f"sequence/fold_{fold}/completed.json", {})
        write(tmp_path / f"sequence/fold_{fold}/repeats_completed.json", {})
    assert input_state(tmp_path)[0] == "waiting"
    write(tmp_path / "extractor/full_training/feature_bundles.json",
          {"status": "complete", "expected_rows": 19716, "completed_rows": 19716})
    assert input_state(tmp_path)[0] == "ready"
    (tmp_path / "sequence/fold_4/repeats_completed.json").unlink()
    assert input_state(tmp_path)[0] == "waiting"


def test_pipeline_halts_on_upstream_failure(tmp_path):
    write(tmp_path / "extractor/full_training/extraction_budget_ledger.json",
          {"completed": {"0": {"status": "failed"}}})
    assert input_state(tmp_path)[0] == "failed"
    write(tmp_path / "extractor/full_training/extraction_budget_ledger.json", {})
    write(tmp_path / "sequence/fold_0/failure.json", {"error": "timeout"})
    assert input_state(tmp_path)[0] == "failed"
