import hashlib
import json
from pathlib import Path
import pytest
from scripts import followup3_energy_recovery as recovery


def test_manual_start_without_result_never_retries(tmp_path):
    identity=dict(allele="A",peptide="P")
    rows=tmp_path/"retry_rows";rows.mkdir()
    key=recovery.core.digest(["A","P"])
    (rows/(key+".started.json")).write_text("{}")
    with pytest.raises(ValueError,match="no second retry"):
        recovery.recover_one((identity,tmp_path,"/missing",dict(protocol_hash="parent"),"manifest","2099-01-01T00:00:00Z",{}))


def test_shared_access_marker_blocks_before_start_or_executable(tmp_path):
    (tmp_path/"STOP_ACCESS.json").write_text("{}")
    with pytest.raises(PermissionError,match="access stop"):
        recovery.recover_one((dict(allele="A",peptide="P"),tmp_path,"/missing",dict(protocol_hash="parent"),"manifest","2099-01-01T00:00:00Z",{}))
    assert not list((tmp_path/"retry_rows").glob("*.started.json"))


def test_completed_manual_record_is_cached_and_identity_checked(tmp_path):
    rows=tmp_path/"retry_rows";rows.mkdir()
    identity=dict(allele="A",peptide="P")
    path=rows/(recovery.core.digest(["A","P"])+".json")
    row=dict(identity,status="ok",protocol_hash="parent",manual_recovery=dict(manifest_hash="manifest"))
    path.write_text(json.dumps(row))
    assert recovery.recover_one((identity,tmp_path,"/missing",dict(protocol_hash="parent"),"manifest","past_unused",{}))==row
    with pytest.raises(ValueError,match="cache identity"):
        recovery.recover_one((identity,tmp_path,"/missing",dict(protocol_hash="parent"),"changed","past_unused",{}))


def test_combined_directory_preserves_original_bytes_and_recovery_parent(tmp_path,monkeypatch):
    import pandas as pd
    old=tmp_path/"original";old.mkdir();retry=tmp_path/"retry_rows";retry.mkdir()
    original=old/"old.json";original.write_text('{"status":"ok"}\n')
    identity=dict(allele="A",peptide="P")
    path=retry/(recovery.core.digest(["A","P"])+".json")
    native=dict(structure_sha256="structure",metadata_sha256="metadata")
    path.write_text(json.dumps(dict(identity,**native,status="ok",protocol_hash="parent",manual_recovery=dict(manifest_hash="manifest"))))
    deadline="2099-01-01T00:00:00Z"
    (tmp_path/"recovery_execution_started.json").write_text(json.dumps(dict(manifest_hash="manifest",workers=1,deadline_utc=deadline)))
    start=path.with_name(path.stem+".started.json")
    start.write_text(json.dumps(dict(identity,manifest_hash="manifest",attempt=2,deadline_utc=deadline)))
    manifest=dict(manifest_hash="manifest",original_success_row_sha256={original.name:recovery.sha(original)},original_interrupted_start_sha256={},recovery_rows=[identity],expected_native_inputs={path.name:native})
    monkeypatch.setattr(recovery,"verify",lambda out:(pd.DataFrame([identity]),dict(protocol_hash="parent",specification=dict(energy_columns=[])),manifest,old))
    monkeypatch.setattr(recovery.core,"validate_energy_rows",lambda *args:None)
    # Production's fixed4449 guard stays closed even after verified copying.
    with pytest.raises(ValueError,match="complete original4449"):
        recovery.combine(tmp_path)
    assert (tmp_path/"combined_rows"/original.name).read_bytes()==original.read_bytes()
    assert (tmp_path/"combined_rows"/path.name).read_bytes()==path.read_bytes()
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    assert original_hash==manifest["original_success_row_sha256"][original.name]
    start.write_text(json.dumps(dict(identity,manifest_hash="manifest",attempt=3,deadline_utc=deadline)))
    with pytest.raises(ValueError,match="start identity/attempt/deadline"):
        recovery.combine(tmp_path)


def test_exact_native_coordinate_and_metadata_hash_preflight(tmp_path):
    cif=tmp_path/"model.cif";meta=tmp_path/"metadata.json"
    cif.write_text("native coordinates");meta.write_text("native metadata")
    expected=dict(structure_sha256=recovery.sha(cif),metadata_sha256=recovery.sha(meta))
    recovery.verify_native(tmp_path,expected)
    cif.write_text("other coordinates")
    with pytest.raises(ValueError,match="native coordinate/metadata bytes"):
        recovery.verify_native(tmp_path,expected)


def test_recovery_is_spawned_and_thread_deadline_graph_is_bounded():
    source=Path(recovery.__file__).read_text()
    assert 'multiprocessing.get_context("spawn")' in source
    assert 'not 1 <= workers <= 16' in source and 'remaining <= 2700' in source
    assert 'threadpool_limits(limits=1)' in source
    assert source.index('write_once(start,') < source.index('row = core.score_repaired')
    assert 'core.score_repaired' in source and 'locate_foldx(ROOT / "foldx5_Linux_0")' in source
