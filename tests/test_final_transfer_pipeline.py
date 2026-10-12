import json
from pathlib import Path
import pytest
from scripts import final_transfer_pipeline as controller
from pepstab.boltz_embeddings import sha256_file


def authorization(tmp_path,workers=1,cells=('hla/h0',)):
    prior=tmp_path/'ranking_exit.json';prior.write_text(json.dumps({'status':'complete'}))
    item=dict(schema=controller.SCHEMA,approved=True,protocol_hash='protocol',controller_sha256=sha256_file(Path(controller.__file__)),cells=list(cells),max_workers=workers,
        wall_seconds=30,ranker_backend='local',prerequisite_exit_receipt=str(prior),prerequisite_exit_receipt_sha256=sha256_file(prior),prerequisite_pid=2147483647)
    path=tmp_path/'authorization.json';path.write_text(json.dumps(item));return path,item


def test_gate_binds_source_exact_axes_and_prerequisite(tmp_path,monkeypatch):
    path,item=authorization(tmp_path);monkeypatch.setattr(controller,'assert_process_absent',lambda pid:None)
    assert controller.check_authorization(path,{'protocol_hash':'protocol'},('hla/h0',),1)==item
    with pytest.raises(ValueError,match='cell axes'):controller.check_authorization(path,{'protocol_hash':'protocol'},('hla/h1',),1)
    with pytest.raises(ValueError,match='worker'):controller.check_authorization(path,{'protocol_hash':'protocol'},('hla/h0',),4)
    Path(item['prerequisite_exit_receipt']).write_text('{}')
    with pytest.raises(ValueError,match='prerequisite'):controller.check_authorization(path,{'protocol_hash':'protocol'},('hla/h0',),1)


def test_live_prerequisite_prevents_launch():
    import os
    with pytest.raises(ValueError,match='still exists'):controller.assert_process_absent(os.getpid())


def test_failed_attempt_has_durable_receipt_and_no_queued_retry(tmp_path,monkeypatch):
    cells=('hla/h0','hla/h1');path,item=authorization(tmp_path,cells=cells);protocol={'protocol_hash':'protocol'};calls=[]
    monkeypatch.setattr(controller,'validate_frozen',lambda out:(None,protocol));monkeypatch.setattr(controller,'assert_process_absent',lambda pid:None)
    def worker(out,regime,cell,approved,ranker_backend=None,cancel_event=None):calls.append(cell);raise RuntimeError('synthetic failure')
    monkeypatch.setattr(controller,'run_cell',worker)
    with pytest.raises(RuntimeError,match='synthetic'):controller.run_controller(tmp_path,path,cells)
    assert calls==['h0'];receipt=json.loads((tmp_path/'controller/exit_receipt.json').read_text());assert receipt['status']=='failed'
    assert (tmp_path/'controller/attempt.json').exists()
    with pytest.raises(FileExistsError):controller.run_controller(tmp_path,path,cells)


def test_modal_requires_root_owned_batch_backend(tmp_path,monkeypatch):
    path,item=authorization(tmp_path);item.update(ranker_backend='modal',cloud_max_active=18);path.write_text(json.dumps(item))
    monkeypatch.setattr(controller,'validate_frozen',lambda out:(None,{'protocol_hash':'protocol'}));monkeypatch.setattr(controller,'assert_process_absent',lambda pid:None)
    with pytest.raises(ValueError,match='backend must be supplied'):controller.run_controller(tmp_path,path,('hla/h0',))
    assert not (tmp_path/'controller/attempt.json').exists()
