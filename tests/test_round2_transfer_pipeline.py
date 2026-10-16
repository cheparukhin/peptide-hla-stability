"""Controller tests use synthetic authorizations, gates and mocked processes only."""
import json
from pathlib import Path

import pytest

from scripts import round2_transfer_pipeline as pipeline


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def harness(tmp_path,monkeypatch):
    calls,children = [],[]
    protocol = {'protocol_hash':'synthetic-frozen'}
    monkeypatch.setattr(pipeline,'validate_frozen',lambda *_:(None,protocol))
    def verify(directory,protocol,regime,cell):
        data = json.loads((directory/'completed.json').read_text())
        assert data == {'status':'complete','protocol_hash':protocol['protocol_hash'],'cell':f'{regime}/{cell}'}
        return data
    monkeypatch.setattr(pipeline,'verify_cell',verify)
    monkeypatch.setattr(pipeline.time,'sleep',lambda _:None)
    class Child:
        def __init__(self,command,exit_code=0):
            self.pid = 5000+len(children);self.command = command
            self.exit_code,self.returncode,self.polls = exit_code,None,0
            self.terminated = False
            children.append(self)
        def poll(self):
            if self.returncode is not None:return self.returncode
            self.polls += 1
            if self.polls < 2:return None
            self.returncode = self.exit_code
            if self.exit_code == 0:
                out = Path(self.command[self.command.index('--out-dir')+1])
                regime = self.command[self.command.index('--regime')+1]
                cell = self.command[self.command.index('--cell')+1]
                write(out/regime/cell/'completed.json',dict(status='complete',
                    protocol_hash='synthetic-frozen',cell=f'{regime}/{cell}'))
            return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def kill(self):self.returncode=-9
        def wait(self,timeout=None):
            if self.returncode is None:self.polls=2;self.poll()
            return self.returncode
    def popen(command,**kwargs):
        calls.append(dict(command=command,kwargs=kwargs))
        return Child(command)
    monkeypatch.setattr(pipeline.subprocess,'Popen',popen)
    def authorize(phase_tag,workers=1,**changes):
        data = dict(schema=pipeline.SCHEMA,approved=True,phase=phase_tag,
            protocol_hash=protocol['protocol_hash'],controller_sha256=pipeline.sha256_file(Path(pipeline.__file__)),
            cells=list(pipeline.phase_cells(phase_tag)),max_workers=workers)
        if phase_tag == 'remaining':
            data['timing_completed_sha256'] = {cell:pipeline.sha256_file(tmp_path/cell/'completed.json')
                                                for cell in pipeline.TIMING_CELLS}
        data.update(changes)
        path = tmp_path/f'{phase_tag}_authorization.json';write(path,data)
        return path
    def execute(phase,workers=1,approval=None):
        return pipeline.execute_phase(phase,approval if approval is not None else authorize(phase,workers),
            out=tmp_path,grouping=tmp_path,main_root=tmp_path,feature_manifest=tmp_path/'features.json',workers=workers)
    return dict(out=tmp_path,calls=calls,children=children,Child=Child,
                authorize=authorize,execute=execute,protocol=protocol)


def test_timing_is_serial_exact_order_and_has_durable_receipts(harness):
    h = harness;state = h['execute']('timing')
    assert state['status'] == 'complete' and state['max_workers'] == 1
    assert [m['cell'] for m in state['finished_children']] == list(pipeline.TIMING_CELLS)
    assert len(state['launch_intents']) == 2 and all(m['pid'] >= 5000 for m in state['finished_children'])
    assert all(m['completed_sha256'] for m in state['finished_children'])
    assert (h['out']/'controller/timing_launch_intent.json').exists()
    assert json.loads((h['out']/'controller/timing_exit_receipt.json').read_text())['status'] == 'complete'
    for call in h['calls']:
        assert call['kwargs']['env']['OMP_NUM_THREADS'] == '2'
        assert str(h['out']/'controller') in call['kwargs']['stdout'].name
    assert all('round2_transfer.py' in command['command'][1] for command in h['calls'])


def test_remaining_exact28_cells_worker_cap_and_timing_hash_binding(harness):
    h = harness;h['execute']('timing')
    peak = 0
    original = pipeline.subprocess.Popen
    def observe(command,**kwargs):
        nonlocal peak
        child = original(command,**kwargs)
        peak = max(peak,sum(c.returncode is None for c in h['children']))
        return child
    pipeline.subprocess.Popen = observe
    state = h['execute']('remaining',workers=3)
    assert state['status'] == 'complete'
    assert state['cells'] == list(pipeline.phase_cells('remaining'))
    assert len(state['finished_children']) == 28 and peak == 3
    assert set(state['cells']).isdisjoint(pipeline.TIMING_CELLS)
    assert len(h['calls']) == 30
    assert not any('report' in arg for call in h['calls'] for arg in call['command'])


@pytest.mark.parametrize('changes',[
    {'approved':False},{'schema':'other'},{'phase':'remaining'},
    {'protocol_hash':'changed'},{'controller_sha256':'changed'},
    {'cells':['hla/h0']},{'cells':['joint/h0_p0','hla/h0']},
    {'max_workers':2},{'max_workers':True}])
def test_invalid_authorization_cannot_launch(harness,changes):
    h = harness;approval=h['authorize']('timing',**changes)
    with pytest.raises(ValueError,match='authorization'):
        h['execute']('timing',approval=approval)
    assert not h['calls']
    assert not (h['out']/'controller/timing_launch_intent.json').exists()


def test_no_approval_and_frozen_gate_failure_cannot_launch(harness,monkeypatch):
    h = harness
    with pytest.raises(ValueError,match='--approved'):
        pipeline.execute_phase('timing',None,out=h['out'],feature_manifest='synthetic')
    def closed(*_):raise ValueError('frozen protocol rejected')
    monkeypatch.setattr(pipeline,'validate_frozen',closed)
    with pytest.raises(ValueError,match='frozen protocol rejected'):
        h['execute']('timing')
    assert not h['calls']


@pytest.mark.parametrize('phase', ['timing','remaining'])
def test_phase_already_attempted_or_stale_intent_never_retries(harness,phase):
    h = harness
    if phase == 'remaining':h['execute']('timing')
    write(h['out']/f'controller/{phase}_launch_intent.json',dict(status='uncertain',pid=9999))
    before=len(h['calls'])
    with pytest.raises(ValueError,match='already attempted or uncertain'):
        h['execute'](phase)
    assert len(h['calls']) == before


def test_successful_phase_also_refuses_duplicate(harness):
    h=harness;h['execute']('timing');before=len(h['calls'])
    with pytest.raises(ValueError,match='already attempted'):
        h['execute']('timing')
    assert len(h['calls']) == before


def test_timing_completion_mutation_blocks_remaining(harness):
    h=harness;h['execute']('timing');approved=h['authorize']('remaining',3)
    path=h['out']/pipeline.TIMING_CELLS[0]/'completed.json'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='completion hash changed'):
        h['execute']('remaining',3,approved)
    assert len(h['calls']) == 2


@pytest.mark.parametrize('workers',[0,4,True])
def test_worker_overflow_or_invalid_type_rejected(harness,workers):
    h=harness
    approved=h['authorize']('timing',workers=workers)
    with pytest.raises(ValueError,match='worker count'):
        h['execute']('timing',workers,approved)
    assert not h['calls']


def test_child_failure_stops_dispatch_records_failure_and_refuses_retry(harness,monkeypatch):
    h=harness
    def failing(command,**kwargs):
        h['calls'].append(dict(command=command,kwargs=kwargs))
        return h['Child'](command,exit_code=7)
    monkeypatch.setattr(pipeline.subprocess,'Popen',failing)
    with pytest.raises(RuntimeError,match='exited 7'):
        h['execute']('timing')
    assert len(h['calls']) == 1
    receipt=json.loads((h['out']/'controller/timing_exit_receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['finished_children'][0]['exit_code'] == 7
    assert receipt['active_children'] == []
    with pytest.raises(ValueError,match='already attempted'):
        h['execute']('timing')
    assert len(h['calls']) == 1


def test_uncertain_popen_failure_retains_prelaunch_intent(harness,monkeypatch):
    h=harness
    def uncertain(*args,**kwargs):raise OSError('uncertain launch')
    monkeypatch.setattr(pipeline.subprocess,'Popen',uncertain)
    with pytest.raises(OSError,match='uncertain launch'):
        h['execute']('timing')
    receipt=json.loads((h['out']/'controller/timing_exit_receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert len(receipt['launch_intents']) == 1
    assert receipt['launch_intents'][0]['launch_status'] == 'intent_persisted'
    with pytest.raises(ValueError,match='already attempted'):
        h['execute']('timing')


def test_single_owner_lock_is_shared_across_phases(harness):
    h=harness;directory=h['out']/'controller';directory.mkdir()
    with (directory/'controller.lock').open('a') as lock:
        pipeline.fcntl.flock(lock,pipeline.fcntl.LOCK_EX|pipeline.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):h['execute']('timing')
    assert not h['calls']


def test_existing_uncertain_cell_lock_bars_launch(harness):
    h=harness;path=h['out']/'hla/h0/running.lock';path.parent.mkdir(parents=True);path.write_text('9999')
    with pytest.raises(ValueError,match='already complete or partially written/running/uncertain'):
        h['execute']('timing')
    assert not h['calls']


@pytest.mark.parametrize('artifact',['progress.jsonl','runtime.json','predictions.csv'])
def test_partial_cell_artifacts_cannot_be_relaunched(harness,artifact):
    h=harness;path=h['out']/'hla/h0'/artifact
    path.parent.mkdir(parents=True);path.write_text('partial')
    with pytest.raises(ValueError,match='partially written'):
        h['execute']('timing')
    assert not h['calls']


def test_failure_lets_started_siblings_settle_and_verify(harness,monkeypatch):
    h=harness;h['execute']('timing')
    def factory(command,**kwargs):
        h['calls'].append(dict(command=command,kwargs=kwargs))
        return h['Child'](command,exit_code=9 if len(h['calls']) == 3 else 0)
    monkeypatch.setattr(pipeline.subprocess,'Popen',factory)
    with pytest.raises(RuntimeError,match='exited 9'):
        h['execute']('remaining',3)
    assert len(h['calls']) == 5
    assert not any(child.terminated for child in h['children'])
    receipt=json.loads((h['out']/'controller/remaining_exit_receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['dispatch_stopped']
    assert [child['disposition'] for child in receipt['finished_children']] == ['failed','complete','complete']
    assert not receipt['active_children']


def test_interrupt_aborts_active_children_and_records_uncertainty(harness,monkeypatch):
    h=harness;h['execute']('timing')
    original_poll=h['Child'].poll
    def still_active(self):
        if self.pid >= 5002 and self.returncode is None:return None
        return original_poll(self)
    monkeypatch.setattr(h['Child'],'poll',still_active)
    def interrupt(_):raise KeyboardInterrupt('synthetic interruption')
    monkeypatch.setattr(pipeline.time,'sleep',interrupt)
    with pytest.raises(KeyboardInterrupt):h['execute']('remaining',3)
    receipt=json.loads((h['out']/'controller/remaining_exit_receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['controller_interrupted']
    assert len(receipt['finished_children']) == 3
    assert all(child['uncertain'] and child['disposition'] == 'aborted_after_controller_interrupt'
               for child in receipt['finished_children'])
    assert all(child.terminated for child in h['children'][2:])
    assert len(h['calls']) == 5


def test_remaining_requires_both_successful_timing_child_receipts(harness):
    h=harness;h['execute']('timing');authorization=h['authorize']('remaining',3)
    path=h['out']/'controller/timing_exit_receipt.json';receipt=json.loads(path.read_text())
    receipt['finished_children'][1]['exit_code']=4;write(path,receipt)
    with pytest.raises(ValueError,match='both timing children'):
        h['execute']('remaining',3,authorization)
    assert len(h['calls']) == 2


def test_exit_zero_requires_hash_verified_cell_artifacts(harness,monkeypatch):
    from scripts.round2_transfer import verify_cell as scientific_verify_cell
    h=harness
    monkeypatch.setattr(pipeline,'verify_cell',scientific_verify_cell)
    def factory(command,**kwargs):
        h['calls'].append(dict(command=command,kwargs=kwargs))
        child=h['Child'](command)
        original_poll=child.poll
        def poll():
            code=original_poll()
            if code == 0:
                regime=command[command.index('--regime')+1];cell=command[command.index('--cell')+1]
                directory=h['out']/regime/cell
                names=('predictions.csv','selection.json','ancestry.json','members.json','runtime.json')
                for name in names:(directory/name).write_text('synthetic artifact\n')
                hashes={name:pipeline.sha256_file(directory/name) for name in names}
                write(directory/'completed.json',dict(status='complete',regime=regime,cell=cell,
                    protocol_hash='synthetic-frozen',sha256=hashes))
                (directory/'predictions.csv').write_text('altered artifact\n')
            return code
        child.poll=poll
        return child
    monkeypatch.setattr(pipeline.subprocess,'Popen',factory)
    with pytest.raises(ValueError,match='artifact hash mismatch'):
        h['execute']('timing')
    assert len(h['calls']) == 1
    receipt=json.loads((h['out']/'controller/timing_exit_receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['finished_children'][0]['exit_code'] == 0
    assert receipt['finished_children'][0]['disposition'] == 'completion_unverified'


@pytest.mark.parametrize('timing_hashes',[None,{}, {'hla/h0':'wrong'}])
def test_remaining_missing_timing_hash_binding_blocks_dispatch(harness,timing_hashes):
    h=harness;h['execute']('timing')
    approval=h['authorize']('remaining',3,timing_completed_sha256=timing_hashes)
    with pytest.raises(ValueError,match='both timing completion hashes'):
        h['execute']('remaining',3,approval)
    assert len(h['calls']) == 2


def test_timing_two_workers_is_forbidden_even_with_matching_authorization(harness):
    h=harness
    approval=h['authorize']('timing',2)
    with pytest.raises(ValueError,match='worker count'):
        h['execute']('timing',2,approval)
    assert not h['calls']


def test_persistent_save_failure_after_popen_still_cleans_child(harness,monkeypatch):
    h=harness
    state=dict(launch_intents=[],active_children=[],finished_children=[])
    original_error=OSError('persistent status-write failure')
    def save():
        if h['children']:raise original_error
    child_poll=h['Child'].poll
    def remain_active(self):
        if self.returncode is None:return None
        return child_poll(self)
    monkeypatch.setattr(h['Child'],'poll',remain_active)
    job=dict(index=0,cell='hla/h0',command=pipeline.cell_command('hla/h0',h['out'],h['out'],h['out'],'fake'),
        log=str(h['out']/'mock.log'))
    with pytest.raises(OSError) as raised:
        pipeline.run_phase_jobs([job],state,save,h['protocol'],h['out'],1)
    assert raised.value is original_error
    assert len(h['children']) == 1 and h['children'][0].terminated
    assert h['children'][0].returncode == -15
    assert not state['active_children']
    assert state['finished_children'][0]['uncertain']
    assert 'persistent status-write failure' in state['cleanup_persistence_error']


def test_cleanup_continues_across_individual_termination_and_wait_errors():
    state=dict(active_children=[],finished_children=[])
    class Broken:
        pid=8000
        def poll(self):raise OSError('poll failed')
        def terminate(self):raise OSError('terminate failed')
        def wait(self,timeout=None):raise OSError('wait failed')
        def kill(self):raise OSError('kill failed')
    class Healthy:
        pid=8001
        terminated=False
        def poll(self):return None
        def terminate(self):self.terminated=True
        def wait(self,timeout=None):return -15
    broken,healthy=Broken(),Healthy()
    active={child:dict(pid=child.pid,cell='synthetic') for child in (broken,healthy)}
    def save():raise OSError('save still failed')
    pipeline.stop_children(active,state,save)
    assert healthy.terminated
    assert active == {broken:dict(pid=8000,cell='synthetic',cleanup_status='uncertain',uncertain=True)}
    assert state['active_children'][0]['pid'] == 8000
    assert state['finished_children'][0]['pid'] == 8001
    assert {error['action'] for error in state['cleanup_errors']} == {'poll','terminate','wait','kill','wait_after_kill'}


def test_exit_persistence_failure_preserves_original_dispatch_error(harness,monkeypatch):
    h=harness
    original_error=OSError('first write failed after launch')
    atomic=pipeline.atomic_json
    def failed_status(path,value):
        if h['children']:raise original_error
        return atomic(path,value)
    monkeypatch.setattr(pipeline,'atomic_json',failed_status)
    original_poll=h['Child'].poll
    def stay_active(self):
        if self.returncode is None:return None
        return original_poll(self)
    monkeypatch.setattr(h['Child'],'poll',stay_active)
    with pytest.raises(OSError) as raised:h['execute']('timing')
    assert raised.value is original_error
    assert h['children'][0].terminated
    assert (h['out']/'controller/timing_launch_intent.json').exists()
    with pytest.raises(ValueError,match='already attempted'):h['execute']('timing')
