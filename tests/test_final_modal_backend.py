import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from pepstab.final_modal_backend import GuardedBatchBackend, CALL_UPPER, RATE
from pepstab.round2_budget import ReservationBudget


class Function:
    def __init__(self, response=None):
        self.calls = 0
        self.response = response
    def map(self, tasks, return_exceptions):
        self.calls += len(tasks)
        if self.response is not None:
            return self.response
        return [dict(task_id=t['task_id'],pair_ids=[r['pair_id'] for r in t['held']],
                     predictions=[1.] * len(t['held']),worker_seconds=10.) for t in tasks]


@pytest.fixture
def setup(tmp_path):
    auth = tmp_path/'auth.json';status = tmp_path/'status.json'
    auth.write_text(json.dumps(dict(stop_flag_path=str(tmp_path/'stop'),status_path=str(status),
        deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())))
    status.write_text(json.dumps(dict(authorization_sha256=hashlib.sha256(auth.read_bytes()).hexdigest(),
        phase='monitoring',updated_at_utc=datetime.now(timezone.utc).isoformat(),guard_pid=os.getpid())))
    budget=ReservationBudget(tmp_path/'ledger.jsonl',ceiling_usd=1.,call_upper_usd=CALL_UPPER,max_calls=2)
    function=Function();backend=GuardedBatchBackend(function,budget,auth,tmp_path/'receipts')
    task=dict(task_id='a'*64,candidate={'id':'fixed_ranker8'},seed=0,regime='hla',gradient=[{}],stopping=[{}],held=[{'pair_id':1}])
    yield backend,budget,function,task,tmp_path
    budget.close()


def test_success_metering_and_no_duplicate_dispatch(setup):
    backend,budget,function,task,path=setup
    assert backend([task])[0]['predictions']==[1.]
    assert budget.status()['settled_usd']==pytest.approx(72*RATE)
    assert budget.status()['inflight_reserved_usd']==0
    with pytest.raises(FileExistsError):backend([task])
    assert function.calls==1


@pytest.mark.parametrize('failure',[RuntimeError('preempted'),{'task_id':'wrong'},
    dict(task_id='a'*64,pair_ids=[1],predictions=[1.],worker_seconds=1001),
    dict(task_id='a'*64,pair_ids=[1],predictions=[float('nan')],worker_seconds=10)])
def test_uncertain_result_burns_full_reservation(setup,failure):
    backend,budget,function,task,path=setup;function.response=[failure]
    with pytest.raises((ValueError,RuntimeError)):backend([task])
    assert budget.status()['settled_usd']==CALL_UPPER
    assert budget.status()['inflight_reserved_usd']==0
    assert function.calls==1


def test_stop_flag_and_stale_guard_prevent_calls(setup):
    backend,budget,function,task,path=setup
    (path/'stop').touch()
    with pytest.raises(RuntimeError,match='froze'):backend([task])
    (path/'stop').unlink();p=path/'status.json';s=json.loads(p.read_text())
    s['updated_at_utc']=(datetime.now(timezone.utc)-timedelta(seconds=26)).isoformat();p.write_text(json.dumps(s))
    with pytest.raises(RuntimeError,match='stale'):backend([task])
    assert function.calls==0 and budget.status()['calls']==0


def test_guard_change_after_reservation_burns_without_dispatch(setup,monkeypatch):
    backend,budget,function,task,path=setup
    original=budget.reserve
    def reserve(*args,**kwargs):
        token=original(*args,**kwargs);(path/'stop').touch();return token
    monkeypatch.setattr(budget,'reserve',reserve)
    with pytest.raises(RuntimeError,match='froze'):backend([task])
    assert function.calls==0 and budget.status()['settled_usd']==CALL_UPPER
