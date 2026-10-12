"""Tiny scalar/source-copy checks only; no model fitting, replay or provider calls."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import pepstab
from pepstab import final_hybrid_replay as replay

SAVED=np.float32(0.8145772814750671)


def adjacent(value,steps):
    for _ in range(steps):value=np.nextafter(value,np.float32(np.inf),dtype=np.float32)
    return value


@pytest.mark.parametrize('steps',[0,1])
def test_zero_or_one_float32_ulp_only(steps):
    record=replay.stopping_mse_check(float(adjacent(SAVED,steps)),float(SAVED),np.float64(SAVED),'bundle','leaf_mlp')
    assert record['float32_ulp_difference']==steps


def test_two_float32_ulps_fail():
    with pytest.raises(ValueError,match='more than one'):
        replay.stopping_mse_check(adjacent(SAVED,2),SAVED,SAVED,'bundle','leaf_mlp')


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,True,'0.5'])
def test_invalid_scalar_fails(value):
    with pytest.raises(ValueError):replay.stopping_mse_check(value,SAVED,SAVED,'bundle','leaf_mlp')


def test_persisted_curve_must_match_exactly():
    with pytest.raises(ValueError,match='persisted'):
        replay.stopping_mse_check(SAVED,adjacent(SAVED,1),SAVED,'bundle','leaf_mlp')


def test_non_float32_value_must_not_be_rounded_to_pass():
    with pytest.raises(ValueError,match='original float32'):
        replay.stopping_mse_check(float(SAVED)+1e-13,SAVED,SAVED,'bundle','leaf_mlp')


def fixture(tmp_path,monkeypatch,*,target_count=1):
    source='''def audit_member_bundle(mse,persisted,selected,context='bundle',name='leaf_mlp',held=1,saved_held=1):
    details={'best_stopping_mse':persisted}
    curves=[selected]
    best=1
    def neural():
'''+('\n'.join([replay.TARGET]*target_count)+'\n' if target_count else '        pass\n')+'''        np.testing.assert_array_equal(held,saved_held)
    neural()
'''
    namespace={'np':np};exec(source,namespace)
    core=ModuleType('pepstab.final_hybrids');core.audit_member_bundle=namespace['audit_member_bundle']
    path=tmp_path/'original.py';path.write_text(source);core.__file__=str(path)
    monkeypatch.setitem(sys.modules,'pepstab.final_hybrids',core)
    monkeypatch.setattr(pepstab,'final_hybrids',core,raising=False)
    monkeypatch.setattr(replay.inspect,'getsource',lambda function:source)
    monkeypatch.setattr(replay.contract,'ROOT',tmp_path)
    metadata=dict(schema='final_hybrid_stopping_mse_replay_amendment.1',max_float32_ulp=1,
        held_predictions_remain_exact=True,original_audit_module_sha256=replay.contract.sha(path),
        original_audit_function_sha256=hashlib.sha256(source.encode()).hexdigest())
    meta_path=tmp_path/'amendment.json';meta_path.write_text(json.dumps(metadata))
    ref={'path':'amendment.json','sha256':replay.contract.sha(meta_path)}
    def bound(reference):
        p=tmp_path/reference['path']
        if replay.contract.sha(p)!=reference['sha256']:raise ValueError('binding changed')
        return p
    monkeypatch.setattr(replay.contract,'bound',bound)
    auth=dict(schema='final_cloud_phase_authorization.1',approved=True,phase='hybrid_report',
        report_replay_amendment=ref,sources={replay.SOURCE:replay.contract.sha(Path(replay.__file__))})
    return core,auth,metadata,meta_path,source


def test_copy_preserves_original_and_exact_held_check(tmp_path,monkeypatch):
    core,auth,_,_,source=fixture(tmp_path,monkeypatch)
    original=core.audit_member_bundle;global_assert=np.testing.assert_allclose
    with pytest.raises(AssertionError):original(adjacent(SAVED,1),SAVED,SAVED)
    replay.install(auth)
    core.audit_member_bundle(adjacent(SAVED,1),SAVED,SAVED)
    with pytest.raises(AssertionError):core.audit_member_bundle(SAVED,SAVED,SAVED,held=2,saved_held=1)
    assert np.testing.assert_allclose is global_assert
    assert (tmp_path/'original.py').read_text()==source
    with pytest.raises(ValueError,match='already installed'):replay.install(auth)


@pytest.mark.parametrize('field,value',[('phase','hybrid_main'),('approved',False),('sources',{})])
def test_wrong_authorization_fails(tmp_path,monkeypatch,field,value):
    _,auth,_,_,_=fixture(tmp_path,monkeypatch);auth[field]=value
    with pytest.raises(ValueError):replay.install(auth)


@pytest.mark.parametrize('field,value',[
    ('max_float32_ulp',2),('max_float32_ulp',True),('held_predictions_remain_exact',False),
    ('original_audit_module_sha256','a'*64),('original_audit_function_sha256','a'*64)])
def test_amendment_and_source_hashes_fail_closed(tmp_path,monkeypatch,field,value):
    core,auth,metadata,p,_=fixture(tmp_path,monkeypatch);original=core.audit_member_bundle
    metadata[field]=value;p.write_text(json.dumps(metadata));auth['report_replay_amendment']['sha256']=replay.contract.sha(p)
    with pytest.raises(ValueError):replay.install(auth)
    assert core.audit_member_bundle is original


@pytest.mark.parametrize('count',[0,2])
def test_replacement_requires_exact_single_original_line(tmp_path,monkeypatch,count):
    core,auth,_,_,_=fixture(tmp_path,monkeypatch,target_count=count);original=core.audit_member_bundle
    with pytest.raises(ValueError,match='single'):replay.install(auth)
    assert core.audit_member_bundle is original


def test_changed_binding_fails_before_install(tmp_path,monkeypatch):
    core,auth,_,p,_=fixture(tmp_path,monkeypatch);original=core.audit_member_bundle
    p.write_text('{}')
    with pytest.raises(ValueError):replay.install(auth)
    assert core.audit_member_bundle is original


def test_finalize_only_after_complete_report_and_all_scalar_checks(tmp_path,monkeypatch):
    core,auth,_,_,_=fixture(tmp_path,monkeypatch);finalize=replay.install(auth)
    out=tmp_path/replay.contract.SCOPE;out.mkdir(parents=True)
    audit=dict(status='passed',new_supervised_fits=480,engineering_fits=12,protocol_hash=replay.contract.SCIENCE_HASH,
        untouched_statistic=0.123)
    (out/'audit.json').write_text(json.dumps(audit));(out/'REPORT.md').write_text('Original report\n')
    with pytest.raises(ValueError,match='328'):finalize(out)
    assert json.loads((out/'audit.json').read_text())==audit
    for bundle in range(82):
        for name in sorted(replay.MLPS):
            core.audit_member_bundle(adjacent(SAVED,int(bundle==0)),SAVED,SAVED,context=f'bundle-{bundle}',name=name)
    summary=finalize(out);result=json.loads((out/'audit.json').read_text())
    assert summary['scalar_checks']==328 and summary['bundles']==82
    assert summary['nonzero_difference_count']==4 and summary['maximum_float32_ulp_difference']==1
    assert result['untouched_statistic']==audit['untouched_statistic']
    assert result['stopping_mse_replay_amendment']['binding']==auth['report_replay_amendment']
    assert 'held-prediction equality' in (out/'REPORT.md').read_text()
    with pytest.raises(ValueError,match='successful original'):finalize(out)


def test_failed_report_never_gets_amendment_finalize(tmp_path,monkeypatch):
    _,auth,_,_,_=fixture(tmp_path,monkeypatch);finalize=replay.install(auth)
    out=tmp_path/replay.contract.SCOPE;out.mkdir(parents=True)
    (out/'audit.json').write_text('{"status":"failed"}');(out/'REPORT.md').write_text('Unfinished report')
    with pytest.raises(ValueError,match='successful original'):finalize(out)
    assert (out/'REPORT.md').read_text()=='Unfinished report'
