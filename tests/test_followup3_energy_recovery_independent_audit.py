import copy
import pytest
from scripts import followup3_energy_recovery_independent_audit as audit


def identity():
    return dict(allele='A',peptide='P',cluster_id=1,outer_fold=2,complex_id='case',folder='native/case')


def attempt():
    row=identity();native=dict(structure_sha256='cif',metadata_sha256='meta')
    start=dict(row,attempt=2,manifest_hash='manifest',deadline_utc='deadline')
    result=dict(row,**native,status='ok',protocol_hash='protocol',repair_interface='ALL',manual_recovery=dict(manifest_hash='manifest',attempt=2,numerical_threads=1))
    return row,start,result,native


def test_attempt_checks_identity_native_bytes_and_one_manual_attempt():
    args=attempt();audit.check_attempt(*args,'manifest','protocol','deadline')
    for field,value in [('attempt',3),('deadline_utc','changed'),('peptide','other')]:
        row,start,result,native=attempt();start[field]=value
        with pytest.raises(AssertionError):audit.check_attempt(row,start,result,native,'manifest','protocol','deadline')
    row,start,result,native=attempt();result['structure_sha256']='changed'
    with pytest.raises(AssertionError):audit.check_attempt(row,start,result,native,'manifest','protocol','deadline')
    row,start,result,native=attempt();result['manual_recovery']['numerical_threads']=2
    with pytest.raises(AssertionError):audit.check_attempt(row,start,result,native,'manifest','protocol','deadline')


def test_provider_requires_terminal_exact_owned_job_and_caps():
    settings=dict(deadline_utc='2026-10-13T17:08:22.283902+00:00',workers=16)
    plan=dict(cpu=16,memory_mib=32768,workers=1,threads=1,timeout_seconds=2700,reservation_usd=3.5,deadline_utc=settings['deadline_utc'],commands=[['python','scripts/followup3_energy_recovery.py','run','--workers','16','--deadline-utc',settings['deadline_utc']]])
    created=dict(sandbox_id=audit.EXPECTED_SANDBOX,app_id='app',plan_sha256='sha',created_epoch=1791908640.581984)
    terminal=dict(created,provider_terminal=True)
    audit.check_provider(plan,created,terminal,'sha',settings)
    for field,value in [('provider_terminal',False),('sandbox_id','other'),('plan_sha256','changed')]:
        changed=dict(terminal);changed[field]=value
        with pytest.raises(AssertionError):audit.check_provider(plan,created,changed,'sha',settings)
    changed=dict(plan);changed['cpu']=32
    with pytest.raises(AssertionError):audit.check_provider(changed,created,terminal,'sha',settings)


def test_incomplete_recovery_cannot_create_success_audit(tmp_path,monkeypatch):
    monkeypatch.setattr('sys.argv',['audit','--directory',str(tmp_path),'--resources',str(tmp_path)])
    with pytest.raises(FileNotFoundError):audit.main()
    assert not list(tmp_path.glob('independent_*audit.json'))
