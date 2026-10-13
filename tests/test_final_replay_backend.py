import pytest
from pepstab.final_replay_backend import cpu_capabilities,check_libraries,FLAGS,VERSIONS


def test_capability_gate_requires_every_exposed_cpu():
    good='flags : '+' '.join(FLAGS)+'\n'
    assert cpu_capabilities(good+good)['exposed_cpu_count']==2
    for text in ('',good+'flags : avx avx2 fma\n'):
        with pytest.raises(ValueError):cpu_capabilities(text)


def libraries():
    return [dict(filepath='/lib/'+marker+'/openblas.so',internal_api='openblas',architecture='SkylakeX',version=version,num_threads=2) for marker,version in VERSIONS.items()]


@pytest.mark.parametrize('field,value',[('architecture','Haswell'),('num_threads',1),('version','different')])
def test_different_actual_backend_is_refused(field,value):
    rows=libraries();assert len(check_libraries(rows))==2
    rows[0][field]=value
    with pytest.raises(ValueError):check_libraries(rows)


def test_both_backends_must_be_present_once():
    rows=libraries()
    for bad in (rows[:1],rows+rows[:1]):
        with pytest.raises(ValueError):check_libraries(bad)


def test_fresh_entry_pins_before_package_init(tmp_path):
    import hashlib,json,os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for name in ('scripts/final_cloud_scientific_entry.py','pepstab/final_replay_backend.py'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/name).read_bytes())
    # A synthetic package proves import order without loading numerical libraries.
    (tmp_path/'pepstab/__init__.py').write_text("import os,sys\nassert os.environ.get('OPENBLAS_CORETYPE')=='SkylakeX'\nassert 'numpy' not in sys.modules\nprint('PINNED_BEFORE_PACKAGE')\nraise SystemExit(0)\n")
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    meta=tmp_path/'amendment.json';meta.write_text(json.dumps({'schema':'final_hybrid_backend_replay_amendment.1','coretype':'SkylakeX'}))
    auth=tmp_path/'auth.json';auth.write_text(json.dumps({'approved':True,'phase':'hybrid_report','sources':{'pepstab/final_replay_backend.py':digest(tmp_path/'pepstab/final_replay_backend.py')},'report_backend_amendment':{'path':'amendment.json','sha256':digest(meta)}}))
    code="""import runpy,sys
from pathlib import Path
original=Path.read_text
def read(self,*a,**kw):
    return 'flags : avx avx2 fma avx512f avx512vl avx512bw avx512dq' if str(self)=='/proc/cpuinfo' else original(self,*a,**kw)
Path.read_text=read
sys.argv=[sys.argv[1],'--approved',sys.argv[2],'--script',sys.argv[3]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'scripts/final_cloud_scientific_entry.py'),str(auth),str(tmp_path/'scripts/final_hybrids_cloud_report.py')],env={**os.environ,'FINAL_CLOUD_AUTH_SHA256':digest(auth)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'PINNED_BEFORE_PACKAGE' in result.stdout
