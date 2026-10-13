"""Match the recorded hybrid replay backend; no model or tolerance changes."""
import json
import hashlib
import os
from pathlib import Path
import sys

SOURCE = 'pepstab/final_replay_backend.py'
FLAGS = {'avx','avx2','fma','avx512f','avx512vl','avx512bw','avx512dq'}
VERSIONS = {'numpy.libs':'0.3.34.106.0','scipy.libs':'0.3.31.dev'}


def cpu_capabilities(text):
    rows=[set(line.split(':',1)[1].split()) for line in text.splitlines() if line.startswith('flags')]
    if not rows or any(not FLAGS <= row for row in rows):
        raise ValueError('recorded SkylakeX backend requires AVX512-capable exposed CPUs; no fallback')
    return dict(required_flags=sorted(FLAGS),exposed_cpu_count=len(rows),all_required_flags_present=True,
                models=sorted({line.split(':',1)[1].strip() for line in text.splitlines() if line.startswith('model name')}))


def check_libraries(libraries):
    selected=[]
    for marker,version in VERSIONS.items():
        matches=[r for r in libraries if marker in r.get('filepath','') and r.get('internal_api')=='openblas']
        if len(matches)!=1 or any(matches[0].get(k)!=v for k,v in dict(version=version,architecture='SkylakeX',num_threads=2).items()):
            raise ValueError('actual OpenBLAS library/backend/threads differ from recorded fitting runtime')
        selected.append(matches[0])
    return selected


def pin_before_imports(auth,path,target):
    """Called by file, before importing the NumPy-importing pepstab package."""
    root=Path(__file__).resolve().parents[1]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if ('numpy' in sys.modules or 'scipy' in sys.modules
            or auth.get('approved') is not True or auth.get('phase')!='hybrid_report'
            or os.environ.get('FINAL_CLOUD_AUTH_SHA256')!=digest(path)
            or auth.get('sources',{}).get(SOURCE)!=digest(__file__)
            or Path(target).resolve()!=root/'scripts/final_hybrids_cloud_report.py'):
        raise ValueError('exact fresh released report context required before backend pin')
    ref=auth['report_backend_amendment'];meta=(root/ref['path']).resolve()
    if not meta.is_relative_to(root) or digest(meta)!=ref['sha256']:
        raise ValueError('backend amendment binding changed')
    amendment=json.loads(meta.read_text())
    if amendment.get('schema')!='final_hybrid_backend_replay_amendment.1' or amendment.get('coretype')!='SkylakeX':
        raise ValueError('exact historical backend required')
    cpu_capabilities(Path('/proc/cpuinfo').read_text())
    os.environ['OPENBLAS_CORETYPE']='SkylakeX'


def prepare(auth):
    from pepstab import final_cloud_protocol as contract
    if auth.get('phase')!='hybrid_report' or auth.get('sources',{}).get(SOURCE)!=contract.sha(Path(__file__)):
        raise ValueError('source-bound report-only backend amendment required')
    ref=auth['report_backend_amendment'];amendment=contract.read(contract.bound(ref))
    if amendment.get('schema')!='final_hybrid_backend_replay_amendment.1' or amendment.get('coretype')!='SkylakeX':
        raise ValueError('exact historical backend amendment required')
    records=amendment.get('fitting_runtime_records',[])
    if len(records)!=12:raise ValueError('both pilot and all ten fitting runtime records required')
    for record in records:
        check_libraries(contract.read(contract.bound(record))['threadpools'])
    if os.environ.get('OPENBLAS_CORETYPE')!='SkylakeX':
        raise ValueError('backend must be pinned by the pre-import entry')
    cpu=cpu_capabilities(Path('/proc/cpuinfo').read_text())
    os.environ['OPENBLAS_CORETYPE']='SkylakeX'
    # Load both exact packaged backends before any model execution.
    import numpy
    import scipy.linalg
    from threadpoolctl import threadpool_info
    libraries=check_libraries(threadpool_info())
    result=dict(amendment=dict(ref),cpu=cpu,coretype='SkylakeX',threads=2,
                libraries=[dict(row,library_sha256=contract.sha(row['filepath'])) for row in libraries],
                held_replay_requirement='array_equal',tolerance_changes=False,
                previous_failure_cause='CPU/BLAS dispatch is a hypothesis, not established for the unlogged failed container')
    print('HISTORICAL_REPLAY_BACKEND '+json.dumps(result,sort_keys=True),flush=True)
    return result


def finalize(out,profile):
    from pepstab import final_cloud_protocol as contract
    out=Path(out);audit=contract.read(out/'audit.json')
    if audit.get('status')!='passed' or audit.get('new_supervised_fits')!=480 or 'replay_backend_profile' in audit:
        raise ValueError('successful complete exact report required before backend disclosure')
    audit['replay_backend_profile']=profile
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    with (out/'REPORT.md').open('a') as f:
        f.write('\nSaved-model replay used the recorded SkylakeX OpenBLAS backend with two numerical threads. CPU capabilities and actual library identities are in audit.json. All held predictions still had to reproduce exactly; no prediction tolerance was introduced. The cause of the earlier unlogged-container mismatch remains unproven.\n')
