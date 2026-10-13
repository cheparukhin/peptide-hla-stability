"""Source-bound cloud entry restoring the exact frozen training labels before import."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

ALLOWED={'scripts/final_transfer.py','scripts/final_transfer_report.py','scripts/final_hybrids_cloud.py',
         'scripts/final_hybrids_cloud_report.py','scripts/final_cloud_phase.py','scripts/final_label_efficiency.py'}
COMPATIBILITY_SOURCES=('pepstab/final_cloud_targets.py','scripts/final_cloud_scientific_entry.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved',type=Path,required=True)
    parser.add_argument('--script',type=Path,required=True)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    # pepstab.__init__ imports NumPy: pin before importing any package member.
    bare=json.loads(args.approved.read_text())
    if 'report_backend_amendment' in bare:
        backend=ROOT/'pepstab/final_replay_backend.py'
        if bare.get('sources',{}).get('pepstab/final_replay_backend.py')!=hashlib.sha256(backend.read_bytes()).hexdigest():
            raise ValueError('pre-import backend source is not authorization-bound')
        runpy.run_path(str(backend))['pin_before_imports'](bare,args.approved,args.script)
    from pepstab import final_cloud_protocol as contract
    auth_path=contract.path_in_root(args.approved);auth=contract.read(auth_path)
    if os.environ.get('FINAL_CLOUD_AUTH_SHA256')!=contract.sha(auth_path) or not os.environ.get('FINAL_CLOUD_SANDBOX_ID','').startswith('sb-'):
        raise ValueError('canonical target entry requires released exact supervisor authorization')
    script=contract.path_in_root(args.script);relative=str(script.relative_to(ROOT))
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    if relative not in ALLOWED or (relative=='scripts/final_cloud_phase.py' and not any(flag in arguments for flag in ('--audit-pilot','--audit-transfer-recovery'))):
        raise ValueError('unsupported scientific entry target')
    for name in COMPATIBILITY_SOURCES:
        if auth['sources'].get(name)!=contract.sha(ROOT/name):raise ValueError('canonical target compatibility source changed')
    backend_profile = None
    if 'report_backend_amendment' in auth:
        if relative != 'scripts/final_hybrids_cloud_report.py':
            raise ValueError('historical backend matching is report-only')
        from pepstab.final_replay_backend import prepare
        backend_profile = prepare(auth)
    reference=auth['canonical_targets'];metadata_path=contract.bound(reference)
    from pepstab import final_cloud_targets as targets
    if metadata_path!=contract.path_in_root(targets.META):raise ValueError('canonical metadata path changed')
    manifest=contract.read(contract.bound(auth['input_manifest']))['files']
    metadata=contract.read(metadata_path)
    if (manifest.get(targets.META)!=reference['sha256'] or manifest.get(targets.ARRAYS)!=metadata['arrays']['sha256']):
        raise ValueError('canonical target inputs absent from immutable source manifest')
    targets.install(metadata_path)
    finalize_replay = None
    if 'report_replay_amendment' in auth:
        if relative != 'scripts/final_hybrids_cloud_report.py' or auth.get('phase') != 'hybrid_report':
            raise ValueError('stopping-MSE compatibility cannot enter fitting')
        from pepstab.final_hybrid_replay import install
        finalize_replay = install(auth)
    sys.argv=[str(script),*arguments]
    runpy.run_path(str(script),run_name='__main__')
    if finalize_replay is not None:
        finalize_replay(ROOT/contract.SCOPE)
    if backend_profile is not None:
        from pepstab.final_replay_backend import finalize
        finalize(ROOT/contract.SCOPE,backend_profile)

if __name__=='__main__':main()
