"""Source-bound cloud entry restoring the exact frozen training labels before import."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pepstab import final_cloud_protocol as contract

ALLOWED={'scripts/final_transfer.py','scripts/final_transfer_report.py','scripts/final_hybrids_cloud.py',
         'scripts/final_hybrids_cloud_report.py','scripts/final_cloud_phase.py'}
COMPATIBILITY_SOURCES=('pepstab/final_cloud_targets.py','scripts/final_cloud_scientific_entry.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved',type=Path,required=True)
    parser.add_argument('--script',type=Path,required=True)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args();auth_path=contract.path_in_root(args.approved);auth=contract.read(auth_path)
    if os.environ.get('FINAL_CLOUD_AUTH_SHA256')!=contract.sha(auth_path) or not os.environ.get('FINAL_CLOUD_SANDBOX_ID','').startswith('sb-'):
        raise ValueError('canonical target entry requires released exact supervisor authorization')
    script=contract.path_in_root(args.script);relative=str(script.relative_to(ROOT))
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    if relative not in ALLOWED or (relative=='scripts/final_cloud_phase.py' and not any(flag in arguments for flag in ('--audit-pilot','--audit-transfer-recovery'))):
        raise ValueError('unsupported scientific entry target')
    for name in COMPATIBILITY_SOURCES:
        if auth['sources'].get(name)!=contract.sha(ROOT/name):raise ValueError('canonical target compatibility source changed')
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

if __name__=='__main__':main()
