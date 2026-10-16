"""Audit completed manual energy recovery; no scoring, labels, fits or bootstrap."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.followup3_representation_independent_audit import read,sha,csv

EXPECTED_MANIFEST='d2aa04836f99311851700772b63e583dd11cb2752bccfdbf4bc7c37f5ab564fd'
EXPECTED_SANDBOX='sb-01M4V17YJGZWCVSR1478SKWA59'
KEYS=('allele','peptide','cluster_id','outer_fold','complex_id','folder')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_attempt(identity,start,result,native,manifest_hash,protocol_hash,deadline):
    assert all(str(start[k])==str(result[k])==str(identity[k]) for k in KEYS)
    assert start['attempt']==2 and start['manifest_hash']==manifest_hash and start['deadline_utc']==deadline
    assert result['manual_recovery']==dict(manifest_hash=manifest_hash,attempt=2,numerical_threads=1)
    assert result['protocol_hash']==protocol_hash and result['status']=='ok' and result['repair_interface']=='ALL'
    assert all(result[k]==native[k] for k in ('structure_sha256','metadata_sha256'))


def check_provider(plan,created,terminal,plan_sha,settings):
    assert terminal['provider_terminal'] is True
    assert terminal['sandbox_id']==created['sandbox_id']==EXPECTED_SANDBOX
    assert terminal['app_id']==created['app_id']
    assert terminal['plan_sha256']==created['plan_sha256']==plan_sha
    assert plan['cpu']==16 and plan['memory_mib']==32768 and plan['workers']==1 and plan['threads']==1
    assert plan['timeout_seconds']==2700 and plan['reservation_usd']==3.5
    assert plan['deadline_utc']==settings['deadline_utc']
    assert plan['commands']==[['python','scripts/followup3_energy_recovery.py','run','--workers','16','--deadline-utc',settings['deadline_utc']]]
    assert settings['workers']==16
    deadline=datetime.fromisoformat(settings['deadline_utc']).timestamp()
    assert 0<deadline-created['created_epoch']<=2700


def audit_recovery(directory,resources,require_features=False):
    from scripts.followup3_energy_recovery import verify
    panel,protocol,manifest,old=verify(directory)
    assert manifest['manifest_hash']==EXPECTED_MANIFEST
    receipt=read(directory/'recovery_receipt.json')
    assert receipt['status']=='passed' and receipt['manifest_hash']==EXPECTED_MANIFEST
    assert receipt['manifest_file_sha256']==sha(directory/'manifest.json') and receipt['parent_protocol_hash']==protocol['protocol_hash']
    assert receipt['preserved_ok_rows']==4376 and receipt['preserved_original_interrupted_starts']==receipt['manual_recovered_rows']==73
    assert receipt['completed_successes_recomputed']==0 and receipt['scientific_recipe_unchanged'] is True
    old_hashes=manifest['original_success_row_sha256'];old_starts=manifest['original_interrupted_start_sha256']
    assert old_hashes==receipt['original_success_row_sha256'] and old_starts==receipt['original_interrupted_start_sha256']
    assert len(old_hashes)==4376 and len(old_starts)==73
    identities=manifest['recovery_rows']
    recovery_names={digest([r['allele'],r['peptide']])+'.json' for r in identities}
    assert len(identities)==len(recovery_names)==73 and not recovery_names&set(old_hashes)
    assert set(receipt['recovery_row_sha256'])==set(manifest['expected_native_inputs'])==recovery_names
    assert set(old_starts)==set(receipt['retry_start_sha256'])=={n[:-5]+'.started.json' for n in recovery_names}
    expected_panel_names={digest([a,p])+'.json' for a,p in panel[['allele','peptide']].itertuples(index=False,name=None)}
    assert len(panel)==4449 and expected_panel_names==set(old_hashes)|recovery_names
    settings=read(directory/'recovery_execution_started.json')
    assert settings['manifest_hash']==EXPECTED_MANIFEST and sha(directory/'recovery_execution_started.json')==receipt['recovery_execution_started_sha256']
    parent=ROOT/manifest['parent_directory'];old_dispatch=read(parent/'function_dispatch_receipt.json')
    statuses={(r['allele'],r['peptide']):r['status'] for r in old_dispatch['statuses']}
    assert len(old_dispatch['statuses'])==len(statuses)==4449
    assert all(statuses[(r['allele'],r['peptide'])]=='interrupted_attempt' for r in identities)
    assert sum(s=='ok' for s in statuses.values())==4376
    assert not (directory/'STOP_ACCESS.json').exists() and not (old/'STOP_ACCESS.json').exists()
    created=read(resources/'created.json');terminal=read(resources/'provider_exit.json');plan=read(resources/'plan.json')
    check_provider(plan,created,terminal,sha(resources/'plan.json'),settings)
    assert plan['inputs'][str((directory/'manifest.json').relative_to(ROOT))]==sha(directory/'manifest.json')
    for path,value in manifest['source_sha256'].items():
        assert plan['inputs'][path]==sha(ROOT/path)==value
    original_terminal=read(ROOT/'reports/followup3/resources/energy_extract_v2/provider_exit.json')
    assert original_terminal['provider_terminal'] is True and original_terminal['app_id']=='ap-nvIT2aCnYh2xc3Ea32CotH'
    records={}
    for name,value in old_hashes.items():
        assert sha(old/name)==value
        records[name]=read(old/name)
    for name,value in old_starts.items():assert sha(old/name)==value
    for identity in identities:
        name=digest([identity['allele'],identity['peptide']])+'.json';startname=name[:-5]+'.started.json'
        start=directory/'retry_rows'/startname;path=directory/'retry_rows'/name
        assert sha(start)==receipt['retry_start_sha256'][startname] and sha(path)==receipt['recovery_row_sha256'][name]
        assert not (old/name).exists()
        native=manifest['expected_native_inputs'][name]
        assert sha(ROOT/native['inverse_score_path'])==native['inverse_score_sha256']
        original_native=read(ROOT/native['inverse_score_path'])
        assert all(native[k]==original_native[k] for k in ('structure_sha256','metadata_sha256'))
        row=read(path);check_attempt(identity,read(start),row,native,EXPECTED_MANIFEST,protocol['protocol_hash'],settings['deadline_utc'])
        records[name]=row
    assert set(p.name for p in (directory/'retry_rows').glob('*.json'))==recovery_names|set(receipt['retry_start_sha256'])
    if receipt['recovery_dispatch_sha256'] is not None:
        path=directory/'recovery_dispatch_receipt.json';dispatch=read(path)
        assert sha(path)==receipt['recovery_dispatch_sha256'] and dispatch['manifest_hash']==EXPECTED_MANIFEST
        assert dispatch['attempted']==len(dispatch['statuses'])==73
        assert {(r['allele'],r['peptide']) for r in dispatch['statuses']}=={(r['allele'],r['peptide']) for r in identities}
        assert all(r['status']=='ok' for r in dispatch['statuses'])
    combined=directory/'combined_rows';hashes={**old_hashes,**receipt['recovery_row_sha256']}
    assert receipt['combined_row_sha256']==hashes and set(p.name for p in combined.glob('*.json'))==expected_panel_names
    columns=protocol['specification']['energy_columns'];assert len(columns)==29
    ordered=[]
    for identity in panel.to_dict(orient='records'):
        name=digest([identity['allele'],identity['peptide']])+'.json';row=records[name]
        assert sha(combined/name)==hashes[name]
        assert all(str(row[k])==str(identity[k]) for k in KEYS)
        assert row['status']=='ok' and row['protocol_hash']==protocol['protocol_hash'] and row['repair_interface']=='ALL'
        assert row['binary_sha256']==protocol['specification']['extraction']['binary_sha256']
        assert set(row['terms'])-{'foldx_group1','foldx_group2'}==set(columns)
        assert all(isinstance(row['terms'][c],(int,float)) and not isinstance(row['terms'][c],bool) and np.isfinite(row['terms'][c]) for c in columns)
        ordered.append([row['terms'][c] for c in columns])
    feature_checked=False;feature_hash=None
    if require_features or (parent/'energy_features.provenance.json').exists():
        provenance=read(parent/'energy_features.provenance.json');feature_hash=sha(parent/'energy_features.npz')
        assert provenance['protocol_hash']==protocol['protocol_hash'] and provenance['row_artifact_sha256']==hashes and provenance['feature_sha256']==feature_hash
        with np.load(parent/'energy_features.npz',allow_pickle=False) as data:
            assert set(data.files)=={'E','allele','peptide'} and data['E'].dtype==np.float64
            np.testing.assert_array_equal(data['allele'],panel.allele.to_numpy(str));np.testing.assert_array_equal(data['peptide'],panel.peptide.to_numpy(str))
            np.testing.assert_array_equal(data['E'],np.asarray(ordered,dtype=np.float64))
        feature_checked=True
    return dict(status='passed',study='energy_manual_recovery',manifest_hash=EXPECTED_MANIFEST,manifest_file_sha256=sha(directory/'manifest.json'),recovery_receipt_sha256=sha(directory/'recovery_receipt.json'),parent_protocol_hash=protocol['protocol_hash'],old_success_hashes_unchanged=4376,original_interrupted_starts_unchanged=73,manual_attempts_verified=73,native_source_hashes_verified=73,combined_rows=4449,energy_columns=29,provider_terminal=True,provider_returncode=terminal['returncode'],provider_receipt_sha256=sha(resources/'provider_exit.json'),resource_plan_sha256=sha(resources/'plan.json'),execution_started_sha256=sha(directory/'recovery_execution_started.json'),feature_matrix_checked=feature_checked,energy_feature_sha256=feature_hash,combined_hashmap_digest=digest(hashes),scoring_or_fits_executed=False,labels_read=False,audit_source_sha256=sha(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'reports/followup3/energy_recovery_v3')
    parser.add_argument('--resources',type=Path,default=ROOT/'reports/followup3/resources/energy_recovery_v3')
    parser.add_argument('--require-features',action='store_true')
    args=parser.parse_args();result=audit_recovery(args.directory,args.resources,args.require_features)
    name='independent_feature_audit.json' if result['feature_matrix_checked'] else 'independent_recovery_audit.json'
    (args.directory/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
