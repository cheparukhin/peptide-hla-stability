"""Source-bound report-only one-ULP stopping-MSE check; frozen fitting is untouched."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
from pathlib import Path
import numpy as np
from pepstab import final_cloud_protocol as contract

TARGET = "        np.testing.assert_allclose([mse,details['best_stopping_mse']],[curves[best-1]]*2,rtol=0,atol=1e-12)"
REPLACEMENT = "        _report_stopping_mse_check(mse,details['best_stopping_mse'],curves[best-1],context,name)"
MLPS = {'standard','leaf_mlp','quantile_mlp','pca_mlp'}
SOURCE = 'pepstab/final_hybrid_replay.py'


def stopping_mse_check(recomputed, persisted, selected, context, component):
    """Only recomputation may differ; persisted stopping-state identity is exact."""
    values = (recomputed,persisted,selected)
    if any(isinstance(v,(bool,str)) or not isinstance(v,(int,float,np.integer,np.floating))
           or not math.isfinite(float(v)) or float(v)<0 for v in values):
        raise ValueError('finite nonnegative numerical MSE scalars required')
    if float(persisted)!=float(selected):
        raise ValueError('persisted best stopping MSE differs from selected curve')
    if any(float(v)>np.finfo(np.float32).max or float(np.float32(v))!=float(v) for v in values):
        raise ValueError('MSE scalar must preserve its original float32 value exactly')
    a,b=(int(np.asarray(np.float32(v)).view(np.uint32)) for v in (recomputed,selected))
    ulps=abs(a-b)
    if ulps>1:raise ValueError('recomputed stopping MSE differs by more than one float32 ULP')
    if component not in MLPS or not isinstance(context,str) or not context:
        raise ValueError('exact MLP component and context required')
    return dict(context=context,component=component,recomputed_mse=float(recomputed),
                persisted_best_mse=float(persisted),selected_curve_mse=float(selected),float32_ulp_difference=ulps)


def install(auth):
    """Install an exact copied auditor only in an authorized hybrid report process."""
    if (not isinstance(auth,dict) or auth.get('schema')!='final_cloud_phase_authorization.1'
            or auth.get('approved') is not True or auth.get('phase')!='hybrid_report'):
        raise ValueError('approved hybrid report authorization required')
    if auth.get('sources',{}).get(SOURCE)!=contract.sha(Path(__file__)):
        raise ValueError('report replay adapter source is not authorization-bound')
    reference=auth['report_replay_amendment'];metadata_path=contract.bound(reference)
    amendment=contract.read(metadata_path)
    if (amendment.get('schema')!='final_hybrid_stopping_mse_replay_amendment.1'
            or type(amendment.get('max_float32_ulp')) is not int or amendment['max_float32_ulp']!=1
            or amendment.get('held_predictions_remain_exact') is not True):
        raise ValueError('exact one-ULP report-only amendment required')
    from pepstab import final_hybrids as core
    original=core.audit_member_bundle
    if getattr(original,'_report_replay_installed',False):raise ValueError('report replay already installed')
    source=inspect.getsource(original)
    if (contract.sha(Path(core.__file__))!=amendment.get('original_audit_module_sha256')
            or hashlib.sha256(source.encode()).hexdigest()!=amendment.get('original_audit_function_sha256')):
        raise ValueError('original frozen auditor source binding changed')
    if source.count(TARGET)!=1:raise ValueError('exact single stopping-MSE assertion required')
    records=[]
    def checker(*arguments):records.append(stopping_mse_check(*arguments))
    namespace=dict(inspect.unwrap(original).__globals__)
    if '_report_stopping_mse_check' in namespace:raise ValueError('checker namespace collision')
    namespace['_report_stopping_mse_check']=checker
    # Exactly one source line changes. Every other original audit check executes.
    exec(compile(source.replace(TARGET,REPLACEMENT,1),str(core.__file__),'exec'),namespace)
    copied=namespace['audit_member_bundle'];copied._report_replay_installed=True
    core.audit_member_bundle=copied

    def finalize(out):
        out=Path(out).resolve()
        if out!=(contract.ROOT/contract.SCOPE).resolve():raise ValueError('exact hybrid report scope required')
        if contract.sha(metadata_path)!=reference['sha256']:raise ValueError('amendment changed during report')
        audit_path=out/'audit.json';report_path=out/'REPORT.md';audit=contract.read(audit_path)
        if (audit.get('status')!='passed' or audit.get('new_supervised_fits')!=480
                or audit.get('engineering_fits')!=12 or audit.get('protocol_hash')!=contract.SCIENCE_HASH
                or 'stopping_mse_replay_amendment' in audit or not report_path.is_file()):
            raise ValueError('successful original complete report required before amendment finalize')
        keys={(r['context'],r['component']) for r in records};contexts={r['context'] for r in records}
        if (len(records)!=328 or len(keys)!=328 or len(contexts)!=82
                or any({r['component'] for r in records if r['context']==context}!=MLPS for context in contexts)):
            raise ValueError('all 328 unique MLP scalar checks across 82 bundles required')
        summary=dict(binding=dict(reference),scalar_checks=len(records),bundles=len(contexts),
            maximum_float32_ulp_difference=max(r['float32_ulp_difference'] for r in records),
            nonzero_difference_count=sum(r['float32_ulp_difference']!=0 for r in records),
            persisted_curve_identity_exact=True,held_predictions_remain_exact=True,records=records)
        audit['stopping_mse_replay_amendment']=summary
        note=(f"\nReport replay used the separately authorized stopping-MSE amendment: {len(records)} "
              f"MLP scalar checks across 82 bundles, {summary['nonzero_difference_count']} nonzero differences, "
              f"maximum {summary['maximum_float32_ulp_difference']} float32 ULP. Only recomputed stopping MSE "
              "may differ by one ULP; persisted best-MSE/curve identity, held-prediction equality and all other "
              "frozen audit checks remain unchanged. No fitting, selection or statistical output was changed. "
              "The amendment binding and every scalar diagnostic are preserved in audit.json.\n")
        for path,text in ((audit_path,json.dumps(audit,indent=2,allow_nan=False)+'\n'),
                          (report_path,report_path.read_text()+note)):
            temporary=path.with_name('.'+path.name+'.replay')
            with temporary.open('x') as handle:handle.write(text)
            temporary.replace(path)
        return summary
    return finalize
