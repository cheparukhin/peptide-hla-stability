"""Freeze, audit and explicitly gated workers for bounded transfer controls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits, threadpool_info
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pepstab import final_transfer as ft
from pepstab import evaluation as ev
from pepstab.boltz_embeddings import sha256_file
from pepstab.round2_nested import config_digest, positive_affine, apply_affine
from scripts import round2_transfer as old
from scripts.round2_transfer_report import validate_evidence as old_audit
from scripts.round2_embedding_screen import atomic_csv,atomic_json
OUT=ROOT/'reports/finalization/transfer_controls'
LEGACY=old.OUT
CODE=('pepstab/final_transfer.py','scripts/final_transfer.py','scripts/final_transfer_pipeline.py','scripts/final_transfer_report.py','modal_app/final_transfer.py','tests/test_final_transfer.py','tests/test_final_transfer_pipeline.py','pepstab/final_modal_backend.py','scripts/final_transfer_cloud.py','scripts/final_runtime_guard.py','pepstab/round2_budget.py','tests/test_final_modal_backend.py','tests/test_final_transfer_cloud.py','tests/test_final_runtime_guard.py','reports/finalization/transfer_controls/SPECIFICATION.md','reports/finalization/transfer_controls/runtime_plan.json')+old.CODE
FILES=('predictions.csv','selection.json','ancestry.json','members.json','runtime.json','raw_inner_predictions.csv','raw_final_predictions.csv','member_predictions.npz')


def now(): return datetime.now(timezone.utc).isoformat()


def legacy_protocol(legacy=LEGACY):
    protocol=json.loads((Path(legacy)/'assessment_protocol.json').read_text())
    if protocol['status']!='FROZEN' or config_digest(protocol['specification'])!=protocol['protocol_hash']: raise ValueError('legacy protocol corrupt')
    if protocol['specification']['nominations']!=['R4_ridge','R5_mlp']: raise ValueError('legacy nominations changed')
    specification=protocol['specification']
    for name,key in [('data/rasmussen_et_al_dataset.csv','source_sha256'),('data/splits.csv','splits_sha256')]:
        if sha256_file(ROOT/name)!=specification[key]:raise ValueError('legacy population source changed')
    for name,h in specification['grouping_hashes'].items():
        if sha256_file(Path(legacy)/name)!=h:raise ValueError('legacy grouping changed')
    for name,h in protocol['code_sha256'].items():
        if sha256_file(ROOT/name)!=h: raise ValueError('legacy source changed')
    return protocol


def reuse_cell(frame,fit,held,regime,cell,protocol,legacy=LEGACY):
    specification=protocol['specification']
    digest=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()
    if digest!=specification['data_hash'] or len(frame)!=specification['n_rows']:raise ValueError('legacy fitting labels/population identity changed')
    directory=Path(legacy)/regime/cell
    done=old.verify_cell(directory,protocol,regime,cell)
    saved=json.loads((directory/'ancestry.json').read_text()); ancestry=ft.TransferAncestry(regime)
    if saved['regime']!=regime: raise ValueError('legacy ancestry regime changed')
    ancestry.records,ancestry.rowsets=saved['records'],saved['rowsets']; ancestry.audit(frame,fit,held)
    selection=old_audit(frame,fit,held,regime,directory,done,ancestry,protocol['specification']['candidate_order'],protocol['feature_hashes'])
    rows=pd.read_csv(directory/'predictions.csv')
    identity=['pair_id','cluster_id','hla_group','allele','hla_fold','peptide_fold']
    predictions={}
    for arm in ft.REUSED_ARMS:
        part=rows[rows.arm.eq(arm)]
        if len(part)!=len(held) or set(part.pair_id)!=set(held.pair_id) or set(part.regime)!={regime} or set(part.cell)!={cell}: raise ValueError('legacy reuse identity mismatch')
        joined=held[identity].merge(part[identity+['y_pred']],on=identity,how='left',validate='one_to_one')
        predictions[arm]=ev.validate_finite(joined.y_pred,'legacy final')
    lookup={s['candidate']:s['pooled_inner_score'] for s in selection['scores']}
    scores={'original':lookup['original'],'mlp_centred':lookup[selection['selected']['mlp_centred']]}
    return dict(predictions=predictions,scores=scores,selection=selection,
        evidence_sha256={name:sha256_file(directory/name) for name in ('completed.json',)+tuple(done['sha256'])})


def specification(frame):
    return dict(schema='final_transfer_controls.1',population='original training only; previously examined development evidence; original test excluded',
        rows=len(frame),hla_groups=int(frame.hla_group.nunique()),outer_cells='existing five HLA and 25 joint cells; all retained',
        candidate_order=list(ft.CANDIDATES),reported_arms=list(ft.ARMS),sequence_order=list(ft.SEQUENCE_ORDER),fit_graph=ft.fit_graph(),
        inner_selection='3 HLA folds or all 3x3 joint intersections; pooled inner median per-allele Spearman min20; argmax with frozen order exact ties',
        stopping='trees/ranker: 3 stops x 2 seeds inner AND final; joint stops diagonal intersections gradients outside both; R0 MLP same feature-head recipe as reused R5, 2 inner/30 final members',
        tree_params=ft.TREE_PARAMS,centred_tree_k=20,ranker='one fixed depth8 PairLogit; positional onehot; within-allele pairs with inverse pair-count allele weights; no ranker grid',
        affine='one positive affine map per outer cell, slope floor1e-8; raw fixed-candidate clean pooled inner OOF scored before mapping; outer-fit labels only; map applied to final raw ranker predictions',
        affine_limit='training-size scale transport from smaller inner fits to larger outer-fit refits; mapped inner values are not independently calibrated; differs from old nested ranker',
        legacy='reuse audited original/centred scores and finals plus R4/R5/unchanged embedding_selector finals; absent legacy raw inner predictions are disclosed and not recreated',
        matched_controls='R4 ridge minus R0 ridge; R5 hidden64 MLP minus R0 hidden64 MLP; same grids/scaling/selection/stopping; R0 positional sequence onehot',
        uncertainty=dict(draws=2000,modes=['peptide','hla','crossed'],panels=[20,50],deletions='all 35 actual HLA groups; conditional on fitted predictions; crossed sensitivity may be conservative'),
        execution=dict(local_workers_max=3,numerical_threads=2,no_retry=True,no_auto_expansion=True,cloud_calls=1620,task_timeout_seconds=1000,startup_timeout_seconds=60),
        data_hash=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest())


def freeze(out=OUT,legacy=LEGACY):
    out,legacy=Path(out),Path(legacy)
    frame=old.training_frame(legacy); old.preflight(frame); prior=legacy_protocol(legacy)
    evidence={}
    for regime,cell,fit,held in ft.outer_cells(frame):
        evidence[f'{regime}/{cell}']=reuse_cell(frame,fit,held,regime,cell,prior,legacy)['evidence_sha256']
    spec=specification(frame)
    item=dict(status='FROZEN_NOT_EXECUTION_APPROVAL',frozen_at_utc=now(),protocol_hash=config_digest(spec),specification=spec,
        legacy_directory=str(legacy.resolve()),legacy_protocol_sha256=sha256_file(legacy/'assessment_protocol.json'),legacy_evidence_sha256=evidence,
        grouping_sha256={name:sha256_file(legacy/name) for name in ('protocol.json','manifest.csv','hla_groups.csv','coverage.csv')},
        code_sha256={name:sha256_file(ROOT/name) for name in CODE})
    path=out/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous[k]!=item[k] for k in item if k!='frozen_at_utc'): raise ValueError('refusing changed frozen protocol')
        return previous
    atomic_json(path,item); atomic_json(out/'preflight.json',old.preflight(frame))
    return item


def validate_frozen(out=OUT):
    item=json.loads((Path(out)/'protocol.json').read_text()); legacy=Path(item['legacy_directory'])
    if item['status']!='FROZEN_NOT_EXECUTION_APPROVAL' or config_digest(item['specification'])!=item['protocol_hash']: raise ValueError('final transfer protocol corrupt')
    for name,h in item['code_sha256'].items():
        if sha256_file(ROOT/name)!=h: raise ValueError(f'frozen source changed: {name}')
    if sha256_file(legacy/'assessment_protocol.json')!=item['legacy_protocol_sha256']: raise ValueError('legacy protocol changed')
    for name,h in item['grouping_sha256'].items():
        if sha256_file(legacy/name)!=h: raise ValueError('grouping changed')
    frame=old.training_frame(legacy)
    if specification(frame)!=item['specification']: raise ValueError('frozen population/specification changed')
    for cell,files in item['legacy_evidence_sha256'].items():
        for name,h in files.items():
            if sha256_file(legacy/cell/name)!=h: raise ValueError('reused legacy artifact changed')
    return frame,item


def verify_cell(directory,protocol,regime,cell):
    directory=Path(directory); done=json.loads((directory/'completed.json').read_text())
    if (done.get('status'),done.get('regime'),done.get('cell'),done.get('protocol_hash'))!=('complete',regime,cell,protocol['protocol_hash']): raise ValueError('completion identity mismatch')
    if set(done['sha256'])!=set(FILES): raise ValueError('incomplete completion hash coverage')
    if done['n_fits']!=ft.fit_graph()['regimes'][regime]['fits_per_cell']: raise ValueError('fit count differs from frozen graph')
    for name,h in done['sha256'].items():
        if sha256_file(directory/name)!=h: raise ValueError('completion artifact changed')
    return done


def run_cell(out,regime,cell,approved,*,ranker_backend=None,cancel_event=None):
    """Durable single-attempt ownership; failures never clear ownership or retry."""
    frame,protocol=validate_frozen(out)
    authorization=json.loads(Path(approved).read_text())
    if authorization.get('approved') is not True or authorization.get('protocol_hash')!=protocol['protocol_hash'] or f'{regime}/{cell}' not in authorization.get('cells',[]): raise ValueError('missing exact scientific/runtime authorization')
    if ranker_backend is None and authorization.get('ranker_backend')!='local': raise ValueError('explicit ranker backend required')
    match=[(f,h) for r,c,f,h in ft.outer_cells(frame) if (r,c)==(regime,cell)]
    if len(match)!=1: raise ValueError('unknown cell')
    fit,held=match[0];directory=Path(out)/regime/cell; directory.mkdir(parents=True,exist_ok=True)
    owner=directory/'attempt.json'; descriptor=os.open(owner,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,'w') as handle:
        json.dump(dict(pid=os.getpid(),started_at_utc=now(),protocol_hash=protocol['protocol_hash'],authorization_sha256=sha256_file(approved)),handle);handle.flush();os.fsync(handle.fileno())
    try:
        reused=reuse_cell(frame,fit,held,regime,cell,legacy_protocol(protocol['legacy_directory']),protocol['legacy_directory'])
        ancestry=ft.TransferAncestry(regime)
        def callback(member):
            if cancel_event is not None and cancel_event.is_set():raise RuntimeError('controller froze further cell fitting')
            with (directory/'progress.jsonl').open('a') as handle: handle.write(json.dumps(member)+'\n');handle.flush();os.fsync(handle.fileno())
        predictor=ft.ControlPredictor(frame,ancestry,callback=callback,ranker_backend=ranker_backend)
        atomic_json(directory/'runtime.json',dict(platform=platform.platform(),python=sys.version,threadpools=threadpool_info(),threads=2,
            versions={p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','catboost','threadpoolctl')},ranker_backend=authorization['ranker_backend']))
        values,selection,inner,raw_final=ft.assess_cell(fit,held,regime,predictor,ancestry,reused)
        audit=ancestry.audit(frame,fit,held)
        if len(predictor.members)!=ft.fit_graph()['regimes'][regime]['fits_per_cell']: raise ValueError('actual fit graph differs')
        identity=['pair_id','cluster_id','hla_group','allele','hla_fold','peptide_fold']
        atomic_csv(directory/'predictions.csv',pd.concat([held[identity].assign(regime=regime,cell=cell,arm=arm,y_pred=p) for arm,p in values.items()],ignore_index=True))
        atomic_csv(directory/'raw_inner_predictions.csv',inner)
        atomic_csv(directory/'raw_final_predictions.csv',pd.concat([held[['pair_id']].assign(procedure=p,raw_prediction=v) for p,v in raw_final.items()],ignore_index=True))
        arrays={f'm{i:04d}_{key}':value for i,member in enumerate(predictor.member_predictions) for key,value in member.items()}
        temporary=directory/'member_predictions.tmp.npz';np.savez_compressed(temporary,**arrays);temporary.replace(directory/'member_predictions.npz')
        for name,item in [('selection',selection),('ancestry',ancestry.as_dict()),('members',predictor.members)]: atomic_json(directory/f'{name}.json',item)
        atomic_json(directory/'completed.json',dict(status='complete',regime=regime,cell=cell,protocol_hash=protocol['protocol_hash'],n_fits=len(predictor.members),n_held=len(held),audit=audit,reused_evidence_sha256=reused['evidence_sha256'],completed_at_utc=now(),sha256={name:sha256_file(directory/name) for name in FILES}))
        atomic_json(directory/'exit_receipt.json',dict(status='complete',exit_code=0,finished_at_utc=now()))
    except BaseException as error:
        atomic_json(directory/'exit_receipt.json',dict(status='failed',error=repr(error),finished_at_utc=now()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','run-cell','validate']);parser.add_argument('--out-dir',type=Path,default=OUT);parser.add_argument('--legacy-dir',type=Path,default=LEGACY);parser.add_argument('--regime',choices=['hla','joint']);parser.add_argument('--cell');parser.add_argument('--approved',type=Path)
    args=parser.parse_args()
    if args.mode=='freeze': print(json.dumps(freeze(args.out_dir,args.legacy_dir)['specification']['fit_graph'],indent=2))
    elif args.mode=='validate':validate_frozen(args.out_dir)
    else:
        if args.approved is None or args.regime is None or args.cell is None: parser.error('--approved, --regime and --cell required')
        with threadpool_limits(limits=2): run_cell(args.out_dir,args.regime,args.cell,args.approved)
if __name__=='__main__':main()
