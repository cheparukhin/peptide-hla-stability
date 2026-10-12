"""Fixed 120-fit label-budget study; no scheduler, provider API or automatic retry.

freeze is metadata-only. run is explicitly authorized Linux/Modal work and
includes the fixed report. No command in this module is run on real data by tests.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SCOPE='reports/finalization/label_efficiency'
OUT=ROOT/SCOPE
HYBRID=ROOT/'reports/finalization/hybrids_cloud_v1'
SALT='pepstab-label-efficiency-v1'
FRACTIONS=(25,50)
DRAWS=(0,1,2)
REPS=('R0','R5')
MEMBERS=((0,0),(1,1))
OWN_SOURCES=('scripts/final_label_efficiency.py',SCOPE+'/SPECIFICATION.md',SCOPE+'/test_design.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def binding(path):return {'path':str(Path(path).resolve().relative_to(ROOT.resolve())),'sha256':sha(path)}
def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def array_hash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def average_members(vectors):
    """Match the original hybrid's finite float64 conversion before averaging."""
    from pepstab import evaluation as ev
    if len(vectors)!=2:raise ValueError('exact two-member schedule required')
    values=[ev.validate_finite(value,'standard member predictions') for value in vectors]
    if values[0].ndim!=1 or values[0].shape!=values[1].shape:raise ValueError('member predictions are not aligned vectors')
    return np.mean(values,axis=0)


def verify_saved_member(directory,arrays,Xheld,Zheld,pred,seed,n_gradient):
    """Same-process disk reload and exact held replay; no fitting or tolerances."""
    from pepstab.final_hybrids import _restore_scaler,_restore_mlp
    with np.load(Path(directory)/'arrays.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(arrays):raise ValueError('saved member payload keys changed')
        for name,value in arrays.items():
            if saved[name].dtype!=np.asarray(value).dtype:raise ValueError('saved member dtype changed')
            np.testing.assert_array_equal(saved[name],value,err_msg=f'saved member state changed: {name}')
        state={f'standard__{name}':saved[name] for name in ('w0','w1','b0','b1','offset')}
        state.update({f'standard__{name}':saved[f'scaler_{name}'] for name in ('mean','var','scale')})
        restored=_restore_scaler(state,'standard',n_gradient).transform(Xheld)
        np.testing.assert_array_equal(restored,Zheld,err_msg='saved scaler held transform changed')
        replay=_restore_mlp(state,'standard',seed).predict(restored)
        np.testing.assert_array_equal(replay,saved['prediction'],err_msg='saved model held predictions changed')
        np.testing.assert_array_equal(replay,pred,err_msg='original held predictions changed')
    csv=pd.read_csv(Path(directory)/'predictions.csv',float_precision='round_trip')
    np.testing.assert_array_equal(csv.y_pred.to_numpy(),np.asarray(pred,dtype=np.float64),err_msg='saved CSV predictions changed')
    return dict(saved_model_predictions_replayed_exactly=True,replayed_prediction_sha256=array_hash(replay))


def ordered_clusters(cluster_ids,draw):
    if draw not in DRAWS:raise ValueError('draw outside fixed three-ordering design')
    return sorted(set(map(int,cluster_ids)),key=lambda cid:(hashlib.sha256(f'{SALT}|draw={draw}|cluster={cid}'.encode()).hexdigest(),cid))


def make_tasks(frame):
    """Only IDs/cluster sizes are consumed; full stopping assignments precede subsets."""
    from pepstab import nested
    if not frame.split.eq('train').all() or frame.pair_id.duplicated().any():raise ValueError('unique original-training rows required')
    outer=nested.outer_folds(frame);tasks=[];reuse=[]
    for fold in range(5):
        fit,held=frame.loc[outer!=fold],frame.loc[outer==fold]
        stops=nested.inner_folds(fit,5)  # Never recomputed on selected rows.
        for stop,seed in MEMBERS:
            reuse.extend(dict(outer_fold=fold,representation=rep,stop_fold=stop,seed=seed,
                gradient_rows=fit.loc[stops!=stop].pair_id.tolist(),stopping_rows=fit.loc[stops==stop].pair_id.tolist(),held_rows=held.pair_id.tolist()) for rep in REPS)
        for draw in DRAWS:
            order=ordered_clusters(fit.cluster_id,draw)
            for fraction in FRACTIONS:
                selected=set(order[:math.ceil(len(order)*fraction/100)])
                mask=fit.cluster_id.isin(selected)
                selected_rows=fit.loc[mask].pair_id.tolist()
                for rep in REPS:
                    for stop,seed in MEMBERS:
                        gradient=fit.loc[mask&(stops!=stop)];stopping=fit.loc[mask&(stops==stop)]
                        if gradient.empty or stopping.empty:raise ValueError('empty subset partition; no redraw')
                        tasks.append(dict(index=len(tasks),fraction=fraction,draw=draw,outer_fold=fold,representation=rep,
                            stop_fold=stop,seed=seed,selected_clusters=sorted(selected),selected_rows=selected_rows,
                            gradient_rows=gradient.pair_id.tolist(),stopping_rows=stopping.pair_id.tolist(),held_rows=held.pair_id.tolist(),
                            outer_fit_clusters=len(order),outer_fit_rows=len(fit),selected_clusters_count=len(selected),
                            selected_rows_count=int(mask.sum()),gradient_clusters=int(gradient.cluster_id.nunique()),
                            stopping_clusters=int(stopping.cluster_id.nunique())))
    if len(tasks)!=120 or len(reuse)!=20:raise ValueError('fixed fit graph changed')
    return dict(new_members=tasks,reused_full_members=reuse,new_supervised_fits=120,new_full_budget_fits=0)


def load_context(canonical_targets,*,features=False):
    from pepstab import final_cloud_protocol as c,final_cloud_targets as targets
    path=c.bound(canonical_targets)
    if path!=c.path_in_root(targets.META):raise ValueError('canonical training target path changed')
    targets.install(path)
    from scripts.final_hybrids_cloud import validate_frozen,validated_features
    frame,hybrid_protocol=validate_frozen(HYBRID)
    return frame,hybrid_protocol,validated_features(frame,hybrid_protocol) if features else None


def baseline_members(frame,hybrid_protocol,certificate_ref,*,features=None):
    """Authenticate full-data reuse; no new model inference or fitting occurs."""
    from pepstab import final_cloud_protocol as c
    from pepstab.final_hybrids import mlp_config,member_identity
    from pepstab.round2_nested import unlabelled,align_predictions
    c.verify_phase_certificate(certificate_ref,'hybrid_report')
    certificate=c.read(c.bound(certificate_ref));output=c.read(c.bound(certificate['output_manifest']))['files']
    audit_path=HYBRID/'audit.json'
    audit=read(audit_path)
    expected_audits={(fold,rep) for fold in range(5) for rep in REPS}
    component_audits=audit.get('component_audits',[])
    if (output.get(str(audit_path.relative_to(ROOT)))!=sha(audit_path) or audit.get('status')!='passed'
            or audit.get('protocol_hash')!=hybrid_protocol['protocol_hash'] or audit.get('n_rows')!=19716
            or audit.get('new_supervised_fits')!=480 or audit.get('engineering_fits')!=12
            or audit.get('original_test_reopened') is not False or len(component_audits)!=10
            or {(a['outer_fold'],a['representation']) for a in component_audits}!=expected_audits
            or any(a.get('predictions_and_selection_replayed') is not True or a.get('bundles')!=8 or a.get('new_supervised_fits')!=48 for a in component_audits)):
        raise ValueError('successful full model/held-prediction/selection audit required for reuse')
    replay=audit.get('stopping_mse_replay_amendment')
    if replay is not None and (replay.get('held_predictions_remain_exact') is not True
            or replay.get('persisted_curve_identity_exact') is not True
            or replay.get('scalar_checks')!=328 or replay.get('maximum_float32_ulp_difference',2)>1):
        raise ValueError('baseline report must preserve exact held replay and persisted stopping state')
    runtime_path=HYBRID/'runtime_summary.json'
    if output.get(str(runtime_path.relative_to(ROOT)))!=sha(runtime_path):raise ValueError('baseline runtime/provenance binding changed')
    main_ref=read(runtime_path)['controller_receipts']['main']
    main_exit=c.read(c.bound({'path':main_ref['path'],'sha256':main_ref['sha256']}))
    main_files=c.read(c.bound(main_exit['output_manifest']))['files']
    index={int(pid):i for i,pid in enumerate(frame.pair_id)};reuse=make_tasks(frame)['reused_full_members'];predictions={}
    for fold in range(5):
        for rep in REPS:
            directory=HYBRID/f'fold_{fold}'/rep;done=read(directory/'completed.json')
            if main_files.get(str((directory/'completed.json').relative_to(ROOT)))!=sha(directory/'completed.json'):
                raise ValueError('original full-data completion is not provider-manifest-bound')
            for filename in ('bundle_manifest.json','raw_final_predictions.csv'):
                if sha(directory/filename)!=done['sha256'][filename] or main_files.get(str((directory/filename).relative_to(ROOT)))!=sha(directory/filename):raise ValueError('full-data reference hash changed')
            finals=[r for r in read(directory/'bundle_manifest.json') if r['role']=='final']
            if {(r['stop_fold'],r['seed']) for r in finals}!=set(MEMBERS) or len(finals)!=2:raise ValueError('exact full-data member schedule required')
            for record in finals:
                task=next(t for t in reuse if (t['outer_fold'],t['representation'],t['stop_fold'],t['seed'])==(fold,rep,record['stop_fold'],record['seed']))
                identity=record['identity'];bundle=directory/record['path'];completion=read(bundle/'completed.json')
                if sha(bundle/'completed.json')!=record['completed_sha256'] or completion['identity']!=identity:raise ValueError('full-member completion binding changed')
                for role,column in [('gradient','gradient_rows'),('stopping','stopping_rows'),('held','held_rows')]:
                    if identity['roles'][role]['pair_ids']!=task[column]:raise ValueError('full-budget member IDs differ')
                if identity['source_feature_hash']!=hybrid_protocol['specification']['feature_hashes'][rep] or identity['mlp']!=json.loads(json.dumps(asdict(mlp_config(task['seed'])))):
                    raise ValueError('full-budget feature/configuration differs')
                if features is not None:
                    role_frames={role:frame.iloc[[index[pid] for pid in task[column]]] for role,column in [('gradient','gradient_rows'),('stopping','stopping_rows'),('held','held_rows')]}
                    role_arrays={role:features[rep][[index[int(pid)] for pid in part.pair_id]] for role,part in role_frames.items()}
                    expected=member_identity(role_arrays,role_frames['gradient'],role_frames['stopping'],unlabelled(role_frames['held']),rep,task['seed'],identity['context'],hybrid_protocol['protocol_hash'],hybrid_protocol['specification']['feature_hashes'][rep])
                    if identity!=expected:raise ValueError('exact full-budget feature/label/recipe identity differs')
                if sha(bundle/'arrays.npz')!=completion['payload_sha256']['arrays.npz']:raise ValueError('audited standard-model payload changed')
            held=frame.iloc[[index[pid] for pid in task['held_rows']]]
            table=pd.read_csv(directory/'raw_final_predictions.csv',float_precision='round_trip')
            rows=table.loc[table.component.eq('standard')]
            predictions[(fold,rep)]=(held.pair_id.tolist(),align_predictions(held,rows))
    return predictions


def freeze(baseline_certificate):
    from pepstab.final_cloud_targets import META
    from pepstab.final_hybrids import mlp_config
    targets=binding(ROOT/META);frame,hybrid,_=load_context(targets)
    baseline=binding(baseline_certificate);baseline_members(frame,hybrid,baseline)
    tasks=make_tasks(frame);sources=dict(hybrid['source_sha256'])
    for name in (*OWN_SOURCES,'scripts/final_hybrids_cloud.py','pepstab/final_cloud_targets.py','pepstab/final_cloud_protocol.py'):sources[name]=sha(ROOT/name)
    specification=dict(schema='final_label_efficiency.1',fractions=[25,50,100],draws=list(DRAWS),hash_ordering_salt=SALT,
        member_axes=[list(x) for x in MEMBERS],config=json.loads(json.dumps(asdict(mlp_config(0)))),new_supervised_fits=120,
        full_budget_reused_members=20,original_training_only=True,stopping_assignments='full outer-fit then intersect both roles',
        subset_manifest_sha256=digest(tasks),scientific_specification_sha256=sha(OUT/'SPECIFICATION.md'),
        primary='mean_draw(R5_25_score-R0_25_score)-(R5_100_score-R0_100_score)',primary_panel=20,sensitivity_panel=50,
        subset_draw_aggregation='mean of scores, never predictions',bootstrap_draws=2000,bootstrap_seed=20261003,
        inclusive_cap_usd=10,canonical_population_sha256=hybrid['specification']['data_hash'],feature_hashes=hybrid['specification']['feature_hashes'])
    protocol=dict(schema='final_label_efficiency_protocol.1',status='frozen_no_execution_approval',protocol_hash=digest(specification),
        specification=specification,source_sha256=sources,canonical_targets=targets,baseline_report_certificate=baseline,
        hybrid_protocol=binding(HYBRID/'protocol.json'),created_at_utc=datetime.now(timezone.utc).isoformat())
    write_once(OUT/'task_manifest.json',tasks);write_once(OUT/'protocol.json',protocol)
    return {'protocol':binding(OUT/'protocol.json'),'tasks':binding(OUT/'task_manifest.json'),'new_fits':120}


def authorize(path):
    from pepstab import final_cloud_protocol as c
    if sys.platform!='linux':raise ValueError('paid scientific work is Modal/Linux only')
    path=Path(path);auth=c.read(path)
    if (auth.get('schema')!='final_label_efficiency_authorization.1' or auth.get('approved') is not True
            or auth.get('phase')!='label_efficiency' or auth.get('backend')!='modal_cpu_sandbox'
            or auth.get('inclusive_cap_usd')!=10 or auth.get('new_supervised_fits')!=120
            or auth.get('resources')!={'cpu_cores':2,'memory_mib':8192,'numerical_threads':2,'max_workers':1}
            or any(type(v) is not int for v in auth.get('resources',{}).values())
            or type(auth.get('work_wall_seconds')) is not int or not 0<auth['work_wall_seconds']<=3600
            or not time.time()<c.timestamp(auth['deadline_utc'])<=c.timestamp(c.OUTER_DEADLINE)
            or os.environ.get('FINAL_CLOUD_AUTH_SHA256')!=sha(path)
            or not os.environ.get('FINAL_CLOUD_SANDBOX_ID','').startswith('sb-')):
        raise ValueError('exact released bounded study authorization required')
    protocol=c.read(c.bound(auth['protocol']))
    if protocol.get('schema')!='final_label_efficiency_protocol.1' or protocol['protocol_hash']!=digest(protocol['specification']):raise ValueError('frozen study protocol changed')
    for name,digest_value in protocol['source_sha256'].items():
        if sha(ROOT/name)!=digest_value:raise ValueError('frozen study source changed')
    if binding(OUT/'protocol.json')!=auth['protocol']:raise ValueError('exact study output scope required')
    tasks=read(OUT/'task_manifest.json')
    if digest(tasks)!=protocol['specification']['subset_manifest_sha256']:raise ValueError('frozen subset manifest changed')
    return auth,protocol,tasks


def run(approved):
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from pepstab.final_hybrids import mlp_config
    from pepstab.mlp import MLPRegressor
    auth,protocol,tasks=authorize(approved);started=time.monotonic()
    def guard():
        if time.monotonic()-started>=auth['work_wall_seconds'] or time.time()>=datetime.fromisoformat(auth['deadline_utc']).timestamp() or sha(approved)!=os.environ['FINAL_CLOUD_AUTH_SHA256']:
            raise ValueError('fixed study deadline/authorization changed; no retry')
    frame,hybrid,features=load_context(protocol['canonical_targets'],features=True)
    if binding(HYBRID/'protocol.json')!=protocol['hybrid_protocol'] or make_tasks(frame)!=tasks:raise ValueError('frozen population/subsets changed')
    full=baseline_members(frame,hybrid,protocol['baseline_report_certificate'],features=features)
    guard();write_once(OUT/'attempt.json',dict(status='attempted',protocol_hash=protocol['protocol_hash'],authorization_sha256=sha(approved),new_fits=120,no_retry=True))
    position={int(pid):i for i,pid in enumerate(frame.pair_id)};tables=[];receipts=[]
    with threadpool_limits(limits=2):
        for task in tasks['new_members']:
            guard();directory=OUT/'members'/f"member_{task['index']:03d}";write_once(directory/'attempt.json',task)
            role_ids={role:task[column] for role,column in [('gradient','gradient_rows'),('stopping','stopping_rows'),('held','held_rows')]}
            X={role:features[task['representation']][[position[pid] for pid in ids]] for role,ids in role_ids.items()}
            scaler=StandardScaler().fit(X['gradient']);Z={role:scaler.transform(value) for role,value in X.items()}
            gradient=frame.iloc[[position[pid] for pid in role_ids['gradient']]];stopping=frame.iloc[[position[pid] for pid in role_ids['stopping']]]
            model=MLPRegressor(mlp_config(task['seed'])).fit(Z['gradient'],gradient.y_log1p.to_numpy(),Z['stopping'],stopping.y_log1p.to_numpy())
            pred=model.predict(Z['held']);guard()
            if pred.shape!=(len(role_ids['held']),) or not np.isfinite(pred).all():raise ValueError('nonfinite/unaligned member predictions; stop without retry')
            arrays={f'w{i}':value for i,value in enumerate(model.weights_)};arrays.update({f'b{i}':value for i,value in enumerate(model.biases_)})
            arrays.update(offset=np.asarray(model.y_offset_),curve=np.asarray(model.dev_curve_),scaler_mean=scaler.mean_,scaler_var=scaler.var_,scaler_scale=scaler.scale_,prediction=pred)
            np.savez_compressed(directory/'arrays.npz',**arrays)
            table=frame.iloc[[position[pid] for pid in role_ids['held']]][['pair_id','cluster_id','allele']].copy()
            table['y_pred']=pred.astype(np.float64);table['member_index']=task['index'];tables.append(table)
            table.to_csv(directory/'predictions.csv',index=False)
            replay=verify_saved_member(directory,arrays,X['held'],Z['held'],pred,task['seed'],len(gradient));guard()
            receipt=dict(status='complete',n_fits=1,task_sha256=digest(task),config=asdict(mlp_config(task['seed'])),
                feature_hashes={role:array_hash(value) for role,value in Z.items()},gradient_labels_sha256=array_hash(gradient.y_log1p.to_numpy()),
                stopping_labels_sha256=array_hash(stopping.y_log1p.to_numpy()),epochs=model.n_epochs_,best_epoch=model.best_epoch_,fit_seconds=model.fit_seconds_,
                payload_sha256={name:sha(directory/name) for name in ('arrays.npz','predictions.csv')})
            receipt.update(replay)
            write_once(directory/'completed.json',receipt);receipts.append(receipt)
        guard();report(frame,tasks,tables,full,protocol,guard);guard()
    write_once(OUT/'completed.json',dict(status='complete',protocol_hash=protocol['protocol_hash'],new_supervised_fits=120,
        reused_full_members=20,member_completed_sha256={str(i):sha(OUT/'members'/f'member_{i:03d}'/'completed.json') for i in range(120)},
        elapsed_seconds=time.monotonic()-started,outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.csv','.md')}))
    return {'new_supervised_fits':120,'reused_full_members':20}


def report(frame,tasks,tables,full,protocol,guard):
    from pepstab import evaluation as ev
    from pepstab.round2_nested import align_predictions
    from pepstab.round2_reporting import cluster_bootstrap_scores
    predictions={};draw_rows=[];curve_rows=[];gap_rows=[];comparison_rows=[];bootstrap_metadata={}
    for fraction in FRACTIONS:
        for draw in DRAWS:
            for rep in REPS:
                parts=[]
                for fold in range(5):
                    indices=[t['index'] for t in tasks['new_members'] if (t['fraction'],t['draw'],t['representation'],t['outer_fold'])==(fraction,draw,rep,fold)]
                    held=tables[indices[0]].drop(columns=['member_index','y_pred']).copy()
                    vectors=[align_predictions(held,tables[i]) for i in indices]
                    held['y_pred']=average_members(vectors);parts.append(held)
                predictions[f'{rep}_{fraction}_draw{draw}']=align_predictions(frame,pd.concat(parts))
    for rep in REPS:
        parts=[pd.DataFrame({'pair_id':full[(fold,rep)][0],'y_pred':full[(fold,rep)][1]}).merge(frame[['pair_id','cluster_id']],on='pair_id',validate='one_to_one') for fold in range(5)]
        predictions[f'{rep}_100']=align_predictions(frame,pd.concat(parts))
    for minimum in (20,50):
        guard();boot,meta=cluster_bootstrap_scores(frame,predictions,min_rows=minimum,n_boot=2000,seed=20261003)
        if meta['n_degenerate_resamples']:raise ValueError('degenerate fixed eligible panel; no scope change')
        bootstrap_metadata[str(minimum)]=meta
        point={name:float(ev.score(name,frame.allele,frame.y_log1p,pred,meta['alleles']).median_spearman) for name,pred in predictions.items()}
        for fraction in (25,50,100):
            means={}
            for rep in REPS:
                names=[f'{rep}_100'] if fraction==100 else [f'{rep}_{fraction}_draw{draw}' for draw in DRAWS]
                mean=float(np.mean([point[name] for name in names]));sample=boot[names].mean(axis=1);means[rep]=mean
                lo,hi=np.nanpercentile(sample,[2.5,97.5]);curve_rows.append(dict(min_rows=minimum,fraction=fraction,representation=rep,mean_score=mean,ci_low=float(lo),ci_high=float(hi),n_subset_draws=len(names)))
            gap_rows.append(dict(min_rows=minimum,fraction=fraction,mean_gap=means['R5']-means['R0']))
            for draw in ([None] if fraction==100 else DRAWS):
                suffix='100' if fraction==100 else f'{fraction}_draw{draw}'
                a,b=point['R0_'+suffix],point['R5_'+suffix]
                draw_rows.append(dict(min_rows=minimum,fraction=fraction,draw=draw,R0_score=a,R5_score=b,gap=b-a))
        for fraction in FRACTIONS:
            partial=sum(boot[f'R5_{fraction}_draw{draw}']-boot[f'R0_{fraction}_draw{draw}'] for draw in DRAWS)/3
            sample=partial-(boot.R5_100-boot.R0_100)
            partial_point=float(np.mean([point[f'R5_{fraction}_draw{draw}']-point[f'R0_{fraction}_draw{draw}'] for draw in DRAWS]))
            delta=partial_point-(point['R5_100']-point['R0_100']);lo,hi=np.nanpercentile(sample,[2.5,97.5])
            comparison_rows.append(dict(role='primary' if fraction==25 and minimum==20 else 'sensitivity',fraction=fraction,min_rows=minimum,
                delta=delta,ci_low=float(lo),ci_high=float(hi),bootstrap_mean_minus_point=float(np.nanmean(sample)-delta),conditional_on_fixed_fits_and_subsets=True))
        boot.to_csv(OUT/f'bootstrap_min{minimum}.csv',index=False)
    for name,data in [('scores_by_draw',draw_rows),('absolute_curves',curve_rows),('matched_gaps',gap_rows),('comparisons',comparison_rows)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    budget=[{k:v for k,v in t.items() if not k.endswith('_rows') and k!='selected_clusters'} for t in tasks['new_members']]
    # Include label-budget rows explicitly: both gradient AND stopping labels count.
    for row,t in zip(budget,tasks['new_members']):row.update(gradient_rows=len(t['gradient_rows']),stopping_rows=len(t['stopping_rows']),selected_rows=t['selected_rows_count'],outer_fit_rows=t['outer_fit_rows'],row_fraction=t['selected_rows_count']/t['outer_fit_rows'],cluster_fraction=t['selected_clusters_count']/t['outer_fit_clusters'])
    pd.DataFrame(budget).to_csv(OUT/'actual_budgets.csv',index=False)
    write_once(OUT/'reporting_metadata.json',dict(protocol_hash=protocol['protocol_hash'],bootstrap=bootstrap_metadata,
        score_aggregation='mean of three draw scores, never predictions',conditional_on_fixed_fits_and_subsets=True))
    primary=next(r for r in comparison_rows if r['role']=='primary')
    (OUT/'REPORT.md').write_text('# Fixed-head label-budget sensitivity\n\n'+f"Primary change in matched R5−R0 gap at 25% versus 100%: {primary['delta']:+.6f} [{primary['ci_low']:+.6f}, {primary['ci_high']:+.6f}].\n\n"+
        'Three predetermined nested subset draws are averaged as scores, never prediction ensembles. Both gradient and stopping labels count. All 120 new MLP fits and 20 audited full-data reused members follow the frozen design.\n\n'+
        'Intervals condition on fixed fits and subsets on previously examined development data. Full-data recipe nomination prevents claims of prospective assay savings. No validation/test, extraction, tuning or independent confirmation was added. See scores_by_draw.csv, absolute_curves.csv, matched_gaps.csv, actual_budgets.csv and comparisons.csv.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','run'])
    parser.add_argument('--baseline-certificate',type=Path);parser.add_argument('--approved',type=Path);args=parser.parse_args()
    if args.mode=='freeze':
        if args.baseline_certificate is None:parser.error('freeze requires completed audited baseline certificate')
        result=freeze(args.baseline_certificate)
    else:
        if args.approved is None:parser.error('run requires separate approved root authorization')
        result=run(args.approved)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
