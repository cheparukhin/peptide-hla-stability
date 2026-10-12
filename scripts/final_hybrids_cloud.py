"""Cloud operational copy of the frozen 480-fit worker; scientific bodies unchanged."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pepstab import evaluation as ev,nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.round2_nested import Ancestry,config_digest,apply_affine,unlabelled,validate_partition
from pepstab.final_hybrid_reporting import COMPONENTS,SUFFIXES,ORDER,WEIGHTS,replay_selection
from scripts.final_ranking import load_features
from scripts.round2_embedding_screen import training_frame,atomic_json,atomic_csv
OUT=ROOT/'reports/finalization/hybrids'
REPRESENTATIONS=('R0','R5')
MEMBER_AXES=((0,0),(1,1))
FILES=('predictions.csv','inner_predictions.csv','raw_final_predictions.csv','selection.json','bundle_manifest.json','ancestry.json','runtime.json')
REFERENCES=('reports/finalization/ranking/scores.csv','reports/finalization/ranking/audit.json','reports/finalization/ranking/completion_audit.json','reports/further_research/round2/results/main/pooled_scores.csv','reports/further_research/round2/results/main/manifest.json')
CODE=('pepstab/final_hybrids.py','scripts/final_hybrids.py','scripts/final_hybrids_pipeline.py','scripts/final_hybrids_report.py','pepstab/final_hybrid_reporting.py','tests/test_final_hybrids.py','tests/test_final_hybrids_pipeline.py','tests/test_final_hybrid_reporting.py','pepstab/mlp.py','pepstab/phase_a.py','pepstab/features.py','pepstab/data.py','pepstab/nested.py','pepstab/splits.py','pepstab/evaluation.py','pepstab/round2_nested.py','pepstab/round2_reporting.py','pepstab/round2_representations.py','pepstab/boltz_embeddings.py','scripts/round2_embedding_screen.py','scripts/final_ranking.py','pepstab/final_ranking.py','pepstab/round2_embedding.py')


def now():return datetime.now(timezone.utc).isoformat()

def write_once(path,item):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as h:json.dump(item,h,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())

def array_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def dependency_versions():return {p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','catboost','threadpoolctl')}

def frame_hash(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()

def fit_graph():
    return dict(jobs=10,bundles_per_job=8,supervised_fits_per_bundle=6,fits_per_job=48,scientific_fit_count=480,scientific_mlp_fits=320,scientific_tree_fits=160,additional_leaf_tree_fits=0,engineering_pilot_fit_count=12,total_scientific_and_pilot_fits=492)

def call_schedule(frame,fold):
    if type(fold) is not int or fold not in range(5):raise ValueError('outer fold outside frozen scope')
    outer=nested.outer_folds(frame);fit,held=frame.loc[outer!=fold],frame.loc[outer==fold];validate_partition(fit,held)
    inner=nested.inner_folds(fit,3)
    for i in range(3):
        call_fit,call_held=fit.loc[inner!=i],fit.loc[inner==i];validate_partition(call_fit,call_held)
        yield 'inner',i,call_fit,call_held
    yield 'final',None,fit,held

def member_partitions(call_fit,call_held):
    stops=nested.inner_folds(call_fit,5)
    for stop,seed in MEMBER_AXES:
        gradient,stopping=call_fit.loc[stops!=stop],call_fit.loc[stops==stop]
        for a,b in ((gradient,stopping),(gradient,call_held),(stopping,call_held)):validate_partition(a,b)
        yield stop,seed,gradient,stopping

def task_manifest(frame):
    scientific=[];pilot=[]
    for rep in REPRESENTATIONS:
        for fold in range(5):
            for role,i,fit,held in call_schedule(frame,fold):
                for stop,seed,gradient,stopping in member_partitions(fit,held):
                    item=dict(representation=rep,outer_fold=fold,role=role,inner_fold=i,stop_fold=stop,seed=seed,n_fits=6,gradient_rows=gradient.pair_id.tolist(),stopping_rows=stopping.pair_id.tolist(),held_rows=held.pair_id.tolist())
                    scientific.append(item)
                    if fold==0 and i==0 and stop==0:pilot.append(dict(item,engineering_only=True,held_scoring=False,reuse_into_scientific=False))
    if len(scientific)!=80 or len(pilot)!=2:raise ValueError('task graph differs from480scientific+12pilot scope')
    return dict(scientific=scientific,engineering_pilot=pilot)

def preflight(frame):
    rows=[]
    ev.validate_finite(frame.y_log1p,'canonical training labels')
    if not frame.split.eq('train').all() or frame.pair_id.duplicated().any():raise ValueError('unique canonical original training rows required')
    for fold in range(5):
        for role,i,fit,held in call_schedule(frame,fold):
            panel=ev.eligible_alleles(held.allele,held.y_log1p,min_rows=20) if role=='inner' else None
            if role=='inner' and not panel:raise ValueError('empty frozen inner min20 panel; no cell may be dropped')
            for stop,seed,gradient,stopping in member_partitions(fit,held):
                if not (gradient.groupby('allele').y_log1p.nunique()>1).any():raise ValueError('no valid within-allele ranking pairs in frozen gradient')
                if len(gradient)<128:raise ValueError('PCA128 unavailable in frozen gradient partition')
                rows.append(dict(outer_fold=fold,role=role,inner_fold=i,stop_fold=stop,seed=seed,n_gradient=len(gradient),n_stopping=len(stopping),n_held=len(held),inner_eligible_alleles=None if panel is None else len(panel)))
    return rows

def freeze(out=OUT):
    from pepstab import final_hybrids as core
    out=Path(out);frame=training_frame();check=preflight(frame);features,provenance=load_features(frame)
    if tuple(core.COMPONENTS)!=COMPONENTS:raise ValueError('core component registry differs')
    graph=json.loads((out/'fit_graph.json').read_text());spec=out/'SPECIFICATION.md'
    if graph['specification_sha256']!=sha256_file(spec) or graph['scientific_fit_count']!=480 or graph['engineering_pilot_fit_count']!=12 or graph['components']!=list(COMPONENTS):raise ValueError('scientific specification/fit graph differs')
    ranked=json.loads((ROOT/'reports/finalization/ranking/protocol.json').read_text())
    hashes={rep:array_hash(features[rep]) for rep in REPRESENTATIONS}
    expected={'R0':'d84f1e8244b0a25838f1c9f12fa19f70437e7dd488388306f8fc35ffc3966828','R5':'8c02288566e92de5e0c8a2b87d0d636f25bf01d14eb8711292f78af28c4030ed'}
    if hashes!=expected:raise ValueError('canonical R0/R5 feature identities differ')
    if len(frame)!=19716 or frame.cluster_id.nunique()!=3587:raise ValueError('canonical training population size changed')
    canonical_identity=dict(source_sha256=sha256_file(ROOT/'data/rasmussen_et_al_dataset.csv'),splits_sha256=sha256_file(ROOT/'data/splits.csv'),data_hash=frame_hash(frame),outer_fold_hash=array_hash(nested.outer_folds(frame).astype('<i8')))
    if canonical_identity!=ranked['specification']['identity']:raise ValueError('completed ranking data/group/label identity differs')
    if ranked['protocol_hash']!='32d3f8f1908bd5bb3df81ed205c61771b567d2e349f92ebfa199da8111b0ef8d':raise ValueError('completed ranking identity differs')
    manifest=task_manifest(frame)
    specification=dict(schema='final_hybrids.1',canonical_ranking_identity=canonical_identity,dependency_versions=dependency_versions(),python_version=platform.python_version(),fit_graph=fit_graph(),scientific_specification_sha256=sha256_file(spec),scientific_graph_sha256=sha256_file(out/'fit_graph.json'),task_manifest_hash=config_digest(manifest),data_hash=frame_hash(frame),feature_hashes=hashes,components=list(COMPONENTS),representations=list(REPRESENTATIONS),outer_fold_vector_sha256=array_hash(nested.outer_folds(frame).astype('<i8')),selection_order=list(ORDER),blend_weights=list(WEIGHTS),member_axes=[list(x) for x in MEMBER_AXES],local_workers_max=2,threads_each=2,cloud_spend_usd=0,original_test_excluded=True)
    protocol=dict(status='FROZEN_NOT_EXECUTION_APPROVAL',protocol_hash=config_digest(specification),specification=specification,frozen_at_utc=now(),source_sha256={name:sha256_file(ROOT/name) for name in CODE},reference_sha256={name:sha256_file(ROOT/name) for name in REFERENCES},feature_provenance=provenance,ranking_protocol_sha256=sha256_file(ROOT/'reports/finalization/ranking/protocol.json'),raw_data_sha256=sha256_file(ROOT/'data/rasmussen_et_al_dataset.csv'),splits_sha256=sha256_file(ROOT/'data/splits.csv'))
    path=out/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=protocol[k] for k in protocol if k!='frozen_at_utc'):raise ValueError('refusing changed frozen hybrid protocol')
        return old
    atomic_json(out/'task_manifest.json',manifest);atomic_json(out/'preflight.json',check);write_once(path,protocol)
    return protocol

def validate_frozen(out=OUT):
    out=Path(out);protocol=json.loads((out/'protocol.json').read_text());spec=protocol['specification']
    if protocol['status']!='FROZEN_NOT_EXECUTION_APPROVAL' or config_digest(spec)!=protocol['protocol_hash']:raise ValueError('hybrid protocol is not intact frozen specification')
    if dependency_versions()!=spec['dependency_versions'] or platform.python_version()!=spec['python_version']:raise ValueError('frozen dependency/Python versions changed')
    for name,h in protocol['source_sha256'].items():
        if sha256_file(ROOT/name)!=h:raise ValueError(f'frozen hybrid source changed: {name}')
    for name,h in protocol['reference_sha256'].items():
        if sha256_file(ROOT/name)!=h:raise ValueError('historical descriptive reference changed')
    for file,key in [(out/'SPECIFICATION.md','scientific_specification_sha256'),(out/'fit_graph.json','scientific_graph_sha256')]:
        if sha256_file(file)!=spec[key]:raise ValueError('scientific design changed')
    for file,key in [(ROOT/'data/rasmussen_et_al_dataset.csv','raw_data_sha256'),(ROOT/'data/splits.csv','splits_sha256'),(ROOT/'reports/finalization/ranking/protocol.json','ranking_protocol_sha256')]:
        if sha256_file(file)!=protocol[key]:raise ValueError('canonical input identity changed')
    frame=training_frame()
    if frame_hash(frame)!=spec['data_hash'] or array_hash(nested.outer_folds(frame).astype('<i8'))!=spec['outer_fold_vector_sha256']:raise ValueError('training population/grouping changed')
    if config_digest(json.loads((out/'task_manifest.json').read_text()))!=spec['task_manifest_hash'] or config_digest(task_manifest(frame))!=spec['task_manifest_hash']:raise ValueError('explicit task manifest changed')
    return frame,protocol

def validated_features(frame,protocol):
    features,provenance=load_features(frame)
    if provenance!=protocol['feature_provenance'] or {rep:array_hash(features[rep]) for rep in REPRESENTATIONS}!=protocol['specification']['feature_hashes']:raise ValueError('cached features/provenance changed')
    return features

def fit_bundle(*args,**kwargs):
    from pepstab.final_hybrids import fit_member_bundle
    from pepstab.final_cloud_protocol import durable_sync, exclusive_json, binding, now
    # The original component_job has already written its exclusive bundle intent.
    durable_sync()
    result = fit_member_bundle(*args,**kwargs)
    durable_sync()
    path = Path(kwargs['artifact_dir'])
    exclusive_json(path/'cloud_commit.json',dict(schema='final_cloud_bundle_commit.1',
        completed=binding(path/'completed.json'),committed_at_utc=now(),method='os.sync',volume_version=2))
    durable_sync()
    return result

def runtime():
    return dict(platform=platform.platform(),python=sys.version,versions={p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','catboost','threadpoolctl')},threadpools=threadpool_info(),threads=2,device='modal_cpu_sandbox',sandbox_id=os.environ.get('FINAL_CLOUD_SANDBOX_ID'),authorization_sha256=os.environ.get('FINAL_CLOUD_AUTH_SHA256'))

def require_authorization(out,approved,phase):
    from pepstab.final_cloud_protocol import scientific_authorization
    return scientific_authorization(out,approved,phase)


def component_job(out,approved,rep,fold,*,pilot=False):
    phase='pilot' if pilot else 'main';frame,protocol,auth=require_authorization(out,approved,phase)
    if rep not in REPRESENTATIONS or type(fold) is not int or fold not in range(5) or (pilot and fold!=0):raise ValueError('job outside exact frozen phase')
    directory=Path(out)/('pilot' if pilot else '')/f'fold_{fold}'/rep;directory.mkdir(parents=True,exist_ok=True)
    write_once(directory/'attempt.json',dict(pid=os.getpid(),protocol_hash=protocol['protocol_hash'],authorization_sha256=sha256_file(approved),phase=phase,representation=rep,outer_fold=fold,started_at_utc=now()))
    start=time.perf_counter();records=[];ancestry=Ancestry()
    try:
        features=validated_features(frame,protocol);position={int(p):i for i,p in enumerate(frame.pair_id)};X=lambda rows:features[rep][[position[int(p)] for p in rows.pair_id]]
        source_feature_hash=protocol['specification']['feature_hashes'][rep]
        inner_tables=[];final_raw={};calls=list(call_schedule(frame,fold));calls=calls[:1] if pilot else calls
        with threadpool_limits(limits=2):
            atomic_json(directory/'runtime.json',runtime())
            for role,i,call_fit,held in calls:
                predictions={name:[] for name in COMPONENTS};members=list(member_partitions(call_fit,held));members=members[:1] if pilot else members
                for stop,seed,gradient,stopping in members:
                    context=f'hybrids/{phase}/outer/{fold}/{rep}/{role}/'+(str(i) if i is not None else 'refit')+f'/stop/{stop}/seed/{seed}'
                    relative=Path('bundles')/(f'inner_{i}' if role=='inner' else 'final')/f'stop_{stop}_seed_{seed}';path=directory/relative
                    write_once(directory/(str(relative).replace('/','_')+'.launch.json'),dict(context=context,started_at_utc=now(),n_fits=6))
                    arrays=dict(gradient=X(gradient),stopping=X(stopping),held=X(held))
                    result=fit_bundle(arrays,gradient,stopping,unlabelled(held),representation=rep,seed=seed,context=context,artifact_dir=path,protocol_hash=protocol['protocol_hash'],source_feature_hash=source_feature_hash)
                    if result['n_fits']!=6 or set(result['predictions'])!=set(COMPONENTS):raise ValueError('six-fit bundle scope changed')
                    receipt=path/'completed.json'
                    for name,p in result['predictions'].items():
                        p=ev.validate_finite(p,f'{name} member')
                        if p.shape!=(len(held),):raise ValueError('bundle member row shape changed')
                        predictions[name].append(p)
                        ancestry.add(role='hybrid_member',fit=gradient,held=held,stopping=None if name in ('tree_rmse','tree_rank') else stopping,component=name,representation=rep,context=context,stop_fold=stop,seed=seed,scaler_rows=ancestry.rows(gradient),transform_rows=ancestry.rows(gradient))
                    records.append(dict(role=role,inner_fold=i,stop_fold=stop,seed=seed,representation=rep,context=context,path=str(relative),completed_sha256=sha256_file(receipt),identity=result['identity'],n_fits=6))
                average={name:np.mean(values,axis=0) for name,values in predictions.items()}
                if pilot:continue
                if role=='inner':
                    for name,p in average.items():inner_tables.append(held[['pair_id','cluster_id','allele']].assign(inner_fold=i,component=name,y_pred=p))
                else:final_raw=average
        atomic_json(directory/'bundle_manifest.json',records)
        if pilot:
            atomic_json(directory/'ancestry.json',ancestry.as_dict())
            audit=ancestry.audit(frame,calls[0][3]);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
            write_once(directory/'completed.json',dict(status='complete',phase='pilot',protocol_hash=protocol['protocol_hash'],representation=rep,outer_fold=fold,n_fits=6,engineering_only=True,held_scores_computed=False,reuse_into_scientific=False,fit_seconds=time.perf_counter()-start,max_rss_mib=rss,bundle_manifest_sha256=sha256_file(directory/'bundle_manifest.json'),runtime_sha256=sha256_file(directory/'runtime.json'),audit=audit,completed_at_utc=now()))
        else:
            fit,held=calls[-1][2:];inner=pd.concat(inner_tables,ignore_index=True);selection=replay_selection(fit,inner)
            best_blend=max(selection['blend_scores'],key=lambda r:r['mean']);best_component=max(selection['component_scores'],key=lambda r:r['mean'])
            selection.update(blend_order=list(WEIGHTS),selector_order=list(ORDER),blend_best=best_blend,blend_threshold=best_blend['mean']-best_blend['se'],selector_best=best_component,selector_threshold=best_component['mean']-best_component['se'],ranker_mapping_input_rows=ancestry.rows(fit),ranker_scale_transport='clean fixed-candidate pooled raw inner OOF; outer-fit labels only; smaller-inner-to-final scale transport; inner scores raw')
            final={name:np.asarray(p).copy() for name,p in final_raw.items()};final['tree_rank']=apply_affine(final_raw['tree_rank'],selection['ranker_mapping']);w=selection['blend_weight'];final['blend']=(1-w)*final['standard']+w*final['tree_rmse'];final['hybrid_selector']=final[selection['hybrid_component']].copy()
            for name in ('ranker_mapping','blend','hybrid_selector'):ancestry.add(role=name,fit=fit,held=held,selection=fit,representation=rep)
            atomic_csv(directory/'inner_predictions.csv',inner)
            atomic_csv(directory/'raw_final_predictions.csv',pd.concat([held[['pair_id','cluster_id','allele']].assign(outer_fold=fold,component=name,y_pred=p) for name,p in final_raw.items()],ignore_index=True))
            atomic_csv(directory/'predictions.csv',pd.concat([held[['pair_id','cluster_id','allele']].assign(outer_fold=fold,arm=f'{rep}_{name}',y_pred=final[name]) for name in SUFFIXES],ignore_index=True))
            atomic_json(directory/'selection.json',selection);atomic_json(directory/'ancestry.json',ancestry.as_dict());audit=ancestry.audit(frame,held)
            write_once(directory/'completed.json',dict(status='complete',phase='main',protocol_hash=protocol['protocol_hash'],representation=rep,outer_fold=fold,n_fits=48,n_bundles=len(records),n_held=len(held),audit=audit,completed_at_utc=now(),sha256={name:sha256_file(directory/name) for name in FILES}))
        write_once(directory/'exit_receipt.json',dict(status='complete',exit_code=0,pid=os.getpid(),phase=phase,n_fits=6 if pilot else 48,completed_sha256=sha256_file(directory/'completed.json'),finished_at_utc=now()))
    except BaseException as error:
        write_once(directory/'exit_receipt.json',dict(status='failed',pid=os.getpid(),phase=phase,error=repr(error),completed_bundles=len(records),finished_at_utc=now(),no_retry=True));raise

def verify_component(directory,protocol,fold,representation):
    directory=Path(directory);done=json.loads((directory/'completed.json').read_text())
    if (done.get('status'),done.get('phase'),done.get('protocol_hash'),done.get('outer_fold'),done.get('representation'),done.get('n_fits'),done.get('n_bundles'))!=('complete','main',protocol['protocol_hash'],fold,representation,48,8):raise ValueError('component completion identity/fit count changed')
    if set(done['sha256'])!=set(FILES):raise ValueError('incomplete component artifact hashes')
    for name,h in done['sha256'].items():
        if sha256_file(directory/name)!=h:raise ValueError('component artifact changed')
    manifest=json.loads((directory/'bundle_manifest.json').read_text())
    if len(manifest)!=8 or sum(b['n_fits'] for b in manifest)!=48:raise ValueError('bundle fit graph incomplete')
    for item in manifest:
        path=(directory/item['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path/'completed.json')!=item['completed_sha256']:raise ValueError('bundle completion hash/path changed')
    return done

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run-component','pilot-component'])
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--approved',type=Path,required=True)
    p.add_argument('--representation',choices=REPRESENTATIONS,required=True)
    p.add_argument('--fold',type=int,required=True)
    a=p.parse_args()
    component_job(a.out_dir,a.approved,a.representation,a.fold,pilot=a.mode=='pilot-component')

if __name__=='__main__':main()
