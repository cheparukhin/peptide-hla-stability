"""Metadata-only freeze; isolated cloud cells; report-only conditional uncertainty."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab import followup3_fewshot as study
OUT = ROOT / 'reports/followup3/fewshot'
SOURCES = ('pepstab/followup3_fewshot.py', 'scripts/followup3_fewshot.py', 'pepstab/mlp.py',
    'pepstab/features.py', 'pepstab/round2_transfer.py', 'pepstab/round2_reporting.py',
    'pepstab/final_cloud_targets.py', 'pepstab/data.py', 'pepstab/evaluation.py',
    'reports/followup3/fewshot/SPECIFICATION.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write_once(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())


def load_frame():
    from pepstab.final_cloud_targets import install, META
    install(ROOT / META)
    from scripts.round2_transfer import training_frame
    return training_frame()


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def freeze(out):
    frame = load_frame(); plan = study.make_plan(frame); panels=study.fixed_panels(frame,plan)
    if len(frame) != 19716 or frame.cluster_id.nunique() != 3587 or frame.hla_group.nunique() != 35:
        raise ValueError('canonical original training cohort changed')
    hybrid = json.loads((ROOT / 'reports/finalization/hybrids_cloud_v1/protocol.json').read_text())
    protocol = dict(schema='followup3_fewshot_protocol.1', status='frozen_no_execution_authorization',
        created_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=study.digest(plan),
        frame_sha256=frame_hash(frame), source_sha256={name: sha(ROOT / name) for name in SOURCES},
        data_sha256={name:sha(ROOT/name) for name in ('data/rasmussen_et_al_dataset.csv','data/splits.csv',
            'reports/further_research/round2/transfer/manifest.csv', 'reports/further_research/round2/transfer/protocol.json',
            'reports/finalization/cloud/canonical_training_targets.json','reports/finalization/cloud/canonical_training_targets.npz')},
        feature_hashes={rep:hybrid['specification']['feature_hashes'][rep] for rep in ('R0','R5')},
        source_ridge_alpha=study.SOURCE_ALPHA, adapter_alpha=study.ADAPTER_ALPHA,
        primary='mean_draw(score_R5_k20-score_R5_k0)', uncertainty_draws=2000,
        uncertainty_seed=20261003, source_fits=450, target_adapter_fits=plan['adapter_fits'],
        fixed_panels={str(k):v for k,v in panels.items()})
    protocol['protocol_hash'] = study.digest(protocol)
    write_once(out / 'task_manifest.json', plan); write_once(out / 'protocol.json', protocol)
    pd.DataFrame(plan['supports']).to_csv(out / 'support_manifest.csv', index=False)
    rows = [dict(index=c['index'], draw=c['draw'], hla_fold=c['hla_fold'], peptide_fold=c['peptide_fold'],
        source_rows=len(c['source_ids']), query_rows=len(c['query_ids']), support_maximum_labels=sum(map(len,c['support_ids'].values())),
        source_ridge_fits=2, source_mlp_fits=4,
        adapter_fits=len(c['support_ids'])*3*3) for c in plan['cells']]
    pd.DataFrame(rows).to_csv(out / 'cell_inventory.csv', index=False)
    query_inventory=[]
    lookup=frame.set_index('pair_id',drop=False)
    for draw in study.DRAWS:
        query=lookup.loc[[r['pair_id'] for r in plan['queries'] if r['draw']==draw]]
        for allele in plan['support_candidates']:
            part=query.loc[query.allele.eq(allele)]
            query_inventory.append(dict(draw=draw,allele=allele,query_rows=len(part),query_clusters=int(part.cluster_id.nunique()),
                panel_min20=allele in panels[20],panel_min50=allele in panels[50]))
    pd.DataFrame(query_inventory).to_csv(out/'query_inventory.csv',index=False)
    return dict(protocol_hash=protocol['protocol_hash'], n_cells=len(rows), source_fits=450,
        adapter_fits=plan['adapter_fits'], support_alleles=len(plan['support_candidates']),
        panel_alleles={str(k):len(v) for k,v in panels.items()},
        source_rows_range=[min(r['source_rows'] for r in rows),max(r['source_rows'] for r in rows)])


def context(out, protocol_path=None):
    protocol_path=Path(protocol_path) if protocol_path is not None else out/'protocol.json'
    if protocol_path.resolve().parent!=out.resolve():raise ValueError('protocol must reside in study output directory')
    protocol=json.loads(protocol_path.read_text()); identity=protocol.pop('protocol_hash')
    if study.digest(protocol)!=identity: raise ValueError('protocol identity changed')
    protocol['protocol_hash']=identity
    amendment=protocol.get('operational_amendment')
    if amendment is not None:
        if (sha(out/amendment['original_protocol_file'])!=amendment['original_protocol_sha256']
                or sha(out/'SCALER_REPLAY_AMENDMENT.md')!=amendment['amendment_sha256']
                or sha(ROOT/'tests/test_followup3_fewshot.py')!=amendment['regression_tests_sha256']
                or amendment['scientific_recipe_changed'] is not False):
            raise ValueError('operational amendment/original protocol/regression binding differs')
    for key in ('source_sha256','data_sha256'):
        for name,expected in protocol[key].items():
            if sha(ROOT/name)!=expected: raise ValueError(f'frozen source/input changed: {name}')
    plan=json.loads((out/'task_manifest.json').read_text())
    if study.digest(plan)!=protocol['plan_sha256']: raise ValueError('frozen task manifest changed')
    frame=load_frame()
    if frame_hash(frame)!=protocol['frame_sha256'] or study.digest(study.make_plan(frame))!=protocol['plan_sha256']:
        raise ValueError('canonical frame or label-blind ancestry differs')
    if {str(k):v for k,v in study.fixed_panels(frame,plan).items()}!=protocol['fixed_panels']:
        raise ValueError('frozen assessment panel differs')
    return frame,protocol,plan


def amend_replay(out):
    """Metadata-only operational correction; retain original freeze unchanged."""
    original_path=out/'protocol.json';original=json.loads(original_path.read_text())
    identity=original.pop('protocol_hash')
    if study.digest(original)!=identity:raise ValueError('original frozen identity differs')
    original['protocol_hash']=identity
    changed=[name for name,expected in original['source_sha256'].items() if sha(ROOT/name)!=expected]
    if changed!=['scripts/followup3_fewshot.py']:
        raise ValueError('replay amendment permits only the fitting/serialization script correction')
    for name,expected in original['data_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('scientific input changed')
    plan=json.loads((out/'task_manifest.json').read_text());frame=load_frame()
    if (study.digest(plan)!=original['plan_sha256'] or study.digest(study.make_plan(frame))!=original['plan_sha256']
            or frame_hash(frame)!=original['frame_sha256']
            or {str(k):v for k,v in study.fixed_panels(frame,plan).items()}!=original['fixed_panels']):
        raise ValueError('replay amendment changed scientific population/partition/panel')
    updated={k:v for k,v in original.items() if k!='protocol_hash'}
    updated['source_sha256']={name:sha(ROOT/name) for name in original['source_sha256']}
    updated['operational_amendment']=dict(schema='followup3_fewshot_scaler_replay_amendment.1',
        original_protocol_file='protocol.json',original_protocol_sha256=sha(original_path),
        original_protocol_hash=identity,amendment_sha256=sha(out/'SCALER_REPLAY_AMENDMENT.md'),
        regression_tests_sha256=sha(ROOT/'tests/test_followup3_fewshot.py'),
        reason='restore saved StandardScaler and use authoritative transform; exact replay retained',
        scientific_recipe_changed=False,failed_attempts_must_be_preserved=True,
        amended_at_utc=datetime.now(timezone.utc).isoformat())
    updated['protocol_hash']=study.digest(updated)
    path=out/'protocol_replay_v2.json';write_once(path,updated)
    return dict(protocol_path=str(path),protocol_hash=updated['protocol_hash'],
        original_protocol_preserved_sha256=sha(original_path),scientific_recipe_changed=False)


def require_cloud():
    if sys.platform!='linux' or os.environ.get('FOLLOWUP3_AUTHORIZED')!='1':
        raise ValueError('scientific fitting/reporting requires root-authorized Modal/Linux execution')


def load_features(path, frame, protocol):
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!={'pair_id','R0','R5','R0_blosum'}:raise ValueError('shared feature bundle keys differ')
        if not np.array_equal(saved['pair_id'],frame.pair_id.to_numpy()):raise ValueError('feature population/order differs')
        features={name:saved[name].copy() for name in ('R0','R5','R0_blosum')}
    for name,value in features.items():
        if value.dtype!=np.float32 or value.ndim!=2 or len(value)!=len(frame) or not np.isfinite(value).all():
            raise ValueError('finite aligned float32 cached features required')
        if name in protocol['feature_hashes'] and study.array_hash(value)!=protocol['feature_hashes'][name]:
            raise ValueError('canonical cached representation identity differs')
    from pepstab.features import build_features
    if not np.array_equal(features['R0_blosum'],build_features(frame,'pep_pseudo','blosum').astype(np.float32)):
        raise ValueError('sequence BLOSUM encoding differs')
    return features


def restore_source_scaler(arrays, source_ids):
    """Restore the estimator, retaining its authoritative dtype semantics."""
    from sklearn.preprocessing import StandardScaler
    np.testing.assert_array_equal(arrays['source_ids'],np.asarray(source_ids),
        err_msg='saved scaler source-fitting row identity differs')
    scaler=StandardScaler()
    for name in ('mean','scale','var'):setattr(scaler,name+'_',arrays[f'scaler_{name}'].copy())
    scaler.n_features_in_=int(arrays['scaler_n_features'])
    scaler.n_samples_seen_=arrays['scaler_n_samples'].copy()
    if (scaler.mean_.shape!=(scaler.n_features_in_,) or scaler.scale_.shape!=scaler.mean_.shape
            or scaler.var_.shape!=scaler.mean_.shape or int(scaler.n_samples_seen_)!=len(source_ids)):
        raise ValueError('saved scaler fitted shape/sample metadata differs')
    return scaler


def replay_source(arrays, X):
    if 'coef' in arrays:return X@arrays['coef']+arrays['intercept']
    value=X
    depth=int(arrays['depth'])
    for i in range(depth):
        value=value@arrays[f'w{i}']+arrays[f'b{i}']
        if i<depth-1:value=np.maximum(value,0.)
    return value.ravel()+float(arrays['offset'])


def run_cell(out, index, features_path, protocol_path=None):
    require_cloud()
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from pepstab.mlp import MLPConfig, MLPRegressor
    from pepstab.round2_transfer import inner_partitions
    frame,protocol,plan=context(out,protocol_path)
    if index not in range(len(plan['cells'])):raise ValueError('cell index outside frozen study')
    cell=plan['cells'][index];directory=out/'cells'/f'cell_{index:03d}'
    directory.mkdir(parents=True,exist_ok=True)
    write_once(directory/'attempt.json',dict(protocol_hash=protocol['protocol_hash'],cell_index=index,
        started_at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
    started=time.monotonic();features=load_features(features_path,frame,protocol)
    positions={int(p):i for i,p in enumerate(frame.pair_id)}
    part=lambda ids: frame.iloc[[positions[p] for p in ids]]
    source=part(cell['source_ids']);query=part(cell['query_ids'])
    support_ids=[p for ids in cell['support_ids'].values() for p in ids]
    support=part(support_ids);study.validate_roles(source,support,query)
    indices=lambda f: np.asarray([positions[int(p)] for p in f.pair_id])
    X=lambda name,f: features[name][indices(f)]
    predict_frame=pd.concat([support,query],ignore_index=True)
    n_support=len(support);base={};scaled={};members=[];adapters=[];adapter_arrays={}
    with threadpool_limits(limits=2):
        for rep in ('R0','R5'):
            begin=time.monotonic();scaler=StandardScaler().fit(X(rep,source))
            zs=scaler.transform(X(rep,source)).astype(np.float32)
            zp=scaler.transform(X(rep,predict_frame)).astype(np.float32)
            model=Ridge(alpha=study.SOURCE_ALPHA,solver='lsqr',tol=1e-5,max_iter=2000).fit(zs,source.y_log1p)
            prediction=model.predict(zp);base[rep]=prediction.astype(np.float64);scaled[rep]=zp
            state=dict(coef=model.coef_,intercept=np.asarray(model.intercept_),scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_,scaler_var=scaler.var_,source_ids=source.pair_id.to_numpy(),
                scaler_n_samples=np.asarray(scaler.n_samples_seen_),scaler_n_features=np.asarray(scaler.n_features_in_),
                prediction=prediction)
            member_path=directory/f'{rep}_ridge.npz';np.savez_compressed(member_path,**state)
            with np.load(member_path,allow_pickle=False) as saved:
                restored_scaler=restore_source_scaler(saved,source.pair_id.to_numpy())
                for name in ('mean','scale','var'):
                    np.testing.assert_array_equal(getattr(restored_scaler,name+'_'),getattr(scaler,name+'_'))
                transformed=restored_scaler.transform(X(rep,predict_frame)).astype(np.float32)
                np.testing.assert_array_equal(restored_scaler.transform(X(rep,source)).astype(np.float32),zs)
                np.testing.assert_array_equal(transformed,zp)
                np.testing.assert_array_equal(replay_source(saved,transformed),prediction)
            members.append(dict(arm=rep,kind='ridge',source_ids=cell['source_ids'],gradient_ids=cell['source_ids'],
                stopping_ids=[],scaler_ids=cell['source_ids'],seconds=time.monotonic()-begin,
                label_sha256=study.array_hash(source.y_log1p.to_numpy()),payload_sha256=sha(member_path)))
        stopping=list(inner_partitions(source,'joint',5,stopping=True));sequence=[]
        for encoding,name in [('onehot','R0'),('blosum','R0_blosum')]:
            for stop,seed in ((0,0),(1,1)):
                stop_cell,gradient,stop_frame=stopping[stop]
                study.validate_roles(gradient,support,query);study.validate_roles(stop_frame,support,query)
                begin=time.monotonic();cfg=MLPConfig(hidden=(256,64),l2=1e-5 if encoding=='onehot' else 1e-3,
                    max_epochs=300,patience=15,seed=seed)
                model=MLPRegressor(cfg).fit(X(name,gradient),gradient.y_log1p.to_numpy(),X(name,stop_frame),stop_frame.y_log1p.to_numpy())
                prediction=model.predict(X(name,predict_frame));sequence.append(prediction.astype(np.float64))
                state=dict(depth=np.asarray(len(model.weights_)),offset=np.asarray(model.y_offset_),prediction=prediction)
                for i,(w,b) in enumerate(zip(model.weights_,model.biases_)):state[f'w{i}']=w;state[f'b{i}']=b
                member_path=directory/f'sequence_{encoding}_{seed}.npz';np.savez_compressed(member_path,**state)
                with np.load(member_path,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(replay_source(saved,X(name,predict_frame)),prediction)
                members.append(dict(arm='sequence_anchor',kind='mlp',encoding=encoding,seed=seed,stopping_cell=stop_cell,
                    gradient_ids=gradient.pair_id.astype(int).tolist(),stopping_ids=stop_frame.pair_id.astype(int).tolist(),
                    source_ids=cell['source_ids'],scaler_ids=[],configuration=asdict(cfg),best_epoch=model.best_epoch_,epochs=model.n_epochs_,
                    gradient_label_sha256=study.array_hash(gradient.y_log1p.to_numpy()),
                    stopping_label_sha256=study.array_hash(stop_frame.y_log1p.to_numpy()),
                    seconds=time.monotonic()-begin,payload_sha256=sha(member_path),stopping_curve=model.dev_curve_))
        base['sequence_anchor']=np.mean(sequence,axis=0)
        support_position={int(pid):i for i,pid in enumerate(support.pair_id)}
        result=query[['pair_id','cluster_id','hla_group','hla_fold','peptide_fold','allele','split','y_log1p']].copy()
        for arm in study.ARMS:
            features_rep='R0' if arm=='sequence_anchor' else arm
            z=scaled[features_rep];zero=base[arm][n_support:]
            for budget in study.BUDGETS:
                pred=zero.copy()
                if budget:
                    for allele,ids in cell['support_ids'].items():
                        si=np.asarray([support_position[pid] for pid in ids[:budget]])
                        qi=np.flatnonzero(query.allele.eq(allele).to_numpy())
                        residual=support.y_log1p.to_numpy()[si]-base[arm][si]
                        update,state=study.fit_adapter(z[si],residual,z[n_support+qi])
                        pred[qi]+=update
                        key=f'adapter_{len(adapters)}'
                        for component,value in state.items():adapter_arrays[f'{key}_{component}']=value
                        replay=((z[n_support+qi].astype(np.float64)-state['mean'])@
                            (z[si].astype(np.float64)-state['mean']).T/z.shape[1])@state['dual']+float(state['offset'])
                        np.testing.assert_array_equal(replay,update)
                        adapters.append(dict(key=key,arm=arm,budget=budget,allele=allele,
                            support_ids=ids[:budget],query_ids=query.iloc[qi].pair_id.astype(int).tolist(),
                            labels=budget,stopping_labels=0,tuning_labels=0,
                            support_label_sha256=study.array_hash(support.y_log1p.to_numpy()[si])))
                if not np.isfinite(pred).all():raise ValueError('nonfinite fitted predictions')
                result[f'{arm}_k{budget}']=pred
    result.to_csv(directory/'predictions.csv',index=False)
    csv=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
    for arm in study.ARMS:
        for k in study.BUDGETS:np.testing.assert_array_equal(csv[f'{arm}_k{k}'],result[f'{arm}_k{k}'])
    np.savez_compressed(directory/'adapters.npz',**adapter_arrays)
    with np.load(directory/'adapters.npz',allow_pickle=False) as saved:
        for name,value in adapter_arrays.items():np.testing.assert_array_equal(saved[name],value)
    write_once(directory/'members.json',members);write_once(directory/'adapters.json',adapters)
    for member in members:
        if not (set(member['gradient_ids']) | set(member['stopping_ids'])) <= set(cell['source_ids']):
            raise ValueError('source member consumed labels outside frozen source pool')
    source_labels_by_arm={arm:len(set(p for member in members if member['arm']==arm
        for p in member['gradient_ids']+member['stopping_ids'])) for arm in study.ARMS}
    completion=dict(status='passed',protocol_hash=protocol['protocol_hash'],cell_index=index,source_fits=len(members),
        adapter_fits=len(adapters),source_rows=len(source),source_clusters=int(source.cluster_id.nunique()),
        source_label_union=len(set(p for member in members for p in member['gradient_ids']+member['stopping_ids'])),
        source_label_union_by_arm=source_labels_by_arm,
        maximum_target_labels=len(support),target_labels_reused_across_query_cells=True,
        saved_models_predictions_replayed_exactly=True,saved_adapters_replayed_exactly=True,
        seconds=time.monotonic()-started,files={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
    write_once(directory/'completed.json',completion)
    return completion


def comparison_specs():
    contrasts=[('primary_R5_adaptation20',{'R5_k20':1.,'R5_k0':-1.})]
    for arm in study.ARMS:
        for k in study.BUDGETS[1:]:
            if not (arm=='R5' and k==20):contrasts.append((f'{arm}_adaptation{k}',{f'{arm}_k{k}':1.,f'{arm}_k0':-1.}))
    contrasts.append(('adaptation_interaction20',{'R5_k20':1.,'R5_k0':-1.,'R0_k20':-1.,'R0_k0':1.}))
    for k in study.BUDGETS:
        for ref in ('R0','sequence_anchor'):
            contrasts.append((f'R5_minus_{ref}_k{k}',{f'R5_k{k}':1.,f'{ref}_k{k}':-1.}))
    return contrasts


def report(out,protocol_path=None):
    require_cloud();frame,protocol,plan=context(out,protocol_path);lookup=frame.set_index('pair_id',drop=False)
    tables={draw:[] for draw in study.DRAWS};fits=0;adapter_fits=0;source_budgets=[]
    for cell in plan['cells']:
        directory=out/'cells'/f"cell_{cell['index']:03d}";done=json.loads((directory/'completed.json').read_text())
        if done['status']!='passed' or done['protocol_hash']!=protocol['protocol_hash'] or done['cell_index']!=cell['index']:
            raise ValueError('missing bound cell completion')
        if done['source_fits']!=6 or done['adapter_fits']!=len(cell['support_ids'])*9:
            raise ValueError('per-cell source/adapter fit coverage differs')
        for arm,used in done['source_label_union_by_arm'].items():
            source_budgets.append(dict(cell_index=cell['index'],draw=cell['draw'],hla_fold=cell['hla_fold'],
                peptide_fold=cell['peptide_fold'],arm=arm,available_source_labels=len(cell['source_ids']),
                actual_source_label_union=used,source_clusters=done['source_clusters']))
        for name,expected in done['files'].items():
            if sha(directory/name)!=expected:raise ValueError('completed payload changed')
        table=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
        if table.pair_id.tolist()!=cell['query_ids']:raise ValueError('query IDs differ')
        for column in ('allele','cluster_id','hla_group','y_log1p'):
            np.testing.assert_array_equal(table[column].to_numpy(),lookup.loc[cell['query_ids'],column].to_numpy())
        tables[cell['draw']].append(table);fits+=done['source_fits'];adapter_fits+=done['adapter_fits']
    if fits!=450 or adapter_fits!=plan['adapter_fits']:raise ValueError('frozen fit graph incomplete')
    tables={draw:pd.concat(parts).sort_values('pair_id').reset_index(drop=True) for draw,parts in tables.items()}
    for draw,table in tables.items():
        expected=sorted(r['pair_id'] for r in plan['queries'] if r['draw']==draw)
        if table.pair_id.tolist()!=expected or table.pair_id.duplicated().any():raise ValueError('pooled query coverage differs')
    panels=study.fixed_panels(frame,plan);comparisons=[];metadata={};points=None
    for mode in ('crossed','peptide','hla'):
        point,boot,meta=study.conditional_scores(frame,tables,panels,mode=mode)
        points=point;metadata[mode]=meta
        absolute=point.groupby(['panel_min_rows','arm','budget'],as_index=False).score.mean()
        for threshold,panel in panels.items():
            scores={f'{r.arm}_k{r.budget}':float(r.score) for r in absolute.loc[absolute.panel_min_rows.eq(threshold)].itertuples()}
            for name,weights in comparison_specs():
                estimate=sum(scores[a]*w for a,w in weights.items());distribution=sum(boot[threshold][a]*w for a,w in weights.items())
                low,high=np.percentile(distribution,[2.5,97.5])
                family_low,family_high=np.percentile(distribution,[1.25,98.75])
                comparisons.append(dict(contrast=name,panel_min_rows=threshold,mode=mode,delta=estimate,ci_low=float(low),ci_high=float(high),
                    family_ci_low=float(family_low),family_ci_high=float(family_high),family_interval_level=0.975,
                    bootstrap_mean_minus_point=float(distribution.mean()-estimate),role='primary' if name=='primary_R5_adaptation20' and threshold==20 and mode=='crossed' else 'secondary'))
        for threshold,vectors in boot.items():pd.DataFrame(vectors).to_csv(out/f'bootstrap_{mode}_min{threshold}.csv',index=False)
    points.to_csv(out/'scores_by_draw.csv',index=False);absolute.to_csv(out/'absolute_curves.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    deletions=[]
    from pepstab.round2_reporting import _weighted_rho
    for threshold,panel in panels.items():
        for group in sorted(frame.hla_group.unique()):
            retained=[a for a in panel if frame.loc[frame.allele.eq(a),'hla_group'].iloc[0]!=group]
            scores={}
            for arm in study.ARMS:
                for k in study.BUDGETS:
                    name=f'{arm}_k{k}';draw_scores=[]
                    for draw in study.DRAWS:
                        rhos=[]
                        for allele in retained:
                            part=tables[draw].loc[tables[draw].allele.eq(allele)]
                            rhos.append(float(_weighted_rho(part.y_log1p.to_numpy(),part[name].to_numpy(),np.ones((1,len(part))))[0]))
                        draw_scores.append(float(np.median(rhos)))
                    scores[name]=float(np.mean(draw_scores))
            for name,weights in comparison_specs():deletions.append(dict(panel_min_rows=threshold,dropped_hla_group=int(group),contrast=name,delta=sum(scores[a]*w for a,w in weights.items())))
    pd.DataFrame(deletions).to_csv(out/'all_hla_group_deletions.csv',index=False)
    budget_rows=[]
    for draw in study.DRAWS:
        for k in study.BUDGETS:
            selected=[r for r in plan['supports'] if r['draw']==draw and r['support_order']<=k]
            budget_rows.append(dict(draw=draw,k=k,target_labels=len(selected),target_alleles=len(plan['support_candidates']),
                unique_target_clusters=len(set(r['cluster_id'] for r in selected)),target_tuning_labels=0,target_stopping_labels=0,
                query_rows=len(tables[draw]),labels_per_allele=k,labels_reused_across_five_query_cells=True))
    pd.DataFrame(budget_rows).to_csv(out/'actual_budgets.csv',index=False)
    pd.DataFrame(source_budgets).to_csv(out/'source_budgets.csv',index=False)
    audit=dict(status='passed',protocol_hash=protocol['protocol_hash'],source_fits=fits,target_adapter_fits=adapter_fits,
        cells=75,panels=panels,support_labels_separate_from_source=True,query_fixed_across_k=True,
        draw_scores_averaged=True,original_validation_test_used=False,
        saved_models_predictions_replayed_exactly=True,uncertainty=metadata)
    write_once(out/'audit.json',audit)
    primary=next(r for r in comparisons if r['role']=='primary')
    (out/'REPORT.md').write_text('# Few-shot unseen-HLA residual adaptation\n\n'+
        f"With 20 target-allele labels, R5 ranking gain versus 0 labels is **{primary['delta']:+.6f} [{primary['ci_low']:+.6f}, {primary['ci_high']:+.6f}]** on the fixed ≥20 panel.\n\n"+
        'Three frozen support draws are averaged as scores. Target support and query peptide families are excluded from source fitting. The same query cohort is used at every budget within each draw. All source stopping labels count; no target labels are used for tuning or stopping.\n\n'+
        'The sequence anchor is a fresh four-member historical two-layer recipe using both one-hot and BLOSUM encodings; it is a reduced ensemble and is not the unchanged 30-member reference or competitive selector. Source ridge heads and residual adapters are fixed. These are cached-feature development comparisons, not neural fine-tuning or prospective assay savings.\n\n'+
        'Crossed, peptide-only and HLA-only intervals use 2,000 shared full-population draws, conditional on fitted models and the three frozen support designs; they omit refitting, selection history and independent assay variability. Secondary intervals are nominal and unadjusted. All 35 actual HLA-group deletions are retained. See absolute_curves.csv, comparisons.csv, actual_budgets.csv, cell_inventory.csv and audit.json.\n')
    return {k:v for k,v in audit.items() if k!='uncertainty'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','amend-replay','run-cell','report'])
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--cell-index',type=int);parser.add_argument('--features',type=Path)
    parser.add_argument('--protocol',type=Path,help='Explicit amended protocol; original remains the default')
    args=parser.parse_args()
    if args.mode=='freeze':result=freeze(args.out)
    elif args.mode=='amend-replay':result=amend_replay(args.out)
    elif args.mode=='report':result=report(args.out,args.protocol)
    else:
        if args.cell_index is None or args.features is None:parser.error('run-cell requires --cell-index and --features')
        result=run_cell(args.out,args.cell_index,args.features,args.protocol)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
