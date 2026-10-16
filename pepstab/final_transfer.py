"""Bounded finalisation transfer controls; original test is never an input.

The fixed ranker common-scale map uses pooled clean inner OOF predictions from
one fixed candidate. It transports scale from smaller inner training sets to
outer-fit refits and is a different procedure from the old nested ranker.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import evaluation as ev
from .features import build_features
from .mlp import MLPConfig, MLPRegressor
from .phase_a import allele_offsets, ranking_pairs, shrunk_allele_means
from .round2_nested import apply_affine, positive_affine, unlabelled
from .round2_transfer import (TransferAncestry, inner_partitions, outer_cells,
    pooled_median, validate_frame, validate_partition, transfer_bootstrap)

SEQUENCE_ORDER = ('original', 'mlp_centred', 'tree_log', 'tree_centred', 'fixed_ranker8')
SEQUENCE_SELECTOR = 'expanded_sequence_selector'
REUSED_ARMS = ('original', 'mlp_centred', 'R4_ridge', 'R5_mlp', 'embedding_selector')
NEW_PROCEDURES = ('tree_log', 'tree_centred', 'R0_ridge', 'R0_mlp', 'fixed_ranker8')
ARMS = REUSED_ARMS + NEW_PROCEDURES + (SEQUENCE_SELECTOR,)
TREE_PARAMS = dict(iterations=2000, od_type='Iter', od_wait=100,
    bootstrap_type='Bernoulli', subsample=.8, rsm=1., random_strength=1.,
    border_count=128, learning_rate=.06, l2_leaf_reg=10., thread_count=2,
    verbose=False, allow_writing_files=False)
CANDIDATES = tuple(dict(id=f'tree_{target}_d{depth}', procedure=f'tree_{target}',
    learner='rmse', target=target, depth=depth, **({'k':20} if target=='centred' else {}))
    for target in ('log', 'centred') for depth in (4,8)) + tuple(
    dict(id=f'R0_{learner}_{value:g}', procedure=f'R0_{learner}', learner=learner,
         regularization=value) for learner,grid in [('ridge',(10000.,100.,1.)),('mlp',(1.,.01,.0001))]
    for value in grid) + (dict(id='fixed_ranker8', procedure='fixed_ranker8',
                              learner='pairlogit', target='log', depth=8),)


def fit_graph():
    regimes, total = {}, {p:0 for p in NEW_PROCEDURES}
    for regime,cells,inner in [('hla',5,3),('joint',25,9)]:
        per = dict(tree_log=(inner*2+1)*6, tree_centred=(inner*2+1)*6,
                   R0_ridge=inner*3+1, R0_mlp=inner*3*2+30,
                   fixed_ranker8=(inner+1)*6)
        regimes[regime] = dict(outer_cells=cells, inner_cells=inner,
            fits_per_cell=sum(per.values()), members_per_procedure=per,
            total_fits=cells*sum(per.values()))
        for p in total: total[p] += cells*per[p]
    assert total == dict(tree_log=3060,tree_centred=3060,R0_ridge=750,R0_mlp=2340,fixed_ranker8=1620)
    return dict(regimes=regimes, by_procedure=total, total_fits=sum(total.values()),
                reused_fits=0, selector_extra_fits=0, mapping_extra_fits=0)


def member_schedule(fit, regime, candidate, *, final=False):
    if candidate not in CANDIDATES: raise ValueError('candidate outside frozen final transfer registry')
    if candidate['learner']=='ridge':
        yield dict(gradient=fit, stopping=None, seed=None)
        return
    feature_head = candidate['learner']=='mlp'
    stops = list(inner_partitions(fit,regime,5 if feature_head else 3,stopping=True))
    axes = ([(s,seed) for s in range(5) for seed in range(6)] if final else [(0,0),(1,1)]) if feature_head else [(s,seed) for s in range(3) for seed in range(2)]
    for stop,seed in axes:
        cell,gradient,stopping = stops[stop]
        spec = dict(gradient=gradient,stopping=stopping,stopping_cell=cell,seed=seed)
        if feature_head:
            spec['config'] = MLPConfig(hidden=(64,),l2=candidate['regularization'],max_epochs=200,patience=15,seed=seed)
        yield spec


def tree_fit(X,y,allele,Xstop,ystop,astop,Xheld,candidate,seed):
    from catboost import CatBoost, CatBoostRegressor, Pool
    ranking = candidate['learner']=='pairlogit'
    params = dict(TREE_PARAMS,depth=candidate['depth'],random_seed=seed,
        loss_function='PairLogit' if ranking else 'RMSE',eval_metric='PairLogit' if ranking else 'RMSE')
    def pool(x,target,groups):
        groups=np.asarray(groups,dtype=str); order=np.argsort(groups,kind='stable')
        groups,target=groups[order],np.asarray(target)[order]
        hi,lo,weight=ranking_pairs(groups,target)
        if not len(hi): raise ValueError('no untied within-allele pairs; no fallback')
        return Pool(x[order],target,group_id=groups,pairs=np.column_stack([hi,lo]),pairs_weight=weight/weight.mean())
    model = CatBoost(params) if ranking else CatBoostRegressor(**params)
    if ranking: model.fit(pool(X,y,allele),eval_set=pool(Xstop,ystop,astop),use_best_model=True)
    else: model.fit(X,y,eval_set=(Xstop,ystop),use_best_model=True)
    return np.asarray(model.predict(Xheld)),dict(config=params,best_iteration=int(model.get_best_iteration()),tree_count=int(model.tree_count_))


def ranker_task(task):
    """Explicit partitions only: no worker-side selection or inferred partitioning."""
    import importlib.metadata
    import platform
    import resource
    payload={k:v for k,v in task.items() if k!='task_id'}
    if task.get('task_id')!=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(): raise ValueError('task identity hash mismatch')
    gradient,stop,held = [pd.DataFrame(task[key]) for key in ('gradient','stopping','held')]
    candidate = next(c for c in CANDIDATES if c['id']=='fixed_ranker8')
    if task.get('candidate') != candidate or task.get('seed') not in (0,1): raise ValueError('ranker task recipe changed')
    if any(k in held for k in ('y_log1p','thalf_hours')): raise ValueError('worker received assessment labels')
    for a,b in ((gradient,held),(stop,held),(gradient,stop)): validate_partition(a,b,task['regime'])
    start=time.perf_counter()
    arrays=[build_features(f,'pep_pseudo','onehot').astype(np.float32) for f in (gradient,stop,held)]
    pred,details=tree_fit(arrays[0],gradient.y_log1p,gradient.allele,arrays[1],stop.y_log1p,stop.allele,arrays[2],candidate,task['seed'])
    elapsed=time.perf_counter()-start
    return dict(task_id=task['task_id'],pair_ids=held.pair_id.tolist(),predictions=ev.validate_finite(pred,'remote ranker').tolist(),details=details,
        worker_seconds=elapsed,fit_seconds=elapsed,platform=platform.platform(),max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','catboost','threadpoolctl')})


class ControlPredictor:
    def __init__(self,population,ancestry,*,callback=None,ranker_backend=None):
        validate_frame(population)
        self.position={int(p):i for i,p in enumerate(population.pair_id)}
        self.features=build_features(population,'pep_pseudo','onehot').astype(np.float32)
        self.feature_hash=hashlib.sha256(self.features.tobytes()).hexdigest()
        self.ancestry,self.callback,self.ranker_backend=ancestry,callback,ranker_backend
        self.members=[]; self.member_predictions=[]
    def X(self,frame): return self.features[[self.position[int(p)] for p in frame.pair_id]]
    def __call__(self,fit,held,candidate,context,*,final=False):
        validate_partition(fit,held,self.ancestry.regime)
        if any(k in held for k in ('y_log1p','thalf_hours')): raise ValueError('assessment labels forbidden')
        schedule=list(member_schedule(fit,self.ancestry.regime,candidate,final=final))
        remote={}
        if candidate['learner']=='pairlogit' and self.ranker_backend is not None:
            columns=['pair_id','cluster_id','hla_group','allele','split','peptide','hla_pseudoseq','y_log1p']
            tasks=[]
            for member in schedule:
                gradient,stop=member['gradient'],member['stopping']
                payload=dict(context=context,candidate=candidate,seed=member['seed'],stopping_cell=member['stopping_cell'],regime=self.ancestry.regime,
                    gradient=gradient[columns].to_dict('records'),stopping=stop[columns].to_dict('records'),held=held[[c for c in columns if c!='y_log1p']].to_dict('records'))
                payload['task_id']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                tasks.append(payload)
            outputs=self.ranker_backend(tasks)
            if len(outputs)!=len(tasks): raise ValueError('remote batch member coverage changed')
            for task,result in zip(tasks,outputs):
                if result['task_id']!=task['task_id'] or result['pair_ids']!=held.pair_id.tolist(): raise ValueError('remote identity changed')
                remote[(task['stopping_cell'],task['seed'])]=result
        values=[]
        for member in schedule:
            start=time.perf_counter(); gradient,stop=member['gradient'],member['stopping']
            frames=(gradient,held) if stop is None else (gradient,stop,held)
            arrays=[self.X(f) for f in frames]
            means,global_mean={},None
            if candidate.get('target')=='centred': means,global_mean=shrunk_allele_means(gradient.allele,gradient.y_log1p,20)
            offsets=lambda f: allele_offsets(f.allele,means,global_mean) if global_mean is not None else np.zeros(len(f))
            y=gradient.y_log1p.to_numpy()-offsets(gradient); scaler=None; details={}
            learner=candidate['learner']
            if learner in ('ridge','mlp'):
                scaler=StandardScaler().fit(arrays[0]); arrays=[scaler.transform(x).astype(np.float32) for x in arrays]
                details.update(scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),scaler_var=scaler.var_.tolist())
            if learner=='ridge':
                model=Ridge(alpha=candidate['regularization'],solver='lsqr',tol=1e-5,max_iter=2000).fit(arrays[0],y)
                pred=model.predict(arrays[1]); details['config']=dict(alpha=candidate['regularization'],solver='lsqr',tol=1e-5,max_iter=2000)
            elif learner=='mlp':
                cfg=member['config']; model=MLPRegressor(cfg).fit(arrays[0],y,arrays[1],stop.y_log1p.to_numpy())
                pred=model.predict(arrays[2]); details.update(config=asdict(cfg),best_epoch=model.best_epoch_,epochs=model.n_epochs_)
            elif learner=='pairlogit' and self.ranker_backend is not None:
                result=remote[(member['stopping_cell'],member['seed'])]
                pred=np.asarray(result['predictions']); details.update(result['details'],remote_meter={k:result[k] for k in ('worker_seconds','platform','max_rss_mib','versions')})
            else:
                pred,details=tree_fit(arrays[0],y,gradient.allele,arrays[1],stop.y_log1p.to_numpy()-offsets(stop),stop.allele,arrays[2],candidate,member['seed'])
                pred=pred+offsets(held)
            pred=ev.validate_finite(pred,'member prediction')
            if pred.shape!=(len(held),): raise ValueError('member shape mismatch')
            record=self.ancestry.add('control_member',gradient,held,stopping=stop,scaler=gradient if scaler else None,context=context,candidate=candidate['id'],seed=member['seed'],stopping_cell=member.get('stopping_cell'))
            item=dict(context=context,candidate=candidate['id'],seed=member['seed'],stopping_cell=member.get('stopping_cell'),final_recipe=final,
                n_fit=len(gradient),n_stop=0 if stop is None else len(stop),n_source=len(fit),excluded_cross_quadrant_rows=len(fit)-len(gradient)-(0 if stop is None else len(stop)),
                feature_hash=self.feature_hash,fit_seconds=time.perf_counter()-start,centering_means=means,centering_global=global_mean,**details,
                **{k:record[k] for k in ('fit_rows','stopping_rows','prediction_rows','scaler_rows') if k in record})
            self.members.append(item)
            self.member_predictions.append(dict(pair_ids=held.pair_id.to_numpy(dtype=np.int64),predictions=pred.copy()))
            if self.callback: self.callback(item)
            values.append(pred)
        return np.mean(values,axis=0)


def select_scores(scores):
    if [s['candidate'] for s in scores]!=[c['id'] for c in CANDIDATES]: raise ValueError('candidate score order changed')
    values=ev.validate_finite([s['pooled_inner_score'] for s in scores],'inner scores')
    return {p:max((i for i,c in enumerate(CANDIDATES) if c['procedure']==p),key=lambda i:values[i]) for p in NEW_PROCEDURES}


def assess_cell(fit,held,regime,predictor,ancestry,reused,*,min_rows=20):
    """Reused contains audited old scores/finals; no legacy raw OOF invention."""
    validate_partition(fit,held,regime)
    panel=ev.eligible_alleles(fit.allele,fit.y_log1p,min_rows=min_rows)
    if not panel: raise ValueError('empty fitting panel')
    index={pid:i for i,pid in enumerate(fit.pair_id)}; raw={}; inner_tables=[]
    for c in CANDIDATES:
        pred=np.full(len(fit),np.nan)
        for cell,train,target in inner_partitions(fit,regime):
            p=predictor(train,unlabelled(target),c,f"select/{c['id']}/{cell}")
            pred[[index[pid] for pid in target.pair_id]]=p
            ancestry.add('inner_prediction',train,target,candidate=c['id'],cell=cell)
            inner_tables.append(pd.DataFrame(dict(pair_id=target.pair_id,inner_cell=cell,candidate=c['id'],raw_prediction=p)))
        raw[c['id']]=ev.validate_finite(pred,'complete clean OOF')
    mapping=positive_affine(raw['fixed_ranker8'],fit.y_log1p,minimum_slope=1e-8)
    ancestry.add('fixed_ranker_affine_map',fit,held,selection=fit,candidate='fixed_ranker8',mapping_rows=ancestry.rows(fit))
    mapped={key:apply_affine(value,mapping) if key=='fixed_ranker8' else value for key,value in raw.items()}
    scores=[dict(candidate=c['id'],pooled_inner_score=pooled_median(fit.allele,fit.y_log1p,raw[c['id']],panel)) for c in CANDIDATES]
    chosen=select_scores(scores)
    final={name:ev.validate_finite(reused['predictions'][name],name).copy() for name in REUSED_ARMS}
    for name,p in final.items():
        if p.shape!=(len(held),): raise ValueError('reused row shape mismatch')
    final_raw={}
    for p,i in chosen.items():
        prediction=predictor(fit,unlabelled(held),CANDIDATES[i],f'final/{p}',final=True)
        final_raw[p]=prediction
        final[p]=apply_affine(prediction,mapping) if p=='fixed_ranker8' else prediction
        ancestry.add('selected_prediction',fit,held,selection=fit,candidate=CANDIDATES[i]['id'],arm=p)
    family_scores={name:float(reused['scores'][name]) for name in ('original','mlp_centred')}
    family_scores.update({p:scores[chosen[p]]['pooled_inner_score'] for p in SEQUENCE_ORDER[2:]})
    ev.validate_finite(list(family_scores.values()),'expanded family scores')
    winner=max(SEQUENCE_ORDER,key=lambda p:family_scores[p]); final[SEQUENCE_SELECTOR]=final[winner].copy()
    ancestry.add('expanded_selector',fit,held,selection=fit,arm=SEQUENCE_SELECTOR,selected_procedure=winner)
    table=pd.concat(inner_tables,ignore_index=True)
    table['mapped_prediction']=table.raw_prediction
    mask=table.candidate.eq('fixed_ranker8'); table.loc[mask,'mapped_prediction']=apply_affine(table.loc[mask,'raw_prediction'],mapping)
    selection=dict(scores=scores,selected={p:CANDIDATES[i]['id'] for p,i in chosen.items()},sequence_family_scores=family_scores,
        expanded_selected=winner,inner_panel=panel,ranker_mapping=mapping,mapping_scope='clean fixed-candidate OOF on outer-fit rows only; smaller-inner-to-outer training-size scale transport; inner scores use raw predictions; mapped inner values are not independently calibrated',
        reused_selection=reused['selection'],selection_rule='pooled inner median argmax; frozen candidate/family order breaks exact ties')
    return final,selection,table,final_raw


def comparisons(frame,predictions,*,min_rows=20,n_boot=2000):
    """Full paired uncertainty and ALL actual HLA-group deletions, no refits."""
    panel=ev.eligible_alleles(frame.allele,frame.y_log1p,min_rows=min_rows)
    point={p:pooled_median(frame.allele,frame.y_log1p,v,panel) for p,v in predictions.items()}
    contrasts=[(ref,p) for ref in ('original',SEQUENCE_SELECTOR) for p in ARMS if p!=ref]
    contrasts += [('R0_ridge','R4_ridge'),('R0_mlp','R5_mlp')]
    rows=[];boots={};deletions=[]
    for mode in ('peptide','hla','crossed'):
        boot,meta=transfer_bootstrap(frame,predictions,min_rows=min_rows,n_boot=n_boot,mode=mode); boots[mode]=boot
        for ref,p in contrasts:
            valid=(boot[p]-boot[ref]).dropna();lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid) else (np.nan,np.nan)
            rows.append(dict(baseline=ref,candidate=p,delta=point[p]-point[ref],baseline_score=point[ref],candidate_score=point[p],ci_low=lo,ci_high=hi,**{k:v for k,v in meta.items() if k!='panel'}))
    for group in sorted(frame.hla_group.unique()):
        keep=frame.hla_group.ne(group).to_numpy(); remaining=frame.loc[keep]; subpanel=[a for a in panel if a in set(remaining.allele)]
        values={p:pooled_median(remaining.allele,remaining.y_log1p,v[keep],subpanel) for p,v in predictions.items()}
        for ref,p in contrasts: deletions.append(dict(dropped_hla_group=int(group),dropped_rows=int((~keep).sum()),baseline=ref,candidate=p,delta=values[p]-values[ref],panel_min_rows=min_rows))
    return pd.DataFrame(rows),boots,pd.DataFrame(deletions)
