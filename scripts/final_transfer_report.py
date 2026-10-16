"""Audit the entire bounded transfer graph before any uncertainty/report output."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pepstab import final_transfer as ft
from pepstab import evaluation as ev
from pepstab.features import build_features
from pepstab.phase_a import shrunk_allele_means
from pepstab.round2_nested import positive_affine,apply_affine
from pepstab.boltz_embeddings import sha256_file
from scripts.final_transfer import OUT,FILES,validate_frozen,verify_cell,reuse_cell,legacy_protocol,atomic_json,atomic_csv


def audit_cell(frame,fit,held,regime,cell,directory,protocol):
    directory=Path(directory); done=verify_cell(directory,protocol,regime,cell)
    if done['n_held']!=len(held): raise ValueError('outer held count changed')
    saved=json.loads((directory/'ancestry.json').read_text()); ancestry=ft.TransferAncestry(regime)
    if saved['regime']!=regime: raise ValueError('ancestry regime mismatch')
    ancestry.records,ancestry.rowsets=saved['records'],saved['rowsets'];ancestry.audit(frame,fit,held)
    reused=reuse_cell(frame,fit,held,regime,cell,legacy_protocol(protocol['legacy_directory']),protocol['legacy_directory'])
    if done['reused_evidence_sha256']!=reused['evidence_sha256']: raise ValueError('legacy reuse receipt mismatch')
    selection=json.loads((directory/'selection.json').read_text()); inner=pd.read_csv(directory/'raw_inner_predictions.csv')
    expected_panel=ev.eligible_alleles(fit.allele,fit.y_log1p,min_rows=20)
    if selection['inner_panel']!=expected_panel or selection['reused_selection']!=reused['selection']: raise ValueError('selection panel/reuse mismatch')
    partitions=list(ft.inner_partitions(fit,regime));raw={};scores=[]
    if set(inner.candidate)!={c['id'] for c in ft.CANDIDATES}: raise ValueError('inner candidate coverage changed')
    if inner.duplicated(['pair_id','candidate']).any(): raise ValueError('duplicate inner predictions')
    for candidate in ft.CANDIDATES:
        part=inner[inner.candidate.eq(candidate['id'])]
        if len(part)!=len(fit) or set(part.pair_id)!=set(fit.pair_id): raise ValueError('incomplete inner OOF coverage')
        for name,train,target in partitions:
            rows=part[part.inner_cell.eq(name)]
            if set(rows.pair_id)!=set(target.pair_id): raise ValueError('inner row ancestry changed')
        if set(part.inner_cell)!={p[0] for p in partitions}: raise ValueError('inner cell changed')
        aligned=fit[['pair_id']].merge(part[['pair_id','raw_prediction','mapped_prediction']],on='pair_id',validate='one_to_one')
        values=ev.validate_finite(aligned.raw_prediction,'saved raw OOF');raw[candidate['id']]=values
        scores.append(dict(candidate=candidate['id'],pooled_inner_score=ft.pooled_median(fit.allele,fit.y_log1p,values,expected_panel)))
    mapping=positive_affine(raw['fixed_ranker8'],fit.y_log1p,minimum_slope=1e-8)
    for k,v in mapping.items():
        if not np.isclose(selection['ranker_mapping'][k],v,rtol=1e-12,atol=1e-12): raise ValueError('affine map changed')
    for candidate in ft.CANDIDATES:
        part=inner[inner.candidate.eq(candidate['id'])]
        expected=apply_affine(part.raw_prediction,mapping) if candidate['id']=='fixed_ranker8' else part.raw_prediction
        if not np.allclose(part.mapped_prediction,expected,rtol=1e-12,atol=1e-12): raise ValueError('mapped inner values changed')
    chosen=ft.select_scores(scores)
    if [s['candidate'] for s in selection['scores']]!=[s['candidate'] for s in scores] or not np.allclose([s['pooled_inner_score'] for s in selection['scores']],[s['pooled_inner_score'] for s in scores],rtol=1e-12,atol=1e-12): raise ValueError('scores differ from raw inner OOF')
    if selection['selected']!={p:ft.CANDIDATES[i]['id'] for p,i in chosen.items()}: raise ValueError('candidate choice changed')
    family=dict(reused['scores']);family.update({p:scores[chosen[p]]['pooled_inner_score'] for p in ft.SEQUENCE_ORDER[2:]})
    winner=max(ft.SEQUENCE_ORDER,key=lambda p:family[p])
    if selection['expanded_selected']!=winner or set(selection['sequence_family_scores'])!=set(family) or not np.allclose([selection['sequence_family_scores'][p] for p in family],list(family.values()),rtol=1e-12,atol=1e-12): raise ValueError('expanded selector changed')
    members=json.loads((directory/'members.json').read_text());records=[r for r in ancestry.records if r['role']=='control_member']
    calls=[(f"select/{c['id']}/{name}",c,train,target,False) for c in ft.CANDIDATES for name,train,target in partitions]
    calls += [(f'final/{p}',ft.CANDIDATES[i],fit,held,True) for p,i in chosen.items()]
    features=build_features(frame,'pep_pseudo','onehot').astype(np.float32);feature_hash=hashlib.sha256(features.tobytes()).hexdigest();position={p:i for i,p in enumerate(frame.pair_id)};scalers={}
    archive=np.load(directory/'member_predictions.npz',allow_pickle=False)
    expected_keys={f'm{i:04d}_{key}' for i in range(len(members)) for key in ('pair_ids','predictions')}
    if set(archive.files)!=expected_keys:raise ValueError('member prediction archive coverage changed')
    expected_members=[];expected_records=[]
    for context,candidate,train,target,final in calls:
        for spec in ft.member_schedule(train,regime,candidate,final=final):
            gradient,stop=spec['gradient'],spec['stopping']; rows=dict(fit_rows=ancestry.rows(gradient),prediction_rows=ancestry.rows(target))
            if stop is not None:rows['stopping_rows']=ancestry.rows(stop)
            if candidate['learner'] in ('mlp','ridge'):rows['scaler_rows']=ancestry.rows(gradient)
            expected_records.append(dict(role='control_member',context=context,candidate=candidate['id'],seed=spec['seed'],stopping_cell=spec.get('stopping_cell'),**rows))
            expected_members.append((context,candidate,train,target,final,spec,rows))
    if records!=expected_records or len(members)!=len(expected_members) or done['n_fits']!=len(members):raise ValueError('member ancestry/fit graph changed')
    aggregated={}
    for number,(member,(context,candidate,train,target,final,spec,rows)) in enumerate(zip(members,expected_members)):
        ids=archive[f'm{number:04d}_pair_ids'];p=ev.validate_finite(archive[f'm{number:04d}_predictions'],'member archive')
        if not np.array_equal(ids,target.pair_id) or p.shape!=(len(target),):raise ValueError('member archive row identity changed')
        aggregated.setdefault(context,[]).append(p)
        gradient,stop=spec['gradient'],spec['stopping']
        for k,v in dict(context=context,candidate=candidate['id'],seed=spec['seed'],stopping_cell=spec.get('stopping_cell'),final_recipe=final,feature_hash=feature_hash,n_fit=len(gradient),n_stop=0 if stop is None else len(stop),n_source=len(train),excluded_cross_quadrant_rows=len(train)-len(gradient)-(0 if stop is None else len(stop)),**rows).items():
            if member.get(k)!=v:raise ValueError(f'member {k} changed')
        means,global_mean=shrunk_allele_means(gradient.allele,gradient.y_log1p,20) if candidate.get('target')=='centred' else ({},None)
        if member['centering_means']!=means or member['centering_global']!=global_mean:raise ValueError('centring ancestry changed')
        if candidate['learner'] in ('ridge','mlp'):
            key=rows['fit_rows']
            if key not in scalers:scalers[key]=StandardScaler().fit(features[[position[p] for p in gradient.pair_id]])
            scaler=scalers[key]
            if not np.allclose(member['scaler_mean'],scaler.mean_,rtol=1e-12,atol=1e-12) or not np.allclose(member['scaler_scale'],scaler.scale_,rtol=1e-12,atol=1e-12) or not np.allclose(member['scaler_var'],scaler.var_,rtol=1e-12,atol=1e-12):raise ValueError('gradient scaler changed')
            config=asdict(spec['config']) if candidate['learner']=='mlp' else dict(alpha=candidate['regularization'],solver='lsqr',tol=1e-5,max_iter=2000)
            config=json.loads(json.dumps(config))
        else:
            loss='PairLogit' if candidate['learner']=='pairlogit' else 'RMSE';config=dict(ft.TREE_PARAMS,depth=candidate['depth'],random_seed=spec['seed'],loss_function=loss,eval_metric=loss)
        if member['config']!=config:raise ValueError('member configuration changed')
        if not np.isfinite(member['fit_seconds']) or member['fit_seconds']<0:raise ValueError('member telemetry invalid')
    raw_finals=pd.read_csv(directory/'raw_final_predictions.csv')
    for context,candidate,train,target,final in calls:
        if final:
            part=raw_finals[raw_finals.procedure.eq(candidate['procedure'])]
        else:
            name=context.rsplit('/',1)[1];part=inner[inner.candidate.eq(candidate['id']) & inner.inner_cell.eq(name)]
        value=target[['pair_id']].merge(part[['pair_id','raw_prediction']],on='pair_id',how='left',validate='one_to_one').raw_prediction
        if not np.allclose(value,np.mean(aggregated[context],axis=0),rtol=1e-12,atol=1e-12):raise ValueError('aggregate prediction differs from member archive')
    archive.close()
    wrappers=[r for r in ancestry.records if r['role']!='control_member']; expected=[]
    for c in ft.CANDIDATES:
        for name,train,target in partitions:expected.append(dict(role='inner_prediction',fit_rows=ancestry.rows(train),prediction_rows=ancestry.rows(target),candidate=c['id'],cell=name))
    expected.append(dict(role='fixed_ranker_affine_map',fit_rows=ancestry.rows(fit),prediction_rows=ancestry.rows(held),selection_rows=ancestry.rows(fit),candidate='fixed_ranker8',mapping_rows=ancestry.rows(fit)))
    for p,i in chosen.items():expected.append(dict(role='selected_prediction',fit_rows=ancestry.rows(fit),prediction_rows=ancestry.rows(held),selection_rows=ancestry.rows(fit),candidate=ft.CANDIDATES[i]['id'],arm=p))
    expected.append(dict(role='expanded_selector',fit_rows=ancestry.rows(fit),prediction_rows=ancestry.rows(held),selection_rows=ancestry.rows(fit),arm=ft.SEQUENCE_SELECTOR,selected_procedure=winner))
    if wrappers!=expected:raise ValueError('wrapper/mapping ancestry changed')
    predictions=pd.read_csv(directory/'predictions.csv'); finals=pd.read_csv(directory/'raw_final_predictions.csv'); aligned=held[['pair_id','cluster_id','hla_group','allele','hla_fold','peptide_fold']].copy();identity=list(aligned.columns)
    if set(predictions.arm)!=set(ft.ARMS) or len(predictions)!=len(held)*len(ft.ARMS) or set(predictions.regime)!={regime} or set(predictions.cell)!={cell}:raise ValueError('outer prediction axes changed')
    if set(finals.procedure)!=set(ft.NEW_PROCEDURES) or len(finals)!=len(held)*len(ft.NEW_PROCEDURES):raise ValueError('raw final axes changed')
    for arm in ft.ARMS:
        part=predictions[predictions.arm.eq(arm)]; joined=aligned[identity].merge(part[identity+['y_pred']],on=identity,how='left',validate='one_to_one');aligned[arm]=ev.validate_finite(joined.y_pred,arm)
        if arm in reused['predictions'] and not np.allclose(aligned[arm],reused['predictions'][arm],rtol=1e-12,atol=1e-12):raise ValueError('reused prediction changed')
        if arm in ft.NEW_PROCEDURES:
            rawpart=finals[finals.procedure.eq(arm)];v=aligned[['pair_id']].merge(rawpart[['pair_id','raw_prediction']],on='pair_id',how='left',validate='one_to_one').raw_prediction
            expected=apply_affine(v,mapping) if arm=='fixed_ranker8' else v
            if not np.allclose(aligned[arm],expected,rtol=1e-12,atol=1e-12):raise ValueError('raw final/map mismatch')
    if not np.array_equal(aligned[ft.SEQUENCE_SELECTOR],aligned[winner]):raise ValueError('expanded selected prediction mismatch')
    return aligned,done


def report(out=OUT):
    frame,protocol=validate_frozen(out);parts={r:[] for r in ('hla','joint')};evidence={}
    for regime,cell,fit,held in ft.outer_cells(frame):
        aligned,done=audit_cell(frame,fit,held,regime,cell,Path(out)/regime/cell,protocol);parts[regime].append(aligned);evidence[f'{regime}/{cell}']=sha256_file(Path(out)/regime/cell/'completed.json')
    destination=Path(out)/'results';destination.mkdir(parents=True,exist_ok=True)
    comparisons=[];deletions=[]
    for regime,pieces in parts.items():
        aligned=pd.concat(pieces,ignore_index=True)
        if len(aligned)!=len(frame) or set(aligned.pair_id)!=set(frame.pair_id) or aligned.pair_id.duplicated().any():raise ValueError('pooled outer coverage changed')
        joined=frame[['pair_id','y_log1p','split']].merge(aligned,on='pair_id',validate='one_to_one').sort_values('pair_id');pred={arm:joined[arm].to_numpy() for arm in ft.ARMS}
        atomic_csv(destination/f'{regime}_pooled_predictions.csv',joined)
        for minimum in (20,50):
            comp,boots,deletion=ft.comparisons(joined,pred,min_rows=minimum,n_boot=2000);comparisons.append(comp.assign(regime=regime));deletions.append(deletion.assign(regime=regime))
            for mode,boot in boots.items():atomic_csv(destination/f'{regime}_min{minimum}_{mode}_bootstrap.csv',boot)
    atomic_csv(destination/'comparisons.csv',pd.concat(comparisons,ignore_index=True));atomic_csv(destination/'all_group_deletions.csv',pd.concat(deletions,ignore_index=True))
    atomic_json(destination/'manifest.json',dict(status='complete',schema='final_transfer_results.1',protocol_hash=protocol['protocol_hash'],cell_completed_sha256=evidence,output_sha256={p.name:sha256_file(p) for p in sorted(destination.glob('*.csv'))},fit_graph=ft.fit_graph(),interpretation='previously examined development evidence; conditional uncertainty; no original test use; fixed-ranker scale transport differs from old nested ranker'))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',type=Path,default=OUT);report(parser.parse_args().out_dir)
