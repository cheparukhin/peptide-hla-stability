"""Complete synthetic 30-cell report; no learned model is fitted."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from pepstab.boltz_embeddings import sha256_file
from pepstab.mlp import MLPConfig
from pepstab.round2_transfer import (ARMS, TransferAncestry, assess_cell, inner_partitions,
                                    outer_cells, transfer_comparisons, member_schedule, candidate_registry, reported_arms, SEQUENCE_SELECTOR, EMBEDDING_SELECTOR)
from scripts.round2_transfer import atomic_csv, atomic_json, verify_cell
from scripts.round2_transfer_report import load_predictions, report, validate_evidence


def synthetic_frame():
    rows = []
    for group in range(15):
        for peptide in range(60):
            rows.append(dict(pair_id=len(rows),cluster_id=peptide,hla_group=group,allele=f"A{group}",
                split="train",hla_fold=group%5,peptide_fold=peptide%5,y_log1p=float(peptide%7),thalf_hours=float(peptide%7)))
    return pd.DataFrame(rows)


def save_synthetic_cell(out, frame, regime, cell, fit, held, candidates):
    ancestry = TransferAncestry(regime)
    members = []
    def predictor(train,predict,candidate,context):
        common = dict(context=context,candidate=candidate["id"],fit_seconds=0.,n_source=len(train))
        final = context.startswith('final/')
        for spec in member_schedule(train,regime,candidate,final=final):
            gradient,stop = spec['gradient'],spec['stopping']
            feature = 'representation' in candidate
            key = dict(seed=spec['seed'])
            if stop is not None:
                key.update(encoding=spec['encoding'],stopping_cell=spec['stopping_cell'])
            r = ancestry.add('ridge' if stop is None else 'mlp_member',gradient,predict,
                stopping=stop,scaler=gradient if feature else None,context=context,candidate=candidate['id'],**key)
            means,global_mean = {},None
            if candidate.get('target') == 'centred':
                from pepstab.phase_a import shrunk_allele_means
                means,global_mean = shrunk_allele_means(gradient.allele,gradient.y_log1p,candidate['k'])
            m = dict(**common,**key,n_fit=len(gradient),n_stop=0 if stop is None else len(stop),
                excluded_cross_quadrant_rows=len(train)-len(gradient)-(0 if stop is None else len(stop)),
                final_recipe=final,centering_means=means,centering_global=global_mean)
            m.update({k:r[k] for k in ('fit_rows','stopping_rows','prediction_rows','scaler_rows') if k in r})
            if stop is not None: m.update(config=asdict(spec['config']),best_epoch=0,epochs=1)
            if feature: m.update(scaler_mean=[0.],scaler_scale=[1.],feature_hash='synthetic')
            members.append(m)
        return predict.cluster_id.to_numpy(dtype=float)%7
    values,selection = assess_cell(fit,held,regime,predictor,ancestry,candidates=candidates)
    directory = out/regime/cell
    directory.mkdir(parents=True)
    tables = []
    for arm in reported_arms(candidates):
        table = held[["pair_id","cluster_id","hla_group","allele","hla_fold","peptide_fold"]].copy()
        tables.append(table.assign(arm=arm,regime=regime,cell=cell,y_pred=values[arm]))
    atomic_csv(directory/"predictions.csv",pd.concat(tables,ignore_index=True))
    atomic_json(directory/"selection.json",selection)
    atomic_json(directory/"ancestry.json",ancestry.as_dict())
    atomic_json(directory/"members.json",members)
    atomic_json(directory/"runtime.json",{})
    done = dict(status="complete",regime=regime,cell=cell,protocol_hash="synthetic",n_held=len(held),n_fits=len(members),
        sha256={name:sha256_file(directory/name) for name in ("predictions.csv","selection.json","ancestry.json","members.json","runtime.json")})
    atomic_json(directory/"completed.json",done)
    return directory,done,ancestry


def test_all30_cells_report_and_coherently_rehashed_tampering(tmp_path,monkeypatch):
    import scripts.round2_transfer_report as reporter
    frame = synthetic_frame()
    candidates = candidate_registry(("R3_mlp","R5_ridge"))
    protocol = {"protocol_hash":"synthetic","specification":{"candidate_order":list(candidates)}}
    saved = []
    for regime,cell,fit,held in outer_cells(frame):
        directory,done,ancestry = save_synthetic_cell(tmp_path,frame,regime,cell,fit,held,candidates)
        saved.append((regime,cell,fit,held,directory,done,ancestry))
    monkeypatch.setattr(reporter,"validate_frozen",lambda *args:(frame,protocol))
    monkeypatch.setattr(reporter,"transfer_comparisons",lambda frame,values,min_rows,n_boot:
                        transfer_comparisons(frame,values,min_rows=min_rows,n_boot=2))
    pd.DataFrame({"regime":["hla"]*5+["joint"]*25,"coverage":"synthetic"}).to_csv(tmp_path/"coverage.csv",index=False)
    report(tmp_path,tmp_path,tmp_path)
    manifest = json.loads((tmp_path/"results/completed.json").read_text())
    assert len(manifest["input_completion_hashes"]) == 30
    assert sum(item[5]["n_fits"] for item in saved) == 13530
    comparisons = pd.read_csv(tmp_path/"results/comparisons.csv")
    assert len(comparisons) == 2*2*3*9
    assert set(comparisons.baseline) == {"original",SEQUENCE_SELECTOR}
    assert set(comparisons.panel_min_rows) == {20,50}
    assert np.allclose(comparisons.delta,0)
    per_cell = pd.read_csv(tmp_path/"results/per_cell.csv")
    assert len(per_cell) == 30*6*2
    # Empty joint cell diagnostic panels are retained as NaN, not discarded.
    assert per_cell[per_cell.regime.eq("joint")].median_spearman.isna().all()
    regime,cell,fit,held,directory,done,ancestry = saved[0]
    def rehash(name):
        done["sha256"][name] = sha256_file(directory/name)
        atomic_json(directory/"completed.json",done)
        verify_cell(directory,protocol,regime,cell)
    members = json.loads((directory/"members.json").read_text())
    original = json.loads(json.dumps(members))
    members[0]["n_fit"] += 1
    atomic_json(directory/"members.json",members)
    rehash("members.json")
    with pytest.raises(ValueError,match="counts"):
        validate_evidence(frame,fit,held,regime,directory,done,ancestry,candidates)
    atomic_json(directory/"members.json",original[:-1])
    rehash("members.json")
    with pytest.raises(ValueError,match="fit graph"):
        validate_evidence(frame,fit,held,regime,directory,done,ancestry,candidates)
    atomic_json(directory/"members.json",original)
    rehash("members.json")
    selection = json.loads((directory/"selection.json").read_text())
    selection["selected"][EMBEDDING_SELECTOR] = selection["selected"]["R3_mlp"]
    atomic_json(directory/"selection.json",selection)
    rehash("selection.json")
    with pytest.raises(ValueError,match="selection violates"):
        validate_evidence(frame,fit,held,regime,directory,done,ancestry,candidates)


def test_partial_29_cell_results_fail_closed(tmp_path):
    frame = synthetic_frame()
    candidates = candidate_registry(('R3_mlp',))
    protocol = dict(protocol_hash='synthetic',specification=dict(candidate_order=list(candidates)))
    for regime,cell,fit,held in list(outer_cells(frame))[:-1]:
        save_synthetic_cell(tmp_path,frame,regime,cell,fit,held,candidates)
    with pytest.raises(FileNotFoundError):
        load_predictions(frame,tmp_path,protocol)


def test_replay_rejects_centering_and_feature_hash_tampering(tmp_path):
    frame = synthetic_frame()
    candidates = candidate_registry(('R3_mlp',))
    regime,cell,fit,held = next(outer_cells(frame))
    directory,done,ancestry = save_synthetic_cell(tmp_path,frame,regime,cell,fit,held,candidates)
    members = json.loads((directory/'members.json').read_text())
    original = json.loads(json.dumps(members))
    arguments = (frame,fit,held,regime,directory,done,ancestry,candidates,{'R3':'synthetic'})
    validate_evidence(*arguments)
    centred = next(m for m in members if m['candidate'].startswith('mlp_centred'))
    centred['centering_global'] += 1
    atomic_json(directory/'members.json',members)
    with pytest.raises(ValueError,match='centring'):
        validate_evidence(*arguments)
    members = original
    feature = next(m for m in members if m['candidate'].startswith('R3'))
    feature['feature_hash'] = 'changed'
    atomic_json(directory/'members.json',members)
    with pytest.raises(ValueError,match='feature hash'):
        validate_evidence(*arguments)
