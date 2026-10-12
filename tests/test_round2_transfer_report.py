"""Complete synthetic 30-cell report; no learned model is fitted."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from pepstab.boltz_embeddings import sha256_file
from pepstab.mlp import MLPConfig
from pepstab.round2_transfer import (ARMS, TransferAncestry, assess_cell, inner_partitions,
                                    outer_cells, transfer_comparisons)
from scripts.round2_transfer import atomic_csv, atomic_json, verify_cell
from scripts.round2_transfer_report import load_predictions, report, validate_evidence


def synthetic_frame():
    rows = []
    for group in range(15):
        for peptide in range(60):
            rows.append(dict(pair_id=len(rows),cluster_id=peptide,hla_group=group,allele=f"A{group}",
                split="train",hla_fold=group%5,peptide_fold=peptide%5,y_log1p=float(peptide%7),thalf_hours=float(peptide%7)))
    return pd.DataFrame(rows)


def save_synthetic_cell(out, frame, regime, cell, fit, held):
    ancestry = TransferAncestry(regime)
    members = []
    def predictor(train,predict,candidate,context):
        common = dict(context=context,candidate=candidate["id"],fit_seconds=0.,n_source=len(train))
        if candidate["learner"] == "ridge":
            ancestry.records.append(dict(role="ridge",context=context,candidate=candidate["id"],
                fit_rows=ancestry.rows(train),scaler_rows=ancestry.rows(train),prediction_rows=ancestry.rows(predict)))
            members.append(dict(**common,n_fit=len(train),n_stop=0,excluded_cross_quadrant_rows=0,seed=None))
        else:
            for stop_cell,gradient,stop in inner_partitions(train,regime,5,stopping=True):
                for enc in ("onehot","blosum"):
                    for seed in range(3):
                        key = dict(stopping_cell=stop_cell,encoding=enc,seed=seed)
                        ancestry.records.append(dict(role="original_member",context=context,**key,
                            fit_rows=ancestry.rows(gradient),stopping_rows=ancestry.rows(stop),prediction_rows=ancestry.rows(predict)))
                        cfg = MLPConfig(hidden=(256,64),l2=1e-5 if enc == "onehot" else 1e-3,max_epochs=300,patience=15,seed=seed)
                        members.append(dict(**common,**key,n_fit=len(gradient),n_stop=len(stop),
                            excluded_cross_quadrant_rows=len(train)-len(gradient)-len(stop),config=asdict(cfg),best_epoch=0,epochs=1))
        return predict.cluster_id.to_numpy(dtype=float)%7
    values,selection = assess_cell(fit,held,regime,predictor,ancestry)
    directory = out/regime/cell
    directory.mkdir(parents=True)
    tables = []
    for arm in ARMS:
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
    protocol = {"protocol_hash":"synthetic"}
    saved = []
    for regime,cell,fit,held in outer_cells(frame):
        directory,done,ancestry = save_synthetic_cell(tmp_path,frame,regime,cell,fit,held)
        saved.append((regime,cell,fit,held,directory,done,ancestry))
    monkeypatch.setattr(reporter,"validate_frozen",lambda *args:(frame,protocol))
    monkeypatch.setattr(reporter,"transfer_comparisons",lambda frame,values,min_rows,n_boot:
                        transfer_comparisons(frame,values,min_rows=min_rows,n_boot=2))
    pd.DataFrame({"regime":["hla"]*5+["joint"]*25,"coverage":"synthetic"}).to_csv(tmp_path/"coverage.csv",index=False)
    report(tmp_path,tmp_path,tmp_path)
    manifest = json.loads((tmp_path/"results/completed.json").read_text())
    assert len(manifest["input_completion_hashes"]) == 30
    comparisons = pd.read_csv(tmp_path/"results/comparisons.csv")
    assert len(comparisons) == 2*2*3*3
    assert set(comparisons.panel_min_rows) == {20,50}
    assert np.allclose(comparisons.delta,0)
    per_cell = pd.read_csv(tmp_path/"results/per_cell.csv")
    assert len(per_cell) == 30*4*2
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
        validate_evidence(frame,fit,held,regime,directory,done,ancestry)
    atomic_json(directory/"members.json",original[:-1])
    rehash("members.json")
    with pytest.raises(ValueError,match="fit graph"):
        validate_evidence(frame,fit,held,regime,directory,done,ancestry)
    atomic_json(directory/"members.json",original)
    rehash("members.json")
    selection = json.loads((directory/"selection.json").read_text())
    selection["selected"]["transfer_selector"] = selection["selected"]["R2_ridge"]
    atomic_json(directory/"selection.json",selection)
    rehash("selection.json")
    with pytest.raises(ValueError,match="selection violates"):
        validate_evidence(frame,fit,held,regime,directory,done,ancestry)
