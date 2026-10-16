"""Synthetic worker/report integration; no real model or scientific-data access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from test_final_hybrids_pipeline import mocked_bundle_integration, worker
from scripts import final_hybrids_cloud_report as report


def test_cloud_replay_matches_float32_members_averaged_by_real_worker(tmp_path,monkeypatch):
    frame,features,protocol,approved=mocked_bundle_integration(tmp_path,monkeypatch)
    original=worker.fit_bundle
    def float32_bundle(*args,**kwargs):
        result=original(*args,**kwargs)
        result['predictions']={key:value.astype(np.float32) for key,value in result['predictions'].items()}
        return result
    monkeypatch.setattr(worker,'fit_bundle',float32_bundle)
    worker.component_job(tmp_path,approved,'R0',0)
    table,_,audit=report.audit_component(tmp_path,protocol,frame,features,'R0',0)
    assert audit['new_supervised_fits']==48 and len(table)==720
    path=tmp_path/'fold_0/R0/raw_final_predictions.csv'
    raw=pd.read_csv(path,float_precision='round_trip')
    assert np.any(raw.y_pred.to_numpy()!=raw.y_pred.to_numpy(dtype=np.float32).astype(float))
    raw.loc[0,'y_pred']+=1e-5
    raw.to_csv(path,index=False)
    completed=path.parent/'completed.json';done=json.loads(completed.read_text())
    done['sha256'][path.name]=worker.sha256_file(path);completed.write_text(json.dumps(done))
    with pytest.raises(AssertionError):
        report.audit_component(tmp_path,protocol,frame,features,'R0',0)
