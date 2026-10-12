"""Restore exact frozen training-target bits across libm implementations.

Only original-training y_log1p values are restored. Raw values, IDs, splits,
nontraining rows, source files and scientific hash checks are never changed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
META='reports/finalization/cloud/canonical_training_targets.json'
ARRAYS='reports/finalization/cloud/canonical_training_targets.npz'


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()


def read_targets(metadata_path):
    from pepstab.final_cloud_protocol import bound,sha,path_in_root
    metadata=json.loads(Path(metadata_path).read_text())
    expected=dict(schema='final_cloud_canonical_training_targets.1',n_rows=19716,split='train',
                  transform='numpy.log1p(thalf_hours)',maximum_platform_ulp_difference=1)
    if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('exact training-target transport contract required')
    for key,path in [('raw_data_sha256','data/rasmussen_et_al_dataset.csv'),('splits_sha256','data/splits.csv')]:
        if sha(ROOT/path)!=metadata[key]:raise ValueError('canonical target raw/split source changed')
    path=bound(metadata['arrays'])
    if path!=path_in_root(ARRAYS):raise ValueError('unexpected target transport path')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'pair_id','thalf_hours','y_log1p'}:raise ValueError('target arrays changed')
        arrays={name:archive[name].copy() for name in archive.files}
    for name,dtype in [('pair_id','<i4'),('thalf_hours','<f8'),('y_log1p','<f8')]:
        value=arrays[name]
        if value.dtype!=np.dtype(dtype) or value.shape!=(19716,) or not np.isfinite(value).all():
            raise ValueError('target array dtype/shape/finiteness changed')
        if hashlib.sha256(value.tobytes()).hexdigest()!=metadata['array_sha256'][name]:
            raise ValueError('target array content changed')
    if np.any(np.diff(arrays['pair_id'])<=0) or np.any(arrays['thalf_hours']<0) or np.any(arrays['y_log1p']<0):
        raise ValueError('ordered unique nonnegative training targets required')
    return metadata,arrays


def restore_training_targets(frame,metadata,arrays):
    train=frame.loc[frame['split'].eq('train')].sort_values('pair_id')
    if (train.pair_id.dtype!=np.dtype('int32') or train.thalf_hours.dtype!=np.dtype('float64')
            or train.y_log1p.dtype!=np.dtype('float64')
            or not np.array_equal(train.pair_id.to_numpy(),arrays['pair_id'])
            or not np.array_equal(train.thalf_hours.to_numpy().view(np.uint64),arrays['thalf_hours'].view(np.uint64))):
        raise ValueError('canonical training IDs/raw target bits changed')
    observed=train.y_log1p.to_numpy();canonical=arrays['y_log1p']
    if not np.isfinite(observed).all() or np.any(observed<0):raise ValueError('invalid current transformed targets')
    a,b=observed.view(np.uint64),canonical.view(np.uint64)
    distance=np.maximum(a,b)-np.minimum(a,b)
    if np.any(distance>1):raise ValueError('target discrepancy exceeds diagnosed one-ULP platform difference')
    result=frame.copy()
    result.loc[train.index,'y_log1p']=canonical
    restored=result.loc[train.index].reset_index(drop=True)
    if frame_hash(restored)!=metadata['training_frame_sha256']:
        raise ValueError('restored canonical training frame does not match frozen hash')
    return result


def install(metadata_path):
    """Install only at the cloud entry, before frozen consumers import the loader."""
    from pepstab import data
    metadata,arrays=read_targets(metadata_path)
    original=data.load_with_splits
    def canonical_load_with_splits():
        return restore_training_targets(original(),metadata,arrays)
    data.load_with_splits=canonical_load_with_splits
