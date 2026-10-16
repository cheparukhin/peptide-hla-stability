"""Synthetic hybrid recipe/artifact checks; no biological data are fitted."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pepstab import final_hybrids as core
from pepstab.mlp import MLPRegressor


def synthetic_roles():
    rng=np.random.default_rng(417)
    counts={'gradient':160,'stopping':32,'held':24}
    features={}
    frames={}
    offset=0
    for role,n in counts.items():
        X=rng.normal(size=(n,860)).astype(np.float32)
        X[:,0]=rng.integers(0,2,n)
        X[:,1]=0.
        features[role]=X
        frame=pd.DataFrame(dict(pair_id=np.arange(offset,offset+n),
            cluster_id=np.arange(offset,offset+n)//2,
            allele=np.tile(['A','B','C','D'],n//4),split='train'))
        if role!='held':
            frame['y_log1p']=np.maximum(0.,1.+.4*X[:,2]+.3*X[:,3]).astype(np.float64)
            frame.loc[frame.index%5==0,'y_log1p']=0.
        frames[role]=frame
        offset+=n
    return features,frames


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    features,frames=synthetic_roles()
    directory=tmp_path_factory.mktemp('complete_hybrid_bundle')
    # Exercise the actual runner's forwarding interface, then the core replay
    # used by the root reporter, with one real synthetic six-fit bundle.
    from scripts.final_hybrids import fit_bundle
    with threadpool_limits(limits=2):
        result=fit_bundle(features,frames['gradient'],frames['stopping'],frames['held'],
            representation='R0',seed=0,context='synthetic/inner0/stop0seed0',
            artifact_dir=directory,protocol_hash='synthetic',source_feature_hash='synthetic_full_matrix')
    return features,frames,directory,result


def audit(features,frames,directory,identity):
    with threadpool_limits(limits=2):
        return core.audit_member_bundle(features,frames['gradient'],frames['stopping'],frames['held'],
            directory,'synthetic',identity)


def test_identity_json_roundtrip_and_fixed_recipe():
    features,frames=synthetic_roles()
    identity=core.member_identity(features,frames['gradient'],frames['stopping'],frames['held'],
        'R0',0,'test','synthetic')
    assert identity==json.loads(json.dumps(identity))
    assert identity['mlp']['hidden']==[64]
    assert identity['mlp']['l2']==.01
    assert identity['tree']['iterations']==512 and identity['tree']['depth']==4
    assert not identity['tree']['use_best_model']
    assert identity['preprocessing']['quantile']['subsample'] is None
    assert identity['preprocessing']['quantile']['n_quantiles']==160
    assert identity['preprocessing']['pca']['n_components']==128
    assert not identity['preprocessing']['pca']['whiten']
    assert core.fit_graph()['scientific_fits']==480
    assert core.fit_graph()['mlp_fits']==320
    assert core.fit_graph()['tree_fits']==160
    assert set(core.COMPONENTS)=={'standard','tree_rmse','tree_rank','leaf_mlp','quantile_mlp','pca_mlp'}


def test_complete_bundle_replays_weights_trees_transforms_without_fit(bundle,monkeypatch):
    features,frames,directory,result=bundle
    def forbidden(*args,**kwargs):
        raise AssertionError('replay attempted a learner/transform refit')
    monkeypatch.setattr(core.MLPRegressor,'fit',forbidden)
    monkeypatch.setattr(core.StandardScaler,'fit',forbidden)
    monkeypatch.setattr(core.QuantileTransformer,'fit',forbidden)
    monkeypatch.setattr(core.PCA,'fit',forbidden)
    actual=audit(features,frames,directory,result['identity'])
    assert actual['n_fits']==6 and actual['audit']['status']=='passed'
    for name in core.COMPONENTS:
        np.testing.assert_array_equal(result['predictions'][name],actual['predictions'][name])
    receipt=json.loads((directory/'completed.json').read_text())
    assert set(receipt['component_details'])==set(core.COMPONENTS)
    assert receipt['total_fit_seconds']>0 and isinstance(receipt['peak_rss_bytes'],int)
    assert receipt['component_details']['tree_rmse']['tree_count']==512
    assert receipt['component_details']['tree_rank']['tree_count']==512
    assert receipt['component_details']['leaf_mlp']['n_features']==1024
    assert receipt['component_details']['pca_mlp']['n_features']==128
    assert len(list(directory.glob('fit_*.launched.json')))==6
    assert len(list(directory.glob('fit_*.completed.json')))==6
    assert receipt['component_details']['leaf_mlp']['tree_rmse_sha256']==receipt['payload_sha256']['tree_rmse.cbm']


def test_standard_neural_head_matches_canonical_mlp_exactly(bundle):
    features,frames,directory,result=bundle
    with threadpool_limits(limits=2):
        scaler=core.StandardScaler().fit(features['gradient'])
        expected=MLPRegressor(core.mlp_config(0)).fit(scaler.transform(features['gradient']),
            frames['gradient'].y_log1p.to_numpy(),scaler.transform(features['stopping']),
            frames['stopping'].y_log1p.to_numpy()).predict(scaler.transform(features['held']))
    np.testing.assert_array_equal(result['predictions']['standard'],expected)


def test_report_runtime_contract_reads_real_synthetic_bundle(bundle):
    from scripts.final_hybrids_report import bundle_runtime
    _,_,directory,_=bundle
    row=bundle_runtime(directory,'R0',0,'inner',0,0,0)
    assert row['total_fit_seconds']>0 and row['bundle_seconds']>row['total_fit_seconds']
    assert row['transform_seconds']>=0 and row['peak_rss_bytes']>0
    assert set(row['component_details'])==set(core.COMPONENTS)
    assert all('dimensions' in d and d['fit_seconds']>=0 for d in row['component_details'].values())


def test_exact_completed_bundle_reuse_makes_no_new_fits(bundle,monkeypatch):
    features,frames,directory,result=bundle
    monkeypatch.setattr(core.MLPRegressor,'fit',lambda *a,**k:pytest.fail('duplicate fitting'))
    with threadpool_limits(limits=2):
        other=core.fit_member_bundle(features,frames['gradient'],frames['stopping'],frames['held'],
            representation='R0',seed=0,context='synthetic/inner0/stop0seed0',artifact_dir=directory,
            protocol_hash='synthetic',source_feature_hash='synthetic_full_matrix')
    assert other['reused_completed_bundle']


@pytest.mark.parametrize('alteration',['held_labels','cluster_overlap','not_train','PCA_short','wrong_width','wrong_dtype'])
def test_scope_and_feature_roles_fail_before_fits(tmp_path,alteration):
    features,frames=synthetic_roles()
    if alteration=='held_labels':frames['held']['y_log1p']=999.
    if alteration=='cluster_overlap':frames['held'].loc[0,'cluster_id']=frames['gradient'].cluster_id.iloc[0]
    if alteration=='not_train':frames['stopping'].loc[0,'split']='test'
    if alteration=='PCA_short':
        frames['gradient']=frames['gradient'].iloc[:100]
        features['gradient']=features['gradient'][:100]
    if alteration=='wrong_width':features={k:v[:,:128] for k,v in features.items()}
    if alteration=='wrong_dtype':features['held']=features['held'].astype(np.float64)
    with pytest.raises(ValueError):
        core.fit_member_bundle(features,frames['gradient'],frames['stopping'],frames['held'],
            representation='R0',seed=0,context='test',artifact_dir=tmp_path,protocol_hash='synthetic')
    assert not (tmp_path/'launched.json').exists()


def test_no_rank_pairs_has_no_fallback_or_partial_fit(tmp_path):
    features,frames=synthetic_roles()
    frames['gradient']['y_log1p']=0.
    with pytest.raises(ValueError,match='no strictly ordered'):
        core.fit_member_bundle(features,frames['gradient'],frames['stopping'],frames['held'],
            representation='R0',seed=0,context='test',artifact_dir=tmp_path,protocol_hash='synthetic')
    assert not (tmp_path/'launched.json').exists()


def test_tree_target_and_feature_partition_does_not_accept_stop_data(bundle):
    _,_,directory,_=bundle
    complete=json.loads((directory/'completed.json').read_text())
    records=complete['ancestry']['records']
    tree_records=[r for r in records if r['role']=='hybrid_tree']
    assert len(tree_records)==2
    assert all('stopping_rows' not in record for record in tree_records)
    for name in ('tree_rmse','tree_rank'):
        d=complete['component_details'][name]
        assert d['n_stop']==0 and d['n_stop_reserved']==32 and d['no_stopping_labels'] and d['no_eval_set']


def test_rank_pair_ties_and_allele_weighting():
    y=np.array([2.,2.,1.,3.,0.])
    allele=np.array(['a','a','a','b','b'])
    order,groups,target,hi,lo,weight,details=core._rank_data(y,allele)
    assert len(hi)==3 and (groups[hi]==groups[lo]).all() and (target[hi]>target[lo]).all()
    assert np.isclose(weight[groups[hi]=='a'].sum(),weight[groups[hi]=='b'].sum())
    assert details['n_pairs']==3 and details['target_ties_omitted']


def test_fixed_leaf_dictionary_handles_unobserved_and_shallow_trees():
    class FakeTree:
        tree_count_=512
        def calc_leaf_indexes(self,X,**kwargs):
            assert kwargs==dict(ntree_start=0,ntree_end=64,thread_count=2)
            # Every tree is shallower than depth4. Keep all16 slots anyway.
            return np.tile(np.arange(64)%4,(len(X),1))
    result=core.encode_leaves(FakeTree(),np.zeros((3,860),dtype=np.float32))
    assert result.shape==(3,1024)
    np.testing.assert_array_equal(result.sum(axis=1),[64,64,64])
    assert len(core.leaf_dictionary())==1024
    class InvalidTree(FakeTree):
        def calc_leaf_indexes(self,X,**kwargs):return np.full((len(X),64),16)
    with pytest.raises(ValueError,match='outside'):
        core.encode_leaves(InvalidTree(),np.zeros((3,860),dtype=np.float32))


@pytest.mark.parametrize('target',['arrays.npz','tree_rmse.cbm','fit_standard.completed.json'])
def test_corrupted_payload_rejected_without_refit(bundle,tmp_path,target):
    features,frames,source,result=bundle
    directory=tmp_path/'copy'
    shutil.copytree(source,directory)
    (directory/target).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='payload changed'):
        audit(features,frames,directory,result['identity'])


def test_changed_labels_or_order_block_completed_reuse(bundle):
    features,frames,directory,result=bundle
    changed={name:frame.copy() for name,frame in frames.items()}
    changed['stopping'].loc[0,'y_log1p']+=1.
    with pytest.raises(ValueError,match='identity'):
        audit(features,changed,directory,result['identity'])


def test_interrupted_bundle_cannot_blind_retry(tmp_path,monkeypatch):
    features,frames=synthetic_roles()
    def fail(*a,**k):raise RuntimeError('synthetic interrupted fit')
    monkeypatch.setattr(core.MLPRegressor,'fit',fail)
    kwargs=dict(representation='R0',seed=0,context='test',artifact_dir=tmp_path,protocol_hash='synthetic')
    with threadpool_limits(limits=2),pytest.raises(RuntimeError,match='interrupted'):
        core.fit_member_bundle(features,frames['gradient'],frames['stopping'],frames['held'],**kwargs)
    assert (tmp_path/'failed.json').exists()
    with pytest.raises(FileExistsError):
        core.fit_member_bundle(features,frames['gradient'],frames['stopping'],frames['held'],**kwargs)
