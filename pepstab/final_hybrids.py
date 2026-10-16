"""Fixed six-component hybrid members, with pickle-free artifact replay.

One bundle is four MSE neural heads and two gradient-only fixed-budget trees.
The RMSE tree is reused for the fixed leaf representation. No selection,
calibration, cloud work or assessment-label access occurs in this module.
"""
from __future__ import annotations

from dataclasses import asdict
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale, _is_constant_feature
from sklearn.utils.extmath import _incremental_mean_and_var
from threadpoolctl import threadpool_limits

from .mlp import MLPConfig, MLPRegressor
from .phase_a import ranking_pairs
from .round2_nested import Ancestry, validate_partition

COMPONENTS = ('standard', 'tree_rmse', 'tree_rank', 'leaf_mlp', 'quantile_mlp', 'pca_mlp')
MLP_COMPONENTS = ('standard', 'leaf_mlp', 'quantile_mlp', 'pca_mlp')
REPRESENTATIONS = {'R0': 860, 'R5': 1024}
SCHEMA = 'pepstab-final-hybrid-bundle-v1'
TREE_PARAMETERS = dict(iterations=512, depth=4, learning_rate=.06, l2_leaf_reg=10.,
    bootstrap_type='Bernoulli', subsample=.8, rsm=1., random_strength=1.,
    border_count=128, thread_count=2, task_type='CPU', grow_policy='SymmetricTree',
    use_best_model=False, verbose=False, allow_writing_files=False)


def array_hash(array):
    if isinstance(array, bytes):
        return hashlib.sha256(array).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_parameter_tree(actual, expected):
    """CBM textual metadata rounds floats; reject every other parameter change."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):
            raise ValueError('effective CatBoost parameter keys changed')
        for key in expected:
            _check_parameter_tree(actual[key],expected[key])
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):
            raise ValueError('effective CatBoost parameter list changed')
        for a,b in zip(actual,expected):
            _check_parameter_tree(a,b)
    elif isinstance(expected,float):
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-9)
    elif actual!=expected:
        raise ValueError('effective CatBoost parameter changed')


def atomic_json(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(item, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def mlp_config(seed):
    return MLPConfig(hidden=(64,), l2=.01, lr=.001, batch_size=256,
                     max_epochs=200, patience=15, seed=seed)


def preprocessing_parameters(n_gradient):
    return dict(standard=StandardScaler().get_params(),
        quantile=QuantileTransformer(n_quantiles=min(256, n_gradient),
            output_distribution='normal', subsample=None, random_state=0).get_params(),
        pca=PCA(n_components=128, svd_solver='randomized', whiten=False, random_state=0).get_params(),
        leaf=dict(n_trees=64, slots_per_tree=16, n_columns=1024,
                  ordering='tree_index then leaf_id; fixed full dictionary'))


def _validate_inputs(features, gradient, stopping, held):
    if set(features) != {'gradient', 'stopping', 'held'}:
        raise ValueError('exact gradient/stopping/held feature roles required')
    if any(name in held for name in ('y_log1p', 'thalf_hours')):
        raise ValueError('held assessment labels must not reach member fitting or replay')
    for role, frame in (('gradient', gradient), ('stopping', stopping), ('held', held)):
        if not {'pair_id','cluster_id','allele','split'} <= set(frame) or len(frame) == 0:
            raise ValueError(f'nonempty original-train metadata required: {role}')
        if not frame.split.eq('train').all() or frame[['pair_id','cluster_id','allele']].isna().any().any():
            raise ValueError('only complete original-training row metadata accepted')
        X = np.asarray(features[role])
        if X.dtype != np.float32 or X.ndim != 2 or len(X) != len(frame) or not np.isfinite(X).all():
            raise ValueError(f'finite aligned float32 role features required: {role}')
        if role != 'held':
            if 'y_log1p' not in frame or not np.isfinite(frame.y_log1p.to_numpy()).all():
                raise ValueError('finite fitting/stopping targets required')
    validate_partition(gradient, stopping)
    validate_partition(gradient, held)
    validate_partition(stopping, held)
    widths = {np.asarray(X).shape[1] for X in features.values()}
    if len(widths) != 1 or min(len(gradient), next(iter(widths))) < 128:
        raise ValueError('128 PCA components unavailable; no dimensional fallback')


def member_identity(features, gradient, stopping, held_unlabelled, representation,
                    seed, context, protocol_hash, source_feature_hash=None):
    _validate_inputs(features, gradient, stopping, held_unlabelled)
    if representation not in REPRESENTATIONS or seed not in (0,1):
        raise ValueError('member outside frozen representation/seed scope')
    if any(np.asarray(X).shape[1] != REPRESENTATIONS[representation] for X in features.values()):
        raise ValueError('frozen representation feature width changed')
    roles = {}
    for role, frame in (('gradient',gradient),('stopping',stopping),('held',held_unlabelled)):
        roles[role] = dict(pair_ids=[int(i) for i in frame.pair_id],
            cluster_ids=frame.cluster_id.astype(str).tolist(), allele=frame.allele.astype(str).tolist(),
            features_sha256=array_hash(features[role]), shape=list(np.asarray(features[role]).shape),
            dtype='float32')
        if role != 'held':
            roles[role]['labels_sha256'] = array_hash(frame.y_log1p.to_numpy())
    identity=dict(schema=SCHEMA, protocol_hash=protocol_hash, representation=representation,
        seed=int(seed), context=context, source_feature_hash=source_feature_hash, roles=roles,
        mlp=asdict(mlp_config(seed)), tree={**TREE_PARAMETERS, 'random_seed':int(seed)},
        preprocessing=preprocessing_parameters(len(gradient)),
        versions={name:version(name) for name in ('numpy','scipy','scikit-learn','catboost','threadpoolctl')},
        python=platform.python_version(),
        source_sha256={name:file_hash(Path(__file__).with_name(name)) for name in
                       ('final_hybrids.py','mlp.py','phase_a.py','round2_nested.py')})
    return json.loads(json.dumps(identity,allow_nan=False))


def leaf_dictionary():
    index=np.arange(1024,dtype=np.int16)
    return np.column_stack((index//16,index%16))


def encode_leaves(model, X):
    if model.tree_count_ < 64:
        raise ValueError('leaf representation requires first64 fitted trees')
    leaves=np.asarray(model.calc_leaf_indexes(X,ntree_start=0,ntree_end=64,thread_count=2))
    if leaves.shape != (len(X),64) or np.any(leaves<0) or np.any(leaves>=16):
        raise ValueError('leaf IDs outside fixed64×16 dictionary')
    result=np.zeros((len(X),1024),dtype=np.float32)
    columns=np.arange(64)[None,:]*16+leaves
    result[np.arange(len(X))[:,None],columns]=1.
    return result


def _save_scaler(arrays, prefix, scaler):
    for name in ('mean','var','scale'):
        arrays[f'{prefix}__{name}']=getattr(scaler,name+'_').copy()


def _restore_scaler(arrays, prefix, n_gradient):
    scaler=StandardScaler()
    scaler.mean_=arrays[f'{prefix}__mean'].copy()
    scaler.var_=arrays[f'{prefix}__var'].copy()
    scaler.scale_=arrays[f'{prefix}__scale'].copy()
    scaler.n_features_in_=len(scaler.mean_)
    scaler.n_samples_seen_=np.int64(n_gradient)
    return scaler


def _check_scaler(arrays, prefix, X):
    # Pure sufficient-statistic reconstruction: no estimator/model refit.
    mean,var,count=_incremental_mean_and_var(X,0.,0.,0.)
    scale=_handle_zeros_in_scale(np.sqrt(var),copy=False,
                                constant_mask=_is_constant_feature(var,mean,count))
    for name,value in (('mean',mean),('var',var),('scale',scale)):
        np.testing.assert_array_equal(arrays[f'{prefix}__{name}'],value,
                                      err_msg=f'gradient-only scaler state changed: {prefix}')
    return _restore_scaler(arrays,prefix,len(X))


def _restore_quantile(arrays, params):
    transformer=QuantileTransformer(**params)
    transformer.quantiles_=arrays['quantile__quantiles'].copy()
    transformer.references_=arrays['quantile__references'].copy()
    transformer.n_quantiles_=len(transformer.references_)
    transformer.n_features_in_=transformer.quantiles_.shape[1]
    return transformer


def _restore_pca(arrays, params, n_gradient):
    transformer=PCA(**params)
    for name in ('components','mean','explained_variance','explained_variance_ratio','singular_values','noise_variance'):
        setattr(transformer,name+'_',arrays[f'pca__{name}'].copy())
    transformer.n_features_in_=len(transformer.mean_)
    transformer.n_components_=128
    transformer.n_samples_=n_gradient
    return transformer


def _restore_mlp(arrays, name, seed):
    model=MLPRegressor(mlp_config(seed))
    model.weights_=[arrays[f'{name}__w{i}'].copy() for i in (0,1)]
    model.biases_=[arrays[f'{name}__b{i}'].copy() for i in (0,1)]
    model.y_offset_=float(arrays[f'{name}__offset'])
    return model


def _rank_data(y, allele):
    allele=np.asarray(allele,dtype=str)
    order=np.argsort(allele,kind='stable')
    groups,target=allele[order],np.asarray(y)[order]
    hi,lo,weight=ranking_pairs(groups,target)
    if len(hi)==0:
        raise ValueError('no strictly ordered within-allele pairs; no ranker fallback')
    weight=weight/weight.mean()
    details=dict(n_pairs=len(hi),ordered_gradient_indices_sha256=array_hash(order.astype('<i8')),
        higher_indices_sha256=array_hash(hi),lower_indices_sha256=array_hash(lo),
        normalized_weights_sha256=array_hash(weight),target_ties_omitted=True,
        equal_total_weight_per_contributing_allele=True)
    return order,groups,target,hi,lo,weight,details


def _rank_pool(X, y, allele):
    from catboost import Pool
    order,groups,target,hi,lo,weight,details=_rank_data(y,allele)
    return Pool(X[order],target,group_id=groups,pairs=np.column_stack((hi,lo)),pairs_weight=weight),details


def _rss_bytes():
    value=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform=='darwin' else value*1024


def fit_graph():
    return dict(scientific_fits=480,engineering_pilot_fits=12,supervised_fits_per_bundle=6,
        bundles_per_representation_outer=8,total_scientific_bundles=80,
        mlp_fits=320,tree_fits=160,rmse_tree_reused_for_leaf=True)


def member_schedule(fit):
    from .nested import inner_folds
    folds=inner_folds(fit,5)
    if set(folds)!=set(range(5)):
        raise ValueError('five nonempty member stopping folds required')
    return [dict(stop_fold=stop,seed=seed,gradient=fit.loc[folds!=stop],stopping=fit.loc[folds==stop])
            for stop,seed in ((0,0),(1,1))]


@threadpool_limits.wrap(limits=2)
def fit_member_bundle(features, gradient, stopping, held_unlabelled, *, representation,
                      seed, context, artifact_dir, protocol_hash, source_feature_hash=None):
    """Fit exactly six fixed learners, or replay an already completed bundle.

    A launched bundle without completion is never automatically retried.
    Tree functions receive only gradient features/labels; MLP stopping targets
    are consumed exclusively by the inherited MSE early-stopping loop.
    """
    from catboost import CatBoost, CatBoostRegressor
    identity=member_identity(features,gradient,stopping,held_unlabelled,
        representation,seed,context,protocol_hash,source_feature_hash)
    directory=Path(artifact_dir)
    if (directory/'completed.json').exists():
        result=audit_member_bundle(features,gradient,stopping,held_unlabelled,
            directory,protocol_hash,identity)
        result['reused_completed_bundle']=True
        return result
    # Check rank pair availability before any fit or launch receipt.
    _rank_data(gradient.y_log1p.to_numpy(),gradient.allele.to_numpy())
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'launched.json').open('x') as handle:
        json.dump(dict(status='launched',identity=identity,pid=os.getpid(),
            at_utc=datetime.now(timezone.utc).isoformat(),planned_n_fits=6),handle)
    started=time.perf_counter()
    arrays={}
    predictions={}
    details={}
    transforms={}
    ancestry=Ancestry()
    payloads={}
    X={role:np.asarray(value) for role,value in features.items()}
    y=gradient.y_log1p.to_numpy()
    ys=stopping.y_log1p.to_numpy()
    parameter_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def launch(name,extra=None):
        path=directory/f'fit_{name}.launched.json'
        with path.open('x') as handle:
            json.dump(dict(component=name,identity_sha256=parameter_hash,gradient_rows=ancestry.rows(gradient),
                stopping_rows=ancestry.rows(stopping),prediction_rows=ancestry.rows(held_unlabelled),
                dependency=extra,status='launched',at_utc=datetime.now(timezone.utc).isoformat()),handle)
        payloads[path.name]=file_hash(path)

    def finish(name,item):
        path=directory/f'fit_{name}.completed.json'
        atomic_json(path,item)
        payloads[path.name]=file_hash(path)
        details[name]=item

    def record(role,name,*,neural=False,**extra):
        ancestry.add(role=role,fit=gradient,held=held_unlabelled,
            stopping=stopping if neural else None,component=name,context=context,
            seed=seed,transform_fit_rows=ancestry.rows(gradient),**extra)

    def neural(name,Z):
        launch(name,extra={'tree_rmse_sha256':payloads['tree_rmse.cbm']} if name=='leaf_mlp' else None)
        begin=time.perf_counter()
        model=MLPRegressor(mlp_config(seed)).fit(Z['gradient'],y,Z['stopping'],ys)
        fit_seconds=time.perf_counter()-begin
        values=model.predict(Z['held'])
        for i,(W,b) in enumerate(zip(model.weights_,model.biases_)):
            arrays[f'{name}__w{i}'],arrays[f'{name}__b{i}']=W.copy(),b.copy()
        arrays[f'{name}__offset']=np.asarray(model.y_offset_)
        arrays[f'{name}__curve']=np.asarray(model.dev_curve_)
        arrays[f'prediction__{name}']=values.copy()
        predictions[name]=values
        state=dict(component=name,learner='mlp',config=identity['mlp'],fit_seconds=fit_seconds,
            n_fit=len(gradient),n_stop=len(stopping),n_predict=len(held_unlabelled),
            n_features=Z['gradient'].shape[1],epochs=model.n_epochs_,best_epoch=model.best_epoch_,
            dimensions=dict(input=Z['gradient'].shape[1],hidden=[64],output=1),
            best_stopping_mse=model.dev_mse_,gradient_features_sha256=array_hash(Z['gradient']),
            stopping_features_sha256=array_hash(Z['stopping']),held_features_sha256=array_hash(Z['held']),
            gradient_rows=ancestry.rows(gradient),stopping_rows=ancestry.rows(stopping),
            prediction_rows=ancestry.rows(held_unlabelled),transform_fit_rows=ancestry.rows(gradient))
        if name=='leaf_mlp':
            state['tree_rmse_sha256']=payloads['tree_rmse.cbm']
            state['leaf_dictionary_sha256']=array_hash(arrays['leaf__dictionary'])
        finish(name,state)
        record('hybrid_mlp',name,neural=True)

    try:
        begin=time.perf_counter()
        standard=StandardScaler().fit(X['gradient'])
        _save_scaler(arrays,'standard_scaler',standard)
        scaled={role:standard.transform(value) for role,value in X.items()}
        transforms['standard_scaler']=dict(parameters=identity['preprocessing']['standard'],
            fit_rows=ancestry.rows(gradient),seconds=time.perf_counter()-begin)
        record('hybrid_transform','standard_scaler')
        neural('standard',scaled)

        trees={}
        for name,loss in (('tree_rmse','RMSE'),('tree_rank','PairLogit')):
            config={**TREE_PARAMETERS,'random_seed':seed,'loss_function':loss,'eval_metric':loss}
            launch(name)
            begin=time.perf_counter()
            if name=='tree_rmse':
                model=CatBoostRegressor(**config)
                model.fit(scaled['gradient'],y)
                pair_details=None
            else:
                model=CatBoost(config)
                pool,pair_details=_rank_pool(scaled['gradient'],y,gradient.allele.to_numpy())
                model.fit(pool)
                del pool
            fit_seconds=time.perf_counter()-begin
            if model.tree_count_!=512 or np.any(np.asarray(model.get_tree_leaf_counts())>16):
                raise ValueError('fixed512/depth4 tree recipe changed')
            path=directory/f'{name}.cbm'
            temporary=directory/f'{name}.tmp.cbm'
            model.save_model(str(temporary),format='cbm')
            temporary.replace(path)
            payloads[path.name]=file_hash(path)
            # CatBoost stores rounded decimal metadata inside CBM. Bind both
            # the original effective parameters and their exact wire form.
            wire_model=CatBoostRegressor() if name=='tree_rmse' else CatBoost()
            wire_model.load_model(str(path),format='cbm')
            wire_parameters=wire_model.get_all_params()
            _check_parameter_tree(wire_parameters,model.get_all_params())
            del wire_model
            values=np.asarray(model.predict(scaled['held'],thread_count=2),dtype=np.float64)
            arrays[f'prediction__{name}']=values.copy()
            predictions[name]=values
            trees[name]=model
            finish(name,dict(component=name,learner='catboost',config=config,effective_parameters=model.get_all_params(),
                serialized_effective_parameters=wire_parameters,
                fit_seconds=fit_seconds,tree_count=int(model.tree_count_),n_features=scaled['gradient'].shape[1],
                dimensions=dict(input=scaled['gradient'].shape[1],tree_count=int(model.tree_count_),output=1),
                n_fit=len(gradient),n_stop=0,n_stop_reserved=len(stopping),n_predict=len(held_unlabelled),
                gradient_features_sha256=array_hash(scaled['gradient']),gradient_rows=ancestry.rows(gradient),
                gradient_targets_sha256=identity['roles']['gradient']['labels_sha256'],
                prediction_rows=ancestry.rows(held_unlabelled),transform_fit_rows=ancestry.rows(gradient),
                cbm_sha256=payloads[path.name],pairing=pair_details,
                no_stopping_labels=True,no_eval_set=True))
            record('hybrid_tree',name)

        begin=time.perf_counter()
        leaf={role:encode_leaves(trees['tree_rmse'],value) for role,value in scaled.items()}
        arrays['leaf__dictionary']=leaf_dictionary()
        leaf_scale=StandardScaler().fit(leaf['gradient'])
        _save_scaler(arrays,'leaf_scaler',leaf_scale)
        leaf_scaled={role:leaf_scale.transform(value) for role,value in leaf.items()}
        transforms['leaf']=dict(parameters=identity['preprocessing']['leaf'],fit_rows=ancestry.rows(gradient),
            dictionary_sha256=array_hash(arrays['leaf__dictionary']),tree_rmse_sha256=payloads['tree_rmse.cbm'],
            seconds=time.perf_counter()-begin)
        record('hybrid_transform','leaf',tree_rmse_sha256=payloads['tree_rmse.cbm'])
        neural('leaf_mlp',leaf_scaled)
        del leaf,leaf_scaled

        begin=time.perf_counter()
        quantile=QuantileTransformer(**identity['preprocessing']['quantile']).fit(X['gradient'])
        arrays['quantile__quantiles']=quantile.quantiles_.copy()
        arrays['quantile__references']=quantile.references_.copy()
        quantiled={role:quantile.transform(value) for role,value in X.items()}
        quantile_scale=StandardScaler().fit(quantiled['gradient'])
        _save_scaler(arrays,'quantile_scaler',quantile_scale)
        quantile_scaled={role:quantile_scale.transform(value) for role,value in quantiled.items()}
        transforms['quantile']=dict(parameters=identity['preprocessing']['quantile'],
            fit_rows=ancestry.rows(gradient),n_quantiles=int(quantile.n_quantiles_),seconds=time.perf_counter()-begin)
        record('hybrid_transform','quantile')
        neural('quantile_mlp',quantile_scaled)
        del quantiled,quantile_scaled

        begin=time.perf_counter()
        pca=PCA(**identity['preprocessing']['pca']).fit(scaled['gradient'])
        for name in ('components','mean','explained_variance','explained_variance_ratio','singular_values','noise_variance'):
            arrays[f'pca__{name}']=np.asarray(getattr(pca,name+'_')).copy()
        projected={role:pca.transform(value) for role,value in scaled.items()}
        transforms['pca']=dict(parameters=identity['preprocessing']['pca'],fit_rows=ancestry.rows(gradient),
            n_components=128,seconds=time.perf_counter()-begin,
            gradient_input_sha256=array_hash(scaled['gradient']),no_second_scaler=True)
        record('hybrid_transform','pca')
        neural('pca_mlp',projected)

        for name,values in predictions.items():
            if values.shape!=(len(held_unlabelled),) or not np.isfinite(values).all():
                raise ValueError(f'nonfinite or unaligned member predictions: {name}')
        array_path=directory/'arrays.npz'
        tmp=directory/'arrays.tmp.npz'
        np.savez_compressed(tmp,**arrays)
        tmp.replace(array_path)
        payloads[array_path.name]=file_hash(array_path)
        complete=dict(schema=SCHEMA,status='complete',identity=identity,n_fits=6,
            component_details=details,transforms=transforms,ancestry=ancestry.as_dict(),payload_sha256=payloads,
            array_schema={key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=array_hash(value)) for key,value in arrays.items()},
            total_fit_seconds=sum(d['fit_seconds'] for d in details.values()),
            transform_seconds=sum(d['seconds'] for d in transforms.values()),
            bundle_seconds=time.perf_counter()-started,peak_rss_bytes=_rss_bytes(),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            timing_scope='supervised-fit sum excludes transforms/serialization; bundle elapsed includes both; process peak RSS includes earlier work')
        atomic_json(directory/'completed.json',complete)
        return dict(predictions=predictions,n_fits=6,identity=identity,receipt=str(directory/'completed.json'),
                    artifacts=payloads,total_fit_seconds=complete['total_fit_seconds'],peak_rss_bytes=complete['peak_rss_bytes'])
    except BaseException as error:
        atomic_json(directory/'failed.json',dict(status='failed',identity=identity,error=repr(error),
            completed_supervised_components=list(details),actual_completed_fits=len(details),
            seconds=time.perf_counter()-started,peak_rss_bytes=_rss_bytes()))
        raise


@threadpool_limits.wrap(limits=2)
def audit_member_bundle(features, gradient, stopping, held_unlabelled, artifact_dir,
                        protocol_hash, expected_identity):
    """Replay immutable transforms and learners with no supervised refitting.

    Scalings and quantiles are checked from gradient sufficient statistics.
    PCA's saved mean, dimensions, orthogonality and variance identities are
    checked; its unchanged source-bound fitted state is then replayed, without
    learning a second PCA. Tree/leaf dependencies and all learned-input hashes
    are checked against the independently supplied role arrays and metadata.
    """
    from catboost import CatBoost, CatBoostRegressor
    directory=Path(artifact_dir)
    receipt_path=directory/'completed.json'
    complete=json.loads(receipt_path.read_text())
    identity=member_identity(features,gradient,stopping,held_unlabelled,
        expected_identity['representation'],expected_identity['seed'],expected_identity['context'],
        protocol_hash,expected_identity.get('source_feature_hash'))
    if complete.get('schema')!=SCHEMA or complete.get('status')!='complete' or complete.get('n_fits')!=6:
        raise ValueError('completed six-fit bundle required')
    if complete['identity']!=expected_identity or identity!=expected_identity:
        raise ValueError('bundle identity/source/recipe/role features changed')
    if set(complete['component_details'])!=set(COMPONENTS):
        raise ValueError('incomplete or changed component registry')
    payloads=complete['payload_sha256']
    expected_payloads={'arrays.npz','tree_rmse.cbm','tree_rank.cbm'} | {
        f'fit_{component}.{state}.json' for component in COMPONENTS for state in ('launched','completed')}
    if set(payloads)!=expected_payloads:
        raise ValueError('bundle payload coverage changed')
    for filename,digest in payloads.items():
        if Path(filename).name!=filename or file_hash(directory/filename)!=digest:
            raise ValueError(f'immutable bundle payload changed: {filename}')
    with np.load(directory/'arrays.npz',allow_pickle=False) as raw:
        if set(raw.files)!=set(complete['array_schema']):
            raise ValueError('bundle array coverage changed')
        arrays={name:raw[name].copy() for name in raw.files}
    expected_arrays={f'{c}__{name}' for c in MLP_COMPONENTS for name in ('w0','w1','b0','b1','offset','curve')}
    expected_arrays |= {f'{p}__{name}' for p in ('standard_scaler','leaf_scaler','quantile_scaler') for name in ('mean','var','scale')}
    expected_arrays |= {'quantile__quantiles','quantile__references','leaf__dictionary'}
    expected_arrays |= {f'pca__{name}' for name in ('components','mean','explained_variance','explained_variance_ratio','singular_values','noise_variance')}
    expected_arrays |= {f'prediction__{c}' for c in COMPONENTS}
    if set(arrays)!=expected_arrays:
        raise ValueError('fixed transform/MLP/prediction array schema changed')
    for name,value in arrays.items():
        saved=complete['array_schema'][name]
        if value.dtype.kind not in 'fi' or not np.isfinite(value).all() or list(value.shape)!=saved['shape'] or str(value.dtype)!=saved['dtype'] or array_hash(value)!=saved['sha256']:
            raise ValueError(f'array shape/dtype/numerical/hash state changed: {name}')
    X={role:np.asarray(value) for role,value in features.items()}
    standard=_check_scaler(arrays,'standard_scaler',X['gradient'])
    scaled={role:standard.transform(value) for role,value in X.items()}
    ancestry=Ancestry()
    source_graph=complete['ancestry']
    ancestry.rowsets.update(source_graph['rowsets'])
    for key,ids in ancestry.rowsets.items():
        if ids!=sorted(set(ids)) or array_hash(np.asarray(ids,dtype='<i8'))!=key:
            raise ValueError('bundle ancestry rowset content changed')
    expected=Ancestry()
    seed=identity['seed']
    context=identity['context']

    def record(role,name,*,neural=False,**extra):
        expected.add(role=role,fit=gradient,held=held_unlabelled,
            stopping=stopping if neural else None,component=name,context=context,
            seed=seed,transform_fit_rows=expected.rows(gradient),**extra)

    gradient_rows=expected.rows(gradient)
    stopping_rows=expected.rows(stopping)
    held_rows=expected.rows(held_unlabelled)
    for name,details in complete['transforms'].items():
        if details['fit_rows']!=gradient_rows or details['seconds']<0:
            raise ValueError('transform fitted outside gradient partition')
    if set(complete['transforms'])!={'standard_scaler','leaf','quantile','pca'}:
        raise ValueError('transform coverage changed')
    if complete['transforms']['standard_scaler']['parameters']!=identity['preprocessing']['standard']:
        raise ValueError('standard scaler recipe changed')
    record('hybrid_transform','standard_scaler')
    output={}

    def neural(name,Z):
        details=complete['component_details'][name]
        required=dict(component=name,learner='mlp',config=identity['mlp'],n_fit=len(gradient),
            n_stop=len(stopping),n_predict=len(held_unlabelled),n_features=Z['gradient'].shape[1],
            dimensions=dict(input=Z['gradient'].shape[1],hidden=[64],output=1),
            gradient_features_sha256=array_hash(Z['gradient']),stopping_features_sha256=array_hash(Z['stopping']),
            held_features_sha256=array_hash(Z['held']),gradient_rows=gradient_rows,stopping_rows=stopping_rows,
            prediction_rows=held_rows,transform_fit_rows=gradient_rows)
        if any(details.get(key)!=value for key,value in required.items()):
            raise ValueError(f'MLP role/feature/recipe metadata changed: {name}')
        dimensions=(Z['gradient'].shape[1],64,1)
        for i,(nin,nout) in enumerate(zip(dimensions[:-1],dimensions[1:])):
            if arrays[f'{name}__w{i}'].shape!=(nin,nout) or arrays[f'{name}__w{i}'].dtype!=np.float32 or arrays[f'{name}__b{i}'].shape!=(nout,) or arrays[f'{name}__b{i}'].dtype!=np.float32:
                raise ValueError('MLP parameter shape/dtype changed')
        model=_restore_mlp(arrays,name,seed)
        # The target offset is another gradient-only learned parameter.
        if model.y_offset_!=float(gradient.y_log1p.to_numpy(dtype=np.float32).mean()):
            raise ValueError('MLP target centering used wrong labels')
        curves=arrays[f'{name}__curve']
        epochs,best=details['epochs'],details['best_epoch']
        if curves.shape!=(epochs,) or not 1<=best<=epochs<=200 or int(np.argmin(curves))+1!=best or details['fit_seconds']<0:
            raise ValueError('MSE stopping curve/epoch changed')
        if epochs<200 and epochs-best!=15:
            raise ValueError('MLP stopped before frozen MSE patience')
        mse=float(np.mean((model.predict(Z['stopping'])-stopping.y_log1p.to_numpy(dtype=np.float32))**2))
        np.testing.assert_allclose([mse,details['best_stopping_mse']],[curves[best-1]]*2,rtol=0,atol=1e-12)
        values=model.predict(Z['held'])
        np.testing.assert_array_equal(values,arrays[f'prediction__{name}'])
        output[name]=values
        record('hybrid_mlp',name,neural=True)
        if json.loads((directory/f'fit_{name}.completed.json').read_text())!=details:
            raise ValueError('MLP completion receipt changed')

    neural('standard',scaled)
    trees={}
    for name,loss in (('tree_rmse','RMSE'),('tree_rank','PairLogit')):
        model=CatBoostRegressor() if name=='tree_rmse' else CatBoost()
        model.load_model(str(directory/f'{name}.cbm'),format='cbm')
        details=complete['component_details'][name]
        config={**TREE_PARAMETERS,'random_seed':seed,'loss_function':loss,'eval_metric':loss}
        required=dict(component=name,learner='catboost',config=config,tree_count=512,
            n_features=scaled['gradient'].shape[1],dimensions=dict(input=scaled['gradient'].shape[1],tree_count=512,output=1),
            n_fit=len(gradient),n_stop=0,n_stop_reserved=len(stopping),n_predict=len(held_unlabelled),
            gradient_features_sha256=array_hash(scaled['gradient']),gradient_rows=gradient_rows,
            gradient_targets_sha256=identity['roles']['gradient']['labels_sha256'],prediction_rows=held_rows,
            transform_fit_rows=gradient_rows,cbm_sha256=payloads[f'{name}.cbm'],no_stopping_labels=True,no_eval_set=True)
        if any(details.get(key)!=value for key,value in required.items()) or details['fit_seconds']<0:
            raise ValueError('tree supervised gradient boundary or recipe changed')
        if model.tree_count_!=512 or np.any(np.asarray(model.get_tree_leaf_counts())>16):
            raise ValueError('saved tree budget/depth changed')
        if model.get_all_params()!=details['serialized_effective_parameters']:
            raise ValueError('saved serialized effective CatBoost parameters changed')
        _check_parameter_tree(model.get_all_params(),details['effective_parameters'])
        for key in ('iterations','depth','random_seed','border_count','grow_policy','bootstrap_type','learning_rate','l2_leaf_reg','rsm','random_strength','subsample','task_type'):
            actual=model.get_all_params()[key]
            if isinstance(config[key],float):
                np.testing.assert_allclose(actual,config[key],rtol=1e-6,atol=1e-8)
            elif actual!=config[key]:
                raise ValueError(f'effective tree parameter changed: {key}')
        if details['pairing']!=(_rank_data(gradient.y_log1p.to_numpy(),gradient.allele.to_numpy())[-1] if name=='tree_rank' else None):
            raise ValueError('gradient rank pairs/ties/allele weights changed')
        values=np.asarray(model.predict(scaled['held'],thread_count=2),dtype=np.float64)
        np.testing.assert_array_equal(values,arrays[f'prediction__{name}'])
        output[name]=values
        trees[name]=model
        record('hybrid_tree',name)
        if json.loads((directory/f'fit_{name}.completed.json').read_text())!=details:
            raise ValueError('tree completion receipt changed')

    np.testing.assert_array_equal(arrays['leaf__dictionary'],leaf_dictionary())
    leaf_details=complete['transforms']['leaf']
    if leaf_details['parameters']!=identity['preprocessing']['leaf'] or leaf_details['dictionary_sha256']!=array_hash(leaf_dictionary()) or leaf_details['tree_rmse_sha256']!=payloads['tree_rmse.cbm']:
        raise ValueError('fixed leaf dictionary or shared RMSE tree identity changed')
    leaf={role:encode_leaves(trees['tree_rmse'],value) for role,value in scaled.items()}
    leaf_scale=_check_scaler(arrays,'leaf_scaler',leaf['gradient'])
    leaf_scaled={role:leaf_scale.transform(value) for role,value in leaf.items()}
    mlp_details=complete['component_details']['leaf_mlp']
    if mlp_details['tree_rmse_sha256']!=payloads['tree_rmse.cbm'] or mlp_details['leaf_dictionary_sha256']!=array_hash(arrays['leaf__dictionary']):
        raise ValueError('leaf MLP did not reuse its exact member RMSE tree')
    record('hybrid_transform','leaf',tree_rmse_sha256=payloads['tree_rmse.cbm'])
    neural('leaf_mlp',leaf_scaled)
    del leaf,leaf_scaled

    qt_params=identity['preprocessing']['quantile']
    if complete['transforms']['quantile']['parameters']!=qt_params:
        raise ValueError('quantile numerical recipe changed')
    references=np.linspace(0,1,min(256,len(gradient)),endpoint=True)
    quantiles=np.nanpercentile(X['gradient'],references*100,axis=0)
    quantiles=np.maximum.accumulate(quantiles)
    np.testing.assert_array_equal(arrays['quantile__references'],references)
    np.testing.assert_array_equal(arrays['quantile__quantiles'],quantiles)
    quantile=_restore_quantile(arrays,qt_params)
    quantiled={role:quantile.transform(value) for role,value in X.items()}
    qt_scale=_check_scaler(arrays,'quantile_scaler',quantiled['gradient'])
    quantile_scaled={role:qt_scale.transform(value) for role,value in quantiled.items()}
    record('hybrid_transform','quantile')
    neural('quantile_mlp',quantile_scaled)
    del quantiled,quantile_scaled

    pca_details=complete['transforms']['pca']
    if pca_details['parameters']!=identity['preprocessing']['pca'] or pca_details['n_components']!=128 or not pca_details['no_second_scaler'] or pca_details['gradient_input_sha256']!=array_hash(scaled['gradient']):
        raise ValueError('PCA gradient source/recipe changed')
    if arrays['pca__components'].shape!=(128,X['gradient'].shape[1]):
        raise ValueError('PCA component width changed')
    np.testing.assert_array_equal(arrays['pca__mean'],scaled['gradient'].mean(axis=0))
    np.testing.assert_allclose(arrays['pca__components']@arrays['pca__components'].T,np.eye(128),rtol=0,atol=5e-5)
    np.testing.assert_allclose(arrays['pca__explained_variance'],arrays['pca__singular_values']**2/(len(gradient)-1),rtol=1e-6,atol=1e-6)
    pca=_restore_pca(arrays,identity['preprocessing']['pca'],len(gradient))
    projected={role:pca.transform(value) for role,value in scaled.items()}
    record('hybrid_transform','pca')
    neural('pca_mlp',projected)

    if source_graph['records']!=expected.records or source_graph['rowsets']!=expected.rowsets:
        raise ValueError('complete supervised/transform ancestry graph changed')
    for name in COMPONENTS:
        launch=json.loads((directory/f'fit_{name}.launched.json').read_text())
        if launch['component']!=name or launch['identity_sha256']!=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest():
            raise ValueError('per-fit launch identity changed')
    np.testing.assert_allclose(complete['total_fit_seconds'],sum(d['fit_seconds'] for d in complete['component_details'].values()),rtol=0,atol=1e-12)
    if complete['peak_rss_bytes']<=0 or complete['bundle_seconds']<0 or complete['transform_seconds']<0:
        raise ValueError('runtime/memory receipt invalid')
    return dict(predictions=output,n_fits=6,identity=identity,receipt=str(receipt_path),
        artifacts=payloads,total_fit_seconds=complete['total_fit_seconds'],peak_rss_bytes=complete['peak_rss_bytes'],
        audit=dict(status='passed',saved_weights_and_CBM_replayed=True,gradient_only_transform_stats=True,
                   PCA_state_replayed_without_refit=True,RMSE_tree_shared_with_leaf=True,
                   held_labels_rejected=True,cluster_roles_pairwise_disjoint=True))
