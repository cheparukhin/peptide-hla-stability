"""Small synthetic guards only; no canonical fitting or reporting."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from pepstab.followup3_fewshot import (make_plan,validate_roles,fit_adapter,conditional_scores,fixed_panels,ARMS,BUDGETS)


def fixture_frame():
    rows=[]
    for allele in range(10):
        for cluster in range(100):
            rows.append(dict(pair_id=len(rows),cluster_id=cluster,hla_group=allele,hla_fold=allele//2,
                peptide_fold=cluster%5,allele=f'HLA-{allele}',split='train',y_log1p=float(cluster%13)))
    return pd.DataFrame(rows)


def test_support_selection_label_blind_and_exact_nested_budget():
    frame=fixture_frame();plan=make_plan(frame)
    changed=frame.copy();changed['y_log1p']=np.arange(len(changed))[::-1]
    assert make_plan(changed)==plan
    assert len(plan['cells'])==75
    assert plan['source_fits']==450
    assert plan['adapter_fits']==10*5*3*3*3
    for cell in plan['cells']:
        lookup=frame.set_index('pair_id',drop=False)
        source=lookup.loc[cell['source_ids']]
        support=lookup.loc[[p for ids in cell['support_ids'].values() for p in ids]]
        query=lookup.loc[cell['query_ids']]
        validate_roles(source,support,query)
        assert not set(source.cluster_id)&set(cell['excluded_support_clusters'])
        assert not set(query.cluster_id)&set(cell['excluded_support_clusters'])
        for ids in cell['support_ids'].values():
            assert len(ids)==20 and len(set(ids))==20
            assert set(ids[:5])<=set(ids[:10])<=set(ids[:20])


def test_cross_allele_support_query_family_leakage_rejected():
    frame=fixture_frame();cell=make_plan(frame)['cells'][0];lookup=frame.set_index('pair_id',drop=False)
    source=lookup.loc[cell['source_ids']];support=lookup.loc[[p for ids in cell['support_ids'].values() for p in ids]]
    query=lookup.loc[cell['query_ids']].copy();query.iloc[0,query.columns.get_loc('cluster_id')]=support.cluster_id.iloc[0]
    with pytest.raises(ValueError,match='support/query'):validate_roles(source,support,query)


def test_source_support_family_leakage_rejected():
    frame=fixture_frame();cell=make_plan(frame)['cells'][0];lookup=frame.set_index('pair_id',drop=False)
    source=lookup.loc[cell['source_ids']].copy();support=lookup.loc[[p for ids in cell['support_ids'].values() for p in ids]]
    query=lookup.loc[cell['query_ids']]
    source.iloc[0,source.columns.get_loc('cluster_id')]=support.cluster_id.iloc[0]
    with pytest.raises(ValueError,match='overlapping cluster_id'):validate_roles(source,support,query)


def test_adapter_dual_matches_intercept_ridge_and_single_label_constant():
    rng=np.random.default_rng(17);zs=rng.normal(size=(5,7));zq=rng.normal(size=(9,7));y=rng.normal(size=5)
    update,state=fit_adapter(zs,y,zq)
    reference=Ridge(alpha=1.,fit_intercept=True).fit(zs/np.sqrt(7),y).predict(zq/np.sqrt(7))
    np.testing.assert_allclose(update,reference,rtol=1e-12,atol=1e-12)
    single,_=fit_adapter(zs[:1],y[:1],zq)
    np.testing.assert_array_equal(single,np.repeat(y[0],len(zq)))
    with pytest.raises(ValueError,match='nonfinite'):fit_adapter(zs,np.full(5,np.nan),zq)


def test_paired_bootstrap_averages_scores_and_preserves_identical_arm_gaps():
    frame=fixture_frame();plan=make_plan(frame);lookup=frame.set_index('pair_id',drop=False);tables={}
    for draw in range(3):
        table=lookup.loc[[r['pair_id'] for r in plan['queries'] if r['draw']==draw]].copy()
        for arm in ARMS:
            for k in BUDGETS:table[f'{arm}_k{k}']=table.y_log1p.to_numpy()+k
        tables[draw]=table
    panels=fixed_panels(frame,plan)
    for mode in ('crossed','peptide','hla'):
        point,boots,meta=conditional_scores(frame,tables,panels,n_boot=5,mode=mode)
        assert len(point)==2*3*3*4
        assert meta['n_boot']==5
        for vectors in boots.values():
            np.testing.assert_array_equal(vectors['R5_k20']-vectors['R5_k0'],np.zeros(5))
            np.testing.assert_array_equal(vectors['R0_k20'],vectors['R5_k20'])


def test_original_validation_or_duplicate_identity_rejected():
    frame=fixture_frame();frame.loc[0,'split']='test'
    with pytest.raises(ValueError,match='original training'):make_plan(frame)
