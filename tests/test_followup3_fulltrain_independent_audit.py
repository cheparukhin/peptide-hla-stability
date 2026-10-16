import numpy as np
import pandas as pd
import pytest
from scripts import followup3_fulltrain_independent_audit as check
from scripts import followup3_representation_independent_audit as shared
from pepstab.followup3_inverse_fulltrain import fixed_query_table


def fixture():
    return pd.DataFrame([dict(pair_id=i,cluster_id=i//2,allele='A' if i<60 else 'B',peptide=f'p{i}',outer_fold=i%5,y_log1p=float(i%11)) for i in range(120)])


def test_fixed_query_replays_nonzero_changes_and_rejects_tampering():
    frame=fixture();y=frame.y_log1p.to_numpy()
    old={'R5':y,'R5_P':np.zeros(len(y)),'R5_PC':-y}
    new={'R5':y,'R5_P':y,'R5_PC':y}
    saved=fixed_query_table(frame,old,new,'mlp')
    assert check.check_fixed_query(frame,old,new,saved,'mlp')=={'20':['A','B'],'50':['A','B']}
    changed=saved.copy();changed.loc[changed.comparison.eq('net'),'old_to_new_change']+=.1
    with pytest.raises(AssertionError):check.check_fixed_query(frame,old,new,changed,'mlp')
    changed=saved.copy();changed.loc[0,'panel_alleles']='A'
    with pytest.raises(AssertionError):check.check_fixed_query(frame,old,new,changed,'mlp')


def test_comparison_replays_saved_quantiles_without_bootstrap_generation():
    frame=fixture();y=frame.y_log1p.to_numpy()
    predictions={'R5':y,'R5_P':np.zeros(len(y)),'R5_PC':-y}
    point,_=shared.scores(frame,predictions,20)
    boot=pd.DataFrame({'R5':np.linspace(-.2,.4,2000),'R5_P':np.linspace(.1,.3,2000),'R5_PC':np.linspace(-.5,.2,2000)})
    rows=[]
    for b,c in check.CONTRASTS:
        lo,hi=np.quantile(boot[c]-boot[b],[.025,.975])
        rows.append(dict(baseline=b,candidate=c,baseline_score=point[b],candidate_score=point[c],delta=point[c]-point[b],ci_low=lo,ci_high=hi,panel_min_rows=20,n_rows=120,n_alleles=2,n_clusters=60,n_boot=2000,bootstrap_seed=20261003))
    saved=pd.DataFrame(rows)
    assert check.check_comparisons(frame,predictions,saved,boot,20)[1]==['A','B']
    changed=saved.copy();changed.loc[0,'ci_low']+=.01
    with pytest.raises(AssertionError):check.check_comparisons(frame,predictions,changed,boot,20)
    with pytest.raises(AssertionError):check.check_comparisons(frame,predictions,saved,boot.iloc[:-1],20)


def test_fulltrain_audit_refuses_unfrozen_candidate(tmp_path,monkeypatch):
    monkeypatch.setattr('sys.argv',['audit','--directory',str(tmp_path)])
    with pytest.raises(FileNotFoundError):check.main()
    assert not (tmp_path/'independent_audit.json').exists()


def test_four_batch_partition_rejects_overlap_cache_leak_and_missing_rows():
    batches=[[1,2],[3,4],[5,6],[7]]
    assert check.check_batch_partition(set(range(8)),{0},batches,(2,2,2,1))==list(map(set,batches))
    for changed in ([[1,2],[3,4],[5,6],[6]], [[0,2],[3,4],[5,6],[7]], [[1,2],[3,4],[5,6],[8]]):
        with pytest.raises(AssertionError):check.check_batch_partition(set(range(8)),{0},changed,(2,2,2,1))
    with pytest.raises(AssertionError):check.check_batch_partition(set(range(8)),{0},batches[:2],(2,2))
