"""Synthetic arithmetic and ancestry checks; no canonical fitting/resampling."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.followup3_representation_independent_audit import balanced, scores, fitting_roles, contrast_weights


def test_independent_rank_metric_handles_ties_and_constant_predictions():
    frame = pd.DataFrame(dict(allele=['A']*6+['B']*6,y_log1p=[1,1,2,3,4,5,1,2,2,3,5,8]))
    p = np.asarray([2,2,1,4,3,5,8,7,7,4,3,1],float)
    result,panel = scores(frame,{'p':p,'constant':np.zeros(12)},4)
    expected = np.median([spearmanr(frame.y_log1p.iloc[:6],p[:6]).statistic,spearmanr(frame.y_log1p.iloc[6:],p[6:]).statistic])
    assert panel==['A','B'] and result['constant']==0
    np.testing.assert_allclose(result['p'],expected,atol=1e-15)


def test_independent_group_assignments_match_frozen_equal_target_contract():
    from pepstab.nested import inner_folds
    frame = pd.DataFrame(dict(cluster_id=[9]*4+[2]*3+[7]*2+[1,3,5,8,10],outer_fold=np.arange(14)%5))
    for n in (3,5):
        np.testing.assert_array_equal(balanced(frame,n),inner_folds(frame,n))
    frame['outer_fold']=frame.cluster_id%5
    for fold in range(5):
        for inner in (None,0,1,2):
            fit,held=fitting_roles(frame,fold,inner)
            assert not set(fit.cluster_id)&set(held.cluster_id)
            assert not set(fit.cluster_id)&set(frame.loc[frame.outer_fold.eq(fold),'cluster_id'])


def test_complete_declared_contrasts_and_interactions():
    inverse=contrast_weights('inverse');energy=contrast_weights('energy')
    assert len(inverse)==7 and len(energy)==3
    assert inverse[('R5_P','R5_PC')]=={'R5_PC':1.,'R5_P':-1.}
    assert energy[('R0_energy_gain','R5_energy_gain')]=={'R5_E':1.,'R5':-1.,'R0_E':-1.,'R0':1.}
