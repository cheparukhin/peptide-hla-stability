import numpy as np
import pandas as pd
import pytest

from pepstab import nested
from pepstab.final_hybrid_reporting import choose_first_one_se, contrasts, paired_statistics, SUFFIXES, COMPONENTS, replay_selection


def test_mse_first_one_se_is_not_outer_argmax():
    rows = [dict(component='standard', mean=.5, se=.01),
            dict(component='tree', mean=.51, se=.02)]
    assert choose_first_one_se(rows)['component'] == 'standard'
    rows[1]['se'] = .001
    assert choose_first_one_se(rows)['component'] == 'tree'
    with pytest.raises(ValueError):
        choose_first_one_se([dict(mean=np.nan, se=.1)])


def test_exact_contrast_registry_and_shared_interaction_covariance():
    arms = [f'{rep}_{name}' for rep in ('R0', 'R5') for name in SUFFIXES]
    base = np.linspace(-.2, .2, 100)
    boot = pd.DataFrame({name: base + i*.001 for i, name in enumerate(arms)})
    # Correlated random shifts cancel exactly in the paired difference of gains.
    boot['R5_tree_rank'] = boot.R5_tree_rmse + base + .05
    boot['R0_tree_rank'] = boot.R0_tree_rmse + base + .02
    point = boot.mean().to_dict()
    rows = paired_statistics(point, boot, 20)
    assert len(contrasts()) == 24 and len(rows) == 25
    assert sum(r['role'] == 'primary' for r in rows) == 1
    sensitivity = paired_statistics(point, boot, 50)
    assert sum(r['role'] == 'primary' for r in rows+sensitivity) == 1
    interaction = rows[-1]
    np.testing.assert_allclose([interaction['delta'], interaction['ci_low'], interaction['ci_high']], .03, atol=1e-15)


def test_selection_replay_preserves_mse_fallback_and_inner_identities():
    fit = pd.DataFrame(dict(pair_id=np.arange(300), cluster_id=np.arange(300),
                            allele=['A']*300, y_log1p=np.linspace(0, 2, 300)))
    folds = nested.inner_folds(fit, 3)
    tables = []
    for inner in range(3):
        held = fit.loc[folds == inner]
        for name in COMPONENTS:
            pred = held.y_log1p.to_numpy() if name == 'standard' else -held.y_log1p.to_numpy()
            tables.append(held[['pair_id', 'cluster_id', 'allele']].assign(
                inner_fold=inner, component=name, y_pred=pred))
    table = pd.concat(tables, ignore_index=True)
    choice = replay_selection(fit, table)
    assert choice['hybrid_component'] == 'standard' and choice['blend_weight'] == 0
    assert choice['ranker_mapping']['slope'] == 1e-8
    wrong = table.copy()
    wrong.loc[0, 'cluster_id'] = 999
    with pytest.raises(ValueError):
        replay_selection(fit, wrong)
