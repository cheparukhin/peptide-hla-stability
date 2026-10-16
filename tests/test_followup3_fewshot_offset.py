"""Tiny synthetic checks; no original diagnostic metrics or paid computation."""
import numpy as np
import pandas as pd
import pytest
import json
from scripts.followup3_fewshot_offset import (
    NAMES, contrasts, offset_cell, score_control, source_closure)
from pepstab.followup3_fewshot import ARMS, BUDGETS, conditional_scores


def cell_fixture():
    table = pd.DataFrame(dict(pair_id=[0, 1, 2, 3], cluster_id=[0, 1, 2, 3],
        hla_group=[0, 0, 1, 1], allele=['a', 'a', 'b', 'b'], y_log1p=[0., 1., 2., 3.],
        R5_k0=[1., 2., 4., 5.], R5_k20=[1., 3., 6., 5.]))
    cell = dict(index=0, draw=0, peptide_fold=0, query_ids=[0, 1, 2, 3],
                support_ids={'a':list(range(100, 120)), 'b':list(range(200, 220))})
    rows = [dict(arm='R5', budget=20, allele=a, key=a,
        support_ids=cell['support_ids'][a], query_ids=table.loc[table.allele.eq(a), 'pair_id'].tolist(),
        labels=20, stopping_labels=0, tuning_labels=0) for a in ('a', 'b')]
    arrays = {'a_offset':np.asarray(7.), 'b_offset':np.asarray(-4.)}
    return table, cell, rows, arrays


def test_saved_scalar_only_and_exact_rank_invariance():
    table, cell, rows, arrays = cell_fixture()
    result, ranks, offsets = offset_cell(table, cell, rows, arrays, ['a', 'b'])
    np.testing.assert_array_equal(result.R5_offset20, [8., 9., 0., 1.])
    assert all(row['ranks_identical'] for row in ranks)
    assert [row['offset'] for row in offsets] == [7., -4.]
    assert table.columns.tolist()[-1] == 'R5_k20'


@pytest.mark.parametrize('kind', ['support', 'query', 'duplicate', 'nonfinite', 'rounding'])
def test_adapter_ancestry_and_bad_offset_rejected(kind):
    table, cell, rows, arrays = cell_fixture()
    if kind == 'support':
        rows[0]['support_ids'] = rows[0]['support_ids'][::-1]
    elif kind == 'query':
        rows[0]['query_ids'] = [1, 0]
    elif kind == 'duplicate':
        rows.append(rows[0])
    elif kind == 'nonfinite':
        arrays['a_offset'] = np.asarray(np.nan)
    else:
        arrays['a_offset'] = np.asarray(1e30)
    with pytest.raises(ValueError):
        offset_cell(table, cell, rows, arrays, ['a', 'b'])


def test_crossed_bootstrap_matches_original_population_rng_and_score_aggregation():
    rows = []
    for allele in range(5):
        for cluster in range(13):
            rows.append(dict(pair_id=len(rows), cluster_id=cluster, hla_group=allele,
                             allele=f'a{allele}', y_log1p=float(cluster % 7)))
    # Two groups and four clusters with no query rows must still enter resampling.
    frame = pd.DataFrame(rows + [dict(pair_id=65+i, cluster_id=13+i, hla_group=5+i % 2,
        allele='offpanel', y_log1p=float(i)) for i in range(4)])
    tables = {}
    for draw in range(3):
        table = frame.iloc[:65].copy()
        x = table.y_log1p.to_numpy()
        # Different ranking quality per draw ensures score averaging is tested.
        zero = x + np.sin(np.arange(65) + draw) * (draw + 1)
        for arm in ARMS:
            for budget in BUDGETS:
                table[f'{arm}_k{budget}'] = zero if budget == 0 else x + np.cos(np.arange(65))
        table['R5_offset20'] = zero + table.cluster_id.to_numpy() % 3
        tables[draw] = table
    panel = [f'a{i}' for i in range(5)]
    point, _, boot = score_control(frame, tables, panel, n_boot=9, seed=741)
    original_points, original_boot, _ = conditional_scores(
        frame, tables, {20:panel}, n_boot=9, seed=741, mode='crossed')
    for name in NAMES[:2]:
        np.testing.assert_array_equal(boot[name], original_boot[20][name])
        selected = original_points.loc[original_points.arm.eq('R5') &
                                      original_points.budget.eq(int(name.split('_k')[1]))]
        np.testing.assert_array_equal(point.loc[point.prediction.eq(name)].score, selected.score)
    comparisons, differences = contrasts(point, boot)
    np.testing.assert_allclose(comparisons.delta.iloc[0] + comparisons.delta.iloc[1],
                               comparisons.delta.iloc[2], atol=1e-15)
    np.testing.assert_allclose(differences['full_minus_offset'] + differences['offset_minus_zero'],
                               differences['original_full_minus_zero'], atol=1e-15)


def test_runtime_closure_includes_indirect_canonical_imports():
    closure = source_closure()
    assert len(closure) == len(set(closure))
    for name in ('scripts/round2_transfer.py', 'scripts/round2_embedding_screen.py',
                 'pepstab/nested.py', 'pepstab/esm.py', 'scripts/phase_d_structural_residual.py',
                 'tests/test_followup3_fewshot.py'):
        assert name in closure


def test_report_refuses_local_execution_before_context(monkeypatch):
    from scripts import followup3_fewshot_offset as diagnostic
    monkeypatch.setattr(diagnostic.original.sys, 'platform', 'darwin')
    with pytest.raises(ValueError, match='Modal/Linux'):
        diagnostic.report()


def test_frozen_input_tampering_rejected_before_scoring(tmp_path, monkeypatch):
    from scripts import followup3_fewshot_offset as diagnostic
    monkeypatch.setattr(diagnostic, 'ROOT', tmp_path)
    payload = tmp_path / 'input.csv'
    payload.write_text('frozen input\n')
    protocol = dict(source_sha256={}, input_sha256={'input.csv':diagnostic.original.sha(payload)})
    protocol['protocol_hash'] = diagnostic.study.digest(protocol)
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    payload.write_text('changed input\n')
    with pytest.raises(ValueError, match='input_sha256 changed'):
        diagnostic.validate_freeze(tmp_path, tmp_path)
