import numpy as np
import pandas as pd

from scripts.round2_transfer_manifest import grouped_hlas, regime_masks, balanced_group_folds


def test_construct_and_near_sequence_groups_are_indivisible():
    data = pd.DataFrame({"allele": ["A", "A(mut)", "B", "C"],
                         "hla_pseudoseq": ["A"*34, "C"*34, "A"*31+"D"*3, "E"*34]})
    groups = grouped_hlas(data).set_index("allele")
    assert groups.loc["A", "hla_group"] == groups.loc["A(mut)", "hla_group"] == groups.loc["B", "hla_group"]
    assert groups.loc["C", "hla_group"] != groups.loc["A", "hla_group"]


def test_joint_excludes_both_cross_quadrants_and_hla_allows_peptide_overlap():
    frame = pd.DataFrame({"hla_fold": [0, 0, 1, 1], "peptide_fold": [0, 1, 0, 1]})
    fit, held = regime_masks(frame, "joint", 0, 0)
    np.testing.assert_array_equal(fit, [False, False, False, True])
    np.testing.assert_array_equal(held, [True, False, False, False])
    fit, held = regime_masks(frame, "hla", 0)
    np.testing.assert_array_equal(fit, [False, False, True, True])
    assert held.sum() == 2


def test_group_folds_deterministic_and_use_all_folds():
    groups = pd.DataFrame({"hla_group": range(8), "rows": [12, 10, 8, 6, 6, 4, 4, 2]})
    first = balanced_group_folds(groups)
    assert first == balanced_group_folds(groups.sample(frac=1, random_state=8))
    assert set(first.values()) == set(range(5))
