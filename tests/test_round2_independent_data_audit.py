"""Guard outcome reservation and exact component exclusion logic."""
import numpy as np
import pandas as pd
import pytest

from scripts.round2_independent_data_audit import component_map, read_metadata, within_three


def test_transitive_bridge_is_excluded_even_without_direct_neighbour():
    seed = "AAAAAAAAA"
    bridge = "CCCAAAAAA"
    candidate = "CCCCCCAAA"
    assert within_three([candidate], [seed]).tolist() == [False]
    graph = component_map({seed, bridge, candidate})
    assert graph[seed] == graph[candidate]


def test_graph_and_direct_indicator_match_exhaustive_reference():
    rng = np.random.default_rng(104)
    peptides = sorted(set("".join(x) for x in rng.choice(list("ACD"), size=(80, 9))))
    graph = component_map(set(peptides))
    distance = np.array([[sum(a != b for a, b in zip(x, y)) for y in peptides] for x in peptides])
    reach = distance <= 3
    for k in range(len(peptides)):
        reach |= reach[:, k, None] & reach[k, None, :]
    actual = np.array([[graph[x] == graph[y] for y in peptides] for x in peptides])
    np.testing.assert_array_equal(actual, reach)
    np.testing.assert_array_equal(within_three(peptides, peptides[:5]), (distance[:, :5] <= 3).any(axis=1))


def test_empty_and_invalid_inputs():
    assert component_map(set()) == {}
    assert within_three(["AAAAAAAAA"], []).tolist() == [False]
    with pytest.raises(ValueError, match="9-mers"):
        within_three(["AAAAAAAA"], ["AAAAAAAAA"])


def test_external_outcomes_are_never_loaded(tmp_path):
    path = tmp_path / "reserved.csv"
    pd.DataFrame({"peptide_sequence": ["AAAAAAAAA"], "mhc_sequence": ["ACD"],
                  "allele": ["HLA-A*02:01 C67S"], "label": ["unreadable-reserved"],
                  "target": ["also-reserved"], "assay": ["SPA"]}).to_csv(path, index=False)
    metadata = read_metadata(path)
    assert "label" not in metadata and "target" not in metadata
    assert metadata.allele.iloc[0] == "HLA-A*02:01 C67S"
