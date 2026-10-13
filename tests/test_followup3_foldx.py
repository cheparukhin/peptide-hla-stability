import pandas as pd
import pytest
from scripts import followup3_foldx as pilot


def fixture():
    rows = [dict(allele=f"HLA-A*{i:02}:01", peptide=f"P{i:07}X", cluster_id=i, outer_fold=i % 5,
                 complex_id=f"case{i}", folder=f"production/case{i}") for i in range(40)]
    panel = pd.DataFrame(rows)
    splits = panel[["allele", "peptide"]].assign(split="train")
    return panel, splits


def test_panel_is_deterministic32_distinct_originaltrain_alleles():
    panel, splits = fixture()
    selected = pilot.select_cases(panel, splits)
    assert len(selected) == selected.allele.nunique() == 32
    pd.testing.assert_frame_equal(selected, pilot.select_cases(panel.sample(frac=1, random_state=4), splits))
    splits.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="ORIGINAL-training"):
        pilot.select_cases(panel, splits)


def test_license_rejection_classification_never_returns_program_text():
    assert pilot.access_rejection("Invalid license XYZ-private", "", 0) is True
    assert pilot.access_rejection("", "License expired", 1) is True
    assert pilot.access_rejection("", "Permission denied", 126) is True
    assert pilot.access_rejection("FoldX licensed research software", "", 0) is False


def test_protocol_scope_has_no_outcome_loader_or_model_fitting():
    source = __import__("inspect").getsource(pilot)
    assert "load_with_splits" not in source
    assert "y_log1p" not in source
    assert "rasmussen_et_al_dataset" not in source
    assert pilot.TIMEOUT_S == 1200


@pytest.mark.parametrize("invalid", ["nan", "inf", "-inf", "unexpected", float("nan"), float("inf")])
def test_nonfinite_component_rejected_before_numeric_filter(invalid):
    row = dict(pdb="native.pdb", group1="C", group2="A", interaction_energy=-12., van_der_waals_clashes=invalid)
    with pytest.raises(ValueError, match="invalid/nonfinite"):
        pilot.validated_interaction([row], ("C", "A"))


def test_shared_parser_nonfinite_string_cannot_be_silently_dropped():
    from pepstab.foldx import parse_fxout
    text = "Pdb\tGroup1\tGroup2\tInteraction Energy\tVan der Waals Clashes\nnative.pdb\tC\tA\t-12.0\tnan\n"
    parsed = parse_fxout(text)
    assert parsed[0]["van_der_waals_clashes"] == "nan"
    with pytest.raises(ValueError, match="invalid/nonfinite"):
        pilot.validated_interaction(parsed, ("C", "A"))
