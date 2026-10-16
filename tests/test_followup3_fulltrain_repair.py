import pandas as pd
import pytest

from scripts import followup3_inverse_fulltrain_repair as repair


def collected():
    row = {name: f"identity-{name}" for name in repair.METADATA_COLUMNS}
    row.update({f"{name}_score": value for name, value in list(row.items())})
    row.update(allele="a", peptide="p", P_pos_1=-0.25, C_pos_1=-0.5)
    return pd.DataFrame([row])


def test_only_redundant_metadata_is_removed_before_original_checks(monkeypatch):
    scores = collected()
    panel = scores[["allele", "peptide", *repair.METADATA_COLUMNS]]
    received = []
    def original(actual_panel, actual_scores):
        assert actual_panel is panel
        received.append(actual_scores)
        return "original-validator-result"
    monkeypatch.setattr(repair, "_ORIGINAL_VALIDATE", original)
    assert repair.validate_collected_scores(panel, scores) == "original-validator-result"
    pd.testing.assert_frame_equal(received[0], scores.drop(columns=[f"{c}_score" for c in repair.METADATA_COLUMNS]))
    assert all(f"{c}_score" in scores for c in repair.METADATA_COLUMNS)


@pytest.mark.parametrize("fault", ["alias_mismatch", "alias_missing", "duplicate_column", "extra_suffix", "merge_indicator"])
def test_malformed_metadata_fails_before_delegation(monkeypatch, fault):
    scores = collected()
    if fault == "alias_mismatch": scores.loc[0, "cluster_id_score"] = "changed"
    elif fault == "alias_missing": scores = scores.drop(columns="outer_fold_score")
    elif fault == "extra_suffix": scores["outer_fold_score_score"] = "unexpected"
    elif fault == "merge_indicator": scores["_merge"] = "both"
    else: scores = pd.concat([scores, scores[["peptide"]]], axis=1)
    def forbidden(*_): raise AssertionError("original validator must not be reached")
    monkeypatch.setattr(repair, "_ORIGINAL_VALIDATE", forbidden)
    with pytest.raises(ValueError): repair.validate_collected_scores(pd.DataFrame(), scores)
