import numpy as np
import pandas as pd
import pytest

from pepstab import evaluation as ev
from pepstab.round2_reporting import cluster_bootstrap_scores
from pepstab.round2_transfer import _frequency_median, transfer_bootstrap


def sample():
    rows = []
    # Unequal numbers of alleles per HLA group expose mistaken group averaging.
    for allele,group in (("A",0),("B",0),("C",1),("D",2)):
        for cluster in range(5):
            rows.append({"pair_id":len(rows),"allele":allele,"hla_group":group,
                "cluster_id":cluster,"split":"train","y_log1p":float(cluster//2),
                "a":float(cluster if allele in ("A","C") else 4-cluster),
                "b":float(cluster%3 if allele != "D" else 1)})
    return pd.DataFrame(rows)


def test_integer_median_matches_expansion_including_even_and_nan():
    for x,w in (([0.,1.,2.],[2,1,1]),([np.nan,1.,4.],[8,3,1]),([2.,3.],[0,0])):
        expanded = np.repeat(np.asarray(x)[np.isfinite(x)],np.asarray(w)[np.isfinite(x)])
        expected = np.median(expanded) if len(expanded) else np.nan
        assert _frequency_median(np.asarray(x),np.asarray(w)) == pytest.approx(expected,nan_ok=True)


@pytest.mark.parametrize("mode",["peptide","hla","crossed"])
def test_bootstrap_matches_physically_replicated_group_clones(mode):
    frame = sample()
    pred = {k:frame[k].to_numpy() for k in ("a","b")}
    actual,meta = transfer_bootstrap(frame,pred,min_rows=2,n_boot=30,seed=123,mode=mode)
    rng = np.random.default_rng(123)
    expected = []
    clusters,groups = np.unique(frame.cluster_id),np.unique(frame.hla_group)
    for draw in range(30):
        picked_p = rng.choice(len(clusters),len(clusters),replace=True) if mode in ("peptide","crossed") else np.arange(len(clusters))
        picked_h = rng.choice(len(groups),len(groups),replace=True) if mode in ("hla","crossed") else np.arange(len(groups))
        parts = []
        for clone,h in enumerate(picked_h):
            for p in picked_p:
                rows = frame[frame.hla_group.eq(groups[h]) & frame.cluster_id.eq(clusters[p])].copy()
                rows["allele"] = rows.allele + f"__clone{clone}"
                parts.append(rows)
        expanded = pd.concat(parts,ignore_index=True)
        panel = ev.eligible_alleles(expanded.allele,expanded.y_log1p,min_rows=2)
        expected.append({k:ev.score(k,expanded.allele,expanded.y_log1p,expanded[k],panel).median_spearman if panel else np.nan for k in pred})
    np.testing.assert_allclose(actual,pd.DataFrame(expected),atol=1e-14,equal_nan=True)
    assert meta["n_hla_groups"] == 3


def test_peptide_mode_matches_existing_reference_exactly():
    frame = sample()
    pred = {k:frame[k].to_numpy() for k in ("a","b")}
    actual,_ = transfer_bootstrap(frame,pred,min_rows=2,n_boot=50,seed=92,mode="peptide")
    expected,_ = cluster_bootstrap_scores(frame,pred,min_rows=2,n_boot=50,seed=92)
    np.testing.assert_allclose(actual,expected,atol=1e-14,equal_nan=True)


def test_panels_separate_fixed_eligibility_and_input_rejection():
    frame = sample()
    pred = {"a":frame.a.to_numpy()}
    with pytest.raises(ValueError,match="no eligible"):
        transfer_bootstrap(frame,pred,min_rows=20,n_boot=4)
    with pytest.raises(ValueError,match="shape"):
        transfer_bootstrap(frame,{"a":np.zeros(3)},min_rows=2,n_boot=4)
    bad = frame.copy()
    bad.loc[0,"hla_group"] = 10
    with pytest.raises(ValueError,match="multiple"):
        transfer_bootstrap(bad,pred,min_rows=2,n_boot=4)
