import json
import numpy as np
import pytest
from pepstab import inverse_folding as original
from pepstab import followup3_inverse_parser as repaired


def test_whole_native_chain_wins_over_earlier_hla_substring():
    chains = {"A":("VVYPAEITLTWVV", None), "B":("VVVV", None), "C":("YPAEITLTW", None)}
    assert repaired.unique_peptide_chain(chains, "YPAEITLTW") == ("C", 0)


@pytest.mark.parametrize("chains", [{"A":("VVYPAEITLTWVV",None)}, {"C":("YPAEITLTW",None), "D":("YPAEITLTW",None)}])
def test_missing_or_ambiguous_whole_native_chain_fails(chains):
    with pytest.raises(ValueError, match="expected one exact whole peptide chain"):
        repaired.unique_peptide_chain(chains, "YPAEITLTW")


def test_successful_original_complex_has_identical_sequences_coordinates_and_order(tmp_path, monkeypatch):
    from scripts.boltz_pose_check import find_subsequence
    sequences = {"A":"VVVVVVVVVVVV", "B":"VVVV", "C":"YPAEITLTW"}
    chains = {c:(s,np.arange(len(s)*12,dtype=float).reshape(len(s),4,3)) for c,s in sequences.items()}
    ca = {c:(s,x[:,1]) for c,(s,x) in chains.items()}
    meta = dict(model="boltz2", seed=0, inputs=dict(allele="HLA-B*56:01", peptide="YPAEITLTW", arm="B", complex_id="case", split="train",
                chains=[dict(id=c,sequence=s) for c,s in sequences.items()]))
    (tmp_path/"metadata.json").write_text(json.dumps(meta));(tmp_path/"model.cif").write_text("mock")
    monkeypatch.setattr(original,"_pose_check_helpers",lambda:(lambda path:None,lambda atoms:ca,find_subsequence))
    monkeypatch.setattr(original,"chain_backbone",lambda atoms,cid:chains[cid])
    before = original.load_complex(tmp_path)
    repaired.install_exact_selector()
    after = original.load_complex(tmp_path)
    assert before.peptide_chain == after.peptide_chain == "C"
    assert before.context_chains == after.context_chains == ["A","B"]
    assert before.name == after.name
    assert before.mpnn_dict() == after.mpnn_dict()


def test_spawn_worker_installs_selector_before_original_worker(monkeypatch):
    from scripts import followup3_inverse as cli
    events = []
    monkeypatch.setattr(repaired,"install_exact_selector",lambda:events.append("selector"))
    monkeypatch.setattr(cli,"_extract_one",lambda args:events.append("original_worker") or "record")
    assert repaired.amended_extract_one(("root",{},"out",False)) == "record"
    assert events == ["selector","original_worker"]
