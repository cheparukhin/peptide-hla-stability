"""Contract tests for the ProteinMPNN inverse-folding feature.

These guard the things that make the number mean what the report says it
means: that we score rather than sample, that the peptide chain is the only
designed chain, that the HLA is decoded first, and that the backbone we hand
ProteinMPNN is the one in the mmCIF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "structures" / "ectodomain_pilot" / "ectodomain-20261004"
ARM_B = PILOT / "boltz2" / "seed_0" / "A0201_LLWNGPMAV_arm_B"
ARM_A = PILOT / "boltz2" / "seed_0" / "A0201_LLWNGPMAV_arm_A"

pytest.importorskip("torch")
pytest.importorskip("biotite")
mpnn = pytest.importorskip("pepstab.inverse_folding")

requires_pilot = pytest.mark.skipif(
    not ARM_B.exists(), reason="stage 4c pilot structures not present"
)
requires_weights = pytest.mark.skipif(
    not (mpnn.MPNN_DIR / "vanilla_model_weights" / mpnn.DEFAULT_CHECKPOINT).exists(),
    reason="ProteinMPNN checkpoint not cloned into external/",
)


def test_provenance_records_commit_and_checkpoint():
    prov = mpnn.proteinmpnn_provenance()
    assert prov["repo"].endswith("ProteinMPNN")
    assert prov["checkpoint"] == "v_48_020.pt"
    # "scoring" not "sampling" is the whole point of the feature.
    assert "forward" in prov["mode"] and "scoring" in prov["mode"]


@requires_pilot
def test_load_complex_identifies_chains_like_the_pose_checker():
    cx = mpnn.load_complex(ARM_B)
    meta = json.loads((ARM_B / "metadata.json").read_text())
    assert cx.peptide == meta["inputs"]["peptide"]
    assert len(cx.chains[cx.peptide_chain][0]) == 9
    # Arm B is HLA ectodomain (275) + beta2m (99) + peptide (9).
    assert sorted(len(cx.chains[c][0]) for c in cx.context_chains) == [99, 275]


@requires_pilot
def test_arm_a_has_groove_only_context():
    cx = mpnn.load_complex(ARM_A)
    assert [len(cx.chains[c][0]) for c in cx.context_chains] == [182]


@requires_pilot
def test_backbone_coordinates_match_the_cif_ca_trace():
    """chain_backbone must agree residue-for-residue with the pose checker."""
    load_prediction, ca_by_chain, _ = mpnn._pose_check_helpers()
    atoms = load_prediction(sorted(ARM_B.glob("*.cif"))[0])
    ca = ca_by_chain(atoms)
    for cid, (seq, coords) in ca.items():
        bb_seq, bb = mpnn.chain_backbone(atoms, cid)
        assert bb_seq == seq
        assert bb.shape == (len(seq), 4, 3)
        assert np.isfinite(bb).all()
        # index 1 of BACKBONE_ATOMS is CA
        assert np.allclose(bb[:, 1, :], coords)


@requires_pilot
def test_mpnn_dict_puts_the_peptide_chain_first_and_keeps_all_chains():
    cx = mpnn.load_complex(ARM_B)
    d = cx.mpnn_dict()
    assert d["num_of_chains"] == 3
    assert d["seq"].startswith(cx.peptide)
    assert len(d["seq"]) == 383
    for cid in cx.chains:
        assert d[f"seq_chain_{cid}"] == cx.chains[cid][0]


@requires_pilot
@requires_weights
def test_scoring_designs_only_the_peptide_and_decodes_hla_first():
    """The HLA/beta2m must be fixed context, decoded before every peptide position."""
    import sys

    import torch

    sys.path.insert(0, str(mpnn.MPNN_DIR))
    from protein_mpnn_utils import tied_featurize

    cx = mpnn.load_complex(ARM_B)
    proto = cx.mpnn_dict()
    chain_dict = {proto["name"]: ([cx.peptide_chain], cx.context_chains)}
    feats = tied_featurize([proto], torch.device("cpu"), chain_dict)
    S, mask, chain_M, chain_M_pos = feats[1], feats[2], feats[4], feats[10]
    design = (chain_M * chain_M_pos * mask)[0].numpy()
    assert design.sum() == 9
    pep_idx = np.where(design > 0)[0]
    decoded = "".join(mpnn.ALPHABET[i] for i in S[0, pep_idx].numpy())
    assert decoded == cx.peptide

    # ProteinMPNN's decoding order: argsort((chain_M + 1e-4) * |randn|).
    # Context positions carry chain_M = 0, so they sort first, always.
    randn = torch.randn(1, S.shape[1], generator=torch.Generator().manual_seed(1))
    order = torch.argsort((chain_M * chain_M_pos * mask + 1e-4) * torch.abs(randn))[0].numpy()
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    assert rank[pep_idx].min() >= (len(order) - 9)


@requires_pilot
@requires_weights
def test_score_is_reproducible_and_negative_log_likelihood_convention():
    cx = mpnn.load_complex(ARM_A)
    a = mpnn.score_complex(cx, n_orders=4, seed=7)["native"]
    b = mpnn.score_complex(cx, n_orders=4, seed=7)["native"]
    assert a["ll_total"] == pytest.approx(b["ll_total"], abs=1e-6)
    assert a["ll_total"] < 0 and a["ll_mean"] < 0
    assert a["mpnn_score"] == pytest.approx(-a["ll_mean"], abs=1e-9)
    assert a["ll_total"] == pytest.approx(sum(a["ll_pos"]), abs=1e-3)
    assert len(a["ll_pos"]) == 9


@requires_pilot
@requires_weights
def test_a_decoy_equal_to_the_native_sequence_is_dropped():
    cx = mpnn.load_complex(ARM_A)
    out = mpnn.score_complex(cx, sequences={"self": cx.peptide}, n_orders=2)
    assert list(out) == ["native"]


@requires_pilot
@requires_weights
def test_score_folder_row_is_keyed_and_finite():
    row = mpnn.score_folder(ARM_B, n_orders=2)
    for key in ("model", "allele", "peptide", "arm", "seed"):
        assert row[key] not in (None, "")
    assert row["status"] == "ok"
    assert np.isfinite(row["pep_ll_total"])
    assert all(np.isfinite(row[f"ll_pos_{i}"]) for i in range(1, 10))


def test_find_predictions_discovers_folders_by_metadata(tmp_path):
    (tmp_path / "x" / "y").mkdir(parents=True)
    (tmp_path / "x" / "y" / "metadata.json").write_text("{}")
    assert mpnn.find_predictions(tmp_path) == [tmp_path / "x" / "y"]
