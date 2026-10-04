"""Guards on the label-side probes and the ported coupling arm.

The probes in ``scripts/biology_probes.py`` produce the one quantity in this
project that bounds every other result -- an estimate of assay reproducibility,
which no other table supplies -- so the arithmetic behind it is pinned here
rather than trusted. The coupling tests guard the controls that make
``scripts/stage3e_coupling.py`` an experiment rather than a pair of runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pepstab.data import AA_INDEX, load_raw  # noqa: E402
from pepstab.features import residue_rows  # noqa: E402
from scripts.biology_probes import (  # noqa: E402
    allele_pair_concordance,
    groove_distance,
    noise_ceiling,
)

#: The four pairs ``reports/BIOLOGY_NOTES.md`` §4 on the ``stage3-esm`` branch
#: published, with the rho and shared-peptide count it reported. Reproducing
#: them here is what licenses re-using that section's conclusion: this port
#: swapped the branch's pasted-in BLOSUM62 for the repo's own scaled copy, and
#: these numbers are the check that the swap changed nothing.
BRANCH_PUBLISHED = [
    ("HLA-A*02:01", "HLA-A*02:16", 369, 0.9205),
    ("HLA-A*02:12", "HLA-A*02:19", 347, 0.9074),
    ("HLA-A*23:01", "HLA-A*24:02", 303, 0.9012),
    ("HLA-B*27:03", "HLA-B*27:05", 245, 0.9007),
]


@pytest.fixture(scope="module")
def concordance() -> pd.DataFrame:
    return allele_pair_concordance(load_raw())


def test_reproduces_the_branch_published_pairs(concordance):
    """Each published pair is present with the same n and rho to 4 decimals."""
    keyed = {(r.a, r.b): r for r in concordance.itertuples()}
    for a, b, n_shared, rho in BRANCH_PUBLISHED:
        assert (a, b) in keyed, f"{a}/{b} missing from the concordance table"
        row = keyed[(a, b)]
        assert row.n_shared == n_shared
        assert row.rho == pytest.approx(rho, abs=5e-5)
        assert row.hamming == 1


def test_groove_distance_is_invariant_to_the_matrix_scale():
    """The normalised kernel cancels BLOSUM_SCALE, so the repo's copy is usable.

    This is the property that let the branch's pasted matrix be replaced by
    ``residue_rows("blosum")``, which is the same matrix divided by 5.
    """
    df = load_raw().drop_duplicates("allele")
    a, b = df.hla_pseudoseq.iloc[0], df.hla_pseudoseq.iloc[7]
    scaled = residue_rows("blosum")
    # Exact in real arithmetic; the achievable tolerance is set by the matrix
    # being float32, so rescaling by 5 and summing 34 terms re-rounds at ~1e-7.
    # Tightening this below float32 epsilon would be testing the FPU.
    assert groove_distance(a, b, scaled) == pytest.approx(
        groove_distance(a, b, scaled * 5.0), rel=1e-6)
    assert groove_distance(a, a, scaled) == pytest.approx(0.0, abs=1e-6)


def test_blosum_is_the_repo_matrix_not_a_copy():
    """A sanity check that the matrix is symmetric and in AA_INDEX order."""
    b = residue_rows("blosum")
    assert b.shape == (len(AA_INDEX), len(AA_INDEX))
    assert np.allclose(b, b.T)
    # Identical residues score at least as well as any substitution for them.
    assert (np.diag(b)[:, None] >= b - 1e-6).all()


def test_near_identical_pair_count_is_reported_not_rounded(concordance):
    """There are twelve Hamming-1 pairs, not four.

    The branch's §4 quoted four and read as though that was all of them. The
    bound it drew survives -- reproducibility is at least the *maximum* observed
    concordance -- but the count has to be visible, so this pins it.
    """
    ones = concordance[concordance.hamming == 1]
    assert len(ones) == 12, f"expected 12 Hamming-1 pairs, found {len(ones)}"
    assert ones.rho.max() == pytest.approx(0.9205, abs=5e-5)
    # The eight the branch did not quote sit materially lower.
    assert ones.rho.min() < 0.70
    ceiling = noise_ceiling(concordance)
    assert ceiling["hamming_le_1"]["n_pairs"] == 13      # 12 at d=1, 1 at d=0
    assert ceiling["hamming_le_1"]["max_rho"] == pytest.approx(0.9205, abs=5e-5)


def test_the_hamming_zero_pair_is_the_censored_c67s_construct_pair(concordance):
    """The one pair identical across all 34 contacts is the C67S pair.

    It ought to be the tightest reproducibility bound available -- true
    similarity on the binding surface is 1 by construction -- and instead it is
    the loosest, because both constructs are among the most censored alleles in
    the dataset. That is why the ceiling is quoted off the Hamming-1 pairs and
    carries a censoring caveat rather than being quoted off this one.
    """
    zeros = concordance[concordance.hamming == 0]
    assert len(zeros) == 1
    row = zeros.iloc[0]
    assert "C67S" in row.a and "C67S" in row.b
    assert row.rho < 0.70
    assert row.floor_share_max > 0.70


# --- the coupling arm's controls --------------------------------------------


def test_coupling_arms_are_parameter_identical():
    """The ablation must match the attention arm exactly in parameter count.

    This is the control that makes a gap between the arms attributable to
    coupling rather than to capacity. If it ever fails, the experiment does not
    measure what its report says it measures.
    """
    torch = pytest.importorskip("torch")
    from pepstab.attn import AttnConfig, CrossAttentionHead

    sizes = {}
    for couple in (True, False):
        cfg = AttnConfig(seed=0, couple=couple)
        m = CrossAttentionHead(64, cfg)
        sizes[couple] = sum(p.numel() for p in m.parameters())
    assert sizes[True] == sizes[False], sizes
    assert sizes[True] == 94913, (
        f"parameter count moved to {sizes[True]}; the stage3-esm branch's "
        f"smoke test recorded 94,913 and reports quote it")
    del torch


def test_initialisation_is_seeded():
    """Two members at the same seed must be bit-identical.

    The branch called ``torch.manual_seed`` inside ``fit``, after the caller had
    constructed the module, so initialisation came from ambient RNG state. This
    is the regression guard for that fix.
    """
    torch = pytest.importorskip("torch")
    from pepstab.attn import AttnConfig, CrossAttentionHead

    a = CrossAttentionHead(64, AttnConfig(seed=0))
    b = CrossAttentionHead(64, AttnConfig(seed=0))
    c = CrossAttentionHead(64, AttnConfig(seed=1))
    assert all(torch.equal(x, y) for x, y in
               zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in
                   zip(a.state_dict().values(), c.state_dict().values()))


def test_l2_excludes_biases_and_layernorm():
    """Weight decay must reach weight matrices only, as pepstab.mlp's does."""
    pytest.importorskip("torch")
    from pepstab.attn import AttnConfig, CrossAttentionHead

    m = CrossAttentionHead(64, AttnConfig(seed=0, l2=0.5))
    groups = m.param_groups()
    decayed = {id(p) for g in groups if g["weight_decay"] for p in g["params"]}
    for name, p in m.named_parameters():
        should_decay = not (name.endswith("bias") or "norm" in name)
        assert (id(p) in decayed) == should_decay, name


def test_ablation_has_no_attention_map():
    """The mean-pool arm must refuse to produce one rather than fabricate it."""
    torch = pytest.importorskip("torch")
    from pepstab.attn import AttnConfig, CrossAttentionHead

    m = CrossAttentionHead(64, AttnConfig(seed=0, couple=False))
    with pytest.raises(ValueError):
        m.attention_map(torch.zeros(2, 9, 64), torch.zeros(2, 34, 64))


def test_attention_rows_sum_to_one():
    torch = pytest.importorskip("torch")
    from pepstab.attn import AttnConfig, CrossAttentionHead

    m = CrossAttentionHead(64, AttnConfig(seed=0, couple=True))
    amap = m.attention_map(torch.randn(3, 9, 64), torch.randn(3, 34, 64))
    assert amap.shape == (3, 9, 34)
    assert torch.allclose(amap.sum(-1), torch.ones(3, 9), atol=1e-5)


def test_bank_avoids_fanning_unique_blocks_out_to_rows():
    """The bank stores 75 groove matrices once, not once per row."""
    torch = pytest.importorskip("torch")
    from pepstab.attn import Bank

    blocks = np.random.default_rng(0).normal(size=(75, 34, 8)).astype(np.float32)
    index = np.random.default_rng(1).integers(0, 75, 5000)
    bank = Bank(blocks, index)
    assert bank.blocks.shape == (75, 34, 8)       # not (5000, 34, 8)
    assert len(bank) == 5000
    rows = torch.arange(10)
    got = bank.gather(rows).numpy()
    assert np.array_equal(got, blocks[index[:10]])
    mask = np.zeros(5000, dtype=bool)
    mask[[3, 9, 11]] = True
    sub = bank.subset(mask)
    assert len(sub) == 3
    assert sub.blocks.data_ptr() == bank.blocks.data_ptr()   # shared, not copied


def test_bank_rejects_an_out_of_range_index():
    pytest.importorskip("torch")
    from pepstab.attn import Bank

    with pytest.raises(ValueError):
        Bank(np.zeros((4, 3, 2), dtype=np.float32), np.array([0, 1, 9]))


def test_cross_features_match_the_matrix_elementwise():
    """The 306 columns are exactly BLOSUM62[peptide_i, contact_j]."""
    from scripts.stage3f_blosum_cross import cross_features

    df = load_raw().head(6)
    X = cross_features(df)
    assert X.shape == (6, 9 * 34)
    b62 = residue_rows("blosum")
    for r in range(len(df)):
        row = df.iloc[r]
        expect = b62[np.array([AA_INDEX[c] for c in row.peptide])[:, None],
                     np.array([AA_INDEX[c] for c in row.hla_pseudoseq])[None, :]]
        assert np.array_equal(X[r].reshape(9, 34), expect)


def test_cross_features_carry_no_information_the_baseline_lacks():
    """The block is a deterministic function of columns the baseline already has.

    This is the whole logic of the control: two rows agreeing on peptide and
    pseudosequence must get identical cross-features, so a gain cannot be new
    information arriving.
    """
    from scripts.stage3f_blosum_cross import cross_features

    df = load_raw()
    dup = df[df.duplicated(subset=["peptide", "hla_pseudoseq"], keep=False)]
    if dup.empty:
        pytest.skip("no two rows share a peptide and a pseudosequence")
    first = dup.groupby(["peptide", "hla_pseudoseq"]).head(2)
    pair = first.groupby(["peptide", "hla_pseudoseq"]).filter(lambda g: len(g) == 2)
    sample = pair.head(10)
    X = cross_features(sample)
    for (_, _), idx in sample.groupby(["peptide", "hla_pseudoseq"]).groups.items():
        pos = [sample.index.get_loc(i) for i in idx]
        if len(pos) > 1:
            assert np.array_equal(X[pos[0]], X[pos[1]])
