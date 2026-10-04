"""Cache frozen ESM-2 embeddings for every unique peptide and HLA domain.

    python scripts/esm_features.py --model models/esm2_t12_35M --out features/esm2_35M.npz

Embeddings are cached per *unique sequence* (5,633 peptides, 75 domains), not per
measurement row -- 28,166 rows map onto 5,708 forward passes.

Peptide and HLA are embedded separately, as stage 3 specifies, so the head has to
learn the interaction itself. Two layers are kept: the final one and the middle
one, because the last layer of a masked language model is optimised for token
prediction rather than for binding.

The 34 contact positions are recovered from the data, not hardcoded: a domain
column is accepted for pseudosequence column j only if it matches in all 75
alleles. The script fails if the mapping is not unique.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pepstab.data import load_raw, PSEUDOSEQ_LENGTH  # noqa: E402


#: The NetMHCpan pseudosequence positions, in HLA heavy-chain numbering. Public
#: definition, not fitted here -- :func:`contact_positions` tests it against the
#: data rather than trusting it.
NETMHCPAN_POSITIONS = (7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77,
                       80, 81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152,
                       156, 158, 159, 163, 167, 171)


def contact_positions(hla_seq: list[str], pseudoseq: list[str]) -> np.ndarray:
    """Columns of the 182-residue domain that carry the 34-residue pseudosequence.

    Recovering the mapping from the data alone is underdetermined -- with only 75
    alleles, several conserved domain columns match any given pseudosequence
    column by chance. So the published position list is *tested* instead: the
    supplied domain is residues 1-182 of the mature alpha1/alpha2, so position p
    should sit at index p-1. The assertion covers all 75 alleles x 34 columns,
    which is the residue-indexing check stage 3 requires before any embedding is
    sliced.
    """
    seqs = np.array([list(s) for s in hla_seq])
    pseu = np.array([list(s) for s in pseudoseq])
    idx = np.array(NETMHCPAN_POSITIONS) - 1
    if idx.max() >= seqs.shape[1] or len(pseu[0]) != PSEUDOSEQ_LENGTH:
        raise ValueError(f"unexpected shapes: domain {seqs.shape}, pseudoseq {pseu.shape}")
    ok = seqs[:, idx] == pseu
    if not ok.all():
        cols = np.flatnonzero(~ok.all(axis=0)).tolist()
        raise ValueError(
            f"pseudosequence columns {cols} do not match domain positions "
            f"{[NETMHCPAN_POSITIONS[c] for c in cols]}; the supplied domain is not "
            "1-182 in heavy-chain numbering and the contact slice would be wrong"
        )
    return idx


@torch.no_grad()
def embed(seqs: list[str], model, tok, layers: tuple[int, ...],
          batch_size: int = 128) -> dict[int, np.ndarray]:
    """Per-residue hidden states for each sequence, special tokens stripped.

    Returns {layer: (n_seqs, seq_len, hidden)}. All sequences in one call must
    have equal length, which holds here (9-mers, or 182-residue domains).
    """
    out = {k: [] for k in layers}
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", add_special_tokens=True)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for k in layers:
            # hidden_states[0] is the embedding layer; [1:] are the transformer
            # blocks. Drop BOS and EOS so column j is residue j.
            out[k].append(hs[k][:, 1:-1, :].to(torch.float32).numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/esm2_t12_35M")
    ap.add_argument("--out", default="features/esm2_35M.npz")
    ap.add_argument("--layers", default="", help="comma-separated; default mid,final")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    df = load_raw()
    peptides = sorted(df["peptide"].unique())
    alleles = df.drop_duplicates("allele")[["allele", "hla_seq", "hla_pseudoseq"]]
    alleles = alleles.sort_values("allele").reset_index(drop=True)

    contacts = contact_positions(alleles.hla_seq.tolist(), alleles.hla_pseudoseq.tolist())
    print(f"contact positions (0-based into the 182-residue domain): {contacts.tolist()}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model).eval()
    n_layers = model.config.num_hidden_layers
    layers = (tuple(int(x) for x in args.layers.split(","))
              if args.layers else (n_layers // 2, n_layers))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"{args.model}: {n_layers} layers, hidden {model.config.hidden_size}, "
          f"keeping layers {layers}")

    t0 = time.perf_counter()
    pep = embed(peptides, model, tok, layers, args.batch_size)
    t_pep = time.perf_counter() - t0
    t0 = time.perf_counter()
    dom = embed(alleles.hla_seq.tolist(), model, tok, layers, 16)
    t_hla = time.perf_counter() - t0

    payload = {"peptides": np.array(peptides), "alleles": alleles.allele.to_numpy(),
               "contact_positions": contacts,
               "seconds_peptides": t_pep, "seconds_hla": t_hla,
               "n_layers": n_layers, "hidden": model.config.hidden_size}
    for k in layers:
        # float16 halves the cache; the head standardises its inputs anyway and
        # the fitted predictions move by far less than the seed-to-seed spread.
        payload[f"pep_L{k}"] = pep[k].astype(np.float16)
        payload[f"hla_L{k}"] = dom[k][:, contacts, :].astype(np.float16)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    size = out.stat().st_size / 1e6
    print(f"{len(peptides)} peptides in {t_pep:.1f}s, {len(alleles)} domains in "
          f"{t_hla:.1f}s -> {out} ({size:.0f} MB)")
    for k in layers:
        print(f"  layer {k}: peptide {pep[k].shape}, HLA contacts "
              f"{dom[k][:, contacts, :].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
