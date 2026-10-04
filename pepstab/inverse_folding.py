"""Inverse-folding (ProteinMPNN) conditional scoring of the peptide chain.

Stage 5 stretch arm. The question ProteinMPNN answers is *"given this 3D
backbone, how probable is this amino-acid sequence?"*. We hold the HLA
ectodomain (and beta2m, where the construct has it) fixed as visible context
and score the **native** peptide sequence against the predicted backbone.

Predeclared feature definition (written before any number was computed)
--------------------------------------------------------------------
Let the complex have chains H (HLA), optionally M (beta2m) and P (peptide,
length 9). ProteinMPNN's autoregressive decoder is run in *scoring* mode
(`ProteinMPNN.forward`, teacher-forced on the true sequence) -- never in
`ProteinMPNN.sample` mode, which designs a new sequence and is not a feature.

The peptide chain is the only *masked* (designed) chain; H and M are *visible*,
so their sequences are supplied to the decoder.  ProteinMPNN's decoding order
is ``argsort((chain_M + 1e-4) * |randn|)``, and visible positions carry
``chain_M = 0``, so every HLA/beta2m position is decoded before any peptide
position: the HLA really is held fixed.  Within the peptide the order is a
uniform random permutation, so a single pass gives one autoregressive
factorisation of ``log P(peptide | backbone, HLA sequence)``.  We average over
``n_orders`` independent permutations drawn from a fixed seed.

Reported per structure:

``pep_ll_total``    mean over decoding orders of ``sum_i log p(s_i | ...)``
                    -- the peptide's total log-likelihood (nats; higher = more
                    compatible with the backbone).
``pep_ll_mean``     ``pep_ll_total / 9``.
``mpnn_score``      ``-pep_ll_mean``, ProteinMPNN's own "score" convention
                    (lower = better), so numbers are comparable to the
                    upstream tool.
``ll_pos_1..9``     per-position mean log-likelihood, P1..P9.
``order_sd``        s.d. across decoding orders of the per-order total; a pure
                    numerical-noise floor for the feature.

Decoys (circularity control): the same backbone is additionally scored against
the *other* pilot peptides, all 9-mers.  ``native_margin`` is
``pep_ll_total(native) - max over decoys``.  A backbone that the folder built
*from* the native sequence will prefer that sequence; the margin quantifies how
much.

Nothing here recomputes splits, touches ``structures/``, or writes into
``data/``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MPNN_DIR = Path(os.environ.get("PROTEINMPNN_DIR", REPO / "external" / "ProteinMPNN"))
# On Modal this module is copied to /root as a flat module, so REPO-relative
# paths do not resolve; both locations are overridable by environment.
SCRIPTS_DIR = Path(os.environ.get("PEPSTAB_SCRIPTS_DIR", REPO / "scripts"))
DEFAULT_CHECKPOINT = "v_48_020.pt"
BACKBONE_ATOMS = ("N", "CA", "C", "O")
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def proteinmpnn_provenance(checkpoint: str = DEFAULT_CHECKPOINT) -> dict:
    """Repo commit and checkpoint identity. A score without this is meaningless."""
    import hashlib
    import subprocess

    sha = "unknown"
    try:
        sha = subprocess.run(
            ["git", "-C", str(MPNN_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance is best-effort
        pass
    path = MPNN_DIR / "vanilla_model_weights" / checkpoint
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {
        "repo": "https://github.com/dauparas/ProteinMPNN",
        "commit": sha,
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest,
        "mode": "forward (teacher-forced scoring of the native sequence)",
    }


# --------------------------------------------------------------------------
# structure -> ProteinMPNN input dict
# --------------------------------------------------------------------------
def _pose_check_helpers():
    """Reuse the existing mmCIF parsing rather than writing a second one."""
    for candidate in (str(SCRIPTS_DIR), str(Path(__file__).resolve().parent)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from boltz_pose_check import ca_by_chain, find_subsequence, load_prediction  # noqa

    return load_prediction, ca_by_chain, find_subsequence


def chain_backbone(atoms, chain_id: str) -> tuple[str, np.ndarray]:
    """One-letter sequence and an (L, 4, 3) N/CA/C/O array for one chain.

    Residue order and the CA row order match ``boltz_pose_check.ca_by_chain``
    exactly: one entry per residue that has a CA and a standard residue name,
    first altloc/insertion-code copy wins.
    """
    import biotite.structure as struc

    _pose_check_helpers()  # puts scripts/ on sys.path
    from boltz_pose_check import THREE_TO_ONE

    chain = atoms[atoms.chain_id == chain_id]
    starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    seq: list[str] = []
    coords: list[np.ndarray] = []
    seen: set = set()
    cats = chain.get_annotation_categories()
    ins = chain.ins_code if "ins_code" in cats else np.array([""] * chain.array_length())
    for lo, hi in zip(starts[:-1], starts[1:]):
        res = chain[lo:hi]
        if not np.any(res.atom_name == "CA"):
            continue
        one = THREE_TO_ONE.get(str(res.res_name[0]).upper())
        if one is None:
            continue
        key = (int(res.res_id[0]), str(ins[lo]))
        if key in seen:
            continue
        seen.add(key)
        xyz = np.full((4, 3), np.nan)
        for j, name in enumerate(BACKBONE_ATOMS):
            hit = np.where(res.atom_name == name)[0]
            if len(hit):
                xyz[j] = res.coord[hit[0]]
        seq.append(one)
        coords.append(xyz)
    return "".join(seq), np.asarray(coords, dtype=float)


@dataclass
class Complex:
    """One scored prediction: its chains, which one is the peptide, provenance."""

    name: str
    model: str
    seed: int
    arm: str
    allele: str
    peptide: str
    complex_id: str
    peptide_chain: str
    chains: dict = field(repr=False, default_factory=dict)  # id -> (seq, coords)
    # Production metadata carries these; the pilot does not. Provenance only.
    # `pair_id` is deliberately NOT carried: it is positional into the raw CSV
    # and the project invariant forbids joining on it. Joins are on
    # (allele, peptide).
    split: str | None = None
    shard: int | None = None

    @property
    def context_chains(self) -> list[str]:
        return sorted(c for c in self.chains if c != self.peptide_chain)

    def mpnn_dict(self) -> dict:
        """The dict shape ``tied_featurize`` expects (same as ``parse_PDB``)."""
        out: dict = {"name": self.name}
        concat = ""
        n = 0
        for cid in [self.peptide_chain] + self.context_chains:
            seq, xyz = self.chains[cid]
            out[f"seq_chain_{cid}"] = seq
            out[f"coords_chain_{cid}"] = {
                f"{a}_chain_{cid}": xyz[:, j, :].tolist()
                for j, a in enumerate(BACKBONE_ATOMS)
            }
            concat += seq
            n += 1
        out["num_of_chains"] = n
        out["seq"] = concat
        return out


def load_complex(folder: Path) -> Complex:
    """Read one prediction folder (mmCIF + metadata.json) into a ``Complex``.

    The peptide chain is found by exact sequence match, exactly as
    ``scripts/ectodomain_pose_check.py`` does, not by hard-coding chain 'C'.
    """
    load_prediction, ca_by_chain, find_subsequence = _pose_check_helpers()
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    case = meta["inputs"]
    cifs = sorted(folder.glob("*.cif"))
    if len(cifs) != 1:
        raise ValueError(f"expected exactly one mmCIF in {folder}, found {len(cifs)}")

    atoms = load_prediction(cifs[0])
    ca = ca_by_chain(atoms)
    hit = find_subsequence(ca, case["peptide"])
    if hit is None:
        raise ValueError(f"peptide {case['peptide']} not found in {folder}")
    pep_chain, offset = hit
    if offset != 0 or len(ca[pep_chain][0]) != len(case["peptide"]):
        raise ValueError(f"peptide chain {pep_chain} is not peptide-only in {folder}")

    chains = {}
    for cid in ca:
        seq, xyz = chain_backbone(atoms, cid)
        if seq != ca[cid][0]:
            raise ValueError(f"backbone/CA sequence mismatch on chain {cid} in {folder}")
        if not np.isfinite(xyz).all():
            raise ValueError(f"missing backbone atoms on chain {cid} in {folder}")
        chains[str(cid)] = (seq, xyz)

    declared = {str(c["id"]): c["sequence"] for c in case["chains"]}
    if {k: v[0] for k, v in chains.items()} != declared:
        raise ValueError(f"chain sequences disagree with metadata in {folder}")

    # The pilot writes `seed` at the top level; production writes it under
    # `settings`. Both are real schemas this harness must read, so neither is
    # hard-coded. Found by the ::smoke invariant, not by inspection.
    seed = meta.get("seed", meta.get("settings", {}).get("seed"))
    if seed is None:
        raise ValueError(f"no seed in metadata.json for {folder}")

    return Complex(
        name=f"{meta['model']}_{case['complex_id']}_arm_{case['arm']}_seed{seed}",
        model=meta["model"],
        seed=int(seed),
        arm=case["arm"],
        allele=case["allele"],
        peptide=case["peptide"],
        complex_id=case["complex_id"],
        peptide_chain=str(pep_chain),
        chains=chains,
        split=case.get("split", meta.get("split")),
        shard=meta.get("shard", meta.get("settings", {}).get("shard")),
    )


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@lru_cache(maxsize=2)
def load_model(checkpoint: str = DEFAULT_CHECKPOINT, backbone_noise: float = 0.0):
    """Load ProteinMPNN on CPU. ``backbone_noise=0`` -> deterministic features."""
    import torch

    if str(MPNN_DIR) not in sys.path:
        sys.path.insert(0, str(MPNN_DIR))
    from protein_mpnn_utils import ProteinMPNN  # noqa

    path = MPNN_DIR / "vanilla_model_weights" / checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=backbone_noise,
        k_neighbors=ckpt["num_edges"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("num_edges"), ckpt.get("noise_level")


def score_complex(
    cx: Complex,
    sequences: dict[str, str] | None = None,
    n_orders: int = 32,
    batch_rows: int = 8,
    checkpoint: str = DEFAULT_CHECKPOINT,
    seed: int = 0,
) -> dict[str, dict]:
    """Conditional log-likelihood of each candidate peptide on this backbone.

    ``batch_rows`` caps *total* rows per forward pass (orders x candidates),
    not orders alone: a 383-residue arm-B complex at 40 rows per pass needs
    well over a gigabyte, which on a shared machine means swap, not speed.

    ``sequences`` maps a label -> a 9-mer to place on the peptide chain; the
    native sequence is always included under ``"native"``. Returns label ->
    {ll_total, ll_mean, mpnn_score, ll_pos, order_sd}.
    """
    import torch

    if str(MPNN_DIR) not in sys.path:
        sys.path.insert(0, str(MPNN_DIR))
    from protein_mpnn_utils import tied_featurize  # noqa

    model, _, _ = load_model(checkpoint)
    device = torch.device("cpu")

    cands = {"native": cx.peptide}
    for label, seq in (sequences or {}).items():
        # A decoy identical to the native sequence is the native case, not a
        # decoy: dropping it keeps native_margin an honest native-vs-other gap.
        if label != "native" and seq != cx.peptide:
            cands[label] = seq
    for label, seq in cands.items():
        if len(seq) != len(cx.peptide):
            raise ValueError(f"candidate {label} has length {len(seq)}, need {len(cx.peptide)}")

    proto = cx.mpnn_dict()
    chain_dict = {proto["name"]: ([cx.peptide_chain], cx.context_chains)}
    feats = tied_featurize([proto], device, chain_dict)
    X, S, mask, chain_M = feats[0], feats[1], feats[2], feats[4]
    chain_encoding_all, chain_M_pos, residue_idx = feats[5], feats[10], feats[12]

    design = (chain_M * chain_M_pos * mask)[0].cpu().numpy()
    pep_idx = np.where(design > 0)[0]
    if len(pep_idx) != len(cx.peptide):
        raise ValueError(f"{len(pep_idx)} designable positions, expected {len(cx.peptide)}")
    native_from_tensor = "".join(ALPHABET[i] for i in S[0, pep_idx].cpu().numpy())
    if native_from_tensor != cx.peptide:
        raise ValueError(
            f"designable positions hold {native_from_tensor}, not the peptide {cx.peptide}"
        )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    labels = list(cands)
    aa_to_i = {a: i for i, a in enumerate(ALPHABET)}
    per_order: dict[str, list[np.ndarray]] = {k: [] for k in labels}

    with torch.no_grad():
        done = 0
        per_pass = max(1, batch_rows // len(labels))
        while done < n_orders:
            k = min(per_pass, n_orders - done)
            B = k * len(labels)
            Xb = X.repeat(B, 1, 1, 1)
            maskb = mask.repeat(B, 1)
            chain_Mb = (chain_M * chain_M_pos).repeat(B, 1)
            ridx = residue_idx.repeat(B, 1)
            cenc = chain_encoding_all.repeat(B, 1)
            Sb = S.repeat(B, 1).clone()
            # Same k decoding orders reused across every candidate sequence, so
            # the comparison between candidates is paired.
            randn_k = torch.randn(k, X.shape[1], generator=gen)
            randn = randn_k.repeat(len(labels), 1)
            for li, label in enumerate(labels):
                rows = slice(li * k, (li + 1) * k)
                Sb[rows, torch.as_tensor(pep_idx)] = torch.tensor(
                    [aa_to_i[a] for a in cands[label]], dtype=Sb.dtype
                )
            log_probs = model(Xb, Sb, maskb, chain_Mb, ridx, cenc, randn)
            gathered = torch.gather(log_probs, 2, Sb.unsqueeze(-1)).squeeze(-1)
            for li, label in enumerate(labels):
                rows = slice(li * k, (li + 1) * k)
                per_order[label].append(
                    gathered[rows][:, torch.as_tensor(pep_idx)].cpu().numpy()
                )
            done += k

    out = {}
    for label in labels:
        arr = np.concatenate(per_order[label], axis=0)  # [n_orders, 9]
        totals = arr.sum(axis=1)
        out[label] = {
            "sequence": cands[label],
            "ll_total": float(totals.mean()),
            "ll_mean": float(totals.mean() / arr.shape[1]),
            "mpnn_score": float(-totals.mean() / arr.shape[1]),
            "ll_pos": arr.mean(axis=0).tolist(),
            "order_sd": float(totals.std(ddof=1)),
            "n_orders": int(arr.shape[0]),
        }
    return out


def score_folder(
    folder: Path,
    decoys: dict[str, str] | None = None,
    n_orders: int = 32,
    checkpoint: str = DEFAULT_CHECKPOINT,
    seed: int = 0,
    batch_rows: int = 8,
) -> dict:
    """Score one prediction folder; returns one tidy row."""
    cx = load_complex(folder)
    scored = score_complex(
        cx, sequences=decoys, n_orders=n_orders, checkpoint=checkpoint,
        seed=seed, batch_rows=batch_rows,
    )
    native = scored["native"]
    row = {
        "model": cx.model,
        "allele": cx.allele,
        "peptide": cx.peptide,
        "arm": cx.arm,
        "seed": cx.seed,
        "complex_id": cx.complex_id,
        "peptide_chain": cx.peptide_chain,
        "context_chains": "+".join(cx.context_chains),
        "n_context_residues": sum(len(cx.chains[c][0]) for c in cx.context_chains),
        "checkpoint": checkpoint,
        "split": cx.split,
        "shard": cx.shard,
        "n_orders": native["n_orders"],
        "pep_ll_total": native["ll_total"],
        "pep_ll_mean": native["ll_mean"],
        "mpnn_score": native["mpnn_score"],
        "order_sd": native["order_sd"],
        "status": "ok",
    }
    for i, v in enumerate(native["ll_pos"], start=1):
        row[f"ll_pos_{i}"] = float(v)
    others = {k: v for k, v in scored.items() if k != "native"}
    if others:
        best = max(others.values(), key=lambda d: d["ll_total"])
        row["decoy_best_ll_total"] = best["ll_total"]
        row["decoy_best_sequence"] = best["sequence"]
        row["decoy_mean_ll_total"] = float(np.mean([d["ll_total"] for d in others.values()]))
        row["native_margin"] = native["ll_total"] - best["ll_total"]
        row["native_rank"] = 1 + sum(
            1 for d in others.values() if d["ll_total"] > native["ll_total"]
        )
        row["n_decoys"] = len(others)
        for k, d in others.items():
            row[f"decoy_ll_{k}"] = d["ll_total"]
    return row


def find_predictions(root: Path) -> list[Path]:
    """Every cohort prediction folder under ``root`` (one ``metadata.json`` each).

    Delegates to ``pepstab.structural_features.discover_folds``, which skips the
    ``_smoke`` / ``_shards`` / ``_failed`` trees that production writes next to
    the cohort output. Sharing that one implementation with the stage 4c.5
    extractor is deliberate: ingesting a harness fold raises nothing and looks
    like a normal row, so two independent exclusion rules are two chances to
    get it silently wrong.
    """
    discover_folds = None
    for mod in ("pepstab.structural_features", "structural_features"):
        try:
            discover_folds = __import__(mod, fromlist=["discover_folds"]).discover_folds
            break
        except ImportError:
            continue
    if discover_folds is None:  # pilot-only fallback; the pilot has no such trees
        return sorted(p.parent for p in Path(root).rglob("metadata.json"))
    return sorted(discover_folds(Path(root)))
