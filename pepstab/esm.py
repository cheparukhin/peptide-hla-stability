"""Frozen ESM-2 representations for stage 3, cached once per unique sequence.

The dataset has 28,166 measurement rows but only **5,633 distinct peptides and
75 distinct HLA domain sequences**, so embedding per row would waste ~5x on the
peptide side and ~375x on the HLA side. Everything here is keyed by the
sequence string; rows are reconstructed by a gather at feature-build time, the
same trick :mod:`pepstab.features` uses for one-hot/BLOSUM.

Why ``fair-esm`` rather than ``transformers``: the only thing stage 3 needs is
``model(tokens, repr_layers=[...])``. ``fair-esm`` is a ~100 kB pure-python
package with no extra runtime dependencies beyond torch, against
``transformers`` + ``tokenizers`` + ``huggingface-hub`` (~50 MB of wheels) for
the same frozen forward pass. The checkpoints are the same weights.

Layout on disk, under ``features/esm/``::

    <checkpoint>/<kind>_L<layer>.npz     reps (n_unique, length, dim) float16
    <checkpoint>/<kind>.index.json       the sequence list + its content hash
    <checkpoint>/manifest.json           timings, peak RSS, device, versions

float16 halves the cache (the 650M peptide block is 260 MB per layer in float32)
and is reversible for our purposes: ESM-2 activations sit in roughly [-20, 20]
and float16 has ~3 decimal digits there, far below the seed-to-seed spread of
the heads trained on them. Everything is cast back to float32 on load.

Nothing in this module touches labels or splits, so building the cache cannot
leak anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "features" / "esm"

#: ESM-2 checkpoints stage 3 can use, with their layer count and embedding width.
#: ``mid`` is the layer compared against the final one -- the plan asks for a
#: middle layer, and for ESM-2 the most transferable representations are
#: generally not the last block.
CHECKPOINTS: dict[str, dict] = {
    "esm2_t12_35M_UR50D": {"n_layers": 12, "dim": 480, "mid": 6, "params_m": 35},
    "esm2_t30_150M_UR50D": {"n_layers": 30, "dim": 640, "mid": 15, "params_m": 150},
    "esm2_t33_650M_UR50D": {"n_layers": 33, "dim": 1280, "mid": 17, "params_m": 650},
}

DEFAULT_CHECKPOINT = "esm2_t12_35M_UR50D"

#: The 34 HLA class I contact positions, 1-based into the 182-residue
#: ``hla_seq`` column (the NetMHCpan pseudosequence positions).
#:
#: **Verified, not assumed** (``scripts/esm_features.py --verify-index`` and
#: ``tests/test_esm.py``): taking these columns out of ``hla_seq`` reproduces
#: the supplied ``hla_pseudoseq`` character-for-character for all 75 alleles.
#: Four of them (7, 84, 118, 159) carry a tyrosine that is invariant across all
#: 75 alleles, so column identity alone cannot pin them down -- seven columns
#: match equally well. They are resolved by the canonical NetMHCpan numbering,
#: and the ambiguity is harmless for a *sequence* reconstruction (every
#: candidate is Y) but not for an *embedding*, which is position-dependent.
CONTACT_POSITIONS_1BASED = (
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84,
    95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171,
)

#: Positions whose assignment is not determined by the pseudosequence alone.
AMBIGUOUS_CONTACT_POSITIONS = (7, 84, 118, 159)

KINDS = ("peptide", "hla")


def layer_ids(checkpoint: str) -> dict[str, int]:
    """The two layers extracted per checkpoint: a middle one and the last one."""
    spec = CHECKPOINTS[checkpoint]
    return {"mid": spec["mid"], "final": spec["n_layers"]}


def content_hash(seqs: list[str]) -> str:
    """A stable digest of a sequence set, so a stale cache cannot be reused."""
    h = hashlib.sha256()
    for s in seqs:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def verify_contact_index(hla_seqs: list[str], pseudoseqs: list[str]) -> dict:
    """Check that :data:`CONTACT_POSITIONS_1BASED` reproduces the pseudosequences.

    Returns a report rather than only a bool, because the interesting part is
    *which* positions the data can pin down on its own.
    """
    idx = np.asarray(CONTACT_POSITIONS_1BASED) - 1
    S = np.array([list(s) for s in hla_seqs])
    P = np.array([list(p) for p in pseudoseqs])
    if S.shape[0] != P.shape[0]:
        raise ValueError("hla_seqs and pseudoseqs must be the same length")
    recon = S[:, idx]
    exact = bool(np.array_equal(recon, P))
    ambiguous = []
    for k in range(P.shape[1]):
        cols = [j + 1 for j in range(S.shape[1]) if np.all(S[:, j] == P[:, k])]
        if len(cols) > 1:
            ambiguous.append({"pseudo_position": k + 1,
                              "assigned": int(CONTACT_POSITIONS_1BASED[k]),
                              "residue": str(P[0, k]),
                              "indistinguishable_columns": cols})
    return {
        "n_alleles": int(S.shape[0]),
        "exact_match": exact,
        "n_mismatched_alleles": int((recon != P).any(axis=1).sum()),
        "n_ambiguous_positions": len(ambiguous),
        "ambiguous": ambiguous,
    }


# --- extraction -------------------------------------------------------------


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB (macOS reports bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if platform.system() == "Darwin" else peak / 1024


@dataclass
class ExtractionResult:
    checkpoint: str
    kind: str
    n_sequences: int
    seq_length: int
    dim: int
    device: str
    load_seconds: float
    embed_seconds: float
    peak_rss_mb: float
    cache_mb: float
    content_hash: str


def pick_device(requested: str = "auto") -> str:
    """Pick a torch device. ``auto`` prefers MPS, which is what this Mac has."""
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract(seqs: list[str], checkpoint: str = DEFAULT_CHECKPOINT,
            device: str = "auto", batch_tokens: int = 8192,
            progress=None) -> tuple[dict[int, np.ndarray], dict]:
    """Embed each sequence once and return ``{layer: (n, length, dim) float16}``.

    All sequences must be the same length -- true for both of our kinds (9-mer
    peptides, 182-residue HLA domains) -- which keeps batching trivial and means
    no padding mask is needed. BOS/EOS token positions are stripped, so index
    ``j`` of the output is residue ``j`` of the input.
    """
    import torch
    import esm as fair_esm

    if not seqs:
        raise ValueError("nothing to embed")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"expected one sequence length, got {sorted(lengths)}")
    length = lengths.pop()

    t0 = time.perf_counter()
    model, alphabet = fair_esm.pretrained.load_model_and_alphabet(checkpoint)
    model.eval()
    dev = pick_device(device)
    model = model.to(dev)
    load_seconds = time.perf_counter() - t0

    layers = sorted(set(layer_ids(checkpoint).values()))
    converter = alphabet.get_batch_converter()
    # +2 for BOS/EOS; cap batches by token count so one setting works for both
    # a 9-mer and a 182-residue domain.
    per_batch = max(1, batch_tokens // (length + 2))

    out = {L: np.empty((len(seqs), length, CHECKPOINTS[checkpoint]["dim"]),
                       dtype=np.float16) for L in layers}
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(seqs), per_batch):
            chunk = seqs[start:start + per_batch]
            _, _, tokens = converter([(str(i), s) for i, s in enumerate(chunk)])
            res = model(tokens.to(dev), repr_layers=layers, return_contacts=False)
            for L in layers:
                # strip BOS (0) and EOS (length + 1)
                r = res["representations"][L][:, 1:length + 1, :]
                out[L][start:start + len(chunk)] = r.to("cpu").float().numpy().astype(np.float16)
            if progress is not None:
                progress(min(start + per_batch, len(seqs)), len(seqs))
    embed_seconds = time.perf_counter() - t0

    del model
    if dev == "mps":
        torch.mps.empty_cache()

    meta = {
        "device": dev,
        "load_seconds": load_seconds,
        "embed_seconds": embed_seconds,
        "batch_size": per_batch,
        "peak_rss_mb": _peak_rss_mb(),
        "torch_version": torch.__version__,
        "fair_esm_version": getattr(fair_esm, "__version__", "2.0.0"),
    }
    return out, meta


# --- cache ------------------------------------------------------------------


def cache_dir(checkpoint: str) -> Path:
    return CACHE_DIR / checkpoint


def _index_path(checkpoint: str, kind: str) -> Path:
    return cache_dir(checkpoint) / f"{kind}.index.json"


def _reps_path(checkpoint: str, kind: str, layer: int) -> Path:
    return cache_dir(checkpoint) / f"{kind}_L{layer}.npz"


def is_cached(seqs: list[str], checkpoint: str, kind: str) -> bool:
    """True if a cache exists for exactly this sequence set."""
    p = _index_path(checkpoint, kind)
    if not p.exists():
        return False
    index = json.loads(p.read_text())
    if index["content_hash"] != content_hash(seqs):
        return False
    return all(_reps_path(checkpoint, kind, L).exists()
               for L in layer_ids(checkpoint).values())


def build_cache(seqs: list[str], checkpoint: str = DEFAULT_CHECKPOINT,
                kind: str = "peptide", device: str = "auto",
                force: bool = False, progress=None) -> ExtractionResult:
    """Embed ``seqs`` (deduplicated, sorted by the caller) and write the cache."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    digest = content_hash(seqs)
    dest = cache_dir(checkpoint)
    dest.mkdir(parents=True, exist_ok=True)

    if not force and is_cached(seqs, checkpoint, kind):
        index = json.loads(_index_path(checkpoint, kind).read_text())
        return ExtractionResult(
            checkpoint=checkpoint, kind=kind, n_sequences=len(seqs),
            seq_length=len(seqs[0]), dim=CHECKPOINTS[checkpoint]["dim"],
            device="cached", load_seconds=0.0, embed_seconds=0.0,
            peak_rss_mb=_peak_rss_mb(),
            cache_mb=sum(_reps_path(checkpoint, kind, L).stat().st_size
                         for L in layer_ids(checkpoint).values()) / 1e6,
            content_hash=index["content_hash"])

    reps, meta = extract(seqs, checkpoint=checkpoint, device=device,
                         progress=progress)
    size = 0
    for L, arr in reps.items():
        path = _reps_path(checkpoint, kind, L)
        np.savez(path, reps=arr)
        size += path.stat().st_size
    _index_path(checkpoint, kind).write_text(json.dumps({
        "checkpoint": checkpoint, "kind": kind, "content_hash": digest,
        "n_sequences": len(seqs), "seq_length": len(seqs[0]),
        "dim": CHECKPOINTS[checkpoint]["dim"],
        "layers": layer_ids(checkpoint), "sequences": seqs, **meta,
    }, indent=1))
    return ExtractionResult(
        checkpoint=checkpoint, kind=kind, n_sequences=len(seqs),
        seq_length=len(seqs[0]), dim=CHECKPOINTS[checkpoint]["dim"],
        device=meta["device"], load_seconds=meta["load_seconds"],
        embed_seconds=meta["embed_seconds"], peak_rss_mb=meta["peak_rss_mb"],
        cache_mb=size / 1e6, content_hash=digest)


def load_cache(checkpoint: str, kind: str, layer: str | int
               ) -> tuple[dict[str, int], np.ndarray]:
    """Return ``({sequence: row}, reps (n, length, dim) float32)``."""
    L = layer_ids(checkpoint)[layer] if isinstance(layer, str) else int(layer)
    p = _index_path(checkpoint, kind)
    if not p.exists():
        raise FileNotFoundError(
            f"no ESM cache at {p}. Build it with "
            f"`.venv/bin/python scripts/esm_features.py --checkpoint {checkpoint}`.")
    index = json.loads(p.read_text())
    reps = np.load(_reps_path(checkpoint, kind, L))["reps"].astype(np.float32)
    return {s: i for i, s in enumerate(index["sequences"])}, reps


# --- row features -----------------------------------------------------------

#: Peptide representations compared on validation.
PEPTIDE_REPS = ("pos", "mean", "none")
#: HLA representations compared on validation.
HLA_REPS = ("contact", "mean", "none")


def _gather(values, lookup: dict[str, int], reps: np.ndarray) -> np.ndarray:
    """Fan a per-unique-sequence array out to one row per measurement."""
    missing = [v for v in set(map(str, values)) if v not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} sequence(s) absent from the cache, "
                       f"e.g. {missing[0][:20]!r}; rebuild it")
    idx = np.fromiter((lookup[str(v)] for v in values), dtype=np.int64,
                      count=len(values))
    return reps[idx]


def peptide_block(peptides, checkpoint: str, layer: str, rep: str) -> np.ndarray:
    """``(n, 9 * dim)`` per-position, ``(n, dim)`` mean-pooled, or empty."""
    if rep == "none":
        return np.zeros((len(peptides), 0), dtype=np.float32)
    lookup, reps = load_cache(checkpoint, "peptide", layer)
    if rep == "mean":
        reps = reps.mean(axis=1)
    elif rep != "pos":
        raise ValueError(f"peptide rep must be one of {PEPTIDE_REPS}")
    rows = _gather(peptides, lookup, reps)
    return rows.reshape(len(rows), -1)


def hla_block(hla_seqs, checkpoint: str, layer: str, rep: str) -> np.ndarray:
    """``(n, 34 * dim)`` contact positions, ``(n, dim)`` mean-pooled, or empty."""
    if rep == "none":
        return np.zeros((len(hla_seqs), 0), dtype=np.float32)
    lookup, reps = load_cache(checkpoint, "hla", layer)
    if rep == "mean":
        reps = reps.mean(axis=1)
    elif rep == "contact":
        reps = reps[:, np.asarray(CONTACT_POSITIONS_1BASED) - 1, :]
    else:
        raise ValueError(f"hla rep must be one of {HLA_REPS}")
    rows = _gather(hla_seqs, lookup, reps)
    return rows.reshape(len(rows), -1)
