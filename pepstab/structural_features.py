"""Stage 4c.5: turn a Boltz-2 (or ESMFold2) fold into a stage-5 feature row.

Implements the "Feature contract for stage 5" section of
``reports/stage4c_ectodomain_pilot.md``. Three rules from that contract drive
the whole design:

1. **Nothing about the token layout is hard-coded.** The anticipated arm-B
   shape is 383 x 383 with HLA ``0:275``, beta2m ``275:374``, peptide
   ``374:383``. That is an expectation, not a fact, so :func:`verify_fold`
   re-derives every boundary from the mmCIF itself and raises
   :class:`MappingError` on any disagreement. A silently-wrong slice would
   produce plausible numbers for the wrong molecule.
2. **Global ipTM stays global.** In a three-chain complex it covers the
   HLA/beta2m interface as well, so it is named ``global_iptm`` and never
   presented as peptide-interface confidence. Boltz 2.1.1 *does* emit a
   pair-specific score (``pair_chains_iptm``); it is exported only under names
   that state both the chain pair and its direction, and only for models whose
   chain-index mapping has been verified (:data:`PAIR_IPTM_VERIFIED_MODELS`).
3. **Confidence is a feature, never a row filter.** Nothing here drops a row on
   a confidence threshold. Rows fail only on a structural/parsing problem, and
   those failures are recorded rather than discarded.

Verified mapping (evidence in ``reports/stage4c5_features.md``):

* Token index ``i`` of ``plddt``/``pae`` is the ``i``-th CA atom in mmCIF
  atom-site order. Checked per fold: ``plddt * 100`` equals the CA B-factor
  column to within :data:`BFACTOR_TOLERANCE`.
* mmCIF chain blocks appear in input-chain order, so chain boundaries in token
  space are the cumulative chain lengths.
* ``pair_chains_iptm`` keys are ``asym_id`` values, assigned by
  ``enumerate`` over the input chains, i.e. the same order. Entry ``[a][b]``
  aggregates PAE **rows in chain b, columns in chain a** (boltz 2.1.1
  ``confidence_utils.compute_ptms``), which is the transpose of the naive
  reading.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# --- Definitions fixed by the contract -------------------------------------

#: HLA residues 1-182 (the alpha1/alpha2 groove). Positional: the first 182
#: residues of the HLA chain, matching ``scripts/ectodomain_pose_check.py``.
GROOVE_RESIDUES = 182

#: Heavy-atom contact cutoff, Angstrom. Same value as the pose checker.
CONTACT_CUTOFF_A = 4.5

#: Shrake-Rupley sampling density for burial. Fixed so burial is reproducible.
SASA_POINT_NUMBER = 200
SASA_PROBE_RADIUS = 1.4
SASA_VDW_RADII = "ProtOr"

#: ``plddt * 100`` must match the CA B-factor column to this tolerance.
BFACTOR_TOLERANCE = 0.01

#: Models whose ``pair_chains_iptm`` chain-index mapping has been verified
#: against source and output. ESMFold2 emits a differently-shaped object and
#: is deliberately excluded rather than guessed at.
PAIR_IPTM_VERIFIED_MODELS = frozenset({"boltz2"})

#: Production sibling trees that are *not* cohort output. Ingesting these would
#: not error and would not look wrong -- it would quietly add harness folds to
#: the feature table.
EXCLUDED_DIR_NAMES = frozenset({"_smoke", "_shards", "_failed"})

ELEMENT_HYDROGEN = ("H", "D")


class MappingError(ValueError):
    """The fold on disk disagrees with the verified chain/token mapping."""


# --- Parsing ----------------------------------------------------------------


@dataclass(frozen=True)
class ChainBlock:
    """One chain, in both residue space and token space."""

    chain_id: str
    length: int
    start: int  # token index of the first residue
    stop: int  # token index one past the last residue

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass
class Fold:
    """A parsed, verified fold."""

    folder: Path
    metadata: dict[str, Any]
    atoms: Any  # biotite AtomArray, heavy atoms, model 1, non-hetero
    pae: np.ndarray
    plddt: np.ndarray
    confidence: dict[str, Any]
    chains: list[ChainBlock]
    hla: ChainBlock
    peptide: ChainBlock
    beta2m: ChainBlock | None
    groove: slice  # token slice of HLA residues 1-182
    alpha3: slice | None  # token slice of HLA residues 183..end, arm B only
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return int(self.plddt.shape[0])


def _read_array(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    with np.load(path) as handle:
        return handle[key]


def _one_file(folder: Path, prefix: str) -> Path:
    hits = sorted(p for p in folder.iterdir() if p.name.startswith(prefix))
    if len(hits) != 1:
        raise MappingError(f"expected exactly one {prefix}* file in {folder}, found {len(hits)}")
    return hits[0]


def _chain_runs(chain_ids: Sequence[str]) -> list[tuple[str, int]]:
    """Consecutive runs of chain id, in file order.

    Runs rather than ``set``: a chain that is split into two non-adjacent
    blocks would break token slicing, and this surfaces it.
    """
    runs: list[tuple[str, int]] = []
    for cid in chain_ids:
        cid = str(cid)
        if runs and runs[-1][0] == cid:
            runs[-1] = (cid, runs[-1][1] + 1)
        else:
            runs.append((cid, 1))
    return runs


def load_fold(folder: Path) -> Fold:
    """Parse and verify one prediction directory.

    Raises :class:`MappingError` if anything about the layout differs from the
    verified mapping. Reuses the mmCIF reading convention of
    ``scripts/boltz_pose_check.py``: model 1, hetero atoms dropped.
    """
    import biotite.structure as struc  # noqa: F401  (import cost stays local)
    from biotite.structure.io.pdbx import CIFFile, get_structure

    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())

    cifs = sorted(folder.glob("*.cif"))
    if len(cifs) != 1:
        raise MappingError(f"expected exactly one mmCIF in {folder}, found {len(cifs)}")
    atoms = get_structure(CIFFile.read(cifs[0]), model=1, extra_fields=["b_factor"])
    if "hetero" in atoms.get_annotation_categories():
        atoms = atoms[~atoms.hetero]
    atoms = atoms[~np.isin(atoms.element, ELEMENT_HYDROGEN)]

    pae = np.asarray(_read_array(_one_file(folder, "pae"), "pae"), dtype=float)
    plddt = np.asarray(_read_array(_one_file(folder, "plddt"), "plddt"), dtype=float)
    conf_files = sorted(p for p in folder.iterdir() if p.name.startswith("confidence"))
    confidence = json.loads(conf_files[0].read_text()) if conf_files else {}

    return verify_fold(folder, metadata, atoms, pae, plddt, confidence)


def verify_fold(folder, metadata, atoms, pae, plddt, confidence) -> Fold:
    """Re-derive every boundary from the files and fail loudly on mismatch."""
    checks: dict[str, Any] = {}

    ca = atoms[atoms.atom_name == "CA"]
    runs = _chain_runs(ca.chain_id)
    checks["cif_chain_runs"] = runs

    declared = [(str(c["id"]), len(c["sequence"])) for c in metadata["inputs"]["chains"]]
    checks["metadata_chains"] = declared
    if runs != declared:
        raise MappingError(
            f"{folder}: mmCIF chain blocks {runs} do not match the declared input "
            f"chains {declared}; token slicing cannot be trusted"
        )

    n = sum(length for _, length in runs)
    checks["n_tokens"] = n
    checks["pae_shape"] = list(pae.shape)
    checks["plddt_shape"] = list(plddt.shape)
    if pae.shape != (n, n) or plddt.shape != (n,):
        raise MappingError(
            f"{folder}: residue count {n} but pae {pae.shape} / plddt {plddt.shape}; "
            "the anticipated shape is not a fact, and this fold disagrees with it"
        )
    if not (np.isfinite(pae).all() and np.isfinite(plddt).all()):
        raise MappingError(f"{folder}: non-finite values in pae or plddt")

    # Token index i == i-th CA in atom-site order. The writer puts pLDDT in the
    # B-factor column, so this is a direct, per-fold proof of the mapping.
    bdiff = float(np.abs(ca.b_factor - plddt * 100.0).max())
    checks["max_abs_bfactor_minus_plddt_x100"] = bdiff
    if not bdiff <= BFACTOR_TOLERANCE:
        raise MappingError(
            f"{folder}: pLDDT token order does not match mmCIF CA order "
            f"(max |b_factor - plddt*100| = {bdiff:.4f} > {BFACTOR_TOLERANCE})"
        )

    bounds = np.cumsum([0] + [length for _, length in runs])
    blocks = [
        ChainBlock(cid, length, int(bounds[i]), int(bounds[i + 1]))
        for i, (cid, length) in enumerate(runs)
    ]

    peptide_seq = metadata["inputs"]["peptide"]
    pep_candidates = [
        b for b, (_, length) in zip(blocks, runs) if length == len(peptide_seq)
    ]
    if len(pep_candidates) != 1:
        raise MappingError(
            f"{folder}: {len(pep_candidates)} chains have the peptide length "
            f"{len(peptide_seq)}; the peptide chain is ambiguous"
        )
    peptide = pep_candidates[0]
    # The pose checker's convention: the peptide is the last chain block.
    if peptide is not blocks[-1]:
        raise MappingError(f"{folder}: peptide chain {peptide.chain_id} is not the last block")
    if _sequence(ca, peptide) != peptide_seq:
        raise MappingError(
            f"{folder}: chain {peptide.chain_id} is {_sequence(ca, peptide)!r}, "
            f"not the declared peptide {peptide_seq!r}"
        )

    hla = blocks[0]
    hla_declared = metadata["inputs"]["chains"][0]["sequence"]
    if _sequence(ca, hla) != hla_declared:
        raise MappingError(f"{folder}: first chain is not the declared HLA sequence")
    if hla.length < GROOVE_RESIDUES:
        raise MappingError(
            f"{folder}: HLA chain has {hla.length} residues, fewer than the "
            f"{GROOVE_RESIDUES}-residue groove"
        )
    groove = slice(hla.start, hla.start + GROOVE_RESIDUES)
    alpha3 = slice(hla.start + GROOVE_RESIDUES, hla.stop) if hla.length > GROOVE_RESIDUES else None

    beta2m = None
    if len(blocks) == 3:
        beta2m = blocks[1]
    elif len(blocks) != 2:
        raise MappingError(f"{folder}: expected 2 or 3 chains, found {len(blocks)}")

    checks["groove_tokens"] = [groove.start, groove.stop]
    checks["peptide_tokens"] = [peptide.start, peptide.stop]
    checks["alpha3_tokens"] = None if alpha3 is None else [alpha3.start, alpha3.stop]
    checks["beta2m_tokens"] = None if beta2m is None else [beta2m.start, beta2m.stop]

    return Fold(
        folder=Path(folder),
        metadata=metadata,
        atoms=atoms,
        pae=pae,
        plddt=plddt,
        confidence=confidence,
        chains=blocks,
        hla=hla,
        peptide=peptide,
        beta2m=beta2m,
        groove=groove,
        alpha3=alpha3,
        checks=checks,
    )


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def _sequence(ca, block: ChainBlock) -> str:
    return "".join(_THREE_TO_ONE.get(str(r).upper(), "X") for r in ca.res_name[block.slice])


# --- Features ---------------------------------------------------------------


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_std": float(values.std()),
    }


def _block_mean(pae: np.ndarray, rows: slice, cols: slice) -> float:
    return float(pae[rows, cols].mean())


def _residue_atom_indices(atoms, block: ChainBlock, n_residues: int | None = None):
    """Atom index arrays, one per residue of ``block``, in file order."""
    import biotite.structure as struc

    where = np.flatnonzero(atoms.chain_id == block.chain_id)
    chain = atoms[where]
    starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    groups = [
        where[a:b]
        for a, b in zip(starts[:-1], starts[1:])
        if np.any(chain.atom_name[a:b] == "CA")
    ]
    if len(groups) != block.length:
        raise MappingError(
            f"chain {block.chain_id}: {len(groups)} residues with a CA, expected {block.length}"
        )
    return groups if n_residues is None else groups[:n_residues]


def extract_features(fold: Fold) -> dict[str, Any]:
    """The stage-5 feature row for one fold. Every value is finite or None."""
    import biotite.structure as struc

    meta = fold.metadata
    inputs = meta["inputs"]
    pep_len = fold.peptide.length
    row: dict[str, Any] = {
        "model": meta["model"],
        "allele": inputs["allele"],
        "peptide": inputs["peptide"],
        "arm": inputs["arm"],
        "seed": meta["seed"],
        "complex_id": inputs["complex_id"],
        "n_chains": len(fold.chains),
        "n_tokens": fold.n_tokens,
        "peptide_length": pep_len,
        "hla_chain_length": fold.hla.length,
        "beta2m_chain_length": None if fold.beta2m is None else fold.beta2m.length,
    }

    pae, plddt = fold.pae, fold.plddt
    pep, groove = fold.peptide.slice, fold.groove

    # --- pLDDT -------------------------------------------------------------
    pep_plddt = plddt[pep]
    row.update(_stats("pep_plddt", pep_plddt))
    for i in range(pep_len):
        row[f"pep_plddt_p{i + 1}"] = float(pep_plddt[i])
    row["groove_plddt_mean"] = float(plddt[groove].mean())
    row["hla_chain_plddt_mean"] = float(plddt[fold.hla.slice].mean())
    row["alpha3_plddt_mean"] = None if fold.alpha3 is None else float(plddt[fold.alpha3].mean())
    row["beta2m_plddt_mean"] = (
        None if fold.beta2m is None else float(plddt[fold.beta2m.slice].mean())
    )
    row["complex_plddt_mean_tokens"] = float(plddt.mean())

    # --- PAE, both directions ----------------------------------------------
    # Direction is in the name: "<rows>_rows_<cols>_cols". pae[i, j] is the
    # expected error in token j's position when the frame is aligned on token i,
    # so the two directions are different quantities, not a convention choice.
    pep_to_groove = pae[pep, groove]
    groove_to_pep = pae[groove, pep]
    row.update(_stats("pae_pep_rows_groove_cols", pep_to_groove))
    row.update(_stats("pae_groove_rows_pep_cols", groove_to_pep))
    row["pae_pep_groove_asymmetry"] = (
        row["pae_pep_rows_groove_cols_mean"] - row["pae_groove_rows_pep_cols_mean"]
    )
    for i in range(pep_len):
        row[f"pae_pep_rows_groove_cols_p{i + 1}"] = float(pep_to_groove[i].mean())
        row[f"pae_groove_rows_pep_cols_p{i + 1}"] = float(groove_to_pep[:, i].mean())
    row["pae_pep_intra_mean"] = _block_mean(pae, pep, pep)
    row["pae_groove_intra_mean"] = _block_mean(pae, groove, groove)

    # Ectodomain / beta2m blocks. Separately named, never folded into the
    # groove features above.
    for name, sl in (("alpha3", fold.alpha3), ("b2m", None if fold.beta2m is None else fold.beta2m.slice)):
        row[f"pae_pep_rows_{name}_cols_mean"] = None if sl is None else _block_mean(pae, pep, sl)
        row[f"pae_{name}_rows_pep_cols_mean"] = None if sl is None else _block_mean(pae, sl, pep)
        row[f"pae_groove_rows_{name}_cols_mean"] = None if sl is None else _block_mean(pae, groove, sl)
        row[f"pae_{name}_rows_groove_cols_mean"] = None if sl is None else _block_mean(pae, sl, groove)

    # --- Contacts and burial against HLA residues 1-182 --------------------
    # One receptor atom selection, used for contacts and for burial alike.
    groove_residues = _residue_atom_indices(fold.atoms, fold.hla, GROOVE_RESIDUES)
    pep_residues = _residue_atom_indices(fold.atoms, fold.peptide)
    coord = fold.atoms.coord
    receptor_idx = np.concatenate(groove_residues)
    receptor = coord[receptor_idx]

    per_pos_contacts: list[int] = []
    per_pos_mindist: list[float] = []
    contacted_groove: set[int] = set()
    groove_offsets = np.cumsum([0] + [len(g) for g in groove_residues])
    for res_idx in pep_residues:
        d = np.linalg.norm(coord[res_idx][:, None, :] - receptor[None, :, :], axis=-1)
        close = d < CONTACT_CUTOFF_A
        per_pos_contacts.append(int(close.sum()))
        per_pos_mindist.append(float(d.min()))
        hit_atoms = np.flatnonzero(close.any(axis=0))
        if hit_atoms.size:
            contacted_groove.update(
                int(np.searchsorted(groove_offsets, a, side="right") - 1) for a in hit_atoms
            )
    row["groove_contacts_4p5A_total"] = int(sum(per_pos_contacts))
    row["groove_contacts_4p5A_mean_per_residue"] = float(np.mean(per_pos_contacts))
    row["groove_residues_contacted"] = len(contacted_groove)
    row["pep_min_dist_to_groove_A"] = float(min(per_pos_mindist))
    row["pep_max_min_dist_to_groove_A"] = float(max(per_pos_mindist))
    for i in range(pep_len):
        row[f"groove_contacts_4p5A_p{i + 1}"] = per_pos_contacts[i]
        row[f"pep_min_dist_to_groove_p{i + 1}"] = per_pos_mindist[i]

    # Burial: Shrake-Rupley SASA of the peptide alone, minus its SASA in the
    # groove-only complex. Same receptor selection as the contacts above, so
    # alpha3 and beta2m never occlude the peptide in one feature and not the
    # other.
    pep_idx = np.concatenate(pep_residues)
    subset = fold.atoms[np.concatenate([receptor_idx, pep_idx])]
    sasa_kwargs = dict(
        probe_radius=SASA_PROBE_RADIUS,
        point_number=SASA_POINT_NUMBER,
        vdw_radii=SASA_VDW_RADII,
    )
    n_rec = len(receptor_idx)
    bound = np.nan_to_num(struc.sasa(subset, **sasa_kwargs)[n_rec:], nan=0.0)
    alone = np.nan_to_num(struc.sasa(subset[n_rec:], **sasa_kwargs), nan=0.0)
    total_alone = float(alone.sum())
    row["pep_sasa_alone_A2"] = total_alone
    row["pep_sasa_in_groove_A2"] = float(bound.sum())
    row["pep_buried_sasa_A2"] = total_alone - float(bound.sum())
    row["pep_buried_sasa_frac"] = (
        (total_alone - float(bound.sum())) / total_alone if total_alone > 0 else None
    )
    starts = np.cumsum([0] + [len(g) for g in pep_residues])
    for i in range(pep_len):
        a, b = starts[i], starts[i + 1]
        tot = float(alone[a:b].sum())
        row[f"pep_buried_sasa_frac_p{i + 1}"] = (
            (tot - float(bound[a:b].sum())) / tot if tot > 0 else None
        )

    # --- Confidence scalars -------------------------------------------------
    # Global ipTM is stored as global ipTM: in a three-chain complex it covers
    # the HLA/beta2m interface too, so it is not peptide-interface confidence.
    conf = fold.confidence
    for src, dst in (
        ("confidence_score", "global_confidence_score"),
        ("ptm", "global_ptm"),
        ("iptm", "global_iptm"),
        ("protein_iptm", "global_protein_iptm"),
        ("ligand_iptm", "global_ligand_iptm"),
        ("complex_plddt", "global_complex_plddt"),
        ("complex_iplddt", "global_complex_iplddt"),
        ("complex_pde", "global_complex_pde"),
        ("complex_ipde", "global_complex_ipde"),
    ):
        value = conf.get(src)
        row[dst] = float(value) if isinstance(value, (int, float)) else None

    chain_index = {b.chain_id: i for i, b in enumerate(fold.chains)}
    names = {fold.hla.chain_id: "hla_chain", fold.peptide.chain_id: "peptide"}
    if fold.beta2m is not None:
        names[fold.beta2m.chain_id] = "b2m"
    chains_ptm = conf.get("chains_ptm") if isinstance(conf.get("chains_ptm"), dict) else {}
    for cid, label in names.items():
        value = chains_ptm.get(str(chain_index[cid]))
        row[f"chain_ptm_{label}"] = float(value) if isinstance(value, (int, float)) else None

    row.update(_pair_iptm(fold, chain_index, names))

    row["status"] = "ok"
    bad = [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
    if bad:
        raise MappingError(f"{fold.folder}: non-finite feature values {bad}")
    return row


def _pair_iptm(fold: Fold, chain_index: dict[str, int], names: dict[str, str]) -> dict[str, Any]:
    """Boltz ``pair_chains_iptm``, exported with its direction in the name.

    ``pair_chains_iptm[a][b]`` aggregates PAE rows in chain ``b`` against
    columns in chain ``a`` (boltz 2.1.1 ``compute_ptms``), so the exported name
    follows the PAE convention, not the JSON key order.
    """
    labels = sorted(names.values())
    out: dict[str, Any] = {
        f"iptm_pair_{r}_rows_{c}_cols": None
        for r in labels
        for c in labels
        if r != c
    }
    raw = fold.confidence.get("pair_chains_iptm")
    if fold.metadata["model"] not in PAIR_IPTM_VERIFIED_MODELS or not isinstance(raw, dict):
        # Unverified chain-index mapping: leave the columns empty rather than
        # emit a number that might belong to the wrong interface.
        return out
    for cid_r, label_r in names.items():
        for cid_c, label_c in names.items():
            if label_r == label_c:
                continue
            a, b = chain_index[cid_c], chain_index[cid_r]
            value = raw.get(str(a), {}).get(str(b))
            if isinstance(value, (int, float)):
                out[f"iptm_pair_{label_r}_rows_{label_c}_cols"] = float(value)
    return out


# --- Discovery --------------------------------------------------------------


def is_excluded(path: Path, root: Path) -> bool:
    """True if ``path`` sits under a ``_smoke`` / ``_shards`` / ``_failed`` tree.

    Production writes those siblings next to the cohort output. Ingesting them
    would add harness folds to the feature table without raising anything.
    """
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        rel = Path(path)
    return any(part in EXCLUDED_DIR_NAMES for part in rel.parts)


def discover_folds(root: Path) -> Iterator[Path]:
    """Prediction directories under ``root``, excluded trees skipped."""
    root = Path(root)
    for meta in sorted(root.rglob("metadata.json")):
        folder = meta.parent
        if is_excluded(folder, root):
            continue
        yield folder


def extract_path(folder: Path) -> dict[str, Any]:
    """Extract one fold, converting any failure into a recorded failure row."""
    folder = Path(folder)
    try:
        return extract_features(load_fold(folder))
    except Exception as exc:  # noqa: BLE001 - failures are data, not crashes
        row: dict[str, Any] = {"status": f"{type(exc).__name__}: {exc}"}
        try:
            meta = json.loads((folder / "metadata.json").read_text())
            row.update(
                model=meta.get("model"),
                seed=meta.get("seed"),
                arm=meta.get("inputs", {}).get("arm"),
                allele=meta.get("inputs", {}).get("allele"),
                peptide=meta.get("inputs", {}).get("peptide"),
                complex_id=meta.get("inputs", {}).get("complex_id"),
            )
        except Exception:  # noqa: BLE001
            pass
        return row


# --- Joining to the dataset -------------------------------------------------

KEY_COLUMNS = ["model", "allele", "peptide", "arm", "seed"]


def attach_pair_ids(frame, splits):
    """Attach ``pair_id``/``split`` by joining on ``(allele, peptide)``.

    Never positional on ``pair_id``: ``pair_id`` indexes the raw CSV, and a
    positional join would silently mis-label rows.
    """
    import pandas as pd  # noqa: F401

    cols = ["allele", "peptide", "pair_id", "split", "cluster_id"]
    right = splits[[c for c in cols if c in splits.columns]].drop_duplicates(["allele", "peptide"])
    merged = frame.merge(right, on=["allele", "peptide"], how="left", validate="many_to_one")
    return merged
