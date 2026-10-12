"""Frozen, label-independent Boltz representations for the bounded round-two screen.

All projection matrices and contact memberships are fixed before assessment.
The model input remains the full three-chain complex; only outputs are cropped.
"""
from __future__ import annotations

from functools import lru_cache
import io
import json
from pathlib import Path

import numpy as np

from pepstab.boltz_embeddings import (
    _atomic_bytes, json_bytes, sha256_bytes, sha256_file, validate_cache_entry,
)
from pepstab.esm import CONTACT_POSITIONS_1BASED

SEED = 20261012
CRYSTALS = ("1X7Q", "1XR9", "4QRU", "5N6B", "7LFZ")
FEATURE_SCHEMA = "pepstab-round2-representations-v1"
CONTACT_SCHEMA = "pepstab-round2-crystal-contact-v1"
FEATURE_DIMENSIONS = {"R2": 1152, "R3": 1528, "R4": 2304, "R5": 1024,
                      "R5_shuffled": 1024, "R5_position9x34": 2448,
                      "R3_pooled": 1528, "R5_position9x34_pooled": 2448}
PAIR_KEYS = ("peptide_to_groove_z", "groove_to_peptide_z")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports/further_research/round2/representations"


@lru_cache(maxsize=2)
def fixed_projection(input_channels: int, output_channels: int) -> np.ndarray:
    """Gaussian QR with canonical positive diagonal; independent of data/labels."""
    if (input_channels, output_channels) not in ((384, 8), (128, 4)):
        raise ValueError("only the prespecified token/pair projections are permitted")
    gaussian = np.random.default_rng(SEED).standard_normal((input_channels, output_channels))
    q, r = np.linalg.qr(gaussian, mode="reduced")
    q *= np.where(np.diag(r) < 0, -1., 1.)
    result = q.astype(np.float32)
    result.flags.writeable = False
    return result


def feature_protocol() -> dict:
    return {
        "schema_version": FEATURE_SCHEMA, "seed": SEED, "label_independent": True,
        "source_plan": "reports/NEXT_RESEARCH_PLAN.md sections 4-5",
        "required_cache_schema": "pepstab-boltz-interface-v2",
        "dtype": "float32", "flatten_order": "C",
        "chain_mean_order": ["A full HLA ectodomain", "B beta2m", "C peptide"],
        "direction_order": list(PAIR_KEYS),
        "representations": {
            "R2": "concatenate full-chain means A,B,C; no replacement by groove mean",
            "R3": "P1-P9 then groove1-182; shared fixed 384-to-8 channel projection; flatten",
            "R4": "groove mean for each P1-P9; layout [peptide,direction,128 channels]",
            "R5": "contact mean per peptide position; layout [P1,P2,P9,mean(P3-P8),direction,128 channels]",
            "R5_shuffled": "same R5 pooling with fixed permutation of HLA contact membership; cardinality per peptide preserved",
            "R5_position9x34": "all P1-P9 x canonical34; layout [peptide,contact,direction,4 projected channels]",
            "R3_pooled": "same tokens, projection and dimensions as R3; tile peptide mean over nine positions and groove mean over182 before projection",
            "R5_position9x34_pooled": "same projection and dimensions as position9x34; per-peptide/direction contact-mask mean of projected channels tiled over34 positions",
        },
        "dimensions": FEATURE_DIMENSIONS,
        "projections": {name: {"input_channels": n, "output_channels": k,
                                "algorithm": "numpy PCG64 Gaussian; reduced QR; positive R diagonal",
                                "float32_sha256": sha256_bytes(fixed_projection(n, k).tobytes())}
                        for name, n, k in (("token", 384, 8), ("pair", 128, 4))},
        "scope_amendment": "Fixed random projections bound fitting cost; never fitted on data or labels. This screen tests compressed positional channels and cannot exclude gains from full-channel or fitting-partition dimensional reduction.",
        "contact_rule": "union across all five existing crystals of minimum heavy-atom distance <=4.5 Angstrom, restricted to canonical NetMHCpan34",
        "central_pooling": "equal mean of six per-position pocket means; each P3-P8 receives equal weight",
        "shuffling": "fixed PCG64(seed) permutation of the 34 groove columns in the contact mask, applied before constrained pooling",
    }


def build_contact_map(crystal_directory: str | Path) -> dict:
    """Use observed coordinates only; verify author IDs equal mature label positions."""
    from biotite.structure.io.pdbx import CIFFile, get_structure
    from biotite.sequence import ProteinSequence

    positions = np.asarray(CONTACT_POSITIONS_1BASED)
    distances, sources = [], []
    for pdb_id in CRYSTALS:
        path = Path(crystal_directory) / f"{pdb_id}.cif"
        cif = CIFFile.read(path)
        scheme = cif.block["pdbx_poly_seq_scheme"]
        columns = {k: scheme[k].as_array(str) for k in scheme.keys()}
        sequences = {}
        for chain, length in (("A", 182), ("C", 9)):
            rows = np.flatnonzero(columns["asym_id"] == chain)
            rows = rows[:length]
            expected = np.arange(1, length + 1).astype(str)
            if len(rows) != length or any(not np.array_equal(columns[k][rows], expected)
                                         for k in ("seq_id", "pdb_seq_num", "auth_seq_num")):
                raise ValueError(f"{pdb_id} {chain}: mature numbering mismatch")
            if np.any(~np.isin(columns["pdb_ins_code"][rows], [".", "?"])):
                raise ValueError("inserted residues cannot be indexed without alignment")
            sequences[chain] = "".join(ProteinSequence.convert_letter_3to1(x) for x in columns["mon_id"][rows])
        if not sequences["A"].startswith("GSHSMR"):
            raise ValueError("HLA mature N-terminal motif not verified")
        atoms = get_structure(cif, model=1, altloc="first", use_author_fields=True)
        heavy = atoms[~atoms.hetero & ~np.isin(atoms.element, ["H", "D"])]
        d = np.empty((9, 34), dtype=np.float64)
        for p in range(1, 10):
            peptide = heavy[(heavy.chain_id == "C") & (heavy.res_id == p)]
            if not len(peptide):
                raise ValueError(f"{pdb_id}: missing peptide P{p}")
            for j, h in enumerate(positions):
                groove = heavy[(heavy.chain_id == "A") & (heavy.res_id == h)]
                if not len(groove):
                    raise ValueError(f"{pdb_id}: missing HLA {h}")
                d[p - 1, j] = np.linalg.norm(peptide.coord[:, None] - groove.coord[None, :], axis=-1).min()
        distances.append(d)
        sources.append({"pdb_id": pdb_id, "path": f"structures/crystal_cache/{pdb_id}.cif",
                        "sha256": sha256_file(path), "hla_groove_sequence": sequences["A"],
                        "peptide_sequence": sequences["C"], "numbering_verified": True,
                        "mapping": "A author/label mature1-182 identity; C author/label P1-P9 identity",
                        "minimum_heavy_atom_distances_angstrom": d.round(6).tolist()})
    minimum = np.min(distances, axis=0)
    mask = minimum <= 4.5
    fallback = []
    for p in range(9):
        if not mask[p].any():
            j = int(np.argmin(minimum[p]))
            mask[p, j] = True
            fallback.append({"peptide_position": p + 1, "hla_position": int(positions[j]),
                             "distance_angstrom": float(minimum[p, j])})
    permutation = np.random.default_rng(SEED).permutation(34)
    shuffled = mask[:, permutation]
    if np.array_equal(mask, shuffled):
        raise ValueError("shuffle failed to change pocket membership")
    result = {"schema_version": CONTACT_SCHEMA, "seed": SEED, "contact_positions_1based": positions.tolist(),
              "contact_positions_source": "pepstab.esm.CONTACT_POSITIONS_1BASED; existing canonical list verified against all 75 dataset alleles",
              "chains": {"groove": "A", "peptide": "C"}, "model": 1, "altloc": "first",
              "heavy_atoms": "non-hetero; exclude element H,D", "threshold_angstrom": 4.5,
              "aggregation": "union of contacts across all five crystals", "sources": sources,
              "fallback_rule": "if no contacts at a peptide position, use lowest-index closest canonical34 residue by minimum distance across crystals",
              "fallback_applied": fallback, "mask": mask.astype(int).tolist(),
              "pockets_1based": [positions[row].tolist() for row in mask],
              "shuffled_column_permutation_0based": permutation.tolist(),
              "shuffled_mask": shuffled.astype(int).tolist()}
    result["contact_map_sha256"] = sha256_bytes(json_bytes(result))
    return result


def validate_contact_map(contact_map: dict) -> tuple[np.ndarray, np.ndarray]:
    if contact_map.get("schema_version") != CONTACT_SCHEMA:
        raise ValueError("unknown contact map schema")
    if contact_map.get("contact_positions_1based") != list(CONTACT_POSITIONS_1BASED):
        raise ValueError("contact position order mismatch")
    payload = {k: v for k, v in contact_map.items() if k != "contact_map_sha256"}
    if contact_map.get("contact_map_sha256") != sha256_bytes(json_bytes(payload)):
        raise ValueError("contact map content hash mismatch")
    masks = []
    for key in ("mask", "shuffled_mask"):
        raw = np.asarray(contact_map[key])
        if raw.shape != (9, 34) or not np.isin(raw, [0, 1]).all() or not raw.any(axis=1).all():
            raise ValueError("invalid contact membership")
        masks.append(raw.astype(bool))
    permutation = np.asarray(contact_map["shuffled_column_permutation_0based"])
    if sorted(permutation.tolist()) != list(range(34)) or not np.array_equal(masks[1], masks[0][:, permutation]):
        raise ValueError("invalid shuffled membership")
    if np.array_equal(*masks):
        raise ValueError("shuffled control leaves membership unchanged")
    return tuple(masks)


def load_contact_map(path: str | Path = REPORT_DIR / "contact_map.json") -> dict:
    contact_map = json.loads(Path(path).read_text())
    validate_contact_map(contact_map)
    return contact_map


def representation_features(blocks: dict, contact_map: dict) -> dict[str, np.ndarray]:
    """Return six prespecified 1-D features; never access outcomes or fit parameters."""
    masks = validate_contact_map(contact_map)
    shapes = {"chain_means_s": (3, 384), "peptide_s": (9, 384), "groove_s": (182, 384),
              **{k: (9, 182, 128) for k in PAIR_KEYS}}
    arrays = {}
    for key, shape in shapes.items():
        if key not in blocks:
            raise ValueError(f"required v2 block missing: {key}")
        a = np.asarray(blocks[key], dtype=np.float32)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"invalid representation block {key}")
        arrays[key] = a
    z = np.stack([arrays[k] for k in PAIR_KEYS], axis=2)  # [p,g,direction,channel]
    contact_z = z[:, np.asarray(CONTACT_POSITIONS_1BASED) - 1]
    pooled_tokens = np.concatenate([np.repeat(arrays["peptide_s"].mean(axis=0, keepdims=True), 9, axis=0),
                                    np.repeat(arrays["groove_s"].mean(axis=0, keepdims=True), 182, axis=0)])
    projected_pairs = contact_z @ fixed_projection(128, 4)
    pooled_pairs = np.stack([projected_pairs[p, masks[0][p]].mean(axis=0) for p in range(9)])
    result = {"R2": arrays["chain_means_s"].ravel(),
              "R3": (np.concatenate([arrays["peptide_s"], arrays["groove_s"]]) @ fixed_projection(384, 8)).ravel(),
              "R4": z.mean(axis=1).ravel(),
              "R5_position9x34": projected_pairs.ravel(),
              "R3_pooled": (pooled_tokens @ fixed_projection(384, 8)).ravel(),
              "R5_position9x34_pooled": np.repeat(pooled_pairs[:, None], 34, axis=1).ravel()}
    for name, mask in zip(("R5", "R5_shuffled"), masks):
        per_position = np.stack([contact_z[p, mask[p]].mean(axis=0) for p in range(9)])
        result[name] = np.stack([per_position[0], per_position[1], per_position[8],
                                 per_position[2:8].mean(axis=0)]).ravel()
    for name, dimension in FEATURE_DIMENSIONS.items():
        if result[name].shape != (dimension,) or not np.isfinite(result[name]).all():
            raise ValueError(f"non-finite or invalid features {name}")
    return {name: result[name].astype(np.float32, copy=False) for name in FEATURE_DIMENSIONS}


def build_feature_bundle(entries: list[dict], output_directory: str | Path,
                         contact_map: dict) -> dict:
    """Validate each raw v2 cache and publish content-addressed compact features."""
    if not entries:
        raise ValueError("empty cache manifest")
    pair_ids = [str(e["pair_id"]) for e in entries]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate pair_id in manifest")
    collected = {name: [] for name in FEATURE_DIMENSIONS}
    rows = []
    for entry, pair_id in zip(entries, pair_ids):
        path = Path(entry["cache_path"])
        info = validate_cache_entry(path, expected_input_hash=entry["input_hash"])
        cache_pair_id = info.get("pair_id", info.get("case", {}).get("pair_id"))
        if str(cache_pair_id) != pair_id:
            raise ValueError("cache pair_id does not match manifest")
        if "pair_id" in info and "pair_id" in info.get("case", {}) and str(info["pair_id"]) != str(info["case"]["pair_id"]):
            raise ValueError("cache metadata pair_id locations disagree")
        if info["schema_version"] != "pepstab-boltz-interface-v2":
            raise ValueError("R2 requires full-chain cache v2")
        complete = json.loads((path / "complete.json").read_text())
        with np.load(path / complete["array_file"], allow_pickle=False) as raw:
            features = representation_features({k: raw[k] for k in raw.files}, contact_map)
        for name, feature in features.items():
            collected[name].append(feature)
        rows.append({"pair_id": pair_id, "input_hash": info["input_hash"],
                     "cache_path": str(path.resolve()), "cache_npz_sha256": complete["npz_sha256"],
                     "cache_metadata_sha256": complete["metadata_sha256"]})
    arrays = {name: np.stack(values) for name, values in collected.items()}
    arrays.update(pair_id=np.asarray(pair_ids), input_hash=np.asarray([row["input_hash"] for row in rows]))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    payload_hash = sha256_bytes(payload)
    protocol = feature_protocol()
    manifest = {"schema_version": FEATURE_SCHEMA, "rows": rows, "n_rows": len(rows),
                "dimensions": FEATURE_DIMENSIONS, "feature_protocol": protocol,
                "feature_protocol_sha256": sha256_bytes(json_bytes(protocol)),
                "contact_map_sha256": contact_map["contact_map_sha256"],
                "npz_sha256": payload_hash, "array_file": f"features-{payload_hash}.npz"}
    manifest_bytes = json_bytes(manifest)
    manifest_hash = sha256_bytes(manifest_bytes)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(directory / manifest["array_file"], payload)
    metadata_name = f"manifest-{manifest_hash}.json"
    _atomic_bytes(directory / metadata_name, manifest_bytes)
    _atomic_bytes(directory / "complete.json", json_bytes({"schema_version": FEATURE_SCHEMA,
                  "array_file": manifest["array_file"], "metadata_file": metadata_name,
                  "npz_sha256": payload_hash, "metadata_sha256": manifest_hash}))
    return manifest
