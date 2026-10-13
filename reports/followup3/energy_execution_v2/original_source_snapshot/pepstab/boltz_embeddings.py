"""Audited, frozen Boltz-2.1.1 trunk extraction and directional interface cache.

No Boltz import or cloud side effect is needed for cache validation. The extraction
method is compiled from the exact installed upstream forward prefix, with a
dedicated return before *any* output head. The original forward is never replaced.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import textwrap
import types
import uuid

import numpy as np

BOLTZ_VERSION = "2.1.1"
PATCH_VERSION = "round2-trunk-v1"
FORWARD_SHA256 = "70ed763b82eca88f6a3e721e4c9407958210d832a8352be27ddbc8802306c51b"
WHEEL_SHA256 = "1d036a689ed7a9bb24ac090b36e9af746e2350df9d4ef277e420b6d6838ba0b8"
CHECKPOINT_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
CHECKPOINT_SHA256 = "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"
SCHEMA_VERSION = "pepstab-boltz-interface-v2"
LEGACY_SCHEMA_VERSION = "pepstab-boltz-interface-v1"
BLOCKS = ("peptide_s", "groove_s", "peptide_to_groove_z", "groove_to_peptide_z", "chain_means_s")
# Declared before runtime. Native bf16 and fp16 use their respective precision;
# validation also reports exact equality and observed same-input repeat variation.
EQUIVALENCE_TOLERANCES = {
    "torch.float32": {"atol": 1e-5, "rtol": 1e-5},
    "torch.float16": {"atol": 2e-3, "rtol": 2e-3},
    "torch.bfloat16": {"atol": 2e-2, "rtol": 2e-2},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: dict | list) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def input_hash(case: dict, settings: dict) -> str:
    return sha256_bytes(json_bytes({"case": case, "settings": settings}))


def compile_trunk_forward(original_forward, *, expected_sha256=FORWARD_SHA256):
    """Copy the pinned forward through its final trunk update, then return s/z.

    A byte-for-byte source check fails closed on package drift. An AST boundary
    check prevents accidentally including distogram/diffusion/confidence heads.
    This does not mutate the package, class, instance forward, or folding path.
    """
    source = textwrap.dedent(inspect.getsource(original_forward))
    actual = sha256_bytes(source.encode())
    if actual != expected_sha256:
        raise ValueError(f"Boltz forward source drift: {actual} != {expected_sha256}")
    module = ast.parse(source)
    fn = module.body[0]
    if not isinstance(fn, ast.FunctionDef) or not isinstance(fn.body[0], ast.With):
        raise ValueError("unexpected Boltz forward structure")
    body = fn.body[0].body
    boundaries = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "pdistogram" for t in node.targets)]
    if len(boundaries) != 1:
        raise ValueError("expected exactly one distogram boundary")
    boundary = boundaries[0]
    if "self.distogram_module(z)" != ast.unparse(body[boundary].value):
        raise ValueError("unexpected distogram boundary expression")
    fn.name = "extract_trunk"
    fn.decorator_list = []
    fn.body[0].body = body[:boundary] + ast.parse("return {'s': s, 'z': z}").body
    fn.body = fn.body[:1]
    ast.fix_missing_locations(module)
    namespace = dict(original_forward.__globals__)
    exec(compile(module, "<pepstab-boltz-2.1.1-trunk-v1>", "exec"), namespace)
    return namespace[fn.name], {"patch_version": PATCH_VERSION, "forward_sha256": actual,
                              "extraction_ast_sha256": sha256_bytes(ast.dump(module).encode())}


def bind_trunk_extractor(model):
    from importlib.metadata import version
    if version("boltz") != BOLTZ_VERSION:
        raise ValueError("extractor requires boltz==2.1.1")
    if model.training or not model.run_trunk_and_structure:
        raise ValueError("extractor requires evaluation mode with trunk enabled")
    fn, audit = compile_trunk_forward(type(model).forward)
    return types.MethodType(fn, model), audit


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # numpy cannot represent bfloat16; preserve its exact values in float32.
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def make_token_map(feats: dict, chains: list[dict], token_letters: dict[int, str]) -> list[dict]:
    """Verify actual token features against input sequences, without order guesses.

    Boltz 2.1.1's protein chain IDs and residue_index are zero-based. Each chain's
    active token sequence and positions must match exactly; padding is excluded.
    Unexpected active non-protein/special/duplicate tokens fail closed.
    """
    arrays = {k: _numpy(feats[k]) for k in ("token_pad_mask", "asym_id", "residue_index", "mol_type", "res_type")}
    if any(a.shape[0] != 1 for a in arrays.values()):
        raise ValueError("cache requires one complex per batch")
    mask = arrays["token_pad_mask"][0].astype(bool)
    asym, res, mol = (arrays[k][0] for k in ("asym_id", "residue_index", "mol_type"))
    restype = arrays["res_type"][0].argmax(axis=-1)
    result = []
    for chain_number, chain in enumerate(chains):
        indices = np.flatnonzero(mask & (asym == chain_number))
        if len(indices) != len(chain["sequence"]):
            raise ValueError(f"token count mismatch for chain {chain['id']}")
        if not np.array_equal(res[indices], np.arange(len(indices))) or np.any(mol[indices] != 0):
            raise ValueError("unexpected residue indices or non-protein tokens")
        actual = "".join(token_letters.get(int(restype[i]), "?") for i in indices)
        if actual != chain["sequence"]:
            raise ValueError(f"token sequence mismatch for chain {chain['id']}")
        for i in indices:
            result.append({"token_index": int(i), "chain_id": chain["id"],
                           "asym_id": chain_number, "residue_index": int(res[i]),
                           "residue_number": int(res[i]) + 1, "residue": actual[int(res[i])]})
    if len(result) != int(mask.sum()) or len({r["token_index"] for r in result}) != len(result):
        raise ValueError("unmapped or duplicate active tokens")
    return result


def crop_interface(s, z, token_map: list[dict], *, peptide_chain="C", groove_chain="A", dtype="float16") -> dict:
    s, z = _numpy(s), _numpy(z)
    if s.ndim != 3 or z.ndim != 4 or s.shape[0] != 1 or z.shape[0] != 1:
        raise ValueError("expected batched token and pair tensors")
    if z.shape[1:3] != (s.shape[1], s.shape[1]):
        raise ValueError("inconsistent trunk tensor lengths")
    p = [x["token_index"] for x in token_map if x["chain_id"] == peptide_chain]
    g = [x["token_index"] for x in token_map if x["chain_id"] == groove_chain and x["residue_number"] <= 182]
    if len(p) != 9 or len(g) != 182:
        raise ValueError("expected peptide P1-P9 and groove residues 1-182")
    blocks = {"peptide_s": s[0, p], "groove_s": s[0, g],
              "peptide_to_groove_z": z[0][np.ix_(p, g)],
              # Retain actual reverse values, in the common [peptide, groove, channel] layout.
              "groove_to_peptide_z": z[0][np.ix_(g, p)].transpose(1, 0, 2)}
    chain_indices = [[x["token_index"] for x in token_map if x["chain_id"] == c] for c in "ABC"]
    if [len(x) for x in chain_indices] != [275, 99, 9]:
        raise ValueError("chain means require the original A275/B99/C9 construct")
    blocks["chain_means_s"] = np.stack([s[0, indices].astype("float32").mean(axis=0) for indices in chain_indices])
    if not all(np.isfinite(x).all() for x in blocks.values()):
        raise ValueError("non-finite trunk outputs")
    converted = {k: v.astype(dtype) for k, v in blocks.items()}
    if not all(np.isfinite(x).all() for x in converted.values()):
        raise ValueError("storage conversion overflow")
    return converted


def compare_tensors(reference, observed) -> dict:
    import torch
    if reference.shape != observed.shape or reference.dtype != observed.dtype:
        raise ValueError("equivalence shape/dtype mismatch")
    tolerance = EQUIVALENCE_TOLERANCES[str(reference.dtype)]
    a, b = reference.float(), observed.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"shape": list(a.shape), "dtype": str(reference.dtype), **tolerance,
            "finite": finite, "exact": bool(torch.equal(reference, observed)),
            "allclose": finite and bool(torch.allclose(a, b, **tolerance)),
            "max_abs": float((a - b).abs().max()),
            "rms": float(torch.mean((a - b).square()).sqrt())}


def _atomic_bytes(path: Path, content: bytes):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_cache_entry(directory: str | Path, blocks: dict, metadata: dict) -> dict:
    """Content-addressed shard files; completion is the final atomic operation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if set(blocks) != set(BLOCKS):
        raise ValueError("missing or unexpected embedding blocks")
    for key, arr in blocks.items():
        expected = (9,) if key == "peptide_s" else (182,) if key == "groove_s" else (3,) if key == "chain_means_s" else (9, 182)
        if arr.shape[:-1] != expected or not np.isfinite(arr).all():
            raise ValueError(f"invalid block {key}")
    if blocks["peptide_s"].shape[-1] != blocks["groove_s"].shape[-1] or blocks[BLOCKS[2]].shape != blocks[BLOCKS[3]].shape:
        raise ValueError("inconsistent channel counts")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **blocks)
    payload = buffer.getvalue()
    payload_hash = sha256_bytes(payload)
    info = {**metadata, "schema_version": SCHEMA_VERSION,
            "chain_mean_order": ["A", "B", "C"], "chain_mean_residue_counts": [275, 99, 9],
            "block_layout": "pair blocks both indexed [peptide P1-P9, groove 1-182, channel]; reverse values are z[groove,peptide]",
            "blocks": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in blocks.items()},
            "npz_sha256": payload_hash, "compressed_bytes": len(payload)}
    info_bytes = json_bytes(info)
    info_hash = sha256_bytes(info_bytes)
    array_name, metadata_name = f"embeddings-{payload_hash}.npz", f"metadata-{info_hash}.json"
    _atomic_bytes(directory / array_name, payload)
    _atomic_bytes(directory / metadata_name, info_bytes)
    complete = {"schema_version": SCHEMA_VERSION, "array_file": array_name, "metadata_file": metadata_name,
                "npz_sha256": payload_hash, "metadata_sha256": info_hash, "input_hash": metadata["input_hash"]}
    # Validation happens before publishing the final marker; reload must exactly
    # preserve every stored block, and corrupt files can never count as complete.
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        for k, v in blocks.items():
            if not np.array_equal(v, loaded[k]):
                raise ValueError("serialization changed stored values")
    _atomic_bytes(directory / "complete.json", json_bytes(complete))
    validate_cache_entry(directory, expected_input_hash=metadata["input_hash"])
    return {**complete, "compressed_bytes": len(payload)}


def validate_cache_entry(directory: str | Path, *, expected_input_hash: str | None = None) -> dict:
    directory = Path(directory)
    complete = json.loads((directory / "complete.json").read_text())
    if complete["schema_version"] not in (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
        raise ValueError("unknown cache schema")
    for field in ("array_file", "metadata_file"):
        if Path(complete[field]).name != complete[field]:
            raise ValueError("cache file must be a basename")
    metadata_path = directory / complete["metadata_file"]
    array_path = directory / complete["array_file"]
    if sha256_file(metadata_path) != complete["metadata_sha256"] or sha256_file(array_path) != complete["npz_sha256"]:
        raise ValueError("cache content hash mismatch")
    info = json.loads(metadata_path.read_text())
    if info["input_hash"] != complete["input_hash"] or info["npz_sha256"] != complete["npz_sha256"]:
        raise ValueError("cache manifest identity mismatch")
    if expected_input_hash is not None and info["input_hash"] != expected_input_hash:
        raise ValueError("stale cache input/settings")
    expected_blocks = BLOCKS if complete["schema_version"] == SCHEMA_VERSION else BLOCKS[:4]
    if info["schema_version"] != complete["schema_version"] or set(info["blocks"]) != set(expected_blocks):
        raise ValueError("invalid cache metadata schema")
    with np.load(array_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected_blocks):
            raise ValueError("missing embedding blocks")
        for k in expected_blocks:
            arr = arrays[k]
            expected = (9,) if k == "peptide_s" else (182,) if k == "groove_s" else (3,) if k == "chain_means_s" else (9, 182)
            if (arr.shape[:-1] != expected or list(arr.shape) != info["blocks"][k]["shape"]
                    or str(arr.dtype) != info["blocks"][k]["dtype"] or not np.isfinite(arr).all()):
                raise ValueError(f"invalid stored block {k}")
    return info
