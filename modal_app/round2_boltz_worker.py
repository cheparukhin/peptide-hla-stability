"""Isolated worker: original Boltz preprocessing/Lightning precision, custom writer.

The process-level predict_step replacement serves the extraction path only. The
original installed Boltz2.forward remains unchanged, including its full fold path.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from pepstab.boltz_embeddings import (
    BOLTZ_VERSION, CHECKPOINT_REVISION, CHECKPOINT_SHA256, EQUIVALENCE_TOLERANCES, PATCH_VERSION, SCHEMA_VERSION,
    bind_trunk_extractor, compare_tensors, crop_interface, input_hash,
    json_bytes, make_token_map, sha256_bytes, sha256_file, write_cache_entry, validate_cache_entry,
)


@contextlib.contextmanager
def head_audit(model, *, forbidden=False):
    """Count and, in extraction mode, reject every expensive downstream head."""
    counts = {}
    restored = []
    for label, obj, name in (("distogram", model.distogram_module, "forward"),
                             ("diffusion_conditioning", model.diffusion_conditioning, "forward"),
                             ("diffusion_sample", model.structure_module, "sample"),
                             ("confidence", model.confidence_module, "forward")):
        original = getattr(obj, name)
        counts[label] = 0
        def guarded(*args, _original=original, _label=label, **kwargs):
            counts[_label] += 1
            if forbidden:
                raise RuntimeError(f"extraction invoked forbidden head {_label}")
            return _original(*args, **kwargs)
        setattr(obj, name, guarded)
        restored.append((obj, name, original))
    try:
        yield counts
    finally:
        for obj, name, original in restored:
            setattr(obj, name, original)


def run(config: dict):
    start = time.monotonic()
    import torch
    import numpy as np
    from boltz.data import const
    import boltz.main as boltz_main
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.write.writer import BoltzWriter

    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    inputs = Path(config["work"]) / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    msa_root = Path(config["msa_root"])
    cache = Path(config["cache"])
    ckpt = cache / "boltz2_conf.ckpt"
    checkpoint_hash = sha256_file(ckpt)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("cached checkpoint differs from pinned Hugging Face revision LFS digest")
    by_id = {case["complex_id"]: case for case in config["cases"]}
    input_audits = {}
    for case in config["cases"]:
        yaml_lines = ["version: 1", "sequences:"]
        actual_msas = {}
        for chain in case["chains"]:
            path = "empty"
            if chain["msa"]:
                source = msa_root / chain["msa"]["csv"]
                if sha256_file(source) != chain["msa"]["csv_sha256"]:
                    raise ValueError(f"MSA hash mismatch: {source}")
                path = source
                if case["msa_depth_cap"] != 1024:
                    path = inputs.parent / f"{case['complex_id']}_{chain['id']}.csv"
                    with source.open() as f:
                        rows = list(csv.DictReader(f))[:case["msa_depth_cap"]]
                    with path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["sequence", "key"])
                        writer.writeheader()
                        writer.writerows(rows)
                actual_msas[chain["id"]] = {"sha256": sha256_file(path), "source_sha256": chain["msa"]["csv_sha256"],
                                            "rows": min(case["msa_depth_cap"], chain["msa"]["depth"])}
            yaml_lines += ["  - protein:", f"      id: {chain['id']}", f"      sequence: {chain['sequence']}", f"      msa: {path}"]
        yaml_text = "\n".join(yaml_lines) + "\n"
        (inputs / f"{case['complex_id']}.yaml").write_text(yaml_text)
        input_audits[case["complex_id"]] = {"yaml_sha256": sha256_bytes(yaml_text.encode()), "msas": actual_msas}

    settings = {"package": f"boltz=={BOLTZ_VERSION}", "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_sha256": checkpoint_hash, "recycling_steps": 3, "actual_trunk_iterations": 4,
                "sampling_steps_full_audit": 200, "diffusion_samples": 1, "precision": "bf16-mixed",
                "max_msa_seqs": 1024, "subsample_msa": False, "seed": 0,
                "seed_scope": "process; exact GPU RNG restored before each comparison of a case",
                "use_kernels": True, "storage_dtype": "float16", "patch_version": PATCH_VERSION,
                "cache_schema": SCHEMA_VERSION, "float32_matmul_precision": "highest",
                "equivalence_tolerances": EQUIVALENCE_TOLERANCES, "mode": config["mode"]}
    report = {"settings": settings, "cases": [], "status": "running", "checkpoint_hash_s": time.monotonic() - start,
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
              "config": config}
    (out / "report.json").write_bytes(json_bytes(report))
    original_predict = Boltz2.predict_step
    original_writer = BoltzWriter.write_on_batch_end
    original_download = boltz_main.download_boltz2
    extraction = None
    extractor_audit = None
    prediction_start = time.monotonic()

    def predict_step(model, batch, batch_idx, dataloader_idx=0):
        nonlocal extraction, extractor_audit
        case_start = time.monotonic()
        if extraction is None:
            extraction, extractor_audit = bind_trunk_extractor(model)
            report["load_preprocess_to_first_predict_s"] = case_start - prediction_start
            model.requires_grad_(False)
        cid = batch["record"][0].id
        case = by_id[cid]
        identity = input_hash(case, settings)
        if config["mode"] == "extract":
            try:
                cached = validate_cache_entry(out / "cache" / cid, expected_input_hash=identity)
            except (OSError, ValueError, KeyError):
                cached = None
            if cached is not None:
                complete = json.loads((out / "cache" / cid / "complete.json").read_text())
                report["cases"].append({"complex_id": cid, "pair_id": case["pair_id"], "ok": True,
                                        "reused": True, "cache": {**complete, "compressed_bytes": cached["compressed_bytes"]}})
                (out / "report.json").write_bytes(json_bytes(report))
                return {"exception": False}
        token_letters = {const.token_ids[name]: letter for letter, name in const.prot_letter_to_token.items()
                         if letter in "ACDEFGHIKLMNPQRSTVWY"}
        token_map = make_token_map(batch, case["chains"], token_letters)
        tensor_input_hash = hashlib.sha256()
        for key in sorted(batch):
            value = batch[key]
            if isinstance(value, torch.Tensor):
                tensor_input_hash.update(key.encode())
                tensor_input_hash.update(str((value.shape, value.dtype)).encode())
                tensor_input_hash.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        def reset_rng():
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(gpu_rng)
        rec = {"complex_id": cid, "pair_id": case["pair_id"], "allele": case["allele"],
               "msa_depth_cap": case["msa_depth_cap"], "technical_depth_variant": case["technical_depth_variant"],
               "preprocessed_tensor_sha256": tensor_input_hash.hexdigest(), "token_count": len(token_map),
               "msa_shape": list(batch["msa"].shape), "ok": False}
        kwargs = dict(recycling_steps=3, num_sampling_steps=200, diffusion_samples=1,
                      max_parallel_samples=None, run_confidence_sequentially=True)
        reference = {}
        if config["mode"] == "verify":
            calls = 0
            pairformer = model.pairformer_module._orig_mod if model.is_pairformer_compiled else model.pairformer_module
            def capture(module, args, output):
                nonlocal calls
                calls += 1
                if calls == 4:
                    # Capture CPU copies at the exact final trunk boundary.
                    reference.update(s=output[0].detach().clone().cpu(), z=output[1].detach().clone().cpu())
            hook = pairformer.register_forward_hook(capture)
            reset_rng()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            try:
                with head_audit(model) as head_counts:
                    full = model(batch, **kwargs)
                torch.cuda.synchronize()
                rec.update(full_forward_s=time.monotonic() - t0, full_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           full_head_calls=head_counts, trunk_iterations=calls,
                           full_has_coordinates="sample_atom_coords" in full, full_has_confidence="plddt" in full)
                if calls != 4 or not all(head_counts.values()) or not rec["full_has_coordinates"] or not rec["full_has_confidence"]:
                    raise ValueError("full forward did not execute the expected trunk and heads")
                del full
            finally:
                hook.remove()
        reset_rng()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with head_audit(model, forbidden=True) as extraction_head_counts:
            embedded = extraction(batch, **kwargs)
        torch.cuda.synchronize()
        rec.update(extract_s=time.monotonic() - t0, extract_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   extraction_head_calls=extraction_head_counts)
        transfer_start = time.monotonic()
        native = {k: v.detach().cpu() for k, v in embedded.items()}
        rec["gpu_to_host_s"] = time.monotonic() - transfer_start
        del embedded
        if reference:
            rec["equivalence"] = {k: compare_tensors(reference[k], native[k]) for k in native}
            if not all(x["allclose"] for x in rec["equivalence"].values()):
                raise ValueError("full/extraction equivalence failed")
            reset_rng()
            with head_audit(model, forbidden=True):
                repeated = extraction(batch, **kwargs)
            rec["repeat_variability"] = {k: compare_tensors(native[k], repeated[k].detach().cpu()) for k in native}
            if not all(x["allclose"] for x in rec["repeat_variability"].values()):
                raise ValueError("same-input repeat exceeds declared tolerance")
            del repeated
        original_blocks = crop_interface(native["s"], native["z"], token_map, dtype="float32")
        blocks = {k: v.astype("float16") for k, v in original_blocks.items()}
        rec["storage_conversion"] = {k: {"max_abs": float(np.max(np.abs(v - blocks[k].astype("float32")))),
                                         "allclose": bool(np.allclose(v, blocks[k].astype("float32"), atol=1e-3, rtol=1e-3))}
                                     for k, v in original_blocks.items()}
        if not all(x["allclose"] for x in rec["storage_conversion"].values()):
            raise ValueError("float16 conversion failed predeclared atol=rtol=1e-3")
        metadata = {"pair_id": case["pair_id"], "case": case, "token_map": token_map, "settings": settings,
                    "extractor": extractor_audit, "input_audit": input_audits[cid],
                    "preprocessed_tensor_sha256": rec["preprocessed_tensor_sha256"],
                    "input_hash": identity, "native_dtypes": {k: str(v.dtype) for k, v in native.items()}}
        storage_start = time.monotonic()
        rec["cache"] = write_cache_entry(out / "cache" / cid, blocks, metadata)
        rec["serialize_reload_s"] = time.monotonic() - storage_start
        if config["mode"] == "verify" and len(report["cases"]) < 2:
            np.savez_compressed(out / f"{cid}_full_audit.npz", **{k: v.float().numpy() for k, v in native.items()})
        rec.update(ok=True, case_wall_s=time.monotonic() - case_start, host_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        report["cases"].append(rec)
        (out / "report.json").write_bytes(json_bytes(report))
        print(json.dumps({k: rec[k] for k in ("complex_id", "ok", "extract_s", "case_wall_s")}), flush=True)
        return {"exception": False}

    # Keep the published checkpoint/cache immutable and prohibit network fallback.
    def verified_existing_cache(path):
        if not (path / "boltz2_conf.ckpt").exists() or not (path / "mols").is_dir():
            raise ValueError("pinned cached weights/molecules unavailable")
    try:
        Boltz2.predict_step = predict_step
        BoltzWriter.write_on_batch_end = lambda *a, **kw: None
        boltz_main.download_boltz2 = verified_existing_cache
        boltz_main.predict.main([str(inputs), "--out_dir", str(inputs.parent / "preprocessed"),
                                "--cache", str(cache), "--model", "boltz2", "--accelerator", "gpu",
                                "--devices", "1", "--recycling_steps", "3", "--sampling_steps", "200",
                                "--diffusion_samples", "1", "--max_msa_seqs", "1024", "--seed", "0",
                                "--num_workers", "0", "--override"], standalone_mode=False)
        if len(report["cases"]) != len(config["cases"]):
            raise ValueError("incomplete panel coverage")
        if config.get("contact_map_path"):
            from pepstab import round2_representations
            contact_map = round2_representations.load_contact_map(config["contact_map_path"])
            entries = [{"pair_id": r["pair_id"], "input_hash": r["cache"]["input_hash"],
                        "cache_path": str(out / "cache" / r["complex_id"])} for r in report["cases"]]
            feature_start = time.monotonic()
            report["compact_features"] = round2_representations.build_feature_bundle(entries, out / "features", contact_map)
            report["compact_features"]["module_sha256"] = sha256_file(round2_representations.__file__)
            report["compact_features"]["contact_file_sha256"] = sha256_file(config["contact_map_path"])
            report["compact_features"]["seconds"] = time.monotonic() - feature_start
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        Boltz2.predict_step = original_predict
        BoltzWriter.write_on_batch_end = original_writer
        boltz_main.download_boltz2 = original_download
        report["worker_wall_s"] = time.monotonic() - start
        report["host_peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        (out / "report.json").write_bytes(json_bytes(report))
    return report


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
