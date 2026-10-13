"""Root-guarded ordinary Functions for frozen repaired energy extraction only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
MOUNT_ROOT = "/Users/cheparukhin/ai_science_hack"
app = modal.App("pepstab-followup3-repaired-energy")
structures = modal.Volume.from_name("pepstab-structures", create_if_missing=False)
outputs = modal.Volume.from_name("pepstab-followup3-20261013", create_if_missing=False)
image = modal.Image.from_id("im-Epc3H2RdNAgJSwtXBYMCfy")


@app.function(image=image, volumes={"/structures":structures.with_mount_options(read_only=True), MOUNT_ROOT:outputs},
              cpu=(1., 1.), memory=(2048, 2048), max_containers=100, timeout=660, retries=0,
              startup_timeout=120, scaledown_window=2)
def score_one(identity, protocol, deadline_utc):
    """One logical attempt; provider restarts cannot repeat committed starts."""
    # Root uploads exact source bytes and its existing FoldX installation to
    # the same authorized workspace volume; neither is embedded in an image.
    source_root = Path(MOUNT_ROOT)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      PYTHONPATH=str(source_root / "runs/followup3/dependencies") + os.pathsep + str(source_root))
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "runs/followup3/dependencies"))
    for name in ("pepstab/followup3_energy.py", "pepstab/foldx.py", "pepstab/structural_features.py", "pepstab/boltz_embeddings.py", "scripts/followup3_foldx.py", "scripts/boltz_pose_check.py", "modal_app/followup3_energy.py"):
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != protocol["source_sha256"][name]:
            raise ValueError("worker extraction source drift before import")
    from pepstab import followup3_energy as core
    from pepstab.foldx import locate_foldx
    from pepstab.boltz_embeddings import sha256_file
    rows = source_root / "reports/followup3/energy/energy_scoring" / protocol["protocol_hash"]
    rows.mkdir(parents=True, exist_ok=True)
    key = core.digest([identity["allele"], identity["peptide"]])
    completed, attempted = rows / (key + ".json"), rows / (key + ".started.json")
    access_marker = rows / "STOP_ACCESS.json"
    outputs.reload()
    if completed.exists():
        cached = json.loads(completed.read_text())
        if cached["protocol_hash"] != protocol["protocol_hash"] or any(str(cached[k]) != str(v) for k, v in identity.items()):
            raise ValueError("completed record identity differs")
        folder = Path("/structures") / identity["folder"]
        cifs = list(folder.glob("*.cif"))
        if len(cifs) != 1 or sha256_file(cifs[0]) != cached["structure_sha256"] or sha256_file(folder / "metadata.json") != cached["metadata_sha256"]:
            raise ValueError("completed/pilot structure provenance differs from mounted source")
        return dict(allele=identity["allele"], peptide=identity["peptide"], status=cached["status"], cached=True)
    if attempted.exists():
        return dict(allele=identity["allele"], peptide=identity["peptide"], status="interrupted_attempt", cached=False)
    if core.utc_deadline(deadline_utc) <= datetime.now(timezone.utc).timestamp():
        return dict(allele=identity["allele"], peptide=identity["peptide"], status="global_deadline", cached=False)
    if access_marker.exists():
        return dict(allele=identity["allele"], peptide=identity["peptide"], status="access_rejected", cached=False)
    checked = dict(protocol)
    expected = checked.pop("protocol_hash")
    if core.digest(checked) != expected:
        raise ValueError("worker protocol payload changed")
    distribution = locate_foldx(source_root / "foldx5_Linux_0")
    if distribution["sha256"] != core.BINARY_SHA256:
        raise ValueError("frozen licensed-build binary identity differs")
    with attempted.open("x") as handle:
        json.dump(dict(identity, protocol_hash=protocol["protocol_hash"], deadline_utc=deadline_utc), handle)
    outputs.commit()
    def check_access():
        outputs.reload()
        if access_marker.exists():
            raise PermissionError("shared access rejection;no subsequent command permitted")
    result = core.score_repaired(identity, Path("/structures") / identity["folder"], distribution, deadline_utc,
                                 protocol["specification"]["energy_columns"], access_stop_check=check_access)
    if result["status"] == "access_rejected":
        try:
            with access_marker.open("x") as handle:
                json.dump(dict(protocol_hash=protocol["protocol_hash"], status="access_rejected"), handle)
        except FileExistsError:
            pass
    result["protocol_hash"] = protocol["protocol_hash"]
    with completed.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    outputs.commit()
    return dict(allele=identity["allele"], peptide=identity["peptide"], status=result["status"], cached=False)


@app.local_entrypoint()
def run(protocol_path: str, authorization_path: str):
    """Root must first bind a fresh resource reservation and independent guard."""
    if os.environ.get("MODAL_PROFILE") != "a-cheparukhin":
        raise ValueError("merged primary workspace profile required")
    sys.path.insert(0, str(ROOT))
    from scripts.followup3_energy import verify, sha
    from pepstab import followup3_energy as core
    protocol_file = Path(protocol_path)
    panel, protocol = verify(protocol_file.parent)
    authorization = json.loads(Path(authorization_path).read_text())
    if not authorization.get("approved") or authorization.get("protocol_file_sha256") != sha(protocol_file):
        raise ValueError("exact root extraction authorization required")
    if not authorization.get("independent_stop_guard_bound") or not authorization.get("budget_reservation_id"):
        raise ValueError("root independent stop guard and ledger reservation required")
    if not 0 < authorization["reservation_usd"] <= 40:
        raise ValueError("repaired extraction reservation outside bounded40 envelope")
    deadline = authorization["deadline_utc"]
    seconds = core.utc_deadline(deadline) - datetime.now(timezone.utc).timestamp()
    if not 0 < seconds <= 5 * 3600:
        raise ValueError("fixed UTC launch deadline outside five-hour maximum")
    guard_handoff = protocol_file.parent / "function_app_created.json"
    with guard_handoff.open("x") as handle:
        json.dump(dict(app_id=app.app_id, root_authorization_sha256=sha(authorization_path), protocol_hash=protocol["protocol_hash"],
                       at_utc=datetime.now(timezone.utc).isoformat()), handle, indent=2, allow_nan=False)
        handle.write("\n")
    seeded = Path(protocol_file.parent) / "pilot_energy_rows.json"
    existing = json.loads(seeded.read_text())
    pilot_keys = {(row["allele"], row["peptide"]) for row in existing}
    with outputs.batch_upload() as batch:
        for row in existing:
            key = core.digest([row["allele"], row["peptide"]])
            data = dict(row, protocol_hash=protocol["protocol_hash"])
            import io
            batch.put_file(io.BytesIO((json.dumps(data, allow_nan=False) + "\n").encode()),
                           f"/reports/followup3/energy/energy_scoring/{protocol['protocol_hash']}/{key}.json")
    remaining = [row for row in panel.to_dict(orient="records") if (row["allele"], row["peptide"]) not in pilot_keys]
    # Cached pilot calls only verify unchanged mounted source hashes;they do
    # not execute FoldX. Every original panel identity is accounted for.
    submitted = panel.to_dict(orient="records")
    result = []
    for row in score_one.map(submitted, [protocol] * len(submitted), [deadline] * len(submitted)):
        result.append(row)
        if row["status"] == "access_rejected":
            (protocol_file.parent / "function_access_rejected.json").write_text(json.dumps(dict(app_id=app.app_id, protocol_hash=protocol["protocol_hash"],
                                      status="access_rejected"), indent=2) + "\n")
            raise RuntimeError("shared FoldX access rejected;root guard must stop this app")
    receipt = dict(protocol_hash=protocol["protocol_hash"], root_authorization_sha256=sha(authorization_path),
                   reused_pilot_rows=len(existing), newly_scored_rows=len(remaining), submitted_rows=len(submitted), deadline_utc=deadline, statuses=result)
    (protocol_file.parent / "function_dispatch_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if any(row["status"] != "ok" for row in result):
        raise RuntimeError("incomplete/failed repaired panel;assessment remains blocked")
