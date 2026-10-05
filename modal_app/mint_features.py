"""Phase C §6.2: frozen MINT feature extraction, joint-chain and separate-chain.

    MODAL_PROFILE=a-cheparukhin modal run modal_app/mint_features.py::forecast
    MODAL_PROFILE=a-cheparukhin modal run modal_app/mint_features.py::smoke
    MODAL_PROFILE=a-cheparukhin modal run --detach modal_app/mint_features.py::extract

Gated by `reports/further_research/phase_c_gate.md`: base MINT only
(`varunullanat2012/mint`), which is the one rung of the candidate ladder with no
supervised exposure to peptide-MHC stability or affinity labels.

**Three arms, and what each contrast isolates.** §6.2 asks for frozen
separate-chain against frozen joint-chain under the same head and the same
position-preserving peptide pooling. Adding the project's already-cached ESM-2
650M as a third arm splits the comparison the plan was worried about:

| contrast | weights | chains attend? | isolates |
|---|---|---|---|
| joint vs separate | MINT = MINT | yes vs no | cross-chain context **at inference** |
| separate vs esm2 | MINT vs ESM-2 | no = no | the STRING PPI **pretraining** |
| joint vs esm2 | MINT vs ESM-2 | yes vs no | both at once -- the headline, confounded |

§6.2's stated worry is that pretraining and architecture change together. They
still do in the third row, but the first two rows each hold one of them fixed,
which is more than the plan assumed was available.

**All three arms share one feature contract**: 9 per-position peptide vectors
plus the mean-pooled partner chain, 10 x 1,280 = 12,800 dims. That is exactly
`pepstab.esm.peptide_block(rep="pos")` concatenated with `hla_block(rep="mean")`,
so arm 3 needs no GPU at all -- it is already on disk.

**fp32, per the gate's §5.5.** bf16 would be 3.6x faster and save about $0.30,
but it produces a different feature matrix, and a dtype difference between arms
would be a second uncontrolled difference in a comparison whose whole point is to
control them. The speed-up is for §6.3, where it buys something unaffordable.

Separate-chain extraction is nearly free, and the reason is itself informative: a
separate-chain representation does not depend on the partner, so it is 5,633
peptide passes plus 75 HLA passes rather than 22,533 pair passes. The joint arm
costs 4x more *because* its representation is pair-dependent, which is the
property under test.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from mint_common import (
    GPU,
    MINUTES,
    WEIGHTS,
    WORKER_CPU,
    WORKER_MEM,
    hourly_usd,
    mint_image,
    mint_weights_vol,
    rates,
)

app = modal.App("pepstab-mint-features")

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "further_research"
FEATURES = Path("/mintfeatures")
features_vol = modal.Volume.from_name("pepstab-mint-features", create_if_missing=True)

PROFILES = ("a-cheparukhin", "colleague")
CKPT = WEIGHTS / "mint_state_dict.pt"
LAYER = 33

#: fp32 batch 8, the fastest fp32 cell in the gate's throughput grid
#: (0.05680 s/pair against 0.06251 at batch 1).
BATCH = 8
STEADY_S = 0.0568
LOAD_S = 27.0

#: Stored half-precision: 22,533 x 10 x 1,280 is 1.15 GB in fp32 and 577 MB in
#: fp16, and the egress is paid per GiB. This is a *storage* precision, not a
#: compute one -- every forward pass above is fp32, and the head casts back to
#: fp32 before fitting. Recorded in the manifest so it is never mistaken for a
#: bf16 arm.
STORE_DTYPE = "float16"


def check_profile(profile: str) -> str:
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}; set both to the same value."
    )
    if active is None:
        print(f"warning: MODAL_PROFILE unset; billing the {profile!r} workspace.")
    return profile


def build_inputs() -> dict:
    """Unique peptides, unique HLA domains, and the dev-set pair list.

    Train + val, because §3.2 runs the outer assessment on training clusters and
    the validation split stays available as the development resource it already
    is. The test split is excluded: it was scored once, at stage 6.

    Joined on ``(allele, peptide)`` and filtered on ``split``, per EVALUATION.md.
    Never on ``pair_id``.
    """
    import pandas as pd

    splits = pd.read_csv(REPO / "data" / "splits.csv")
    raw = pd.read_csv(REPO / "data" / "rasmussen_et_al_dataset.csv")
    dev = splits[splits.split.isin(["train", "val"])].merge(
        raw[["allele", "peptide", "hla_seq"]],
        on=["allele", "peptide"], how="left", validate="one_to_one",
    )
    assert dev.hla_seq.notna().all(), "dev rows lost hla_seq in the (allele, peptide) join"

    return {
        "peptides": sorted(dev.peptide.unique().tolist()),
        "hla": dev.drop_duplicates("hla_seq").set_index("hla_seq").index.tolist(),
        "pairs": [
            {"allele": r.allele, "peptide": r.peptide, "hla_seq": r.hla_seq}
            for r in dev.itertuples()
        ],
        "n_rows": len(dev),
    }


@app.function(
    image=mint_image,
    gpu=GPU,
    volumes={WEIGHTS: mint_weights_vol, FEATURES: features_vol},
    timeout=120 * MINUTES,
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    max_containers=1,
    retries=0,
)
def extract_features(inputs: dict, run: str, limit: int = 0,
                     resume: bool = True) -> dict:
    """Write joint-chain and separate-chain frozen MINT features to the Volume.

    Layout deliberately mirrors `pepstab.esm`'s cache -- an index JSON of keys
    plus a `reps` array -- so the separate-chain arm loads through the same code
    path as the ESM-2 arm and the two cannot silently disagree about pooling.
    """
    import numpy as np
    import torch

    from mint.data import Alphabet
    from mint.model.esm2 import ESM2

    alphabet = Alphabet.from_architecture("ESM-1b")
    cfg = _config()

    t0 = time.monotonic()
    model = ESM2(
        num_layers=cfg["encoder_layers"],
        embed_dim=cfg["encoder_embed_dim"],
        attention_heads=cfg["encoder_attention_heads"],
        token_dropout=cfg["token_dropout"],
        use_multimer=True,
    )
    blob = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(
        {k.replace("model.", "", 1): v for k, v in blob["state_dict"].items()}, strict=False
    )
    model.eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    load_s = round(time.monotonic() - t0, 1)

    dest = FEATURES / run
    dest.mkdir(parents=True, exist_ok=True)
    dim = cfg["encoder_embed_dim"]
    store = getattr(np, STORE_DTYPE)
    timings: dict[str, float] = {}

    def enc(seq: str) -> list[int]:
        return list(alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>"))

    def run_batches(rows: list[list[str]], take) -> np.ndarray:
        """Forward `rows` (each a chain list) in equal-length batches, fp32.

        Grouped by token length so a batch never carries pad tokens: the gate
        measured its throughput on a padding-free batch and a padded one would
        not reproduce it.
        """
        encoded = [[enc(s) for s in chains] for chains in rows]
        order = sorted(range(len(encoded)), key=lambda i: sum(len(e) for e in encoded[i]))
        out = None
        for start in range(0, len(order), BATCH):
            group = [order[i] for i in range(start, min(start + BATCH, len(order)))]
            lengths = {sum(len(e) for e in encoded[i]) for i in group}
            if len(lengths) > 1:  # ragged tail: fall back to one row at a time
                group_iter = [[i] for i in group]
            else:
                group_iter = [group]
            for sub in group_iter:
                ids, cids = [], []
                for i in sub:
                    row_ids, row_cids = [], []
                    for c, e in enumerate(encoded[i]):
                        row_ids += e
                        row_cids += [c] * len(e)
                    ids.append(row_ids)
                    cids.append(row_cids)
                tok = torch.tensor(ids, dtype=torch.int64).cuda()
                cid = torch.tensor(cids, dtype=torch.int32).cuda()
                with torch.no_grad():
                    rep = model(tok, cid, repr_layers=[LAYER])["representations"][LAYER]
                real = (
                    ~tok.eq(model.cls_idx) & ~tok.eq(model.eos_idx) & ~tok.eq(model.padding_idx)
                )
                vals = take(rep, real, cid)
                if out is None:
                    out = np.zeros((len(encoded),) + tuple(vals.shape[1:]), dtype=store)
                out[sub] = vals.float().cpu().numpy().astype(store)
        return out

    # --- separate chains: peptide alone, then HLA domain alone ---------------
    def done(name: str) -> bool:
        """Already on the Volume from an earlier, possibly interrupted run.

        The first production attempt was cancelled externally after the
        separate-chain arms had committed. That compute is paid for; redoing it
        would charge the workspace twice for the same array.
        """
        if resume and (dest / f"{name}_L{LAYER}.npz").exists():
            print(f"resume: {name} already on the Volume, skipping", flush=True)
            return True
        return False

    peptides = inputs["peptides"][: limit or None]
    if done("sep_peptide"):
        pep_sep = np.load(dest / f"sep_peptide_L{LAYER}.npz")["reps"]
        timings["separate_peptide_s"] = 0.0
    else:
      t = time.monotonic()
      pep_sep = run_batches(
        [[p] for p in peptides],
        lambda rep, real, cid: torch.stack(
            [rep[i, real[i].nonzero(as_tuple=True)[0], :] for i in range(len(rep))]
        ),
      )
      timings["separate_peptide_s"] = round(time.monotonic() - t, 1)
      np.savez_compressed(dest / f"sep_peptide_L{LAYER}.npz", reps=pep_sep)
      (dest / "sep_peptide.index.json").write_text(json.dumps({"sequences": peptides}))
      features_vol.commit()

    hlas = inputs["hla"][: limit or None]
    if done("sep_hla"):
        hla_sep = np.load(dest / f"sep_hla_L{LAYER}.npz")["reps"]
        timings["separate_hla_s"] = 0.0
    else:
      t = time.monotonic()
      hla_sep = run_batches(
        [[h] for h in hlas],
        lambda rep, real, cid: torch.stack(
            [rep[i, real[i].nonzero(as_tuple=True)[0], :].mean(dim=0) for i in range(len(rep))]
        ),
      )
      timings["separate_hla_s"] = round(time.monotonic() - t, 1)
      np.savez_compressed(dest / f"sep_hla_L{LAYER}.npz", reps=hla_sep)
      (dest / "sep_hla.index.json").write_text(json.dumps({"sequences": hlas}))
      features_vol.commit()

    # --- joint chains: the pair-dependent representation --------------------
    pairs = inputs["pairs"][: (limit * 2) or None]
    keys = [f"{p['allele']}|{p['peptide']}" for p in pairs]

    def take_joint(rep, real, cid):
        rows = []
        for i in range(len(rep)):
            pep_pos = (real[i] & (cid[i] == 0)).nonzero(as_tuple=True)[0]
            par_pos = (real[i] & (cid[i] > 0)).nonzero(as_tuple=True)[0]
            rows.append(
                torch.cat([rep[i, pep_pos, :], rep[i, par_pos, :].mean(dim=0, keepdim=True)])
            )
        return torch.stack(rows)

    # Chunked and committed as it goes. This is the expensive arm -- 22,533
    # pair-dependent passes against the separate arms' 4,584 -- and the first
    # attempt lost all of it to an external cancellation 11 minutes in because it
    # only wrote at the end. A chunk boundary caps the loss at one chunk.
    CHUNK = 4000
    chunk_dir = dest / "_joint_chunks"
    chunk_dir.mkdir(exist_ok=True)
    t = time.monotonic()
    for start in range(0, len(pairs), CHUNK):
        part = chunk_dir / f"joint_{start:06d}.npz"
        if resume and part.exists():
            print(f"resume: joint chunk at {start} present, skipping", flush=True)
            continue
        block = pairs[start : start + CHUNK]
        arr = run_batches([[p["peptide"], p["hla_seq"]] for p in block], take_joint)
        np.savez_compressed(part, reps=arr)
        features_vol.commit()
        print(f"joint {start + len(block)}/{len(pairs)} "
              f"({time.monotonic() - t:.0f}s elapsed)", flush=True)

    joint = np.concatenate(
        [np.load(p)["reps"] for p in sorted(chunk_dir.glob("joint_*.npz"))]
    )
    assert len(joint) == len(pairs), f"{len(joint)} rows from chunks, {len(pairs)} pairs"
    timings["joint_s"] = round(time.monotonic() - t, 1)
    np.savez_compressed(dest / f"joint_L{LAYER}.npz", reps=joint)
    (dest / "joint.index.json").write_text(json.dumps({"keys": keys}))

    manifest = {
        "run": run,
        "checkpoint": "varunullanat2012/mint @ 75b1602c34e9df1a89089a1245f000a51ee3c746",
        "layer": LAYER,
        "compute_dtype": "float32",
        "store_dtype": STORE_DTYPE,
        "batch": BATCH,
        "load_seconds": load_s,
        "timings": timings,
        "shapes": {
            "sep_peptide": list(pep_sep.shape),
            "sep_hla": list(hla_sep.shape),
            "joint": list(joint.shape),
        },
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "seconds_per_joint_pair": round(timings["joint_s"] / max(len(pairs), 1), 5),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    features_vol.commit()
    manifest["files"] = sorted(p.name for p in dest.iterdir())
    manifest["total_mb"] = round(
        sum(p.stat().st_size for p in dest.iterdir() if p.is_file()) / 1e6, 1
    )
    return manifest


def _config() -> dict:
    import urllib.request

    import mint

    root = Path(mint.__file__).resolve().parent
    rel = "data/esm2_t33_650M_UR50D.json"
    for p in (root / rel, root.parent / rel):
        if p.exists():
            return json.loads(p.read_text())
    url = (
        "https://raw.githubusercontent.com/VarunUllanat/mint/"
        "06694b7606e2d00b76ec58daf5c7aecdaf7cd283/" + rel
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _price(n_pairs: int, n_pep: int, n_hla: int) -> dict:
    rate = hourly_usd(GPU, WORKER_CPU, WORKER_MEM)
    # Peptide-alone passes are 11 tokens against the joint pass's 195, so they
    # are charged at a generous tenth of the joint rate rather than the full one.
    gpu_s = LOAD_S + n_pairs * STEADY_S + (n_pep * STEADY_S / 10) + n_hla * STEADY_S + 120
    egress_gib = (n_pairs * 10 + n_pep * 9 + n_hla) * 1280 * 2 / 2**30
    usd = gpu_s / 3600 * rate + egress_gib * rates()["egress_gib_cost"]
    return {
        "gpu_seconds": round(gpu_s),
        "gpu_minutes": round(gpu_s / 60, 1),
        "egress_gib": round(egress_gib, 2),
        "usd": round(usd, 3),
        "with_margin_usd": round(1.25 * usd, 3),
    }


@app.local_entrypoint()
def forecast():
    """Price §6.2 extraction. Spawns no worker."""
    inp = build_inputs()
    p = _price(len(inp["pairs"]), len(inp["peptides"]), len(inp["hla"]))
    print(f"dev rows {inp['n_rows']:,} | unique peptides {len(inp['peptides']):,} | "
          f"unique HLA domains {len(inp['hla'])}")
    print(f"arms: joint ({len(inp['pairs']):,} pair passes), separate "
          f"({len(inp['peptides']):,} peptide + {len(inp['hla'])} HLA passes), "
          f"esm2 (cached, $0)")
    print(f"\nfp32 batch {BATCH} at {STEADY_S} s/pair (gate's fastest fp32 cell)")
    print(f"  {p['gpu_minutes']} min GPU, {p['egress_gib']} GiB egress")
    print(f"  ${p['usd']}, ${p['with_margin_usd']} with the plan's 25% margin")
    print("\nLog this before launching. CLAUDE.md requires ::smoke to pass first.")


@app.local_entrypoint()
def smoke(profile: str = "a-cheparukhin", n: int = 4):
    """End-to-end on a few examples, per CLAUDE.md. Writes to a _smoke run dir."""
    check_profile(profile)
    inp = build_inputs()
    res = extract_features.remote(inp, run="_smoke", limit=n)
    print(json.dumps(res, indent=2, default=str))
    exp_pep, exp_joint = [n, 9, 1280], [n * 2, 10, 1280]
    assert res["shapes"]["sep_peptide"] == exp_pep, (res["shapes"], exp_pep)
    assert res["shapes"]["joint"] == exp_joint, (res["shapes"], exp_joint)
    assert res["shapes"]["sep_hla"][1] == 1280
    print(f"\nSMOKE PASSED -- shapes match the 9 x 1280 / 10 x 1280 contract.")


@app.local_entrypoint()
def extract(profile: str = "a-cheparukhin", run: str = "mint_base"):
    """Full §6.2 extraction over the dev rows."""
    check_profile(profile)
    inp = build_inputs()
    p = _price(len(inp["pairs"]), len(inp["peptides"]), len(inp["hla"]))
    print(f"{inp['n_rows']:,} dev rows, ${p['with_margin_usd']} with margin, "
          f"~{p['gpu_minutes']} min, profile {profile}")

    t0 = time.monotonic()
    res = extract_features.remote(inp, run=run)
    res["wall_clock_seconds"] = round(time.monotonic() - t0, 1)
    res["forecast"] = p
    res["profile"] = profile

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "phase_c_features_manifest.json"
    dest.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(REPO)}")
    print(f"\nFetch the features:\n  MODAL_PROFILE={profile} modal volume get "
          f"pepstab-mint-features {run} features/mint/")
