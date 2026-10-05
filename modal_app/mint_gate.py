"""Phase C gate: checkpoint access, then an extraction/training smoke benchmark.

`reports/FURTHER_RESEARCH_PLAN.md` §6 gates the whole Phase C GPU investment on
one measurement: "perform an extraction/training smoke benchmark for memory,
time and checkpoint access before specifying a full GPU budget." This is that
benchmark. It is not the Phase C comparison, and it reads no validation or test
label -- the 5 example pairs come from the `train` split of `data/splits.csv`.

Run in order; each step is gated on the previous one reporting success:

    # free-ish: imports and the HF manifest, no weights, no GPU
    MODAL_PROFILE=a-cheparukhin modal run modal_app/mint_gate.py::probe

    # ~$0.03: pull 3.25 GB to the Volume, resolve the Lightning unpickle on CPU
    MODAL_PROFILE=a-cheparukhin modal run modal_app/mint_gate.py::setup

    # local only, spawns nothing: print the priced forecast
    modal run modal_app/mint_gate.py::forecast

    # ~$0.20: the gate measurement itself, on one L4
    MODAL_PROFILE=a-cheparukhin modal run modal_app/mint_gate.py::smoke

`CLAUDE.md` forbids a batch GPU job without a passing end-to-end pilot; this
module is that pilot for a Phase C production run, and `::smoke` is the
entrypoint that has to pass before one is specified.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

from mint_common import (
    CONTRACTS,
    GPU,
    MINT_CKPT_FILE,
    MINT_CONFIG,
    MINT_GITHUB_SHA,
    MINT_HF_REPO,
    MINT_HF_REVISION,
    MINUTES,
    SPEARMINT_REPOS,
    WEIGHTS,
    WORKER_CPU,
    WORKER_MEM,
    hourly_usd,
    mint_image,
    mint_weights_vol,
    probe_image,
    rates,
)

app = modal.App("pepstab-mint-gate")

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "further_research"

#: Local Modal profile names. `colleague` points at the `sofyaleyn` workspace.
#: Unlike the folding stages there is no cohort half to get wrong here, but the
#: assertion stays: the gate's spend has to land in the workspace whose ledger
#: entry records it.
PROFILES = ("a-cheparukhin", "colleague")

#: Raw checkpoint, and the clean state dict `::setup` writes beside it so the
#: GPU worker never repeats the Lightning unpickle.
CKPT = WEIGHTS / MINT_CKPT_FILE
CLEAN = WEIGHTS / "mint_state_dict.pt"

#: Forecast inputs, replaced by `::smoke`'s measurements once it has run. These
#: are pre-measurement estimates and the `::forecast` output says so.
EST_LOAD_S = 60.0
EST_EMBED_S_PER_PAIR = 0.05
EST_HEAD_STEP_S = 0.01


def check_profile(profile: str) -> str:
    """Refuse to run if the named profile and the active Modal profile disagree."""
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. "
        f"The gate's spend has to land in the workspace its ledger entry names; "
        f"set both to the same value."
    )
    if active is None:
        print(
            f"warning: MODAL_PROFILE is unset, so this runs against your active "
            f"profile. Billing the {profile!r} workspace -- confirm that is the "
            f"same one."
        )
    return profile


def build_cases(n: int = 5) -> list[dict]:
    """`n` example pairs from the **train** split, with both construct variants.

    Joined on `(allele, peptide)` and filtered on `split`, per `EVALUATION.md`;
    never on `pair_id`, which is positional into the raw CSV. Train only: the
    gate measures seconds and gibibytes, and has no business reading a held-out
    label to do it.

    Deterministic without a seed to defend: the first `n` distinct alleles in
    allele order, one pair each, so the panel spans alleles rather than
    measuring one allele five times.
    """
    import pandas as pd

    splits = pd.read_csv(REPO / "data" / "splits.csv")
    raw = pd.read_csv(REPO / "data" / "rasmussen_et_al_dataset.csv")
    ecto = pd.read_csv(REPO / "hla75_ectodomain_b2m.csv")

    train = splits[splits.split == "train"].merge(
        raw[["allele", "peptide", "hla_seq", "hla_pseudoseq"]],
        on=["allele", "peptide"],
        how="left",
        validate="one_to_one",
    )
    assert train.hla_seq.notna().all(), "train rows lost hla_seq in the (allele, peptide) join"

    picked = (
        train.sort_values(["allele", "peptide"])
        .groupby("allele", as_index=False)
        .first()
        .sort_values("allele")
        .head(n)
    )
    ecto_by_allele = ecto.set_index("allele")

    cases = []
    for row in picked.itertuples():
        case = {
            "allele": row.allele,
            "peptide": row.peptide,
            "cluster_id": int(row.cluster_id),
            "split": row.split,
            "chains_pep_domain": [row.peptide, row.hla_seq],
        }
        if row.allele in ecto_by_allele.index:
            e = ecto_by_allele.loc[row.allele]
            case["chains_pep_ectodomain_b2m"] = [row.peptide, e.ecto, e.b2m]
        cases.append(case)
    assert cases, "no train cases built"
    return cases


# --------------------------------------------------------------------------
# step 1 -- probe. No weights, no GPU, 0.25 cores.
# --------------------------------------------------------------------------
@app.function(
    image=probe_image,
    timeout=10 * MINUTES,
    cpu=0.25,
    memory=2048,
    max_containers=1,
    retries=0,
)
def probe_imports() -> dict:
    """Report the real API surface and the weight manifest. Loads no weights.

    Answers, cheapest-to-establish first, the three things that can each cost a
    GPU container:

    1. Does `import mint` work from a pip-install of the pinned commit, with
       only torch in the image? The published install path is a conda
       environment pinned to python 3.7.12 and pytorch 1.12.1, which we are not
       reproducing.
    2. Does the architecture config ship inside the installed package, or only
       in the git tree? `MINTWrapper` needs `data/esm2_t33_650M_UR50D.json`, and
       `setup.py` has no `install_requires` and relies on
       `include_package_data`, so this is not safe to assume.
    3. Is a per-residue representation reachable? `MINTWrapper.forward` returns
       mean-pooled embeddings only, but Phase C §6.2 requires
       position-preserving peptide pooling. If `ESM2.forward` does not accept
       `repr_layers`, the whole feature contract has to change.
    """
    out: dict = {}

    try:
        import torch

        out["torch_version"] = torch.__version__
    except Exception as exc:
        return {"fatal": f"import torch failed: {exc!r}"}

    try:
        import mint

        out["mint_version"] = getattr(mint, "__version__", "unknown")
        out["mint_path"] = getattr(mint, "__file__", "unknown")
    except Exception as exc:
        return {"fatal": f"import mint failed: {exc!r}"}

    import inspect

    try:
        from mint.helpers.extract import CollateFn, MINTWrapper, load_config
        from mint.model.esm2 import ESM2

        out["sig::MINTWrapper"] = str(inspect.signature(MINTWrapper))
        out["sig::MINTWrapper.forward"] = str(inspect.signature(MINTWrapper.forward))
        out["sig::ESM2"] = str(inspect.signature(ESM2))
        out["sig::ESM2.forward"] = str(inspect.signature(ESM2.forward))
        out["esm2_accepts_repr_layers"] = (
            "repr_layers" in inspect.signature(ESM2.forward).parameters
        )
        out["sig::CollateFn.__call__"] = str(inspect.signature(CollateFn.__call__))
    except Exception as exc:
        out["fatal"] = f"import mint.helpers.extract failed: {exc!r}"
        return out

    # Whether the config rides along in the wheel, and if not, that we can still
    # reach it from the pinned raw URL. Either answer is fine; not knowing is not.
    pkg_root = Path(mint.__file__).resolve().parent
    candidates = [
        pkg_root / MINT_CONFIG,
        pkg_root.parent / MINT_CONFIG,
    ]
    found = [str(p) for p in candidates if p.exists()]
    out["config_in_package"] = found
    if not found:
        try:
            import urllib.request

            url = (
                f"https://raw.githubusercontent.com/VarunUllanat/mint/"
                f"{MINT_GITHUB_SHA}/{MINT_CONFIG}"
            )
            with urllib.request.urlopen(url, timeout=30) as r:
                cfg = json.loads(r.read())
            out["config_from_raw_url"] = {
                k: cfg[k]
                for k in ("encoder_layers", "encoder_embed_dim", "encoder_attention_heads")
            }
        except Exception as exc:
            out["config_fetch_error"] = repr(exc)

    # The alphabet is the only other thing the collate path needs, and a wrong
    # architecture name silently changes the token ids.
    try:
        from mint.data import Alphabet

        alpha = Alphabet.from_architecture("ESM-1b")
        out["alphabet"] = {
            "size": len(alpha),
            "cls_idx": alpha.cls_idx,
            "eos_idx": alpha.eos_idx,
            "padding_idx": alpha.padding_idx,
        }
    except Exception as exc:
        out["alphabet_error"] = repr(exc)

    # Weight manifests for the whole candidate ladder, so the gate report's
    # access table is produced by a call and not by a claim. The three SPEARMINT
    # repos are recorded, not downloaded: the leakage audit disqualifies them.
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        ladder = {MINT_HF_REPO: MINT_HF_REVISION, **SPEARMINT_REPOS}
        out["hf"] = {}
        for repo, revision in ladder.items():
            info = api.model_info(repo, revision=revision, files_metadata=True)
            out["hf"][repo] = {
                "revision_requested": revision,
                "sha": info.sha,
                "revision_matches": info.sha == revision,
                "license": (info.cardData or {}).get("license"),
                "gated": info.gated,
                "private": info.private,
                "total_gb": round(
                    sum((s.size or 0) for s in (info.siblings or [])) / 1e9, 3
                ),
                "files": sorted(
                    (s.rfilename, round((s.size or 0) / 1e9, 3))
                    for s in (info.siblings or [])
                ),
            }
    except Exception as exc:
        out["hf_manifest_error"] = repr(exc)

    return out


# --------------------------------------------------------------------------
# step 2 -- setup. Pull weights, resolve the unpickle, write a clean state dict.
# --------------------------------------------------------------------------
@app.function(
    image=probe_image,
    volumes={WEIGHTS: mint_weights_vol},
    timeout=60 * MINUTES,
    cpu=4.0,  # the download is throughput-bound; the load is memory-bound
    memory=16384,
    max_containers=1,
    retries=0,
)
def fetch_weights(force: bool = False) -> dict:
    """Snapshot MINT to the Volume, then de-risk the load on a CPU worker.

    `mint.ckpt` is a PyTorch Lightning checkpoint and `MINTWrapper` opens it
    with `torch.load(..., weights_only=False)`. Whether that unpickles in an
    image without `lightning` installed depends on what Lightning serialised
    into the file, which no manifest reports. Finding out on a GPU costs the
    card's rate per attempt, so it is established here, where the worker is
    CPU-only, and the resulting `state_dict` is written back to the Volume as
    plain tensors. The GPU step then loads that and cannot hit the problem.
    """
    import torch
    from huggingface_hub import hf_hub_download

    result: dict = {"attempts": []}
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    if CKPT.exists() and not force:
        result["download"] = {"skipped": True, "gb": round(CKPT.stat().st_size / 1e9, 3)}
    else:
        t0 = time.monotonic()
        path = hf_hub_download(
            MINT_HF_REPO,
            MINT_CKPT_FILE,
            revision=MINT_HF_REVISION,
            local_dir=str(WEIGHTS),
        )
        mint_weights_vol.commit()
        result["download"] = {
            "path": path,
            "seconds": round(time.monotonic() - t0, 1),
            "gb": round(Path(path).stat().st_size / 1e9, 3),
        }

    # Ladder of load strategies, cheapest and safest first. Whichever one works
    # is recorded, so the GPU step and any later Phase C run inherit a measured
    # fact rather than a guess.
    state = None
    for label, loader in (
        ("weights_only=True", lambda: torch.load(CKPT, map_location="cpu", weights_only=True)),
        ("weights_only=False", lambda: torch.load(CKPT, map_location="cpu", weights_only=False)),
    ):
        t0 = time.monotonic()
        try:
            obj = loader()
            result["attempts"].append(
                {"strategy": label, "ok": True, "seconds": round(time.monotonic() - t0, 1)}
            )
            state = obj
            break
        except Exception as exc:
            result["attempts"].append(
                {"strategy": label, "ok": False, "error": repr(exc)[:400]}
            )

    if state is None:
        result["fatal"] = "no torch.load strategy succeeded; do not start the GPU step"
        return result

    result["top_level_keys"] = sorted(k for k in state)[:20] if isinstance(state, dict) else None
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
    result["state_dict"] = {
        "entries": len(sd),
        "tensor_entries": len(tensors),
        "params_millions": round(sum(v.numel() for v in tensors.values()) / 1e6, 1),
        "dtypes": sorted({str(v.dtype) for v in tensors.values()}),
        "sample_keys": sorted(tensors)[:5],
        # Confirms the cross-chain module is actually in the weights and we are
        # not about to benchmark a plain ESM-2 with extra steps.
        "cross_chain_keys": sorted(
            k for k in tensors if "cross" in k.lower() or "multimer" in k.lower()
        )[:5],
        "cross_chain_key_count": sum(
            1 for k in tensors if "cross" in k.lower() or "multimer" in k.lower()
        ),
    }

    torch.save({"state_dict": tensors}, CLEAN)
    mint_weights_vol.commit()
    result["clean_state_dict"] = {
        "path": str(CLEAN),
        "gb": round(CLEAN.stat().st_size / 1e9, 3),
    }

    # Verify the architecture instantiates and the weights fit it, still on CPU.
    try:
        from mint.model.esm2 import ESM2

        cfg = _config()
        model = ESM2(
            num_layers=cfg["encoder_layers"],
            embed_dim=cfg["encoder_embed_dim"],
            attention_heads=cfg["encoder_attention_heads"],
            token_dropout=cfg["token_dropout"],
            use_multimer=True,
        )
        stripped = {k.replace("model.", "", 1): v for k, v in tensors.items()}
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        result["load_state_dict"] = {
            "missing": list(missing)[:10],
            "n_missing": len(missing),
            "unexpected": list(unexpected)[:10],
            "n_unexpected": len(unexpected),
            "model_params_millions": round(
                sum(p.numel() for p in model.parameters()) / 1e6, 1
            ),
        }
    except Exception as exc:
        result["load_state_dict_error"] = repr(exc)[:400]

    return result


def _config() -> dict:
    """The ESM-2 650M architecture config, from the package or the pinned commit."""
    import mint

    pkg_root = Path(mint.__file__).resolve().parent
    for p in (pkg_root / MINT_CONFIG, pkg_root.parent / MINT_CONFIG):
        if p.exists():
            return json.loads(p.read_text())
    import urllib.request

    url = f"https://raw.githubusercontent.com/VarunUllanat/mint/{MINT_GITHUB_SHA}/{MINT_CONFIG}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------
# step 3 -- the gate measurement, on one GPU.
# --------------------------------------------------------------------------
@app.function(
    image=mint_image,
    gpu=GPU,
    volumes={WEIGHTS: mint_weights_vol},
    timeout=30 * MINUTES,
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    max_containers=1,
    retries=0,
)
def bench(cases: list[dict], head_steps: int = 50, adapt_steps: int = 10) -> dict:
    """Measure what §6's gate asks for: memory, wall time, per-pair cost.

    Three regimes, because they have different budgets and Phase C needs all
    three priced separately:

    * **frozen extraction** (§6.2) -- one forward pass per pair, no grad. This
      is the per-pair cost that scales with 28,166 rows.
    * **downstream head training** (§6.2) -- the small head on cached features.
      Cheap, but it is the step that runs once per fold per seed per config, so
      its per-step time multiplies by the search budget.
    * **limited adaptation** (§6.3) -- top-layer fine-tuning, forward and
      backward through the encoder. This is the regime that sets the real Phase
      C ceiling and the one that can exceed the card, so its peak memory is the
      number that decides whether L4 is the right shape.
    """
    import torch
    import torch.nn as nn

    out: dict = {"gpu": GPU, "torch": torch.__version__, "cases": len(cases)}
    out["cuda"] = {
        "device": torch.cuda.get_device_name(0),
        "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    }

    from mint.data import Alphabet
    from mint.model.esm2 import ESM2

    alphabet = Alphabet.from_architecture("ESM-1b")
    cfg = _config()

    # --- weight load ------------------------------------------------------
    t0 = time.monotonic()
    model = ESM2(
        num_layers=cfg["encoder_layers"],
        embed_dim=cfg["encoder_embed_dim"],
        attention_heads=cfg["encoder_attention_heads"],
        token_dropout=cfg["token_dropout"],
        use_multimer=True,
    )
    src = CLEAN if CLEAN.exists() else CKPT
    blob = torch.load(src, map_location="cpu", weights_only=CLEAN.exists())
    tensors = blob.get("state_dict", blob)
    model.load_state_dict(
        {k.replace("model.", "", 1): v for k, v in tensors.items()}, strict=False
    )
    model.eval().cuda()
    torch.cuda.synchronize()
    out["load"] = {
        "source": src.name,
        "seconds": round(time.monotonic() - t0, 1),
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "weights_gib": round(
            sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30, 2
        ),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }

    def tokenize(chains: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """One case to (tokens, chain_ids). Mirrors `CollateFn`, batch of one.

        Written out rather than reused because `CollateFn` takes a batch of
        same-arity cases and we deliberately time a 2-chain and a 3-chain
        contract in the same run.
        """
        ids, cids = [], []
        for i, seq in enumerate(chains):
            enc = alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>")
            ids += list(enc)
            cids += [i] * len(enc)
        return (
            torch.tensor([ids], dtype=torch.int64).cuda(),
            torch.tensor([cids], dtype=torch.int32).cuda(),
        )

    layer = cfg["encoder_layers"]

    # --- frozen extraction, per contract ----------------------------------
    out["extraction"] = {}
    cached: dict[str, list] = {}
    for contract in CONTRACTS:
        key = f"chains_{contract}"
        usable = [c for c in cases if key in c]
        if not usable:
            out["extraction"][contract] = {"skipped": "no case carries this contract"}
            continue

        torch.cuda.reset_peak_memory_stats()
        # One untimed warm-up: the first CUDA kernel launch of a shape pays
        # autotuning that the per-pair steady cost should not be charged for.
        with torch.no_grad():
            t, c = tokenize(usable[0][key])
            model(t, c, repr_layers=[layer])
        torch.cuda.synchronize()

        per_pair, feats = [], []
        for case in usable:
            t, c = tokenize(case[key])
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.no_grad():
                rep = model(t, c, repr_layers=[layer])["representations"][layer]
            torch.cuda.synchronize()
            per_pair.append(time.monotonic() - t0)

            # Position-preserving peptide pooling, per §6.2: the 9 peptide
            # positions kept separately, the partner chains mean-pooled. Taken
            # from the per-residue representation, which `MINTWrapper.forward`
            # does not expose -- it mean-pools every chain.
            #
            # Indexed off the special-token mask rather than arithmetic on
            # lengths: `CollateFn` wraps *each* chain as `<cls>seq<eos>`, so the
            # layout is [cls, pep, eos, cls, hla, eos, ...] and a boundary costs
            # two tokens, not one. Guessing that wrong silently shifts every
            # position by one and would make the whole comparison meaningless.
            real = (
                ~t.eq(model.cls_idx) & ~t.eq(model.eos_idx) & ~t.eq(model.padding_idx)
            )
            pep_pos = (real[0] & (c[0] == 0)).nonzero(as_tuple=True)[0]
            partner_pos = (real[0] & (c[0] > 0)).nonzero(as_tuple=True)[0]
            assert len(pep_pos) == len(case["peptide"]), (
                len(pep_pos), len(case["peptide"]), case["allele"]
            )
            pep = rep[0, pep_pos, :]
            partner = rep[0, partner_pos, :].mean(dim=0)
            feats.append(torch.cat([pep.flatten(), partner.flatten()]).float().cpu())

        cached[contract] = feats
        out["extraction"][contract] = {
            "pairs": len(usable),
            "tokens": int(tokenize(usable[0][key])[0].shape[1]),
            "seconds_per_pair_median": round(sorted(per_pair)[len(per_pair) // 2], 4),
            "seconds_per_pair_max": round(max(per_pair), 4),
            "feature_dim": int(feats[0].numel()),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

    # --- downstream head training ----------------------------------------
    # The §6.2 "same small downstream head": one hidden layer on the
    # position-preserving features. Trained on 5 pairs, which measures per-step
    # cost and nothing about fit -- the loss is reported so a silent no-op
    # (identical loss, dead gradient) is visible, not as a result.
    contract = CONTRACTS[0]
    if contract in cached:
        X = torch.stack(cached[contract]).cuda()
        y = torch.zeros(len(X), 1).cuda()
        head = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
        opt = torch.optim.Adam(head.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        losses = []
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(head_steps):
            opt.zero_grad()
            loss = nn.functional.mse_loss(head(X), y)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        out["head_training"] = {
            "contract": contract,
            "params_millions": round(sum(p.numel() for p in head.parameters()) / 1e6, 3),
            "steps": head_steps,
            "seconds_total": round(elapsed, 3),
            "seconds_per_step": round(elapsed / head_steps, 5),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "loss_first": round(losses[0], 6),
            "loss_last": round(losses[-1], 6),
        }

    # --- limited adaptation ----------------------------------------------
    # §6.3 compares frozen against limited adaptation of the joint encoder.
    # That regime holds activations for a backward pass through the top layers,
    # so it is the one that can exhaust the card. Measured here at the smallest
    # realistic setting -- top 4 of 33 layers, batch of 5 -- so the forecast
    # scales from a measurement instead of an assumption.
    if contract in cached:
        unfrozen = 4
        for name, p in model.named_parameters():
            parts = name.split(".")
            keep = len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit() and int(parts[1]) >= layer - unfrozen
            p.requires_grad_(bool(keep))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        usable = [c for c in cases if f"chains_{contract}" in c]
        head2 = nn.Sequential(nn.Linear(cfg["encoder_embed_dim"], 1)).cuda()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad] + list(head2.parameters()),
            lr=1e-5,
        )
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        oom = None
        try:
            for _ in range(adapt_steps):
                opt.zero_grad()
                total = 0.0
                for case in usable:
                    t, c = tokenize(case[f"chains_{contract}"])
                    rep = model(t, c, repr_layers=[layer])["representations"][layer]
                    real = (
                        ~t.eq(model.cls_idx)
                        & ~t.eq(model.eos_idx)
                        & ~t.eq(model.padding_idx)
                    )
                    pep_pos = (real[0] & (c[0] == 0)).nonzero(as_tuple=True)[0]
                    pred = head2(rep[0, pep_pos, :].mean(dim=0))
                    total = total + nn.functional.mse_loss(pred, torch.zeros_like(pred))
                (total / len(usable)).backward()
                opt.step()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            oom = repr(exc)[:300]
        elapsed = time.monotonic() - t0
        out["limited_adaptation"] = {
            "contract": contract,
            "unfrozen_top_layers": unfrozen,
            "trainable_params_millions": round(trainable / 1e6, 1),
            "batch_pairs": len(usable),
            "steps": adapt_steps,
            "oom": oom,
            "seconds_total": round(elapsed, 3),
            "seconds_per_step": round(elapsed / adapt_steps, 4) if not oom else None,
            "seconds_per_pair_step": (
                round(elapsed / (adapt_steps * len(usable)), 4) if not oom else None
            ),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

    out["host_peak_gib"] = _cgroup_peak()
    return out


@app.function(
    image=mint_image,
    gpu=GPU,
    volumes={WEIGHTS: mint_weights_vol},
    timeout=30 * MINUTES,
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    max_containers=1,
    retries=0,
)
def throughput(cases: list[dict], batches: tuple[int, ...] = (1, 8, 16, 32), steps: int = 10) -> dict:
    """§5.2 probe 1: what batching and bf16 are actually worth.

    The gate's adaptation figure was 97.2 ms per pair-step at **batch 1 in
    fp32**, because the smoke looped pairs inside a step. That measures the
    harness as much as the model, and a $238 no-go resting on the least
    favourable configuration available is a decision resting on an artefact.
    This probe replaces the artefact with a grid.

    Every `pep_domain` construct is exactly 195 tokens -- all peptides are
    9-mers and `hla_seq` is uniformly 182 residues -- so a batch needs no
    padding and the per-pair cost is not diluted by pad tokens. That is a
    property of this dataset, not a general one, and it is why batching here
    should convert almost fully into throughput.
    """
    import torch
    import torch.nn as nn

    from mint.data import Alphabet
    from mint.model.esm2 import ESM2

    alphabet = Alphabet.from_architecture("ESM-1b")
    cfg = _config()
    layer = cfg["encoder_layers"]

    t0 = time.monotonic()
    model = ESM2(
        num_layers=cfg["encoder_layers"],
        embed_dim=cfg["encoder_embed_dim"],
        attention_heads=cfg["encoder_attention_heads"],
        token_dropout=cfg["token_dropout"],
        use_multimer=True,
    )
    blob = torch.load(CLEAN, map_location="cpu", weights_only=True)
    model.load_state_dict(
        {k.replace("model.", "", 1): v for k, v in blob["state_dict"].items()}, strict=False
    )
    model.cuda()
    torch.cuda.synchronize()

    out: dict = {
        "gpu": GPU,
        "device": torch.cuda.get_device_name(0),
        "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "load_seconds": round(time.monotonic() - t0, 1),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cells": [],
    }

    key = "chains_pep_domain"
    usable = [c for c in cases if key in c]
    lengths = {sum(len(s) for s in c[key]) for c in usable}
    assert len(lengths) == 1, f"ragged constructs would need padding: {lengths}"
    out["residues_per_pair"] = lengths.pop()

    def batch_of(k: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """`k` pairs as one padding-free batch, cycling the panel if short."""
        picked = [usable[i % len(usable)][key] for i in range(k)]
        ids, cids = [], []
        for chains in picked:
            row_ids, row_cids = [], []
            for i, seq in enumerate(chains):
                enc = alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>")
                row_ids += list(enc)
                row_cids += [i] * len(enc)
            ids.append(row_ids)
            cids.append(row_cids)
        return (
            torch.tensor(ids, dtype=torch.int64).cuda(),
            torch.tensor(cids, dtype=torch.int32).cuda(),
            len(ids[0]),
        )

    unfrozen = 4
    head = nn.Linear(cfg["encoder_embed_dim"], 1).cuda()

    for regime in ("frozen_extraction", "adaptation_top4"):
        for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            for bs in batches:
                cell = {
                    "regime": regime,
                    "dtype": dtype_name,
                    "batch": bs,
                    "steps": steps,
                }
                try:
                    tokens, chain_ids, n_tok = batch_of(bs)
                    cell["tokens_per_row"] = n_tok

                    if regime == "frozen_extraction":
                        model.eval()
                        for p in model.parameters():
                            p.requires_grad_(False)
                        opt = None
                    else:
                        model.train()
                        for name, p in model.named_parameters():
                            parts = name.split(".")
                            keep = (
                                len(parts) > 1
                                and parts[0] == "layers"
                                and parts[1].isdigit()
                                and int(parts[1]) >= layer - unfrozen
                            )
                            p.requires_grad_(bool(keep))
                        opt = torch.optim.Adam(
                            [p for p in model.parameters() if p.requires_grad]
                            + list(head.parameters()),
                            lr=1e-5,
                        )

                    def one_step() -> None:
                        ctx = (
                            torch.autocast("cuda", dtype=dtype)
                            if dtype is torch.bfloat16
                            else torch.autocast("cuda", enabled=False)
                        )
                        if opt is None:
                            with torch.no_grad(), ctx:
                                model(tokens, chain_ids, repr_layers=[layer])
                        else:
                            opt.zero_grad()
                            with ctx:
                                rep = model(tokens, chain_ids, repr_layers=[layer])[
                                    "representations"
                                ][layer]
                                pred = head(rep[:, 1 : 1 + 9, :].mean(dim=1))
                                loss = nn.functional.mse_loss(
                                    pred.float(), torch.zeros_like(pred, dtype=torch.float32)
                                )
                            loss.backward()
                            opt.step()

                    torch.cuda.reset_peak_memory_stats()
                    one_step()  # untimed warm-up: first launch of a shape autotunes
                    torch.cuda.synchronize()
                    t = time.monotonic()
                    for _ in range(steps):
                        one_step()
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - t

                    cell["seconds_per_step"] = round(elapsed / steps, 4)
                    cell["seconds_per_pair"] = round(elapsed / (steps * bs), 5)
                    cell["pairs_per_second"] = round(steps * bs / elapsed, 1)
                    cell["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                except torch.cuda.OutOfMemoryError:
                    cell["oom"] = True
                    torch.cuda.empty_cache()
                except Exception as exc:  # a failed cell must not lose the others
                    cell["error"] = repr(exc)[:300]
                    torch.cuda.empty_cache()
                out["cells"].append(cell)

    return out


def _cgroup_peak() -> float | None:
    for name in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        p = Path(name)
        if p.exists():
            return round(int(p.read_text()) / 1e9, 2)
    return None


# --------------------------------------------------------------------------
# local entrypoints
# --------------------------------------------------------------------------
def _price(load_s: float, embed_s: float, head_step_s: float, n: int) -> dict:
    """Priced forecast for the gate's own spend, from the rate card."""
    gpu_hr = hourly_usd(GPU, WORKER_CPU, WORKER_MEM)
    cpu_hr = hourly_usd(None, 4.0, 16384)
    r = rates()

    # Two contracts x (1 warm-up + n pairs), head training, adaptation, plus
    # container startup. Generous: the point is a ceiling, not a tight estimate.
    gpu_s = load_s + 2 * (n + 1) * embed_s + 50 * head_step_s + 10 * n * 20 * embed_s + 120
    probe_s, setup_s = 180.0, 600.0
    gb_month = 3.3 * 2  # raw ckpt + clean state dict, on the Volume

    items = {
        "probe (0.25 cpu, 2 GiB)": probe_s / 3600 * hourly_usd(None, 0.25, 2048),
        "setup (4 cpu, 16 GiB, 3.3 GB pull + CPU load)": setup_s / 3600 * cpu_hr,
        f"smoke ({GPU}, {WORKER_CPU} cpu, {WORKER_MEM // 1024} GiB)": gpu_s / 3600 * gpu_hr,
        "volume storage, one month": gb_month * r["volume_storage_gib_month_cost"],
    }
    subtotal = sum(items.values())
    return {
        "items": {k: round(v, 4) for k, v in items.items()},
        "gpu_hourly_usd": round(gpu_hr, 4),
        "cpu_hourly_usd": round(cpu_hr, 4),
        "gpu_seconds": round(gpu_s),
        "subtotal_usd": round(subtotal, 3),
        # The plan's 25% operational allowance, not a statistical guarantee.
        "with_margin_usd": round(1.25 * subtotal, 3),
    }


@app.local_entrypoint()
def forecast(n: int = 5):
    """Price the gate before launching it.

    Spawns no worker. It does trigger the image build on first run, because
    Modal builds an app's images at app creation; that is a few cents once and
    is counted in the `probe` line below.
    """
    cases = build_cases(n)
    print(f"{len(cases)} train-split cases, alleles: {[c['allele'] for c in cases]}")
    print(f"clusters: {[c['cluster_id'] for c in cases]} (train only, no val/test label read)")
    for contract in CONTRACTS:
        have = [c for c in cases if f"chains_{contract}" in c]
        if have:
            lens = [sum(len(s) for s in c[f"chains_{contract}"]) for c in have]
            print(f"  {contract}: {len(have)} cases, {min(lens)}-{max(lens)} residues")

    p = _price(EST_LOAD_S, EST_EMBED_S_PER_PAIR, EST_HEAD_STEP_S, n)
    print("\nPRE-MEASUREMENT forecast (estimates, not measurements):")
    for k, v in p["items"].items():
        print(f"  {v:>8.4f}  {k}")
    print(f"  {'-' * 8}")
    print(f"  {p['subtotal_usd']:>8.3f}  subtotal")
    print(f"  {p['with_margin_usd']:>8.3f}  with the plan's 25% margin")
    print(f"\n{GPU} container rate ${p['gpu_hourly_usd']}/h, CPU container ${p['cpu_hourly_usd']}/h")
    print("Gate ceiling is $5. Log this figure in reports/compute_ledger.csv before launching.")


@app.local_entrypoint()
def smoke(profile: str = "a-cheparukhin", n: int = 5):
    """The gate measurement. Writes reports/further_research/phase_c_smoke.json."""
    check_profile(profile)
    cases = build_cases(n)
    p = _price(EST_LOAD_S, EST_EMBED_S_PER_PAIR, EST_HEAD_STEP_S, n)
    print(f"{len(cases)} train cases, ~${p['with_margin_usd']} forecast with margin, profile {profile}")

    t0 = time.monotonic()
    res = bench.remote(cases)
    res["wall_clock_seconds"] = round(time.monotonic() - t0, 1)
    res["profile"] = profile
    res["forecast"] = p
    res["checkpoint"] = {
        "hf_repo": MINT_HF_REPO,
        "hf_revision": MINT_HF_REVISION,
        "github": f"{MINT_GITHUB_SHA}",
    }
    res["panel"] = [
        {k: c[k] for k in ("allele", "peptide", "cluster_id", "split")} for c in cases
    ]

    # Realised GPU cost from the measured seconds, so the ledger entry is a
    # measurement and not the forecast repeated.
    gpu_s = res.get("load", {}).get("seconds", 0)
    for block in ("extraction", "head_training", "limited_adaptation"):
        b = res.get(block) or {}
        vals = b.values() if block == "extraction" else [b]
        for v in vals:
            if isinstance(v, dict):
                gpu_s += v.get("seconds_total", 0) or 0
                gpu_s += (v.get("seconds_per_pair_median", 0) or 0) * (v.get("pairs", 0) or 0)
    res["realised_gpu_seconds_measured"] = round(gpu_s, 1)
    res["realised_gpu_usd_lower_bound"] = round(
        gpu_s / 3600 * hourly_usd(GPU, WORKER_CPU, WORKER_MEM), 4
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "phase_c_smoke.json"
    dest.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(REPO)}")


@app.local_entrypoint()
def probe():
    """Step 1. Imports and manifests; no weights, no GPU."""
    info = probe_imports.remote()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "phase_c_probe.json"
    dest.write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(json.dumps(info, indent=2, default=str))
    if info.get("fatal"):
        print("\nPROBE FAILED -- resolve this before ::setup, and before any GPU.")
        sys.exit(1)
    print(f"\nwrote {dest.relative_to(REPO)}\nNext: ::setup")


@app.local_entrypoint()
def setup(force: bool = False, profile: str = "a-cheparukhin"):
    """Step 2. Pull weights and resolve the Lightning unpickle on CPU."""
    check_profile(profile)
    res = fetch_weights.remote(force=force)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "phase_c_setup.json"
    dest.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps(res, indent=2, default=str))
    if res.get("fatal"):
        print("\nSETUP FAILED -- do not start ::smoke.")
        sys.exit(1)
    print(f"\nwrote {dest.relative_to(REPO)}\nNext: ::forecast, then ::smoke")


@app.local_entrypoint()
def throughput_probe(profile: str = "a-cheparukhin", n: int = 32):
    """§5.2 probe 1. Replaces the gate's batch-1 fp32 adaptation figure with a grid.

    Priced from the gate's own measurement: 16 cells x 11 steps (10 timed plus a
    warm-up) at the measured batch-1 rates is an upper bound, since every cell
    with batch > 1 is faster per pair than the batch-1 cell it is bounded by.
    """
    check_profile(profile)
    cases = build_cases(n)

    smoke = OUT_DIR / "phase_c_smoke.json"
    adapt_s = 0.0972
    if smoke.exists():
        adapt_s = json.loads(smoke.read_text())["limited_adaptation"]["seconds_per_pair_step"]

    # Upper bound: charge every cell at the batch-1 adaptation rate, which
    # over-states the frozen cells and every batched cell.
    pair_steps = sum(b * 11 for b in (1, 8, 16, 32)) * 4  # 4 regime x dtype combos
    gpu_s = EST_LOAD_S + pair_steps * adapt_s + 120
    usd = gpu_s / 3600 * hourly_usd(GPU, WORKER_CPU, WORKER_MEM)
    print(f"{len(cases)} train pairs; {pair_steps} pair-steps charged at {adapt_s} s (upper bound)")
    print(f"forecast ${usd:.3f}, ${1.25 * usd:.3f} with the plan's 25% margin, ~{gpu_s / 60:.0f} min")

    res = throughput.remote(cases)
    res["profile"] = profile
    res["forecast_usd_with_margin"] = round(1.25 * usd, 3)

    dest = OUT_DIR / "phase_c_throughput.json"
    dest.write_text(json.dumps(res, indent=2, default=str) + "\n")

    import collections

    best = collections.defaultdict(dict)
    print(f"\n{res['device']}, bf16 supported: {res['bf16_supported']}, "
          f"{res['residues_per_pair']} residues/pair\n")
    print(f"{'regime':<20} {'dtype':<6} {'batch':>5} {'s/step':>9} {'s/pair':>9} "
          f"{'pairs/s':>8} {'peak GiB':>9}")
    for c in res["cells"]:
        if "seconds_per_pair" in c:
            print(f"{c['regime']:<20} {c['dtype']:<6} {c['batch']:>5} "
                  f"{c['seconds_per_step']:>9.4f} {c['seconds_per_pair']:>9.5f} "
                  f"{c['pairs_per_second']:>8.1f} {c['peak_gib']:>9.2f}")
            best[c["regime"]][c["seconds_per_pair"]] = c
        else:
            print(f"{c['regime']:<20} {c['dtype']:<6} {c['batch']:>5}   "
                  f"{'OOM' if c.get('oom') else 'ERROR'}")

    print()
    for regime, cells in best.items():
        fastest = cells[min(cells)]
        baseline = next(
            (c for c in res["cells"] if c["regime"] == regime and c["dtype"] == "fp32"
             and c["batch"] == 1 and "seconds_per_pair" in c), None
        )
        speedup = baseline["seconds_per_pair"] / fastest["seconds_per_pair"] if baseline else None
        print(f"{regime}: fastest {fastest['dtype']} batch {fastest['batch']} at "
              f"{fastest['seconds_per_pair']:.5f} s/pair"
              + (f", {speedup:.1f}x the batch-1 fp32 cell" if speedup else ""))
    print(f"\nwrote {dest.relative_to(REPO)}")


@app.function(
    image=mint_image,
    gpu=GPU,
    volumes={WEIGHTS: mint_weights_vol},
    timeout=150 * MINUTES,
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    max_containers=1,
    retries=0,
)
def epoch_curve(
    hla: dict[str, str],
    rows: list[dict],
    epoch_cap: int = 12,
    batch: int = 8,
    unfrozen: int = 4,
    lr: float = 1e-5,
) -> dict:
    """§5.2 probe 2: does early stopping bite before the assumed 20 epochs?

    §5's §6.3 forecast multiplies by an epoch count, and that count was assumed,
    not measured -- SPEARMINT used 200 for its stability stage, which at any rate
    we can reach would be unaffordable per *single* fit. This traces one
    adaptation run's inner-validation curve under a prespecified epoch cap and
    reports where the best epoch fell.

    **What this cannot conclude.** The fold here is a deterministic whole-cluster
    partition of the training rows built for timing, *not* the §3 nested protocol
    fold, and one run at one configuration is not a tuning result. No number from
    it is reportable as a Phase C outcome. It answers exactly one question: is the
    20-epoch multiplier in the budget optimistic, about right, or conservative.
    If the curve is still improving at the cap, that is a real outcome and the
    honest answer is that the multiplier is unresolved.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    from mint.data import Alphabet
    from mint.model.esm2 import ESM2

    alphabet = Alphabet.from_architecture("ESM-1b")
    cfg = _config()
    layer = cfg["encoder_layers"]

    t0 = time.monotonic()
    model = ESM2(
        num_layers=cfg["encoder_layers"],
        embed_dim=cfg["encoder_embed_dim"],
        attention_heads=cfg["encoder_attention_heads"],
        token_dropout=cfg["token_dropout"],
        use_multimer=True,
    )
    blob = torch.load(CLEAN, map_location="cpu", weights_only=True)
    model.load_state_dict(
        {k.replace("model.", "", 1): v for k, v in blob["state_dict"].items()}, strict=False
    )
    model.cuda()
    load_s = round(time.monotonic() - t0, 1)

    for name, p in model.named_parameters():
        parts = name.split(".")
        p.requires_grad_(
            bool(
                len(parts) > 1
                and parts[0] == "layers"
                and parts[1].isdigit()
                and int(parts[1]) >= layer - unfrozen
            )
        )
    head = nn.Sequential(nn.Linear(cfg["encoder_embed_dim"] * 10, 1)).cuda()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=lr
    )

    def encode(row: dict) -> list[int]:
        ids = []
        for seq in (row["peptide"], hla[row["allele"]]):
            ids += list(alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>"))
        return ids

    def forward(chunk: list[dict]) -> "torch.Tensor":
        tok = torch.tensor([encode(r) for r in chunk], dtype=torch.int64).cuda()
        cid = torch.tensor(
            [[0] * 11 + [1] * (tok.shape[1] - 11)] * len(chunk), dtype=torch.int32
        ).cuda()
        rep = model(tok, cid, repr_layers=[layer])["representations"][layer]
        # Position-preserving peptide pooling plus the mean-pooled partner, the
        # same 10 x 1,280 contract the smoke's extraction used.
        pep = rep[:, 1:10, :].flatten(1)
        partner = rep[:, 12:-1, :].mean(dim=1)
        return head(torch.cat([pep, partner], dim=1).float())

    fit = [r for r in rows if not r["inner_val"]]
    val = [r for r in rows if r["inner_val"]]
    rng = np.random.default_rng(0)

    def evaluate() -> dict:
        model.eval()
        preds, ys, alleles = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(val), batch):
                chunk = val[i : i + batch]
                preds += forward(chunk).squeeze(-1).float().cpu().tolist()
                ys += [r["y"] for r in chunk]
                alleles += [r["allele"] for r in chunk]
        model.train()
        p, y, a = np.array(preds), np.array(ys), np.array(alleles)
        # Median per-allele Spearman, the project's primary metric, on alleles
        # with enough inner-val rows to make a rank correlation meaningful.
        from scipy.stats import spearmanr

        per = []
        for al in np.unique(a):
            m = a == al
            if m.sum() >= 20 and len(np.unique(y[m])) > 1:
                per.append(spearmanr(p[m], y[m]).correlation)
        return {
            "mse": round(float(np.mean((p - y) ** 2)), 5),
            "median_per_allele_spearman": round(float(np.median(per)), 4) if per else None,
            "eligible_alleles": len(per),
        }

    # Checkpoint the curve to the Volume after every epoch. The first attempt at
    # this probe was cancelled externally 42 minutes in and returned nothing,
    # because the history existed only as a return value: a cancelled or
    # disconnected run threw away numbers it had already computed. A run stopped
    # at epoch 7 should still answer the question at epoch 7.
    progress = WEIGHTS / "epoch_curve_progress.json"

    def checkpoint(history: list[dict]) -> None:
        progress.write_text(json.dumps({"history": history, "epoch_cap": epoch_cap}, indent=2))
        mint_weights_vol.commit()

    history = [{"epoch": 0, **evaluate(), "train_seconds": 0.0}]
    checkpoint(history)
    model.train()
    for epoch in range(1, epoch_cap + 1):
        order = rng.permutation(len(fit))
        t = time.monotonic()
        losses = []
        for i in range(0, len(order) - batch + 1, batch):
            chunk = [fit[j] for j in order[i : i + batch]]
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = forward(chunk).squeeze(-1)
                loss = nn.functional.mse_loss(
                    pred.float(), torch.tensor([r["y"] for r in chunk]).cuda()
                )
            loss.backward()
            opt.step()
            losses.append(float(loss))
        train_s = time.monotonic() - t
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 5),
                "train_seconds": round(train_s, 1),
                **evaluate(),
            }
        )
        checkpoint(history)
        print(f"epoch {epoch}: val_mse {history[-1]['mse']} "
              f"scc {history[-1]['median_per_allele_spearman']} "
              f"{history[-1]['train_seconds']}s", flush=True)

    mses = [h["mse"] for h in history]
    best = int(np.argmin(mses))
    return {
        "gpu": GPU,
        "load_seconds": load_s,
        "config": {
            "epoch_cap": epoch_cap,
            "batch": batch,
            "unfrozen_top_layers": unfrozen,
            "lr": lr,
            "dtype": "bf16 autocast",
        },
        "fit_rows": len(fit),
        "inner_val_rows": len(val),
        "history": history,
        "best_epoch_by_mse": best,
        "still_improving_at_cap": best == epoch_cap,
        "seconds_per_epoch_median": round(
            float(np.median([h["train_seconds"] for h in history[1:]])), 1
        ),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


@app.local_entrypoint()
def epoch_probe(profile: str = "a-cheparukhin", epoch_cap: int = 12, inner_folds: int = 5):
    """§5.2 probe 2. Priced from probe 1's measured bf16 batch-8 rates."""
    import numpy as np
    import pandas as pd

    check_profile(profile)

    splits = pd.read_csv(REPO / "data" / "splits.csv")
    raw = pd.read_csv(REPO / "data" / "rasmussen_et_al_dataset.csv")
    train = splits[splits.split == "train"].merge(
        raw[["allele", "peptide", "thalf_hours", "hla_seq"]],
        on=["allele", "peptide"], how="left", validate="one_to_one",
    )
    assert train.hla_seq.notna().all() and train.thalf_hours.notna().all()

    # Deterministic whole-cluster partition of the TRAINING rows only. A timing
    # fold, not the §3 nested protocol fold -- see `epoch_curve`'s docstring.
    clusters = sorted(train.cluster_id.unique())
    held = {c for i, c in enumerate(clusters) if i % inner_folds == 0}
    train["inner_val"] = train.cluster_id.isin(held)

    hla = train.drop_duplicates("allele").set_index("allele").hla_seq.to_dict()
    rows = [
        {"allele": r.allele, "peptide": r.peptide,
         "y": float(np.log1p(r.thalf_hours)), "inner_val": bool(r.inner_val)}
        for r in train.itertuples()
    ]
    n_fit = sum(1 for r in rows if not r["inner_val"])
    n_val = len(rows) - n_fit

    t = json.loads((OUT_DIR / "phase_c_throughput.json").read_text())
    cell = {(c["regime"], c["dtype"], c["batch"]): c for c in t["cells"] if "seconds_per_pair" in c}
    ad = cell[("adaptation_top4", "bf16", 8)]["seconds_per_pair"]
    fr = cell[("frozen_extraction", "bf16", 8)]["seconds_per_pair"]
    gpu_s = EST_LOAD_S + epoch_cap * (n_fit * ad + n_val * fr) + n_val * fr + 120
    usd = gpu_s / 3600 * hourly_usd(GPU, WORKER_CPU, WORKER_MEM)
    print(f"{n_fit:,} fit rows, {n_val:,} inner-val rows ({len(held):,} held clusters), "
          f"cap {epoch_cap} epochs")
    print(f"priced from probe 1: {ad} s/pair adapt, {fr} s/pair eval")
    print(f"forecast ${usd:.2f}, ${1.25 * usd:.2f} with margin, ~{gpu_s / 60:.0f} min\n")

    res = epoch_curve.remote(hla, rows, epoch_cap=epoch_cap)
    res["profile"] = profile
    res["forecast_usd_with_margin"] = round(1.25 * usd, 2)
    dest = OUT_DIR / "phase_c_epoch_curve.json"
    dest.write_text(json.dumps(res, indent=2, default=str) + "\n")

    print(f"{'epoch':>5} {'train_loss':>11} {'val_mse':>9} {'med SCC':>8} {'s/epoch':>8}")
    for h in res["history"]:
        print(f"{h['epoch']:>5} {str(h.get('train_loss', '-')):>11} {h['mse']:>9.5f} "
              f"{str(h['median_per_allele_spearman']):>8} {h['train_seconds']:>8.1f}")
    print(f"\nbest epoch by inner-val MSE: {res['best_epoch_by_mse']}")
    print(f"still improving at the cap: {res['still_improving_at_cap']}")
    print(f"median s/epoch {res['seconds_per_epoch_median']}, peak {res['peak_gib']} GiB")
    print(f"\nwrote {dest.relative_to(REPO)}")
