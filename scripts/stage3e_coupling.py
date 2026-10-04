"""Stage 3e: coupling versus concatenation, on the features stage 3 already has.

    .venv/bin/python scripts/stage3e_coupling.py smoke
    .venv/bin/python scripts/stage3e_coupling.py ladder
    .venv/bin/python scripts/stage3e_coupling.py ensemble --selected reports/stage3e_selected.json

``reports/stage3_esm.md`` §9 leaves exactly one mechanism open for the stage 3
null: peptide and HLA are embedded independently, so the head has to learn the
peptide-groove interaction itself from 19,716 rows, and concatenation asks it to
do that in its first layer. "Frozen ESM-2 carries nothing useful here" and
"concatenation cannot use what it carries" predict the same flat result, and
nothing in stage 3 separates them.

This separates them. :mod:`pepstab.attn` lets each of the 9 peptide positions
attend over the 34 HLA contact residues; the control is the same module with the
attention replaced by a uniform average over those 34 residues, at an identical
parameter count. Both arms get the same folds, seeds, layers, ensemble size and
**tuning budget**, so the gap between them isolates coupling from capacity.

Three references, in descending order of what they would mean:

* **xattn vs meanpool** -- the controlled comparison, and the only one here
  whose difference is a single mechanism. A gap says concatenation was the
  limitation. No gap says it was not, which closes §9's open mechanism.
* **xattn vs esm_ensemble** (stage 3's shipped concatenation arm, same
  checkpoint, same layers, same 30 members) -- cross-architecture, so a
  difference confounds coupling with everything else that differs between a
  numpy MLP on flattened PCA components and a torch attention block. Reported,
  not leaned on.
* **xattn vs seq_ensemble_pep_pseudo** (stage 2) -- the project's standing
  reference, which answers "does any of this beat one-hot".

**On tuning.** ``reports/stage3_esm.md`` §2 is the reason this script has a
``ladder`` mode at all: transplanting a regularisation strength selected for one
feature geometry onto another cost that stage 0.109 SCC and nearly produced a
confident wrong negative. The ``stage3-esm`` branch, which wrote this experiment
and never ran it, would have inherited ``l2=1e-5`` from an :class:`AttnConfig`
default. Here each arm selects its own L2 on a ladder of at least three points,
the selection is checked for interiority, and the *budget* is identical across
arms even though the selected values need not be.
"""
from __future__ import annotations

# --- BLAS thread pinning, before numpy/torch are imported --------------------
# Same reasoning as scripts/esm_arm.py: several workstreams share 8 cores, every
# one of them links numpy against Accelerate, and the pool is sized at import.
# Torch gets its own cap below, from --threads.
import os as _os  # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import esm as pesm  # noqa: E402
from pepstab.attn import AttnConfig, Bank, CrossAttentionHead, fit, predict  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
)
from scripts.baseline_ensemble import N_FOLDS, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

#: Residue vectors are reduced on the *hidden* axis before either arm sees them:
#: (9, 480) -> (9, d_in). Position structure is untouched -- each residue vector
#: is one PCA sample -- so this is not the flattening reduction stage 3 applies.
#: 64 keeps the head at ~95k parameters, which is the same order as stage 2's
#: MLP (860 -> 256 -> 64 -> 1 is ~237k), so the comparison is not a capacity
#: mismatch dressed up as an architecture one.
D_IN = 64

#: Both arms' ladder. Three points, so "interior" is satisfiable; extended by
#: ``--l2-grid`` if a selection lands on an edge. Centred an order of magnitude
#: below stage 3's selected 1e-2 because these inputs are a learned projection
#: of standardised components rather than the components themselves, and the
#: decay here also reaches two projection matrices stage 3's MLP does not have.
#: That is a prior about where to *look*, not a selected value.
L2_GRID = (1e-4, 1e-3, 1e-2)

ARMS = ("xattn", "meanpool")
LAYERS = ("mid", "final")
BASELINES = ("seq_ensemble_pep_pseudo", "esm_ensemble")


def load_data():
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    return df, train, val, alleles


def _residue_pca(blocks: np.ndarray, fit_rows: np.ndarray, d_in: int
                 ) -> tuple[np.ndarray, float]:
    """Reduce the hidden axis of ``(n_unique, positions, dim)`` to ``d_in``.

    Each residue vector is one sample, so the output is still
    ``(n_unique, positions, d_in)`` -- only the basis changes. Fitted on the
    unique sequences appearing in the training split, with no labels, so it
    cannot leak; and fitted on *unique* sequences rather than rows, so a peptide
    measured on 36 alleles does not get 36 votes in the covariance.

    Unlike the branch this is called **once per kind**, giving the peptide block
    and the groove block their own bases. A single peptide-fitted basis applied
    to both, which is what the branch did, measurably discards groove variance
    (``reports/stage3e_cost.json`` records how much) and buys nothing: each
    block has its own learned input projection inside the head anyway, so there
    is no coordinate system the two need to share.
    """
    flat = blocks[fit_rows].reshape(-1, blocks.shape[-1]).astype(np.float64)
    centre = flat.mean(axis=0)
    centred = flat - centre
    k = min(d_in, min(centred.shape) - 1)
    from sklearn.utils.extmath import randomized_svd

    _, s, Vt = randomized_svd(centred, n_components=k, random_state=0)
    total = float((centred ** 2).sum())
    explained = float((s ** 2).sum() / total) if total else 1.0

    out = (blocks.reshape(-1, blocks.shape[-1]).astype(np.float64) - centre) @ Vt.T
    # Scale each component to unit variance on the fit residues, so the learned
    # projection starts from an isotropic input rather than one whose first
    # component is 30x the last.
    scale = ((flat - centre) @ Vt.T).std(axis=0, keepdims=True) + 1e-6
    out = (out / scale).reshape(*blocks.shape[:-1], k)
    return out.astype(np.float32), explained


def _variance_retained(blocks: np.ndarray, fit_rows: np.ndarray,
                       basis_blocks: np.ndarray, d_in: int) -> float:
    """Share of ``blocks``'s residue variance kept by a basis fitted on another.

    Measures what the branch's shared peptide-fitted basis would have cost the
    groove block, so the choice to fit separately is reported rather than
    asserted.
    """
    src = basis_blocks[: len(basis_blocks)].reshape(-1, basis_blocks.shape[-1]).astype(np.float64)
    centre = src.mean(axis=0)
    from sklearn.utils.extmath import randomized_svd

    k = min(d_in, min(src.shape) - 1)
    _, _, Vt = randomized_svd(src - centre, n_components=k, random_state=0)
    tgt = blocks[fit_rows].reshape(-1, blocks.shape[-1]).astype(np.float64) - centre
    total = float((tgt ** 2).sum())
    kept = float(((tgt @ Vt.T) ** 2).sum())
    return kept / total if total else 1.0


def build_banks(train: pd.DataFrame, frames: list[pd.DataFrame], checkpoint: str,
                layer: str, d_in: int) -> tuple[list[tuple[Bank, Bank]], dict]:
    """Deduplicated residue banks plus row indices, per frame.

    Returns ``([(pep_bank, hla_bank), ...], info)`` with one entry per frame in
    ``frames``.
    """
    pep_lookup, pep_reps = pesm.load_cache(checkpoint, "peptide", layer)
    hla_lookup, hla_reps = pesm.load_cache(checkpoint, "hla", layer)
    hla_reps = hla_reps[:, np.asarray(pesm.CONTACT_POSITIONS_1BASED) - 1, :]

    pep_fit = np.unique([pep_lookup[str(v)] for v in train.peptide.unique()])
    hla_fit = np.unique([hla_lookup[str(v)] for v in train.hla_seq.unique()])

    pep_block, pep_var = _residue_pca(pep_reps, pep_fit, d_in)
    hla_block, hla_var = _residue_pca(hla_reps, hla_fit, d_in)
    shared_cost = _variance_retained(hla_reps, hla_fit, pep_reps[pep_fit], d_in)

    out = []
    for frame in frames:
        pep_idx = np.fromiter((pep_lookup[str(v)] for v in frame.peptide),
                              dtype=np.int64, count=len(frame))
        hla_idx = np.fromiter((hla_lookup[str(v)] for v in frame.hla_seq),
                              dtype=np.int64, count=len(frame))
        out.append((Bank(pep_block, pep_idx), Bank(hla_block, hla_idx)))

    info = {
        "layer": layer, "d_in": int(pep_block.shape[-1]),
        "n_unique_peptides": int(len(pep_block)), "n_unique_hla": int(len(hla_block)),
        "peptide_variance_retained": round(pep_var, 6),
        "hla_variance_retained": round(hla_var, 6),
        "hla_variance_under_shared_peptide_basis": round(shared_cost, 6),
        "bank_mb": round((pep_block.nbytes + hla_block.nbytes) / 1e6, 1),
        "fanned_out_mb_avoided": round(
            (len(frames[0]) * hla_block.shape[1] * hla_block.shape[2] * 4) / 1e6, 1),
    }
    return out, info


def _member(arm: str, l2: float, seed: int, d_in: int) -> CrossAttentionHead:
    cfg = AttnConfig(d_model=D_IN, n_heads=4, hidden=128, l2=l2, seed=seed,
                     couple=(arm == "xattn"), max_epochs=MAX_EPOCHS,
                     patience=PATIENCE)
    return CrossAttentionHead(d_in, cfg)


def _parsimonious(sub: pd.DataFrame) -> float:
    """The largest L2 whose mean rho is within one standard error of the best.

    The plain argmax is not usable on this ladder, and that is a measurement
    rather than an inconvenience. Over five points spanning 1e-6 to 1e-2 the
    mean validation rho moves by ~0.04 while the **seed** spread at a single
    point is 0.03-0.07, so most of the ladder is a statistical tie and the
    argmax lands wherever the noise happens to peak. Chasing it is what produced
    two successive boundary hits here -- the attention arm selected the bottom of
    a three-point ladder, and after extending downward by two decades the
    ablation selected the new bottom -- and extending again would not terminate,
    because there is no real optimum to walk toward.

    So the rule is the standard one for a flat selection curve: among points that
    are statistically indistinguishable from the best, take the most regularised,
    which is also the most stable. It terminates, it is applied identically to
    both arms, and both the argmax and the parsimonious choice are recorded in
    ``reports/stage3e_tuning.csv`` so a reader can check the rule did not
    manufacture the result.
    """
    means = sub.groupby("l2").member_median_rho.mean()
    per_point = sub.groupby("l2").member_median_rho
    best_l2 = means.idxmax()
    n = max(int(per_point.count()[best_l2]), 2)
    se = float(per_point.std()[best_l2]) / np.sqrt(n)
    ok = means[means >= means.max() - se]
    return float(max(ok.index))


def _fit_one(arm: str, l2: float, seed: int, fold: int, banks, y_train, fold_id,
             val_banks, val, y_val, alleles) -> tuple[np.ndarray, dict]:
    """Fit one member and score it on validation. Returns (prediction, record)."""
    pep, hla = banks
    is_fit = fold_id != fold
    model = _member(arm, l2, seed, pep.d_in)
    info = fit(model, pep.subset(is_fit), hla.subset(is_fit), y_train[is_fit],
               pep.subset(~is_fit), hla.subset(~is_fit), y_train[~is_fit])
    t0 = time.perf_counter()
    pred = predict(model, val_banks[0], val_banks[1], info["offset"])
    infer = 1000 * (time.perf_counter() - t0) / len(val)
    rho = score("m", val.allele, y_val, pred, alleles).median_spearman
    return pred, {
        "arm": arm, "l2": l2, "seed": seed, "fold": fold,
        "n_parameters": info["n_parameters"], "best_epoch": info["best_epoch"],
        "n_epochs": info["n_epochs"], "dev_mse": round(info["dev_mse"], 6),
        "fit_seconds": round(info["fit_seconds"], 1),
        "infer_seconds_per_1k": round(infer, 4),
        "member_median_rho": round(rho, 6),
    }


# --- mode: smoke ------------------------------------------------------------


def mode_smoke(args) -> int:
    """End-to-end pilot: both arms, one fold, one seed, few epochs.

    Required before the batch run by the project's own rule -- a new runner gets
    a pilot as much as a new model does. Checks the three things that would
    invalidate the experiment: that the arms are parameter-identical, that the
    ablation really has no attention map, and that two members at the same seed
    are bit-identical (which the branch's unseeded initialisation would fail).
    """
    df, train, val, alleles = load_data()
    fold_id = cv_folds(train, N_FOLDS)
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    (train_banks, val_banks), info = build_banks(
        train, [train, val], args.checkpoint, "mid", args.d_in)
    print(json.dumps(info, indent=1))

    sizes, preds = {}, {}
    for arm in ARMS:
        pep, hla = train_banks
        is_fit = fold_id != 0
        cfg = AttnConfig(d_model=D_IN, n_heads=4, hidden=128, l2=L2_GRID[1],
                         seed=0, couple=(arm == "xattn"), max_epochs=args.epochs,
                         patience=PATIENCE)
        m = CrossAttentionHead(pep.d_in, cfg)
        res = fit(m, pep.subset(is_fit), hla.subset(is_fit), y_train[is_fit],
                  pep.subset(~is_fit), hla.subset(~is_fit), y_train[~is_fit])
        p = predict(m, val_banks[0], val_banks[1], res["offset"])
        sizes[arm] = res["n_parameters"]
        preds[arm] = p
        s = score(arm, val.allele, y_val, p, alleles)
        print(f"{arm:9s} {res['n_parameters']:,} params, "
              f"{res['n_epochs']} epochs ({res['fit_seconds']:.0f}s), "
              f"dev mse {res['dev_mse']:.4f}, val median rho {s.median_spearman:.4f}")

    assert sizes["xattn"] == sizes["meanpool"], (
        f"arms differ in size: {sizes} -- the capacity control is broken")
    print(f"\nparameter identity: both arms {sizes['xattn']:,} -- OK")

    pep, hla = val_banks
    rows = torch.arange(4)
    m = _member("meanpool", L2_GRID[1], 0, pep.d_in)
    try:
        m.attention_map(pep.gather(rows), hla.gather(rows))
    except ValueError:
        print("ablation has no attention map -- OK")
    else:
        raise AssertionError("the mean-pool ablation returned an attention map")

    a = _member("xattn", L2_GRID[1], 0, pep.d_in)
    b = _member("xattn", L2_GRID[1], 0, pep.d_in)
    same = all(torch.equal(x, y) for x, y in
               zip(a.state_dict().values(), b.state_dict().values()))
    assert same, "two members at seed 0 differ: initialisation is not seeded"
    print("seeded initialisation is reproducible -- OK")

    amap = a.attention_map(pep.gather(rows), hla.gather(rows))
    assert amap.shape == (4, 9, 34), amap.shape
    sums = amap.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums
    print(f"attention map {tuple(amap.shape)}, rows sum to 1 -- OK")
    print(f"\nprojected full run: {len(ARMS)} arms x {N_FOLDS} folds x "
          f"{len(LAYERS)} layers x {len(SEEDS)} seeds = "
          f"{len(ARMS) * N_FOLDS * len(LAYERS) * len(SEEDS)} networks")
    return 0


# --- mode: ladder -----------------------------------------------------------


def mode_ladder(args) -> int:
    """Select L2 per arm on its own ladder, and record whether it was interior.

    One fold and one layer, three seeds per point, so the cost is
    ``len(grid) x 2 arms x 3 seeds`` networks rather than a full ensemble per
    point. Selection is on validation median per-allele rho, which is the same
    surface ``reports/stage3_esm.md`` §4 selected its representation on; the
    test split is not loaded here or anywhere in this file.
    """
    df, train, val, alleles = load_data()
    fold_id = cv_folds(train, N_FOLDS)
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()
    grid = tuple(args.l2_grid) if args.l2_grid else L2_GRID

    banks, info = build_banks(train, [train, val], args.checkpoint, args.layer,
                              args.d_in)
    train_banks, val_banks = banks
    print(f"layer {args.layer}, d_in {info['d_in']}, "
          f"ladder {[f'{g:g}' for g in grid]}\n")

    records = []
    for arm in ARMS:
        for l2 in grid:
            for seed in SEEDS:
                _, rec = _fit_one(arm, l2, seed, args.fold, train_banks, y_train,
                                  fold_id, val_banks, val, y_val, alleles)
                rec["layer"] = args.layer
                rec["stage"] = "ladder"
                records.append(rec)
            got = [r for r in records if r["arm"] == arm and r["l2"] == l2]
            print(f"  {arm:9s} l2={l2:<7g} rho "
                  f"{np.mean([r['member_median_rho'] for r in got]):.4f} "
                  f"(spread {np.ptp([r['member_median_rho'] for r in got]):.4f}), "
                  f"{np.mean([r['fit_seconds'] for r in got]):.0f}s/net")

    runs = pd.DataFrame(records)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / "stage3e_ladder.csv"
    if path.exists():
        runs = pd.concat([pd.read_csv(path), runs], ignore_index=True)
    runs.to_csv(path, index=False)

    selected, sens = {}, []
    for arm in ARMS:
        sub = runs[runs.arm == arm]
        means = sub.groupby("l2").member_median_rho.mean()
        tried = sorted(sub.l2.unique())
        best = float(means.idxmax())
        spread = sub[sub.l2 == best].member_median_rho
        # One-standard-error parsimony rule. See _parsimonious() for why the
        # plain argmax is not usable on this ladder.
        pars = _parsimonious(sub)
        interior = len(tried) >= 3 and min(tried) < pars < max(tried)
        selected[arm] = pars
        sens.append({"arm": arm, "ladder": " ".join(f"{t:g}" for t in tried),
                     "n_points": len(tried), "argmax": best, "selected": pars,
                     "rule": "largest l2 within 1 SE of the best",
                     "interior": bool(interior),
                     "rho_at_argmax": round(float(means[best]), 4),
                     "rho_at_selected": round(float(means[pars]), 4),
                     "seed_spread_at_argmax": round(float(np.ptp(spread)), 4),
                     "seed_spread_at_selected": round(float(np.ptp(
                         sub[sub.l2 == pars].member_median_rho)), 4),
                     "ladder_range_rho": round(float(means.max() - means.min()), 4),
                     "rho_at_min": round(float(means[min(tried)]), 4),
                     "rho_at_max": round(float(means[max(tried)]), 4)})
        flag = "interior" if interior else "AT BOUNDARY -- extend with --l2-grid"
        print(f"\n{arm}: argmax l2={best:g}, selected l2={pars:g} "
              f"from {len(tried)} points ({flag})")

    # The comparison this experiment exists to make is xattn vs meanpool, and
    # those two arms have *identical* feature geometry, width and parameter
    # count -- the only difference is the coupling switch. Stage 3 §2's lesson
    # is about arms whose feature geometry differs; applied here it would be
    # actively harmful, because letting the two arms sit at different L2 adds a
    # second difference and the gap would no longer isolate coupling. So the
    # controlled comparison runs both arms at one common value, chosen on the
    # pooled ladder by the same parsimony rule.
    common = _parsimonious(runs)
    tried = sorted(runs.l2.unique())
    pooled = runs.groupby("l2").member_median_rho.mean()
    common_interior = len(tried) >= 3 and min(tried) < common < max(tried)
    print(f"\npooled over both arms: argmax l2={float(pooled.idxmax()):g}, "
          f"common l2={common:g} ({'interior' if common_interior else 'AT BOUNDARY'})")
    sens.append({"arm": "COMMON (pooled)", "ladder": " ".join(f"{t:g}" for t in tried),
                 "n_points": len(tried), "argmax": float(pooled.idxmax()),
                 "selected": common, "rule": "largest l2 within 1 SE of the best",
                 "interior": bool(common_interior),
                 "rho_at_argmax": round(float(pooled.max()), 4),
                 "rho_at_selected": round(float(pooled[common]), 4),
                 "ladder_range_rho": round(float(pooled.max() - pooled.min()), 4)})

    pd.DataFrame(sens).to_csv(REPORT_DIR / "stage3e_tuning.csv", index=False)
    (REPORT_DIR / "stage3e_selected.json").write_text(json.dumps(
        {**selected, "_common": common}, indent=1))
    print(f"\nwrote {(REPORT_DIR / 'stage3e_tuning.csv').relative_to(REPO_ROOT)} "
          f"and stage3e_selected.json")
    if not common_interior:
        print("\nThe common selection is on an edge. Extend the ladder and "
              "re-run before the ensemble.")
    return 0


# --- mode: ensemble ---------------------------------------------------------


def mode_ensemble(args) -> int:
    df, train, val, alleles = load_data()
    fold_id = cv_folds(train, N_FOLDS)
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    chosen = json.loads(Path(args.selected).read_text()) if args.selected else {}
    if args.l2 is not None:
        chosen = {a: args.l2 for a in ARMS}
    elif "_common" in chosen:
        # Default to the common pooled value: the arms share a feature geometry,
        # so holding L2 equal is what keeps the gap attributable to coupling.
        chosen = {a: chosen["_common"] for a in ARMS}
    missing = [a for a in ARMS if a not in chosen]
    if missing:
        raise SystemExit(
            f"no selected L2 for {missing}. Run `ladder` first -- this script "
            f"will not fall back to a default, because inheriting an unexamined "
            f"regularisation strength is the error stage 3 §2 exists to document.")
    if len(set(chosen[a] for a in ARMS)) != 1:
        print(f"NOTE: arms are at different L2 ({chosen}). The xattn-vs-meanpool "
              f"gap then confounds coupling with regularisation; prefer --l2.")

    n_members = N_FOLDS * len(LAYERS) * len(SEEDS)
    print(f"checkpoint {args.checkpoint}, d_in {args.d_in}")
    print(f"per arm: {N_FOLDS} CV folds x {len(LAYERS)} ESM layers x "
          f"{len(SEEDS)} seeds = {n_members} networks "
          f"(stage 2 and stage 3 both use 30)")
    print(f"selected L2: " + ", ".join(f"{a}={chosen[a]:g}" for a in ARMS) + "\n")

    members: dict[tuple[str, str], list[np.ndarray]] = {}
    records, costs = [], {"bank_info": []}
    started = time.perf_counter()
    for layer in LAYERS:
        banks, info = build_banks(train, [train, val], args.checkpoint, layer,
                                  args.d_in)
        costs["bank_info"].append(info)
        train_banks, val_banks = banks
        for arm in ARMS:
            for k in range(N_FOLDS):
                for seed in SEEDS:
                    pred, rec = _fit_one(arm, chosen[arm], seed, k, train_banks,
                                         y_train, fold_id, val_banks, val, y_val,
                                         alleles)
                    rec["layer"] = layer
                    rec["stage"] = "ensemble_member"
                    records.append(rec)
                    members.setdefault((arm, layer), []).append(pred)
            got = [r for r in records if r["arm"] == arm and r["layer"] == layer]
            print(f"  {arm:9s} layer {layer:5s}: {len(got)} networks, "
                  f"{got[0]['n_parameters']:,} params, member rho "
                  f"{np.mean([r['member_median_rho'] for r in got]):.4f}, "
                  f"{np.mean([r['fit_seconds'] for r in got]):.0f}s/net")
    wall = time.perf_counter() - started

    runs = pd.DataFrame(records)
    runs.to_csv(REPORT_DIR / "stage3e_runs.csv", index=False)
    by_arm = runs.groupby("arm").n_parameters.nunique()
    assert (by_arm == 1).all(), f"parameter count varies within an arm: {by_arm}"
    sizes = runs.groupby("arm").n_parameters.first().to_dict()
    assert len(set(sizes.values())) == 1, (
        f"arms differ in size: {sizes} -- the capacity control is broken, and a "
        f"gap between them would not isolate coupling")
    print(f"\nparameter identity holds: both arms {sizes['xattn']:,}")

    # --- ensembles, and the comparisons ------------------------------------
    ens = {arm: np.mean([p for L in LAYERS for p in members[(arm, L)]], axis=0)
           for arm in ARMS}
    per_layer = {(arm, L): np.mean(members[(arm, L)], axis=0)
                 for arm in ARMS for L in LAYERS}

    PRED_DIR.mkdir(exist_ok=True)
    for arm, pred in ens.items():
        pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": pred}).to_csv(
            PRED_DIR / f"stage3e_{arm}.csv", index=False)

    refs: dict[str, np.ndarray] = {}
    for base in BASELINES:
        path = PRED_DIR / f"{base}.csv"
        if not path.exists():
            print(f"  (missing {path.name}; skipping that comparison)")
            continue
        merged = val[["pair_id"]].merge(pd.read_csv(path), on="pair_id",
                                        how="left", validate="one_to_one")
        if merged.y_pred.isna().any():
            raise ValueError(f"{path.name} does not cover every validation row")
        refs[base] = merged.y_pred.to_numpy()

    summary = []
    for name, pred in ([(a, ens[a]) for a in ARMS]
                       + [(f"{a}_layer_{L}", per_layer[(a, L)])
                          for a in ARMS for L in LAYERS]):
        row = score(name, val.allele, y_val, pred, alleles).as_row()
        row["n_members"] = n_members if "_layer_" not in name else n_members // len(LAYERS)
        summary.append(row)
    for base, pred in refs.items():
        row = score(base, val.allele, y_val, pred, alleles).as_row()
        row["n_members"] = 30
        summary.append(row)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(REPORT_DIR / "stage3e_summary.csv", index=False)

    comparisons = []
    # The controlled comparison first: it is the only one whose difference is a
    # single mechanism, and the only one this experiment was built to make.
    pairs = [("xattn", "meanpool", ens["meanpool"],
              "coupling vs its parameter-identical ablation")]
    for base, pred in refs.items():
        pairs.append(("xattn", base, pred, "coupling vs external reference"))
        pairs.append(("meanpool", base, pred, "ablation vs external reference"))
    for arm, base_name, base_pred, kind in pairs:
        res = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                       base_pred, ens[arm], alleles=alleles)
        lo, hi = res["ci95"]
        a = score(arm, val.allele, y_val, ens[arm], alleles)
        b = score(base_name, val.allele, y_val, base_pred, alleles)
        print(f"\n{arm} vs {base_name} ({kind}):")
        print(f"  {base_name} {b.median_spearman:.4f}  {arm} {a.median_spearman:.4f}  "
              f"delta {res['delta_median_spearman']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"  verdict: {describe_delta(res)}")
        comparisons.append({
            "arm": arm, "reference": base_name, "comparison": kind,
            "n_rows": len(val), "n_alleles": len(alleles),
            "reference_median_rho": round(b.median_spearman, 4),
            "arm_median_rho": round(a.median_spearman, 4),
            "delta": round(res["delta_median_spearman"], 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "verdict": describe_delta(res),
            "n_members_arm": n_members,
            "n_members_reference": 30,
            "l2_arm": chosen[arm],
        })
    pd.DataFrame(comparisons).to_csv(REPORT_DIR / "stage3e_comparisons.csv",
                                     index=False)

    costs |= {
        "n_networks": int(len(runs)), "wall_minutes": round(wall / 60, 1),
        "total_fit_seconds": float(runs.fit_seconds.sum()),
        "median_fit_seconds": float(runs.fit_seconds.median()),
        "median_infer_seconds_per_1k": float(runs.infer_seconds_per_1k.median()),
        "n_parameters": {a: int(v) for a, v in sizes.items()},
        "selected_l2": chosen,
        "median_best_epoch": {a: float(g.best_epoch.median())
                              for a, g in runs.groupby("arm")},
    }
    (REPORT_DIR / "stage3e_cost.json").write_text(json.dumps(costs, indent=2))
    print(f"\n{len(runs)} networks in {wall / 60:.1f} min "
          f"({wall / len(runs):.0f}s per network)")
    print("wrote reports/stage3e_{runs,summary,comparisons}.csv, "
          "stage3e_cost.json, preds/stage3e_*.csv")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("smoke", "ladder", "ensemble"))
    ap.add_argument("--checkpoint", default=pesm.DEFAULT_CHECKPOINT)
    ap.add_argument("--d-in", type=int, default=D_IN)
    ap.add_argument("--layer", default="mid", choices=LAYERS)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--l2-grid", type=float, nargs="*", default=None)
    ap.add_argument("--selected", default=None)
    ap.add_argument("--l2", type=float, default=None,
                    help="run both arms at this L2 (the controlled comparison)")
    ap.add_argument("--epochs", type=int, default=6, help="smoke mode only")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    return {"smoke": mode_smoke, "ladder": mode_ladder,
            "ensemble": mode_ensemble}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
