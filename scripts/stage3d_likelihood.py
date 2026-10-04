#!/usr/bin/env python
"""Stage 3d -- ESM-2 *likelihood* features for peptide-HLA stability.

Stage 3 asked whether ESM-2's hidden states carry stability signal and
answered no (ESM-only rho 0.683, seq+ESM 0.676, both inconclusive against the
0.693 sequence ensemble). This stage asks a different question of the same
checkpoint: not what the model *represents*, but what it considers
*probable*. A peptide that ESM-2 finds surprising sits in a thinly populated
region of sequence space, and the hypothesis -- stated before the run -- is
that surprisal tracks stability because unusual residues at anchor positions
disrupt pocket complementarity.

The hypothesis has a specific reason to fail, and it is worth naming up front
so the negative is interpretable if it comes: ESM-2 is trained on UniRef, in
which a 9-mer is never an entity. Every peptide here is an interior fragment
of some longer protein, presented to the model without its flanks. The
masked-marginal distribution is therefore conditioned on eight neighbours and
nothing else, which is a far weaker context than the model normally works in.

Two subcommands:

``extract``
    Masked-marginal pseudo-log-likelihood over unique peptides. For each
    peptide and each of its 9 positions, the position is replaced by ``<mask>``
    and the model scores the full vocabulary there. This is 9 forward passes
    per peptide -- the standard masked-marginal estimator (Meier et al. 2021),
    not the cheaper single-pass wild-type-marginal approximation, because at
    length 9 a single pass leaks the residue it is scoring through the
    unmasked input.

``arm``
    The stage 3 protocol applied to those features: the same ``cv_folds``
    fold assignment, the same seeds, the same 30-network ensemble count, the
    same L2 ladder with a boundary check, scored by the same paired cluster
    bootstrap against the stage 2 sequence ensemble.

The ensemble axis is the *checkpoint* (35M, 650M) rather than stage 3's layer
pair, so the member count matches stage 2 and stage 3 exactly: 5 folds x 2
checkpoints x 3 seeds = 30 networks.
"""

from __future__ import annotations

# Thread defaults before numpy/torch import; override in the environment.
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    MIN_WORTHWHILE_DELTA_SPEARMAN,
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
)
from pepstab.features import INPUT_SETS, build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import N_FOLDS, cv_folds  # noqa: E402
from scripts.baseline_sequence import (  # noqa: E402
    HIDDEN_GRID,
    MAX_EPOCHS,
    PATIENCE,
    SEEDS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = REPO_ROOT / "features"
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"
RUNS_CSV = REPORT_DIR / "stage3d_runs.csv"

#: Checkpoints scored. Both are run for every peptide; the ensemble averages
#: over them, and the per-checkpoint arms are reported separately so a reader
#: can see whether 19x the parameters bought anything.
CHECKPOINTS = ("esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D")

#: Dense standardised features need stronger shrinkage than stage 2's sparse
#: one-hot columns; this is stage 3's ladder, reused unchanged so the two
#: stages are comparable. Three points, so the interior check is meaningful.
L2_GRID = (1e-3, 1e-2, 1e-1)

#: The 20 canonical amino acids. Entropy is computed over these after
#: renormalising, not over ESM-2's full 33-token vocabulary: the vocabulary
#: carries mask/pad/unk/gap tokens whose probability mass is noise here, and
#: an entropy that moves when the model shifts mass onto ``<pad>`` would not
#: be a statement about residue preference.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

#: The stage 2 sequence block the ``seq_lik`` arm appends to. Named here and
#: checked against the package so a rename upstream fails loudly instead of
#: silently scoring a different arm than the report claims.
SEQ_INPUT_SET = "pep_pseudo"
if SEQ_INPUT_SET not in INPUT_SETS:
    raise ImportError(f"input set {SEQ_INPUT_SET!r} is gone; "
                      f"pepstab.features offers {sorted(INPUT_SETS)}")


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------

def masked_marginals(peptides: list[str], checkpoint: str, device: str,
                     batch_size: int = 1024) -> tuple[np.ndarray, list[str], dict]:
    """Masked-marginal features for each peptide.

    Returns ``(features, column_names, timing)``. One row per peptide.
    """
    import esm as fair_esm
    import torch

    t0 = time.perf_counter()
    model, alphabet = fair_esm.pretrained.load_model_and_alphabet(checkpoint)
    model = model.to(device).eval()
    load_s = time.perf_counter() - t0
    torch.set_num_threads(int(_os.environ.get("OMP_NUM_THREADS", "1")))

    length = len(peptides[0])
    if any(len(p) != length for p in peptides):
        raise ValueError("stage 3d assumes a fixed peptide length")

    aa_ids = torch.tensor([alphabet.get_idx(a) for a in CANONICAL_AA],
                          device=device)
    if (aa_ids < 0).any():
        raise ValueError(f"{checkpoint}: canonical amino acid missing from vocab")

    converter = alphabet.get_batch_converter()
    _, _, ids = converter([(str(i), p) for i, p in enumerate(peptides)])
    ids = ids.to(device)                           # (N, length + 2)
    n, n_tok = ids.shape
    if n_tok != length + 2:
        raise ValueError(f"expected {length + 2} tokens, got {n_tok}")

    # The indexing the rest of this function assumes: token 0 is BOS and
    # token i+1 is peptide residue i. Checked rather than trusted, because a
    # silent off-by-one here would score the wrong residue and still produce
    # plausible numbers.
    for probe in range(min(5, n)):
        for pos in range(length):
            if ids[probe, pos + 1].item() != alphabet.get_idx(peptides[probe][pos]):
                raise AssertionError(
                    f"token layout mismatch at peptide {probe} position {pos}")

    # True residue token at each sequence position, for the log p(true) lookup.
    true_ids = ids[:, 1:length + 1]                # (N, length)

    logp_true = torch.empty((n, length), dtype=torch.float32, device=device)
    entropy = torch.empty((n, length), dtype=torch.float32, device=device)
    margin = torch.empty((n, length), dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    n_passes = 0
    with torch.no_grad():
        for pos in range(length):
            tok_idx = pos + 1                      # skip <cls>
            for lo in range(0, n, batch_size):
                hi = min(lo + batch_size, n)
                batch = ids[lo:hi].clone()
                batch[:, tok_idx] = alphabet.mask_idx
                logits = model(batch)["logits"][:, tok_idx, :]
                n_passes += hi - lo

                # log p(true residue) over the FULL vocabulary -- the standard
                # masked-marginal quantity, left unrenormalised so it stays
                # comparable with published pseudo-log-likelihoods.
                full_logp = torch.log_softmax(logits.float(), dim=-1)
                logp_true[lo:hi, pos] = full_logp.gather(
                    1, true_ids[lo:hi, pos:pos + 1]).squeeze(1)

                # Entropy and wild-type margin over the 20 canonical residues.
                aa_logp = torch.log_softmax(logits.float()[:, aa_ids], dim=-1)
                p = aa_logp.exp()
                entropy[lo:hi, pos] = -(p * aa_logp).sum(dim=-1)
                best = aa_logp.max(dim=-1).values
                # position of the true residue within the canonical ordering
                true_rank = (aa_ids[None, :] == true_ids[lo:hi, pos:pos + 1])
                true_aa_logp = (aa_logp * true_rank).sum(dim=-1)
                margin[lo:hi, pos] = true_aa_logp - best
    embed_s = time.perf_counter() - t0

    logp_true_np = logp_true.cpu().numpy()
    entropy_np = entropy.cpu().numpy()
    margin_np = margin.cpu().numpy()

    pll = logp_true_np.sum(axis=1, keepdims=True)
    pll_mean = pll / length
    perplexity = np.exp(-pll_mean)
    ent_mean = entropy_np.mean(axis=1, keepdims=True)

    feats = np.hstack([pll, pll_mean, perplexity, ent_mean,
                       logp_true_np, entropy_np, margin_np]).astype(np.float32)
    cols = (["pll", "pll_mean", "perplexity", "entropy_mean"]
            + [f"logp_P{i + 1}" for i in range(length)]
            + [f"entropy_P{i + 1}" for i in range(length)]
            + [f"margin_P{i + 1}" for i in range(length)])
    if feats.shape[1] != len(cols):
        raise AssertionError("feature/column mismatch")
    if not np.isfinite(feats).all():
        raise AssertionError(f"{checkpoint}: non-finite likelihood feature")

    timing = {
        "checkpoint": checkpoint,
        "device": device,
        "n_peptides": n,
        "n_forward_passes": n_passes,
        "load_seconds": round(load_s, 2),
        "extract_seconds": round(embed_s, 2),
        "seconds_per_1k_passes": round(1000 * embed_s / max(n_passes, 1), 3),
        "n_features": feats.shape[1],
    }
    return feats, cols, timing


def cmd_extract(args: argparse.Namespace) -> int:
    df = load_with_splits()
    peptides = sorted(df["peptide"].unique())
    print(f"{len(peptides):,} unique peptides, "
          f"{len(df):,} rows, {len(peptides) * 9:,} masked passes per checkpoint")

    device = args.device
    if device == "auto":
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    timings = []
    for ckpt in args.checkpoints:
        out = FEATURE_DIR / f"stage3d_lik_{ckpt}.npz"
        if out.exists() and not args.force:
            print(f"{ckpt}: cached, skipping")
            continue
        feats, cols, timing = masked_marginals(peptides, ckpt, device,
                                               args.batch_size)
        np.savez_compressed(out, peptides=np.array(peptides), features=feats,
                            columns=np.array(cols), checkpoint=ckpt)
        timings.append(timing)
        print(f"{ckpt}: {timing['extract_seconds']}s for "
              f"{timing['n_forward_passes']:,} passes "
              f"({timing['seconds_per_1k_passes']}s/1k), -> {out.name}")

    if timings:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / "stage3d_extraction_cost.csv"
        prev = pd.read_csv(path) if path.exists() else None
        new = pd.DataFrame(timings)
        out_df = pd.concat([prev, new]) if prev is not None else new
        out_df.to_csv(path, index=False)
        print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------
# arm
# --------------------------------------------------------------------------

def load_likelihood(df: pd.DataFrame, checkpoint: str) -> np.ndarray:
    """Per-row likelihood block, broadcast from the per-peptide cache."""
    path = FEATURE_DIR / f"stage3d_lik_{checkpoint}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run `stage3d_likelihood.py extract`")
    z = np.load(path, allow_pickle=False)
    index = {p: i for i, p in enumerate(z["peptides"].tolist())}
    missing = set(df["peptide"]) - set(index)
    if missing:
        raise KeyError(f"{len(missing)} peptides absent from {path.name}")
    rows = np.fromiter((index[p] for p in df["peptide"]), dtype=np.int64,
                       count=len(df))
    return z["features"][rows]


def standardise(train_block: np.ndarray, *blocks: np.ndarray):
    mu = train_block.mean(axis=0, keepdims=True)
    sd = train_block.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return [((b - mu) / sd).astype(np.float32) for b in (train_block, *blocks)]


def fit_predict(X_fit, y_fit, X_dev, y_dev, X_eval, cfg) -> tuple[np.ndarray, int]:
    model = MLPRegressor(cfg).fit(X_fit, y_fit, X_dev, y_dev)
    return model.predict(X_eval), getattr(model, "best_epoch_", -1)


def build_blocks(df: pd.DataFrame, arm: str, checkpoint: str) -> np.ndarray:
    """Feature matrix for one arm.

    ``lik``       likelihood features alone
    ``seq_lik``   stage 2's one-hot peptide+pseudosequence block with the
                  likelihood features appended
    """
    lik = load_likelihood(df, checkpoint)
    if arm == "lik":
        return lik
    if arm == "seq_lik":
        seq = build_features(df, SEQ_INPUT_SET, "onehot").astype(np.float32)
        return np.hstack([seq, lik])
    raise ValueError(f"unknown arm {arm}")


def cmd_arm(args: argparse.Namespace) -> int:
    df = load_with_splits()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    y_train = train["y_log1p"].to_numpy(np.float32)
    y_val = val["y_log1p"].to_numpy(np.float32)
    alleles = eligible_alleles(val["allele"], y_val, split="val")
    print(f"train {len(train):,}  val {len(val):,}  "
          f"{len(alleles)} eligible alleles")

    fold = cv_folds(train, N_FOLDS)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for arm in args.arms:
        # ---- head grid: pick (hidden, l2) on validation, 3 seeds each ------
        grid_rows = []
        ckpt0 = args.checkpoints[0]
        Xtr_raw = build_blocks(train, arm, ckpt0)
        Xva_raw = build_blocks(val, arm, ckpt0)
        Xtr, Xva = standardise(Xtr_raw, Xva_raw)
        inner = fold != 0
        for hidden in HIDDEN_GRID:
            for l2 in L2_GRID:
                rhos = []
                for seed in SEEDS:
                    cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                    max_epochs=MAX_EPOCHS, patience=PATIENCE)
                    pred, _ = fit_predict(Xtr[inner], y_train[inner],
                                          Xtr[~inner], y_train[~inner],
                                          Xva, cfg)
                    rhos.append(score(f"{arm}_grid", val["allele"], y_val, pred,
                                      alleles, split="val").median_spearman)
                grid_rows.append({"arm": arm, "hidden": "x".join(map(str, hidden)),
                                  "l2": l2, "rho": float(np.mean(rhos)),
                                  "rho_min": float(min(rhos)),
                                  "rho_max": float(max(rhos))})
                print(f"  grid {arm:9s} h{hidden} l2={l2:<6g} "
                      f"rho={np.mean(rhos):+.4f} "
                      f"[{min(rhos):+.4f}, {max(rhos):+.4f}]")
        best = max(grid_rows, key=lambda r: r["rho"])
        hidden = tuple(int(x) for x in best["hidden"].split("x"))
        l2 = best["l2"]
        edge = l2 in (min(L2_GRID), max(L2_GRID))
        print(f"  {arm}: selected h{hidden} l2={l2:g} of {L2_GRID}"
              f"{'  AT BOUNDARY -- ladder truncated' if edge else '  (interior)'}")
        pd.DataFrame(grid_rows).to_csv(
            REPORT_DIR / f"stage3d_grid_{arm}.csv", index=False)

        # ---- ensemble: 5 folds x len(checkpoints) x 3 seeds ---------------
        members = []
        epochs = []
        t0 = time.perf_counter()
        for ckpt in args.checkpoints:
            Xtr_raw = build_blocks(train, arm, ckpt)
            Xva_raw = build_blocks(val, arm, ckpt)
            Xtr, Xva = standardise(Xtr_raw, Xva_raw)
            for f in range(N_FOLDS):
                hold = fold == f
                for seed in SEEDS:
                    cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                    max_epochs=MAX_EPOCHS, patience=PATIENCE)
                    pred, ep = fit_predict(Xtr[~hold], y_train[~hold],
                                           Xtr[hold], y_train[hold], Xva, cfg)
                    members.append(pred)
                    epochs.append(ep)
                    s = score("m", val["allele"], y_val, pred, alleles,
                              split="val")
                    rows.append({"arm": arm, "level": "member",
                                 "checkpoint": ckpt, "fold": f, "seed": seed,
                                 "hidden": best["hidden"], "l2": l2,
                                 "best_epoch": ep,
                                 "median_spearman": s.median_spearman})
        fit_s = time.perf_counter() - t0
        ens = np.mean(members, axis=0)
        s = score(arm, val["allele"], y_val, ens, alleles, split="val")
        member_med = float(np.median([r["median_spearman"] for r in rows
                                      if r["arm"] == arm
                                      and r["level"] == "member"]))
        rows.append({"arm": arm, "level": "ensemble", "checkpoint": "all",
                     "fold": -1, "seed": -1, "hidden": best["hidden"], "l2": l2,
                     "best_epoch": float(np.mean(epochs)),
                     "median_spearman": s.median_spearman,
                     "mean_spearman": s.mean_spearman,
                     "n_members": len(members), "fit_seconds": round(fit_s, 1)})
        # ``MLPRegressor`` does not expose the early-stopping epoch, so
        # ``epochs`` is all -1 unless a future version adds the attribute;
        # only report it when it is real.
        ep_note = (f", mean best_epoch {np.mean(epochs):.1f}"
                   if min(epochs) >= 0 else "")
        print(f"{arm}: ensemble rho={s.median_spearman:+.4f} "
              f"(members median {member_med:+.4f}, "
              f"n={len(members)}, {fit_s:.0f}s{ep_note})")
        pd.DataFrame({"pair_id": val["pair_id"], "y_pred": ens}).to_csv(
            PRED_DIR / f"stage3d_{arm}.csv", index=False)

    pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)
    print(f"wrote {RUNS_CSV}")

    # ---- paired bootstrap against the stage 2 sequence ensemble -----------
    if args.baseline and Path(args.baseline).exists():
        base = pd.read_csv(args.baseline)
        pred_col = next((c for c in ("y_pred", "y_pred_log1p") if c in base), None)
        if pred_col is None:
            raise ValueError(f"{args.baseline}: no prediction column "
                             f"(found {list(base.columns)})")
        merged = val.merge(base, on="pair_id", how="left", validate="one_to_one")
        if merged[pred_col].isna().any():
            raise ValueError(f"{args.baseline} does not cover every val row")
        base_pred = merged[pred_col].to_numpy(np.float32)
        bs = score("baseline", val["allele"], y_val, base_pred, alleles,
                   split="val")
        print(f"\nbaseline (stage 2 ensemble): rho={bs.median_spearman:+.4f}")
        deltas = []
        for arm in args.arms:
            arm_pred = pd.read_csv(PRED_DIR / f"stage3d_{arm}.csv")
            arm_pred = val.merge(arm_pred, on="pair_id",
                                 validate="one_to_one")["y_pred"].to_numpy()
            res = paired_cluster_bootstrap(val["cluster_id"], val["allele"],
                                           y_val, arm_pred, base_pred,
                                           alleles, split="val")
            print(f"{arm} vs baseline: {describe_delta(res, MIN_WORTHWHILE_DELTA_SPEARMAN)}")
            deltas.append({"arm": arm, **{k: v for k, v in res.items()
                                          if np.isscalar(v)}})
        pd.DataFrame(deltas).to_csv(REPORT_DIR / "stage3d_deltas.csv",
                                    index=False)
        print(f"wrote {REPORT_DIR / 'stage3d_deltas.csv'}")
    else:
        print("\nno baseline predictions supplied -- skipping paired bootstrap")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="masked-marginal likelihood cache")
    e.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINTS))
    e.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    e.add_argument("--batch-size", type=int, default=1024)
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=cmd_extract)

    a = sub.add_parser("arm", help="head grid + 30-network ensemble")
    a.add_argument("--arms", nargs="+", default=["lik", "seq_lik"])
    a.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINTS))
    a.add_argument("--baseline", default=str(PRED_DIR / "seq_ensemble_pep_pseudo.csv"))
    a.set_defaults(func=cmd_arm)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
