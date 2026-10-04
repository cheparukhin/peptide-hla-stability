"""Stage 3c: score a trained stability model against MHC Motif Atlas ligands.

    .venv/bin/python scripts/stage3c_elution_validation.py
    .venv/bin/python scripts/stage3c_elution_validation.py --arms seq_baseline
    .venv/bin/python scripts/stage3c_elution_validation.py --emit-scoring-set \
        reports/stage3c_scoring_set.csv
    .venv/bin/python scripts/stage3c_elution_validation.py --arms esm2 \
        --scores preds/esm2_elution_scores.csv

**Scoring only. Nothing is fitted to the atlas.** The question is whether a
model trained on dissociation half-life ranks mass-spec eluted ligands above
length- and allele-matched human-proteome decoys -- a different assay measuring
a different biological event. The decoy protocol (ratio, seed, caps, exclusion
rules) is frozen in :mod:`pepstab.elution` and was written down before any
score existed.

No model is persisted in this repository, so the model under test is
*reproduced* from its frozen configuration: refit on ``split == "train"``
exactly as ``scripts/baseline_sequence.py`` and ``scripts/baseline_ensemble.py``
do, with the same cluster-respecting inner folds and the same seeds. ``--verify``
checks the refit reproduces ``preds/`` on the validation split before anything
is scored. The atlas is never in a fit set, and the test split is never touched.

The arm is an argument. ``--scores`` takes a CSV of ``allele,peptide,y_pred``
from any external model (the ESM-2 arm, say) scored on the rows that
``--emit-scoring-set`` writes, so a second arm is measured on identical rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import elution as E  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import SELECTED, cv_folds  # noqa: E402
from scripts.baseline_sequence import (  # noqa: E402
    MAX_EPOCHS,
    PATIENCE,
    SEEDS,
    inner_folds,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
PRED_DIR = REPO_ROOT / "preds"

INPUT_SET = "pep_pseudo"

#: Rows per feature-building chunk. 901k scored rows x 860 float32 columns is
#: 3.1 GB in one block; at 100k it is 344 MB, which leaves the machine usable.
CHUNK = 100_000

#: The two reproducible arms. ``seq_baseline`` is the single network stage 2
#: selected (reports/stage2_headline.json); ``seq_ensemble`` is the strong
#: stage 2 baseline, 5 CV folds x 2 encodings x 3 seeds.
BUILTIN_ARMS = ("seq_baseline", "seq_ensemble")


def fit_seq_baseline(df: pd.DataFrame) -> list[tuple[str, MLPRegressor]]:
    """The stage 2 headline single network: onehot pep_pseudo, h256x64, seed 0."""
    train = df[df.split == "train"]
    fold = inner_folds(train)
    is_fit = (fold == "fit").to_numpy()
    X = build_features(train, INPUT_SET, "onehot")
    y = train.y_log1p.to_numpy()
    cfg = MLPConfig(hidden=(256, 64), l2=1e-5, seed=SEEDS[0],
                    max_epochs=MAX_EPOCHS, patience=PATIENCE)
    model = MLPRegressor(cfg).fit(X[is_fit], y[is_fit], X[~is_fit], y[~is_fit])
    return [("onehot", model)]


def fit_seq_ensemble(df: pd.DataFrame) -> list[tuple[str, MLPRegressor]]:
    """The 30-network stage 2 ensemble, member for member."""
    train = df[df.split == "train"]
    fold = cv_folds(train)
    y = train.y_log1p.to_numpy()
    members: list[tuple[str, MLPRegressor]] = []
    for enc in ("onehot", "blosum"):
        hidden, l2 = SELECTED[(INPUT_SET, enc)]
        X = build_features(train, INPUT_SET, enc)
        for k in range(int(fold.max()) + 1):
            is_fit = fold != k
            for seed in SEEDS:
                cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE)
                members.append((enc, MLPRegressor(cfg).fit(
                    X[is_fit], y[is_fit], X[~is_fit], y[~is_fit])))
    return members


FITTERS = {"seq_baseline": fit_seq_baseline, "seq_ensemble": fit_seq_ensemble}


def predict(members: list[tuple[str, MLPRegressor]], frame: pd.DataFrame,
            chunk: int = CHUNK) -> np.ndarray:
    """Mean prediction over members, built and scored in row chunks."""
    encodings = sorted({enc for enc, _ in members})
    total = np.zeros(len(frame), dtype=np.float64)
    for start in range(0, len(frame), chunk):
        block = frame.iloc[start:start + chunk]
        feats = {enc: build_features(block, INPUT_SET, enc) for enc in encodings}
        acc = np.zeros(len(block), dtype=np.float64)
        for enc, model in members:
            acc += model.predict(feats[enc])
        total[start:start + len(block)] = acc / len(members)
    return total


def verify(arm: str, members: list[tuple[str, MLPRegressor]],
           df: pd.DataFrame) -> dict:
    """Check the refit reproduces the committed validation predictions."""
    path = PRED_DIR / ({"seq_baseline": "seq_baseline.csv",
                        "seq_ensemble": "seq_ensemble_pep_pseudo.csv"}[arm])
    if not path.exists():
        return {"checked": False, "reason": f"{path.name} absent"}
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    got = predict(members, val)
    want = pd.read_csv(path).sort_values("pair_id").y_pred.to_numpy()
    return {"checked": True, "file": path.name,
            "max_abs_diff": float(np.max(np.abs(got - want))),
            "pearson": float(np.corrcoef(got, want)[0, 1]),
            "spearman_rank_identical": bool(
                np.array_equal(np.argsort(got), np.argsort(want)))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=list(BUILTIN_ARMS),
                    help=f"arms to score. Built in: {', '.join(BUILTIN_ARMS)}. "
                         "Any other name needs --scores.")
    ap.add_argument("--scores", default=None,
                    help="CSV of allele,peptide,y_pred for an external arm, "
                         "covering every row of the scoring set.")
    ap.add_argument("--emit-scoring-set", default=None,
                    help="write the ligand/decoy table here and exit, so another "
                         "arm can be scored on identical rows.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip reproducing preds/ on the validation split.")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the wrong-allele specificity control.")
    ap.add_argument("--no-swapped", action="store_true",
                    help="skip the allele-swapped decoy control.")
    args = ap.parse_args()

    started = time.perf_counter()
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_with_splits()
    non_c67s = df[~df.allele.str.contains("C67S", regex=False)]
    pseudoseq = dict(zip(non_c67s.allele.astype(str),
                         non_c67s.hla_pseudoseq.astype(str)))

    print(f"atlas      : {E.ATLAS_TXT.relative_to(REPO_ROOT)}")
    print(f"proteome   : {E.PROTEOME_FASTA.relative_to(REPO_ROOT)}")
    print(f"panel      : {len(pseudoseq)} non-C67S alleles with a pseudosequence")
    print(f"protocol   : ratio {E.DECOY_RATIO}:1, seed {E.SEED}, "
          f">={E.MIN_LIGANDS_PER_ALLELE} ligands/allele, "
          f"cap {E.MAX_LIGANDS_PER_ALLELE}\n")

    atlas = E.load_atlas()
    scoring, audit = E.build_scoring_set(
        atlas, set(df.peptide.astype(str)), pseudoseq)
    audit["atlas_sha256"] = sha256(E.ATLAS_TXT)
    audit["atlas_url"] = E.ATLAS_URL
    audit["proteome_sha256"] = sha256(E.PROTEOME_FASTA)
    audit["build_seconds"] = round(time.perf_counter() - started, 1)
    print(f"scoring set: {audit['alleles_scored']} alleles, "
          f"{audit['ligands_scored']:,} ligands, "
          f"{audit['rows_scored'] - audit['ligands_scored']:,} decoys "
          f"({audit['build_seconds']:.0f}s)\n")

    swapped = None
    if not args.no_swapped:
        ligands = E.eligible_atlas_ligands(
            atlas, set(df.peptide.astype(str)), pseudoseq)
        swapped, swap_audit = E.build_swapped_scoring_set(
            ligands, scoring, pseudoseq)
        audit["swapped_decoy_control"] = swap_audit
        print(f"swapped set: same {swap_audit['ligands_scored']:,} ligands, "
              f"{swap_audit['rows_scored'] - swap_audit['ligands_scored']:,} "
              f"decoys drawn from {swap_audit['decoy_pool_distinct_peptides']:,} "
              "eluted ligands of other alleles\n")

    if args.emit_scoring_set:
        out = Path(args.emit_scoring_set)
        frames = [scoring.assign(decoy_set="proteome")]
        if swapped is not None:
            frames.append(swapped.assign(decoy_set="swapped"))
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"wrote {out}; nothing scored.")
        return 0

    per_allele_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []
    verifications: dict[str, dict] = {}

    for arm in args.arms:
        t0 = time.perf_counter()
        if arm in FITTERS:
            print(f"arm {arm}: refitting from the frozen stage 2 configuration")
            members = FITTERS[arm](df)
            print(f"  {len(members)} network(s) in {time.perf_counter() - t0:.0f}s")
            if not args.no_verify:
                verifications[arm] = verify(arm, members, df)
                v = verifications[arm]
                if v.get("checked"):
                    print(f"  reproduces {v['file']}: max|diff|="
                          f"{v['max_abs_diff']:.2e} r={v['pearson']:.6f}")
            def score_rows(frame, _m=members):
                return predict(_m, frame)
        else:
            if not args.scores:
                raise SystemExit(f"arm {arm!r} is not built in; pass --scores")
            ext = pd.read_csv(args.scores).drop_duplicates(["allele", "peptide"])
            members = []

            def score_rows(frame, _e=ext):
                merged = frame[["allele", "peptide"]].merge(
                    _e[["allele", "peptide", "y_pred"]],
                    on=["allele", "peptide"], how="left", validate="many_to_one")
                if merged.y_pred.isna().any():
                    raise SystemExit(
                        f"--scores does not cover the scoring set "
                        f"({int(merged.y_pred.isna().sum())} rows missing). "
                        "Generate it from --emit-scoring-set.")
                return merged.y_pred.to_numpy(dtype=float)

        y_pred = score_rows(scoring)

        table = E.score_panel(scoring, y_pred)
        per_allele_tables[arm] = table
        summaries.append({"arm": arm, "control": "none",
                          **flatten(E.summarise_panel(table))})
        print(f"  AUROC median {table.auroc.median():.4f} "
              f"[{table.auroc.quantile(.25):.4f}, {table.auroc.quantile(.75):.4f}]"
              f"   AUPRC median {table.auprc.median():.4f} "
              f"(chance {table.auprc_baseline.iloc[0]:.4f})"
              f"   ({time.perf_counter() - t0:.0f}s)")

        if swapped is not None:
            y_swap = score_rows(swapped)
            stab = E.score_panel(swapped, y_swap)
            per_allele_tables[f"{arm}__swapped_decoys"] = stab
            summaries.append({"arm": arm, "control": "swapped_decoys",
                              **flatten(E.summarise_panel(stab))})
            print(f"  allele-swapped decoys: AUROC median {stab.auroc.median():.4f} "
                  f"[{stab.auroc.quantile(.25):.4f}, {stab.auroc.quantile(.75):.4f}]"
                  f"   AUPRC median {stab.auprc.median():.4f}")
            grad = E.score_by_donor_distance(swapped, y_swap)
            grad.to_csv(REPORT_DIR / f"stage3c_donor_distance_{arm}.csv", index=False)
            near, far = grad[grad.bin == 0], grad[grad.bin == 1]
            print(f"    donor distance <= {near.donor_distance_high.iloc[0]:.0f}: "
                  f"AUROC median {near.auroc.median():.4f}   "
                  f"> {near.donor_distance_high.iloc[0]:.0f}: "
                  f"{far.auroc.median():.4f}")

        if members and not args.no_control:
            swap_map = E.derangement(sorted(scoring.allele.unique()))
            shuffled = scoring.copy()
            shuffled["hla_pseudoseq"] = shuffled.allele.map(swap_map).map(pseudoseq)
            ctrl = E.score_panel(scoring, predict(members, shuffled))
            per_allele_tables[f"{arm}__wrong_allele"] = ctrl
            summaries.append({"arm": arm, "control": "wrong_allele_pseudoseq",
                              **flatten(E.summarise_panel(ctrl))})
            print(f"  wrong-allele pseudosequence: AUROC median "
                  f"{ctrl.auroc.median():.4f}")

    for name, table in per_allele_tables.items():
        path = REPORT_DIR / f"stage3c_per_allele_{name}.csv"
        table.sort_values("allele").to_csv(path, index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(REPORT_DIR / "stage3c_summary.csv", index=False)
    audit["runtime_seconds"] = round(time.perf_counter() - started, 1)
    audit["verification"] = verifications
    audit["arms"] = args.arms
    (REPORT_DIR / "stage3c_provenance.json").write_text(
        json.dumps(audit, indent=2) + "\n")

    print(f"\nwrote reports/stage3c_summary.csv, "
          f"{len(per_allele_tables)} per-allele tables, "
          f"reports/stage3c_provenance.json")
    print(f"total runtime: {audit['runtime_seconds'] / 60:.1f} min")
    return 0


def flatten(summary: dict) -> dict:
    """``{"auroc": {"median": ...}}`` -> ``{"auroc_median": ...}``."""
    out = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            out.update({f"{key}_{k}": v for k, v in value.items()})
        else:
            out[key] = value
    return out


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
