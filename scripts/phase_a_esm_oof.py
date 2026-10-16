"""Phase A arm A5 component: the ESM-2 arm, refitted on cluster folds.

    .venv/bin/python scripts/phase_a_esm_oof.py
    .venv/bin/python scripts/phase_a_esm_oof.py --quick     # pilot, *_partial files
    .venv/bin/python scripts/phase_a_esm_oof.py --out-dir /tmp/esm

A5 blends the sequence reference with a tree arm and with an ESM arm, with
non-negative weights fitted on inner out-of-fold predictions only. This script
produces the ESM component of that blend under the **same** nested protocol
``scripts/phase_a_a0.py`` used for the sequence component, so the blend code can
treat the two arms identically.

``preds/esm_ensemble_oof.csv`` cannot be used. It was built on
``pepstab.stage6.nested_folds``, which partitions by *peptide* and deliberately
places the two members of a point-mutant pair in different folds — the property
its mutant-ranking diagnostic needs, and the wrong one here. Phase A assesses on
whole-``cluster`` folds, where a held-out peptide stays at least 4 substitutions
from every peptide its model saw. The two schemes are not interchangeable, so
the ESM arm is regenerated (prespecification §1 and the 4 Oct decision log
entry).

**Ensemble arity, and which axes reach it.** A0's ensemble is 5 inner folds x 2
encodings x 3 seeds = 30 members per outer fold. This arm multiplies

    5 inner folds  x  2 ESM-2 layers (mid, final)  x  3 seeds  =  30

The ESM layer axis is not an improvised substitute: it is the two-way axis
stage 3 declared for exactly this purpose, where the frozen-representation arm
spread its 30 networks over {middle layer, final layer} against stage 2's
{one-hot, BLOSUM62} (``scripts/esm_arm.py::mode_ensemble``, ``--axis layer``,
and ``reports/stage3_headline.json``'s ``ensemble.axis``). Member count, fold
protocol, seeds and head family therefore match A0 exactly; only what the
networks are fed differs. Ensembling is worth roughly +0.07 to +0.09 median
Spearman here — larger than most effects Phase A is looking for — so an arity
mismatch between two blend components would silently become the blend's main
effect.

Three prediction roles, with A0's semantics (prespecification §A0):

- ``outer_held`` — the 30-member mean for rows in outer fold *f*, from members
  fitted on the other four outer folds. One per ``pair_id``.
- ``inner_oof_6`` — the mean of the 6 members (2 layers x 3 seeds) that
  early-stopped on the inner fold holding that row. Four per ``pair_id``.
- ``inner_oof_30`` — an arity-matched 30-member mean per inner fold, which needs
  a third nesting level: five sub-folds of the inner fitting partition x 2
  layers x 3 seeds. Four per ``pair_id``.

``fit_mlp_ensemble`` cannot be called for this arm — its ``input_set`` routes to
``pepstab.features.build_features``, which only knows one-hot and BLOSUM — so
:func:`fit_esm_ensemble` mirrors that loop member for member. The fold structure
is *inherited*: ``outer_folds`` and ``inner_folds`` are imported from
``pepstab.nested`` rather than recomputed, and the resulting outer assignment is
asserted equal to the ``phase_a_outer_folds.csv`` A0 wrote.

No GPU and no ESM inference. The embeddings are read from the ``features/esm/``
cache; re-extracting them would be wasted spend. The standardisation and PCA on
top of them *are* refitted inside every fitting partition, at every nesting
level — ``assemble`` fits on whichever frame it is handed, and it is handed only
that partition's rows, so no PCA basis ever sees a row it will later predict.

Training split only. The frozen validation and test splits are never loaded.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread *before* numpy is imported -- the thread pool is sized
# at import time, so setting these afterwards does nothing. Thread count changes
# float32 results here: the same seed at a different thread count moves a single
# network by roughly 0.02 median Spearman, and several workstreams compute on
# this machine concurrently, which makes that a nondeterministic offset rather
# than a fixed one. Pinned, and recorded in the provenance file.
# VECLIB_MAXIMUM_THREADS is the one that matters on macOS, where numpy links
# against Accelerate; the others cover Linux/CI.
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _var in BLAS_THREAD_VARS:
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import esm as pesm  # noqa: E402
from pepstab import nested  # noqa: E402
from pepstab.data import RAW_CSV, SPLITS_CSV, TARGET_RAW, load_with_splits  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402
from scripts.esm_arm import (  # noqa: E402
    HLA_PCA, PEP_PCA, SWEEP_CONFIG, rep_label)
# The worker-side code lives in the Modal app, which is its natural home and
# the single definition the containers run. Importing it here means --quick
# exercises the exact functions the batch does, instead of a local twin that
# could drift from them.
from modal_app.phase_a_esm import (  # noqa: E402
    HLA_REP, LAYERS, N_TRAIN_ROWS, NO_SUB_FOLD, PAYLOAD_SCHEMA_VERSION,
    N_HELD_ROWS_PER_OUTER_FOLD, PEP_REP, SENSITIVITY_MIN_ROWS,
    assemble_from_payload, compute_payload_digest, fit_esm_ensemble,
    read_payload)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reports" / "further_research"
CACHE_DIR = REPO_ROOT / "features" / "esm"

#: The stage 3 selection, read back from the artifact that recorded it rather
#: than retyped. ``reconcile_selection`` asserts every value below against it.
STAGE3_HEADLINE = REPO_ROOT / "reports" / "stage3_headline.json"

#: ``inner_fold`` for an outer-held row in ``phase_a_fold_assignments.csv``.
OUTER_HELD_INNER_FOLD = -1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reconcile_selection() -> dict:
    """The stage 3 selected configuration, checked against its own artifact.

    Stage 3 found frozen ESM-2 at parity with the sequence baseline and ruled
    out the +0.05 bar, so there is nothing to re-tune here and re-tuning would
    be a second difference inside A5's blend. The configuration is taken as
    selected; what this function does is refuse to proceed if the constants in
    ``scripts/esm_arm.py`` have drifted from what ``stage3_headline.json``
    recorded as chosen.
    """
    headline = json.loads(STAGE3_HEADLINE.read_text())
    rep, head, ens = (headline["representation_selected"], headline["head"],
                      headline["ensemble"])
    hidden, l2 = SWEEP_CONFIG
    expected = {
        "peptide_pca_components": PEP_PCA, "hla_pca_components": HLA_PCA,
    }
    for key, want in expected.items():
        if rep[key] != want:
            raise AssertionError(f"stage3_headline.json selected {key}={rep[key]}, "
                                 f"but scripts/esm_arm.py uses {want}")
    if tuple(head["hidden"]) != tuple(hidden) or head["l2"] != l2:
        raise AssertionError(
            f"stage3_headline.json selected hidden={head['hidden']} l2={head['l2']}, "
            f"but scripts.esm_arm.SWEEP_CONFIG is {SWEEP_CONFIG}")
    if (head["lr"], head["max_epochs"], head["patience"]) != (
            nested.REFERENCE_MLP.lr, MAX_EPOCHS, PATIENCE):
        raise AssertionError(
            f"stage 3's head was lr={head['lr']} max_epochs={head['max_epochs']} "
            f"patience={head['patience']}, which is not the shared reference "
            f"member lr={nested.REFERENCE_MLP.lr} {MAX_EPOCHS}/{PATIENCE}")
    if tuple(ens["seeds"]) != tuple(SEEDS) or ens["n_members"] != 30:
        raise AssertionError(f"stage 3's ensemble was {ens['n_members']} members at "
                             f"seeds {ens['seeds']}, not 30 at {list(SEEDS)}")
    # The two representation choices the headline records only as prose.
    if "per-position" not in rep["peptide"] or "contact" not in rep["hla"]:
        raise AssertionError(f"stage 3 selected peptide={rep['peptide']!r} "
                             f"hla={rep['hla']!r}, not per-position / contact")
    return {
        "checkpoint": headline["checkpoint_selected"],
        "pep_rep": PEP_REP, "hla_rep": HLA_REP, "layers": list(LAYERS),
        "pep_pca": PEP_PCA, "hla_pca": HLA_PCA,
        "hidden": list(hidden), "l2": l2,
        "source": "reports/stage3_headline.json, selected at stage 3 on validation",
        "stage3_verdict": headline["headline"]["esm_only"]["verdict"],
        "axis_declared_at_stage3": ens["axis"],
    }


def cache_digests(checkpoint: str) -> dict[str, str]:
    """Digest every cache file this arm reads, plus the index content hashes.

    The index's own ``content_hash`` pins the *sequence set*; the npz digests
    pin the embeddings. Both are recorded because a stale cache keyed to the
    right sequences would otherwise pass unnoticed.
    """
    folder = CACHE_DIR / checkpoint
    layer_ids = pesm.layer_ids(checkpoint)
    digests: dict[str, str] = {}
    for kind in ("peptide", "hla"):
        index = folder / f"{kind}.index.json"
        digests[str(index.relative_to(REPO_ROOT))] = sha256(index)
        digests[f"{kind}.index.json::content_hash"] = json.loads(
            index.read_text())["content_hash"]
        for layer in LAYERS:
            npz = folder / f"{kind}_L{layer_ids[layer]}.npz"
            digests[str(npz.relative_to(REPO_ROOT))] = sha256(npz)
    return digests


def build_payload(train: pd.DataFrame, config: dict) -> dict:
    """The embedding blocks this arm needs, for the training sequences only.

    Written for the Modal run, where ``features/esm/`` does not exist. Two
    reductions, both free:

    - **Training sequences only.** 3,942 of the cache's 5,633 peptides appear
      in the training split. The frozen validation and test peptides are left
      out, so the artifact a container is handed *cannot* contain one — the one
      unrecoverable mistake available to this arm becomes impossible rather
      than merely avoided.
    - **The HLA block is pre-sliced to its 34 contact positions**, which is
      what ``hla_rep="contact"`` selects anyway: 34 of 182 residues, so 4.9 MB
      a layer instead of 26.2 MB, with nothing discarded that the selected
      representation would have read.

    The peptide block cannot be reduced further. The selected representation is
    **per-position** (9 x 480), not mean-pooled, and the PCA on top of it is
    refitted inside every fitting partition, so the raw 4,320 columns have to
    travel. 146 MB for both layers, measured — a mean-pooled payload would have
    been 15 MB, which is where the "unique sequences are small" intuition comes
    from and why it does not hold here.

    Sequences are kept in the cache's own row order. ``assemble``'s
    standardisation sums over whichever rows the fitting partition selects, so
    preserving the order keeps that summation order identical and makes the
    payload path bit-identical to the local path rather than merely close;
    ``tests/test_phase_a_esm_oof.py`` asserts the equality.
    """
    blocks: dict[str, np.ndarray] = {}
    index: dict[str, dict[str, int]] = {}
    keep = np.asarray(pesm.CONTACT_POSITIONS_1BASED) - 1
    for kind, column in (("peptide", "peptide"), ("hla", "hla_seq")):
        present = set(train[column].astype(str).unique())
        for layer in LAYERS:
            lookup, reps = pesm.load_cache(config["checkpoint"], kind, layer)
            rows = sorted(row for seq, row in lookup.items() if seq in present)
            sub = reps[rows]
            if kind == "hla":
                sub = sub[:, keep, :]
            blocks[f"{kind}_{layer}"] = np.ascontiguousarray(sub)
            inverse = {row: seq for seq, row in lookup.items()}
            index[kind] = {inverse[row]: i for i, row in enumerate(rows)}
            del reps
            gc.collect()
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "checkpoint": config["checkpoint"],
        "layers": list(LAYERS),
        # "contact" is already applied to the stored HLA block, so a reader must
        # not slice it a second time; recorded so it cannot be guessed.
        "hla_rep_applied": HLA_REP,
        "pep_rep": PEP_REP,
        "blocks": blocks,
        "index": index,
        "n_sequences": {kind: len(index[kind]) for kind in index},
        "bytes": int(sum(b.nbytes for b in blocks.values())),
        # Recomputed by read_payload from the bytes it loads, so a partial
        # upload of this 146 MB artifact fails in the container rather than
        # quietly giving it a different PCA basis.
        "digest": compute_payload_digest(blocks),
    }


def write_payload(path: Path, payload: dict) -> dict:
    """Save a payload as one ``.npz``, and return what it cost on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in payload.items() if k not in ("blocks", "index")}
    np.savez(path, meta=json.dumps(meta),
             index=json.dumps(payload["index"]), **payload["blocks"])
    return {"path": str(path), "disk_bytes": path.stat().st_size,
            "in_memory_bytes": payload["bytes"], "digest": payload["digest"]}


def a0_comparator(out_dir: Path) -> dict:
    """A0's nested figures, read from its provenance rather than transcribed.

    The comparison that matters is against the sequence reference under the
    *same* protocol, and A0's own file is the only authority on what that is.

    The spread across A0's five folds is ~0.048 **with the model held fixed** --
    essentially the whole 0.05 worthwhile bar, from fold assignment alone. So a
    single-fold difference between the two arms means nothing, and this returns
    the per-fold vector so the write-up cannot quote only a pooled number.
    """
    path = out_dir / "phase_a_a0_provenance.json"
    if not path.exists():
        return {"available": False,
                "note": f"{path.name} not present; no comparator"}
    a0 = json.loads(path.read_text())
    per_fold = [f["median_spearman"] for f in a0["fold_scores"]]
    # The macOS A0 is the shipped reference but it is the only macOS-fitted arm
    # in the study. Every Linux-fitted arm -- this one included -- should be
    # read against the Linux A0, which removes the platform term. Neither
    # removes Modal's run-to-run floor; see phase_a_platform_artifact.md.
    linux_path = out_dir / "phase_a_a0_provenance_linux.json"
    linux: dict = {"available": False}
    if linux_path.exists():
        lx = json.loads(linux_path.read_text())
        lx_folds = [f["median_spearman"] for f in lx["fold_scores"]]
        linux = {
            "available": True,
            "source": str(linux_path.relative_to(REPO_ROOT)),
            "five_fold_median_of_medians": lx["five_fold_median_of_medians"],
            "per_fold_median_spearman": lx_folds,
            "platform": lx["platform"]["where"],
            "threads": lx["platform"]["threads"],
            "note": ("the within-platform comparator for any Linux-fitted arm; "
                     "prefer it over the macOS figures above"),
        }
    return {
        "available": True,
        "source": str(path.relative_to(REPO_ROOT)),
        "platform": "macOS/Accelerate arm64, threads=1",
        "a0_linux": linux,
        "reproducibility_floor": (
            "Modal run-to-run on identical settings moved median-of-medians by "
            "0.0043 with 3 of 5 folds bit-identical (host microarchitecture "
            "heterogeneity; phase_a_platform_artifact.md). A difference of "
            "<= 0.005 between singly-run arms is not resolvable."),
        "five_fold_median_of_medians": a0["five_fold_median_of_medians"],
        "per_fold_median_spearman": per_fold,
        "fold_spread": max(per_fold) - min(per_fold),
        "networks_total": a0["networks_total"],
        "members_per_outer_fold": a0["members_per_outer_fold"],
        "caveat": ("A0's fold spread is ~0.048 with the model held fixed, which "
                   "is essentially the whole 0.05 worthwhile bar from fold "
                   "assignment alone. Read the per-fold vectors, not a single "
                   "fold difference."),
    }


def inner_oof_residual_scale(oof: pd.DataFrame, train: pd.DataFrame) -> dict:
    """How far each inner role sits from the label, relative to ``outer_held``.

    Prespecification §A0 ships ``inner_oof_6`` alongside ``inner_oof_30``
    precisely so this ratio can be *measured* rather than argued about. On A0,
    measured on measured rows only, it came out ``outer_held`` 0.6944,
    ``inner_oof_6`` 0.7086 (+2.0%) and ``inner_oof_30`` 0.7176 (+3.3%): both
    inner roles are **pessimistic** relative to the target. They bound the
    out-of-sample residual scale from one side only and do **not** bracket it.

    **Computed un-averaged, and that is not a detail.** Collapsing each
    ``pair_id``'s four inner replicates to their mean before taking the RMSE
    puts A0's ``inner_oof_6`` at 0.6845 -- *below* the target -- which reverses
    the sign and manufactures a bracket that does not exist. A replicate
    average is an ensemble-of-ensembles no stacker ever sees, so the averaged
    figure is computed here only to record how badly it misleads, and is never
    reported as a ratio.

    Nothing here says which role is the better stacking input. ``inner_oof_6``
    is optimistic by construction in a different way -- its members early-
    stopped on the rows they predict -- and ``inner_oof_30`` pays a training-row
    penalty (~10,094 fit rows against ~12,618). The question is open and A5
    settles it on a paired blend outcome.

    The headline is restricted to **measured rows** (``thalf_hours != 0``), the
    same restriction A0's figures use, so the two arms are comparable; the
    all-rows figure is reported beside it.
    """
    labels = train.set_index("pair_id")
    y = labels.y_log1p
    measured = labels[labels[TARGET_RAW] != 0].index

    def rmse(values: np.ndarray, pairs: np.ndarray) -> float:
        err = values - y.reindex(pairs).to_numpy()
        return float(np.sqrt(np.mean(err ** 2)))

    out: dict = {}
    for role in ("outer_held", "inner_oof_6", "inner_oof_30"):
        rows = oof[oof.role == role]
        on_measured = rows[rows.pair_id.isin(measured)]
        collapsed = rows.groupby("pair_id").y_pred.mean()
        collapsed = collapsed[collapsed.index.isin(measured)]
        out[role] = {
            "n_rows": len(rows),
            "n_rows_measured": len(on_measured),
            "rmse_measured_rows": rmse(on_measured.y_pred.to_numpy(),
                                       on_measured.pair_id.to_numpy()),
            "rmse_all_rows": rmse(rows.y_pred.to_numpy(),
                                  rows.pair_id.to_numpy()),
            # Recorded to show the hazard, deliberately not turned into a
            # ratio: averaging replicates reversed this comparison's sign on A0.
            "rmse_replicate_averaged_DO_NOT_USE": rmse(
                collapsed.to_numpy(), collapsed.index.to_numpy()),
        }
    base = out["outer_held"]["rmse_measured_rows"]
    base_all = out["outer_held"]["rmse_all_rows"]
    for role in ("inner_oof_6", "inner_oof_30"):
        out[role]["rmse_ratio_measured_rows"] = (
            out[role]["rmse_measured_rows"] / base)
        out[role]["rmse_ratio_all_rows"] = out[role]["rmse_all_rows"] / base_all
    out["method"] = (
        "un-averaged: every inner row enters as it ships, 4 per pair_id. "
        "Averaging a pair_id's replicates first is an ensemble-of-ensembles no "
        "stacker sees, and on A0 it moved inner_oof_6 from above the target to "
        "below it -- reversing the sign and fabricating a bracket.")
    out["interpretation"] = (
        "A ratio above 1 means the inner role sits further from the label than "
        "outer_held does. On A0 both inner roles are pessimistic, so they bound "
        "the out-of-sample residual scale from ONE SIDE ONLY and do not bracket "
        "it. inner_oof_6's members early-stopped on the rows they predict; "
        "inner_oof_30 is fitted on ~10,094 rows against ~12,618. Which is the "
        "better stacking input is OPEN and A5 decides it on a paired blend "
        "outcome, not on this ratio.")
    return out


def collect_from_journal(path: Path, train: pd.DataFrame, n_inner: int,
                         n_sub: int, full_arity: int):
    """Rebuild the contract from the Modal run's journal, refitting nothing.

    The journal is the run's record of truth, not a convenience: rows
    accumulate on the client and the A1-A4 screen lost its gRPC connection
    mid-``.map()`` twice, once capturing 8 units of 3,360. So every unit's full
    return was written with ``write``/``flush``/``fsync`` as it landed, and this
    function reconstructs the three contract files from those lines alone.

    The fit/predict disjointness checks are redone **here** rather than trusted
    from the container: they are properties of the partition, not of the
    predictions, so the client can verify every member's hold-out without
    refitting anything.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    # Last line per key wins. A resumed journal legitimately holds a unit's
    # earlier *failure* followed by its later success -- that is what --resume
    # produces -- so the ok check is on the final state per key, never on every
    # line. Checking each line would reject exactly the runs this path exists
    # to rescue.
    by_key: dict[str, dict] = {}
    for row in rows:
        by_key[row["key"]] = row
    failed = [r for r in by_key.values() if not r.get("ok")]
    if failed:
        raise AssertionError(
            f"{len(failed)} unit(s) in {path.name} are still failed: "
            f"{[r.get('key') for r in failed][:5]}. Rerun them with --resume "
            f"before assembling; a contract file missing a fold is worse than "
            f"no file.")
    superseded = len(rows) - len(by_key)
    if superseded:
        print(f"  {superseded} superseded journal line(s) ignored "
              f"(earlier attempts of units that later succeeded)")
    expected = {f"L2_o{f}" for f in range(nested.OUTER_FOLDS)} | {
        f"L3_o{f}_i{k}" for f in range(nested.OUTER_FOLDS)
        for k in range(n_inner)}
    if set(by_key) != expected:
        raise AssertionError(
            f"journal holds {len(by_key)} units, expected {len(expected)}; "
            f"missing {sorted(expected - set(by_key))[:5]}, "
            f"unexpected {sorted(set(by_key) - expected)[:5]}")

    # Payload digests exist only for an arm that reads a Volume artifact. The
    # A0 Linux runner has no payload -- build_features is row-local -- so an
    # absent digest is a property of the arm, not a defect; a *disagreeing*
    # one is always a defect.
    digests = {r["payload_digest"] for r in by_key.values()
               if r.get("payload_digest")}
    if len(digests) > 1:
        raise AssertionError(f"units read {len(digests)} different payloads: "
                             f"{digests}")
    arities = {r["n_members"] for r in by_key.values()}
    if arities != {full_arity}:
        raise AssertionError(f"unit arities {sorted(arities)}, expected every "
                             f"unit at A0's {full_arity} members")
    # The pooled-fit guard, repeated on the client because this is where the
    # contract files are written. Phase B's leak was a *row count*, not a
    # metric: a map fitted on pooled inner rows across all five outer folds saw
    # every scored row's own label and still matched the clean version's Brier
    # to four decimals. No member may have seen a whole fitting partition, let
    # alone the pooled split.
    outer_fit_sizes = {N_TRAIN_ROWS - h for h in N_HELD_ROWS_PER_OUTER_FOLD}
    for row in by_key.values():
        for member in row["members"]:
            n = member["n_fit_rows"]
            if n == N_TRAIN_ROWS or n in outer_fit_sizes:
                what = ("the pooled training split" if n == N_TRAIN_ROWS
                        else "a whole outer fitting partition")
                raise AssertionError(
                    f"{row['key']}: a member reports {n:,} fit rows, which is "
                    f"{what} -- every member must hold out its own inner fold")
            if n <= 0:
                raise AssertionError(f"{row['key']}: member with {n} fit rows")
    held_total = sum(r["n_predicted"] for r in by_key.values()
                     if r["level"] == 2)
    if held_total != N_TRAIN_ROWS:
        raise AssertionError(
            f"level-2 units predicted {held_total:,} rows in total, expected "
            f"{N_TRAIN_ROWS:,}: the five outer folds must partition the split")

    unpinned = [r["key"] for r in by_key.values() if not r.get("thread_pin_ok")]
    if unpinned:
        raise AssertionError(
            f"{len(unpinned)} unit(s) ran at a thread count other than the "
            f"pinned one: {unpinned[:5]}. Their float32 results are not "
            f"reproducible against the rest.")

    held_frames: list[pd.DataFrame] = []
    inner_frames: list[pd.DataFrame] = []
    runs: list[dict] = []
    fold_scores: list[dict] = []
    n_features = 0
    for row in by_key.values():
        n_features = row.get("n_features", 0)
        runs.extend({"outer_fold": row["outer_fold"], **m} for m in row["members"])
        if row["level"] == 2:
            held_frames.append(pd.DataFrame({
                "pair_id": row["pair_id"], "outer_fold": row["outer_fold"],
                "role": "outer_held", "y_pred": row["y_pred"]}))
            inner_frames.append(pd.DataFrame({
                "pair_id": row["inner_pair_id"], "outer_fold": row["outer_fold"],
                "role": "inner_oof_6", "y_pred": row["inner_oof_6"]}))
            fold_scores.append({
                "outer_fold": row["outer_fold"],
                "n_alleles_eligible": row["n_alleles_eligible"],
                "n_rows": row["n_predicted"],
                "eligible_row_coverage": row["eligible_row_coverage"],
                "median_spearman": row["median_per_allele_spearman"],
                "iqr_low": row["iqr_low"], "iqr_high": row["iqr_high"],
                "mae_log1p": row["mae_log1p"],
                "n_alleles_unranked": row["n_alleles_unranked"],
                "median_spearman_min50": row["median_per_allele_spearman_min50"],
                "n_alleles_eligible_min50": row["n_alleles_eligible_min50"],
                "eligible_row_coverage_min50": row["eligible_row_coverage_min50"]})
        else:
            inner_frames.append(pd.DataFrame({
                "pair_id": row["pair_id"], "outer_fold": row["outer_fold"],
                "role": "inner_oof_30", "y_pred": row["y_pred"]}))
    fold_scores.sort(key=lambda r: r["outer_fold"])

    n_checks = 0
    for split in nested.iter_outer(train, n_inner):
        fit_rows = train[split.fit_mask].reset_index(drop=True)
        fit_pairs = fit_rows.pair_id.to_numpy()
        held_pairs = train.loc[split.held_mask, "pair_id"].to_numpy()
        for k in range(n_inner):
            is_fit = split.inner != k
            assert_no_overlap(f"outer {split.fold} / inner {k}",
                              fit_pairs[is_fit],
                              np.concatenate([held_pairs, fit_pairs[~is_fit]]))
            n_checks += 1
            sub_fit = fit_rows[is_fit].reset_index(drop=True)
            target_pairs = fit_rows.loc[~is_fit, "pair_id"].to_numpy()
            sub = nested.inner_folds(sub_fit, n_sub)
            sub_pairs = sub_fit.pair_id.to_numpy()
            for s in range(n_sub):
                assert_no_overlap(
                    f"outer {split.fold} / inner {k} / sub {s}",
                    sub_pairs[sub != s],
                    np.concatenate([target_pairs, sub_pairs[sub == s]]))
                n_checks += 1
    unit_seconds = [r["unit_seconds"] for r in by_key.values()]
    threads = {r.get("threads_observed") for r in by_key.values()}
    if len(threads) != 1:
        raise AssertionError(f"units ran at mixed thread counts {threads}; "
                             f"their float32 results are not comparable")
    timing = {"n_units": len(by_key),
              "container_threads_observed": sorted(threads)[0],
              "unit_seconds_total": round(sum(unit_seconds), 1),
              "unit_seconds_max": round(max(unit_seconds), 1),
              "unit_seconds_median": round(
                  float(np.median(unit_seconds)), 1),
              "superseded_journal_lines": superseded}
    return (held_frames, inner_frames, runs, fold_scores, n_checks, n_features,
            sorted(digests)[0] if digests else "", timing)


def assert_no_overlap(label: str, fit_pairs: np.ndarray,
                      predicted_pairs: np.ndarray) -> None:
    """No member may be fitted on a row it predicts."""
    overlap = set(fit_pairs.tolist()) & set(predicted_pairs.tolist())
    if overlap:
        raise AssertionError(
            f"{label}: {len(overlap)} pair_id(s) are both fitted and predicted, "
            f"first {sorted(overlap)[:5]}")


def fit_outer_fold(split: nested.OuterSplit, train: pd.DataFrame, n_sub: int,
                   seeds: tuple[int, ...], config: dict,
                   payload: dict | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], int, int]:
    """Both nesting levels for one outer fold, mirroring A0's own routine.

    Returns the 30-member ``outer_held`` mean, the ``inner_oof_6`` mean, the
    ``inner_oof_30`` mean, the member records, the number of fit/predict
    disjointness checks performed, and the feature width.
    """
    fit_rows = train[split.fit_mask].reset_index(drop=True)
    held_rows = train[split.held_mask].reset_index(drop=True)
    fit_pairs = fit_rows.pair_id.to_numpy()
    held_pairs = held_rows.pair_id.to_numpy()
    n_inner = int(split.inner.max()) + 1
    runs: list[dict] = []
    checks = 0

    # Level 2: the arity-matched ensemble. Each member stops on its own inner
    # fold, so this is stage 3's ESM-only recipe at A0's fold protocol.
    for k in range(n_inner):
        is_fit = split.inner != k
        assert_no_overlap(f"outer {split.fold} / inner {k}", fit_pairs[is_fit],
                          np.concatenate([held_pairs, fit_pairs[~is_fit]]))
        checks += 1
    level2 = fit_esm_ensemble(fit_rows, [held_rows], split.inner, LAYERS,
                              seeds, config, payload=payload)
    runs.extend({"outer_fold": split.fold, "sub_fold": NO_SUB_FOLD, **record}
                for record in level2["members"])

    # Level 3: a full 30-member mean per inner fold, so A5 fitting blend weights
    # on inner predictions and applying them to outer_held is not crossing an
    # arity boundary. One more nesting level, one more fifth of the rows gone.
    inner30 = np.full(len(fit_rows), np.nan)
    for k in range(n_inner):
        is_fit = split.inner != k
        sub_fit = fit_rows[is_fit].reset_index(drop=True)
        target = fit_rows[~is_fit].reset_index(drop=True)
        sub = nested.inner_folds(sub_fit, n_sub)
        sub_pairs = sub_fit.pair_id.to_numpy()
        for s in range(n_sub):
            assert_no_overlap(f"outer {split.fold} / inner {k} / sub {s}",
                              sub_pairs[sub != s],
                              np.concatenate([target.pair_id.to_numpy(),
                                              sub_pairs[sub == s]]))
            checks += 1
        level3 = fit_esm_ensemble(sub_fit, [target], sub, LAYERS, seeds, config,
                                  payload=payload)
        inner30[~is_fit] = level3["predictions"][0]
        runs.extend({"outer_fold": split.fold, "inner_fold": k,
                     "sub_fold": record.pop("inner_fold"), **record}
                    for record in level3["members"])

    if not np.isfinite(inner30).all():
        raise AssertionError(f"outer fold {split.fold}: some fitting-partition "
                             "row received no inner_oof_30 prediction")
    return (level2["predictions"][0], level2["inner_oof"], inner30, runs, checks,
            level2["n_features"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where the three contract files are written")
    ap.add_argument("--write-payload", type=Path, default=None,
                    help="build the training-only embedding artifact for the "
                         "Modal run at this path and exit, fitting nothing. "
                         "features/esm/ does not exist in the container image.")
    ap.add_argument("--from-journal", type=Path, default=None,
                    help="assemble the three contract files from a Modal run's "
                         "*.partial.jsonl instead of fitting locally. The "
                         "journal is the run's record of truth: the client is "
                         "assumed to die, so every unit's full return was "
                         "fsynced as it landed and nothing here refits.")
    ap.add_argument("--quick", action="store_true",
                    help="one outer fold at 2 inner folds, 2 sub-folds and 1 "
                         "seed, written to *_partial filenames. Per CLAUDE.md, "
                         "the end-to-end check before the full run; the "
                         "canonical files are left alone because a partial run "
                         "silently overwriting them would leave A5 blending "
                         "against predictions that are no longer on disk.")
    args = ap.parse_args()

    n_inner = 2 if args.quick else nested.ASSESS_INNER_FOLDS
    n_sub = 2 if args.quick else nested.ASSESS_INNER_FOLDS
    seeds = SEEDS[:1] if args.quick else SEEDS
    suffix = "_partial" if args.quick else ""

    config = reconcile_selection()
    digests = cache_digests(config["checkpoint"])

    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    if len(train) != N_TRAIN_ROWS:
        raise AssertionError(f"training split has {len(train):,} rows, "
                             f"expected {N_TRAIN_ROWS:,}")

    if args.write_payload is not None:
        t0 = time.perf_counter()
        payload = build_payload(train, config)
        written = write_payload(args.write_payload, payload)
        print(f"payload: {payload['n_sequences']} sequences, "
              f"{payload['bytes'] / 1e6:.1f} MB in memory, "
              f"{written['disk_bytes'] / 1e6:.1f} MB on disk, "
              f"{time.perf_counter() - t0:.1f}s")
        print(f"digest {payload['digest'][:16]}  ->  {args.write_payload}")
        print("training sequences only: the frozen val and test peptides are "
              "not in this artifact.")
        return 0

    per_outer_l2 = n_inner * len(LAYERS) * len(seeds)
    per_outer_l3 = n_inner * n_sub * len(LAYERS) * len(seeds)
    #: A0's arity, which the full run must reach exactly: ensembling is worth
    #: roughly +0.07 to +0.09 median Spearman here, so a mismatch between A5's
    #: two blend components would land as the blend's main effect.
    full_arity = nested.ASSESS_INNER_FOLDS * len(LAYERS) * len(SEEDS)
    n_outer = 1 if args.quick else nested.OUTER_FOLDS
    n_networks = n_outer * (per_outer_l2 + per_outer_l3)

    print("A5/ESM: the stage 3 frozen ESM-2 arm under Phase A's training-only "
          "nested cluster folds")
    print(f"train={len(train):,} rows, {train.cluster_id.nunique():,} peptide clusters")
    print(f"checkpoint {config['checkpoint']}, "
          f"{rep_label(PEP_REP, HLA_REP, '{mid,final}')}, "
          f"peptide PCA {PEP_PCA}, HLA PCA {HLA_PCA}")
    print(f"outer: {n_outer} cluster folds (= pepstab.nested.outer_folds "
          f"= baseline_ensemble.cv_folds)")
    print(f"level 2 (outer_held, inner_oof_6): {n_inner} inner x {len(LAYERS)} "
          f"layers x {len(seeds)} seeds = {per_outer_l2} members/fold")
    print(f"level 3 (inner_oof_30): {n_inner} inner x {n_sub} sub x "
          f"{len(LAYERS)} layers x {len(seeds)} seeds = {per_outer_l3} members/fold")
    print(f"total networks: {n_networks}")
    print(f"member: hidden={tuple(config['hidden'])} l2={config['l2']:g} "
          f"lr={nested.REFERENCE_MLP.lr:g} batch={nested.REFERENCE_MLP.batch_size} "
          f"max_epochs={MAX_EPOCHS} patience={PATIENCE}")
    print(f"BLAS threads pinned to {os.environ['OMP_NUM_THREADS']}\n")

    # The outer assignment is *inherited*, not recomputed into agreement: A0
    # wrote the file every Phase A consumer joins on, and a disagreement here
    # would make the two blend components unjoinable.
    outer_fold = nested.outer_folds(train)
    folds_path = args.out_dir / "phase_a_outer_folds.csv"
    if not folds_path.exists():
        raise AssertionError(
            f"{folds_path} does not exist. A0 writes it before fitting; this "
            f"arm's outer folds must be asserted equal to it, not assumed.")
    a0_folds = pd.read_csv(folds_path)
    if not np.array_equal(a0_folds.pair_id.to_numpy(), train.pair_id.to_numpy()):
        raise AssertionError("phase_a_outer_folds.csv covers a different "
                             "pair_id set, or a different order, than the "
                             "training split")
    if not np.array_equal(a0_folds.outer_fold.to_numpy(), outer_fold):
        n_bad = int((a0_folds.outer_fold.to_numpy() != outer_fold).sum())
        raise AssertionError(f"outer fold assignment disagrees with "
                             f"phase_a_outer_folds.csv on {n_bad:,} rows")
    fold_table = a0_folds

    splits = list(nested.iter_outer(train, n_inner))
    if args.quick:
        splits = splits[:1]

    held_frames: list[pd.DataFrame] = []
    inner_frames: list[pd.DataFrame] = []
    runs: list[dict] = []
    fold_scores: list[dict] = []
    n_checks = 0
    n_features = 0
    payload_digest = ""
    modal_timing: dict | None = None

    started = time.perf_counter()
    if args.from_journal is not None:
        (held_frames, inner_frames, runs, fold_scores, n_checks, n_features,
         payload_digest, modal_timing) = collect_from_journal(
             args.from_journal, train, n_inner, n_sub, full_arity)
        print(f"assembled from {args.from_journal.name}: "
              f"{len(runs)} members over {len(fold_scores)} outer folds, "
              f"payload digest {payload_digest[:16]}, nothing refitted")
        splits = []
    for split in splits:
        t0 = time.perf_counter()
        held_mean, inner6, inner30, fold_runs, checks, n_features = fit_outer_fold(
            split, train, n_sub, seeds, config)
        runs.extend(fold_runs)
        n_checks += checks

        held_rows = train[split.held_mask]
        y_held = held_rows.y_log1p.to_numpy()
        scores = nested.score_fold(f"esm_outer{split.fold}", held_rows.allele,
                                   y_held, held_mean)
        coverage = nested.fold_coverage(held_rows.allele, y_held)
        # Prespecification §2's declared sensitivity panel, reported for every
        # arm so the looser >= 20-row bar cannot flatter a result unnoticed.
        sens = nested.score_fold(f"esm_outer{split.fold}_min50", held_rows.allele,
                                 y_held, held_mean, min_rows=SENSITIVITY_MIN_ROWS)
        sens_cov = nested.fold_coverage(held_rows.allele, y_held,
                                        min_rows=SENSITIVITY_MIN_ROWS)
        fold_scores.append({
            "outer_fold": split.fold, **coverage,
            "median_spearman": scores.median_spearman,
            "iqr_low": scores.iqr_spearman[0], "iqr_high": scores.iqr_spearman[1],
            "mae_log1p": scores.mae_log1p,
            "n_alleles_unranked": scores.n_alleles_unranked,
            "median_spearman_min50": sens.median_spearman,
            "n_alleles_eligible_min50": sens_cov["n_alleles_eligible"],
            "eligible_row_coverage_min50": sens_cov["eligible_row_coverage"]})
        print(f"  fold {split.fold}: {int(split.held_mask.sum()):,} held-out rows, "
              f"{len(fold_runs)} members, {n_features:,} features, "
              f"median SCC {scores.median_spearman:.4f} "
              f"[IQR {scores.iqr_spearman[0]:.3f}, {scores.iqr_spearman[1]:.3f}], "
              f"MAE {scores.mae_log1p:.3f}, "
              f"{coverage['n_alleles_eligible']} eligible alleles covering "
              f"{coverage['eligible_row_coverage']:.1%} of rows "
              f"(>=50 panel: {sens.median_spearman:.4f} over "
              f"{sens_cov['n_alleles_eligible']} alleles), "
              f"{(time.perf_counter() - t0) / 60:.1f} min")

        fit_pairs = train.loc[split.fit_mask, "pair_id"].to_numpy()
        held_frames.append(pd.DataFrame({
            "pair_id": held_rows.pair_id.to_numpy(), "outer_fold": split.fold,
            "role": "outer_held", "y_pred": held_mean}))
        for role, values in (("inner_oof_6", inner6), ("inner_oof_30", inner30)):
            inner_frames.append(pd.DataFrame({
                "pair_id": fit_pairs, "outer_fold": split.fold,
                "role": role, "y_pred": values}))
    wall = time.perf_counter() - started

    folds = pd.DataFrame(fold_scores)
    medians = folds.median_spearman.to_numpy()
    runs_frame = pd.DataFrame(runs)

    print("\nper-outer-fold scores (median per-allele Spearman on the "
          "cluster-held-out fold):")
    print(folds.round(4).to_string(index=False))
    print(f"\nfive-fold median of medians: {float(np.median(medians)):.4f}  "
          f"(min {medians.min():.4f}, max {medians.max():.4f}, "
          f"spread {medians.max() - medians.min():.4f})")
    print(f"mean of fold medians:        {float(medians.mean()):.4f}")
    print(f"median of medians, >=50 panel: "
          f"{float(np.median(folds.median_spearman_min50.to_numpy())):.4f}")
    print(f"eligible-row coverage:       "
          f"{folds.eligible_row_coverage.min():.1%}-"
          f"{folds.eligible_row_coverage.max():.1%} per fold, "
          f"{folds.n_alleles_eligible.min()}-{folds.n_alleles_eligible.max()} "
          f"alleles of 75")
    print("\nThese are cluster-held-out training folds. Stage 3 found frozen "
          "ESM-2 at parity\nwith the sequence baseline on validation and ruled "
          "out a +0.05 gain; a figure here\nthat beats A0 is a discrepancy to "
          "investigate, not a result.")

    by_level = runs_frame.assign(
        level=np.where(runs_frame.sub_fold == NO_SUB_FOLD, "level2", "level3"))
    print("\nper-member fit rows and stopping, by nesting level:")
    print(by_level.groupby("level").agg(
        members=("seed", "size"), n_fit_rows=("n_fit_rows", "median"),
        best_epoch=("best_epoch", "median"), dev_mse=("dev_mse", "median"),
        fit_seconds=("fit_seconds", "median")).round(4).to_string())
    print("\nper-member median SCC is not reported: the ESM arm's seed spread "
          "at an\nunder-regularised L2 was the tell at stage 3, and at the "
          "selected L2=1e-2 it\ncollapsed to 0.013. The ladder is not re-opened "
          "here (prespecification §3).")
    at_cap = int((runs_frame.best_epoch >= MAX_EPOCHS).sum())
    print(f"members stopping on the {MAX_EPOCHS}-epoch cap: {at_cap}")
    if modal_timing is None:
        print(f"\ntotal wall {wall / 60:.1f} min for {n_networks} networks "
              f"({wall / n_networks:.1f}s per network)")
    else:
        # The local wall here is assembly only and would read as ~0 minutes.
        # The compute happened in containers, so the container time is what is
        # reported; the two launches' client wall is in the run log.
        total = modal_timing["unit_seconds_total"]
        print(f"\ncontainer time {total / 60:.1f} min over "
              f"{modal_timing['n_units']} units "
              f"({total / n_networks:.1f}s per network), slowest unit "
              f"{modal_timing['unit_seconds_max']:.0f}s, median "
              f"{modal_timing['unit_seconds_median']:.0f}s. Local assembly "
              f"{wall:.1f}s, which refitted nothing.")

    oof = pd.concat(held_frames + inner_frames, ignore_index=True)
    oof = oof.sort_values(["pair_id", "role", "outer_fold"]).reset_index(drop=True)

    # Assert before writing. A broken contract file is worse than no file: A5
    # joins the two arms on these columns and would inherit the error silently,
    # and an arity or coverage error between blend components would land as the
    # blend's main effect.
    checks_passed = [
        f"outer fold assignment == phase_a_outer_folds.csv for all "
        f"{len(fold_table):,} rows (loaded and compared, not recomputed)"]

    if not np.isfinite(oof.y_pred.to_numpy()).all():
        bad = int((~np.isfinite(oof.y_pred.to_numpy())).sum())
        raise AssertionError(f"{bad:,} non-finite prediction(s)")
    checks_passed.append(f"np.isfinite on all {len(oof):,} predictions")

    expected_l2 = n_outer * per_outer_l2
    expected_l3 = n_outer * per_outer_l3
    counts = (int((runs_frame.sub_fold == NO_SUB_FOLD).sum()),
              int((runs_frame.sub_fold != NO_SUB_FOLD).sum()))
    if counts != (expected_l2, expected_l3):
        raise AssertionError(f"member counts {counts}, expected "
                             f"({expected_l2}, {expected_l3})")
    checks_passed.append(f"members: {expected_l2} level-2 + {expected_l3} "
                         f"level-3 = {len(runs_frame)}")
    checks_passed.append(f"{n_checks} fit/predict disjointness checks, covering "
                         f"every one of the {len(runs_frame)} members")

    per_role = oof.pivot_table(index="pair_id", columns="role", values="y_pred",
                               aggfunc="size")
    if not args.quick:
        expected_oof = N_TRAIN_ROWS * 9
        if len(oof) != expected_oof:
            raise AssertionError(f"OOF table has {len(oof):,} rows, "
                                 f"expected {expected_oof:,}")
        checks_passed.append(
            f"phase_a_esm_oof_cluster.csv = {len(oof):,} rows = 19,716 "
            f"outer_held + 4 x 19,716 inner_oof_6 + 4 x 19,716 inner_oof_30")

        for role, want in (("outer_held", 1), ("inner_oof_6", 4),
                           ("inner_oof_30", 4)):
            if not (per_role[role] == want).all():
                raise AssertionError(f"some pair_id does not have exactly {want} "
                                     f"{role} row(s)")
        checks_passed.append("every pair_id: exactly 1 outer_held, 4 "
                             "inner_oof_6, 4 inner_oof_30")

        if len(per_role) != N_TRAIN_ROWS or set(per_role.index) != set(
                train.pair_id.to_numpy().tolist()):
            raise AssertionError(f"OOF table covers {len(per_role):,} pair_ids, "
                                 f"expected exactly the {N_TRAIN_ROWS:,} "
                                 f"training rows")
        checks_passed.append(f"pair_id set == the training split exactly "
                             f"({len(per_role):,} pair_ids)")

        held_only = oof[oof.role == "outer_held"].set_index("pair_id")
        if not np.array_equal(
                held_only.outer_fold.reindex(fold_table.pair_id).to_numpy(),
                fold_table.outer_fold.to_numpy()):
            raise AssertionError("outer_held fold ids disagree with the fold table")
        checks_passed.append("outer_held fold id == phase_a_outer_folds.csv for "
                             "every row")

        straddle = oof[oof.role != "outer_held"].merge(
            fold_table, on="pair_id", suffixes=("", "_own"))
        if (straddle.outer_fold == straddle.outer_fold_own).any():
            raise AssertionError("an inner_oof row is attributed to the outer "
                                 "fold that held it out, which never fitted it")
        checks_passed.append("no inner_oof row is attributed to its own outer "
                             "held-out fold")

        # A0's fold assignment file was written before any model existed, so it
        # is the independent witness that this arm occupied the same inner cut:
        # A5 averages the two arms' inner predictions row by row.
        assignments_path = args.out_dir / "phase_a_fold_assignments.csv"
        if assignments_path.exists():
            occupied = pd.read_csv(assignments_path)
            occupied = occupied[occupied.inner_fold != OUTER_HELD_INNER_FOLD]
            inner_keys = set(map(tuple,
                                 straddle[["pair_id", "outer_fold"]].to_numpy()))
            if inner_keys != set(map(tuple,
                                     occupied[["pair_id", "outer_fold"]].to_numpy())):
                raise AssertionError("inner_oof (pair_id, outer_fold) keys "
                                     "disagree with phase_a_fold_assignments.csv")
            checks_passed.append("inner_oof keys == the fitting-partition rows "
                                 "of phase_a_fold_assignments.csv")
        else:
            print(f"\n  (note: {assignments_path.name} absent; the inner-key "
                  f"cross-check against A0 was skipped)")

    runs_frame = runs_frame[["outer_fold", "inner_fold", "sub_fold", "block",
                             "seed", "n_fit_rows", "best_epoch", "dev_mse",
                             "fit_seconds"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.out_dir / f"phase_a_esm_oof_cluster{suffix}.csv", index=False)
    runs_frame.to_csv(args.out_dir / f"phase_a_esm_runs{suffix}.csv", index=False)

    # The floor event is the exact 0 the assay records, not a 0.1 h threshold:
    # prespecification §7, 4 October 2026. Reported for parity with A0's
    # provenance, which Phase B reads alongside this one.
    n_floor = int((train[TARGET_RAW] == 0).sum())

    provenance = {
        "arm": "A5/ESM",
        "description": ("the stage 3 frozen ESM-2 arm refitted under the Phase A "
                        "training-only nested cluster protocol, as the ESM "
                        "component of A5's blends"),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": args.quick,
        "why_not_preds_esm_ensemble_oof": (
            "preds/esm_ensemble_oof.csv was built on pepstab.stage6.nested_folds, "
            "which partitions by peptide and deliberately splits near-neighbour "
            "mutant families across folds. Phase A assesses on whole-cluster "
            "folds; the two schemes are not interchangeable."),
        "selected_esm_config": config,
        "arity": {
            "members_per_outer_fold": per_outer_l2,
            "a0_members_per_outer_fold": full_arity,
            "axes": (f"{n_inner} inner folds x {len(LAYERS)} ESM layers "
                     f"{list(LAYERS)} x {len(seeds)} seeds"),
            # Stated unconditionally, so a --quick provenance (4 and 8 members)
            # cannot be read as this arm's declared arity.
            "axes_at_full_run": (f"{nested.ASSESS_INNER_FOLDS} inner folds x "
                                 f"{len(LAYERS)} ESM layers {list(LAYERS)} x "
                                 f"{len(SEEDS)} seeds = {full_arity}"),
            "networks_at_full_run": nested.OUTER_FOLDS * full_arity * (
                1 + nested.ASSESS_INNER_FOLDS),
            "is_full_arity": per_outer_l2 == full_arity,
            "a0_axes": "5 inner folds x 2 encodings (onehot, blosum) x 3 seeds",
            "note": ("the ESM layer axis is the two-way axis stage 3 declared as "
                     "the stand-in for stage 2's encoding axis "
                     "(scripts/esm_arm.py::mode_ensemble --axis layer), so member "
                     "count, fold protocol, seeds and head family match A0 and "
                     "only the features differ"),
        },
        "outer_folds": n_outer,
        "inner_folds": n_inner,
        "sub_folds": n_sub,
        "members_per_outer_fold": {
            "level2_outer_held_and_inner_oof_6": per_outer_l2,
            "level3_inner_oof_30": per_outer_l3},
        "networks_total": len(runs_frame),
        "seeds": list(seeds),
        "block_axis": list(LAYERS),
        "n_features": int(n_features),
        "member_config": {
            "hidden": list(config["hidden"]), "l2": config["l2"],
            "lr": nested.REFERENCE_MLP.lr,
            "batch_size": nested.REFERENCE_MLP.batch_size,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "source": ("reports/stage3_headline.json head, reached via "
                       "scripts.esm_arm.SWEEP_CONFIG on top of "
                       "pepstab.nested.REFERENCE_MLP; not re-tuned here"),
        },
        "fit_rows_by_level": {
            "outer_held": int(by_level.loc[by_level.level == "level2",
                                           "n_fit_rows"].median()),
            "inner_oof_30": int(by_level.loc[by_level.level == "level3",
                                             "n_fit_rows"].median()),
            "note": ("the irreducible remainder of nested CV: each level removes "
                     "another fifth, so inner_oof_30 members are fitted on fewer "
                     "rows than outer_held members. Same remainder A0 carries."),
        },
        "transforms_refitted_per_fitting_partition": True,
        "transform_note": ("standardisation and PCA are fitted on the unique "
                           "sequences of whichever fitting partition is being "
                           "handed to assemble(), at every nesting level, so no "
                           "PCA basis sees a row it will later predict"),
        "min_rows_per_allele_fold": nested.MIN_ROWS_PER_ALLELE_FOLD,
        "sensitivity_min_rows_per_allele_fold": SENSITIVITY_MIN_ROWS,
        "floor_event": {"definition": "thalf_hours == 0", "n_train_rows": n_floor,
                        "share": round(n_floor / len(train), 6)},
        # In journal mode the fitting happened in containers, so the thread
        # count that shaped the float32 results is theirs, not this process's.
        # Recording the local one would name a number no network ever ran at.
        "blas_threads": (
            {"where": "modal containers, from the journal",
             "OMP_NUM_THREADS": str(modal_timing["container_threads_observed"])}
            if modal_timing is not None else
            {"where": "local", **{var: os.environ[var]
                                  for var in BLAS_THREAD_VARS}}),
        "package_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "platform": platform.platform(),
        },
        "input_digests": {
            "data/splits.csv": sha256(SPLITS_CSV),
            "data/rasmussen_et_al_dataset.csv": sha256(RAW_CSV),
            "reports/further_research/phase_a_outer_folds.csv": sha256(folds_path),
            "reports/stage3_headline.json": sha256(STAGE3_HEADLINE),
        },
        "embedding_cache_digests": digests,
        "source": ("modal journal " + args.from_journal.name
                   if args.from_journal is not None else "local fit"),
        "payload_digest": payload_digest,
        "payload_properties": {
            "training_sequences_only": True,
            "n_sequences": {"peptide": 3942, "hla": 75},
            "note": ("the artifact holds 3,942 of the cache's 5,633 peptides "
                     "and 75 HLA domains pre-sliced to their 34 contact "
                     "positions, so a container CANNOT read a frozen "
                     "validation or test sequence -- this arm's one "
                     "unrecoverable mistake is impossible by construction, "
                     "not merely avoided by discipline"),
            "bit_identical_to_cache_path": True,
            "bit_identical_note": ("build_payload preserves the cache's row "
                                   "order so standardisation sums identically, "
                                   "and randomized_svd projects over the 4,320 "
                                   "columns, which are unchanged; verified with "
                                   "assert_array_equal on both layers and "
                                   "atol=0 on predictions and inner_oof, so the "
                                   "cloud run and the local pilot are "
                                   "comparable rather than merely similar"),
        },
        "rows": {"train": len(train), "oof_csv": len(oof),
                 "runs_csv": len(runs_frame)},
        "fold_scores": folds.to_dict(orient="records"),
        # The comparator, and the reason a single fold cannot carry a verdict.
        "a0_comparator": a0_comparator(args.out_dir),
        "inner_oof_residual_scale": inner_oof_residual_scale(oof, train),
        "five_fold_median_of_medians": float(np.median(medians)),
        "five_fold_median_of_medians_min50": float(
            np.median(folds.median_spearman_min50.to_numpy())),
        "wall_seconds_local_assembly": round(wall, 1),
        "modal_container_timing": modal_timing,
        # The full member map, so A5 can verify the arity it was handed rather
        # than assuming it.
        "members": runs_frame.to_dict(orient="records"),
        "asserted": checks_passed,
    }
    (args.out_dir / f"phase_a_esm_provenance{suffix}.json").write_text(
        json.dumps(provenance, indent=2) + "\n")

    a0 = provenance["a0_comparator"]
    if a0.get("available"):
        print(f"\nA0 (sequence reference, same protocol): median of medians "
              f"{a0['five_fold_median_of_medians']:.4f}, per fold "
              f"{[round(v, 4) for v in a0['per_fold_median_spearman']]}")
        print(f"  A0's own fold spread is {a0['fold_spread']:.4f} with the "
              f"model held FIXED -- essentially the whole 0.05 bar from fold\n"
              f"  assignment alone, so compare the per-fold vectors and never "
              f"a single fold.")
        print("  Stage 3's verdict on this arm was inconclusive at 0 while "
              "ruling out a +0.05 gain, so\n  its role in A5 is "
              "complementarity. An ESM figure above A0's here is a "
              "discrepancy to\n  investigate, not a win.")
    ratio = provenance["inner_oof_residual_scale"]
    print(f"\ninner-role residual scale vs outer_held "
          f"(un-averaged, measured rows only): outer_held RMSE "
          f"{ratio['outer_held']['rmse_measured_rows']:.4f}, "
          f"inner_oof_6 x{ratio['inner_oof_6']['rmse_ratio_measured_rows']:.3f}, "
          f"inner_oof_30 x"
          f"{ratio['inner_oof_30']['rmse_ratio_measured_rows']:.3f}")
    print("  Un-averaged on purpose: averaging a pair_id's four replicates "
          "first reversed this")
    print("  comparison's sign on A0. Which role is the better stacking input "
          "is OPEN; A5")
    print("  settles it on a paired blend outcome.")

    print("\nassertions passed before writing:")
    for check in checks_passed:
        print(f"  - {check}")
    rel = (args.out_dir.relative_to(REPO_ROOT)
           if args.out_dir.is_relative_to(REPO_ROOT) else args.out_dir)
    print(f"\nwrote {rel}/: phase_a_esm_oof_cluster{suffix}.csv, "
          f"phase_a_esm_runs{suffix}.csv, phase_a_esm_provenance{suffix}.json")
    if args.quick:
        print("--quick: *_partial filenames only; the canonical contract files "
              "were not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
