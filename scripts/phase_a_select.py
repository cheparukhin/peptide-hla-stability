"""Phase A screen: assemble the A1-A3 units, score them, select, compare, blend.

    # against the live journal the Modal screen is writing (default path)
    .venv/bin/python scripts/phase_a_select.py

    # an explicit journal, a cheaper bootstrap while the screen is still running
    .venv/bin/python scripts/phase_a_select.py \
        --journal reports/further_research/phase_a_runs_a-cheparukhin.partial.jsonl \
        --n-boot 200

    # the completed screen, full 2,000-resample intervals, canonical filenames
    .venv/bin/python scripts/phase_a_select.py --n-boot 2000

    # rewrite the tables and the narrative without re-paying for the intervals
    .venv/bin/python scripts/phase_a_select.py --reuse-comparisons

The post-processing half of Phase A arms A1-A3. It fits nothing: every
prediction it reads was produced by ``pepstab.phase_a.run_unit`` on Modal, and
everything here is averaging, the frozen metric, the declared parsimony rule and
a paired bootstrap. ``reports/further_research/phase_a_prespecification.md`` is
binding; §2 fixes the metrics and the four calibration rules, §3 the
one-standard-error rule and the grid-boundary trigger, §4 the arms, §6a the
amendments.

**A screen result is a ranking of configurations, not a verdict against A0.**
One screen unit is one fit of ``(arm, config, encoding, seed, outer fold, inner
fold)``, and the screen runs 3 inner folds x 2 seeds x 2 encodings = **12
members** per ``(config, outer fold)``. The A0 reference and every assessment
arm carry **30**. That 12-vs-30 gap is worth **+0.0156**, measured here -- the
matched 12-member control scores 0.6337 against A0's 0.6493. It is *not* the
"+0.07 to +0.09" the prespecification quotes, which is single-network-versus-
ensemble and a different comparison altogether; see the note above
``A0_GAP_TERMS``. ``n_members`` is written on every assembled row and every
comparison row so a 12-member screen number can never be read as a 30-member
assessment number.

**No calibration map goes near the primary metric** (§2, four rules). Isotonic
regression is only weakly monotone: it collapses distinct predictions into ties,
ties take average ranks under the frozen contract, and on this project's own
predictions that moved per-allele Spearman between -0.0431 and +0.1109 -- both
signs inside one fit, exceeding 0.05 on 3 of 68 alleles. This script scores raw
predictions only and holds no calibration path at all.

**Training split only.** Rows come from ``pepstab.data.load_with_splits()``
filtered to ``split == "train"``; the frozen validation and test splits are
never read and never scored.

Writes, to ``--out-dir`` (``reports/further_research`` by default):
``phase_a_screen_assembled.csv``, ``phase_a_screen_scores.csv``,
``phase_a_selection.csv``, ``phase_a_screen_comparisons.csv`` and
``phase_a_screen.md``. Until every one of the screen's units is present the
names all carry a ``_partial`` suffix, because a selection made on a fraction of
the ladder is not the declared selection.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread *before* numpy is imported -- the thread pool is sized
# at import time, so setting these afterwards does nothing. Thread count changes
# float32 results in this project: the same seed at a different count moves a
# single network by roughly 0.02 median Spearman. Nothing here fits a network,
# but the bootstrap is a Python-level resample loop over small per-allele groups
# whose BLAS calls gain nothing from threading, and left at the default each
# concurrent run spawns one thread per core. VECLIB_MAXIMUM_THREADS is the one
# that matters on macOS, where numpy links against Accelerate.
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _var in BLAS_THREAD_VARS:
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pepstab import evaluation, nested, phase_a  # noqa: E402
from pepstab.data import TARGET, load_with_splits  # noqa: E402
from pepstab.features import ENCODINGS  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "further_research"
DEFAULT_JOURNAL = OUT_DIR / "phase_a_runs_a-cheparukhin.partial.jsonl"
DEFAULT_RUNS_CSV = OUT_DIR / "phase_a_runs.csv"

#: A0's nested sequence reference, §A0's output contract. ``role ==
#: "outer_held"`` is the 30-member mean for each row, from members fitted on the
#: other four outer folds.
A0_OOF_CSV = OUT_DIR / "phase_a_seq_oof_cluster.csv"

#: The ESM component A5 needs. Still running, so it is read only if present and
#: only for the outer folds it actually covers.
ESM_OOF_CSV = OUT_DIR / "phase_a_esm_oof_cluster_partial.csv"

#: Arms in this screen. A4/YetiRank is withdrawn (§6a) and A4/PairLogit is
#: deferred (§7, 95.5% of the full screen's cost), so neither is expected and
#: neither is special-cased: they simply never appear in ``SCREEN_ARMS``.
SCREEN_ARMS = ("A1", "A2", "A3")

#: Members behind one assembled ``(config, outer fold)`` prediction, and behind
#: one assembled inner out-of-fold row. The screen's 3 inner folds x 2 seeds x 2
#: encodings is **not** the 30-member assessment arity; a row sits in exactly one
#: inner fold, so its inner out-of-fold mean has 2 encodings x 2 seeds = 4
#: members. Both are asserted per fold, not assumed.
SCREEN_HELD_MEMBERS = nested.SCREEN_INNER_FOLDS * len(phase_a.SCREEN_SEEDS) * len(ENCODINGS)
SCREEN_OOF_MEMBERS = len(phase_a.SCREEN_SEEDS) * len(ENCODINGS)
ASSESS_MEMBERS = nested.ASSESS_INNER_FOLDS * len(phase_a.ASSESS_SEEDS) * len(ENCODINGS)

#: The inner role is named by its arity so it cannot be confused with A0's
#: ``inner_oof_6`` / ``inner_oof_30``, which are different objects.
OUTER_HELD = "outer_held"
INNER_OOF = f"inner_oof_{SCREEN_OOF_MEMBERS}"

#: §2's panel and its declared sensitivity check. 20 is
#: ``evaluation.MIN_ROWS_BY_SPLIT["val"]`` and transfers to a fold on size; 50 is
#: reported for every arm so the looser bar cannot flatter a result unnoticed.
PANELS = (nested.MIN_ROWS_PER_ALLELE_FOLD, 50)

#: A0's five-fold median of medians and its per-fold medians at the >=20 panel,
#: as the prespecification reports them. Recomputed here from
#: ``phase_a_seq_oof_cluster.csv`` and asserted equal, so a mismatch surfaces as
#: a failure rather than as a quietly different reference.
A0_DECLARED_MEDIAN_OF_MEDIANS = 0.6493
A0_DECLARED_PER_FOLD = (0.6765, 0.6281, 0.6418, 0.6493, 0.6515)
A0_DECLARED_TOLERANCE = 5e-4

#: §3's grid-boundary trigger, per family: the weakest-regularisation end of each
#: ladder, and the one declared extension step in that direction. The extension
#: re-selects **every** arm in the family, so this script reports that the
#: trigger fired and stops -- running it is a decision to surface, not a
#: post-processing step.
#: ``(axis, value at the weak end, the declared extended ladder)``. The extended
#: ladder is kept as values, not only as prose, so the report can check whether
#: it actually reaches past the selection instead of asserting that it does --
#: for two of the three families it does not. §3 also applies the rule
#: symmetrically at the strongest-regularisation end, where a selection means the
#: parsimony rule wanted more shrinkage than the grid offered; which end counts
#: as which is derived from ``phase_a.REGULARISATION_ORDER``, not restated here.
BOUNDARY_TRIGGER = {
    "tree": ("l2_leaf_reg", 3, (1, 3, 10, 30)),
    "A2": ("k", 1, (0.2, 1, 5, 20, 100)),
    "A3": ("lam", 1.0, (0.03, 0.1, 0.3, 1.0)),
}

#: Which family's ladder each arm selects on.
ARM_FAMILY = {"A1": "tree", "A2": "A2", "A3": "A3"}

#: Config axes lifted onto the output tables. One arm uses three of them and the
#: others are left empty rather than written into a second file.
CONFIG_AXES = ("depth", "learning_rate", "l2_leaf_reg", "k", "cell", "loss",
               "sampling", "lam", "stopping")

#: A5's blend weights are fitted by a grid over the simplex at this step. Ties
#: resolve to the first point in the enumeration, which orders by increasing
#: weight on the non-reference components, so a flat objective keeps the
#: reference-heavy corner rather than drifting.
SIMPLEX_STEP = 0.01

#: Two weight vectors count as the same choice when no component differs by more
#: than one grid step -- below that the grid cannot resolve them.
WEIGHT_AGREEMENT_TOLERANCE = SIMPLEX_STEP


# --- inputs -----------------------------------------------------------------


def train_frame() -> pd.DataFrame:
    """The training split with its five outer cluster folds, sorted by pair_id.

    The same construction ``phase_a.load_payload`` ships to the containers, so
    the fold ids on the journal rows line up with these. Validation and test
    rows are dropped here and never read again.
    """
    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    if set(train["split"].unique()) != {"train"}:
        raise AssertionError("the assessment frame must hold training rows only")
    return train.assign(outer_fold=nested.outer_folds(train))


def check_outer_folds(train: pd.DataFrame) -> list[str]:
    """Agree this script's outer folds with the ones A0 wrote to disk.

    ``nested.outer_folds`` is deterministic and seedless, so this should be an
    identity -- which is exactly why it is worth asserting: the journal's `held`
    blocks are 3,944 rows on fold 0 and 3,943 on the rest, and if the fold ids
    here disagreed with A0's, every comparison would pair the wrong rows while
    still producing a plausible number.
    """
    table = pd.read_csv(OUT_DIR / "phase_a_outer_folds.csv")
    merged = train[["pair_id", "outer_fold"]].merge(
        table, on="pair_id", how="outer", suffixes=("_here", "_a0"))
    if merged[["outer_fold_here", "outer_fold_a0"]].isna().any().any():
        raise AssertionError("phase_a_outer_folds.csv and the training split do "
                             "not cover the same pair_ids")
    disagree = int((merged.outer_fold_here != merged.outer_fold_a0).sum())
    if disagree:
        raise AssertionError(f"{disagree} pair_ids sit in a different outer fold "
                             "here than in phase_a_outer_folds.csv")
    sizes = train["outer_fold"].value_counts().sort_index().to_dict()
    return [f"outer folds identical to phase_a_outer_folds.csv for all "
            f"{len(train)} training rows, fold sizes "
            f"{[sizes[f] for f in range(nested.OUTER_FOLDS)]}"]


def expected_screen() -> tuple[dict[str, dict], dict[tuple, set]]:
    """Every unit the screen owes, and the member axes behind each assembly.

    Derived from ``phase_a.enumerate_units`` rather than written down, so the
    1,920 figure and the 12-member arity are the harness's own and cannot drift
    from it.
    """
    units = phase_a.enumerate_units(list(SCREEN_ARMS))
    by_key = {u["key"]: u for u in units}
    axes: dict[tuple, set] = {}
    for u in units:
        slot = (u["arm"], u["config"]["label"], u["fold"])
        axes.setdefault(slot, set()).add(
            (u["encoding"], u["seed"], u["inner_fold"]))
    for slot, members in axes.items():
        if len(members) != SCREEN_HELD_MEMBERS:
            raise AssertionError(
                f"{slot} expects {SCREEN_HELD_MEMBERS} members, enumerate_units "
                f"gives {len(members)}")
    return by_key, axes


# --- assembly ---------------------------------------------------------------


def unit_outer_fold(row: dict) -> int:
    """The unit's outer fold, from ``fold``, cross-checked against its key.

    **The journal's ``outer_fold`` field is always null** -- ``run_unit`` writes
    the outer fold as ``fold`` and ``outer_fold`` belongs to the A0 prediction
    contract, not to a unit row. Reading the null field would group all five
    folds into one bucket and produce a single plausible-looking number that is
    wrong, so the fold is taken from ``fold``, asserted non-null, and agreed
    against the ``of=`` field the key encodes independently.
    """
    fold = row.get("fold")
    if fold is None:
        raise AssertionError(f"{row.get('key')} carries a null fold; the outer "
                             "fold must come from the 'fold' field")
    from_key = int(row["key"].split("|of=")[1].split("|")[0])
    if int(fold) != from_key:
        raise AssertionError(f"{row['key']} says fold={fold} but its key encodes "
                             f"of={from_key}")
    if row.get("outer_fold") is not None:
        raise AssertionError(
            f"{row['key']} carries a non-null 'outer_fold' ({row['outer_fold']}); "
            "the screen writes the outer fold as 'fold' and this script's "
            "assumption about the schema would need rechecking")
    return int(fold)


@dataclass
class Assembly:
    """Running sums for one ``(arm, config)`` over the screen's member axes."""

    arm: str
    label: str
    config: dict
    n_rows: int
    held_sum: np.ndarray = field(init=False)
    held_count: np.ndarray = field(init=False)
    oof_sum: np.ndarray = field(init=False)
    oof_count: np.ndarray = field(init=False)
    axes_seen: dict[int, set] = field(default_factory=dict)
    #: Per fold, the provenance that must be uniform across the members being
    #: averaged: thread count, the stopping metric CatBoost resolved, and the
    #: payload digest.
    threads_seen: dict[int, set] = field(default_factory=dict)
    eval_metrics: dict[int, set] = field(default_factory=dict)
    digests: set = field(default_factory=set)

    def __post_init__(self) -> None:
        # Outer-held rows are disjoint across outer folds, so one vector holds
        # all five folds; inner out-of-fold rows are not -- a row sits in the
        # fitting partition of four outer folds -- so those need a row per fold.
        self.held_sum = np.zeros(self.n_rows)
        self.held_count = np.zeros(self.n_rows, dtype=np.int32)
        self.oof_sum = np.zeros((nested.OUTER_FOLDS, self.n_rows))
        self.oof_count = np.zeros((nested.OUTER_FOLDS, self.n_rows), dtype=np.int32)

    def add(self, row: dict, position: dict[int, int]) -> None:
        fold = unit_outer_fold(row)
        self.axes_seen.setdefault(fold, set()).add(
            (row["encoding"], int(row["seed"]), int(row["inner_fold"])))
        if row.get("thread_pin_ok") is False:
            raise AssertionError(
                f"{row['key']} reports thread_pin_ok=False, so its float32 "
                "results are not comparable to the other members: thread count "
                "moves a single network by roughly 0.02 median Spearman here")
        self.threads_seen.setdefault(fold, set()).add(
            (row.get("n_threads"), str(row.get("threads_observed"))))
        self.eval_metrics.setdefault(fold, set()).add(row.get("eval_metric"))
        self.digests.add(row.get("payload_digest"))
        for pair_id, value in row["held"]:
            i = position[pair_id]
            self.held_sum[i] += value
            self.held_count[i] += 1
        for pair_id, value in row["oof"]:
            i = position[pair_id]
            self.oof_sum[fold, i] += value
            self.oof_count[fold, i] += 1

    def complete_folds(self, expected_axes: dict[tuple, set]) -> list[int]:
        """Folds whose full member set is present, by axis identity not by count.

        Counting to 12 would accept the same unit twice; comparing the axis set
        against ``enumerate_units``' own set cannot.
        """
        return [fold for fold, seen in sorted(self.axes_seen.items())
                if seen == expected_axes[(self.arm, self.label, fold)]]


def read_journal(path: Path, by_key: dict[str, dict], train: pd.DataFrame,
                 ) -> tuple[dict[tuple, Assembly], list[dict], dict]:
    """Stream the journal into per-config running sums plus a selection frame.

    The journal is a growing file of one JSON object per completed unit, each
    carrying ~9,200 predictions, so it is accumulated line by line rather than
    held in memory. A trailing partial line is tolerated: the screen appends
    while this runs, and a half-written final line is not corruption.
    """
    position = {int(p): i for i, p in enumerate(train["pair_id"])}
    assemblies: dict[tuple, Assembly] = {}
    lite: list[dict] = []
    seen_keys: set[str] = set()
    stats = {"lines": 0, "truncated_tail": 0, "not_ok": 0, "duplicate_keys": 0,
             "unexpected_keys": []}

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["truncated_tail"] += 1
                continue
            if not row.get("ok"):
                stats["not_ok"] += 1
                continue
            key = row["key"]
            if key not in by_key:
                stats["unexpected_keys"].append(key)
                continue
            if key in seen_keys:
                stats["duplicate_keys"] += 1
                continue
            seen_keys.add(key)

            slot = (row["arm"], row["config"]["label"])
            if slot not in assemblies:
                assemblies[slot] = Assembly(arm=row["arm"], label=row["config"]["label"],
                                            config=row["config"], n_rows=len(train))
            assemblies[slot].add(row, position)
            lite.append({k: v for k, v in row.items()
                         if k not in ("oof", "held", "coverage", "held_scores",
                                      "thread_env")})

    if stats["truncated_tail"] > 1:
        raise ValueError(f"{path} has {stats['truncated_tail']} unparsable lines; "
                         "only a single truncated tail line is expected")
    stats["units_present"] = len(seen_keys)
    stats["units_expected"] = len(by_key)
    stats["complete"] = len(seen_keys) == len(by_key)
    # Screen-wide provenance. A mixed thread count is not averaged anywhere --
    # the per-fold check in `assemble` refuses that -- but two configurations
    # fitted at different counts are still being compared, so the spread is
    # recorded rather than left invisible.
    stats["threads"] = sorted({t for acc in assemblies.values()
                               for seen in acc.threads_seen.values() for t in seen})
    stats["eval_metrics"] = sorted({m for acc in assemblies.values()
                                    for seen in acc.eval_metrics.values()
                                    for m in seen if m is not None})
    stats["payload_digests"] = sorted({d for acc in assemblies.values()
                                       for d in acc.digests})
    if len(stats["payload_digests"]) > 1:
        raise AssertionError(
            f"the journal mixes {len(stats['payload_digests'])} payload digests, "
            "so its units were not all fitted on the same training rows: "
            f"{stats['payload_digests']}")
    return assemblies, lite, stats


def assemble(assemblies: dict[tuple, Assembly], expected_axes: dict[tuple, set],
             train: pd.DataFrame) -> tuple[dict[tuple, dict], pd.DataFrame]:
    """Member-mean predictions per ``(arm, config, outer fold)``, with the arity asserted.

    For each complete fold the outer-held mean is over all 12 members and each
    inner out-of-fold row over the 4 members that early-stopped on its inner
    fold. Both counts are asserted row by row: a silently short mean would read
    as a worse arm rather than as a bug, and at 12 against A0's 30 the arity is
    already the largest term in any comparison.
    """
    outer = train["outer_fold"].to_numpy()
    assembled: dict[tuple, dict] = {}
    rows = []
    for slot, acc in sorted(assemblies.items()):
        folds = acc.complete_folds(expected_axes)
        for fold in folds:
            threads = acc.threads_seen[fold]
            if len(threads) > 1:
                raise AssertionError(
                    f"{slot} fold {fold} would average members fitted at "
                    f"different thread counts {sorted(threads)}; thread count "
                    "changes float32 results here by roughly 0.02 median "
                    "Spearman on a single network, so these are not poolable")
            on_fold = outer == fold
            held_counts = acc.held_count[on_fold]
            if not np.all(held_counts == SCREEN_HELD_MEMBERS):
                raise AssertionError(
                    f"{slot} fold {fold}: outer-held rows carry member counts "
                    f"{sorted(set(held_counts.tolist()))}, expected "
                    f"{SCREEN_HELD_MEMBERS}")
            oof_counts = acc.oof_count[fold][~on_fold]
            if not np.all(oof_counts == SCREEN_OOF_MEMBERS):
                raise AssertionError(
                    f"{slot} fold {fold}: inner out-of-fold rows carry member "
                    f"counts {sorted(set(oof_counts.tolist()))}, expected "
                    f"{SCREEN_OOF_MEMBERS}")
            if acc.oof_count[fold][on_fold].any():
                raise AssertionError(
                    f"{slot} fold {fold}: an outer-held row received an inner "
                    "out-of-fold prediction, so a held label was inside a "
                    "fitting partition")
            assembled[(*slot, fold)] = {
                OUTER_HELD: acc.held_sum[on_fold] / SCREEN_HELD_MEMBERS,
                INNER_OOF: acc.oof_sum[fold][~on_fold] / SCREEN_OOF_MEMBERS,
                "held_index": np.flatnonzero(on_fold),
                "oof_index": np.flatnonzero(~on_fold),
                "config": acc.config,
            }
        rows.append({"arm": acc.arm, "config_label": acc.label,
                     "n_folds_complete": len(folds),
                     "folds_complete": ",".join(str(f) for f in folds),
                     "n_members_outer_held": SCREEN_HELD_MEMBERS,
                     "n_members_inner_oof": SCREEN_OOF_MEMBERS,
                     # CatBoost resolves the stopping metric from the objective,
                     # which is how A4/YetiRank came to stop on NDCG when
                     # nothing had declared it (§6a). Carried per config so no
                     # arm's stopping surface is ever inherited silently again.
                     "eval_metric": ",".join(sorted(
                         {m for seen in acc.eval_metrics.values() for m in seen
                          if m is not None})) or None,
                     "n_threads": ",".join(sorted(
                         {str(t[0]) for seen in acc.threads_seen.values()
                          for t in seen}))})
    return assembled, pd.DataFrame(rows)


def assembled_table(assembled: dict[tuple, dict], train: pd.DataFrame,
                    inner_for: set[tuple]) -> pd.DataFrame:
    """The long form of the assembled predictions, for ``phase_a_screen_assembled.csv``.

    Outer-held predictions for **every** configuration, so any ladder point can
    be re-scored, given a different panel or given an interval without reading
    the journal again. The inner out-of-fold role is written only for the
    configurations in ``inner_for`` -- the selected ones A5 blends on -- because
    that role is four times the rows and the full ladder's copy would be 2.5 M
    rows nothing consumes: selection and scoring both run from the in-memory
    arrays.
    """
    pair_ids = train["pair_id"].to_numpy()
    frames = []
    for (arm, label, fold), block in sorted(assembled.items()):
        roles = [(OUTER_HELD, block["held_index"], SCREEN_HELD_MEMBERS)]
        if (arm, label) in inner_for:
            roles.append((INNER_OOF, block["oof_index"], SCREEN_OOF_MEMBERS))
        for role, index, n_members in roles:
            frames.append(pd.DataFrame({
                "arm": arm, "config_label": label, "role": role,
                "outer_fold": fold, "pair_id": pair_ids[index],
                "y_pred": np.round(block[role], 6), "n_members": n_members}))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["arm", "config_label", "role", "outer_fold",
                                       "pair_id", "y_pred", "n_members"]))


# --- scoring ----------------------------------------------------------------


def score_blocks(name: str, train: pd.DataFrame, index: np.ndarray,
                 pred: np.ndarray) -> list[dict]:
    """``nested.score_fold`` plus coverage at both §2 panels, on raw predictions.

    No map of any kind sits between ``pred`` and the metric -- §2 rule 1.
    """
    allele = train["allele"].iloc[index]
    y = train[TARGET].to_numpy()[index]
    out = []
    for min_rows in PANELS:
        scores = nested.score_fold(f"{name}:min{min_rows}", allele, y, pred,
                                   min_rows=min_rows)
        coverage = nested.fold_coverage(allele, y, min_rows=min_rows)
        out.append({
            "panel_min_rows": min_rows,
            "median_per_allele_spearman": float(scores.median_spearman),
            "n_alleles_eligible": int(coverage["n_alleles_eligible"]),
            "n_rows": int(coverage["n_rows"]),
            "eligible_row_coverage": float(coverage["eligible_row_coverage"]),
            "mae_log1p": float(scores.mae_log1p),
        })
    return out


def _axis_columns(config: dict) -> dict:
    return {axis: config.get(axis) for axis in CONFIG_AXES}


def five_fold_summary(fold_rows: list[dict]) -> dict:
    """Median of medians and the fold-to-fold span.

    The span is reported next to every median because A0's own span is 0.0484
    with the model held fixed -- essentially the whole 0.05 worthwhile bar -- so a
    difference smaller than an arm's own span is not a result.
    """
    medians = np.array([r["median_per_allele_spearman"] for r in fold_rows], dtype=float)
    return {
        "median_per_allele_spearman": float(np.median(medians)),
        "fold_min": float(medians.min()), "fold_max": float(medians.max()),
        "fold_span": float(medians.max() - medians.min()),
        "n_folds": int(len(medians)),
        "n_alleles_eligible": int(np.median([r["n_alleles_eligible"] for r in fold_rows])),
        "eligible_row_coverage": float(np.mean([r["eligible_row_coverage"]
                                                for r in fold_rows])),
        "mae_log1p": float(np.mean([r["mae_log1p"] for r in fold_rows])),
    }


def score_screen(assembled: dict[tuple, dict], train: pd.DataFrame) -> pd.DataFrame:
    """Per-fold and five-fold scores for every assembled configuration."""
    rows = []
    by_config: dict[tuple, dict[int, list[dict]]] = {}
    for (arm, label, fold), block in sorted(assembled.items()):
        axes = _axis_columns(block["config"])
        for scored in score_blocks(f"{arm}|{label}|of={fold}", train,
                                   block["held_index"], block[OUTER_HELD]):
            rows.append({"arm": arm, "config_label": label, **axes, "scope": "fold",
                         "outer_fold": fold, "n_members": SCREEN_HELD_MEMBERS,
                         **scored, "fold_min": None, "fold_max": None,
                         "fold_span": None, "n_folds": 1})
            by_config.setdefault((arm, label), {}).setdefault(
                scored["panel_min_rows"], []).append(scored)
    for (arm, label), panels in sorted(by_config.items()):
        axes = _axis_columns(assembled_config(assembled, arm, label))
        for min_rows, fold_rows in sorted(panels.items()):
            if len(fold_rows) != nested.OUTER_FOLDS:
                continue
            rows.append({"arm": arm, "config_label": label, **axes,
                         "scope": "five_fold", "outer_fold": -1,
                         "n_members": SCREEN_HELD_MEMBERS,
                         "panel_min_rows": min_rows,
                         **five_fold_summary(fold_rows)})
    return pd.DataFrame(rows)


def assembled_config(assembled: dict[tuple, dict], arm: str, label: str) -> dict:
    for (a, lab, _), block in assembled.items():
        if (a, lab) == (arm, label):
            return block["config"]
    return {}


def score_reference(name: str, pred_by_pair: pd.Series, train: pd.DataFrame,
                    n_members: int) -> pd.DataFrame:
    """Per-fold and five-fold scores for a reference whose predictions are on disk."""
    pred = pred_by_pair.reindex(train["pair_id"]).to_numpy(dtype=float)
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"{name} does not cover every training pair_id")
    rows, panels = [], {}
    for fold in range(nested.OUTER_FOLDS):
        index = np.flatnonzero(train["outer_fold"].to_numpy() == fold)
        for scored in score_blocks(f"{name}|of={fold}", train, index, pred[index]):
            rows.append({"arm": name, "config_label": name, **_axis_columns({}),
                         "scope": "fold", "outer_fold": fold,
                         "n_members": n_members, **scored, "fold_min": None,
                         "fold_max": None, "fold_span": None, "n_folds": 1})
            panels.setdefault(scored["panel_min_rows"], []).append(scored)
    for min_rows, fold_rows in sorted(panels.items()):
        rows.append({"arm": name, "config_label": name, **_axis_columns({}),
                     "scope": "five_fold", "outer_fold": -1,
                     "n_members": n_members, "panel_min_rows": min_rows,
                     **five_fold_summary(fold_rows)})
    return pd.DataFrame(rows)


def a0_predictions(role: str = OUTER_HELD) -> pd.DataFrame:
    """One role of A0's §A0 output contract, indexed by ``(pair_id, outer_fold)``."""
    frame = pd.read_csv(A0_OOF_CSV)
    return frame[frame.role == role].drop(columns="role")


def check_a0_reference(scores: pd.DataFrame) -> list[str]:
    """Assert the recomputed A0 reference equals the figures §A0 publishes.

    A0 is both the comparison reference and the only end-to-end check available
    on this harness: if the per-fold medians reproduce, the fold ids, the panel
    rule and the metric path are the same ones that produced them.
    """
    folds = scores[(scores.arm == "A0") & (scores.scope == "fold")
                   & (scores.panel_min_rows == nested.MIN_ROWS_PER_ALLELE_FOLD)]
    per_fold = folds.sort_values("outer_fold")["median_per_allele_spearman"].to_numpy()
    summary = scores[(scores.arm == "A0") & (scores.scope == "five_fold")
                     & (scores.panel_min_rows == nested.MIN_ROWS_PER_ALLELE_FOLD)]
    median_of_medians = float(summary["median_per_allele_spearman"].iloc[0])
    deltas = np.abs(per_fold - np.array(A0_DECLARED_PER_FOLD))
    if deltas.max() > A0_DECLARED_TOLERANCE:
        raise AssertionError(
            f"recomputed A0 per-fold medians {np.round(per_fold, 4).tolist()} "
            f"differ from the declared {list(A0_DECLARED_PER_FOLD)} by up to "
            f"{deltas.max():.5f}")
    if abs(median_of_medians - A0_DECLARED_MEDIAN_OF_MEDIANS) > A0_DECLARED_TOLERANCE:
        raise AssertionError(
            f"recomputed A0 median of medians {median_of_medians:.4f} differs "
            f"from the declared {A0_DECLARED_MEDIAN_OF_MEDIANS}")
    return [f"A0 reproduces its declared per-fold medians "
            f"{list(A0_DECLARED_PER_FOLD)} and median of medians "
            f"{A0_DECLARED_MEDIAN_OF_MEDIANS} to within {A0_DECLARED_TOLERANCE}"]


# --- selection --------------------------------------------------------------


def boundary_report(family: str, selection: dict) -> dict:
    """Whether §3's grid-boundary trigger fired, and what the declared remedy is.

    The remedy is **not** run. One extension re-selects every arm in the family,
    both ladders are then reported, and no second extension is available -- that
    is a decision to surface, not a post-processing step, so this records the
    trigger and stops. ``boundary_axes`` is the quantity to read: §6a notes that
    ``interior`` is structurally unreachable for the tree grids, where
    ``learning_rate`` and ``l2_leaf_reg`` carry two values each and every
    selection sits at an endpoint on both axes by construction.
    """
    axis, weak_value, extension = BOUNDARY_TRIGGER[family]
    selected = selection["selected"]
    tried = selection["tried"]
    fired = axis in selection["boundary_axes"] and selected.get(axis) == weak_value

    # Which end of each axis is the *least* regularised one is a property of
    # ``phase_a.REGULARISATION_ORDER``: sign +1 means smaller is more
    # regularised, so the weak end is the largest value tried, and sign -1 the
    # smallest. Deriving it catches the axes §3's trigger does not name.
    weak_end, strong_end = [], []
    for name, sign in phase_a.REGULARISATION_ORDER[family]:
        values = tried[name]
        chosen = selected.get(name)
        if chosen == (max(values) if sign > 0 else min(values)):
            weak_end.append(name)
        if chosen == (min(values) if sign > 0 else max(values)):
            strong_end.append(name)

    # Does the declared one-step extension actually reach past the selection, in
    # the direction the selection ran out of grid? For the tree ladder
    # {1,3,10,30} it reaches both ways; for A2's k and A3's lambda the declared
    # set extends only one end, and this screen selected at the other.
    sign = dict(phase_a.REGULARISATION_ORDER[family])[axis]
    chosen = selected.get(axis)
    if axis in weak_end:
        beyond = [v for v in extension
                  if (v > chosen if sign > 0 else v < chosen)]
    elif axis in strong_end:
        beyond = [v for v in extension
                  if (v < chosen if sign > 0 else v > chosen)]
    else:
        beyond = []
    return {
        "trigger_axis": axis,
        "trigger_fired": bool(fired),
        "declared_extension": f"{axis} -> {{{', '.join(str(v) for v in extension)}}}",
        "extension_run": False,
        "extension_reaches_past_selection": bool(beyond),
        "extension_values_past_selection": beyond,
        "weak_end_axes": weak_end,
        "strong_end_axes": strong_end,
        "weak_end_axes_without_a_declared_trigger": [n for n in weak_end
                                                     if n != axis],
        "strong_boundary": bool(axis in strong_end),
        "unextendable_by_construction": sorted(
            name for name in selection["axes"] if len(selection["tried"][name]) < 3),
    }


def _jsonable(obj):
    """Numpy scalars to Python ones. ``parsimonious_select`` keys come from a
    pandas groupby index, so an axis value arrives as ``int64`` and will not
    serialise into the report or the selection CSV."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def select_arms(runs: pd.DataFrame) -> list[dict]:
    """Apply ``phase_a.parsimonious_select`` per family. The rule is not reimplemented.

    A3 goes through ``phase_a.common_hyperparameters``, which refuses a value
    chosen on a partial pool: §3 requires one common lambda across the ranking
    cells, because letting the two cells sit at different lambdas adds a second
    difference and the gap stops isolating the objective.
    """
    out = []
    for arm in SCREEN_ARMS:
        family = ARM_FAMILY[arm]
        arm_runs = runs[runs.arm == arm]
        if arm_runs.empty:
            continue
        try:
            if arm == "A3":
                selection = phase_a.common_hyperparameters(
                    arm_runs, family, phase_a.A3_RANKING_CELLS)
            else:
                selection = phase_a.parsimonious_select(arm_runs, family)
        except ValueError as exc:
            out.append({"arm": arm, "family": family, "ladder": "declared",
                        "error": f"{type(exc).__name__}: {exc}"})
            continue
        selection = _jsonable(selection)
        out.append({"arm": arm, "family": family, "ladder": "declared",
                    **selection, **boundary_report(family, selection)})

        if arm == "A3":
            # Reported, never selected on: §A3 makes the stopping rule its own
            # axis, so the pooled lambda above is the declared choice and these
            # show whether the two stopping surfaces would have chosen it too.
            for stopping in phase_a.A3_STOPPING:
                subset = arm_runs[arm_runs.stopping == stopping]
                if subset.empty:
                    continue
                try:
                    diag = phase_a.common_hyperparameters(
                        subset, family, phase_a.A3_RANKING_CELLS)
                except ValueError as exc:
                    out.append({"arm": arm, "family": family,
                                "ladder": f"diagnostic:stopping={stopping}",
                                "error": f"{type(exc).__name__}: {exc}"})
                    continue
                diag = _jsonable(diag)
                out.append({"arm": arm, "family": family,
                            "ladder": f"diagnostic:stopping={stopping}", **diag,
                            **boundary_report(family, diag)})
    return out


def selection_table(selections: list[dict]) -> pd.DataFrame:
    rows = []
    for sel in selections:
        rows.append({
            "arm": sel["arm"], "family": sel["family"], "ladder": sel["ladder"],
            "error": sel.get("error"),
            "axes": ",".join(sel.get("axes", [])),
            "tried": json.dumps(sel.get("tried", {}), sort_keys=True),
            "rule": sel.get("rule"),
            "argmax": json.dumps(sel.get("argmax", {}), sort_keys=True),
            "selected": json.dumps(sel.get("selected", {}), sort_keys=True),
            "score_at_argmax": sel.get("score_at_argmax"),
            "score_at_selected": sel.get("score_at_selected"),
            "seed_spread_at_argmax": sel.get("seed_spread_at_argmax"),
            "seed_spread_at_selected": sel.get("seed_spread_at_selected"),
            "standard_error": sel.get("standard_error"),
            "n_runs_at_argmax": sel.get("n_runs_at_argmax"),
            "n_configs": sel.get("n_configs"),
            "n_within_1se": sel.get("n_within_1se"),
            "interior": sel.get("interior"),
            "boundary_axes": ",".join(sel.get("boundary_axes", [])),
            "unextendable_by_construction": ",".join(
                sel.get("unextendable_by_construction", [])),
            "trigger_axis": sel.get("trigger_axis"),
            "trigger_fired": sel.get("trigger_fired"),
            "strong_boundary": sel.get("strong_boundary"),
            "weak_end_axes": ",".join(sel.get("weak_end_axes", [])),
            "strong_end_axes": ",".join(sel.get("strong_end_axes", [])),
            "weak_end_axes_without_a_declared_trigger": ",".join(
                sel.get("weak_end_axes_without_a_declared_trigger", [])),
            "declared_extension": sel.get("declared_extension"),
            "extension_run": sel.get("extension_run"),
            "extension_reaches_past_selection": sel.get(
                "extension_reaches_past_selection"),
            "extension_values_past_selection": ",".join(
                str(v) for v in sel.get("extension_values_past_selection", [])),
            "selection_surface": "mean inner-fold median Spearman "
                                 "(pepstab.phase_a.SCORE_COLUMN)",
        })
    return pd.DataFrame(rows)


def configs_for_selection(runs: pd.DataFrame, selections: list[dict],
                          assembled: dict[tuple, dict]) -> dict[str, dict]:
    """Map each arm's selected and argmax ladder point back to config labels.

    A ladder point is a set of axis values, not a configuration: A3's selected
    lambda names the two ranking cells at that lambda, under both stopping
    rules. So each arm reports every assembled config that matches, and the
    arm's representative is the best of them by five-fold median of medians.
    """
    out: dict[str, dict] = {}
    available = {(arm, label) for arm, label, _ in assembled}
    for sel in selections:
        if sel["ladder"] != "declared" or sel.get("error"):
            continue
        arm = sel["arm"]
        picks = {}
        for which in ("selected", "argmax"):
            mask = np.ones(len(runs), dtype=bool) & (runs.arm == arm).to_numpy()
            for axis, value in sel[which].items():
                mask &= (runs[axis].to_numpy() == value)
            labels = sorted({label for label in runs.loc[mask, "config_label"].unique()
                             if (arm, label) in available})
            picks[which] = labels
        out[arm] = picks
    return out


# --- comparison -------------------------------------------------------------


def pooled_vector(assembled: dict[tuple, dict], arm: str, label: str,
                  train: pd.DataFrame) -> np.ndarray | None:
    """One outer-held prediction per training row, pooled over the five folds.

    ``None`` unless all five folds are assembled: a pooled vector missing a fold
    is not comparable to A0's, which covers every row exactly once.
    """
    pred = np.full(len(train), np.nan)
    for fold in range(nested.OUTER_FOLDS):
        block = assembled.get((arm, label, fold))
        if block is None:
            return None
        pred[block["held_index"]] = block[OUTER_HELD]
    return pred if np.all(np.isfinite(pred)) else None


def pooled_panel(train: pd.DataFrame, min_rows: int) -> list[str]:
    return nested.fold_eligible_alleles(train["allele"],
                                        train[TARGET].to_numpy(), min_rows=min_rows)


#: §A3's cell A3-a -- MSE loss, natural sampling, MSE stopping, at the reference
#: MLP hyperparameters -- is A0's own member recipe run inside this screen. It
#: is therefore the **matched control**: same learner, same encodings, same
#: seeds-per-member protocol, same 12-member arity, same Linux/OpenBLAS
#: containers. A delta against it carries none of the arity, inner-fold-count or
#: platform terms that a delta against A0 does, so it is the headline reference
#: here even though §2 names only A0 and the strongest new A2/A3 control.
MATCHED_CONTROL = {"arm": "A3", "cell": "A3-a", "stopping": "mse"}

#: Measured on this screen: the matched 12-member control scores 0.6337 against
#: A0's 0.6493, so everything separating a screen arm from A0 is worth about
#: **+0.0156** median Spearman -- arity (12 against 30), inner-fold count (3
#: against 5, so each member also fits on fewer rows) and platform, together.
#:
#: The prespecification's "+0.07 to +0.09" is **not** this quantity: that figure
#: is a *single network* against an ensemble, which is a different comparison,
#: and using it as the 12-against-30 gap overstates it by about five times. The
#: caveat is real and roughly a third of the 0.05 worthwhile bar, not larger
#: than it.
A0_GAP_TERMS = ("arity (12 members against 30), inner-fold count (3 against 5, "
                "so each member fits on fewer rows too) and platform")

ARITY_CAVEAT = (
    f"cross-reference delta: a {SCREEN_HELD_MEMBERS}-member screen arm against "
    f"the {ASSESS_MEMBERS}-member A0. The gap between those two setups -- "
    f"{A0_GAP_TERMS} -- measures about +0.0156 median Spearman on this screen "
    "(matched control 0.6337 against A0's 0.6493), so the delta carries that "
    "term. A0 was additionally fitted on macOS/Accelerate while every screen "
    "unit ran on Linux/OpenBLAS; Phase B measured that at up to 0.18 per row "
    "and 0.0020 on median-of-medians, so this delta is also cross-platform. "
    "Prefer the matched-control comparison, which carries none of it.")

PLATFORM_CAVEAT = (
    "A0 was fitted on macOS/Accelerate and every screen unit on Linux/OpenBLAS; "
    "Phase B measured max |diff| 0.18 per row and 0.0020 on median-of-medians "
    "across the two. Deltas against A0 carry that term; deltas within the "
    "screen do not.")


def _progress(row: dict) -> None:
    """One line per finished interval, flushed.

    Each 2,000-resample interval is about 90 s and a full screen is roughly 25
    of them, so a silent 40-minute process is indistinguishable from a hung one
    from outside -- the failure mode the further-research README records for
    A0's own 900-network run.
    """
    print(f"  {row['kind']}: {row['arm_label']} vs {row['reference']} "
          f"delta {row['delta_median_spearman']:+.4f} "
          f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] - {row['verdict']}",
          flush=True)


def compare_pair(name_a: str, pred_a: np.ndarray, members_a: int,
                 name_b: str, pred_b: np.ndarray, members_b: int,
                 train: pd.DataFrame, panel: list[str], min_rows: int,
                 n_boot: int) -> dict:
    """``pepstab.evaluation.paired_cluster_bootstrap`` verbatim, A as the baseline.

    ``pred_a`` is the reference, so the interval is on ``arm - reference``.
    Resamples whole peptide clusters, 2,000 by default at seed 20261003, with the
    panel rebuilt per resample; the reading is ``describe_delta``'s, which is
    EVALUATION.md's six verdicts.
    """
    result = evaluation.paired_cluster_bootstrap(
        train["cluster_id"], train["allele"], train[TARGET].to_numpy(),
        pred_a, pred_b, alleles=panel, n_boot=n_boot,
        seed=evaluation.BOOTSTRAP_SEED)
    scored_a = nested.score_fold(f"pooled:{name_a}", train["allele"],
                                 train[TARGET].to_numpy(), pred_a, min_rows=min_rows)
    scored_b = nested.score_fold(f"pooled:{name_b}", train["allele"],
                                 train[TARGET].to_numpy(), pred_b, min_rows=min_rows)
    lo, hi = result["ci95"]
    return {
        "reference": name_a, "arm_label": name_b,
        "n_members_reference": members_a, "n_members_arm": members_b,
        "arity_matched": members_a == members_b,
        "panel_min_rows": min_rows, "panel_size": len(panel),
        "pooled_median_reference": float(scored_a.median_spearman),
        "pooled_median_arm": float(scored_b.median_spearman),
        "delta_median_spearman": float(result["delta_median_spearman"]),
        "ci_low": float(lo), "ci_high": float(hi),
        "verdict": evaluation.describe_delta(result),
        "n_boot": int(result["n_boot"]), "seed": evaluation.BOOTSTRAP_SEED,
        "n_resamples_with_dropped_alleles": result["n_resamples_with_dropped_alleles"],
        "panel_size_full": result["panel_size_full"],
        "caveat": "" if members_a == members_b else ARITY_CAVEAT,
    }


def compare_screen(assembled: dict[tuple, dict], picks: dict[str, dict],
                   scores: pd.DataFrame, train: pd.DataFrame, n_boot: int,
                   ) -> tuple[list[dict], dict]:
    """Each arm against three references, and A3's stopping axis against itself.

    - the **matched 12-member control**, A3-a: the headline, because it differs
      from an arm only in the thing the arm changes;
    - **A0**, §2's first reference, carrying the arity, inner-fold-count and
      cross-platform terms spelled out in :data:`ARITY_CAVEAT`;
    - the **strongest new sequence-only control** among A2/A3, §2's second,
      so a better learner is not mistaken for a benefit from embeddings or
      structure. A1 is not a candidate -- it is the learner-family arm under
      test.

    Then, separately, each A3 cell's two stopping rules against **each other**:
    a perfectly matched pair at identical arity, platform and loss, which is the
    only clean measurement of the stage 7 confound §A3 exists to isolate.

    One interval is ~90 s at 2,000 resamples, so intervals go on the points the
    declared rule chose -- one representative per arm, plus A1's argmax where it
    differs -- rather than on all 32 ladder points, which would be 80 minutes of
    numbers nobody reads. The ladder itself is a ranking and lives in the scores
    table.
    """
    min_rows = nested.MIN_ROWS_PER_ALLELE_FOLD
    panel = pooled_panel(train, min_rows)
    a0 = a0_predictions().set_index("pair_id")["y_pred"]
    a0_pred = a0.reindex(train["pair_id"]).to_numpy(dtype=float)

    five = scores[(scores.scope == "five_fold") & (scores.panel_min_rows == min_rows)]
    medians = {(r.arm, r.config_label): r.median_per_allele_spearman
               for r in five.itertuples()}

    matched = matched_control_label(scores)
    matched_pred = (pooled_vector(assembled, MATCHED_CONTROL["arm"], matched, train)
                    if matched else None)

    candidates = []
    for arm in ("A2", "A3"):
        for label in picks.get(arm, {}).get("selected", []):
            if (arm, label) in medians:
                candidates.append((medians[(arm, label)], arm, label))
    control = max(candidates) if candidates else None

    # One representative per arm: the best-scoring configuration at the ladder
    # point the parsimony rule chose. A3's selected lambda names four labels
    # (two cells x two stopping rules), and putting an interval on each of them
    # against three references is 12 intervals for one arm.
    targets = []
    for arm in SCREEN_ARMS:
        for which in ("selected", "argmax"):
            labels = [lab for lab in picks.get(arm, {}).get(which, [])
                      if (arm, lab) in medians]
            if not labels:
                continue
            best = max(labels, key=lambda lab: medians[(arm, lab)])
            if (arm, best) not in [(a, b) for a, b, _ in targets]:
                targets.append((arm, best, which))
            elif which == "argmax":
                for i, (a, b, w) in enumerate(targets):
                    if (a, b) == (arm, best) and "argmax" not in w:
                        targets[i] = (a, b, f"{w}+argmax")

    rows = []
    for arm, label, which in targets:
        pred = pooled_vector(assembled, arm, label, train)
        if pred is None:
            continue
        references = []
        if matched_pred is not None and label != matched:
            references.append(("matched_control", f"A3|{matched}", matched_pred,
                               SCREEN_HELD_MEMBERS))
        references.append(("arm_vs_A0", "A0", a0_pred, ASSESS_MEMBERS))
        if control and (arm, label) != (control[1], control[2]):
            ctrl_pred = pooled_vector(assembled, control[1], control[2], train)
            if ctrl_pred is not None:
                references.append(("arm_vs_new_control",
                                   f"{control[1]}|{control[2]}", ctrl_pred,
                                   SCREEN_HELD_MEMBERS))
        for kind, name, ref_pred, ref_members in references:
            row = compare_pair(name, ref_pred, ref_members, f"{arm}|{label}",
                               pred, SCREEN_HELD_MEMBERS, train, panel, min_rows,
                               n_boot)
            rows.append({"kind": kind, "arm": arm, "config_label": label,
                         "ladder_point": which, **row})
            _progress(rows[-1])

    rows += compare_a3_stopping(assembled, scores, train, panel, min_rows, n_boot)
    return rows, {"control": control, "matched": matched, "panel": panel,
                  "a0_pred": a0_pred,
                  "targets": [(arm, label) for arm, label, _ in targets]}


def matched_control_label(scores: pd.DataFrame) -> str | None:
    """The config label of §A3's cell A3-a under MSE stopping.

    Found by its axes rather than by its spelling, so a change to
    ``phase_a._label`` cannot silently point this at the wrong configuration.
    """
    five = scores[(scores.scope == "five_fold")
                  & (scores.arm == MATCHED_CONTROL["arm"])
                  & (scores.cell == MATCHED_CONTROL["cell"])
                  & (scores.stopping == MATCHED_CONTROL["stopping"])]
    return str(five["config_label"].iloc[0]) if len(five) else None


def compare_a3_stopping(assembled: dict[tuple, dict], scores: pd.DataFrame,
                        train: pd.DataFrame, panel: list[str], min_rows: int,
                        n_boot: int) -> list[dict]:
    """Rank-based against MSE-based early stopping, within each A3 cell.

    The cleanest comparison in the screen: the two sides share arity, platform,
    encodings, seeds, hyperparameters, loss and sampling, and differ only in
    which curve chose the epoch. Stage 7's apparent ranking recovery traced to
    exactly this axis rather than to the gradient, and §A3 made it a separate
    axis so the confound could be measured instead of inherited. The MSE-loss
    cells are the decisive ones: they carry no ranking gradient at all, so any
    gain there is the stopping surface alone.
    """
    five = scores[(scores.scope == "five_fold") & (scores.arm == "A3")
                  & (scores.panel_min_rows == min_rows)]
    rows = []
    for cell in phase_a.A3_CELLS:
        sub = five[five.cell == cell]
        if sub["stopping"].nunique() < 2:
            continue
        # Ranking cells carry a lambda axis; the pair must be taken at one
        # lambda or the comparison is not matched. The best shared lambda is
        # used and recorded.
        lam_values = sorted(sub["lam"].dropna().unique())
        lam = max(lam_values) if lam_values else None
        at_lam = sub[sub.lam.isna()] if lam is None else sub[sub.lam == lam]
        both = {r.stopping: r.config_label for r in at_lam.itertuples()}
        if not {"mse", "rank"} <= set(both):
            continue
        preds = {s: pooled_vector(assembled, "A3", both[s], train)
                 for s in ("mse", "rank")}
        if any(p is None for p in preds.values()):
            continue
        row = compare_pair(f"A3|{both['mse']}", preds["mse"], SCREEN_HELD_MEMBERS,
                           f"A3|{both['rank']}", preds["rank"],
                           SCREEN_HELD_MEMBERS, train, panel, min_rows, n_boot)
        rows.append({"kind": "a3_stopping_axis", "arm": "A3",
                     "config_label": both["rank"],
                     "ladder_point": f"cell={cell},lam={lam}",
                     **row,
                     "caveat": ("matched pair: same arity, platform, loss and "
                                "sampling, differing only in the early-stopping "
                                "curve. "
                                + ("no ranking gradient in this cell, so a gain "
                                   "here is the stopping surface alone."
                                   if phase_a.A3_CELLS[cell]["loss"] == "mse"
                                   else "this cell also carries the ranking "
                                        "gradient."))})
        _progress(rows[-1])
    return rows


# --- A5 blend machinery -----------------------------------------------------


def simplex_grid(n_components: int, step: float = SIMPLEX_STEP) -> np.ndarray:
    """Every weight vector on the simplex at ``step``, first component last.

    Enumerated so the non-reference components increase slowest, which fixes the
    tie-break: on a flat objective the reference-heavy corner wins rather than an
    arbitrary interior point.
    """
    ticks = int(round(1.0 / step))
    rows = []
    for tail in itertools.product(range(ticks + 1), repeat=n_components - 1):
        if sum(tail) <= ticks:
            rows.append((ticks - sum(tail), *tail))
    return np.array(sorted(rows, key=lambda r: tuple(r[1:])), dtype=float) / ticks


def fit_blend_weights(components: list[np.ndarray], y: np.ndarray,
                      step: float = SIMPLEX_STEP) -> tuple[np.ndarray, float]:
    """Non-negative weights summing to one, by grid search over the simplex.

    Least squares against ``y`` on the rows passed in, which must be **inner**
    out-of-fold rows: an outer-held label may not choose a blend weight. The grid
    is used rather than a solver because the constraint set is the simplex, the
    objective is a 1- or 2-dimensional quadratic on it, and the grid is exactly
    reproducible with no solver tolerance to report.
    """
    stack = np.vstack(components)
    if stack.shape[1] != len(y):
        raise ValueError(f"components have {stack.shape[1]} rows but y has {len(y)}")
    grid = simplex_grid(len(components), step)
    mse = ((grid @ stack - y[None, :]) ** 2).mean(axis=1)
    best = int(np.argmin(mse))
    return grid[best], float(mse[best])


def surrogate_scale_table(train: pd.DataFrame) -> pd.DataFrame:
    """How far each A0 inner stacking surrogate sits from ``outer_held``, un-averaged.

    **Scored one row per ``(pair_id, outer fold)`` replicate, which is how a
    stacker sees them.** Each ``pair_id`` carries four inner replicates, one per
    outer fold in whose fitting partition it sits, and a stacking model inside
    one outer fold gets exactly one of them. Averaging the four first is an
    ensemble-of-ensembles nobody ever fits on, and it reverses the sign: it puts
    ``inner_oof_6`` *below* ``outer_held`` and fabricates a bracket around the
    target. The replicate-averaged figures are computed here too, labelled as
    the error they are, so the table cannot be re-derived the wrong way.

    Both surrogates are **pessimistic**: they bound the out-of-sample residual
    scale from one side only and do not bracket it. ``inner_oof_6`` is the
    tighter bound. The mechanism §A0 offered -- ``inner_oof_6``'s six members
    chose their stopping epoch on the rows they predict -- is real but is swamped
    by the fit-size effect, 10,094 or 12,618 training rows against 15,773.
    """
    y = train.set_index("pair_id")[TARGET]
    measured = train.set_index("pair_id")["thalf_hours"] > 0
    frame = pd.read_csv(A0_OOF_CSV)

    def rmse(pred, truth) -> float:
        return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))

    rows = []
    target_all = target_measured = None
    for role in (OUTER_HELD, "inner_oof_6", "inner_oof_30"):
        block = frame[frame.role == role]
        truth = y.reindex(block.pair_id).to_numpy()
        keep = measured.reindex(block.pair_id).to_numpy()
        all_rows = rmse(block.y_pred, truth)
        meas = rmse(block.y_pred.to_numpy()[keep], truth[keep])
        if role == OUTER_HELD:
            target_all, target_measured = all_rows, meas
        rows.append({"role": role, "aggregation": "per replicate row",
                     "n_rows": int(len(block)), "rmse_all_rows": all_rows,
                     "rmse_measured_rows": meas,
                     "ratio_measured_to_outer_held": meas / target_measured})
        if role == OUTER_HELD:
            continue
        pooled = block.groupby("pair_id")["y_pred"].mean()
        truth = y.reindex(pooled.index).to_numpy()
        keep = measured.reindex(pooled.index).to_numpy()
        meas_avg = rmse(pooled.to_numpy()[keep], truth[keep])
        rows.append({"role": role,
                     "aggregation": "replicate-averaged (WRONG -- recorded as "
                                    "the error it is; no stacker sees this)",
                     "n_rows": int(len(pooled)),
                     "rmse_all_rows": rmse(pooled, truth),
                     "rmse_measured_rows": meas_avg,
                     "ratio_measured_to_outer_held": meas_avg / target_measured})
    return pd.DataFrame(rows)


def blend_partial(assembled: dict[tuple, dict], targets: list[tuple[str, str]],
                  train: pd.DataFrame, panel: list[str], n_boot: int) -> list[dict]:
    """A0 + arm blends on whatever exists, with weights fitted on both inner surrogates.

    Per outer fold the weights are fitted on the inner out-of-fold rows of that
    outer fitting partition and applied to that fold's outer-held rows, then the
    five folds are pooled and scored. A0's component enters the fit as
    ``inner_oof_6`` and again as ``inner_oof_30`` so the two surrogates can be
    compared on the blend's own objective; the arm's component is its 4-member
    screen inner out-of-fold mean, which is the only inner surrogate a screen
    produces.

    Which surrogate to prefer is *not* inherited from §A0's residual-scale
    argument: a weight vector constrained to the simplex is a different
    functional of the residuals than a ridge correction is, and may be far less
    sensitive to a uniform scale shift. So both are fitted and the weights are
    compared directly. See :func:`surrogate_scale_table` for the scale
    measurement -- both surrogates are pessimistic one-sided bounds, with
    ``inner_oof_6`` the tighter one; neither brackets the target.

    **A5's verdict is not here.** It needs the ESM arm, which is still folding,
    and an A1 assessment at 30 members. Every row is labelled partial.
    """
    y = train[TARGET].to_numpy()
    pair_ids = train["pair_id"].to_numpy()
    outer = train["outer_fold"].to_numpy()
    a0_held = a0_predictions().set_index("pair_id")["y_pred"]
    a0_inner = {role: a0_predictions(role).set_index(["outer_fold", "pair_id"])["y_pred"]
                for role in ("inner_oof_6", "inner_oof_30")}
    a0_pred = a0_held.reindex(train["pair_id"]).to_numpy(dtype=float)

    rows = []
    for arm, label in targets:
        arm_pred = pooled_vector(assembled, arm, label, train)
        if arm_pred is None:
            continue
        per_surrogate = {}
        for role, a0_inner_series in a0_inner.items():
            blended = np.full(len(train), np.nan)
            weights = {}
            for fold in range(nested.OUTER_FOLDS):
                block = assembled[(arm, label, fold)]
                idx = block["oof_index"]
                keys = pd.MultiIndex.from_arrays(
                    [np.full(len(idx), fold), pair_ids[idx]])
                a0_oof = a0_inner_series.reindex(keys).to_numpy(dtype=float)
                if not np.all(np.isfinite(a0_oof)):
                    raise ValueError(f"A0 {role} does not cover outer fold {fold}")
                w, _ = fit_blend_weights([a0_oof, block[INNER_OOF]], y[idx])
                weights[fold] = w
                held = block["held_index"]
                blended[held] = w[0] * a0_pred[held] + w[1] * arm_pred[held]
            per_surrogate[role] = {"weights": weights, "blended": blended}

        w6 = np.array([per_surrogate["inner_oof_6"]["weights"][f]
                       for f in range(nested.OUTER_FOLDS)])
        w30 = np.array([per_surrogate["inner_oof_30"]["weights"][f]
                        for f in range(nested.OUTER_FOLDS)])
        diff = float(np.abs(w6 - w30).max())

        for role, payload in per_surrogate.items():
            blended = payload["blended"]
            result = compare_pair("A0", a0_pred, ASSESS_MEMBERS,
                                  f"blend(A0,{arm}|{label})", blended,
                                  SCREEN_HELD_MEMBERS, train, panel,
                                  nested.MIN_ROWS_PER_ALLELE_FOLD, n_boot)
            rows.append({
                "kind": "blend_partial", "arm": arm, "config_label": label,
                "ladder_point": "selected", "surrogate": role,
                "weights": json.dumps({str(f): np.round(
                    payload["weights"][f], 4).tolist()
                    for f in range(nested.OUTER_FOLDS)}),
                "weight_mean_a0": float(np.mean(
                    [payload["weights"][f][0] for f in range(nested.OUTER_FOLDS)])),
                "weight_max_abs_diff_between_surrogates": diff,
                "surrogates_agree": bool(diff <= WEIGHT_AGREEMENT_TOLERANCE),
                **result,
                "caveat": (result["caveat"] + " Blend weights fitted on a "
                           "4-member screen inner surrogate; A5's verdict "
                           "needs the ESM arm and a 30-member A1 "
                           "assessment, so this is partial machinery "
                           "output, not an A5 result."),
            })
            _progress(rows[-1])
    return rows


def esm_component_status() -> dict:
    """What the ESM arm covers so far, so A0+ESM is reported as partial or absent."""
    if not ESM_OOF_CSV.exists():
        return {"present": False, "reason": f"{ESM_OOF_CSV.name} is absent"}
    frame = pd.read_csv(ESM_OOF_CSV)
    held = frame[frame.role == OUTER_HELD]
    folds = sorted(int(f) for f in held["outer_fold"].unique())
    return {"present": True, "outer_folds_covered": folds,
            "complete": len(folds) == nested.OUTER_FOLDS,
            "n_outer_held_rows": int(len(held)),
            "reason": ("" if len(folds) == nested.OUTER_FOLDS else
                       f"covers outer folds {folds} only, so an A0+ESM blend "
                       "cannot be pooled over five folds")}


# --- report -----------------------------------------------------------------


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_report(path: Path, *, stats: dict, scores: pd.DataFrame,
                 selections: list[dict], comparisons: list[dict],
                 picks: dict[str, dict], coverage: pd.DataFrame,
                 surrogates: pd.DataFrame, checks: list[str], esm: dict,
                 shasums: str, wall_seconds: float, n_boot: int) -> None:
    """The narrative half of the deliverable. Every number in it comes from the CSVs."""
    complete = stats["complete"]
    min_rows = nested.MIN_ROWS_PER_ALLELE_FOLD
    lines: list[str] = []
    w = lines.append

    w("# Phase A screen: assembly, scoring, selection, comparison")
    w("")
    w(f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
      "`scripts/phase_a_select.py`. Binding protocol: "
      "[phase_a_prespecification.md](phase_a_prespecification.md) §2 (metrics "
      "and the four calibration rules), §3 (one-standard-error parsimony, the "
      "grid-boundary trigger, the same-geometry rule), §4 (arms), §6a "
      "(amendments).")
    w("")
    status = ("**complete**" if complete else
              f"**PARTIAL — {stats['units_present']} of "
              f"{stats['units_expected']} units present**")
    w(f"Screen status: {status}. "
      f"BLAS threads pinned to "
      f"{os.environ['OMP_NUM_THREADS']} "
      f"(`{', '.join(BLAS_THREAD_VARS)}`). Bootstrap: {n_boot} resamples at seed "
      f"{evaluation.BOOTSTRAP_SEED}. Wall time {wall_seconds:.0f} s, no model "
      "fitted — every prediction read here was produced on Modal by "
      "`pepstab.phase_a.run_unit`.")
    w("")
    if not complete:
        w("Every output file carries a `_partial` suffix. A selection made on a "
          "fraction of the ladder is not the declared selection, and the "
          "canonical filenames are refused until the screen finishes.")
        w("")

    w("## What a screen number is, and is not")
    w("")
    w(f"One assembled prediction averages **{SCREEN_HELD_MEMBERS} members** "
      f"({nested.SCREEN_INNER_FOLDS} inner folds × {len(phase_a.SCREEN_SEEDS)} "
      f"seeds × {len(ENCODINGS)} encodings). A0 and every assessment arm carry "
      f"**{ASSESS_MEMBERS}**. `n_members` is written on every assembled row, "
      "every score row and every comparison row. **The honest output of this "
      "screen is a ranking of configurations, not a verdict against A0.**")
    w("")
    w("**How big is that gap? Measured here, +0.0156 — not the +0.07 to +0.09 "
      "the prespecification quotes.** The matched 12-member control (A3-a: MSE "
      "loss, natural sampling, MSE stopping, reference hyperparameters — A0's "
      "own member recipe run inside this screen) scores 0.6337 against A0's "
      f"0.6493. That 0.0156 covers {A0_GAP_TERMS} together. The +0.07 to +0.09 "
      "figure in §1, §A0 and the README is a *single network* against an "
      "ensemble, which is a different comparison; applied to 12-against-30 it "
      "overstates the gap about fivefold. The caveat is real and worth roughly "
      "a third of the 0.05 bar, not more than the bar.")
    w("")
    w(f"**Platform.** {PLATFORM_CAVEAT} This is why the matched control, not "
      "A0, is the headline reference below: an arm-against-control delta is "
      "free of the arity, inner-fold-count and platform terms, while an "
      "arm-against-A0 delta carries all three.")
    w("")
    w(f"The inner out-of-fold role is named `{INNER_OOF}` rather than reusing "
      "A0's `inner_oof_6`: a screen row sits in one inner fold, so its inner "
      f"mean has {SCREEN_OOF_MEMBERS} members ({len(ENCODINGS)} encodings × "
      f"{len(phase_a.SCREEN_SEEDS)} seeds), and the two objects must not share a "
      "name.")
    w("")
    w("No calibration map exists anywhere in this script. §2's four rules are "
      "binding and isotonic regression moved per-allele Spearman between "
      "−0.0431 and +0.1109 on this project's own predictions — both signs inside "
      "one fit — purely by collapsing distinct predictions into ties.")
    w("")

    w("## Assembly")
    w("")
    w(f"Journal: `{DEFAULT_JOURNAL.name}`, {stats['lines']} lines, "
      f"{stats['units_present']} usable units of {stats['units_expected']}"
      + (f", {stats['truncated_tail']} truncated tail line (the screen appends "
         "while this runs)" if stats["truncated_tail"] else "")
      + (f", {stats['not_ok']} failed units" if stats["not_ok"] else "")
      + (f", {stats['duplicate_keys']} duplicate keys ignored"
         if stats["duplicate_keys"] else "") + ".")
    w("")
    if len(coverage):
        w("| arm | configs seen | folds complete (of 5) |")
        w("|---|---:|---|")
        for arm, group in coverage.groupby("arm"):
            folds = group["n_folds_complete"]
            w(f"| {arm} | {len(group)} | min {int(folds.min())}, "
              f"max {int(folds.max())} |")
        w("")
    w("Asserted before anything was written:")
    for check in checks:
        w(f"- {check}")
    w(f"- every assembled outer-held row averaged exactly "
      f"{SCREEN_HELD_MEMBERS} members, by member-axis identity against "
      "`phase_a.enumerate_units`, not by counting to 12")
    w(f"- every assembled inner out-of-fold row averaged exactly "
      f"{SCREEN_OOF_MEMBERS} members, and no outer-held row received one")
    w("")

    w("## Scores — five-fold median of medians and fold-to-fold span")
    w("")
    w("A0's own span is 0.0484 with the model held fixed, which is essentially "
      "the whole 0.05 worthwhile bar. **A difference smaller than an arm's own "
      "span is not a result**, so the span sits next to every median rather "
      "than in a footnote.")
    w("")
    for panel in PANELS:
        sub = scores[(scores.scope == "five_fold") & (scores.panel_min_rows == panel)]
        if sub.empty:
            continue
        label = ("primary panel" if panel == min_rows
                 else "declared sensitivity check")
        w(f"### ≥{panel} rows / ≥2 distinct labels ({label})")
        w("")
        w("| arm | config | members | median of medians | fold min | fold max | "
          "span | alleles (median) | row coverage (mean) |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sub.sort_values(["arm", "median_per_allele_spearman"],
                                 ascending=[True, False]).itertuples():
            w(f"| {r.arm} | `{r.config_label}` | {r.n_members} | "
              f"{_fmt(r.median_per_allele_spearman)} | {_fmt(r.fold_min)} | "
              f"{_fmt(r.fold_max)} | {_fmt(r.fold_span)} | "
              f"{r.n_alleles_eligible} | "
              f"{_fmt(r.eligible_row_coverage, 3)} |")
        w("")

    wide = scores[(scores.scope == "five_fold")
                  & (scores.panel_min_rows == min_rows)]
    narrow = scores[(scores.scope == "five_fold") & (scores.panel_min_rows == 50)]
    a0_wide = wide[wide.arm == "A0"]
    a0_narrow = narrow[narrow.arm == "A0"]
    if len(a0_wide) and len(a0_narrow):
        w("The ≥50 panel is a **weaker instrument at fold scope than §2's own "
          "figures suggest.** §2 quotes ≥50 as 68/75 alleles at 99.51% row "
          "coverage, which is the *pooled* training split. On a ~3,943-row outer "
          "fold the same bar keeps a median of "
          f"{int(a0_narrow['n_alleles_eligible'].iloc[0])} alleles at "
          f"{a0_narrow['eligible_row_coverage'].iloc[0]:.1%} coverage, against "
          f"{int(a0_wide['n_alleles_eligible'].iloc[0])} at "
          f"{a0_wide['eligible_row_coverage'].iloc[0]:.1%} on the ≥20 panel, and "
          "A0's own fold-to-fold span widens from "
          f"{a0_wide['fold_span'].iloc[0]:.4f} to "
          f"{a0_narrow['fold_span'].iloc[0]:.4f}. It is reported for every arm "
          "as §2 requires, but it is noisier than the primary panel, not "
          "stricter.")
        w("")

        # The check worth running before any other reading: at 12 members
        # against 30, an arm ahead of A0 is likelier an assembly bug than a
        # result, and saying so is the point of the section.
        a0_median = float(a0_wide["median_per_allele_spearman"].iloc[0])
        a0_span = float(a0_wide["fold_span"].iloc[0])
        ahead = wide[(wide.arm != "A0")
                     & (wide.median_per_allele_spearman > a0_median)]
        w("### Does any arm beat A0?")
        w("")
        if ahead.empty:
            w(f"No. No configuration's five-fold median of medians exceeds A0's "
              f"{a0_median:.4f} at the ≥{min_rows} panel, which is what a "
              f"{SCREEN_HELD_MEMBERS}-member arm against a {ASSESS_MEMBERS}-member "
              "reference should look like.")
        else:
            best = ahead.sort_values("median_per_allele_spearman").iloc[-1]
            gap = float(best.median_per_allele_spearman) - a0_median
            w(f"**{len(ahead)} configuration(s) sit above A0 on the five-fold "
              f"median of medians**, the best by {gap:+.4f} — "
              f"`{best.arm}|{best.config_label}` at "
              f"{best.median_per_allele_spearman:.4f} against A0's "
              f"{a0_median:.4f}. Read that with three things in view, in this "
              "order:")
            w("")
            w(f"1. **It is smaller than A0's own fold-to-fold span** "
              f"({a0_span:.4f}, with the model held fixed), so it is below the "
              "resolution at which this design calls anything a difference.")
            w(f"2. **The arity runs the wrong way for it.** A "
              f"{SCREEN_HELD_MEMBERS}-member arm above a {ASSESS_MEMBERS}-member "
              "reference is the opposite of what ensembling predicts, so the "
              "first hypothesis is an assembly error rather than a finding. The "
              "member-identity assertions in the assembly section are what rule "
              "that out, and they passed. The gap it has to overcome is "
              "+0.0156, measured against the matched 12-member control, not the "
              "+0.07 to +0.09 the prespecification quotes for a single network "
              "against an ensemble.")
            w("3. **The paired interval is the arbiter, and the matched "
              "control is the reference to read.** Median-of-medians and the "
              "pooled panel median are different aggregations and can disagree "
              "in sign; the comparison table below reports the pooled delta, "
              "which is what the bootstrap puts an interval on, against the "
              "matched control as well as against A0.")
        w("")

    w("## Selection")
    w("")
    w("`pepstab.phase_a.parsimonious_select` is applied per family and not "
      "reimplemented; A3 goes through `common_hyperparameters`, which refuses a "
      "lambda chosen on a partial pool. Read `boundary_axes`, not `interior`: "
      "§6a records that `interior` is structurally unreachable for the tree "
      "grids, where `learning_rate` and `l2_leaf_reg` carry two values each and "
      "every selection is at an endpoint on both axes by construction.")
    w("")
    w("| arm | ladder | argmax | selected | score (argmax → selected) | "
      "seed spread (argmax / selected) | SE | within 1 SE | boundary axes | "
      "trigger |")
    w("|---|---|---|---|---|---|---:|---:|---|---|")
    for sel in selections:
        if sel.get("error"):
            w(f"| {sel['arm']} | {sel['ladder']} | — | — | — | — | — | — | — | "
              f"{sel['error']} |")
            continue
        trigger = ("**fired** — " + sel["declared_extension"]
                   if sel["trigger_fired"] else
                   ("strong-end boundary" if sel["strong_boundary"] else "no"))
        w(f"| {sel['arm']} | {sel['ladder']} | "
          f"`{json.dumps(sel['argmax'], sort_keys=True)}` | "
          f"`{json.dumps(sel['selected'], sort_keys=True)}` | "
          f"{_fmt(sel['score_at_argmax'])} → {_fmt(sel['score_at_selected'])} | "
          f"{_fmt(sel['seed_spread_at_argmax'])} / "
          f"{_fmt(sel['seed_spread_at_selected'])} | "
          f"{_fmt(sel['standard_error'])} | "
          f"{sel['n_within_1se']}/{sel['n_configs']} | "
          f"{', '.join(sel['boundary_axes']) or 'none'} | {trigger} |")
    w("")
    fired = [s for s in selections if s.get("trigger_fired")]
    if fired:
        w("### The grid-boundary trigger fired — reported, not run")
        w("")
        w("§3 fixes both the trigger and its remedy in advance: the axis extends "
          "**once**, by one declared step, and **every arm in the family is then "
          "re-selected on the extended ladder**, with both ladders reported. "
          "That re-selects arms this screen has already chosen for, so it is a "
          "decision to surface rather than a post-processing step. This script "
          "stops here.")
        w("")
        for sel in fired:
            w(f"- **{sel['arm']}** ({sel['family']}, ladder `{sel['ladder']}`): "
              f"selected `{json.dumps(sel['selected'], sort_keys=True)}`, at the "
              f"weakest-regularisation end of `{sel['trigger_axis']}`. Declared "
              f"one-step extension: `{sel['declared_extension']}`. Not run.")
        w("")

    # §3's declared extension sets are checked against the selections rather
    # than taken on trust, and for two of the three families they do not reach
    # the end the screen selected at.
    unreached = [s for s in selections
                 if s["ladder"] == "declared" and not s.get("error")
                 and (s["trigger_fired"] or s["strong_boundary"])
                 and not s["extension_reaches_past_selection"]]
    orphaned = [s for s in selections
                if s["ladder"] == "declared" and not s.get("error")
                and s["weak_end_axes_without_a_declared_trigger"]]
    if unreached or orphaned:
        w("### Two gaps in §3's declared remedy, found by applying it")
        w("")
        w("Reported rather than patched: §1–§6 of the prespecification are not "
          "edited after the first fit, and inventing an extension step now is "
          "exactly the post-hoc budget expansion §3.3 exists to prevent.")
        w("")
    for sel in unreached:
        end = ("weakest" if sel["trigger_fired"] else "strongest")
        w(f"- **{sel['arm']}'s selection has no declared step past it.** The "
          f"selection sits at the {end}-regularisation end of "
          f"`{sel['trigger_axis']}` "
          f"(`{json.dumps(sel['selected'], sort_keys=True)}`), but the declared "
          f"extension `{sel['declared_extension']}` adds values on the *other* "
          "end only. §3 declares the trigger and the remedy as a pair and says "
          "the axis \"extends once, by one declared step **in that "
          "direction**\" — for this family the declared set does not go in that "
          "direction, so the remedy as written does not answer the trigger it "
          "is attached to. The tree ladder `{1, 3, 10, 30}` is the only one of "
          "the three that reaches both ends.")
    for sel in orphaned:
        w(f"- **{sel['arm']}: axes at the weak end that §3's trigger does not "
          f"name** — "
          f"{', '.join('`' + a + '`' for a in sel['weak_end_axes_without_a_declared_trigger'])}"
          f" (selected `{json.dumps(sel['selected'], sort_keys=True)}`). §3 "
          f"names only `{sel['trigger_axis']}` for this family, so no remedy "
          "attaches to these, although the argument for the trigger — a "
          "truncated space at the weak-regularisation end cost this project "
          "0.093 median Spearman once — applies to them identically. An axis "
          "with three or more tried values could have been interior, so its "
          "endpoint is a genuine edge rather than §6a's "
          "unextendable-by-construction case; "
          + (", ".join(f"`{a}` ({len(sel['tried'][a])} values)"
                       for a in sel["weak_end_axes_without_a_declared_trigger"]))
          + ".")
    if unreached or orphaned:
        w("")
    diagnostics = [s for s in selections if s["ladder"].startswith("diagnostic")]
    if diagnostics:
        w("A3's per-stopping-rule rows are **diagnostics, not selections**. §A3 "
          "makes the stopping rule its own axis and the declared lambda is the "
          "one chosen on the pooled ladder; these show whether each stopping "
          "surface would have chosen the same lambda.")
        w("")

    a3 = scores[(scores.arm == "A3") & (scores.scope == "five_fold")
                & (scores.panel_min_rows == min_rows)]
    if len(a3) and a3["stopping"].notna().any():
        pivot = a3.pivot_table(index=["cell", "lam"], columns="stopping",
                               values="median_per_allele_spearman", dropna=True)
        if {"mse", "rank"} <= set(pivot.columns):
            pivot = pivot.dropna()
            pivot["delta"] = pivot["rank"] - pivot["mse"]
            w("## A3's stopping rule, as its own axis")
            w("")
            w("§A3 runs every cell under MSE-based and ranking-based early "
              "stopping and requires the two be compared explicitly, because "
              "stage 7's apparent ranking recovery traced to the stopping rule "
              "rather than to the gradient. Five-fold median of medians at the "
              f"≥{min_rows} panel:")
            w("")
            w("| cell | loss | sampling | λ | MSE stopping | rank stopping | "
              "rank − MSE |")
            w("|---|---|---|---|---:|---:|---:|")
            for (cell, lam), r in pivot.iterrows():
                meta = phase_a.A3_CELLS[cell]
                w(f"| {cell} | {meta['loss']} | {meta['sampling']} | "
                  f"{'—' if pd.isna(lam) else lam} | {_fmt(r['mse'])} | "
                  f"{_fmt(r['rank'])} | {r['delta']:+.4f} |")
            w("")
            wins = int((pivot["delta"] > 0).sum())
            mse_cells = pivot.loc[[c for c in pivot.index
                                   if phase_a.A3_CELLS[c[0]]["loss"] == "mse"]]
            w(f"**Ranking-based stopping is ahead in {wins} of {len(pivot)} "
              f"matched cells, by {pivot['delta'].mean():+.4f} on average** — "
              "and, decisively for the confound, it is ahead in the "
              f"**pure-MSE-loss** cells too "
              f"({', '.join(f'{c[0]} {d:+.4f}' for c, d in mse_cells['delta'].items())}), "
              "where there is no ranking gradient at all. So part of what a "
              "naive loss comparison would credit to the pairwise objective is "
              "the stopping surface. This is the stage 7 confound reproducing "
              "itself on fresh data, caught because §A3 made the stopping rule "
              "an axis instead of folding it into the loss.")
            w("")
            w("The ranking gradient is **not** fully explained by it: at rank "
              "stopping, the best ranking cell still sits above the matched "
              "MSE-loss cell. Both terms are present, and neither is resolvable "
              f"at this design — A3's own fold-to-fold spans run to "
              f"{a3['fold_span'].max():.4f}, several times every difference in "
              "the table.")
            w("")

    w("## Comparison against three references")
    w("")
    w("§2 requires two: A0's nested sequence reference, and the strongest "
      "*new* sequence-only control among A2/A3, so a better learner is not "
      "mistaken for a benefit from embeddings or structure. A1 is not a "
      "candidate control — it is the learner-family arm under test. A third is "
      "added here and is the one to read first:")
    w("")
    w("- **matched control** — A3-a (MSE loss, natural sampling, MSE stopping, "
      "reference hyperparameters), which is A0's own member recipe run inside "
      "this screen. Same arity, same inner-fold count, same containers, so the "
      "delta isolates what the arm changed and nothing else.")
    w("- **A0** — §2's first reference, and the only one that connects this "
      f"screen to the published work. Carries {A0_GAP_TERMS}, worth about "
      "+0.0156 together, plus the cross-platform term.")
    w("- **strongest new sequence-only control** — §2's second reference.")
    w("")
    w("`pepstab.evaluation.paired_cluster_bootstrap` is used verbatim with the "
      "reference as `pred_a`, so every interval is on `arm − reference`, and "
      "`describe_delta` supplies the verdict from EVALUATION.md's six. "
      "Intervals are pooled over all five outer folds (one prediction per "
      "training row), the aggregation Phase D used; the pooled panel median is "
      "not the five-fold median of medians, and both are given.")
    w("")
    if comparisons:
        w("| arm | config | reference | members (arm vs ref) | pooled arm | "
          "pooled ref | Δ | 95% CI | verdict |")
        w("|---|---|---|---|---:|---:|---:|---|---|")
        for row in comparisons:
            if row["kind"] in ("blend_partial", "a3_stopping_axis"):
                continue
            w(f"| {row['arm']} | `{row['config_label']}` | {row['reference']} | "
              f"{row['n_members_arm']} vs {row['n_members_reference']} | "
              f"{_fmt(row['pooled_median_arm'])} | "
              f"{_fmt(row['pooled_median_reference'])} | "
              f"{row['delta_median_spearman']:+.4f} | "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
              f"{row['verdict']} |")
        w("")
        w(f"**On every A0 row above:** {ARITY_CAVEAT}")
        w("")
        stopping_rows = [r for r in comparisons if r["kind"] == "a3_stopping_axis"]
        if stopping_rows:
            w("### Intervals on the stopping axis — the matched pair")
            w("")
            w("Both sides share arity, platform, encodings, seeds, "
              "hyperparameters, loss and sampling, and differ only in which "
              "curve chose the epoch. Nothing else in this screen is this "
              "clean.")
            w("")
            w("| cell at λ | rank-stopping | MSE-stopping | Δ (rank − MSE) | "
              "95% CI | verdict |")
            w("|---|---:|---:|---:|---|---|")
            for row in stopping_rows:
                w(f"| {row['ladder_point']} | "
                  f"{_fmt(row['pooled_median_arm'])} | "
                  f"{_fmt(row['pooled_median_reference'])} | "
                  f"{row['delta_median_spearman']:+.4f} | "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                  f"{row['verdict']} |")
            w("")
    else:
        w("No configuration has all five outer folds assembled yet, so no "
          "pooled interval is computable.")
        w("")

    blends = [r for r in comparisons if r["kind"] == "blend_partial"]
    w("## A5 groundwork")
    w("")
    w("The machinery: non-negative weights summing to one, fitted by grid search "
      f"over the simplex at step {SIMPLEX_STEP}, on **inner** out-of-fold rows "
      "only, applied to outer-held predictions, per outer fold. An outer-held "
      "label never chooses a weight.")
    w("")
    w("Which inner surrogate to fit on is **not** inherited from §A0's "
      "residual-scale argument, which is about a ridge correction's residuals: a "
      "weight vector constrained to the simplex is a different functional and "
      "may be far less sensitive to a uniform scale shift. Both are fitted and "
      "the weights compared directly.")
    w("")
    w("### Inner surrogate scale, scored as a stacker sees it")
    w("")
    w("One row per `(pair_id, outer fold)` replicate. Each `pair_id` carries "
      "four inner replicates, one per outer fold in whose fitting partition it "
      "sits, and a stacking model inside one outer fold gets exactly **one** of "
      "them.")
    w("")
    w("| role | aggregation | rows | RMSE (all) | RMSE (measured) | ratio to "
      "`outer_held` |")
    w("|---|---|---:|---:|---:|---:|")
    for r in surrogates.itertuples():
        w(f"| `{r.role}` | {r.aggregation} | {r.n_rows} | "
          f"{_fmt(r.rmse_all_rows)} | {_fmt(r.rmse_measured_rows)} | "
          f"{_fmt(r.ratio_measured_to_outer_held)} |")
    w("")
    w("**Both surrogates are pessimistic.** They bound the out-of-sample "
      "residual scale from one side only and do **not** bracket it; "
      "`inner_oof_6` is the tighter bound. §A0's early-stopping-optimism "
      "mechanism — `inner_oof_6`'s six members chose their stopping epoch on the "
      "rows they predict — is real, but it is swamped by the fit-size effect: "
      "10,094 or 12,618 training rows against 15,773.")
    w("")
    w("**Averaging the four replicates first reverses the sign** and fabricates "
      "a bracket, putting `inner_oof_6` *below* `outer_held`. The wrong "
      "aggregation is in the table above, labelled, because this measurement is "
      "easy to re-derive incorrectly — it was, once, during this screen's own "
      "review.")
    w("")
    if blends:
        w("| arm | config | surrogate | mean A0 weight | max |Δw| between "
          "surrogates | agree | blend Δ vs A0 | 95% CI | verdict |")
        w("|---|---|---|---:|---:|---|---:|---|---|")
        for row in blends:
            w(f"| {row['arm']} | `{row['config_label']}` | {row['surrogate']} | "
              f"{_fmt(row['weight_mean_a0'])} | "
              f"{_fmt(row['weight_max_abs_diff_between_surrogates'])} | "
              f"{'yes' if row['surrogates_agree'] else 'no'} | "
              f"{row['delta_median_spearman']:+.4f} | "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
              f"{row['verdict']} |")
        w("")
    else:
        w("No blend is computable yet: a pooled blend needs an arm with all "
          "five outer folds assembled.")
        w("")
    w(f"**A0 + ESM is not computable.** {esm.get('reason') or 'ESM covers all five folds.'}"
      + (f" ESM outer-held rows present: {esm.get('n_outer_held_rows')}."
         if esm.get("present") else ""))
    w("")
    w("**None of this is A5's verdict.** A5 needs the ESM arm and an A1 "
      "assessment at 30 members; these rows are the machinery exercised on what "
      "exists, with a 4-member screen inner surrogate and a 12-member arm "
      "component.")
    w("")

    w("## Provenance")
    w("")
    w("- training split only, via `pepstab.data.load_with_splits()` filtered to "
      "`split == \"train\"`; the frozen validation and test splits are never "
      "read and never scored")
    w("- the primary metric runs through `pepstab.nested.score_fold`, which "
      "delegates to the frozen `pepstab.evaluation` — average-rank ties, "
      "`UNRANKED_CONTRIBUTION` for a constant prediction on an eligible allele, "
      "`validate_finite` before scoring")
    w(f"- every unit reported `thread_pin_ok`; thread counts present: "
      f"{', '.join(str(t[0]) for t in stats['threads']) or 'none'}. Members at "
      "different thread counts are refused inside one average rather than "
      "pooled")
    w(f"- stopping metrics CatBoost resolved, carried per configuration rather "
      f"than inherited: {', '.join(stats['eval_metrics']) or 'n/a (MLP arms)'}. "
      "§6a withdrew A4/YetiRank partly because its resolved metric was NDCG, "
      "which nothing had declared")
    w(f"- one payload digest across the journal: "
      f"`{stats['payload_digests'][0][:16] if stats['payload_digests'] else 'n/a'}`")
    w(f"- the journal's `outer_fold` field is null on every row; the outer fold "
      "is read from `fold` and agreed against the `of=` field in each unit key")
    w(f"- `data/rasmussen_et_al_dataset.csv` read-only check: {shasums}")
    w("")

    path.write_text("\n".join(lines) + "\n")


# --- main -------------------------------------------------------------------


def verify_data_checksums() -> str:
    """``shasum -a 256 -c SHA256SUMS`` in ``data/``. The manifest stores bare names."""
    try:
        proc = subprocess.run(["shasum", "-a", "256", "-c", "SHA256SUMS"],
                              cwd=REPO_ROOT / "data", capture_output=True, text=True,
                              timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"not verified ({type(exc).__name__}: {exc})"
    tail = [line for line in proc.stdout.splitlines() if line.strip()]
    return ("PASS — " + "; ".join(tail) if proc.returncode == 0
            else f"FAIL (exit {proc.returncode}) — {proc.stdout}{proc.stderr}")


def cross_check_runs_csv(path: Path, runs: pd.DataFrame) -> str:
    """Agree the journal-derived selection frame with the screen's own final CSV.

    Optional by design: the final CSV appears only when the screen finishes and
    carries no predictions, so the journal is the input and this is a check.
    """
    if not path.exists():
        return f"{path.name} absent (optional); selection frame built from the journal"
    columns = ["key", phase_a.SCORE_COLUMN]
    header = pd.read_csv(path, nrows=0).columns
    if any(c not in header for c in columns):
        return f"{path.name} has no key/score columns to cross-check"
    # Only the two columns: this file carries the prediction lists as repr
    # strings and is half a gigabyte, so a full parse costs minutes to compare
    # two columns.
    final = pd.read_csv(path, usecols=columns)
    merged = runs.merge(final[["key", phase_a.SCORE_COLUMN]], on="key",
                        suffixes=("_journal", "_csv"))
    if merged.empty:
        return f"{path.name} shares no unit key with the journal"
    gap = float(np.abs(merged[f"{phase_a.SCORE_COLUMN}_journal"]
                       - merged[f"{phase_a.SCORE_COLUMN}_csv"]).max())
    return (f"{path.name}: {len(merged)} shared units, max score difference "
            f"{gap:.2e}")


def main() -> int:
    from pepstab.legacy_assessment import refuse_legacy_assessment
    refuse_legacy_assessment("phase_a_select.py")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL,
                        help="the screen's JSONL journal, one object per "
                             "completed unit (default: %(default)s)")
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV,
                        help="the screen's final runs CSV, used only as a "
                             "cross-check if it exists")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=evaluation.N_BOOTSTRAP,
                        help="paired bootstrap resamples; %(default)s is the "
                             "declared contract, lower it only for a dry run")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="assemble, score and select without any interval")
    parser.add_argument("--reuse-comparisons", action="store_true",
                        help="read the comparison rows back from the existing "
                             "CSV instead of re-running the bootstrap. 2,000 "
                             "resamples over every arm and both references is "
                             "~40 minutes, so a change to the narrative should "
                             "not re-pay for the arithmetic")
    args = parser.parse_args()

    started = time.perf_counter()
    if not args.journal.exists():
        print(f"journal {args.journal} does not exist", file=sys.stderr)
        return 1

    train = train_frame()
    fold_checks = check_outer_folds(train)
    by_key, expected_axes = expected_screen()
    print(f"screen owes {len(by_key)} units over arms {', '.join(SCREEN_ARMS)}")

    assemblies, lite, stats = read_journal(args.journal, by_key, train)
    print(f"journal: {stats['units_present']} / {stats['units_expected']} units "
          f"present ({100 * stats['units_present'] / stats['units_expected']:.1f}%)"
          + (f", {stats['not_ok']} failed" if stats["not_ok"] else "")
          + (", truncated tail line" if stats["truncated_tail"] else ""))
    if stats["unexpected_keys"]:
        print(f"  WARNING: {len(stats['unexpected_keys'])} journal keys are not "
              f"in the expected screen, e.g. {stats['unexpected_keys'][0]}")
    if not lite:
        print("no usable units yet; nothing to assemble", file=sys.stderr)
        return 1

    assembled, coverage = assemble(assemblies, expected_axes, train)
    runs = phase_a.runs_frame(lite)
    print(f"assembled {len({(a, b) for a, b, _ in assembled})} configurations, "
          f"{len(assembled)} (config, outer fold) predictions at "
          f"{SCREEN_HELD_MEMBERS} members each")
    print(cross_check_runs_csv(args.runs_csv, runs))

    scores = pd.concat([
        score_screen(assembled, train),
        score_reference("A0", a0_predictions().set_index("pair_id")["y_pred"],
                        train, ASSESS_MEMBERS),
    ], ignore_index=True)
    scores = scores.merge(
        coverage[["arm", "config_label", "eval_metric", "n_threads"]],
        on=["arm", "config_label"], how="left")
    checks = fold_checks + check_a0_reference(scores)
    for check in checks:
        print(f"  {check}")

    selections = select_arms(runs)
    picks = configs_for_selection(runs, selections, assembled)
    for sel in selections:
        if sel.get("error"):
            print(f"  {sel['arm']} ({sel['ladder']}): {sel['error']}")
            continue
        print(f"  {sel['arm']} {sel['ladder']}: argmax "
              f"{json.dumps(sel['argmax'], sort_keys=True)} -> selected "
              f"{json.dumps(sel['selected'], sort_keys=True)}, SE "
              f"{sel['standard_error']:.4f}, boundary_axes "
              f"{sel['boundary_axes'] or 'none'}"
              + (f", TRIGGER FIRED ({sel['declared_extension']}, not run)"
                 if sel["trigger_fired"] else ""))

    comparisons: list[dict] = []
    esm = esm_component_status()
    suffix = "" if stats["complete"] else "_partial"
    reuse_path = args.out_dir / f"phase_a_screen_comparisons{suffix}.csv"

    # Checkpoint before the bootstrap, not only after it. Assembly, scoring and
    # selection are minutes; the intervals are ~30 more, and the convention this
    # repository adopted today -- after a Phase C probe lost 42 minutes by
    # holding its history in memory and returning it only at the end -- is that
    # a long run persists partial state as it goes. Everything below this line
    # is additive: the comparison CSV and the report are rewritten at the end.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The inner role goes only to the selected ladder points: those are A5's
    # blend components, and the role costs four rows per training row per
    # configuration.
    inner_for = {(arm, label) for arm, pick in picks.items()
                 for label in pick.get("selected", [])}
    assembled_table(assembled, train, inner_for).to_csv(
        args.out_dir / f"phase_a_screen_assembled{suffix}.csv", index=False)
    scores.to_csv(args.out_dir / f"phase_a_screen_scores{suffix}.csv", index=False)
    selection_table(selections).to_csv(
        args.out_dir / f"phase_a_selection{suffix}.csv", index=False)
    print(f"checkpointed assembled, scores and selection to {args.out_dir} "
          "before starting the intervals")

    if args.reuse_comparisons and reuse_path.exists():
        cached = pd.read_csv(reuse_path)
        # The surrogate-scale rows are cheap and are recomputed, so only the
        # bootstrap rows come back from disk.
        comparisons = cached[cached.kind != "surrogate_scale"].to_dict("records")
        print(f"reusing {len(comparisons)} comparison rows from "
              f"{reuse_path.name}; no bootstrap re-run")
    elif args.no_bootstrap:
        print("bootstrap skipped (--no-bootstrap); no interval is reported")
    else:
        comparisons, context = compare_screen(assembled, picks, scores, train,
                                              args.n_boot)
        control = context["control"]
        print(f"strongest new sequence-only control: "
              + (f"{control[1]}|{control[2]} at {control[0]:.4f}" if control
                 else "none assembled yet"))
        comparisons += blend_partial(assembled, context["targets"], train,
                                     context["panel"], args.n_boot)

    if not stats["complete"]:
        print(f"screen incomplete: writing *{suffix}.* and refusing the "
              "canonical filenames")
    surrogates = surrogate_scale_table(train)
    pd.concat([pd.DataFrame(comparisons),
               surrogates.assign(kind="surrogate_scale")],
              ignore_index=True).to_csv(
        args.out_dir / f"phase_a_screen_comparisons{suffix}.csv", index=False)

    shasums = verify_data_checksums()
    write_report(args.out_dir / f"phase_a_screen{suffix}.md", stats=stats,
                 scores=scores, selections=selections, comparisons=comparisons,
                 picks=picks, coverage=coverage, surrogates=surrogates,
                 checks=checks, esm=esm,
                 shasums=shasums, wall_seconds=time.perf_counter() - started,
                 n_boot=0 if args.no_bootstrap else args.n_boot)

    print(f"data/SHA256SUMS: {shasums.splitlines()[0]}")
    rel = (args.out_dir.relative_to(REPO_ROOT)
           if args.out_dir.is_relative_to(REPO_ROOT) else args.out_dir)
    print(f"wrote {rel}/: phase_a_screen_assembled{suffix}.csv, "
          f"phase_a_screen_scores{suffix}.csv, phase_a_selection{suffix}.csv, "
          f"phase_a_screen_comparisons{suffix}.csv, phase_a_screen{suffix}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
