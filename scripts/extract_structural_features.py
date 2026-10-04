#!/usr/bin/env python
"""Stage 4c.5 structural feature extraction.

Three modes:

``pilot``
    Extract the 90 local stage-4c pilot folds (both models, all arms and
    seeds). Free, local, no cloud. This is the dry run for every parsing and
    feature-definition question.

``extract``
    Extract one tree of production folds into a feature table. A single
    profile's half is a **first-class** mode, not a failure mode: the two Modal
    workspaces hold disjoint, pair-by-pair-interleaved halves, so either half
    alone is already balanced across alleles and splits and is a usable
    unbiased diagnostic on ~14,083 pairs. The output is labelled with the
    profile and its ``cohort_coverage`` column says ``half``.

``concat``
    Concatenate per-profile halves into the full cohort table. Separate step,
    and it asserts the expected row count -- nothing in the output path makes a
    half-sized table obvious, and a half-cohort table that silently becomes
    "the" feature table is the kind of error that reaches a results section.

Usage::

    python scripts/extract_structural_features.py pilot
    python scripts/extract_structural_features.py extract --root <dir> \
        --profile a-cheparukhin --out reports/stage4c5_features_a-cheparukhin.csv
    python scripts/extract_structural_features.py concat \
        reports/stage4c5_features_a-cheparukhin.csv \
        reports/stage4c5_features_colleague.csv \
        --out reports/stage4c5_features_production.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pepstab.structural_features import (  # noqa: E402
    alpha3_provenance,
    attach_pair_ids,
    discover_folds,
    extract_path,
)

PILOT_ROOT = ROOT / "structures/ectodomain_pilot/ectodomain-20261004"
COHORT_CSV = ROOT / "data/structural_cohort.csv"
PROFILES = ("a-cheparukhin", "colleague")

# Seven workstreams share this 8-core / 16 GB machine and two long-lived fold
# driver processes must survive the night, so the pool stays at 1 by default.
DEFAULT_WORKERS = 1


def _run(folders: list[Path], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [extract_path(f) for f in folders]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(extract_path, folders, chunksize=4))
    return pd.DataFrame(rows)


def _cohort() -> pd.DataFrame:
    """The frozen production cohort. Loaded, never recomputed."""
    return pd.read_csv(COHORT_CSV)


def _splits() -> pd.DataFrame:
    from pepstab.data import load_with_splits

    return load_with_splits()


def _annotate(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach split/pair_id and the alpha3 construct provenance.

    Three alleles fold a partly-synthetic 275-residue construct (alpha3 taken
    from a relative). The flag travels with the features so stage 5 can test
    whether those rows behave differently instead of the caveat living only in
    prose.
    """
    frame = attach_pair_ids(frame, _splits())
    return frame.merge(alpha3_provenance(), on="allele", how="left")


def _summarise(frame: pd.DataFrame) -> dict:
    ok = frame["status"] == "ok"
    out = {
        "rows": int(len(frame)),
        "ok": int(ok.sum()),
        "failed": int((~ok).sum()),
    }
    if (~ok).any():
        out["failures"] = (
            frame.loc[~ok, "status"].value_counts().head(10).to_dict()
        )
    return out


def cmd_pilot(args: argparse.Namespace) -> int:
    folders = list(discover_folds(args.root))
    print(f"discovered {len(folders)} pilot folds under {args.root}")
    frame = _run(folders, args.workers)
    frame = _annotate(frame)
    frame.insert(0, "run_id", args.root.name)
    frame["cohort_coverage"] = "pilot"
    frame = frame.sort_values(["model", "complex_id", "arm", "seed"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = _summarise(frame)
    summary["unmatched_pair_id"] = int(frame["pair_id"].isna().sum())
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} ({len(frame)} rows x {frame.shape[1]} columns)")
    if summary["failed"]:
        return 1
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    folders = list(discover_folds(args.root))
    if args.limit:
        folders = folders[: args.limit]
    print(f"discovered {len(folders)} folds under {args.root} (excluded trees skipped)")
    if not folders:
        print("nothing to extract", file=sys.stderr)
        return 1
    frame = _run(folders, args.workers)
    frame = _annotate(frame)
    frame["profile"] = args.profile
    frame["cohort_coverage"] = "half" if args.profile else "unknown"

    cohort = _cohort()
    if args.profile:
        expected = cohort[cohort["profile"] == args.profile]
        print(
            f"profile {args.profile}: {len(expected)} pairs in the frozen cohort, "
            f"{frame['status'].eq('ok').sum()} extracted ok"
        )
        wrong = frame.merge(
            cohort[["allele", "peptide", "profile"]].rename(columns={"profile": "cohort_profile"}),
            on=["allele", "peptide"],
            how="left",
        )
        mismatched = wrong["cohort_profile"].ne(args.profile) & wrong["cohort_profile"].notna()
        if mismatched.any():
            print(
                f"ERROR: {int(mismatched.sum())} extracted pairs belong to the other "
                "profile's half -- the wrong Volume was mounted",
                file=sys.stderr,
            )
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(json.dumps(_summarise(frame), indent=2))
    print(f"wrote {args.out} ({len(frame)} rows x {frame.shape[1]} columns)")
    return 0


def cmd_concat(args: argparse.Namespace) -> int:
    parts = [pd.read_csv(p) for p in args.inputs]
    frame = pd.concat(parts, ignore_index=True)
    cohort = _cohort()
    expected = len(cohort)

    dups = frame.duplicated(["model", "allele", "peptide", "arm", "seed"]).sum()
    if dups:
        print(f"ERROR: {dups} duplicate (model, allele, peptide, arm, seed) keys", file=sys.stderr)
        return 1

    covered = set(zip(frame["allele"], frame["peptide"]))
    cohort_pairs = set(zip(cohort["allele"], cohort["peptide"]))
    missing = cohort_pairs - covered
    extra = covered - cohort_pairs
    print(
        json.dumps(
            {
                "rows": int(len(frame)),
                "expected_cohort_pairs": expected,
                "cohort_pairs_covered": len(cohort_pairs & covered),
                "cohort_pairs_missing": len(missing),
                "rows_not_in_cohort": len(extra),
                "profiles": sorted(frame.get("profile", pd.Series(dtype=str)).dropna().unique().tolist()),
            },
            indent=2,
        )
    )
    frame["cohort_coverage"] = "full" if not missing else "partial"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    if args.require_full and (missing or len(frame) != expected):
        print(
            f"ERROR: --require-full set but the table holds {len(frame)} rows covering "
            f"{len(cohort_pairs & covered)} of {expected} cohort pairs",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="process pool size; capped low, this machine is shared")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pilot", help="extract the 90 local stage-4c pilot folds")
    p.add_argument("--root", type=Path, default=PILOT_ROOT)
    p.add_argument("--out", type=Path, default=ROOT / "reports/stage4c5_pilot_features.csv")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("extract", help="extract one tree of folds (one profile's half)")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", choices=PROFILES, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="stop after N folds (smoke runs)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("concat", help="concatenate per-profile halves, asserting coverage")
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-full", action="store_true",
                   help="exit non-zero unless every frozen cohort pair is present")
    p.set_defaults(func=cmd_concat)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
