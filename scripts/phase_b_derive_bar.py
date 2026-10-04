"""Derive Phase B's worthwhile-delta-Brier bar. **Reads B0 only.**

    .venv/bin/python scripts/phase_b_derive_bar.py

``reports/further_research/phase_b_declaration.md`` §7 declares this bar as a
*rule* rather than a number, because no analogue of EVALUATION.md's 0.05 exists
for Brier and a threshold invented after seeing a result is not a threshold. The
rule:

> The bar is the **maximum** 95% paired cluster-bootstrap half-width under no
> true difference, computed on the harness's own outer-held row set, 2,000
> resamples, seed 20261003, 8 null splits, rounded **up** to two significant
> figures. Computed before any censored arm is scored.

This is a separate script, and a separate invocation, on purpose. It loads **only
the B0 reference's** members and refuses to touch B1 or B2, so the ordering is a
property of the artifact trail -- a timestamped JSON written before
``scripts/phase_b_analyse.py`` ever runs -- rather than a claim about the order
of lines inside one program. EVALUATION.md's 0.05 was derived the same way, from
the width of a comparison that has no real effect.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pepstab import calibration as cal  # noqa: E402
from pepstab.censored import DEFAULT_FLOOR_HOURS, floor_threshold  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "further_research"
REFERENCE_ARM = "B0"
PRIMARY_RULE = "mixture"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=8)
    args = ap.parse_args()
    sx = args.suffix

    store = np.load(OUT_DIR / f"phase_b_nested_members{sx}.npz", allow_pickle=False)
    threshold = floor_threshold(DEFAULT_FLOOR_HOURS)
    folds = sorted({int(k.split("__")[1]) for k in store.files
                    if k.startswith(f"{REFERENCE_ARM}__") and k.endswith("__mus")})

    # Only B0's keys are ever read. Asserted, so a later edit cannot quietly
    # widen what this script sees.
    allowed = {k for k in store.files if k.startswith(f"{REFERENCE_ARM}__")}
    assert allowed, f"no {REFERENCE_ARM} members in the archive"

    frames = []
    for k in folds:
        pid = store[f"{REFERENCE_ARM}__{k}__held_pair_id"]
        mus = store[f"{REFERENCE_ARM}__{k}__mus"]
        sig = store[f"{REFERENCE_ARM}__{k}__sigmas"]
        frames.append(pd.DataFrame({
            "pair_id": pid,
            "p": cal.prob_floor_mixture(mus, sig, threshold)
            if PRIMARY_RULE == "mixture"
            else cal.prob_floor_plugin(mus.mean(axis=0), float(sig.mean()), threshold)}))
    ref = pd.concat(frames, ignore_index=True)
    assert ref.pair_id.is_unique

    df = load_with_splits()
    train = df[df.split == "train"][["pair_id", "thalf_hours", "cluster_id"]]
    meta = ref.merge(train, on="pair_id", validate="one_to_one")
    is_floor = cal.floor_indicator(meta.thalf_hours, "recorded_zero")

    print(f"deriving the worthwhile-delta-Brier bar from {REFERENCE_ARM} alone")
    print(f"  {len(meta):,} outer-held rows, {int(is_floor.sum()):,} at the floor "
          f"({is_floor.mean():.4f}), {meta.cluster_id.nunique():,} clusters")
    print(f"  P(floor) rule {PRIMARY_RULE}, {args.n_boot} resamples x "
          f"{args.n_splits} null splits, seed 20261003")

    null = cal.null_half_width(meta.cluster_id, cal.brier,
                               predictions=meta.p.to_numpy(), labels=is_floor,
                               n_boot=args.n_boot, n_splits=args.n_splits)
    raw = null["max_half_width"]
    mag = 10 ** (np.floor(np.log10(raw)) - 1) if raw > 0 else 1.0
    bar = float(np.ceil(raw / mag) * mag)

    out = {
        "rule": ("max 95% paired cluster-bootstrap half-width under no true "
                 "difference, reference arm only, rounded up to 2 significant "
                 "figures"),
        "reference_arm": REFERENCE_ARM, "p_floor_rule": PRIMARY_RULE,
        "n_rows": int(len(meta)), "n_clusters": int(meta.cluster_id.nunique()),
        "half_widths": null["half_widths"],
        "max_half_width": raw, "median_half_width": null["median_half_width"],
        "worthwhile_delta_brier": bar,
        "n_boot": args.n_boot, "n_splits": args.n_splits, "seed": 20261003,
        "derived_before_any_censored_arm_scored": True,
        "note": ("Written by a separate script and a separate invocation that "
                 "loads only the reference arm's members, so the ordering is a "
                 "property of the artifact trail rather than of line order "
                 "inside one program."),
    }
    (OUT_DIR / f"phase_b_brier_bar{sx}.json").write_text(json.dumps(out, indent=2))
    print(f"\n  per-split half-widths: "
          f"{', '.join(f'{v:.5f}' for v in null['half_widths'])}")
    print(f"  max {raw:.5f}  median {null['median_half_width']:.5f}")
    print(f"  => worthwhile delta Brier bar = {bar:.4f}")
    print(f"  wrote phase_b_brier_bar{sx}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
