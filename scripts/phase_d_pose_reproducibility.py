#!/usr/bin/env python
"""Phase D step 3: multi-pose reproducibility from **existing records only**.

No new poses. Reads the three-seed pilot (`reports/stage4c5_pilot_features.csv`,
2 models x 3 arms x 3 seeds x 5 complexes) and the production extraction
(`reports/stage4c5_features_production.csv`, seed 0 / arm B, 28,166 pairs).

What this measures, and what it does not
----------------------------------------
Across-seed spread in a predicted structure's features is **prediction
ambiguity and sampler behaviour**. It is not a thermodynamic ensemble, not a
conformational population, and not an unbinding rate. Nothing energetic or
kinetic is inferred from it here.

What it *is* good for is reliability: how much of a feature's usable spread
survives being measured once. A feature measured with reliability R correlates
with any outcome at sqrt(R) times its true correlation (classical attenuation),
so R answers "sample sufficiency" operationally -- how many seeds would have to
be averaged to recover a given fraction of the signal.

Three variance terms, and why the third is the one that matters
---------------------------------------------------------------
``seed``  within-complex spread across the 3 pilot seeds (pooled over the 5
          pilot complexes). The measurement noise of a single pose.
``between_complex``  spread across complexes. This is what stage 4c5 reported,
          and it mixes between-allele with within-allele variation.
``within_allele``  spread across peptides *inside* one allele, pooled over the
          75 production alleles. The primary metric is median per-allele
          Spearman, which ranks peptides inside an allele and is blind to any
          between-allele offset -- so this is the only numerator that bears on
          the metric Phase D is judged by. A feature can have a healthy
          between-complex ratio and still be near-useless for ranking if its
          spread is almost all between alleles.

The pilot's 5 complexes span 5 alleles, so a within-allele seed term cannot be
estimated from it; the seed variance is assumed transferable from the pilot to
production. That assumption is stated in the report, not hidden here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.stage5_structural_arm import (  # noqa: E402
    CONFIDENCE_PREFIXES, GEOMETRY_PREFIXES, NON_FEATURE,
)

OUT = REPO / "reports" / "further_research"
PILOT = REPO / "reports" / "stage4c5_pilot_features.csv"
PRODUCTION = REPO / "reports" / "stage4c5_features_production.csv"

#: Production is seed 0, arm B, Boltz-2 only. Reliability is reported for that
#: configuration; the other pilot arms are carried as context, not as the
#: thing production depends on.
PRODUCTION_MODEL = "boltz2"
PRODUCTION_ARM = "B"

#: Fractions of the asymptotic (noise-free) correlation to solve seeds for.
TARGET_FRACTIONS = (0.95, 0.99, 0.999)


def feature_names(frame: pd.DataFrame) -> list[str]:
    """Geometry and confidence columns, using the structural arm's own groups."""
    out = []
    for col in frame.columns:
        if col in NON_FEATURE or col.endswith("_sha256") or col.startswith("msa_"):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if col.startswith(GEOMETRY_PREFIXES + CONFIDENCE_PREFIXES):
            out.append(col)
    return out


def group_of(col: str) -> str:
    return "geometry" if col.startswith(GEOMETRY_PREFIXES) else "confidence"


def pooled_within_variance(frame: pd.DataFrame, by: str, cols: list[str]
                           ) -> tuple[pd.Series, int]:
    """Pooled within-group variance, and its degrees of freedom.

    Sum of squared deviations from each group's own mean, divided by
    ``n_rows - n_groups``. Pooling rather than averaging per-group variances
    weights each group by how much it actually contributes, which matters here
    because production alleles range from 5 to 727 rows.
    """
    sub = frame[[by] + cols]
    centred = sub[cols] - sub.groupby(by)[cols].transform("mean")
    ss = (centred ** 2).sum(axis=0)
    df = len(sub) - sub[by].nunique()
    return ss / df, int(df)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    pilot = pd.read_csv(PILOT)
    cols = feature_names(pilot)
    print(f"pilot: {len(pilot)} folds, {len(cols)} candidate feature columns")

    # ---- seed variance, from the pilot -----------------------------------
    rows = []
    for (model, arm), grp in pilot.groupby(["model", "arm"]):
        grp = grp.copy()
        n_seeds = grp.seed.nunique()
        seed_var, seed_df = pooled_within_variance(grp, "complex_id", cols)
        complex_means = grp.groupby("complex_id")[cols].mean()
        between_var = complex_means.var(ddof=1)
        for c in cols:
            rows.append({
                "model": model, "arm": arm, "feature": c, "group": group_of(c),
                "n_complexes": int(grp.complex_id.nunique()), "n_seeds": n_seeds,
                "seed_var": float(seed_var[c]), "seed_df": seed_df,
                "between_complex_var_of_means": float(between_var[c]),
            })
    pilot_var = pd.DataFrame(rows)
    pilot_var.to_csv(OUT / "phase_d_pose_seed_variance.csv", index=False)
    print(f"wrote {(OUT / 'phase_d_pose_seed_variance.csv').relative_to(REPO)}")

    # ---- production spread: total, between-allele, within-allele ---------
    usecols = ["allele", "peptide", "model", "arm", "seed", "status"] + cols
    prod = pd.read_csv(PRODUCTION, usecols=lambda c: c in set(usecols))
    prod = prod[(prod.model == PRODUCTION_MODEL) & (prod.arm == PRODUCTION_ARM)]
    print(f"production: {len(prod)} pairs, {prod.allele.nunique()} alleles, "
          f"seeds {sorted(prod.seed.unique())}")

    total_var = prod[cols].var(ddof=1)
    wa_var, wa_df = pooled_within_variance(prod, "allele", cols)
    allele_means = prod.groupby("allele")[cols].mean()
    ba_var = allele_means.var(ddof=1)

    # ---- reliability and sample sufficiency ------------------------------
    seed = pilot_var[(pilot_var.model == PRODUCTION_MODEL)
                     & (pilot_var.arm == PRODUCTION_ARM)].set_index("feature")

    out = []
    for c in cols:
        if c not in seed.index:
            continue
        sv = seed.loc[c, "seed_var"]
        for numerator, nv in (("between_complex", float(total_var[c])),
                              ("within_allele", float(wa_var[c]))):
            # Reliability of an n-seed average, and the attenuation sqrt(R)
            # it implies for that feature's correlation with any outcome.
            def rel(n: int, nv=nv, sv=sv) -> float:
                denom = nv + sv / n
                return float(nv / denom) if denom > 0 else float("nan")

            r1 = rel(1)
            row = {
                "feature": c, "group": group_of(c), "numerator": numerator,
                "signal_var": nv, "seed_var": float(sv),
                "sd_signal": float(np.sqrt(nv)), "sd_seed": float(np.sqrt(sv)),
                "ratio_sd_signal_over_seed": (float(np.sqrt(nv / sv))
                                              if sv > 0 else float("inf")),
                "reliability_1_seed": r1,
                "attenuation_1_seed": float(np.sqrt(r1)),
                "reliability_3_seed": rel(3),
                "attenuation_3_seed": float(np.sqrt(rel(3))),
            }
            # Seeds needed so that sqrt(R_n) >= f. Solve analytically:
            # R_n = nv/(nv + sv/n) >= f^2  <=>  n >= (sv/nv) * f^2/(1-f^2).
            for f in TARGET_FRACTIONS:
                if nv <= 0:
                    need = float("nan")
                elif sv == 0:
                    need = 1.0
                else:
                    need = (sv / nv) * (f ** 2) / (1 - f ** 2)
                row[f"seeds_for_{int(f * 1000)}permille"] = (
                    int(max(1, np.ceil(need))) if np.isfinite(need) else need)
            out.append(row)
    rel_frame = pd.DataFrame(out)
    rel_frame.to_csv(OUT / "phase_d_pose_reliability.csv", index=False)
    print(f"wrote {(OUT / 'phase_d_pose_reliability.csv').relative_to(REPO)}")

    # ---- variance partition, for the near-constant-column diagnostic -----
    part = pd.DataFrame({
        "feature": cols, "group": [group_of(c) for c in cols],
        "total_var": total_var[cols].to_numpy(),
        "within_allele_var": wa_var[cols].to_numpy(),
        "between_allele_var_of_means": ba_var[cols].to_numpy(),
    })
    part["within_allele_share"] = part.within_allele_var / part.total_var
    part["sd_total"] = np.sqrt(part.total_var)
    part["sd_within_allele"] = np.sqrt(part.within_allele_var)
    part = part.sort_values("within_allele_share")
    part.to_csv(OUT / "phase_d_feature_variance_partition.csv", index=False)
    print(f"wrote {(OUT / 'phase_d_feature_variance_partition.csv').relative_to(REPO)}")

    # ---- console summary --------------------------------------------------
    print(f"\nproduction within-allele df = {wa_df:,}")
    for numerator in ("between_complex", "within_allele"):
        s = rel_frame[rel_frame.numerator == numerator]
        print(f"\n--- numerator: {numerator} ({len(s)} features) ---")
        print(f"  median sd ratio (signal/seed)   {s.ratio_sd_signal_over_seed.median():.2f}")
        print(f"  features with ratio < 1          {int((s.ratio_sd_signal_over_seed < 1).sum())} of {len(s)}")
        print(f"  median 1-seed reliability        {s.reliability_1_seed.median():.4f}")
        print(f"  median 1-seed attenuation        {s.attenuation_1_seed.median():.4f}")
        print(f"  worst 1-seed attenuation         {s.attenuation_1_seed.min():.4f}"
              f"  ({s.loc[s.attenuation_1_seed.idxmin(), 'feature']})")
        print(f"  median seeds for 99% of signal   {s.seeds_for_990permille.median():.0f}")
        print(f"  max    seeds for 99% of signal   {s.seeds_for_990permille.max():.0f}")

    print("\nlowest within-allele variance share (near-constant within allele):")
    print(part.head(8)[["feature", "group", "sd_total", "sd_within_allele",
                        "within_allele_share"]].to_string(index=False))
    print("\nhighest within-allele variance share:")
    print(part.tail(5)[["feature", "group", "sd_total", "sd_within_allele",
                        "within_allele_share"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
