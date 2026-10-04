"""Build the submission film's data payload from the frozen project artifacts.

Emits `film_data.js` (a single `window.__FILM__ = {...}` assignment) next to
this script, so the film is a static page with no network calls and no build
step beyond running this file.

Every number in the payload is read out of a committed file. Nothing here
recomputes a split, re-scores a model, or writes anywhere under `data/`.

Inputs (all read-only):

  data/rasmussen_et_al_dataset.csv            measured half-lives
  data/splits.csv                             the frozen train/val/test split
  reports/stage6_test_summary.csv             stage 6 test metrics per arm
  reports/stage6_test_paired_ci.csv           paired deltas vs the baseline arm
  reports/figures/figure_data.csv             display names and $/1k rates
  reports/ectodomain-20261004/
      production_verification.json            Boltz-2 production fold audit
  reports/compute_ledger.csv                  the ledger's cost-status label
  reports/stage3c_summary.csv                 elution AUROC per arm/control
  reports/stage3c_provenance.json             elution panel counts

Run from anywhere:

    python3 src/animations/submission-film/build_film_data.py

The output is deterministic: two runs produce byte-identical `film_data.js`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
OUT = HERE / "film_data.js"

# Source paths, relative to the repository root. The provenance block in the
# payload is built from this table, so a path can never drift from its claim.
SRC = {
    "dataset": ["data/rasmussen_et_al_dataset.csv"],
    "splits": ["data/splits.csv"],
    "arms": [
        "reports/stage6_test_summary.csv",
        "reports/stage6_test_paired_ci.csv",
        "reports/figures/figure_data.csv",
    ],
    "compute": [
        "reports/ectodomain-20261004/production_verification.json",
        "reports/compute_ledger.csv",
    ],
    "elution": [
        "reports/stage3c_summary.csv",
        "reports/stage3c_provenance.json",
    ],
}

# The arm every paired contrast in stage 6 is measured against.
BASELINE_ARM = "seq_ensemble_pep_pseudo"

# The statistic the predeclared verdict rule applies to.
HEADLINE_STAT = "median_per_allele_spearman"

# The elution arm the submission ships (the 30-network sequence ensemble).
ELUTION_ARM = "seq_ensemble"

# Controls shown next to the headline elution AUROC, in film order.
# (payload key, arm in stage3c_summary.csv, control in stage3c_summary.csv)
ELUTION_CONTROLS = [
    ("random_scores", "random_control", "none"),
    ("other_allele_ligand_decoys", ELUTION_ARM, "swapped_decoys"),
    ("wrong_hla_pseudosequence", ELUTION_ARM, "wrong_allele_pseudoseq"),
]


def rows(rel: str) -> list[dict]:
    with (ROOT / rel).open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_json(rel: str):
    with (ROOT / rel).open() as fh:
        return json.load(fh)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_dataset() -> dict:
    """Shape of the measured dataset. Read, never recomputed or assumed."""
    n = 0
    floor = 0
    peptides: set[str] = set()
    alleles: set[str] = set()
    lengths: set[int] = set()
    # Index 0 is the assay floor; 1..24 are 0.5 h bins up to 12 h; the last is
    # the > 12 h overflow. The film plots this directly and captions the chart
    # "measured", so the tail has to be counted here rather than modelled in
    # the scene.
    bin_h = 0.5
    n_bins = 24
    hist = [0] * (n_bins + 2)
    for r in rows(SRC["dataset"][0]):
        n += 1
        peptides.add(r["peptide"])
        alleles.add(r["allele"])
        lengths.add(len(r["peptide"]))
        t = fnum(r["thalf_hours"])
        if t == 0.0:
            floor += 1
            hist[0] += 1
        elif t > n_bins * bin_h:
            hist[-1] += 1
        else:
            hist[1 + min(n_bins - 1, int((t - 1e-12) // bin_h))] += 1
    return {
        # Emitted as fractions of all pairs (they sum to 1), which is what the
        # scene plots; raw counts made its axis auto-scale to ~100%.
        "histogram": [c / n for c in hist],
        "hist_bin_hours": bin_h,
        "hist_overflow": True,
        "pairs": n,
        "unique_peptides": len(peptides),
        "unique_alleles": len(alleles),
        # A single value if the assay is one peptide length, else the sorted set.
        "peptide_length": (
            sorted(lengths)[0] if len(lengths) == 1 else sorted(lengths)
        ),
        "floor_rows": floor,
        "floor_fraction": floor / n,
    }


def build_splits() -> dict:
    """Row count per split, straight off the frozen split file."""
    counts: dict[str, int] = {}
    for r in rows(SRC["splits"][0]):
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def build_arms() -> list[dict]:
    """One record per stage 6 test arm, best test Spearman first."""
    # Display name and price per 1,000 predictions, keyed by model id.
    meta = {}
    for r in rows("reports/figures/figure_data.csv"):
        meta[r["file"]] = {
            "name": r["model"],
            "usd_per_1k": fnum(r["usd_per_1k"]),
            "cost_status": r["cost_status"],
        }

    # Paired delta vs the baseline arm, headline statistic only.
    paired = {}
    for r in rows("reports/stage6_test_paired_ci.csv"):
        if r["statistic"] != HEADLINE_STAT or r["baseline"] != BASELINE_ARM:
            continue
        paired[r["model"]] = {
            "delta": fnum(r["delta"]),
            "ci_low": fnum(r["ci_low"]),
            "ci_high": fnum(r["ci_high"]),
            "verdict": r["verdict"],
            "n_boot": int(r["n_boot"]),
        }

    # Resample count, scoped to the headline statistic. It is NOT uniform in the
    # source: the primary metric uses 2,000 but the secondary metrics
    # (differential_concordance, median_precision_at_10) use 500, so a blanket
    # "2,000 resamples" caption would be wrong. Emitted only if every headline
    # contrast agrees, so the film can never state a count the data does not
    # support.
    boots = {p["n_boot"] for p in paired.values()}
    n_boot = boots.pop() if len(boots) == 1 else None

    arms = []
    for r in rows("reports/stage6_test_summary.csv"):
        key = r["model"]
        m = meta.get(key, {})
        arms.append(
            {
                "key": key,
                "name": m.get("name"),
                "is_baseline": key == BASELINE_ARM,
                "spearman": fnum(r["median_per_allele_spearman"]),
                "iqr_low": fnum(r["iqr_low"]),
                "iqr_high": fnum(r["iqr_high"]),
                "mae_log1p": fnum(r["mae_log1p"]),
                "median_precision_at_10": fnum(r["median_precision_at_10"]),
                "n_rows": int(r["n_rows"]),
                "n_alleles": int(r["n_alleles"]),
                # The baseline has no contrast against itself.
                "paired": paired.get(key),
                "usd_per_1k": m.get("usd_per_1k"),
                "cost_status": m.get("cost_status"),
            }
        )
    arms.sort(key=lambda a: -a["spearman"])
    return arms, n_boot


def build_compute() -> dict:
    """The Boltz-2 production fold, as audited after the run."""
    v = load_json(SRC["compute"][0])
    per = v["per_profile"]
    gpu_hours = sum(p["gpu_hours"] for p in per.values())
    usd_from_profiles = round(sum(p["usd"] for p in per.values()), 2)

    # The ledger's label for how the dollar figure was arrived at.
    ledger_status = None
    ledger_usd = None
    for r in rows(SRC["compute"][1]):
        if r["source"].endswith("production_verification.json") and str(
            v["folds_total"]
        ) in r["item"]:
            ledger_status = r["status"]
            ledger_usd = fnum(r["usd"])
            break

    return {
        "model": v["model"],
        "run_id": v["run_id"],
        "folds": v["folds_total"],
        "failures": v["failed_total"],
        "shards": v["shards_total"],
        "gpu_hours": gpu_hours,
        "gpu_type": "A10G",
        "wall_hours": v["wall_hours"],
        "launched": v["launched_bst"],
        "completed": v["completed_bst"],
        "usd_total": v["usd_total"],
        # Independent reconstruction of the total from the two halves; the
        # film can assert these agree rather than trusting one field.
        "usd_from_profiles": usd_from_profiles,
        "usd_ledger": ledger_usd,
        "cost_status": ledger_status,
        "nonzero_returncodes": v["checks"]["nonzero_returncodes"],
        # Profiles are keyed "worker_a"/"worker_b", not by the real Modal
        # profile names. The film is a public submission and the payload ships
        # to the browser, so the account identifiers must not leave the repo —
        # the per-worker counts below are the part that carries meaning.
        "profiles": {
            "worker_%s" % chr(ord("a") + i): {
                "shards": p["shards"],
                "folds": p["folds"],
                "failed": p["failed"],
                "gpu_hours": p["gpu_hours"],
                "usd": p["usd"],
                "median_steady_fold_s": p["median_steady_fold_s"],
            }
            for i, (name, p) in enumerate(sorted(per.items()))
        },
    }


def build_elution() -> dict:
    """Eluted-ligand retrieval, plus the three controls that bound it."""
    table = {
        (r["arm"], r["control"]): r for r in rows(SRC["elution"][0])
    }
    head = table[(ELUTION_ARM, "none")]

    controls = {}
    for key, arm, control in ELUTION_CONTROLS:
        r = table[(arm, control)]
        controls[key] = {
            "arm": arm,
            "control": control,
            "auroc_median": fnum(r["auroc_median"]),
            "auroc_iqr_low": fnum(r["auroc_iqr_low"]),
            "auroc_iqr_high": fnum(r["auroc_iqr_high"]),
            "n_alleles": int(r["n_alleles"]),
        }

    prov = load_json(SRC["elution"][1])
    decoys = sum(a["n_decoys"] for a in prov["per_allele"])

    return {
        "arm": ELUTION_ARM,
        "auroc_median": fnum(head["auroc_median"]),
        "auroc_iqr_low": fnum(head["auroc_iqr_low"]),
        "auroc_iqr_high": fnum(head["auroc_iqr_high"]),
        "auroc_min": fnum(head["auroc_min"]),
        "auroc_max": fnum(head["auroc_max"]),
        "auprc_median": fnum(head["auprc_median"]),
        "auprc_chance": fnum(head["auprc_baseline_median"]),
        "controls": controls,
        "n_alleles": int(head["n_alleles"]),
        "n_peptides": prov["ligands_scored"],
        "n_decoys": decoys,
        "n_rows": prov["rows_scored"],
        "decoy_ratio": prov["decoy_ratio"],
    }


def build() -> dict:
    arms, n_boot = build_arms()
    return {
        "dataset": build_dataset(),
        "splits": build_splits(),
        "arms": arms,
        # Scoped to the headline statistic — see build_arms().
        "bootstrap": {"n_boot": n_boot, "statistic": HEADLINE_STAT},
        "compute": build_compute(),
        "elution": build_elution(),
        "provenance": {k: list(v) for k, v in SRC.items()},
    }


def main() -> None:
    payload = build()
    blob = json.dumps(
        payload, separators=(",", ":"), sort_keys=False, ensure_ascii=True
    )
    assert blob.isascii(), "payload must be pure ASCII"
    OUT.write_text("window.__FILM__=" + blob + ";\n", encoding="ascii")

    d, s, c, e = (
        payload["dataset"],
        payload["splits"],
        payload["compute"],
        payload["elution"],
    )
    top = payload["arms"][0]
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(blob):,} bytes)")
    print(
        f"  dataset: {d['pairs']:,} pairs, {d['unique_peptides']:,} peptides, "
        f"{d['unique_alleles']} alleles, {d['floor_fraction']:.1%} at the floor"
    )
    print("  splits: " + ", ".join(f"{k}={v:,}" for k, v in s.items()))
    print(f"  top arm: {top['key']} at rho={top['spearman']}")
    print(
        f"  compute: {c['folds']:,} folds, {c['gpu_hours']} GPU-h, "
        f"{c['wall_hours']} h wall, {c['failures']} failures, "
        f"${c['usd_total']}"
    )
    print(
        f"  elution: AUROC {e['auroc_median']} over {e['n_alleles']} alleles, "
        f"{e['n_peptides']:,} ligands vs {e['n_decoys']:,} decoys"
    )


if __name__ == "__main__":
    main()
