"""Reproduce milestone-1 measurements and a conservative priced extraction gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pepstab.data import load_with_splits
from pepstab.boltz_embeddings import json_bytes, sha256_file

OUT = ROOT / "reports/further_research/round2/extractor"


def summarise(verification, profile):
    if verification["status"] != "passed" or profile["status"] != "passed":
        raise ValueError("cannot forecast a passed gate from failed runs")
    v, p = verification["cases"], profile["cases"]
    if len(v) < 8 or len(p) != 64 or len({r["pair_id"] for r in p}) != 64:
        raise ValueError("technical/profile coverage does not meet gate")
    if not all(all(x["allclose"] for x in r["equivalence"].values()) and
               all(x["allclose"] for x in r["repeat_variability"].values()) and
               not any(r["extraction_head_calls"].values()) for r in v):
        raise ValueError("equivalence or no-head gate failed")
    training = load_with_splits().query("split == 'train'")
    n = len(training)
    rate = profile["rate_usd_per_hour"]
    measured_seconds = profile["container_wall_s"]
    # End-to-end profile includes weight hashing/load, preparation, Dataloader,
    # inference, host transfer, compression, and verified cache writes.
    per_case_all_in = measured_seconds / len(p)
    startup = profile["checkpoint_hash_s"] + profile["load_preprocess_to_first_predict_s"]
    amortised_warm = max(0, measured_seconds - startup) / len(p)
    conservative = 1.25 * per_case_all_in
    bytes_per_case = float(np.mean([r["cache"]["compressed_bytes"] for r in p]))
    result = {
        "gate": "engineering_passed_forecast_only_full_cohort_not_launched",
        "training_pairs": n, "training_clusters": int(training.cluster_id.nunique()),
        "verification_cases": len(v), "profile_unique_cases": len(p),
        "all_s_z_bitwise_equal": all(all(x["exact"] for x in r["equivalence"].values()) for r in v),
        "all_repeats_bitwise_equal": all(all(x["exact"] for x in r["repeat_variability"].values()) for r in v),
        "max_equivalence_absolute_error": max(x["max_abs"] for r in v for x in r["equivalence"].values()),
        "median_extract_s": float(np.median([r["extract_s"] for r in p[1:]])),
        "p95_extract_s": float(np.quantile([r["extract_s"] for r in p[1:]], .95)),
        "median_case_wall_s": float(np.median([r["case_wall_s"] for r in p[1:]])),
        "profile_container_wall_s": measured_seconds,
        "startup_preprocessing_hash_load_s": startup,
        "warm_seconds_per_case_including_between_batch_work": amortised_warm,
        "all_in_seconds_per_case_at_64_case_shards": per_case_all_in,
        "conservative_seconds_per_case_25pct_margin": conservative,
        "compressed_bytes_per_case_mean": bytes_per_case,
        "projected_compressed_arrays_gb_decimal": bytes_per_case * n / 1e9,
        "metadata_completion_allowance_bytes_per_case": 65536,
        "projected_total_cache_gb_with_metadata_allowance": (bytes_per_case + 65536) * n / 1e9,
        "volume_list_rate_usd_per_gib_month": .09,
        "projected_cache_monthly_usd_before_free_allowance": (bytes_per_case + 65536) * n / 2**30 * .09,
        "storage_pricing_source": "https://modal.com/pricing (includes 1 TiB/month free Volume storage; remaining workspace allowance not assumed)",
        "rate_usd_per_hour": rate,
        "full_train_estimated_usd_at_64_case_shards": n * per_case_all_in * rate / 3600,
        "full_train_conservative_usd_25pct_margin": n * conservative * rate / 3600,
        "gpu_hours_estimated_at_64_case_shards": n * per_case_all_in / 3600,
        "estimated_wall_hours_8_workers": n * per_case_all_in / 3600 / 8,
        "profile_estimated_usd": profile["runtime_estimate_usd"],
        "verification_estimated_usd": verification["runtime_estimate_usd"],
        "peak_gpu_allocated_gb": max(r["extract_peak_allocated_bytes"] for r in p) / 1e9,
        "peak_worker_host_rss_gb": profile["host_peak_rss_gb"],
        "median_gpu_to_host_s": float(np.median([r["gpu_to_host_s"] for r in p])),
        "median_serialization_reload_s": float(np.median([r["serialize_reload_s"] for r in p])),
        "limitations": [
            "Engineering validation only; no labels or prediction performance were inspected.",
            "The 64-case panel covers one training peptide per selected allele, not all sequence or MSA variability.",
            "The all-in estimate conservatively replicates 64-case startup for every shard; long-lived workers may amortise it.",
            "The 25% cost margin is a planning allowance, not a statistical confidence interval.",
            "Provider billing can lag. Runtime estimates exclude retained-volume storage, image storage, and subsequent training.",
            "Instrumented full-forward timings include capture overhead and are not a production folding benchmark.",
            "Full-cohort extraction requires a separately authorized budget and frozen cohort manifest."
        ]
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    result = summarise(json.loads(args.verification.read_text()), json.loads(args.profile.read_text()))
    result["source_report_hashes"] = {str(p.relative_to(ROOT) if p.is_absolute() else p): sha256_file(p)
                                      for p in (args.verification, args.profile)}
    (OUT / "forecast.json").write_bytes(json_bytes(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
