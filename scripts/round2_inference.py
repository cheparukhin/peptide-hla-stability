#!/usr/bin/env python3
"""Explicit, bounded development refit and cached-head inference CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from threadpoolctl import threadpool_limits

from pepstab.boltz_embeddings import json_bytes, sha256_file
from pepstab.round2_inference import (
    array_digest, candidate_config, digest, fit_count, fit_head, load_feature_batch, load_head,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="show configuration and fit count; no training")
    plan.add_argument("--candidate", required=True)
    fit = sub.add_parser("fit", help="export one locked development recipe; original train rows only")
    fit.add_argument("--candidate", required=True)
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--training-csv", type=Path, required=True)
    fit.add_argument("--authorization", type=Path, required=True)
    fit.add_argument("--max-fits", type=int, required=True)
    fit.add_argument("--out", type=Path, required=True)
    predict = sub.add_parser("predict", help="predict from a verified cached feature bundle")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--features", type=Path, required=True)
    predict.add_argument("--pair-ids", type=Path, help="optional one pair_id per line, in desired output order")
    predict.add_argument("--out", type=Path, required=True)
    inspect = sub.add_parser("inspect", help="verify package hashes and print metadata")
    inspect.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with threadpool_limits(limits=2):
            if args.command == "plan":
                print(json.dumps({"candidate": candidate_config(args.candidate), "fits": fit_count(args.candidate),
                                  "scope": "development-refit", "cloud_calls": 0}, sort_keys=True))
            elif args.command == "inspect":
                print(json.dumps(load_head(args.model).metadata, indent=2, sort_keys=True))
            elif args.command == "predict":
                model = load_head(args.model)
                batch = load_feature_batch(args.features, model.metadata["candidate"]["representation"],
                                           expected_contract=model.metadata["contract"])
                if args.pair_ids:
                    batch = batch.select(args.pair_ids.read_text().splitlines())
                output = model.predict(batch)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                # Output files are immutable as well; explicit new paths keep provenance clear.
                with args.out.open("x") as handle:
                    output.to_csv(handle, index=False)
                print(json.dumps({"rows": len(output), "output": str(args.out), "scope": model.metadata["scope"]}))
            else:
                candidate = candidate_config(args.candidate)
                count = fit_count(args.candidate)
                if args.max_fits < count:
                    raise ValueError(f"recipe requires {count} fits, exceeds max_fits={args.max_fits}")
                if args.out.exists():
                    raise ValueError("model output already exists")
                training = pd.read_csv(args.training_csv, dtype={"pair_id": str, "cluster_id": str})
                # A CSV's self-declared split is insufficient: verify membership,
                # cluster identity and targets against the frozen canonical data.
                from pepstab.data import RAW_CSV, SPLITS_CSV, load_with_splits
                import numpy as np
                canonical = load_with_splits().query("split == 'train'").copy()
                canonical["pair_id"] = canonical.pair_id.astype(str)
                canonical = canonical.set_index("pair_id")
                if (not {"pair_id", "cluster_id", "split", "y_log1p"} <= set(training)
                        or training.pair_id.duplicated().any()
                        or not training["split"].eq("train").all()
                        or set(training.pair_id) - set(canonical.index)):
                    raise ValueError("training CSV must contain only canonical original train rows")
                expected = canonical.loc[training.pair_id]
                if (expected.cluster_id.astype(str).tolist() != training.cluster_id.astype(str).tolist()
                        or not np.allclose(expected.y_log1p, training.y_log1p, rtol=0, atol=1e-12)):
                    raise ValueError("training labels/clusters differ from canonical original train rows")
                batch = load_feature_batch(args.features, candidate["representation"])
                selected = batch.select(training.pair_id)
                authorization = json.loads(args.authorization.read_text())
                prefit = {"authorization": authorization, "authorization_sha256": sha256_file(args.authorization),
                          "candidate": candidate, "fits": count, "max_fits": args.max_fits,
                          "training_csv_sha256": sha256_file(args.training_csv),
                          "canonical_source_sha256": sha256_file(RAW_CSV), "frozen_splits_sha256": sha256_file(SPLITS_CSV),
                          "training_feature_sha256": array_digest(selected.values),
                          "training_pair_ids_sha256": digest(selected.pair_ids),
                          "contract_sha256": digest(batch.contract),
                          "inference_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "pepstab/round2_inference.py"),
                          "numerical_threads": 2, "scope": "development-refit"}
                args.out.parent.mkdir(parents=True, exist_ok=True)
                record = args.out.parent / (args.out.name + ".authorization.json")
                with record.open("xb") as handle:
                    handle.write(json_bytes(prefit))
                model = fit_head(batch, training, args.candidate, max_fits=args.max_fits, authorization=prefit)
                model.save(args.out)
                # Verify every parameter and exact serialization round trip before success.
                import numpy as np
                restored = load_head(args.out)
                np.testing.assert_array_equal(model.predict(selected).prediction_y_log1p,
                                              restored.predict(selected).prediction_y_log1p)
                print(json.dumps({"model": str(args.out), "fits": count, "scope": "development-refit",
                                  "roundtrip_prediction_exact": True}))
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"round2 inference: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
