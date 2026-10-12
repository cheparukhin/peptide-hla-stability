"""Freeze, execute and resume corrected sequence controls; original train only.

Run ``--freeze`` before ``--fold N``. Each ensemble is checkpointed beneath
``runs/round2_sequence``; completed folds and full ancestry are durable reports.
No original split, legacy prediction, validation or test output is modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation, nested
from pepstab.data import load_with_splits
from pepstab.round2_nested import (Ancestry, apply_affine, clean_crossfit,
    config_digest, positive_affine, select, unlabelled)
from pepstab.round2_sequence import EnsemblePredictor, FAMILY_ORDER, candidates, fit_graph

OUT = ROOT / "reports/further_research/round2/sequence"


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def frame():
    return load_with_splits().query("split == 'train'").sort_values("pair_id").reset_index(drop=True)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_hashes(directory, *, repeats=False):
    """Bind completed diagnostics to the exact files written before the marker."""
    files = ({"repeat_prediction_sha256": "repeat_predictions.csv",
              "ancestry_sha256": "ancestry_with_repeats.json",
              "member_table_sha256": "repeat_members.csv",
              "primary_completion_sha256": "completed.json"}
             if repeats else {
                 "prediction_sha256": "predictions.csv", "seed_sha256": "seed_predictions.csv",
                 "ancestry_sha256": "ancestry.json", "member_table_sha256": "members.csv",
                 "selections_sha256": "selections.json", "selector_sha256": "selector.json",
                 "ranker_map_sha256": "ranker_map.json",
                 "ranker_mapping_inputs_sha256": "ranker_mapping_inputs.csv"})
    return {key: hash_file(Path(directory)/name) for key, name in files.items()}


def repeat_member_counts(members, fold, repeats):
    """Count actual locked-repeat members by arm, rejecting partial repeat sets."""
    counts = []
    for repeat in range(1, repeats+1):
        for arm, expected in (("mlp_log", 30), ("tree_log", 6), ("pairlogit", 6)):
            context = f"outer/{fold}/{arm}/repeat/{repeat}"
            n = sum(m["context"] == context for m in members)
            if n != expected:
                raise ValueError(f"{context}: expected {expected} members, observed {n}")
            counts.append({"repeat": repeat, "arm": arm, "members": n})
    if len(members) != sum(row["members"] for row in counts):
        raise ValueError("repeat member table contains unexpected contexts")
    return counts


def specification(train):
    fold = nested.outer_folds(train)
    return {"schema": "round2_sequence.1", "canonical_plan": "reports/NEXT_RESEARCH_PLAN.md",
            "population": "original training partition only; development evidence",
            "source_sha256": hash_file(ROOT / "data/rasmussen_et_al_dataset.csv"),
            "splits_sha256": hash_file(ROOT / "data/splits.csv"),
            "data_hash": hashlib.sha256(pd.util.hash_pandas_object(train, index=False).values.tobytes()).hexdigest(),
            "fold_hash": hashlib.sha256(fold.astype("<i8").tobytes()).hexdigest(),
            "n_rows": len(train), "n_clusters": int(train.cluster_id.nunique()),
            "candidate_grids": candidates(), "family_order": FAMILY_ORDER,
            "selection": "per-outer-fold, three inner cluster folds; first in frozen parsimony order within one fold-SE of best mean inner panel median",
            "family_selector": "same parsimony rule on selected candidate's inner scores; original-only is first and always available",
            "features": {"mlp": ["position-preserving onehot", "position-preserving BLOSUM"],
                         "tree": ["position-preserving onehot"], "input": "peptide+pseudosequence; no allele ID"},
            "centering": "gradient-fitting rows only; shrink toward shared fitting mean; unseen alleles use that shared mean",
            "early_stopping": "further cluster folds wholly within each candidate fitting partition; MLP MSE, trees RMSE/PairLogit; predictions never on stopping rows",
            "ranker_scale": "positive affine slope>=1e-8 fit to clean nested-selected crossfit scores inside outer fitting partition; raw fold scores reported separately",
            "ranker_pair_recipe": "all untied within-allele pairs, equal total weight per contributing allele, weights rescaled to mean 1",
            "mapping_training_size_difference": "crossfit base models fit 2/3 of outer-fitting rows before their stopping splits, whereas outer base models use the complete outer fitting partition before stopping; same 6-member recipe",
            "inner_final_recipe_difference": "inner MLP screening uses 3 stopping folds x 2 encodings x seeds0,1=12; final original and centred MLP use original 5 x2 x seeds0,1,2=30; trees always 3 x1 x seeds0,1=6",
            "fit_graph": fit_graph(), "min_rows_primary": 20, "min_rows_sensitivity": 50,
            "primary_statistic": "pooled outer-held median per-allele Spearman; fixed evaluation tie/constant rules",
            "execution": {"device": "local CPU", "threads_per_worker": 2, "dtype_mlp": "float32", "cloud_spend_usd": 0},
            "comparisons": "every component and inner-selected sequence vs original; no Boltz comparison until extraction passes",
            "repeats": "two additional identical-seed refits of original, A1 and A4 after initial fit; report paired seed0/1 diagnostics separately",
            "runtime_ceiling": "local only; no automatic cloud launch or grid expansion"}


def freeze(out):
    train = frame()
    spec = specification(train)
    digest = config_digest(spec)
    path = out / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["protocol_hash"] != digest:
            raise ValueError("frozen protocol differs; require a documented version amendment")
        return existing
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_hash": digest, "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_sha256": {str(p.relative_to(ROOT)): hash_file(p) for p in (
                    ROOT/"pepstab/round2_nested.py", ROOT/"pepstab/round2_sequence.py", Path(__file__))},
                "specification": spec, "hardware": platform.platform(),
                "python": platform.python_version(), "threadpools": threadpool_info()}
    atomic_json(path, manifest)
    train[["pair_id", "cluster_id", "allele"]].assign(
        outer_fold=nested.outer_folds(train)).to_csv(out/"fold_manifest.csv", index=False)
    for bar in (20, 50):
        panel = evaluation.eligible_alleles(train.allele, train.y_log1p, min_rows=bar)
        atomic_json(out/f"panel_{bar}.json", {"min_rows": bar, "alleles": panel,
                    "covered_rows": int(train.allele.isin(panel).sum()), "n_rows": len(train)})
    return manifest


def select_family(selections):
    rows = []
    for family in FAMILY_ORDER:
        choice = selections[family]
        selected_score = next(x for x in choice["scores"] if x["candidate"] == choice["candidate"]["id"])
        rows.append({"family": family, **selected_score})
    best = max(rows, key=lambda x: x["mean"])
    chosen = next(r for r in rows if r["mean"] >= best["mean"] - best["se"])
    return {"family": chosen["family"], "scores": rows,
            "rule": "first frozen family within 1 fold-SE of best inner mean; no outer outcomes"}


def run_fold(out, fold, *, repeats=0, ranker_backend=None, ranker_backend_name="local"):
    train = frame()
    protocol = json.loads((out/"protocol.json").read_text())
    if protocol["protocol_hash"] != config_digest(specification(train)):
        raise ValueError("data or frozen specification changed")
    fold_path = out/f"fold_{fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    if (fold_path/"completed.json").exists() and not repeats:
        print(f"fold {fold}: already complete", flush=True)
        return
    atomic_json(fold_path/"runtime_manifest.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol["protocol_hash"], "platform": platform.platform(),
        "python": platform.python_version(), "threadpools": threadpool_info(),
        "thread_environment": {k: os.environ.get(k) for k in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "code_sha256": {str(p.relative_to(ROOT)): hash_file(p) for p in (
            ROOT/"pepstab/round2_nested.py", ROOT/"pepstab/round2_sequence.py", Path(__file__))},
        "ranker_backend": ranker_backend_name,
        "cloud_spend_usd": 0 if ranker_backend is None else "see modal ledger"})
    started = time.perf_counter()
    outer = nested.outer_folds(train)
    fit, held = train[outer != fold], train[outer == fold]
    ancestry = Ancestry()
    progress_time = [0.]
    def checkpoint_member(member):
        with (fold_path/"members.jsonl").open("a") as handle:
            handle.write(json.dumps(member, allow_nan=False) + "\n")
        now = time.perf_counter()
        if now-progress_time[0] > 45:
            progress_time[0] = now
            print(f"fold {fold}: {member['context']}, {member['encoding']}, seed{member['seed']}, {member['fit_seconds']:.1f}s", flush=True)
    predictor = EnsemblePredictor(train, ancestry, callback=checkpoint_member,
        cache_dir=ROOT/"runs/round2_sequence"/protocol["protocol_hash"]/f"fold_{fold}",
        ranker_backend=ranker_backend, ranker_backend_name=ranker_backend_name)
    selections, tables, seed_tables = {}, [], []
    mappings = {}
    for family in FAMILY_ORDER:
        context = f"outer/{fold}/{family}"
        choice = select(fit, candidates()[family], predictor, ancestry=ancestry, context=context)
        selections[family] = asdict(choice)
        raw = predictor.predict(fit, unlabelled(held), choice.candidate, context+"/outer_refit", final=True)
        by_seed = predictor.last_by_seed.copy()
        ancestry.add(role="outer_selected_prediction", fit=fit, held=held, selection=fit,
                     candidate=choice.candidate["id"], context=context)
        prediction = raw.copy()
        if family == "pairlogit":
            clean, cross_choices = clean_crossfit(fit, candidates()[family], predictor,
                                                 ancestry=ancestry, context=context+"/mapping")
            mapping = positive_affine(clean, fit.y_log1p)
            mappings[family] = {**mapping, "crossfit_selections": cross_choices}
            prediction = apply_affine(raw, mapping)
            fit[["pair_id", "cluster_id"]].assign(y_pred_raw=clean).to_csv(
                fold_path/"ranker_mapping_inputs.csv", index=False)
            atomic_json(fold_path/"ranker_map.json", mappings[family])
        table = held[["pair_id", "cluster_id", "allele"]].copy()
        table["outer_fold"], table["arm"] = fold, family
        table["y_pred"], table["y_pred_raw"] = prediction, raw
        table["selected_candidate"] = choice.candidate["id"]
        tables.append(table)
        for seed, values in by_seed.items():
            seed_table = table.copy()
            seed_table["seed"] = seed
            seed_table["y_pred_raw"] = values
            seed_table["y_pred"] = apply_affine(values, mappings[family]) if family in mappings else values
            seed_tables.append(seed_table)
        atomic_json(fold_path/"selections.json", selections)
        pd.concat(tables).to_csv(fold_path/"predictions.partial.csv", index=False)
        print(f"fold {fold}: {family} complete; selected {choice.candidate['id']}", flush=True)
    selector = select_family(selections)
    selected = next(t for t in tables if t.arm.iloc[0] == selector["family"]).copy()
    selected["arm"] = "sequence_selector"
    tables.append(selected)
    atomic_json(fold_path/"selector.json", selector)
    pd.concat(tables).to_csv(fold_path/"predictions.csv", index=False)
    pd.concat(seed_tables).to_csv(fold_path/"seed_predictions.csv", index=False)
    audit = ancestry.audit(train, held)
    atomic_json(fold_path/"ancestry.json", ancestry.as_dict())
    pd.DataFrame([{k:v for k,v in member.items() if k != "centering_means"}
                  for member in predictor.members]).to_csv(fold_path/"members.csv", index=False)
    initial_member_count = len(predictor.members)
    expected_members = sum(fit_graph()["per_outer"].values())
    if initial_member_count != expected_members:
        raise ValueError(f"incomplete primary fitting graph: {initial_member_count} vs {expected_members}")
    atomic_json(fold_path/"completed.json", {"schema": "round2_sequence_completion.2",
        "protocol_hash": protocol["protocol_hash"], "outer_fold": fold,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "audit": audit,
        "n_rows": len(held), "members": len(predictor.members),
        "sum_member_fit_seconds": sum(m["fit_seconds"] for m in predictor.members),
        "invocation_wall_seconds": time.perf_counter()-started,
        "cloud_cost_usd": 0 if ranker_backend is None else "see modal ledger",
        **completion_hashes(fold_path)})
    if repeats:
        repeat_tables = []
        for repeat in range(1, repeats+1):
            for family in ("mlp_log", "tree_log", "pairlogit"):
                raw = predictor.predict(fit, unlabelled(held), selections[family]["candidate"],
                                        f"outer/{fold}/{family}/repeat/{repeat}", final=True)
                table = held[["pair_id", "cluster_id", "allele"]].copy()
                table["outer_fold"], table["arm"], table["repeat"] = fold, family, repeat
                table["y_pred_raw"] = raw
                table["y_pred"] = apply_affine(raw, mappings[family]) if family in mappings else raw
                repeat_tables.append(table)
                pd.concat(repeat_tables).to_csv(fold_path/"repeat_predictions.csv", index=False)
        repeat_members = predictor.members[initial_member_count:]
        counts = repeat_member_counts(repeat_members, fold, repeats)
        pd.DataFrame([{k:v for k,v in member.items() if k != "centering_means"}
                      for member in repeat_members]).to_csv(fold_path/"repeat_members.csv", index=False)
        atomic_json(fold_path/"ancestry_with_repeats.json", ancestry.as_dict())
        atomic_json(fold_path/"repeats_completed.json", {
            "schema": "round2_sequence_repeats_completion.2",
            "protocol_hash": protocol["protocol_hash"], "outer_fold": fold,
            "repeats": repeats, "scope": "locked member refits; selected configurations and ranker calibration mapping held fixed",
            "additional_members": len(repeat_members), "members_per_repeat": 42,
            "member_counts": counts,
            "ancestry_audit": ancestry.audit(train, held), **completion_hashes(fold_path, repeats=True)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--repeats", type=int, default=0, choices=(0, 2))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        if args.freeze:
            result = freeze(args.out_dir)
            print(json.dumps({"protocol_hash": result["protocol_hash"], "fit_graph": fit_graph()}, indent=2))
        if args.fold is not None:
            if not (args.out_dir/"protocol.json").exists():
                raise ValueError("freeze the protocol before fitting")
            run_fold(args.out_dir, args.fold, repeats=args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
