"""Report completed, identity-matched round-two procedures without selecting arms.

No partial-fold outcomes are loaded. All intervals share 2,000 whole-peptide-
cluster draws and condition on the saved fitted predictions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation as ev, nested
from pepstab.boltz_embeddings import _atomic_bytes, sha256_file
from pepstab.data import load_with_splits
from pepstab.round2_nested import Ancestry, align_predictions, config_digest
from pepstab.round2_reporting import compare_predictions
from pepstab.round2_sequence import FAMILY_ORDER

OUT = ROOT / "reports/further_research/round2/results"
SEQUENCE = ROOT / "reports/further_research/round2/sequence"
EMBEDDING = ROOT / "reports/further_research/round2/embedding"
N_BOOT = 2000


def atomic_json(path, item):
    _atomic_bytes(Path(path), (json.dumps(item, indent=2, allow_nan=False) + "\n").encode())


def atomic_csv(path, table):
    _atomic_bytes(Path(path), table.to_csv(index=False).encode())


def training_frame():
    return load_with_splits().query("split == 'train'").sort_values("pair_id").reset_index(drop=True)


def population_contract(train):
    return {"source_sha256": sha256_file(ROOT / "data/rasmussen_et_al_dataset.csv"),
            "splits_sha256": sha256_file(ROOT / "data/splits.csv"),
            "data_hash": hashlib.sha256(pd.util.hash_pandas_object(train, index=False).values.tobytes()).hexdigest(),
            "fold_hash": hashlib.sha256(nested.outer_folds(train).astype("<i8").tobytes()).hexdigest(),
            "n_rows": len(train)}


def validate_identity(held, table, fold):
    """Full ordered identity coverage; never drop rows to manufacture a match."""
    required = {"pair_id", "cluster_id", "allele", "outer_fold", "arm", "y_pred"}
    if not required.issubset(table):
        raise ValueError("prediction metadata incomplete")
    if not table.outer_fold.eq(fold).all():
        raise ValueError("prediction outer-fold identity mismatch")
    prediction = align_predictions(held, table)
    metadata = held[["pair_id", "allele"]].merge(table[["pair_id", "allele"]], on="pair_id",
                                                validate="one_to_one", suffixes=("", "_pred"), sort=False)
    if not metadata.allele.eq(metadata.allele_pred).all():
        raise ValueError("prediction allele identity mismatch")
    return prediction


def _read_json(path, sources):
    sources[str(Path(path).resolve())] = sha256_file(path)
    return json.loads(Path(path).read_text())


def _read_csv(path, sources):
    sources[str(Path(path).resolve())] = sha256_file(path)
    return pd.read_csv(path)


def _audit_ancestry(path, train, held, sources):
    item = _read_json(path, sources)
    ancestry = Ancestry()
    ancestry.rowsets.update(item["rowsets"])
    ancestry.records.extend(item["records"])
    if not ancestry.records:
        raise ValueError("completed procedure has empty ancestry")
    return ancestry.audit(train, held)


def _validate_protocol(directory, kind, contract, sources):
    protocol = _read_json(directory / "protocol.json", sources)
    spec = protocol["specification"]
    if config_digest(spec) != protocol["protocol_hash"]:
        raise ValueError("frozen protocol content hash mismatch")
    for key in ("source_sha256", "splits_sha256", "data_hash", "n_rows"):
        if spec.get(key) != contract[key]:
            raise ValueError(f"{kind} protocol original-training population mismatch: {key}")
    fold_key = "fold_hash" if kind == "sequence" else "outer_fold_hash"
    if spec.get(fold_key) != contract["fold_hash"]:
        raise ValueError("frozen outer-fold hash mismatch")
    return protocol


def _validate_selection(choice, permitted):
    if choice["candidate"] not in permitted:
        raise ValueError("saved selected candidate is outside frozen grid")
    expected = [candidate["id"] for candidate in permitted]
    if [row["candidate"] for row in choice["scores"]] != expected:
        raise ValueError("saved inner score grid incomplete or reordered")
    for row in choice["scores"]:
        if len(row["fold_scores"]) != 3 or not np.isfinite(row["fold_scores"] + [row["mean"], row["se"]]).all():
            raise ValueError("saved inner selection scores malformed")
        expected_mean = float(np.mean(row["fold_scores"]))
        expected_se = float(np.std(row["fold_scores"], ddof=1) / np.sqrt(3))
        if not np.isclose(row["mean"], expected_mean, rtol=0, atol=1e-12) or not np.isclose(row["se"], expected_se, rtol=0, atol=1e-12):
            raise ValueError("saved inner selection arithmetic mismatch")
    if choice["candidate"]["id"] not in expected:
        raise ValueError("missing selected candidate score")
    best = max(choice["scores"], key=lambda row: row["mean"])
    expected_id = next(row["candidate"] for row in choice["scores"] if row["mean"] >= best["mean"] - best["se"])
    if choice["candidate"]["id"] != expected_id:
        raise ValueError("saved selection violates frozen inner-only parsimony rule")


def load_completed_study(directory, kind, train, *, contract=None):
    """Read only when all five completion markers exist, then verify every arm."""
    directory = Path(directory)
    markers = [directory / f"fold_{fold}/completed.json" for fold in range(5)]
    missing = [str(path) for path in markers if not path.exists()]
    if missing:
        raise ValueError(f"{kind} requires all five completed folds; missing {', '.join(missing)}")
    sources = {}
    contract = population_contract(train) if contract is None else contract
    protocol = _validate_protocol(directory, kind, contract, sources)
    spec = protocol["specification"]
    components = list(spec["family_order"] if kind == "sequence" else spec["component_order"])
    if kind == "sequence" and components != list(FAMILY_ORDER):
        raise ValueError("sequence components differ from declared procedures")
    arms = components + (["sequence_selector"] if kind == "sequence"
                         else ["embedding_selector", "mlp_log", "sequence_selector"])
    predictions = {arm: np.empty(len(train)) for arm in arms}
    selections, completions, audits, saved_choices, saved_selectors = [], [], [], [], []
    folds = nested.outer_folds(train)
    manifest = _read_csv(directory / "fold_manifest.csv", sources)
    if set(manifest.pair_id) != set(train.pair_id) or manifest.pair_id.duplicated().any():
        raise ValueError("fold manifest row coverage mismatch")
    expected = train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds)
    joined = expected.merge(manifest, on="pair_id", validate="one_to_one", suffixes=("", "_saved"), sort=False)
    if any(not joined[key].eq(joined[key + "_saved"]).all() for key in ("cluster_id", "allele", "outer_fold")):
        raise ValueError("saved fold manifest identity mismatch")
    for fold, marker in enumerate(markers):
        fold_path = marker.parent
        complete = _read_json(marker, sources)
        if complete["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("fold completion protocol mismatch")
        if not complete["audit"].get("outer_cluster_exclusion") or not complete["audit"].get("predicted_cluster_exclusion"):
            raise ValueError("fold completion ancestry audit not passed")
        if complete.get("schema") == "round2_sequence_completion.2":
            for field, filename in (("ancestry_sha256", "ancestry.json"), ("member_table_sha256", "members.csv"),
                                    ("selections_sha256", "selections.json"), ("selector_sha256", "selector.json"),
                                    ("ranker_map_sha256", "ranker_map.json"),
                                    ("ranker_mapping_inputs_sha256", "ranker_mapping_inputs.csv")):
                if sha256_file(fold_path / filename) != complete[field]:
                    raise ValueError(f"completed {filename} content hash mismatch")
        csv = fold_path / "predictions.csv"
        if sha256_file(csv) != complete["prediction_sha256"]:
            raise ValueError("completed prediction content hash mismatch")
        table = _read_csv(csv, sources)
        if set(table.arm) != set(arms):
            raise ValueError("completed prediction arm set mismatch")
        held = train[folds == fold]
        choices = (_read_json(fold_path / "selections.json", sources) if kind == "sequence" else
                   {arm: _read_json(fold_path / arm / "selection.json", sources) for arm in components})
        if set(choices) != set(components):
            raise ValueError("per-component selections incomplete")
        for arm in components:
            choice = choices[arm]
            _validate_selection(choice, spec["candidate_grids"][arm])
            selected = table[table.arm == arm]
            if "selected_candidate" not in selected or not selected.selected_candidate.eq(choice["candidate"]["id"]).all():
                raise ValueError("predictions do not identify saved inner-selected candidate")
            for score in choice["scores"]:
                selections.append({"study": kind, "outer_fold": fold, "arm": arm,
                                   "selected": score["candidate"] == choice["candidate"]["id"],
                                   "candidate": score["candidate"], "mean_inner_score": score["mean"],
                                   "inner_fold_se": score["se"],
                                   **{f"inner_fold_{i}": value for i, value in enumerate(score["fold_scores"])}})
        selector = _read_json(fold_path / "selector.json", sources)
        selector_arm = selector["family"] if kind == "sequence" else selector["arm"]
        permitted_selector = components if kind == "sequence" else spec["selector_order"]
        if selector_arm not in permitted_selector:
            raise ValueError("saved selector outside frozen candidate set")
        if kind == "sequence":
            from scripts.round2_sequence import select_family
            expected_selector = select_family(choices)
            if selector_arm != expected_selector["family"] or ("scores" in selector and selector["scores"] != expected_selector["scores"]):
                raise ValueError("saved sequence selector violates frozen inner-only rule")
        saved_choices.append(choices)
        saved_selectors.append(selector)
        selector_name = f"{kind}_selector"
        selected_values = validate_identity(held, table[table.arm == selector_arm], fold)
        selector_values = validate_identity(held, table[table.arm == selector_name], fold)
        if not np.allclose(selected_values, selector_values, rtol=0, atol=1e-12):
            raise ValueError("selector predictions differ from its declared selected procedure")
        selections.append({"study": kind, "outer_fold": fold, "arm": selector_name,
                           "selected": True, "candidate": selector_arm})
        for arm in arms:
            predictions[arm][folds == fold] = validate_identity(held, table[table.arm == arm], fold)
        audits.append({"fold": fold, **_audit_ancestry(fold_path / "ancestry.json", train, held, sources)})
        completions.append(complete)
    return {"directory": directory, "kind": kind, "protocol": protocol,
            "predictions": predictions, "selections": selections, "sources": sources,
            "completions": completions, "audits": audits,
            "saved_choices": saved_choices, "saved_selectors": saved_selectors}


def declared_contrasts(sequence, embedding=None):
    rows = []
    def add(baseline, candidate, kind, interpretation):
        if baseline != candidate:
            rows.append({"baseline": baseline, "candidate": candidate, "comparison_kind": kind,
                         "interpretation": interpretation})
    for arm in sequence["protocol"]["specification"]["family_order"] + ["sequence_selector"]:
        add("mlp_log", arm, "sequence_reference", "corrected sequence procedure versus original recipe")
        add("sequence_selector", arm, "sequence_selected_control", "component versus competitive inner-selected sequence")
    if embedding:
        spec = embedding["protocol"]["specification"]
        add("sequence_selector", "embedding_selector", "primary", "inner-selected embedding-or-original procedure versus competitive sequence")
        add("mlp_log", "embedding_selector", "embedding_original_reference", "inner-selected embedding-or-original procedure versus original")
        for arm in spec["component_order"]:
            add("mlp_log", arm, "embedding_original_reference", "declared component versus original sequence")
            add("sequence_selector", arm, "embedding_selected_control", "declared component versus competitive sequence")
        for learner in ("ridge", "mlp"):
            for rep in spec["representations"]:
                add(f"R0_{learner}", f"{rep}_{learner}", "matched_head", "same learner and head recipe; regularization selected separately")
            for candidate, baseline in spec["comparisons"]["positional_per_learner"]:
                add(f"{baseline}_{learner}", f"{candidate}_{learner}", "isolated_position_control", "frozen matched positional/membership control")
            for candidate, baseline in spec["comparisons"]["confounded_component_contrasts"]:
                add(f"{baseline}_{learner}", f"{candidate}_{learner}", "confounded_component", "changes positions plus chain context, channels or central pooling; cannot isolate position")
    return pd.DataFrame(rows).drop_duplicates(["baseline", "candidate", "comparison_kind"]).reset_index(drop=True)


def load_diagnostics(sequence, train):
    """Optional diagnostics require all five complete, matching full-cohort inputs."""
    directory = sequence["directory"]
    folds = nested.outer_folds(train)
    sources, status, predictions, contrasts = {}, {}, {}, []
    families = sequence["protocol"]["specification"]["family_order"]
    seed_paths = [directory / f"fold_{fold}/seed_predictions.csv" for fold in range(5)]
    if all(path.exists() for path in seed_paths):
        for fold, path in enumerate(seed_paths):
            if sequence["completions"][fold].get("seed_sha256") != sha256_file(path):
                raise ValueError("seed diagnostic missing/mismatched completion hash")
        seed_tables = [_read_csv(path, sources) for path in seed_paths]
        seed_keys = [(arm, seed) for arm in families for seed in (0, 1)]
        complete_seed = all(all(len(table[(table.arm == arm) & (table.seed == seed)]) == int((folds == fold).sum())
                                    for arm, seed in seed_keys) for fold, table in enumerate(seed_tables))
        if complete_seed:
            for arm, seed in seed_keys:
                name = f"{arm}__seed{seed}"
                predictions[name] = np.empty(len(train))
                for fold, table in enumerate(seed_tables):
                    selected = table[(table.arm == arm) & (table.seed == seed)]
                    predictions[name][folds == fold] = validate_identity(train[folds == fold], selected, fold)
            for arm in families:
                contrasts.append({"baseline": f"{arm}__seed0", "candidate": f"{arm}__seed1",
                                  "comparison_kind": "seed_diagnostic", "interpretation": "paired execution seed diagnostic; not assay noise"})
        status["paired_seed0_1"] = {"complete": complete_seed, "folds_available": 5,
                                     "expected_procedures": families, "completion_hash_bound": True}
    else:
        status["paired_seed0_1"] = {"complete": False, "folds_available": sum(path.exists() for path in seed_paths),
                                     "reason": "no partial-cohort diagnostic comparisons"}
    repeat_markers = [directory / f"fold_{fold}/repeats_completed.json" for fold in range(5)]
    if all(path.exists() for path in repeat_markers):
        markers = [_read_json(path, sources) for path in repeat_markers]
        repeat_counts = {marker["repeats"] for marker in markers}
        if repeat_counts != {2}:
            raise ValueError("execution repeat counts differ from declared two repeats")
        repeat_families = ["mlp_log", "tree_log", "pairlogit"]
        for fold, marker in enumerate(markers):
            fold_path = directory / f"fold_{fold}"
            if marker.get("protocol_hash") != sequence["protocol"]["protocol_hash"] or marker.get("primary_completion_sha256") != sha256_file(fold_path / "completed.json"):
                raise ValueError("repeat completion protocol/primary reference mismatch")
            for field, filename in (("repeat_prediction_sha256", "repeat_predictions.csv"),
                                    ("ancestry_sha256", "ancestry_with_repeats.json"),
                                    ("member_table_sha256", "repeat_members.csv")):
                if marker.get(field) != sha256_file(fold_path / filename):
                    raise ValueError(f"repeat completed {filename} content hash mismatch")
        tables = [_read_csv(directory / f"fold_{fold}/repeat_predictions.csv", sources) for fold in range(5)]
        for fold in range(5):
            _audit_ancestry(directory / f"fold_{fold}/ancestry_with_repeats.json", train, train[folds == fold], sources)
        for arm in repeat_families:
            for repeat in (1, 2):
                name = f"{arm}__repeat{repeat}"
                predictions[name] = np.empty(len(train))
                for fold, table in enumerate(tables):
                    selected = table[(table.arm == arm) & (table.repeat == repeat)]
                    predictions[name][folds == fold] = validate_identity(train[folds == fold], selected, fold)
                contrasts.append({"baseline": arm, "candidate": name, "comparison_kind": "execution_repeat",
                                  "interpretation": "identical-seed locked refit; not paired seed or assay variation"})
            contrasts.append({"baseline": f"{arm}__repeat1", "candidate": f"{arm}__repeat2",
                              "comparison_kind": "execution_repeat", "interpretation": "two identical-seed locked refits"})
        status["identical_seed_repeats"] = {"complete": True, "folds_available": 5,
                                             "repeats": 2, "completion_hash_bound": True}
    else:
        status["identical_seed_repeats"] = {"complete": False, "folds_available": sum(path.exists() for path in repeat_markers),
                                             "reason": "no partial-cohort repeat comparisons"}
    return predictions, pd.DataFrame(contrasts), status, sources


def load_embedding_seed_diagnostics(embedding, train):
    """Read paired seeds only with complete, hash-bound full-cohort coverage."""
    folds = nested.outer_folds(train)
    arms = [arm for arm in embedding["protocol"]["specification"]["component_order"] if arm.endswith("_mlp")]
    paths = [embedding["directory"] / f"fold_{fold}/seed_predictions.csv" for fold in range(5)]
    if not all(path.exists() for path in paths):
        return {}, pd.DataFrame(), {"complete": False, "folds_available": sum(p.exists() for p in paths)}, {}
    sources, tables = {}, []
    for fold, path in enumerate(paths):
        if embedding["completions"][fold].get("seed_sha256") != sha256_file(path):
            raise ValueError("embedding seed diagnostic completion hash mismatch")
        tables.append(_read_csv(path, sources))
    predictions, contrasts = {}, []
    for arm in arms:
        for seed in (0, 1):
            name = f"{arm}__seed{seed}"
            predictions[name] = np.empty(len(train))
            for fold, table in enumerate(tables):
                selected = table[(table.arm == arm) & (table.seed == seed)]
                predictions[name][folds == fold] = validate_identity(train[folds == fold], selected, fold)
        contrasts.append({"baseline": f"{arm}__seed0", "candidate": f"{arm}__seed1",
                          "comparison_kind": "seed_diagnostic", "interpretation": "embedding paired seeds; five stopping folds per seed, not execution repeats or assay noise"})
    return predictions, pd.DataFrame(contrasts), {"complete": True, "folds_available": 5,
        "completion_hash_bound": True, "procedures": arms}, sources


def _score_row(frame, name, prediction, minimum, *, scope, fold=None):
    panel = ev.eligible_alleles(frame.allele, frame.y_log1p, min_rows=minimum)
    row = {"procedure": name, "scope": scope, "panel_min_rows": minimum,
           "n_rows": len(frame), "n_clusters": int(frame.cluster_id.nunique()), "n_eligible_alleles": len(panel),
           "n_eligible_rows": int(frame.allele.isin(panel).sum()), "outer_fold": fold}
    if not panel:
        return {**row, "score_available": False, "median_per_allele_spearman": None,
                "mae_log1p": None, "pooled_spearman": None, "pooled_pearson_log1p": None,
                "median_precision_at_10": None, "n_unranked_alleles": 0}, pd.DataFrame()
    score = ev.score(name, frame.allele, frame.y_log1p.to_numpy(), prediction, alleles=panel)
    per_allele = score.per_allele_frame().reset_index(names="allele")
    per_allele["procedure"], per_allele["panel_min_rows"] = name, minimum
    return {**row, "score_available": True, "median_per_allele_spearman": score.median_spearman,
            "mae_log1p": score.mae_log1p, "pooled_spearman": score.pooled_spearman,
            "pooled_pearson_log1p": score.pooled_pearson_log1p,
            "median_precision_at_10": score.median_precision_at_k,
            "n_unranked_alleles": score.n_alleles_unranked}, per_allele


def report(train, sequence_directory, output_directory, *, embedding_directory=None,
           n_boot=N_BOOT, contract=None):
    """Pure arithmetic on completed procedures; test-only n_boot can be reduced."""
    if "split" not in train or not train.split.eq("train").all() or train.pair_id.duplicated().any():
        raise ValueError("results accept unique original-training rows only")
    sequence = load_completed_study(sequence_directory, "sequence", train, contract=contract)
    embedding = (load_completed_study(embedding_directory, "embedding", train, contract=contract)
                 if embedding_directory is not None else None)
    predictions = dict(sequence["predictions"])
    sources = dict(sequence["sources"])
    if embedding:
        from scripts.round2_embedding_screen import choose_embedding_selector
        for fold, completion in enumerate(embedding["completions"]):
            if completion["sequence_protocol_hash"] != sequence["protocol"]["protocol_hash"] or completion["sequence_prediction_sha256"] != sequence["completions"][fold]["prediction_sha256"]:
                raise ValueError("embedding selector used a different sequence reference")
            expected_selector = choose_embedding_selector(embedding["saved_choices"][fold], sequence["saved_choices"][fold])
            selector = embedding["saved_selectors"][fold]
            if selector["arm"] != expected_selector["arm"] or ("scores" in selector and selector["scores"] != expected_selector["scores"]):
                raise ValueError("saved embedding selector violates frozen inner-only rule")
        for arm in ("mlp_log", "sequence_selector"):
            if not np.allclose(predictions[arm], embedding["predictions"][arm], rtol=0, atol=1e-12):
                raise ValueError("embedding/sequence saved references differ")
        predictions.update(embedding["predictions"])
        sources.update(embedding["sources"])
    contrasts = declared_contrasts(sequence, embedding)
    diagnostic_predictions, diagnostic_contrasts, diagnostic_status, diagnostic_sources = load_diagnostics(sequence, train)
    if embedding:
        emb_pred, emb_contrasts, emb_status, emb_sources = load_embedding_seed_diagnostics(embedding, train)
        diagnostic_predictions.update(emb_pred)
        diagnostic_contrasts = pd.concat([diagnostic_contrasts, emb_contrasts], ignore_index=True)
        diagnostic_sources.update(emb_sources)
        diagnostic_status["embedding_paired_seed0_1"] = emb_status
    predictions.update(diagnostic_predictions)
    sources.update(diagnostic_sources)
    if not diagnostic_contrasts.empty:
        contrasts = pd.concat([contrasts, diagnostic_contrasts], ignore_index=True)
    points, fold_scores, per_allele, comparison_tables, bootstrap_meta, bootstrap_tables = [], [], [], [], {}, {}
    folds = nested.outer_folds(train)
    for minimum in (20, 50):
        comparisons, boot, meta = compare_predictions(train, predictions,
            list(contrasts[["baseline", "candidate"]].itertuples(index=False, name=None)),
            min_rows=minimum, n_boot=n_boot, seed=ev.BOOTSTRAP_SEED)
        # The contrast table may intentionally label a pair in multiple scopes.
        for i, (_, contrast) in enumerate(contrasts.iterrows()):
            for key in ("comparison_kind", "interpretation"):
                comparisons.loc[i, key] = contrast[key]
        comparison_tables.append(comparisons)
        bootstrap_meta[str(minimum)], bootstrap_tables[minimum] = meta, boot
        for name, prediction in predictions.items():
            point, detail = _score_row(train, name, prediction, minimum, scope="pooled_outer_held")
            points.append(point)
            per_allele.append(detail)
            for fold in range(5):
                keep = folds == fold
                row, _ = _score_row(train[keep], name, prediction[keep], minimum,
                                    scope="outer_fold_only", fold=fold)
                fold_scores.append(row)
    coverage = train[["allele", "y_log1p"]].groupby("allele").agg(
        n_rows=("y_log1p", "size"), n_distinct_labels=("y_log1p", "nunique"))
    coverage["eligible_min20"] = (coverage.n_rows >= 20) & (coverage.n_distinct_labels >= 2)
    coverage["eligible_min50"] = (coverage.n_rows >= 50) & (coverage.n_distinct_labels >= 2)
    coverage["n_clusters"] = train.groupby("allele").cluster_id.nunique()
    if "thalf_hours" in train:
        coverage["floor_rate"] = train.assign(floor=train.thalf_hours.eq(0)).groupby("allele").floor.mean()
    differences = []
    for contrast in diagnostic_contrasts.to_dict("records"):
        difference = predictions[contrast["candidate"]] - predictions[contrast["baseline"]]
        differences.append({**contrast, "n_rows": len(train), "mean_absolute_prediction_difference": float(np.abs(difference).mean()),
                            "maximum_absolute_prediction_difference": float(np.abs(difference).max()),
                            "prediction_rms_difference": float(np.sqrt(np.mean(difference ** 2)))})
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"pooled_scores.csv": pd.DataFrame(points), "fold_scores.csv": pd.DataFrame(fold_scores),
              "paired_comparisons.csv": pd.concat(comparison_tables, ignore_index=True),
              "per_allele_scores.csv": pd.concat(per_allele, ignore_index=True),
              "coverage.csv": coverage.reset_index(), "declared_contrasts.csv": contrasts,
              "selections.csv": pd.DataFrame(sequence["selections"] + (embedding["selections"] if embedding else [])),
              "execution_differences.csv": pd.DataFrame(differences, columns=["baseline", "candidate", "comparison_kind",
                "interpretation", "n_rows", "mean_absolute_prediction_difference",
                "maximum_absolute_prediction_difference", "prediction_rms_difference"])}
    for filename, table in tables.items():
        atomic_csv(out / filename, table)
    for minimum, boot in bootstrap_tables.items():
        atomic_csv(out / f"bootstrap_scores_min{minimum}.csv", boot.assign(resample=np.arange(len(boot))))
    prediction_table = train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds)
    for name, values in predictions.items():
        prediction_table[name] = values
    atomic_csv(out / "matched_predictions.csv", prediction_table)
    status = {"schema": "round2_results.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "sequence_and_embedding" if embedding else "sequence_only",
              "all_five_folds_completed": True, "n_rows": len(train), "n_clusters": int(train.cluster_id.nunique()),
              "protocol_hashes": {"sequence": sequence["protocol"]["protocol_hash"],
                                  **({"embedding": embedding["protocol"]["protocol_hash"]} if embedding else {})},
              "bootstrap": bootstrap_meta, "shared_draws_across_all_procedures": True,
              "diagnostics": diagnostic_status, "source_sha256": sources,
              "ancestry": {"sequence": sequence["audits"], **({"embedding": embedding["audits"]} if embedding else {})},
              "uncertainty": "conditional paired cluster intervals; previously examined development data, not independent confirmation",
              "selection": "report all frozen procedures; no selection or nomination from report outcomes",
              "output_sha256": {path.name: sha256_file(path) for path in out.glob("*.csv")}}
    atomic_json(out / "manifest.json", status)
    primary = tables["paired_comparisons.csv"].query("panel_min_rows == 20 and comparison_kind == 'primary'")
    point_table = tables["pooled_scores.csv"].query("panel_min_rows == 20")
    lines = ["# Completed round-two development results", "",
             f"Scope: **{status['scope']}**. All five outer folds cover exactly **{len(train):,} original training rows** and **{train.cluster_id.nunique():,} peptide clusters**. Saved predictions, source/fold contracts and ancestry were validated before scoring. Labels are joined through the original training partition only.", "",
             f"All declared comparisons use **{n_boot:,} paired whole-peptide-cluster bootstrap draws**, seed {ev.BOOTSTRAP_SEED}, shared across every procedure. The 20-row panel is primary; the 50-row panel is reported separately. These intervals condition on fitted predictions and do not include search/execution uncertainty or constitute independent confirmation.", "",
             "| Procedure | Pooled median per-allele Spearman (min20) |", "|---|---:|"]
    lines += [f"| {row.procedure} | {row.median_per_allele_spearman:.6f} |" for row in point_table.itertuples() if "__" not in row.procedure]
    if len(primary):
        row = primary.iloc[0]
        lines += ["", f"Prespecified primary comparison, embedding selector versus competitive sequence selector: **{row.delta:+.6f} [{row.ci_low:+.6f}, {row.ci_high:+.6f}]**."]
    else:
        lines += ["", "Embedding results were not included. The embedding comparison remains pending; no incomplete embedding-fold outcomes were examined."]
    lines += ["", "Every component/reference contrast is retained in `paired_comparisons.csv`. Matched-head and isolated positional controls have explicit comparison labels; `confounded_component` contrasts change chain context, channels or pooling and cannot isolate positional information. No arm is selected from these report outcomes.", "",
              "`fold_scores.csv` contains fold-specific medians and panels. Those statistics are separate from the pooled primary statistic; their median is not substituted for it. `coverage.csv` and `per_allele_scores.csv` retain allele eligibility, tied/constant predictions and row coverage.", "",
              f"Sequence paired seed0/1 diagnostics complete: **{diagnostic_status['paired_seed0_1']['complete']}**. Sequence identical-seed execution repeats complete: **{diagnostic_status['identical_seed_repeats']['complete']}**. Embedding paired seed0/1 diagnostics complete: **{diagnostic_status.get('embedding_paired_seed0_1', {}).get('complete', False)}**. Available complete diagnostics are in the paired/pooled tables and `execution_differences.csv`, with file hashes verified against completion markers. Incomplete diagnostics receive no partial-cohort comparison.", "",
              "Protocols, all per-fold inner selections, source/output hashes and audited ancestry are retained in `manifest.json` and `selections.csv`. Seed variation, identical-seed execution variation, paired-cluster uncertainty and assay variation remain distinct; no assay noise ceiling is inferred."]
    _atomic_bytes(out / "REPORT.md", ("\n".join(lines) + "\n").encode())
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-directory", type=Path, default=SEQUENCE)
    parser.add_argument("--embedding-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, default=OUT)
    args = parser.parse_args()
    status = report(training_frame(), args.sequence_directory, args.output_directory,
                    embedding_directory=args.embedding_directory)
    print(json.dumps({"scope": status["scope"], "n_rows": status["n_rows"],
                      "report": str((args.output_directory / "REPORT.md").resolve())}))


if __name__ == "__main__":
    main()
