"""Report all completed complementarity procedures on identical training rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation as ev, nested
from pepstab.boltz_embeddings import _atomic_bytes, sha256_file
from pepstab.round2_complementarity import (
    BLEND_WEIGHTS, CORRECTION_WEIGHTS, REPRESENTATIONS, RESIDUAL_ALPHA, _partition_key, choose_meta_weight,
)
from pepstab.round2_nested import Ancestry, blend, config_digest
from pepstab.round2_reporting import compare_predictions
from pepstab.round2_sequence import candidates
from scripts.round2_complementarity import _validate_fixed_original_members
from scripts.round2_results import (
    _score_row, atomic_csv, atomic_json, load_completed_study,
    population_contract, training_frame, validate_identity, _validate_selection,
)

DIRECTORY = ROOT / "reports/further_research/round2/complementarity"
SEQUENCE = ROOT / "reports/further_research/round2/sequence"
EMBEDDING = ROOT / "reports/further_research/round2/embedding"
OUT = DIRECTORY / "results"
ARMS = ("mlp_log", "sequence_selector", "R2_ridge", "R4_ridge",
        "R2_blend", "R4_blend", "R2_residual", "R4_residual")
N_BOOT = 2000


def _record(path, sources):
    sources[str(Path(path).resolve())] = sha256_file(path)


def _json(path, sources):
    _record(path, sources)
    return json.loads(Path(path).read_text())


def _csv(path, sources):
    _record(path, sources)
    return pd.read_csv(path)


def _same_choice(saved, expected):
    if saved["weight"] != expected["weight"] or len(saved["scores"]) != len(expected["scores"]):
        return False
    for a, b in zip(saved["scores"], expected["scores"]):
        if a["weight"] != b["weight"] or not np.allclose(
            a["fold_scores"] + [a["mean"], a["se"]], b["fold_scores"] + [b["mean"], b["se"]], rtol=0, atol=1e-12):
            return False
    return True


def _meta_frame(fit, table):
    if table.pair_id.duplicated().any() or set(table.pair_id) != set(fit.pair_id):
        raise ValueError("meta input fitting-row coverage mismatch")
    merged = fit[["pair_id", "cluster_id", "allele", "y_log1p"]].merge(
        table, on="pair_id", validate="one_to_one", suffixes=("", "_saved"), sort=False)
    for key in ("cluster_id", "allele"):
        if not merged[key].eq(merged[key + "_saved"]).all():
            raise ValueError("meta input cluster/allele identity mismatch")
    if not np.array_equal(merged.meta_fold, nested.inner_folds(fit, 3)):
        raise ValueError("meta inner-fold identity mismatch")
    for column in ["original_clean_oof"] + [f"{rep}_{suffix}" for rep in REPRESENTATIONS
                                             for suffix in ("nested_selected_oof", "nested_residual_oof")]:
        ev.validate_finite(merged[column], column)
    return merged


def _audit_meta_ancestry(ancestry, fit, provenance):
    fold = nested.inner_folds(fit, 3)
    if [item["meta_fold"] for item in provenance] != [0, 1, 2]:
        raise ValueError("residual target provenance incomplete")
    for k, item in enumerate(provenance):
        inner_fit, inner_held = fit[fold != k], fit[fold == k]
        if item["fit_rows"] != ancestry.rows(inner_fit) or item["prediction_rows"] != ancestry.rows(inner_held):
            raise ValueError("residual target provenance fitting partition mismatch")
        prefix = f"complementarity/base/{_partition_key(inner_fit)}/crossfit/"
        base_records = [record for record in ancestry.records if record["role"] == "fixed_original_clean_oof"
                        and record.get("context", "").startswith(prefix)]
        if len(base_records) != 3:
            raise ValueError("residual targets lack three fresh inner-fit base crossfits")
        predicted = [row for record in base_records for row in ancestry.rowsets[record["prediction_rows"]]]
        if len(predicted) != len(inner_fit) or set(predicted) != set(inner_fit.pair_id):
            raise ValueError("residual clean crossfit target coverage mismatch")
        forbidden = set(inner_held.pair_id)
        for record in ancestry.records:
            context = record.get("context", "")
            if f"/meta/{k}/" not in context and not context.startswith(prefix):
                continue
            for role in ("fit_rows", "stopping_rows", "selection_rows", "scaler_fit_rows"):
                if role in record and forbidden.intersection(ancestry.rowsets[record[role]]):
                    raise ValueError("meta-held fitting/stopping/selection/scaling ancestry violation")
            if record["role"] == "residual_ridge_meta_prediction" and (
                    record["base_oof_target_hash"] != item["base_oof_sha256"]
                    or record["residual_target_hash"] != item["residual_target_sha256"]):
                raise ValueError("residual target hash linkage mismatch")


def load_completed(directory, train, sequence, embedding):
    """Require complete five-fold artifacts, ancestry and immutable inner choices."""
    directory = Path(directory)
    markers = [directory / f"fold_{fold}/completed.json" for fold in range(5)]
    if not all(path.exists() for path in markers):
        raise ValueError("complementarity requires all five completed folds before outcome reads")
    sources = {}
    protocol = _json(directory / "protocol.json", sources)
    spec = protocol["specification"]
    if config_digest(spec) != protocol["protocol_hash"]:
        raise ValueError("complementarity protocol content hash mismatch")
    contract = population_contract(train)
    for key, value in contract.items():
        if spec.get(key) != value:
            raise ValueError(f"complementarity original-training population mismatch: {key}")
    if (spec["sequence_protocol_hash"] != sequence["protocol"]["protocol_hash"]
            or spec["embedding_protocol_hash"] != embedding["protocol"]["protocol_hash"]):
        raise ValueError("complementarity frozen reference protocol mismatch")
    if (spec["representations"] != list(REPRESENTATIONS)
            or spec["blending"]["simplex_weights"] != list(BLEND_WEIGHTS)
            or spec["residual"]["correction_weights"] != list(CORRECTION_WEIGHTS)
            or spec["residual"]["alpha"] != RESIDUAL_ALPHA):
        raise ValueError("complementarity frozen grids differ from declared scope")
    folds = nested.outer_folds(train)
    fold_manifest = _csv(directory / "fold_manifest.csv", sources)
    expected = train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds)
    if fold_manifest.pair_id.duplicated().any() or set(fold_manifest.pair_id) != set(train.pair_id):
        raise ValueError("complementarity fold manifest coverage mismatch")
    check = expected.merge(fold_manifest, on="pair_id", validate="one_to_one", suffixes=("", "_saved"))
    if any(not check[key].eq(check[key + "_saved"]).all() for key in ("cluster_id", "allele", "outer_fold")):
        raise ValueError("complementarity fold manifest identity mismatch")
    predictions = {arm: np.empty(len(train)) for arm in ARMS}
    selection_rows, audits, completions = [], [], []
    for fold, marker in enumerate(markers):
        path = marker.parent
        held, fit = train[folds == fold], train[folds != fold]
        complete = _json(marker, sources)
        if complete.get("schema") != "round2_complementarity_completion.1" or complete["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("complementarity fold completion protocol mismatch")
        if complete["n_rows"] != len(held):
            raise ValueError("complementarity completion row count mismatch")
        if not complete["audit"].get("outer_cluster_exclusion") or not complete["audit"].get("predicted_cluster_exclusion"):
            raise ValueError("complementarity completion ancestry audit failed")
        expected_artifacts = {"selections.json", "residual_target_provenance.json", "meta_inputs.csv",
                              "ancestry.json", "members.csv", "input_identity.json"}
        if set(complete["artifact_sha256"]) != expected_artifacts:
            raise ValueError("complementarity artifact hash set incomplete")
        for name, digest in {"predictions.csv": complete["prediction_sha256"], **complete["artifact_sha256"]}.items():
            if sha256_file(path / name) != digest:
                raise ValueError(f"complementarity {name} completion hash mismatch")
        identity = _json(path / "input_identity.json", sources)
        if identity["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("complementarity input identity protocol mismatch")
        # Verify original reference snapshots, rather than trusting reused values.
        for filename, digest in identity["reference_hashes"].items():
            if sha256_file(filename) != digest:
                raise ValueError("complementarity audited source reference hash changed")
        required_reference = {str((sequence["directory"] / f"fold_{fold}/predictions.csv").resolve()),
                              str((embedding["directory"] / f"fold_{fold}/predictions.csv").resolve())}
        if not required_reference.issubset(identity["reference_hashes"]):
            raise ValueError("complementarity source prediction hash references incomplete")
        _validate_fixed_original_members(sequence["directory"], fold)
        if sequence["saved_choices"][fold]["mlp_log"]["candidate"] != candidates()["mlp_log"][0]:
            raise ValueError("source original baseline recipe is not fixed")
        _record(sequence["directory"] / f"fold_{fold}/members.csv", sources)
        table = _csv(path / "predictions.csv", sources)
        if set(table.arm) != set(ARMS):
            raise ValueError("complementarity reported arm set mismatch")
        for arm in ARMS:
            predictions[arm][folds == fold] = validate_identity(held, table[table.arm == arm], fold)
        for arm, reference in (("mlp_log", sequence), ("sequence_selector", sequence),
                               ("R2_ridge", embedding), ("R4_ridge", embedding)):
            if not np.allclose(predictions[arm][folds == fold], reference["predictions"][arm][folds == fold], rtol=0, atol=1e-12):
                raise ValueError("complementarity reused outer reference differs")
        meta = _meta_frame(fit, _csv(path / "meta_inputs.csv", sources))
        choices = _json(path / "selections.json", sources)
        if set(choices) != set(REPRESENTATIONS):
            raise ValueError("complementarity per-representation selections incomplete")
        for rep in REPRESENTATIONS:
            nested_choices = choices[rep]["nested_embedding_choices"]
            if [item["meta_fold"] for item in nested_choices] != [0, 1, 2]:
                raise ValueError("nested embedding selections incomplete")
            for item in nested_choices:
                _validate_selection(item, spec["embedding"][rep])
            for kind, weights, residual, suffix in (("blend", BLEND_WEIGHTS, False, "nested_selected_oof"),
                                                    ("residual", CORRECTION_WEIGHTS, True, "nested_residual_oof")):
                # Audit the already frozen choice; this does not select report arms.
                expected_choice = choose_meta_weight(meta, meta.original_clean_oof.to_numpy(),
                    meta[f"{rep}_{suffix}"].to_numpy(), weights, residual=residual,
                    min_rows=spec["min_rows_inner"])
                saved = choices[rep][kind]
                if not _same_choice(saved, expected_choice):
                    raise ValueError("complementarity stored weight/inner scores violate frozen rule")
                for score in saved["scores"]:
                    selection_rows.append({"outer_fold": fold, "arm": f"{rep}_{kind}",
                                           "weight": score["weight"], "selected": score["weight"] == saved["weight"],
                                           "mean_inner_score": score["mean"], "inner_fold_se": score["se"],
                                           **{f"inner_fold_{i}": value for i, value in enumerate(score["fold_scores"])}})
                if saved["weight"] == 0 and not np.array_equal(predictions[f"{rep}_{kind}"][folds == fold], predictions["mlp_log"][folds == fold]):
                    raise ValueError("zero-weight complementarity endpoint differs from baseline")
            expected_blend = blend(predictions["mlp_log"][folds == fold], predictions[f"{rep}_ridge"][folds == fold], choices[rep]["blend"]["weight"])
            if not np.allclose(predictions[f"{rep}_blend"][folds == fold], expected_blend, rtol=0, atol=1e-12):
                raise ValueError("selected outer blend does not reproduce declared weight")
        ancestry_item = _json(path / "ancestry.json", sources)
        ancestry = Ancestry()
        ancestry.rowsets.update(ancestry_item["rowsets"])
        ancestry.records.extend(ancestry_item["records"])
        if not ancestry.records:
            raise ValueError("complementarity ancestry empty")
        _audit_meta_ancestry(ancestry, fit, _json(path / "residual_target_provenance.json", sources))
        audits.append({"outer_fold": fold, **ancestry.audit(train, held)})
        completions.append(complete)
    return {"protocol": protocol, "predictions": predictions, "selections": selection_rows,
            "sources": sources, "audits": audits, "completions": completions}


def contrasts():
    rows = []
    for rep in REPRESENTATIONS:
        for procedure in ("ridge", "blend", "residual"):
            arm = f"{rep}_{procedure}"
            for baseline in ("mlp_log", "sequence_selector"):
                rows.append({"baseline": baseline, "candidate": arm,
                             "comparison_kind": "standalone" if procedure == "ridge" else "complementarity"})
            if procedure != "ridge":
                rows.append({"baseline": f"{rep}_ridge", "candidate": arm,
                             "comparison_kind": "complementarity_vs_embedding_alone"})
    return pd.DataFrame(rows)


def report(directory=DIRECTORY, sequence_directory=SEQUENCE, embedding_directory=EMBEDDING,
           output_directory=OUT, *, train=None, n_boot=N_BOOT):
    directory = Path(directory)
    # Complete preflight precedes any source or complementarity outcome read.
    if not all((directory / f"fold_{fold}/completed.json").exists() for fold in range(5)):
        raise ValueError("complementarity requires all five completed folds before outcome reads")
    train = training_frame() if train is None else train
    if not train.split.eq("train").all() or train.pair_id.duplicated().any():
        raise ValueError("complementarity results accept unique original-training rows only")
    sequence = load_completed_study(sequence_directory, "sequence", train)
    embedding = load_completed_study(embedding_directory, "embedding", train)
    for fold in range(5):
        marker = embedding["completions"][fold]
        if (marker["sequence_protocol_hash"] != sequence["protocol"]["protocol_hash"]
                or marker["sequence_prediction_sha256"] != sequence["completions"][fold]["prediction_sha256"]):
            raise ValueError("source embedding used different sequence provenance")
    study = load_completed(directory, train, sequence, embedding)
    definitions = contrasts()
    points, comparisons, fold_scores, details, boots, metadata = [], [], [], [], {}, {}
    folds = nested.outer_folds(train)
    for minimum in (20, 50):
        comparison, boot, meta = compare_predictions(train, study["predictions"],
            list(definitions[["baseline", "candidate"]].itertuples(index=False, name=None)),
            min_rows=minimum, n_boot=n_boot, seed=ev.BOOTSTRAP_SEED)
        comparison["comparison_kind"] = definitions.comparison_kind.to_numpy()
        comparisons.append(comparison)
        boots[minimum], metadata[str(minimum)] = boot, meta
        for arm, values in study["predictions"].items():
            row, detail = _score_row(train, arm, values, minimum, scope="pooled_outer_held")
            points.append(row)
            details.append(detail)
            for fold in range(5):
                keep = folds == fold
                row, _ = _score_row(train[keep], arm, values[keep], minimum, scope="outer_fold_only", fold=fold)
                fold_scores.append(row)
    coverage = train.groupby("allele").agg(n_rows=("pair_id", "size"), n_clusters=("cluster_id", "nunique"),
                                           n_distinct_labels=("y_log1p", "nunique"))
    for minimum in (20, 50):
        coverage[f"eligible_min{minimum}"] = (coverage.n_rows >= minimum) & (coverage.n_distinct_labels >= 2)
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"pooled_scores.csv": pd.DataFrame(points), "paired_comparisons.csv": pd.concat(comparisons, ignore_index=True),
              "fold_scores.csv": pd.DataFrame(fold_scores), "per_allele_scores.csv": pd.concat(details, ignore_index=True),
              "coverage.csv": coverage.reset_index(), "selections.csv": pd.DataFrame(study["selections"]),
              "declared_contrasts.csv": definitions}
    for name, table in tables.items():
        atomic_csv(out / name, table)
    for minimum, boot in boots.items():
        atomic_csv(out / f"bootstrap_scores_min{minimum}.csv", boot.assign(resample=np.arange(len(boot))))
    matched = train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds)
    for name, values in study["predictions"].items():
        matched[name] = values
    atomic_csv(out / "matched_predictions.csv", matched)
    status = {"schema": "round2_complementarity_results.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "all_five_folds_complete": True, "protocol_hash": study["protocol"]["protocol_hash"],
              "n_rows": len(train), "n_clusters": int(train.cluster_id.nunique()), "bootstrap": metadata,
              "shared_draws_across_procedures": True, "selection": "all frozen procedures reported; no nomination or selection from report outcomes",
              "source_sha256": {**sequence["sources"], **embedding["sources"], **study["sources"]},
              "audits": study["audits"], "uncertainty": "conditional paired cluster intervals; development data, not independent confirmation",
              "output_sha256": {name: sha256_file(out / name) for name in tables}}
    atomic_json(out / "manifest.json", status)
    pooled = tables["pooled_scores.csv"].query("panel_min_rows == 20")
    lines = ["# Completed complementarity development results", "",
             f"All five completed outer folds cover exactly **{len(train):,} original training rows**. Protocols, completion hashes, source references, row/cluster/allele identity and ancestry passed validation. Fixed original baseline members were checked against the 30-member recipe. Stored blend/correction choices reproduce their frozen inner-fold rule; zero endpoints reproduce the baseline exactly.", "",
             f"All 16 declared contrasts use **{n_boot:,} paired whole-peptide-cluster bootstrap draws**, seed {ev.BOOTSTRAP_SEED}, shared across every procedure. The 20-row panel is primary; min50 sensitivity and fold-specific statistics are separate files.", "",
             "| Procedure | Pooled median per-allele Spearman (min20) |", "|---|---:|"]
    lines += [f"| {row.procedure} | {row.median_per_allele_spearman:.6f} |" for row in pooled.itertuples()]
    lines += ["", "`paired_comparisons.csv` reports every blend/residual procedure and R2/R4 embedding alone against both original and competitive sequence references, plus each complementary procedure against its embedding-alone arm. Standalone and complementarity gains answer different questions. No arm is selected from these outcomes.", "",
              "Each assessed cluster was excluded from base fitting, stopping and ridge selection. Residual targets came from fresh clean original cross-fits inside the residual fitting partition. Other outer-fold predictions never supplied stacking inputs. Inner-to-outer training-size differences remain a declared limitation.", "",
              "Intervals condition on saved fitted predictions and exclude search/execution uncertainty. These previously examined training data provide development evidence, not independent confirmation. `selections.csv`, `coverage.csv`, `per_allele_scores.csv` and `manifest.json` retain the selections, coverage and provenance."]
    _atomic_bytes(out / "REPORT.md", ("\n".join(lines) + "\n").encode())
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--sequence-directory", type=Path, default=SEQUENCE)
    parser.add_argument("--embedding-directory", type=Path, default=EMBEDDING)
    parser.add_argument("--output-directory", type=Path, default=OUT)
    args = parser.parse_args()
    result = report(args.directory, args.sequence_directory, args.embedding_directory, args.output_directory)
    print(json.dumps({"n_rows": result["n_rows"], "report": str((args.output_directory / "REPORT.md").resolve())}))


if __name__ == "__main__":
    main()
