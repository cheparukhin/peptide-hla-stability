#!/usr/bin/env python3
"""Build/check the offline round-two submission snapshot from saved evidence.

Writes the marked site section, evidence JSON and copies verified figure exports.
No model execution, source-result mutation, network calls, or dependencies.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "reports/further_research/round2/"
REPO_URL = "https://github.com/cheparukhin/peptide-hla-stability/blob/main/"
START = "<!-- ROUND2 GENERATED START -->"
END = "<!-- ROUND2 GENERATED END -->"
SOURCES = {}


def read(name, kind="json"):
    return read_path(BASE + name, kind)


def read_path(path, kind="json"):
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    if kind == "csv":
        return list(csv.DictReader(raw.decode().splitlines()))
    return json.loads(raw) if kind == "json" else raw.decode()


def one(rows, **values):
    matches = [r for r in rows if all(str(r[k]) == str(v) for k, v in values.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one evidence row for {values}; found {len(matches)}")
    return matches[0]


def compare(rows, baseline, candidate, panel=20, **extra):
    row = one(rows, baseline=baseline, candidate=candidate, panel_min_rows=panel, **extra)
    for key in ("baseline_score", "candidate_score", "delta", "ci_low", "ci_high"):
        row[key] = float(row[key])
        if not math.isfinite(row[key]):
            raise ValueError(f"Nonfinite {key}: {row}")
    if not math.isclose(row["candidate_score"] - row["baseline_score"], row["delta"], abs_tol=1e-12):
        raise ValueError("Comparison scores and delta disagree")
    if not row["ci_low"] <= row["delta"] <= row["ci_high"]:
        raise ValueError("Reported point falls outside its interval")
    return row


def fmt(value, signed=False):
    return format(float(value), "+.6f" if signed else ".6f").replace("-", "−")


def interval(row):
    return f'{fmt(row["delta"], True)} [{fmt(row["ci_low"], True)}, {fmt(row["ci_high"], True)}]'


def link(name, label):
    return path_link(BASE + name, label)


def path_link(path, label):
    return f'<a href="../{path}">{html.escape(label)}</a>'


def full_cohort_presentation():
    """Prepare tables only after complete saved outputs pass independent audit.

    The default snapshot includes this completed, gated presentation.
    It performs no fitting, metric recomputation or resampling.
    """
    base = "reports/followup3/inverse_fulltrain_candidate/"
    required = ("completion_audit.json", "independent_audit.json", "comparisons.csv", "fixed_query_descriptive.csv", "INDEPENDENT_FINAL_REVIEW.json")
    missing = [name for name in required if not (ROOT / base / name).is_file()]
    if missing:
        raise ValueError("Full-cohort presentation unavailable: missing " + ", ".join(missing))
    complete = read_path(base + "completion_audit.json")
    independent = read_path(base + "independent_audit.json")
    protocol = read_path(base + "protocol.json")
    final_review = read_path(base + "INDEPENDENT_FINAL_REVIEW.json")
    extraction = read_path(base + "independent_extraction_audit.json")
    repair_base = "reports/followup3/inverse_fulltrain_repair/"
    repair_manifest = read_path(repair_base + "manifest.json")
    repair_review = read_path(repair_base + "PRELAUNCH_REVIEW.json")
    read_path("scripts/followup3_inverse_fulltrain_repair.py", "text")
    if final_review["status"] != "passed" or final_review["protocol_hash"] != protocol["protocol_hash"] or extraction["status"] != "passed" or repair_review["status"] != "passed_prelaunch_repair_review":
        raise ValueError("Passed full-cohort final/extraction and repair reviews required")
    for key,name in (("independent_audit_sha256","independent_audit.json"),("completion_audit_sha256","completion_audit.json"),("independent_extraction_audit_sha256","independent_extraction_audit.json")):
        if final_review[key] != SOURCES[base + name]:
            raise ValueError("Full-cohort final review binds stale audit bytes")
    repair = final_review["repair"]
    if repair["manifest_file_sha256"] != SOURCES[repair_base+"manifest.json"] or repair["review_sha256"] != SOURCES[repair_base+"PRELAUNCH_REVIEW.json"] or repair["wrapper_sha256"] != SOURCES["scripts/followup3_inverse_fulltrain_repair.py"] or repair["manifest_hash"] != repair_manifest["manifest_hash"] or not repair["scientific_recipe_unchanged"] or not repair["metadata_revalidation_exact_parity"] or repair["prior_fits"] != 0:
        raise ValueError("Reviewed full-cohort operational repair ancestry mismatch")
    audit_source = "scripts/followup3_fulltrain_independent_audit.py"
    shared_source = "scripts/followup3_representation_independent_audit.py"
    read_path(audit_source, "text")
    read_path(shared_source, "text")
    if complete["status"] != "passed" or independent["status"] != "passed" or independent["study"] != "inverse_fulltrain_selected_development" or independent["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Passed complete and independent full-cohort audits required")
    if (complete["rows"], complete["mlp_fits"], complete["ridge_fits"]) != (19716,300,150) or independent["rows"] != 19716 or independent["fit_counts"] != {"mlp":300,"ridge":150} or independent["arms"] != ["R5","R5_P","R5_PC"]:
        raise ValueError("Full-cohort frozen coverage/fit graph mismatch")
    if complete["saved_weight_replay"] or complete["tranche_primary_success_claim"] or independent["saved_weight_replay"] or independent["bootstrap_regenerated"] or independent["tranche_primary"] or independent["independent_refits"] != 0:
        raise ValueError("Full-cohort selected-development audit scope mismatch")
    if (independent["cached_rows_unchanged"],independent["new_rows"],independent["fixed_query_rows"],independent["comparison_rows_checked"],independent["fixed_query_entries_checked"]) != (4449,15267,4449,12,24):
        raise ValueError("Full-cohort cached/new/fixed-query coverage mismatch")
    if independent["audit_source_sha256"] != SOURCES[audit_source] or independent["shared_audit_source_sha256"] != SOURCES[shared_source]:
        raise ValueError("Full-cohort independent auditor source changed")
    comparisons = read_path(base + "comparisons.csv", "csv")
    fixed = read_path(base + "fixed_query_descriptive.csv", "csv")
    for name in ("comparisons.csv", "fixed_query_descriptive.csv", "bootstrap_mlp_min20.csv", "bootstrap_mlp_min50.csv", "bootstrap_ridge_min20.csv", "bootstrap_ridge_min50.csv"):
        if name.startswith("bootstrap_"):
            read_path(base + name,"text")
        if complete["outputs"][name] != SOURCES[base + name]:
            raise ValueError("Full-cohort saved table changed after audit")
    if len(comparisons) != 12 or len(fixed) != 24:
        raise ValueError("All full-cohort result rows must remain visible")
    arms = ("R5","R5_P","R5_PC")
    labels = {"R5":"R5", "R5_P":"R5 + P", "R5_PC":"R5 + P + C"}
    contrasts = (("R5","R5_PC","Net"),("R5_P","R5_PC","Context"),("R5","R5_P","Peptide-only"))
    absolute = independent["absolute_scores"]
    score_cells = {}
    contrast_rows = []
    fixed_rows = []
    for learner in ("mlp","ridge"):
        for minimum in (20,50):
            points = absolute[f"{learner}_min{minimum}"]
            if set(points) != set(arms) or not all(math.isfinite(float(v)) and -1 <= float(v) <= 1 for v in points.values()):
                raise ValueError("Complete finite full-cohort absolute scores required")
            score_cells[learner,minimum] = points
            panel = independent["panels"][str(minimum)]
            for baseline,candidate,label in contrasts:
                row = compare(comparisons,baseline,candidate,panel=minimum,learner=learner)
                if int(row["n_rows"]) != 19716 or int(row["n_alleles"]) != len(panel) or int(row["n_boot"]) != 2000 or int(row["bootstrap_seed"]) != 20261003 or not math.isclose(row["baseline_score"],points[baseline],abs_tol=1e-12) or not math.isclose(row["candidate_score"],points[candidate],abs_tol=1e-12):
                    raise ValueError("Full-cohort contrast/absolute score mismatch")
                contrast_rows.append(f'<tr><td>{learner.upper()}</td><td>≥{minimum}</td><th scope="row">{label}: {labels[candidate]} minus {labels[baseline]}</th><td class="num">{interval(row)}</td></tr>')
            fixed_panel = independent["fixed_query_panels"][str(minimum)]
            for kind,names in (("point",arms),("increment",("net","context","peptide"))):
                for name in names:
                    row = one(fixed,learner=learner,min_rows=minimum,kind=kind,comparison=name)
                    before,after,change = (float(row[k]) for k in ("original_subset","fulltrained_same_queries","old_to_new_change"))
                    if not all(math.isfinite(v) for v in (before,after,change)) or not math.isclose(after-before,change,abs_tol=1e-12) or int(row["query_rows"]) != 4449 or int(row["eligible_alleles"]) != len(fixed_panel) or row["panel_alleles"].split("|") != fixed_panel:
                        raise ValueError("Full-cohort fixed-query descriptive arithmetic/panel mismatch")
                    fixed_rows.append(f'<tr><td>{learner.upper()}</td><td>≥{minimum}</td><th scope="row">{html.escape(labels.get(name,name))} {kind}</th><td class="num">{fmt(before)}</td><td class="num">{fmt(after)}</td><td class="num">{fmt(change,True)}</td></tr>')
    score_rows = "\n".join(f'<tr><th scope="row">{labels[arm]}</th>' + "".join(f'<td class="num">{fmt(score_cells[learner,minimum][arm])}</td>' for learner,minimum in (("mlp",20),("ridge",20),("mlp",50),("ridge",50))) + '</tr>' for arm in arms)
    main = compare(comparisons,"R5","R5_PC",learner="mlp")
    net_ridge = compare(comparisons,"R5","R5_PC",learner="ridge")
    fixed_net = one(fixed,learner="mlp",min_rows=20,kind="increment",comparison="net")
    fixed_r5 = one(fixed,learner="mlp",min_rows=20,kind="point",comparison="R5")
    if any(not float(row["ci_low"]) < 0 < float(row["ci_high"]) for row in comparisons) or not main["delta"] < 0:
        raise ValueError("Final inconclusive full-cohort interpretation requires review")
    body = f'''<section id="inverse-full-cohort" aria-labelledby="inverse-full-title">
    <div class="shead"><p class="eyebrow">Selected development extension · 19,716 original-training pairs · 450 fits</p><h2 class="title" id="inverse-full-title">Full-cohort scoring does not establish added ProteinMPNN value.</h2><p class="lede">MLP R5 scores {fmt(main["baseline_score"])} versus {fmt(main["candidate_score"])} for R5 + P + C: {interval(main)} under its conditional pointwise 95% interval. Ridge net is {interval(net_ridge)}. Every declared contrast spans zero, establishing neither added benefit nor harm or equivalence. This extension was selected after balanced-panel outcomes; it is not independent confirmation or a fourth original-family primary.</p></div>
    <details class="evidence-detail"><summary>All three arms, both learners and both panels</summary><div class="scroller" tabindex="0" role="region" aria-label="Full-cohort absolute scores"><table><caption>Median per-allele Spearman · ≥20: {len(independent["panels"]["20"])} alleles · ≥50: {len(independent["panels"]["50"])} alleles</caption><thead><tr><th scope="col">Arm</th><th scope="col">MLP ≥20</th><th scope="col">Ridge ≥20</th><th scope="col">MLP ≥50</th><th scope="col">Ridge ≥50</th></tr></thead><tbody>{score_rows}</tbody></table></div><div class="scroller" tabindex="0" role="region" aria-label="All full-cohort comparisons"><table><caption>All 12 contrasts · conditional pointwise 95% intervals</caption><thead><tr><th scope="col">Learner</th><th scope="col">Panel</th><th scope="col">Comparison</th><th scope="col">Difference [95% interval]</th></tr></thead><tbody>{"".join(contrast_rows)}</tbody></table></div></details>
    <details class="evidence-detail"><summary>Original 4,449 fixed queries: descriptive old/new comparison</summary><p>The MLP net changes from {fmt(fixed_net["original_subset"],True)} to {fmt(fixed_net["fulltrained_same_queries"],True)}, while R5 rises from {fmt(fixed_r5["original_subset"])} to {fmt(fixed_r5["fulltrained_same_queries"])}. Old and new held predictions are compared on the same identities and fixed eligible panels, without an additional fit or confidence interval. Source coverage, inner selection and stopping change together, so these points do not isolate a causal sample-size effect.</p><div class="scroller" tabindex="0" role="region" aria-label="All fixed-query descriptive points"><table><caption>All 24 fixed-query entries · no additional interval</caption><thead><tr><th scope="col">Learner</th><th scope="col">Panel</th><th scope="col">Entry</th><th scope="col">Old</th><th scope="col">New</th><th scope="col">Change</th></tr></thead><tbody>{"".join(fixed_rows)}</tbody></table></div></details>
    <p class="note"><b>Stop computational search; independent kinetic-assay confirmation is next.</b> Retain the simpler embedding head. No new grid, combined arm or full FoldX expansion is admitted. Few-shot remains the sole positive primary in the original three-question family.</p>
    <p class="note">Independent saved-output arithmetic, selection and ancestry audits cover all 300 MLP members and 150 ridge fits. No independent refitting, bootstrap regeneration or saved-weight replay is claimed. Intervals omit assay and repeated fitting/selection variation and extension-choice uncertainty.</p><p class="source-line">{path_link(base+"comparisons.csv","Every contrast")} · {path_link(base+"fixed_query_descriptive.csv","All fixed-query points")} · {path_link(base+"independent_audit.json","Independent audit")} · {path_link(base+"completion_audit.json","Completion audit")} · {path_link(base+"INDEPENDENT_FINAL_REVIEW.json","Final independent review")} · {path_link(repair_base+"PRELAUNCH_REVIEW.json","Operational repair provenance")}</p>
    </section>'''
    return body, {"status":"complete and independently audited; selected development", "main":main,"absolute_scores":absolute,"comparisons":comparisons,"fixed_query_descriptive":fixed,"independent_audit":independent,"final_independent_review":final_review}


def build():
    full_cohort_body, full_cohort_data = full_cohort_presentation()
    followup_base = "reports/followup3/"
    read_path(followup_base + "REPORT.md", "text")
    read_path(followup_base + "EXECUTION.md", "text")
    read_path("reports/NEXT_RESEARCH_PLAN.md", "text")
    followup_cost = read_path(followup_base + "budget.json")
    followup_closeout = read_path(followup_base + "resources/cloud_closeout.json")
    if followup_cost["reserved_usd"] != 0 or any(j["status"] != "closed" for j in followup_cost["jobs"]) or followup_cost["conservative_closed_usd"] > followup_cost["inclusive_new_tranche_ceiling_usd"] or followup_closeout["conservative_closed_usd"] != followup_cost["conservative_closed_usd"] or followup_closeout["reserved_usd"] != 0:
        raise ValueError("Closed bounded follow-up accounting required")
    if followup_closeout["status"] != "closed_scientific_outputs_preserved_temporary_volume_deleted" or not followup_closeout["historical_structure_volume_untouched"]:
        raise ValueError("Completed preservation and temporary-volume closeout required")
    deletion_path = followup_closeout["temporary_volume_delete_receipt"]
    deletion = read_path(deletion_path)
    if SOURCES[deletion_path] != followup_closeout["temporary_volume_delete_receipt_sha256"] or not deletion["delete_api_returned_success"] or not deletion["fresh_lookup_not_found"]:
        raise ValueError("Verified temporary-volume deletion receipt required")
    for path,expected in followup_closeout["scientific_output_preservation"].items():
        preservation = read_path(path)
        if SOURCES[path] != expected or preservation["status"] != "complete" or preservation["errors"] or preservation["conflicts"]:
            raise ValueError("Conflict-free scientific output preservation required")
    read_path(followup_base + "PROTOCOL.md", "text")
    read_path(followup_base + "THIRD_PRIMARY_AMENDMENT.md", "text")
    read_path(followup_base + "ASSESSMENT_PRECHECK.md", "text")
    read_path(followup_base + "inverse_parser_recovery/SOURCE_BOUND_PARSER_AMENDMENT.md", "text")
    read_path(followup_base + "inverse_parser_recovery/FAILED_STRUCTURE_INSPECTION.json", "text")
    inverse_methods = read_path(followup_base + "inverse_execution_v2/protocol.json")["specification"]["scoring"]
    inverse_base = followup_base + "inverse_execution_v2/"
    inverse_scores = read_path(inverse_base + "pooled_scores.csv", "csv")
    inverse_comparisons = read_path(inverse_base + "comparisons.csv", "csv")
    inverse_completion = read_path(inverse_base + "completion_audit.json")
    inverse_independent = read_path(inverse_base + "independent_audit.json")
    inverse_family = read_path(inverse_base + "primary_family3_interval.json")
    inverse_admission = read_path(inverse_base + "HEAD_ADMISSION_AUDIT.json")
    read_path(inverse_base + "parser_recovery_receipt.json", "text")
    read_path(inverse_base + "bootstrap_mlp_min20.csv", "text")
    read_path(followup_base + "FULL_COHORT_DECISION_FRAMEWORK.md", "text")
    read_path("scripts/followup3_representation_independent_audit.py", "text")
    if any(a["status"] != "passed" for a in (inverse_completion, inverse_independent, inverse_admission)) or inverse_completion["scientific_fits"] != 900 or inverse_independent["fit_counts"] != {"mlp":600,"ridge":300} or inverse_independent["rows"] != 4449:
        raise ValueError("Completed 900-fit inverse study and independent audit required")
    if inverse_completion["saved_weight_replay"] or inverse_independent["saved_weight_replay"] or len(inverse_independent["panels"]["20"]) != 69 or len(inverse_independent["panels"]["50"]) != 68:
        raise ValueError("Inverse audit scope and panel statements require review")
    recovery = inverse_independent["parser_recovery"]
    if (recovery["inherited_row_hashes_preserved"], recovery["corrected_or_missing_rows"], recovery["completed_rows"]) != (4163,286,4449) or not recovery["frozen_scientific_sources_unchanged"] or recovery["receipt_sha256"] != SOURCES[inverse_base + "parser_recovery_receipt.json"]:
        raise ValueError("Completed inverse parser-recovery ancestry required")
    if inverse_independent["amendment_sha256"] != SOURCES[followup_base + "THIRD_PRIMARY_AMENDMENT.md"] or inverse_independent["audit_source_sha256"] != SOURCES["scripts/followup3_representation_independent_audit.py"]:
        raise ValueError("Inverse independent audit provenance mismatch")
    for name in ("pooled_scores.csv", "comparisons.csv"):
        if inverse_completion["outputs"][name] != SOURCES[inverse_base + name]:
            raise ValueError("Inverse saved table changed after completion")
    if len(inverse_scores) != 24 or len(inverse_comparisons) != 28 or inverse_independent["comparison_rows_checked"] != 28:
        raise ValueError("All frozen inverse score and comparison rows required")
    inverse_primary = compare(inverse_comparisons, "R5_P", "R5_PC", learner="mlp")
    inverse_net = compare(inverse_comparisons, "R5", "R5_PC", learner="mlp")
    inverse_sensitivity = compare(inverse_comparisons, "R5_P", "R5_PC", panel=50, learner="mlp")
    audited_family = one(inverse_independent["three_primary_intervals"], learner="mlp", panel_min_rows=20, primary=True)
    if inverse_family != audited_family or not math.isclose(inverse_family["interval_level"], 59/60) or inverse_family["saved_draws_sha256"] != SOURCES[inverse_base + "bootstrap_mlp_min20.csv"] or not math.isclose(inverse_family["delta"], inverse_primary["delta"], abs_tol=1e-12):
        raise ValueError("Current three-primary inverse family interval required")
    if not all(float(r["ci_low"]) < 0 < float(r["ci_high"]) for r in (inverse_primary, inverse_net, inverse_family)):
        raise ValueError("Inverse inconclusive interpretation requires review")
    inverse_absolute = []
    for arm,label in (("R0","R0"),("R0_P","R0 + P"),("R0_PC","R0 + P + C"),("R5","R5"),("R5_P","R5 + P"),("R5_PC","R5 + P + C")):
        mlp = one(inverse_scores, learner="mlp", arm=arm, min_rows=20)
        ridge_row = one(inverse_scores, learner="ridge", arm=arm, min_rows=20)
        inverse_absolute.append({"arm":arm,"label":label,"mlp":float(mlp["median_spearman"]),"ridge":float(ridge_row["median_spearman"])})
    inverse_r5 = next(r["mlp"] for r in inverse_absolute if r["arm"] == "R5")
    inverse_r5p = next(r["mlp"] for r in inverse_absolute if r["arm"] == "R5_P")
    inverse_r5pc = next(r["mlp"] for r in inverse_absolute if r["arm"] == "R5_PC")
    if not math.isclose(inverse_r5pc-inverse_r5p,inverse_primary["delta"],abs_tol=1e-12) or not math.isclose(inverse_r5pc-inverse_r5,inverse_net["delta"],abs_tol=1e-12):
        raise ValueError("Inverse absolute score arithmetic mismatch")
    inverse_html = "\n".join(f'<tr><th scope="row">{r["label"]}</th><td class="num">{fmt(r["mlp"])}</td><td class="num">{fmt(r["ridge"])}</td></tr>' for r in inverse_absolute)
    influence_base = inverse_base + "exploratory_influence/"
    inverse_influence = read_path(influence_base + "summary.json")
    influence_alleles = read_path(influence_base + "leave_one_allele.csv", "csv")
    influence_folds = read_path(influence_base + "leave_one_outer_fold.csv", "csv")
    read_path(influence_base + "per_allele_rhos.csv", "csv")
    read_path("scripts/followup3_inverse_influence.py", "text")
    if not inverse_influence["posthoc"] or inverse_influence["primary_family_changed"] or inverse_influence["fixed_eligible_panel"] != inverse_independent["panels"]["20"] or inverse_influence["source_audit_sha256"] != SOURCES[inverse_base + "independent_audit.json"] or inverse_influence["audit_source_sha256"] != SOURCES["scripts/followup3_inverse_influence.py"]:
        raise ValueError("Fixed-panel post-hoc influence provenance requires review")
    for name, expected in inverse_influence["outputs"].items():
        if SOURCES[influence_base + name] != expected:
            raise ValueError("Inverse influence saved table changed")
    for learner in ("mlp", "ridge"):
        allele_values = [float(r["net_R5_delta"]) for r in influence_alleles if r["learner"] == learner]
        fold_values = [float(r["fixed_panel_net_R5_delta"]) for r in influence_folds if r["learner"] == learner]
        summary = inverse_influence["summary"][learner]
        if len(allele_values) != 69 or len(fold_values) != 5 or [min(allele_values),max(allele_values)] != summary["leave_one_allele_range"] or [min(fold_values),max(fold_values)] != summary["leave_one_fold_range"] or not summary["leave_one_fold_all_panels_rankable"]:
            raise ValueError("Inverse influence descriptive ranges require review")
    if min(inverse_influence["summary"]["mlp"]["leave_one_allele_range"] + inverse_influence["summary"]["mlp"]["leave_one_fold_range"]) <= 0 or inverse_influence["summary"]["ridge"]["full_net_R5_delta"] >= 0:
        raise ValueError("Inverse descriptive influence interpretation requires review")


    energy_base = followup_base + "energy_execution_v2/"
    energy_scores = read_path(energy_base + "pooled_scores.csv", "csv")
    energy_comparisons = read_path(energy_base + "comparisons.csv", "csv")
    energy_completion = read_path(energy_base + "completion_audit.json")
    energy_independent = read_path(energy_base + "independent_audit.json")
    energy_review = read_path(energy_base + "INDEPENDENT_FINAL_REVIEW.json")
    energy_family = read_path(energy_base + "primary_family3_interval.json")
    read_path(energy_base + "bootstrap_mlp_min20.csv", "text")
    energy_recovery_base = followup_base + "energy_recovery_v3/"
    energy_recovery = read_path(energy_recovery_base + "independent_recovery_audit.json")
    energy_features = read_path(energy_recovery_base + "independent_feature_audit.json")
    if any(a["status"] != "passed" for a in (energy_completion, energy_independent, energy_review, energy_recovery, energy_features)) or energy_independent["fit_counts"] != {"mlp":200,"ridge":100} or energy_independent["rows"] != 4449 or energy_completion["baseline_fits_repeated"] != 0:
        raise ValueError("Completed independently audited energy study required")
    if energy_independent["protocol_hash"] != energy_completion["protocol_hash"] or energy_independent["amendment_sha256"] != SOURCES[followup_base + "THIRD_PRIMARY_AMENDMENT.md"] or energy_independent["audit_source_sha256"] != SOURCES["scripts/followup3_representation_independent_audit.py"] or energy_independent["panels"] != inverse_independent["panels"]:
        raise ValueError("Energy matched-panel and independent audit ancestry mismatch")
    if any((a["old_success_hashes_unchanged"], a["original_interrupted_starts_unchanged"], a["manual_attempts_verified"], a["combined_rows"], a["energy_columns"]) != (4376,73,73,4449,29) for a in (energy_recovery, energy_features)) or not energy_features["feature_matrix_checked"]:
        raise ValueError("Complete energy recovery and exact feature-matrix audit required")
    for key, path in (("independent_assessment_audit_sha256",energy_base+"independent_audit.json"), ("family_primary_interval_sha256",energy_base+"primary_family3_interval.json"), ("recovery_audit_sha256",energy_recovery_base+"independent_recovery_audit.json"), ("feature_audit_sha256",energy_recovery_base+"independent_feature_audit.json")):
        if energy_review[key] != SOURCES[path]:
            raise ValueError("Energy final independent review has stale ancestry")
    for name in ("pooled_scores.csv", "comparisons.csv"):
        if energy_completion["output_sha256"][name] != SOURCES[energy_base + name]:
            raise ValueError("Completed energy table changed")
    if len(energy_scores) != 16 or len(energy_comparisons) != 12 or energy_independent["comparison_rows_checked"] != 12 or energy_independent["saved_weight_replay"] or energy_independent["bootstrap_regenerated"]:
        raise ValueError("Energy complete table coverage/audit scope requires review")
    energy_primary = compare(energy_comparisons, "R5", "R5_E", learner="mlp")
    energy_ridge = compare(energy_comparisons, "R5", "R5_E", learner="ridge")
    energy_sensitivity = compare(energy_comparisons, "R5", "R5_E", panel=50, learner="mlp")
    if energy_family != one(energy_independent["three_primary_intervals"], learner="mlp", panel_min_rows=20, primary=True) or not math.isclose(energy_family["interval_level"],59/60) or energy_family["saved_draws_sha256"] != SOURCES[energy_base+"bootstrap_mlp_min20.csv"] or not math.isclose(energy_family["delta"],energy_primary["delta"],abs_tol=1e-12):
        raise ValueError("Current three-primary energy family interval required")
    if not all(float(r["ci_low"]) < 0 < float(r["ci_high"]) for r in (energy_primary,energy_ridge,energy_sensitivity,energy_family)):
        raise ValueError("Energy inconclusive interpretation requires review")
    energy_absolute = []
    for arm,label in (("R0","R0"),("R0_E","R0 + E"),("R5","R5"),("R5_E","R5 + E")):
        mlp = one(energy_scores,learner="mlp",arm=arm,min_rows=20)
        ridge_row = one(energy_scores,learner="ridge",arm=arm,min_rows=20)
        energy_absolute.append({"arm":arm,"label":label,"mlp":float(mlp["median_spearman"]),"ridge":float(ridge_row["median_spearman"])})
        if arm in ("R0","R5"):
            for learner,row in (("mlp",mlp),("ridge",ridge_row)):
                if row["median_spearman"] != one(inverse_scores,learner=learner,arm=arm,min_rows=20)["median_spearman"]:
                    raise ValueError("Energy baseline is not the exact reused inverse baseline")
    energy_html = "\n".join(f'<tr><th scope="row">{r["label"]}</th><td class="num">{fmt(r["mlp"])}</td><td class="num">{fmt(r["ridge"])}</td></tr>' for r in energy_absolute)
    full_metadata_base = followup_base + "inverse_fulltrain_metadata/"
    full_metadata = read_path(full_metadata_base + "cohort_metadata_comparison.json")
    metadata_counts = full_metadata["counts"]
    if full_metadata["numeric_labels_predictions_mpnn_scores_read"] or (metadata_counts["balanced_rows"],metadata_counts["full_rows"],metadata_counts["balanced_alleles"],metadata_counts["full_alleles"],metadata_counts["balanced_clusters"],metadata_counts["full_clusters"],metadata_counts["newly_covered_clusters"]) != (4449,19716,75,75,2082,3587,1505):
        raise ValueError("Label-free full-cohort metadata scope/counts changed")
    density = full_metadata["row_density"]
    if (density["additional_rows_in_previously_covered_clusters"],density["rows_in_newly_covered_clusters"]) != (11656,3611) or 11656+3611 != metadata_counts["additional_rows"] or not math.isclose(metadata_counts["row_expansion_factor"],19716/4449) or not math.isclose(metadata_counts["cluster_expansion_factor"],3587/2082):
        raise ValueError("Label-free coverage/density arithmetic mismatch")
    for source,expected in full_metadata["sources"].items():
        SOURCES[source] = hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        if SOURCES[source] != expected:
            raise ValueError("Label-free cohort identity inputs changed")
    for name,expected in full_metadata["count_table_sha256"].items():
        read_path(full_metadata_base+name,"csv")
        if SOURCES[full_metadata_base+name] != expected:
            raise ValueError("Label-free count table changed")
    full_decision = read_path(followup_base + "FULL_COHORT_DECISION.json")
    if not full_decision["approve_fulltrain_inverse"] or not full_decision["energy_final_audit_complete"] or full_decision["full_energy_approved"] or full_decision["scope"]["rows"] != 19716 or full_decision["scope"]["arms"] != ["R5","R5_P","R5_PC"] or (full_decision["scope"]["mlp_fits"],full_decision["scope"]["ridge_fits"]) != (300,150) or full_decision["resources"]["full_study_maximum_usd"] != 40:
        raise ValueError("Selected full-cohort extension scope requires review")

    foldx_methods = read_path(followup_base + "foldx/protocol.json")["specification"]
    foldx_technical = read_path(followup_base + "foldx/technical_receipt.json")
    if (foldx_technical["status"], foldx_technical["attempted"], foldx_technical["successful"], foldx_technical["failures"]) != ("passed", 32, 32, 0):
        raise ValueError("FoldX technical status requires editorial review")
    fewshot_base = followup_base + "fewshot/"
    fewshot_curves = read_path(fewshot_base + "absolute_curves.csv", "csv")
    fewshot_comparisons = read_path(fewshot_base + "comparisons.csv", "csv")
    fewshot_audit = read_path(fewshot_base + "audit.json")
    fewshot_independent = read_path(fewshot_base + "independent_audit.json")
    fewshot_family = read_path(fewshot_base + "primary_family3_interval.json")
    fewshot_target_budgets = read_path(fewshot_base + "actual_budgets.csv", "csv")
    fewshot_source_budgets = read_path(fewshot_base + "source_budgets.csv", "csv")
    fewshot_deletions = read_path(fewshot_base + "all_hla_group_deletions.csv", "csv")
    read_path(fewshot_base + "scores_by_draw.csv", "csv")
    read_path(fewshot_base + "bootstrap_crossed_min20.csv", "text")
    for name in ("REPORT.md", "SPECIFICATION.md", "SCALER_REPLAY_AMENDMENT.md", "protocol_replay_v2.json"):
        read_path(fewshot_base + name, "text")
    read_path("scripts/followup3_fewshot_independent_audit.py", "text")
    if any(a["status"] != "passed" for a in (fewshot_audit, fewshot_independent)) or (fewshot_independent["cells"], fewshot_independent["source_fits"], fewshot_independent["target_adapters"]) != (75, 450, 9180):
        raise ValueError("Completed independent few-shot audit required")
    if not fewshot_audit["saved_models_predictions_replayed_exactly"] or fewshot_audit["panels"]["20"] != fewshot_audit["panels"]["50"] or len(fewshot_audit["panels"]["20"]) != 68:
        raise ValueError("Few-shot replay and identical-panel statements require review")
    fewshot_primary = one(fewshot_comparisons, contrast="primary_R5_adaptation20", panel_min_rows=20, mode="crossed")
    fewshot_zero = float(one(fewshot_curves, panel_min_rows=20, arm="R5", budget=0)["score"])
    fewshot_twenty = float(one(fewshot_curves, panel_min_rows=20, arm="R5", budget=20)["score"])
    if not math.isclose(fewshot_twenty - fewshot_zero, float(fewshot_primary["delta"]), abs_tol=1e-12) or not math.isclose(fewshot_family["point"], float(fewshot_primary["delta"]), abs_tol=1e-12):
        raise ValueError("Few-shot primary arithmetic mismatch")
    if fewshot_family["saved_draws_sha256"] != SOURCES[fewshot_base + "bootstrap_crossed_min20.csv"] or fewshot_family["amendment_sha256"] != SOURCES[followup_base + "THIRD_PRIMARY_AMENDMENT.md"] or not math.isclose(fewshot_family["interval_level"], 59/60):
        raise ValueError("Few-shot family interval provenance mismatch")
    if fewshot_independent["available_source_rows_range"] != [11213, 11950] or fewshot_independent["sequence_consumed_source_rows_range"] != [10213, 11055]:
        raise ValueError("Few-shot source-label budget ranges require editorial review")
    if float(fewshot_primary["ci_low"]) <= 0 or fewshot_family["ci_low"] <= 0:
        raise ValueError("Positive few-shot primary interpretation requires review")
    primary_deletions = [float(r["delta"]) for r in fewshot_deletions if r["contrast"] == "primary_R5_adaptation20" and r["panel_min_rows"] == "20"]
    if len(primary_deletions) != 35 or min(primary_deletions) <= 0:
        raise ValueError("Few-shot HLA-group deletion statement requires review")
    fewshot_rows = []
    for budget in (0, 5, 10, 20):
        target_rows = [one(fewshot_target_budgets, draw=draw, k=budget) for draw in (0, 1, 2)]
        if any(int(r["target_labels"]) != 68*budget or int(r["target_tuning_labels"]) or int(r["target_stopping_labels"]) for r in target_rows):
            raise ValueError("Few-shot target-label budget statement requires review")
        scores = [float(one(fewshot_curves, panel_min_rows=20, arm=arm, budget=budget)["score"]) for arm in ("R0", "R5", "sequence_anchor")]
        fewshot_rows.append((budget, 68*budget, *scores))
    fewshot_html = "\n".join(f'<tr><th scope="row">{budget}</th><td class="num">{labels:,}</td><td class="num">{fmt(r0)}</td><td class="num">{fmt(r5)}</td><td class="num">{fmt(sequence)}</td></tr>' for budget, labels, r0, r5, sequence in fewshot_rows)
    offset_base = followup_base + "fewshot_offset/"
    offset_comparisons = read_path(offset_base + "comparisons.csv", "csv")
    offset_audit = read_path(offset_base + "audit.json")
    offset_independent = read_path(offset_base + "independent_audit.json")
    for name in ("SPECIFICATION.md", "protocol.json", "REPORT.md"):
        read_path(offset_base + name, "text")
    read_path("scripts/followup3_offset_independent_audit.py", "text")
    if any(a["status"] != "passed" or not a["posthoc"] or a["primary_family_changed"] or a["independent_fits"] != 0 for a in (offset_audit, offset_independent)):
        raise ValueError("Passed post-hoc offset audit with unchanged primary family required")
    if (offset_independent["prediction_rows"], offset_independent["within_cell_rank_checks"], offset_independent["paired_bootstrap_rows_checked"]) != (44199, 1020, 2000) or not offset_audit["all_within_cell_ranks_identical"] or not offset_audit["point_and_all_bootstrap_differences_telescope"]:
        raise ValueError("Offset diagnostic coverage and rank/arithmetic statements require review")
    if offset_audit["prior_protocol_hash"] != fewshot_audit["protocol_hash"] or offset_independent["protocol_hash"] != offset_audit["protocol_hash"] or offset_independent["audit_source_sha256"] != SOURCES["scripts/followup3_offset_independent_audit.py"] or offset_audit["output_sha256"]["comparisons.csv"] != SOURCES[offset_base + "comparisons.csv"]:
        raise ValueError("Offset diagnostic provenance mismatch")
    if any((a["cells"], a["alleles"]) != (75, 68) for a in (offset_audit, offset_independent)) or not offset_independent["original_sources_results_family_unchanged"]:
        raise ValueError("Offset diagnostic population or original primary preservation mismatch")
    if any(float(r["interval_level"]) != .95 or r["family_adjusted"] != "False" for r in offset_comparisons):
        raise ValueError("Offset diagnostic nominal interval statement requires review")
    offset_full = one(offset_comparisons, contrast="full_minus_offset")
    offset_zero = one(offset_comparisons, contrast="offset_minus_zero")
    offset_reference = one(offset_comparisons, contrast="original_full_minus_zero")
    offset_scores = offset_independent["absolute_scores"]
    if not math.isclose(offset_scores["R5_k20"] - offset_scores["R5_offset20"], float(offset_full["delta"]), abs_tol=1e-12) or not math.isclose(offset_scores["R5_offset20"] - offset_scores["R5_k0"], float(offset_zero["delta"]), abs_tol=1e-12) or not math.isclose(float(offset_full["delta"]) + float(offset_zero["delta"]), float(fewshot_primary["delta"]), abs_tol=1e-12):
        raise ValueError("Offset diagnostic arithmetic mismatch")
    if float(offset_full["ci_low"]) <= 0 or not float(offset_zero["ci_low"]) < 0 < float(offset_zero["ci_high"]):
        raise ValueError("Offset diagnostic interpretation requires review")
    figure_base = followup_base + "figures/"
    fewshot_figure = read_path(figure_base + "fewshot_curves.json")
    read_path(figure_base + "build_fewshot.py", "text")
    for name in ("fewshot_curves.svg", "fewshot_curves.png"):
        SOURCES[figure_base + name] = hashlib.sha256((ROOT / figure_base / name).read_bytes()).hexdigest()
        if SOURCES[figure_base + name] != fewshot_figure["output_sha256"][name]:
            raise ValueError("Few-shot figure asset/manifest mismatch")
    for source, expected in fewshot_figure["input_sha256"].items():
        if SOURCES.get(source) != expected:
            raise ValueError("Few-shot figure has stale scientific inputs")
    if fewshot_figure["renderer"]["script_sha256"] != SOURCES[figure_base + "build_fewshot.py"] or fewshot_figure["displayed_curve_rows"] != [r for r in fewshot_curves if r["panel_min_rows"] == "20"] or fewshot_figure["primary_family_interval"] != fewshot_family:
        raise ValueError("Few-shot figure plotted values or renderer are stale")

    descriptor_figure = read_path(figure_base + "descriptor_comparison.json")
    read_path(figure_base + "build_descriptor_comparison.py", "text")
    for name in ("descriptor_comparison.svg", "descriptor_comparison.png"):
        SOURCES[figure_base + name] = hashlib.sha256((ROOT / figure_base / name).read_bytes()).hexdigest()
        if SOURCES[figure_base + name] != descriptor_figure["output_sha256"][name]:
            raise ValueError("Descriptor figure asset/manifest mismatch")
    for source, expected in descriptor_figure["input_sha256"].items():
        if SOURCES.get(source) != expected:
            raise ValueError("Descriptor figure scientific input hashes are stale")
    if descriptor_figure["renderer"]["script_sha256"] != SOURCES[figure_base+"build_descriptor_comparison.py"] or descriptor_figure["eligible_alleles"] != inverse_independent["panels"]["20"] or descriptor_figure["primary_comparisons"][0]["family_98_333333"] != inverse_family or descriptor_figure["primary_comparisons"][1]["family_98_333333"] != energy_family:
        raise ValueError("Descriptor figure family intervals/renderer are stale")
    expected_descriptor_rows = [{"arm":rep+suffix,"representation":rep,"descriptor":label,"median_spearman":float(one(energy_scores if suffix == "_E" else inverse_scores,learner="mlp",arm=rep+suffix,min_rows=20)["median_spearman"])} for suffix,label in (("","Baseline"),("_P","+ P"),("_PC","+ P + C"),("_E","+ E")) for rep in ("R0","R5")]
    if descriptor_figure["displayed_absolute_rows"] != expected_descriptor_rows:
        raise ValueError("Descriptor figure does not show all eight audited MLP scores")

    metadata_base = "reports/followup3/independent_data/"
    metadata = read_path(metadata_base + "candidate_metadata_audit.json")
    metadata_groups = read_path(metadata_base + "candidate_source_groups.csv", "csv")
    read_path(metadata_base + "PROVENANCE.md", "text")
    read_path(metadata_base + "study_method_status.csv", "csv")
    read_path(metadata_base + "publication_metadata.json")
    if (metadata["rows"], metadata["peptides"], metadata["components"], len(metadata["allele_counts"]), len(metadata["source_pmids"])) != (129, 126, 113, 9, 36):
        raise ValueError("Independent candidate counts require editorial review")
    largest_metadata_group = max(metadata_groups, key=lambda row: int(row["retained_rows"]))
    if (largest_metadata_group["pmid"], largest_metadata_group["allele"], int(largest_metadata_group["retained_rows"])) != ("27198675", "HLA-A*02:01", 23):
        raise ValueError("Independent largest-stratum statement requires editorial review")
    main = read("results/main/paired_comparisons.csv", "csv")
    scores = read("results/main/pooled_scores.csv", "csv")
    comp = read("complementarity/results/paired_comparisons.csv", "csv")
    transfer = read("transfer/results/comparisons.csv", "csv")
    costs = read("resources/runtime_cost_summary.json")["cloud_accounting_closeout"]
    cache = read("resources/cache_completion_audit.json")
    repeats = read("resources/embedding_repeat_completion_audit.json")
    for name in ("REPORT.md", "resources/embedding_completion_audit.json", "resources/transfer_completion_audit.json",
                 "resources/complementarity_completion_audit.json", "independent_data/README.md"):
        read(name, "text")
    boltz_methods = cache["frozen_extraction_settings"]
    primary = compare(main, "sequence_selector", "embedding_selector")
    sensitivity = compare(main, "sequence_selector", "embedding_selector", 50)
    controls = [
        ("Retain R3 token positions vs. matched tiled pooling", compare(main, "R3_pooled_mlp", "R3_mlp")),
        ("Retain full 9×34 positions vs. matched pooling", compare(main, "R5_position9x34_pooled_mlp", "R5_position9x34_mlp")),
        ("Crystal contact membership vs. shuffled membership", compare(main, "R5_shuffled_mlp", "R5_mlp")),
    ]
    blend = compare(comp, "sequence_selector", "R4_blend")
    residual = compare(comp, "sequence_selector", "R4_residual")
    transfer_rows = [compare(transfer, "bounded_MLP_sequence_selector", "embedding_selector", regime=regime, mode="crossed") for regime in ("hla", "joint")]
    original_transfer = [compare(transfer, "original", "embedding_selector", regime=regime, mode="crossed") for regime in ("hla", "joint")]
    extension_base = "reports/finalization/transfer_cloud_recovery/results/"
    extension = read_path(extension_base + "comparisons.csv", "csv")
    extension_manifest = read_path(extension_base + "manifest.json")
    read_path(extension_base + "REPORT.md", "text")
    extension_audit = read_path(extension_base + "coverage_audit.json")
    extension_matched = [
        (f"{regime.upper()} · {candidate} − {baseline}",
         compare(extension, baseline, candidate, regime=regime, mode="crossed"))
        for regime in ("hla", "joint")
        for baseline, candidate in (("R0_ridge", "R4_ridge"), ("R0_mlp", "R5_mlp"))
    ]
    extension_references = [
        compare(extension, baseline, "embedding_selector", regime=regime, mode="crossed")
        for regime in ("hla", "joint") for baseline in ("original", "expanded_sequence_selector")
    ]
    extension_sequence = [compare(extension, "original", "expanded_sequence_selector", regime=regime, mode="crossed") for regime in ("hla", "joint")]
    extension_matched_html = "\n".join(f'<tr><th scope="row">{label}</th><td class="num">{interval(row)}</td></tr>' for label, row in extension_matched)
    extension_reference_html = "\n".join(f'<tr><th scope="row">{row["regime"].upper()} · {row["baseline"].replace("_", " ")}</th><td class="num">{fmt(row["baseline_score"])}</td><td class="num">{fmt(row["candidate_score"])}</td><td class="num">{interval(row)}</td></tr>' for row in extension_references)
    ranking_base = "reports/finalization/ranking/"
    ranking_comparisons = read_path(ranking_base + "comparisons.csv", "csv")
    ranking_scores = read_path(ranking_base + "scores.csv", "csv")
    ranking_selections = read_path(ranking_base + "selections.csv", "csv")
    ranking_audit = read_path(ranking_base + "completion_audit.json")
    hybrid_base = "reports/finalization/hybrids_cloud_v1/"
    hybrid_scores = read_path(hybrid_base + "scores.csv", "csv")
    hybrid_comparisons = read_path(hybrid_base + "comparisons.csv", "csv")
    hybrid_selections = read_path(hybrid_base + "selections.csv", "csv")
    hybrid_audit = read_path(hybrid_base + "coverage_audit.json")
    hybrid_replay = read_path(hybrid_base + "audit.json")
    read_path(hybrid_base + "REPORT.md", "text")
    if hybrid_audit["status"] != "passed" or hybrid_replay["new_supervised_fits"] != 480:
        raise ValueError("Completed hybrid audit required")
    hybrid_primary = one(hybrid_comparisons, contrast="R5_hybrid_selector-R5_standard", min_rows=20)
    hybrid_sensitivity = one(hybrid_comparisons, contrast="R5_hybrid_selector-R5_standard", min_rows=50)
    hybrid_standard = one(hybrid_scores, arm="R5_standard", min_rows=20)
    hybrid_selected = one(hybrid_scores, arm="R5_hybrid_selector", min_rows=20)
    if not math.isclose(float(hybrid_selected["median_spearman"])-float(hybrid_standard["median_spearman"]), float(hybrid_primary["delta"]), abs_tol=1e-12):
        raise ValueError("Hybrid score/difference mismatch")
    if not all(float(r["ci_low"]) < 0 < float(r["ci_high"]) for r in (hybrid_primary,hybrid_sensitivity)):
        raise ValueError("Hybrid inconclusive statement requires review")
    r5_choices = [r for r in hybrid_selections if r["representation"] == "R5"]
    if len(r5_choices)!=5 or sum(r["hybrid_component"]=="standard" for r in r5_choices)!=3 or sum(r["hybrid_component"]=="blend" and float(r["blend_weight"])==.25 for r in r5_choices)!=2:
        raise ValueError("Hybrid fallback statement requires review")
    hybrid_rows = [(label,one(hybrid_comparisons,contrast=f"R5_{arm}-R5_standard",min_rows=20)) for label,arm in
        (("Regression tree","tree_rmse"),("Ranking tree","tree_rank"),("Tree-leaf MLP","leaf_mlp"),("PCA MLP","pca_mlp"),("Quantile MLP","quantile_mlp"))]
    hybrid_html = "\n".join(f'<tr><th scope="row">{label}</th><td class="num">{interval(row)}</td></tr>' for label,row in hybrid_rows)
    label_base = "reports/finalization/label_efficiency/"
    label_curves = read_path(label_base + "absolute_curves.csv", "csv")
    label_comparisons = read_path(label_base + "comparisons.csv", "csv")
    label_audit = read_path(label_base + "coverage_audit.json")
    read_path(label_base + "REPORT.md", "text")
    read_path(label_base + "reporting_metadata.json")
    final_cost = read_path("reports/finalization/resources/final_cloud_closeout.json")
    if label_audit["status"] != "passed":
        raise ValueError("Completed label-budget audit required")
    label_primary = one(label_comparisons, role="primary", min_rows=20, fraction=25)
    label_sensitivity = one(label_comparisons, min_rows=50, fraction=25)
    label_rows = []
    for fraction in (25, 50, 100):
        r0 = float(one(label_curves, min_rows=20, fraction=fraction, representation="R0")["mean_score"])
        r5 = float(one(label_curves, min_rows=20, fraction=fraction, representation="R5")["mean_score"])
        label_rows.append((fraction,r0,r5,r5-r0))
    if not math.isclose(label_rows[0][3]-label_rows[2][3],float(label_primary["delta"]),abs_tol=1e-12):
        raise ValueError("Label-budget primary arithmetic mismatch")
    if float(label_primary["ci_low"]) <= 0 or not (label_rows[0][1]<label_rows[2][1] and label_rows[0][2]<label_rows[2][2]):
        raise ValueError("Label-budget interpretation requires review")
    label_html = "\n".join(f'<tr><th scope="row">{fraction}%</th><td class="num">{fmt(r0)}</td><td class="num">{fmt(r5)}</td><td class="num">{fmt(gap,True)}</td></tr>' for fraction,r0,r5,gap in label_rows)
    read_path(ranking_base + "protocol.json")
    read_path(ranking_base + "REPORT.md", "text")
    inference = read_path("reports/finalization/inference/distribution_manifest.json")
    read_path("reports/finalization/inference/README.md", "text")
    read_path("docs/ROUND2_INFERENCE.md", "text")
    read_path("reports/finalization/INDEPENDENT_ASSAY_BRIEF.md", "text")
    read_path("reports/finalization/FINAL_REPORT.md", "text")
    read_path("reports/finalization/SUBMISSION_BRIEF.md", "text")
    read_path("reports/finalization/transfer_controls/SPECIFICATION.md", "text")
    if ranking_audit["status"] != "passed" or ranking_audit["new_scientific_fits"] != 840 or ranking_audit["report_invocations"] != 1:
        raise ValueError("Ranking completion audit is missing or incomplete")
    ranking_rows = []
    for label, contrast in (("R5 mixed ranking loss − R5 MSE", "R5_ranked-R5_mse"),
                           ("R0 mixed ranking loss − R0 MSE", "R0_ranked-R0_mse"),
                           ("Loss × representation interaction", "deltaR5-deltaR0")):
        row = one(ranking_comparisons, contrast=contrast, min_rows=20)
        for key in ("delta", "ci_low", "ci_high"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite ranking comparison")
        expected = (one(ranking_scores, arm="R5_ranked", min_rows=20), one(ranking_scores, arm="R5_mse", min_rows=20),
                    one(ranking_scores, arm="R0_ranked", min_rows=20), one(ranking_scores, arm="R0_mse", min_rows=20))
        r5, m5, r0, m0 = [float(r["median_spearman"]) for r in expected]
        delta = r5-m5 if contrast.startswith("R5_") else r0-m0 if contrast.startswith("R0_") else (r5-m5)-(r0-m0)
        if not math.isclose(delta, row["delta"], abs_tol=1e-12):
            raise ValueError("Ranking scores and paired difference disagree")
        if not row["ci_low"] <= 0 <= row["ci_high"]:
            raise ValueError("Ranking uncertainty statement needs review")
        ranking_rows.append((label, row))
    ranking_primary = ranking_rows[0][1]
    ranking_sensitivity = one(ranking_comparisons, contrast="R5_ranked-R5_mse", min_rows=50)
    if len(ranking_selections) != 10 or any(row["deployment"] != "mse" for row in ranking_selections if row["representation"] == "R5"):
        raise ValueError("Ranking fallback statement needs review")
    ranking_html = "\n".join(f'<tr><th scope="row">{label}</th><td class="num">{interval(row)}</td></tr>' for label, row in ranking_rows)
    if inference["actual_fits"] != 31 or inference["cloud_calls"] != 0 or not all(a["roundtrip_prediction_exact"] for a in inference["artifacts"]):
        raise ValueError("Portable inference completion needs review")
    ridge = one(inference["artifacts"], candidate="R4_ridge_100")
    neural = one(inference["artifacts"], candidate="R5_mlp_0.01")
    display_arms = [
        ("mlp_log", "Original 30-member sequence MLP"),
        ("sequence_selector", "Competitive sequence selector"),
        ("R2_mlp", "R2 chain-mean MLP"),
        ("R4_ridge", "R4 position-aware ridge"),
        ("R4_mlp", "R4 position-aware MLP"),
        ("R5_mlp", "R5 contact-pooled MLP"),
        ("R5_shuffled_mlp", "R5 shuffled-membership MLP"),
        ("R5_position9x34_mlp", "Full 9×34 positional MLP"),
        ("embedding_selector", "Prespecified embedding-or-original selector"),
    ]
    score_rows = [one(scores, procedure=arm, panel_min_rows=20, scope="pooled_outer_held") for arm, _ in display_arms]
    score_html = "\n".join(f'<tr class="{"row-best" if arm == "embedding_selector" else ""}"><th scope="row">{label}</th><td class="num">{fmt(row["median_per_allele_spearman"])}</td></tr>' for (arm, label), row in zip(display_arms, score_rows))
    controls_html = "\n".join(f'<tr><th scope="row">{label}</th><td class="num">{interval(row)}</td></tr>' for label, row in controls)
    transfer_html = "\n".join(f'<tr><th scope="row">{label}</th><td class="num">{fmt(row["candidate_score"])}</td><td class="num">{fmt(row["baseline_score"])}</td><td class="num">{interval(row)}</td></tr>' for label, row in zip(("HLA-held-out", "Joint-held-out"), transfer_rows))
    cohort = cache["cohort"]
    total = costs["conservative_total_including_full_engineering_allowance_usd"]
    if not primary["ci_low"] > 0 or primary["ci_low"] > .05 or total > costs["incremental_limit_usd"]:
        raise ValueError("Evidence no longer supports the authored assessment; review the copy")
    if any(float(r["max_abs_prediction_difference"]) != 0 for r in repeats["execution_differences"]):
        raise ValueError("Execution-repeat statement needs review")
    body = f'''
  <section id="round2" aria-labelledby="round2-title">
    <div class="shead">
      <p class="eyebrow">Audited round 2 · 12 October 2026 · development evidence</p>
      <h2 class="title" id="round2-title">Frozen interface embeddings add useful ranking information.</h2>
      <p class="lede">In peptide-cluster holdout, the prespecified embedding-or-original selector improves on a competitive sequence selector. Independent confirmation remains open.</p>
    </div>
    <div class="finding">
      <div><p class="eyebrow">Primary paired difference in Spearman</p><p class="gain">{fmt(primary["delta"], True)}</p><p class="interval">95% interval [{fmt(primary["ci_low"], True)}, {fmt(primary["ci_high"], True)}]</p></div>
      <div class="finding-reading"><span class="chip pos">Positive conditional development gain</span><p>The lower bound is above zero, but below <b>+0.05</b>. The prespecified worthwhile-gain criterion is <b>not met</b>.</p><p>Embedding selector <b>{fmt(primary["candidate_score"])}</b> · competitive sequence selector <b>{fmt(primary["baseline_score"])}</b>.</p></div>
    </div>
    <dl class="spec">
      <div><dt>Original training rows</dt><dd>{cohort["n_rows"]:,}</dd></div><div><dt>Peptide clusters</dt><dd>{cohort["n_clusters"]:,}</dd></div>
      <div><dt>HLA alleles</dt><dd>{cohort["n_alleles"]}</dd></div><div><dt>Outer folds</dt><dd>5</dd></div><div><dt>Primary eligible panel</dt><dd>{primary["n_alleles"]} alleles</dd></div>
    </dl>
    <p class="source-line">{link("results/main/REPORT.md", "Main assessment")} · {link("results/main/paired_comparisons.csv", "Saved paired comparisons")} · {link("resources/embedding_completion_audit.json", "Completion audit")}</p>
    <details class="evidence-detail"><summary>Declared procedures and matched controls</summary>
      <p>Scores describe declared procedures; the highest component score does not select a new winner. The inner selector retained the original fallback and chose R5 MLP in four folds and R4 ridge in one.</p>
      <div class="scroller" tabindex="0" role="region" aria-label="Round 2 procedure scores"><table><caption>Peptide-held-out · ≥20 rows per allele · pooled median per-allele Spearman</caption><thead><tr><th scope="col">Procedure</th><th scope="col">Score</th></tr></thead><tbody>{score_html}</tbody></table></div>
      <div class="scroller" tabindex="0" role="region" aria-label="Matched positional controls"><table><caption>Matched MLP controls · paired whole-cluster 95% intervals</caption><thead><tr><th scope="col">Declared contrast</th><th scope="col">Difference [95% interval]</th></tr></thead><tbody>{controls_html}</tbody></table></div>
      <p>Positions help within these frozen compressed representations. The particular crystal-derived contact membership has no established benefit. Weak chain averages do not reject internal embeddings generally.</p>
    </details>
  </section>

  <section id="ranking-followup" aria-labelledby="ranking-followup-title">
    <div class="shead"><p class="eyebrow">Completed matched follow-up · 12 October 2026</p><h2 class="title" id="ranking-followup-title">Ranking loss adds no established gain.</h2><p class="lede">The mixed MSE + within-allele ranking objective slightly raises the R5 point estimate. Its paired interval crosses zero, and the inner deployment selector retains MSE in all five R5 folds.</p></div>
    <div class="scroller" tabindex="0" role="region" aria-label="Matched ranking-loss comparisons"><table><caption>Original-training peptide-cluster assessment · ≥20 rows per allele</caption><thead><tr><th scope="col">Frozen comparison</th><th scope="col">Difference [paired 95% interval]</th></tr></thead><tbody>{ranking_html}</tbody></table></div>
    <p class="note">R5 mixed-loss MLP scores <b>{fmt(r5)}</b>, compared with <b>{fmt(m5)}</b> for its matched MSE head. The ≥50-row sensitivity gives {interval(ranking_sensitivity)}. These comparisons do not establish a ranking-loss benefit or equivalence.</p>
    <details class="evidence-detail"><summary>Matched recipe, audit and limits</summary><p>R0 onehot and R5 Boltz features use the same 64-unit head, MSE stopping, regularisation grid and final 30-member ensemble. Positive ranking weights 0.1, 0.3 and 1 were selected inside fitting partitions only. All 840 new fits, 480 reused MSE members, fresh inner predictions and gradient-only scaler statistics were audited; both real zero-weight checks reproduced the cached MSE member exactly.</p><p>The interaction subtracts the two loss differences on the same 2,000 paired cluster draws. Intervals condition on fitted predictions and inner selections. Screening uses two members and final prediction 30; feature dimensions and parameter counts still differ. MSE anchors prediction scale. This outcome triggers no extra grid, new stopping rule or transfer expansion.</p></details>
    <p class="source-line">{path_link(ranking_base + "REPORT.md", "Ranking readout")} · {path_link(ranking_base + "comparisons.csv", "All paired comparisons")} · {path_link(ranking_base + "completion_audit.json", "Independent arithmetic and artifact audit")}</p>
  </section>

  <section id="portable-inference" aria-labelledby="portable-inference-title">
    <div class="shead"><p class="eyebrow">Completed portable prediction package</p><h2 class="title" id="portable-inference-title">Use the trained heads with verified Boltz features.</h2><p class="lede">Locked demonstration recipes export R4 ridge (alpha 100) and a 30-member R5 MLP (L2 0.01). Save/reload predictions match exactly.</p></div>
    <p class="col">Warm single-row cached-head medians are <b>{ridge["single_seconds"]["median"]*1000:.2f} ms</b> for ridge and <b>{neural["single_seconds"]["median"]*1000:.2f} ms</b> for the MLP on the recorded host. These are performance-only measurements on identified training-cache inputs, excluding Boltz extraction, MSA generation, cache reads and startup.</p>
    <p class="note">The package accepts compatible verified compact features. Novel peptide/HLA inputs need real feature extraction; this page does not simulate sequence-to-prediction output. The two recipes were locked before new outcomes and are deployment demonstrations, without a new accuracy or universal-winner claim.</p>
    <p class="source-line">{path_link("docs/ROUND2_INFERENCE.md", "Input contract and prediction commands")} · {path_link("reports/finalization/inference/README.md", "Exports and measured timing")} · {path_link("reports/finalization/inference/distribution_manifest.json", "Model artifact hashes")}</p>
  </section>

  <section id="round2-methods" aria-labelledby="round2-methods-title">
    <div class="shead"><p class="eyebrow">How to read the evidence</p><h2 class="title" id="round2-methods-title">Select inside, assess outside.</h2></div>
    <ol class="method-flow"><li><b>Freeze features and partitions.</b> All original training rows remain in five outer peptide-cluster folds; assessed clusters stay outside fitting, stopping and scaling.</li><li><b>Select with inner folds.</b> Regularisation and procedure choices use inner outcomes only, with an original-sequence fallback.</li><li><b>Pair the comparisons.</b> Subtract the two pooled medians of per-allele Spearman on the same eligible panel, then resample whole peptide clusters with 2,000 shared draws.</li></ol>
    <div class="note flag"><p><b>The intervals condition on fitted predictions.</b> They exclude selection and execution uncertainty. They are neither medians of allele differences nor medians of fold scores. The primary ≥20-row panel covers {int(primary["n_rows_eligible"]):,} rows; the ≥50-row sensitivity gives {interval(sensitivity)} across {sensitivity["n_alleles"]} alleles.</p><p>These data and earlier benchmark/test results influenced development history. Preserving the original test in this execution does not create an independent cohort; the original test has not been rescored or used to select these round 2 procedures.</p></div>
  </section>

  <section id="round2-transfer" aria-labelledby="round2-transfer-title">
    <div class="shead"><p class="eyebrow">Completed stronger transfer controls · 30 cells · 10,830 retained fits</p><h2 class="title" id="round2-transfer-title">Transfer gains survive matched head recipes.</h2><p class="lede">Boltz representations outperform their sequence counterparts under matched ridge and MLP recipes. Five HLA folds and all 25 joint intersections passed the full audit.</p></div>
    <div class="scroller" tabindex="0" role="region" aria-label="Matched transfer results"><table><caption>≥20 rows per allele · crossed peptide-cluster × HLA-group paired 95% intervals</caption><thead><tr><th scope="col">Matched contrast</th><th scope="col">Difference [95% interval]</th></tr></thead><tbody>{extension_matched_html}</tbody></table></div>
    <p class="note">All matched contrasts remain positive on the ≥50-row panel, under peptide-only and HLA-only uncertainty, and after deleting any one of the 35 HLA groups. Different feature dimensions, pretraining and representation construction still prevent causal attribution specifically to structure. These are conditional development results.</p>
    <details class="evidence-detail"><summary>Both sequence references, including the stronger original joint baseline</summary>
      <div class="scroller" tabindex="0" role="region" aria-label="Original and expanded sequence references"><table><caption>≥20 rows per allele · pooled median per-allele Spearman</caption><thead><tr><th scope="col">Regime / sequence reference</th><th scope="col">Sequence</th><th scope="col">Embedding selector</th><th scope="col">Difference [crossed 95% interval]</th></tr></thead><tbody>{extension_reference_html}</tbody></table></div>
      <p>The expanded sequence selector changes the original by {interval(extension_sequence[0])} in HLA and {interval(extension_sequence[1])} in joint holdout. Its joint point estimate is lower than the original, so presenting only the expanded comparator would exaggerate the advantage. Outer outcomes do not select a replacement procedure.</p>
    </details>
    <details class="evidence-detail"><summary>Scope and uncertainty</summary><p>The extension adds sequence regression/centred trees, a fixed ranker and matched R0 ridge/MLP heads. Inner selection, stopping and scaling obey each holdout regime. Its 264 comparison rows cover both panels and three uncertainty modes; all 12 bootstrap tables have 2,000 draws, and 3,080 rows cover every group-deletion contrast.</p><p>Intervals condition on fitted predictions and selections; group deletion does not retrain. Crossed resampling is a dependence sensitivity, not independent confirmation. HLA sequence groups are not validated functional distance. The ranker retains an inner-to-final scale-transport limitation. These transfer findings do not retrospectively change the main worthwhile-gain criterion.</p></details>
    <p class="source-line">{path_link(extension_base + "REPORT.md", "Complete stronger-transfer report")} · {path_link(extension_base + "comparisons.csv", "All comparisons")} · {path_link(extension_base + "coverage_audit.json", "Independent coverage audit")} · {link("transfer/results/README.md", "Historical narrower assessment")}</p>
  </section>

  <section id="hybrid-followup" aria-labelledby="hybrid-followup-title">
    <div class="shead"><p class="eyebrow">Completed hybrid comparison · 480 fits · 16 arms</p><h2 class="title" id="hybrid-followup-title">Keep the simpler embedding head.</h2><p class="lede">Trees, blending and preprocessing did not establish an improvement over the matched standard Boltz MLP.</p></div>
    <p class="col">The inner-selected R5 hybrid scores <b>{fmt(hybrid_selected["median_spearman"])}</b> versus <b>{fmt(hybrid_standard["median_spearman"])}</b>: <b>{interval(hybrid_primary)}</b>. Its ≥50-row sensitivity is {interval(hybrid_sensitivity)}. Both are inconclusive, not equivalence tests.</p>
    <p class="note">The selector keeps the standard head in three folds and a 25% regression-tree blend in two. Selector and blend predictions coincide; they are one finding. Both arms use two members; historical 30-member heads and the stronger main sequence selector are descriptive references, not these matched controls.</p>
    <div class="scroller" tabindex="0" role="region" aria-label="R5 hybrid component comparisons"><table><caption>R5 challenger minus standard MLP · ≥20 panel · exploratory paired 95% intervals</caption><thead><tr><th scope="col">Challenger</th><th scope="col">Difference [95% interval]</th></tr></thead><tbody>{hybrid_html}</tbody></table></div>
    <p class="note">These results concern the frozen recipes, not all possible trees or preprocessing methods. All saved held predictions and 328 stopping-MSE checks replayed exactly under the recorded numerical backend. The separate 12-fit pilot was unscored. Independent assay confirmation remains open.</p>
    <p class="source-line">{path_link(hybrid_base + "REPORT.md", "Full hybrid study")} · {path_link(hybrid_base + "comparisons.csv", "Every frozen comparison")} · {path_link(hybrid_base + "coverage_audit.json", "Independent coverage audit")} · {path_link("reports/finalization/cloud/BACKEND_REPLAY.md", "Numerical replay disclosure")}</p>
  </section>

  <section id="label-budget" aria-labelledby="label-budget-title">
    <div class="shead"><p class="eyebrow">Completed label-budget sensitivity · 120 new fits</p><h2 class="title" id="label-budget-title">Boltz loses less performance when labels shrink.</h2><p class="lede">Under the same fixed MLP recipe, the representation advantage grows at lower training-label budgets. Both models still worsen.</p></div>
    <div class="scroller" tabindex="0" role="region" aria-label="Label-budget results"><table><caption>Mean subset scores · ≥20 panel · stopping labels included</caption><thead><tr><th scope="col">Fitting-cluster budget</th><th scope="col">Sequence MLP</th><th scope="col">Boltz MLP</th><th scope="col">Gap</th></tr></thead><tbody>{label_html}</tbody></table></div>
    <p class="col">The primary increase in the gap at 25% versus 100% is <b>{interval(label_primary)}</b>. The ≥50-row sensitivity is {interval(label_sensitivity)}. The 50% budget agrees.</p>
    <p class="note">Three predetermined nested subsets are averaged as scores, never as prediction ensembles. Budgets count peptide clusters and include both fitting and stopping labels. Intervals condition on fixed models and subsets; recipes were nominated using earlier full-data development. This is relative robustness, not prospective assay savings or a comparison with the stronger main sequence selector.</p>
    <p class="source-line">{path_link(label_base + "REPORT.md", "Label-budget report")} · {path_link(label_base + "absolute_curves.csv", "All absolute scores and intervals")} · {path_link(label_base + "comparisons.csv", "Paired gap comparisons")} · {path_link(label_base + "coverage_audit.json", "Independent table and subset audit")}</p>
  </section>

  <section id="focused-followups" aria-labelledby="focused-followups-title">
    <div class="shead"><p class="eyebrow">New bounded development tranche · 13 October 2026</p><h2 class="title" id="focused-followups-title">Context, target labels and repaired energy.</h2><p class="lede">Three frozen primary questions extend the development evidence. Their different assessment populations and recipes prevent direct subtraction from the main score.</p></div>
    <div class="scroller" tabindex="0" role="region" aria-label="New follow-up study status"><table><caption>Current evidence status · original-training outcomes only</caption><thead><tr><th scope="col">Question</th><th scope="col">Frozen primary comparison</th><th scope="col">Status</th></tr></thead><tbody>
      <tr><th scope="row">ProteinMPNN context</th><td>R5 + peptide-only + full context minus R5 + peptide-only</td><td>900 fits complete and independently audited; increment beyond Boltz inconclusive</td></tr>
      <tr><th scope="row">Few-shot HLA adaptation</th><td>R5 at 20 target labels per allele minus R5 at zero labels</td><td>Audited positive adaptation gain; 450 source fits and 9,180 adapters complete</td></tr>
      <tr><th scope="row">Repaired FoldX complementarity</th><td>R5 + repaired energy and interface descriptors minus R5</td><td>4,449 descriptors and 300 new fits complete and independently audited; R5 increment inconclusive</td></tr>
    </tbody></table></div>
    <p class="note">The third-primary decision was recorded at 12:35 UTC before the manager inspected new efficacy results. Any tranche-wide success claim uses 98.333333% intervals for all three primaries. Individual 95% and historically specified 97.5% intervals remain visible, with the latter superseded for the three-question family. Few-shot adaptation is the sole positive primary under this family correction. Technical extraction success is not evidence of half-life prediction benefit.</p>
    <details class="evidence-detail"><summary>Primary method references and pinned model artifacts</summary><p><b>ProteinMPNN:</b> <a href="https://doi.org/10.1126/science.add2187">Dauparas et al., 2022</a> · <a href="https://github.com/dauparas/ProteinMPNN">official implementation</a>. Full-backbone vanilla checkpoint <code>{html.escape(inverse_methods["checkpoint"])}</code>, hash {inverse_methods["checkpoint_sha256"][:12]}…; scoring uses zero added coordinate noise and 16 explicit common peptide orders.</p><p><b>FoldX:</b> <a href="https://doi.org/10.1093/bioinformatics/btz184">FoldX 5.0 (Delgado et al., 2019)</a> · <a href="https://doi.org/10.1016/S0022-2836(02)00442-4">original force field (Guerois et al., 2002)</a> · <a href="https://doi.org/10.1093/nar/gki387">complex-analysis methods (Schymkowitz et al., 2005)</a>. The recorded Linux executable is hash-pinned ({foldx_methods["binary_sha256"][:12]}…); its finer semantic release is not recorded. RepairPDB then AnalyseComplex supplies 29 energy and interface/clash/residue descriptors.</p><p><b>Boltz-2:</b> <a href="https://doi.org/10.1101/2025.06.14.659707">Passaro et al., 2025, technical preprint</a> · <a href="https://github.com/jwohlwend/boltz">official implementation</a>. Reused features use package <code>{html.escape(boltz_methods["package"])}</code>, weight revision <code>{boltz_methods["checkpoint_revision"]}</code> and checkpoint hash {boltz_methods["checkpoint_sha256"][:12]}…. Package version, model family and weight revision are separate identifiers.</p><p>Sequence compatibility and computed equilibrium energies do not directly measure dissociation half-life. The supervised kinetic-ranking comparisons determine their predictive value here. The {path_link(followup_base + "ASSESSMENT_PRECHECK.md", "pre-results source audit")} found no blocking defect before head fitting; all three original studies now have completed independent audits. The selected full-cohort extension also passed completed-output independent auditing.</p></details>
    <p class="source-line">{path_link(followup_base + "REPORT.md", "Current follow-up report")} · {path_link(followup_base + "THIRD_PRIMARY_AMENDMENT.md", "Pre-efficacy third-primary amendment")} · {path_link(followup_base + "foldx/technical_receipt.json", "FoldX technical receipt")}</p>
  </section>

  <section id="inverse-followup" aria-labelledby="inverse-followup-title">
    <div class="shead"><p class="eyebrow">Completed ProteinMPNN assessment · balanced 4,449 pairs · 900 fits</p><h2 class="title" id="inverse-followup-title">The context increment beyond Boltz remains unresolved.</h2><p class="lede">Native-peptide compatibility with full-complex context does not establish an additional gain over its matched Boltz controls. A small useful increment remains possible; these comparisons do not establish equivalence.</p></div>
    <div class="scroller" tabindex="0" role="region" aria-label="ProteinMPNN primary and net comparison"><table><caption>Matched two-member MLP · ≥20 panel · 69 alleles</caption><thead><tr><th scope="col">Comparison</th><th scope="col">Difference [individual 95% interval]</th></tr></thead><tbody><tr><th scope="row">Primary: R5 + P + C minus R5 + P</th><td class="num">{interval(inverse_primary)}</td></tr><tr><th scope="row">Exploratory net: R5 + P + C minus R5</th><td class="num">{interval(inverse_net)}</td></tr></tbody></table></div>
    <p class="col">The required <b>98.333333% three-primary family interval is [{fmt(inverse_family["ci_low"],True)}, {fmt(inverse_family["ci_high"],True)}]</b>. Absolute MLP scores are {fmt(inverse_r5)} for R5, {fmt(inverse_r5p)} for R5 + P and {fmt(inverse_r5pc)} for R5 + P + C. The original CSV's 97.5% fields are historical two-primary intervals, superseded for tranche-wide claims.</p>
    <details class="evidence-detail"><summary>All six arms, sensitivities and completed audit</summary>
      <div class="scroller" tabindex="0" role="region" aria-label="ProteinMPNN absolute MLP and ridge scores"><table><caption>All absolute scores · ≥20 panel · pooled median per-allele Spearman</caption><thead><tr><th scope="col">Arm</th><th scope="col">Two-member MLP</th><th scope="col">Ridge sensitivity</th></tr></thead><tbody>{inverse_html}</tbody></table></div>
      <p>P is nine position-wise mean native-residue log probabilities on the peptide backbone alone; C uses full complex context. Each position averages over the same 16 explicit decoding orders. P's backbone was predicted using native peptide/HLA context. Removing chains changes graph neighbors too, so the ablation does not isolate structural causality.</p><p>The ≥50 sensitivity has 68 alleles and gives {interval(inverse_sensitivity)} for the MLP context increment. R0 context controls are positive exploratory comparisons; R5 ridge increments remain inconclusive. All 28 contrasts and 24 absolute score rows remain in the source tables; no secondary learner/panel replaces the primary. These balanced-panel recipes differ from the main competitive selector and strict joint few-shot ridge design.</p><p>All 600 MLP members and 300 ridge fits passed independent score/selection/ancestry checks. Parser recovery completed all 4,449 identities, preserving 4,163 inherited record hashes and archiving the failed attempt. Fitted states are not saved; no fitted-weight replay, refitting or bootstrap regeneration is claimed. Conditional intervals exclude new assay and repeated fitting/selection variation.</p>
      <p><b>Post-hoc descriptive influence.</b> With original eligibility fixed, every single-allele or outer-held prediction-fold deletion keeps the MLP net point positive, from +0.016474 to +0.028273 across allele deletions and +0.011860 to +0.040450 across fold deletions. Individual allele effects vary: 43 rise and 26 fall. Ridge's net point remains negative under all single-allele deletions and its fold deletions disagree in sign. No models are refitted, no new interval/testing or inference upgrade is introduced, and the original MLP uncertainty remains inconclusive. {path_link(influence_base + "summary.json", "Descriptive source summary")}</p>
    </details>
    <p class="source-line">{path_link(followup_base + "REPORT.md", "Full follow-up readout")} · {path_link(inverse_base + "comparisons.csv", "All declared contrasts")} · {path_link(inverse_base + "pooled_scores.csv", "All absolute scores")} · {path_link(inverse_base + "independent_audit.json", "Independent assessment audit")} · {path_link(inverse_base + "primary_family3_interval.json", "Current family interval")}</p>
  </section>

  <section id="energy-followup" aria-labelledby="energy-followup-title">
    <div class="shead"><p class="eyebrow">Completed repaired descriptors · balanced 4,449 pairs · 300 new fits</p><h2 class="title" id="energy-followup-title">Repaired descriptors do not establish extra Boltz value.</h2><p class="lede">R5 + E minus R5 is {interval(energy_primary)} under its individual 95% interval. Its 98.333333% family interval is [{fmt(energy_family["ci_low"],True)}, {fmt(energy_family["ci_high"],True)}]. A useful increment remains possible; this is not equivalence.</p></div>
    <figure><p class="figure-scroll-hint">Scroll sideways to see the full figure.</p><div class="figure-scroll" tabindex="0" role="region" aria-label="All balanced-panel MLP descriptors and two primary family intervals"><img class="figure-img scientific-figure" src="figures/descriptor_comparison.svg" width="1040" height="560" loading="lazy" alt="All eight balanced-panel MLP scores. ProteinMPNN context adds +0.023960 and repaired descriptors +0.014991; both 98.333333% family intervals span zero."></div><figcaption><b>Same balanced panel, distinct assessment.</b> All eight matched MLP arms and two primary family intervals; absolute-score intervals are not shown. P is peptide-only compatibility, C full-complex compatibility and E the 29 repaired energy/interface/clash/residue descriptors. These recipes differ from the main selector and strict joint few-shot ridge study. <a href="figures/descriptor_comparison.svg">SVG export</a> · <a href="figures/descriptor_comparison.png">PNG export</a> · {path_link(figure_base+"descriptor_comparison.json","Exact rows and source hashes")}</figcaption></figure>
    <details class="evidence-detail"><summary>All repaired-descriptor arms, sensitivities and completed audits</summary><div class="scroller" tabindex="0" role="region" aria-label="Repaired descriptor MLP and ridge absolute scores"><table><caption>≥20 panel · 69 alleles · pooled median per-allele Spearman</caption><thead><tr><th scope="col">Arm</th><th scope="col">Two-member MLP</th><th scope="col">Ridge sensitivity</th></tr></thead><tbody>{energy_html}</tbody></table></div><p>R5 ridge adds {interval(energy_ridge)}; the ≥50 MLP sensitivity is {interval(energy_sensitivity)}. Both learners and both eligible panels remain inconclusive. A positive exploratory weak R0 ridge increment does not establish added incumbent R5 value. All 16 absolute score rows and 12 contrasts remain linked.</p><p>The 73 interrupted cases were recovered with the same frozen recipe, preserving 4,376 inherited success hashes and original attempts. Independent recovery and exact feature-matrix audits passed before assessment. All 200 new MLP members and 100 ridge fits passed score/selection/ancestry auditing, with R0/R5 baselines reused exactly and zero baseline refits. Independent checks did not refit or regenerate bootstrap draws. Fitted states are not saved; fitted-weight replay is not claimed.</p></details>

    <details class="evidence-detail"><summary>Full-cohort coverage and density: label-free metadata</summary><p>The same 75 alleles grow from 4,449 to 19,716 rows (4.43×), and 2,082 to 3,587 inherited peptide clusters (1.72×), adding 1,505 clusters. Of the additional rows, 11,656 belong to previously covered clusters and 3,611 to new clusters. These are coverage/density counts, not effective independent observations. The metadata's ≥20/≥50 thresholds count rows only and do not establish assay-rankability eligibility. No labels, predictions or ProteinMPNN scores were read. {path_link(full_metadata_base+"cohort_metadata_comparison.json","Source metadata comparison")}</p></details>
    <p class="source-line">{path_link(energy_base+"comparisons.csv","All declared contrasts")} · {path_link(energy_base+"pooled_scores.csv","All absolute scores")} · {path_link(energy_base+"INDEPENDENT_FINAL_REVIEW.json","Final independent review")} · {path_link(energy_base+"primary_family3_interval.json","Current family interval")} · {path_link(followup_base+"REPORT.md","Full methods and limitations")}</p>
  </section>

  {full_cohort_body}

  <section id="fewshot-followup" aria-labelledby="fewshot-followup-title">
    <div class="shead"><p class="eyebrow">Completed few-shot adaptation · 68 target alleles · 3 support draws</p><h2 class="title" id="fewshot-followup-title">A few target-allele measurements improve ranking.</h2><p class="lede">Twenty support labels per allele raise the fixed R5 source ridge score from {fmt(fewshot_zero)} to {fmt(fewshot_twenty)} on peptide families excluded from source fitting.</p></div>
    <div class="finding"><div><p class="eyebrow">R5 k=20 minus zero-shot</p><p class="gain">{fmt(fewshot_primary["delta"], True)}</p><p class="interval">Individual crossed 95% interval [{fmt(fewshot_primary["ci_low"], True)}, {fmt(fewshot_primary["ci_high"], True)}]</p></div><div class="finding-reading"><span class="chip pos">Positive conditional adaptation gain</span><p>Three-primary family interval, 98.333333%: <b>[{fmt(fewshot_family["ci_low"], True)}, {fmt(fewshot_family["ci_high"], True)}]</b>.</p><p>Original two-primary 97.5% interval [{fmt(fewshot_primary["family_ci_low"], True)}, {fmt(fewshot_primary["family_ci_high"], True)}], retained historically and superseded for tranche-wide claims.</p></div></div>
    <figure>
      <p class="figure-scroll-hint">Scroll sideways to see the full figure.</p>
      <div class="figure-scroll" tabindex="0" role="region" aria-label="Three few-shot curves and primary family interval"><img class="figure-img scientific-figure" src="figures/fewshot_curves.svg" width="1020" height="530" loading="lazy" alt="R0, R5 and reduced sequence-anchor absolute curves at zero, five, ten and twenty labels. R5 increases from 0.529583 to 0.576337; primary gain +0.046754, with a 98.333333% family interval from +0.014042 to +0.119457."></div>
      <figcaption><b>Strict joint transfer.</b> Three fixed support-draw scores are averaged on 68 target alleles. Source fitting excludes target HLA groups and support/query peptide families. The sequence comparator is a reduced four-member historical anchor. The primary family interval is conditional development evidence; curve intervals are not shown. <a href="figures/fewshot_curves.svg">SVG export</a> · <a href="figures/fewshot_curves.png">PNG export</a> · {path_link(figure_base + "fewshot_curves.json", "Exact values and source hashes")}</figcaption>
    </figure>
    <div class="scroller" tabindex="0" role="region" aria-label="Few-shot absolute learning curves"><table><caption>Mean draw scores · fixed 68-allele panel · median per-allele Spearman</caption><thead><tr><th scope="col">Labels per allele</th><th scope="col">Target labels per draw</th><th scope="col">R0 ridge</th><th scope="col">R5 ridge</th><th scope="col">Four-member sequence anchor</th></tr></thead><tbody>{fewshot_html}</tbody></table></div>
    <p class="note">Supports are chosen by identity. Within each held-out HLA fold, the union of its target alleles' maximum-support clusters is excluded from that fold's source/query cells at every label budget. Source fitting also excludes both held grouping axes. Query rows stay fixed across budgets within each draw; three draw scores are averaged, never predictions. Target tuning and stopping use zero labels. Support labels are reused across five query cells and count once per draw/budget. Source stopping labels count; source availability ranges from 11,213–11,950 labels and the actual sequence gradient/stopping union from 10,213–11,055.</p>
    <details class="evidence-detail"><summary>Matched interpretation, sensitivities and audit</summary><p>The sequence anchor uses the previously nominated two-layer one-hot/BLOSUM recipe with four source members. It is a reduced historical anchor, distinct from the full main competitive selector. Strict joint exclusions, query population and target-label access differ from the main study; these scores cannot be directly subtracted from its results.</p><p>The ≥20 and ≥50 panels are the identical 68 alleles, so their equality supplies no additional population robustness. Every one of the 35 HLA-group deletions keeps a positive primary point difference, from {fmt(min(primary_deletions), True)} to {fmt(max(primary_deletions), True)}, without refitting. R0 and the sequence anchor gain more in nominal secondary comparisons but start and finish below R5; the crossed representation-by-adaptation interaction remains inconclusive. All declared secondary contrasts and uncertainty modes are retained.</p><p>Cloud completion records exact saved-state replay for 450 source fits and 9,180 adapters. The independent audit checks 825 payload hashes, all 72 draw scores, 108 comparisons, 1,260 group-deletion rows, residual offsets and ancestry. It verifies cloud replay assertions without independent refitting, full matrix replay or bootstrap regeneration. Intervals condition on fixed source fits, adapters and support designs; they do not establish prospective assay savings or independent confirmation.</p></details>
    <div class="note"><p><b>Exploratory post-hoc offset control.</b> Chosen after the positive primary, this diagnostic uses saved fits and the same labels. R5 scores {fmt(offset_scores["R5_k0"])} zero-shot, {fmt(offset_scores["R5_offset20"])} with scalar offsets alone and {fmt(offset_scores["R5_k20"])} with the full adapter. Full minus offset is <b>{interval(offset_full)}</b>; offset minus zero is <b>{interval(offset_zero)}</b>, with nominal unadjusted 95% intervals. Feature-dependent correction adds ranking beyond this control on saved predictions; offset-only benefit remains unresolved. Offsets preserve within-cell ranks but can align pooled predictions across source/query cells. The independent audit checks 44,199 query predictions, 1,020 offset/rank checks and paired-bootstrap arithmetic. No new fits or labels are used; this is not a fourth primary and leaves the three-primary family unchanged. {path_link(offset_base + "REPORT.md", "Diagnostic report")} · {path_link(offset_base + "independent_audit.json", "Independent audit")}</p></div>
    <p class="source-line">{path_link(followup_base + "REPORT.md", "Full follow-up readout")} · {path_link(fewshot_base + "absolute_curves.csv", "All absolute curves")} · {path_link(fewshot_base + "comparisons.csv", "All frozen comparisons")} · {path_link(fewshot_base + "independent_audit.json", "Independent numerical and ancestry audit")} · {path_link(fewshot_base + "primary_family3_interval.json", "Three-primary family interval")}</p>
  </section>

  <section id="round2-complementarity" aria-labelledby="round2-complementarity-title">
    <div class="shead"><p class="eyebrow">Clean nested complementarity</p><h2 class="title" id="round2-complementarity-title">Sequence and interface information can complement each other.</h2></div>
    <p class="col">R4 ridge blended with the exact original reference scores <b>{fmt(blend["candidate_score"])}</b>: {interval(blend)} against the competitive selector. R4 residual correction gives {interval(residual)}. Both support a conditional development gain; neither lower bound exceeds 0.05.</p>
    <p class="note">Assessed clusters never enter base fitting, stopping, ridge selection or residual-target construction. Predictions from other outer folds do not supply stacking inputs. Numerical proximity to the embedding selector is not a tested superiority claim.</p>
    <p class="source-line">{link("complementarity/results/REPORT.md", "Complementarity report")} · {link("resources/complementarity_completion_audit.json", "Completion audit")}</p>
  </section>

  <section id="round2-provenance" aria-labelledby="round2-provenance-title">
    <div class="shead"><p class="eyebrow">Reproducibility, provenance and resources</p><h2 class="title" id="round2-provenance-title">Audited artifacts, with a disclosed launch-source gap.</h2></div>
    <div class="note"><p><b>Locked repeats matched exactly on the recorded platform.</b> Two fresh identical-setting embedding-selector repeats used {repeats["actual_new_model_fits"]} local fits and distinct caches. Prediction and metric differences were zero over all {cohort["n_rows"]:,} rows. This is execution reproducibility, not cross-platform equivalence or an assay-noise ceiling.</p><p><b>The cache fanout wrapper has an inherited provenance gap.</b> Current/committed and launch-authorization SHA-256 digests differ; exact launch-time wrapper bytes cannot be certified. Both digests are preserved in the {link("resources/cache_completion_audit.json", "cache audit")}. The four extraction fingerprint sources, cohort, protocol and local artifact checks match; the discrepancy itself is not evidence of changed features.</p></div>
    <div class="budget-box"><p class="eyebrow">Closed round-two accounting</p><p class="budget-total">${total:.4f} <span>against the ${costs["incremental_limit_usd"]:.0f} ceiling</span></p><p>Shared cache ${costs["cache_runtime_budget_accounting_usd"]:.4f} + sequence ${costs["sequence_conservative_accounting_including_preemptions_usd"]:.4f} + retained engineering allowance ${costs["engineering_full_allowance_retained_usd"]:.2f}.</p><p><b>Not a final invoice or live credit balance.</b> Retaining the engineering allowance does not mean it was spent. Provider snapshots lag and differ, leaving ${costs["metering_difference_unattributed_usd"]:.4f} unattributed; storage is separate. These assessment costs do not establish new-case end-to-end latency. A subsequent portable package measures cached-head latency separately; see the repository inference guide.</p></div>
    <p class="note">The separate finalization phase carries <b>${final_cost["closed_conservative_compute_usd"]:.2f}</b> in conservative closed compute plus a retained <b>${final_cost["overhead_allowance_retained_usd"]:.0f}</b> overhead allowance, within its additional $100 ceiling. All compute and temporary finalization Volumes are closed. Existing durable feature and weight caches remain. {path_link("reports/finalization/resources/final_cloud_closeout.json", "Final resource closeout")}</p>
    <p class="note">The separately authorized 13 October tranche is closed at <b>${followup_cost["conservative_closed_usd"]:.6f}</b> conservatively, with <b>$0 reserved</b>, against its ${followup_cost["inclusive_new_tranche_ceiling_usd"]:.0f} ceiling. Failed attempts are included and the application is stopped. Scientific outputs are preserved locally; the temporary tranche volume is deleted and historical structure caches retained. This is not a provider invoice or live credit balance. {path_link(followup_base+"budget.json","Tranche ledger")} · {path_link(followup_base+"resources/cloud_closeout.json","Cloud closeout")}</p>
    <p class="source-line">{link("resources/embedding_repeat_completion_audit.json", "Embedding repeat audit")} · {link("resources/sequence_repeat_completion_audit.json", "Sequence repeat audit")} · {link("resources/runtime_cost_summary.json", "Runtime/cost accounting")} · {link("resources/transfer_runtime_closeout.json", "Local transfer closeout")}</p>
  </section>

  <section id="round2-confirmation" aria-labelledby="round2-confirmation-title">
    <div class="question"><p class="eyebrow">Still open</p><h2 id="round2-confirmation-title">Independent confirmation.</h2><p class="task-body">The public stability release overlaps the canonical corpus. Conservative exclusions preserve <b>{metadata["rows"]} candidate rows, {metadata["peptides"]} peptides, {len(metadata["allele_counts"])} alleles and {len(metadata["source_pmids"])} cited studies</b>. A completed metadata/methods audit verifies heterogeneous endpoints, temperatures and constructs, with unresolved measurement mapping. The largest single-study/allele stratum has only 23 rows (Strønen 2016, A*02:01) and remains gated; it cannot provide representative multi-allele confirmation. External candidate outcome columns remain reserved. Incidental public-paper result visibility during literature retrieval is disclosed in the audit.</p><p class="task-body">No assay quote, partner commitment or adequately powered independent cohort is secured. The synthetic design supports asking for comparable assay quotes, not a guaranteed power or assay-noise claim. Lock the final procedure and analysis before a one-time independent confirmation.</p><p>{path_link(metadata_base + "PROVENANCE.md", "Completed metadata/methods audit")} · {link("independent_data/README.md", "Original identity audit")} · {link("independent_data/design/README.md", "Conditional assay-design study")} · {link("REPORT.md", "Current round 2 assessment")}</p></div>
  </section>

  <section id="round2-evidence" aria-labelledby="round2-evidence-title">
    <div class="shead"><h2 class="title" id="round2-evidence-title">Inspect the evidence.</h2><p class="lede">This page is a local static submission snapshot. Numeric tables are generated from saved result CSVs, with source SHA-256 hashes in the <a href="round2-data.json">downloadable evidence payload</a>. Reading this page starts no model jobs.</p></div>
    <p class="source-line">{path_link("reports/finalization/FINAL_REPORT.md", "Combined finalisation readout")} · {link("REPORT.md", "Closed round-two assessment")} · {path_link("reports/finalization/INDEPENDENT_ASSAY_BRIEF.md", "Unsent independent-assay brief")} · {path_link("reports/NEXT_RESEARCH_PLAN.md", "Research plan and execution gates")} · <a href="#archive">Original hackathon archive</a></p>
  </section>
'''
    payload = {"schema": "round2-site-snapshot.1", "assessment_date": "2026-10-13", "scope": "development only; original historical test separate; independent confirmation open", "cohort": cohort,
               "primary": primary, "sensitivity": sensitivity, "procedure_scores": score_rows,
               "matched_controls": [{"label": label, "comparison": row} for label, row in controls],
               "complementarity": {"blend": blend, "residual": residual}, "transfer": transfer_rows,
               "transfer_original_reference": original_transfer,
               "ranking_followup": {"comparisons": [{"label": label, "comparison": row} for label, row in ranking_rows],
                                    "sensitivity": ranking_sensitivity, "scores": ranking_scores,
                                    "selections": ranking_selections, "audit_status": ranking_audit["status"]},
               "portable_inference": {"recipes": [ridge, neural], "actual_fits": 31,
                                      "latency_scope": inference["latency_scope"]},
               "transfer_extension_status": "complete; independently audited; development only",
               "transfer_extension": {"matched_heads": [{"label": label, "comparison": row} for label, row in extension_matched],
                                      "both_sequence_references": extension_references, "expanded_minus_original": extension_sequence},
               "hybrid_status": "complete; independently audited; primary gain inconclusive; standard head retained",
               "hybrid_followup": {"primary": hybrid_primary, "sensitivity": hybrid_sensitivity,
                                    "scores": hybrid_scores, "comparisons": hybrid_comparisons,
                                    "selections": hybrid_selections, "audit_status": hybrid_audit["status"]},
               "label_budget": {"primary": label_primary, "sensitivity": label_sensitivity, "curves": label_curves, "comparisons": label_comparisons, "audit_status": label_audit["status"]},
               "method_artifacts": {"ProteinMPNN": inverse_methods, "FoldX": foldx_methods, "Boltz": boltz_methods},
               "followup_pipeline": {"status": "all three original studies complete and independently audited; few-shot sole family-positive primary; selected full-cohort extension complete; computational search stopped; independent assays next", "primary_count": 3, "family_interval_level": 0.9833333333333333, "foldx_technical": foldx_technical},
               "fewshot_followup": {"primary": fewshot_primary, "family_three": fewshot_family, "absolute_curves": fewshot_curves, "comparisons": fewshot_comparisons, "target_budgets": fewshot_target_budgets, "source_budgets": fewshot_source_budgets, "independent_audit": fewshot_independent, "figure": fewshot_figure},
               "inverse_followup": {"primary": inverse_primary, "net_secondary": inverse_net, "sensitivity": inverse_sensitivity, "family_three": inverse_family, "absolute_scores": inverse_scores, "all_comparisons": inverse_comparisons, "independent_audit": inverse_independent, "exploratory_influence": inverse_influence},
               "energy_followup": {"primary": energy_primary, "family_three": energy_family, "absolute_scores": energy_scores, "all_comparisons": energy_comparisons, "independent_audit": energy_independent, "final_review": energy_review, "figure": descriptor_figure},
               "full_cohort_extension": {**full_cohort_data, "decision": full_decision, "cohort_metadata": full_metadata},
               "fewshot_offset_diagnostic": {"status": "complete, independently audited, exploratory post-hoc", "primary_family_changed": False, "absolute_scores": offset_scores, "comparisons": offset_comparisons, "independent_audit": offset_independent},
               "independent_metadata": {k: metadata[k] for k in ("checked_date", "outcome_policy", "rows", "peptides", "components", "allele_counts", "source_pmids")},
               "followup_cost_accounting": followup_cost,
               "followup_cloud_closeout": followup_closeout,
               "finalization_cost_accounting": final_cost,
               "cost_accounting": {k: v for k, v in costs.items() if k != "interpretation"},
               "source_sha256": dict(sorted(SOURCES.items()))}
    return body.strip(), json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the saved HTML or payload differs from current sources")
    parser.add_argument("--check-full-cohort", action="store_true", help="Check completed full-cohort tables without writing the snapshot")
    args = parser.parse_args()
    if args.check_full_cohort:
        try:
            full_cohort_presentation()
        except (ValueError, KeyError) as exc:
            raise SystemExit(f"Full-cohort tables not admitted: {exc}")
        print("Full-cohort table gate passed: all three arms, 12 contrasts and 24 fixed-query entries; no snapshot written.")
        return
    body, payload = build()
    index = ROOT / "site/index.html"
    content = index.read_text()
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError("Expected one generated section boundary pair")
    before, remainder = content.split(START)
    _, after = remainder.split(END)
    updated = before + START + "\n" + body + "\n  " + END + after
    outputs = {index: updated.encode(), ROOT / "site/round2-data.json": payload.encode()}
    for name in ("fewshot_curves.svg", "fewshot_curves.png", "descriptor_comparison.svg", "descriptor_comparison.png"):
        outputs[ROOT / "site/figures" / name] = (ROOT / "reports/followup3/figures" / name).read_bytes()
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, text in outputs.items() if not path.exists() or path.read_bytes() != text]
        if stale:
            raise SystemExit("Stale submission snapshot: " + ", ".join(stale))
        print("Submission snapshot matches saved evidence; numeric invariants passed.")
    else:
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text)
        print("Built static submission snapshot and source-hashed evidence payload.")


if __name__ == "__main__":
    main()
