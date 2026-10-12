#!/usr/bin/env python3
"""Build/check the offline round-two submission snapshot from saved evidence.

Only writes site/index.html's marked section and site/round2-data.json.
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
    return f'<a href="{REPO_URL}{path}">{html.escape(label)}</a>'


def build():
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
    read_path(ranking_base + "protocol.json")
    read_path(ranking_base + "REPORT.md", "text")
    inference = read_path("reports/finalization/inference/distribution_manifest.json")
    read_path("reports/finalization/inference/README.md", "text")
    read_path("docs/ROUND2_INFERENCE.md", "text")
    read_path("reports/finalization/INDEPENDENT_ASSAY_BRIEF.md", "text")
    read_path("reports/finalization/FINAL_REPORT.md", "text")
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
      <p>The expanded sequence selector changes the original by {interval(extension_sequence[0])} in HLA and {interval(extension_sequence[1])} in joint holdout. It underperforms the original in joint holdout, so presenting only the expanded comparator would exaggerate the advantage. Outer outcomes do not select a replacement procedure.</p>
    </details>
    <details class="evidence-detail"><summary>Scope and uncertainty</summary><p>The extension adds sequence regression/centred trees, a fixed ranker and matched R0 ridge/MLP heads. Inner selection, stopping and scaling obey each holdout regime. Its 264 comparison rows cover both panels and three uncertainty modes; all 12 bootstrap tables have 2,000 draws, and 3,080 rows cover every group-deletion contrast.</p><p>Intervals condition on fitted predictions and selections; group deletion does not retrain. Crossed resampling is a dependence sensitivity, not independent confirmation. HLA sequence groups are not validated functional distance. The ranker retains an inner-to-final scale-transport limitation. These transfer findings do not retrospectively change the main worthwhile-gain criterion.</p></details>
    <p class="note"><b>Hybrid fitting is complete; final audit is running.</b> The separate Boltz tree/ranking, tree–MLP blend and preprocessing study completed 480 scientific fits on four Modal CPUs after a 12-fit unscored pilot passed. Its primary comparison is the R5 hybrid selector minus the matched two-member R5 standard MLP. No hybrid outcome is claimed yet.</p>
    <p class="source-line">{path_link(extension_base + "REPORT.md", "Complete stronger-transfer report")} · {path_link(extension_base + "comparisons.csv", "All comparisons")} · {path_link(extension_base + "coverage_audit.json", "Independent coverage audit")} · {link("transfer/results/README.md", "Historical narrower assessment")}</p>
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
    <div class="budget-box"><p class="eyebrow">Conservative incremental accounting</p><p class="budget-total">${total:.4f} <span>against the ${costs["incremental_limit_usd"]:.0f} ceiling</span></p><p>Shared cache ${costs["cache_runtime_budget_accounting_usd"]:.4f} + sequence ${costs["sequence_conservative_accounting_including_preemptions_usd"]:.4f} + retained engineering allowance ${costs["engineering_full_allowance_retained_usd"]:.2f}.</p><p><b>Not a final invoice or live credit balance.</b> Retaining the engineering allowance does not mean it was spent. Provider snapshots lag and differ, leaving ${costs["metering_difference_unattributed_usd"]:.4f} unattributed; storage is separate. These assessment costs do not establish new-case end-to-end latency. A subsequent portable package measures cached-head latency separately; see the repository inference guide.</p></div>
    <p class="source-line">{link("resources/embedding_repeat_completion_audit.json", "Embedding repeat audit")} · {link("resources/sequence_repeat_completion_audit.json", "Sequence repeat audit")} · {link("resources/runtime_cost_summary.json", "Runtime/cost accounting")} · {link("resources/transfer_runtime_closeout.json", "Local transfer closeout")}</p>
  </section>

  <section id="round2-confirmation" aria-labelledby="round2-confirmation-title">
    <div class="question"><p class="eyebrow">Still open</p><h2 id="round2-confirmation-title">Independent confirmation.</h2><p class="task-body">The public stability release overlaps the canonical corpus. Conservative exclusions leave only <b>129 candidate rows across heterogeneous assays</b>, insufficient for representative confirmation; their outcomes remain reserved. Construct, endpoint, checkpoint ancestry and cross-length exposure gaps remain.</p><p class="task-body">No assay quote, partner commitment or adequately powered independent cohort is secured. The synthetic design supports asking for comparable assay quotes, not a guaranteed power or assay-noise claim. Lock the final procedure and analysis before a one-time independent confirmation.</p><p>{link("independent_data/README.md", "Identity/provenance audit")} · {link("independent_data/design/README.md", "Conditional assay-design study")} · {link("REPORT.md", "Current round 2 assessment")}</p></div>
  </section>

  <section id="round2-evidence" aria-labelledby="round2-evidence-title">
    <div class="shead"><h2 class="title" id="round2-evidence-title">Inspect the evidence.</h2><p class="lede">This page is a local static submission snapshot. Numeric tables are generated from saved result CSVs, with source SHA-256 hashes in the <a href="round2-data.json">downloadable evidence payload</a>. Reading this page starts no model jobs.</p></div>
    <p class="source-line">{path_link("reports/finalization/FINAL_REPORT.md", "Combined finalisation readout")} · {link("REPORT.md", "Closed round-two assessment")} · {path_link("reports/finalization/INDEPENDENT_ASSAY_BRIEF.md", "Unsent independent-assay brief")} · <a href="{REPO_URL}reports/NEXT_RESEARCH_PLAN.md">Research plan and execution gates</a> · <a href="#archive">Original hackathon archive</a></p>
  </section>
'''
    payload = {"schema": "round2-site-snapshot.1", "assessment_date": "2026-10-12", "scope": "development only; original historical test separate; independent confirmation open", "cohort": cohort,
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
               "hybrid_status": "480 scientific fits complete; saved-model/statistical audit running on Modal; no outcome claimed",
               "cost_accounting": {k: v for k, v in costs.items() if k != "interpretation"},
               "source_sha256": dict(sorted(SOURCES.items()))}
    return body.strip(), json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the saved HTML or payload differs from current sources")
    args = parser.parse_args()
    body, payload = build()
    index = ROOT / "site/index.html"
    content = index.read_text()
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError("Expected one generated section boundary pair")
    before, remainder = content.split(START)
    _, after = remainder.split(END)
    updated = before + START + "\n" + body + "\n  " + END + after
    outputs = {index: updated, ROOT / "site/round2-data.json": payload}
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, text in outputs.items() if not path.exists() or path.read_text() != text]
        if stale:
            raise SystemExit("Stale submission snapshot: " + ", ".join(stale))
        print("Submission snapshot matches saved evidence; numeric invariants passed.")
    else:
        for path, text in outputs.items():
            path.write_text(text)
        print("Built static submission snapshot and source-hashed evidence payload.")


if __name__ == "__main__":
    main()
