#!/usr/bin/env python3
"""Build and validate the scientific submission from saved results.

Reads evidence without refitting models or changing research artifacts.
Uses only the Python standard library and Git for publication-link checks.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
BASE = "reports/further_research/round2/"
REPO_URL = "https://github.com/cheparukhin/peptide-hla-stability/blob/main/"
START = "<!-- ROUND2 GENERATED START -->"
END = "<!-- ROUND2 GENERATED END -->"
SOURCES = {}


def read(name, kind="json"):
    return read_path(BASE + name, kind)


def read_path(path, kind="json"):
    source = ROOT / path
    if source.is_file():
        raw = source.read_bytes()
    else:
        with zipfile.ZipFile(ROOT / "reports/protocol_sources.zip") as archive:
            raw = archive.read(path)
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
    return format(float(value), "+.4f" if signed else ".4f").replace("-", "−")


def interval(row):
    return f'{fmt(row["delta"], True)} [{fmt(row["ci_low"], True)}, {fmt(row["ci_high"], True)}]'


def link(name, label):
    return path_link(BASE + name, label)


def path_link(path, label):
    return f'<a href="{html.escape(REPO_URL + quote(path, safe="/#"))}">{html.escape(label)}</a>'


def publication_link_check(outputs):
    """Reject path escapes, missing files and untracked release dependencies.

    Requires a Git checkout and CLI; use a clean clone for release verification.
    """
    if not (ROOT / ".git").exists():
        raise ValueError("Publication validation requires a Git checkout, not a source archive")
    tracked = set(subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.split("\0"))
    site = ROOT / "site"
    problems = []
    checked = 0

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ("href", "src", "poster") and value:
                    check(value, self.path)

    def check(value, document):
        nonlocal checked
        parsed = urlsplit(value)
        if value.startswith(REPO_URL):
            target = (ROOT / unquote(urlsplit(value[len(REPO_URL):]).path)).resolve()
        elif parsed.scheme or parsed.netloc or not parsed.path:
            return
        else:
            target = (document.parent / unquote(parsed.path)).resolve()
            if parsed.path.startswith("/") or not target.is_relative_to(site):
                problems.append(f"{document.relative_to(ROOT)}: link escapes site artifact: {value}")
                return
        if not parsed.scheme and parsed.path.endswith("/"):
            target = target / "index.html"
        if not target.is_relative_to(ROOT):
            problems.append(f"{document.relative_to(ROOT)}: repository link escapes checkout: {value}")
        elif str(target.relative_to(ROOT)) not in tracked:
            problems.append(f"{document.relative_to(ROOT)}: untracked publication dependency: {value}")
        elif not target.is_file() and target not in outputs:
            problems.append(f"{document.relative_to(ROOT)}: missing publication dependency: {value}")
        checked += 1

    for document in sorted(site.rglob("*.html")):
        if str(document.relative_to(ROOT)) not in tracked:
            problems.append(f"Untracked HTML would be absent from the release: {document.relative_to(ROOT)}")
        parser = Links()
        parser.path = document
        parser.feed(outputs.get(document, document.read_bytes()).decode())
    if problems:
        raise ValueError("Publication links failed:\n" + "\n".join(problems))
    return checked


def full_cohort_evidence():
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
    contrasts = (("R5","R5_PC","Net"),("R5_P","R5_PC","Context"),("R5","R5_P","Peptide-only"))
    absolute = independent["absolute_scores"]
    score_cells = {}
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
            fixed_panel = independent["fixed_query_panels"][str(minimum)]
            for kind,names in (("point",arms),("increment",("net","context","peptide"))):
                for name in names:
                    row = one(fixed,learner=learner,min_rows=minimum,kind=kind,comparison=name)
                    before,after,change = (float(row[k]) for k in ("original_subset","fulltrained_same_queries","old_to_new_change"))
                    if not all(math.isfinite(v) for v in (before,after,change)) or not math.isclose(after-before,change,abs_tol=1e-12) or int(row["query_rows"]) != 4449 or int(row["eligible_alleles"]) != len(fixed_panel) or row["panel_alleles"].split("|") != fixed_panel:
                        raise ValueError("Full-cohort fixed-query descriptive arithmetic/panel mismatch")
    main = compare(comparisons,"R5","R5_PC",learner="mlp")
    if any(not float(row["ci_low"]) < 0 < float(row["ci_high"]) for row in comparisons) or not main["delta"] < 0:
        raise ValueError("Final inconclusive full-cohort interpretation requires review")
    return {"main": main, "absolute_scores": absolute, "comparisons": comparisons, "fixed_query_descriptive": fixed}


def checked_interval(row):
    row = dict(row)
    for key in ('delta', 'ci_low', 'ci_high'):
        row[key] = float(row[key])
        if not math.isfinite(row[key]):
            raise ValueError('Nonfinite interval')
    if not row['ci_low'] <= row['delta'] <= row['ci_high']:
        raise ValueError('Point lies outside saved interval')
    return row


def passed(path):
    audit = read_path(path)
    if audit['status'] != 'passed':
        raise ValueError(f'Passed audit required: {path}')
    return audit


def family_interval(base, row, draws):
    family = read_path(base + 'primary_family3_interval.json')
    read_path(base + draws, 'text')
    if (not math.isclose(family['interval_level'], 59/60)
            or not math.isclose(family.get('delta', family.get('point')), float(row['delta']), abs_tol=1e-12)
            or family['saved_draws_sha256'] != SOURCES[base + draws]):
        raise ValueError('Family interval differs from saved comparison or bootstrap draws')
    if family.get('amendment_sha256') not in (None, SOURCES['reports/followup3/THIRD_PRIMARY_AMENDMENT.md']):
        raise ValueError('Family interval amendment changed')
    return family


def build():
    SOURCES.clear()
    main_path = BASE + 'results/main/paired_comparisons.csv'
    main = read_path(main_path, 'csv')
    scores = read('results/main/pooled_scores.csv', 'csv')
    primary = compare(main, 'sequence_selector', 'embedding_selector')
    contact = compare(main, 'R5_shuffled_mlp', 'R5_mlp')
    for arm, key in [('sequence_selector', 'baseline_score'), ('embedding_selector', 'candidate_score')]:
        score = one(scores, procedure=arm, panel_min_rows=20, scope='pooled_outer_held')
        if not math.isclose(float(score['median_per_allele_spearman']), primary[key], abs_tol=1e-12):
            raise ValueError('Main comparison and score table disagree')
    cohort = read('resources/cache_completion_audit.json')['cohort']
    if (cohort['n_rows'], cohort['n_clusters'], cohort['n_alleles']) != (19716, 3587, 75):
        raise ValueError('Cohort description requires revision')
    if not 0 < primary['ci_low'] < .05 or not contact['ci_low'] < 0 < contact['ci_high']:
        raise ValueError('Main or contact interpretation requires revision')

    transfer_base = 'reports/finalization/transfer_cloud_recovery/results/'
    passed(transfer_base + 'coverage_audit.json')
    transfer = read_path(transfer_base + 'comparisons.csv', 'csv')
    transfer_rows = []
    for regime, label in [('hla', 'HLA group'), ('joint', 'HLA group + peptide cluster')]:
        for baseline, candidate, method in [('R0_mlp', 'R5_mlp', 'Matched MLPs'), ('expanded_sequence_selector', 'embedding_selector', 'Embedding vs expanded sequence selector')]:
            row = compare(transfer, baseline, candidate, regime=regime, mode='crossed')
            if row['ci_low'] <= 0:
                raise ValueError('Positive transfer conclusion requires revision')
            transfer_rows.append({'holdout': label, 'comparison': method, **row})

    transfer_original = {regime: compare(transfer, 'original', 'embedding_selector', regime=regime, mode='crossed') for regime in ('hla', 'joint')}

    read_path('reports/followup3/THIRD_PRIMARY_AMENDMENT.md', 'text')
    few_base = 'reports/followup3/fewshot/'
    few_audit = passed(few_base + 'audit.json')
    few_independent = passed(few_base + 'independent_audit.json')
    if ((few_independent['cells'], few_independent['source_fits'], few_independent['target_adapters']) != (75, 450, 9180)
            or not few_audit['saved_models_predictions_replayed_exactly']
            or len(few_audit['panels']['20']) != 68):
        raise ValueError('Few-shot description requires revision')
    curves = read_path(few_base + 'absolute_curves.csv', 'csv')
    few = checked_interval(one(read_path(few_base + 'comparisons.csv', 'csv'), contrast='primary_R5_adaptation20', panel_min_rows=20, mode='crossed'))
    few_family = family_interval(few_base, few, 'bootstrap_crossed_min20.csv')
    few_zero = float(one(curves, panel_min_rows=20, arm='R5', budget=0)['score'])
    few_twenty = float(one(curves, panel_min_rows=20, arm='R5', budget=20)['score'])
    if not math.isclose(few_twenty - few_zero, few['delta'], abs_tol=1e-12) or few_family['ci_low'] <= 0:
        raise ValueError('Few-shot arithmetic or conclusion requires revision')
    figure_base = 'reports/followup3/figures/'
    figure = read_path(figure_base + 'fewshot_curves.json')
    if figure['displayed_curve_rows'] != [r for r in curves if r['panel_min_rows'] == '20'] or figure['primary_family_interval'] != few_family:
        raise ValueError('Few-shot figure does not match saved results')
    for name in ('fewshot_curves.svg', 'fewshot_curves.png'):
        raw = (ROOT / figure_base / name).read_bytes()
        SOURCES[figure_base + name] = hashlib.sha256(raw).hexdigest()
        if SOURCES[figure_base + name] != figure['output_sha256'][name]:
            raise ValueError('Few-shot figure changed after validation')

    descriptor_rows = []
    for directory, baseline, candidate, label in [
        ('inverse_execution_v2', 'R5_P', 'R5_PC', 'ProteinMPNN complex context'),
        ('energy_execution_v2', 'R5', 'R5_E', 'Repaired FoldX descriptors')]:
        base = f'reports/followup3/{directory}/'
        complete = passed(base + 'completion_audit.json')
        independent = passed(base + 'independent_audit.json')
        comparisons = read_path(base + 'comparisons.csv', 'csv')
        expected = complete.get('outputs', complete.get('output_sha256'))['comparisons.csv']
        if expected != SOURCES[base + 'comparisons.csv'] or independent['rows'] != 4449:
            raise ValueError('Descriptor comparison changed after audit')
        row = compare(comparisons, baseline, candidate, learner='mlp')
        family = family_interval(base, row, 'bootstrap_mlp_min20.csv')
        if not family['ci_low'] < 0 < family['ci_high']:
            raise ValueError('Descriptor conclusion requires revision')
        descriptor_rows.append({'label': label, 'comparison': row, 'family': family})
    full = full_cohort_evidence()
    ranking_base = 'reports/finalization/ranking/'
    hybrid_base = 'reports/finalization/hybrids_cloud_v1/'
    passed(ranking_base + 'completion_audit.json')
    passed(hybrid_base + 'coverage_audit.json')
    ranking = checked_interval(one(read_path(ranking_base + 'comparisons.csv', 'csv'), contrast='R5_ranked-R5_mse', min_rows=20))
    hybrid = checked_interval(one(read_path(hybrid_base + 'comparisons.csv', 'csv'), contrast='R5_hybrid_selector-R5_standard', min_rows=20))
    if any(not row['ci_low'] < 0 < row['ci_high'] for row in (ranking, hybrid)):
        raise ValueError('Ranking or hybrid conclusion requires revision')

    transfer_html = '\n'.join(f'<tr><th scope="row">{r["holdout"]}<span>{r["comparison"]}</span></th><td>{fmt(r["baseline_score"])}</td><td>{fmt(r["candidate_score"])}</td><td>{interval(r)}</td></tr>' for r in transfer_rows)
    descriptor_html = '\n'.join(f'<tr><th scope="row">{r["label"]}</th><td>{fmt(r["comparison"]["baseline_score"])}</td><td>{fmt(r["comparison"]["candidate_score"])}</td><td>{fmt(r["comparison"]["delta"], True)} [{fmt(r["family"]["ci_low"], True)}, {fmt(r["family"]["ci_high"], True)}]</td></tr>' for r in descriptor_rows)
    body = f'''
<header class="masthead">
  <p class="eyebrow">Peptide–HLA stability · Research findings</p>
  <h1>Boltz embeddings improve peptide–HLA stability ranking</h1>
  <p class="standfirst">Boltz-2 features improve within-allele ranking of peptide–HLA complex stability in development assessments and across held-out HLA groups. Adding twenty measurements per target allele further improves adaptation. Independent kinetic measurements are needed to test these findings.</p>
  <nav aria-label="Research navigation"><a href="#results">Results</a><a href="#transfer">Transfer</a><a href="#adaptation">Adaptation</a><a href="#interpretation">Interpretation</a>{path_link('reports/REPORT.md', 'Full report ↗')}</nav>
</header>
<section class="abstract" aria-labelledby="question">
  <h2 id="question">Can representations from a structure model predict complex stability?</h2>
  <p>Class I human leukocyte antigen (HLA) molecules display intracellular peptides on the cell surface for T-cell recognition. How long a peptide–HLA complex remains intact influences antigen presentation. Ordering candidate peptides by measured stability within each allele therefore complements binding-affinity prediction.</p>
  <p>The trunk is Boltz-2's main processing stage. Its embeddings are numerical features describing individual residues and residue pairs in context. We test whether these frozen features improve peptide stability ranking, measured by median within-allele Spearman correlation.</p>
  <dl class="cohort"><div><dt>Development pairs</dt><dd>19,716</dd></div><div><dt>Peptide clusters</dt><dd>3,587</dd></div><div><dt>Eligible HLA alleles</dt><dd>69</dd></div></dl>
  <p class="note">The development cohort comes from the original training split and spans 75 HLA alleles, of which 69 have at least 20 observations and receive equal weight in the primary endpoint. Nested cross-validation separates held-out peptide clusters from model fitting and selection.</p>
</section>
<section id="results" aria-labelledby="results-title">
  <p class="eyebrow">01 · Peptide-cluster holdout</p>
  <h2 id="results-title">Embeddings improve ranking over competitive sequence models</h2>
  <div class="main-result"><div><p class="metric-label">Paired improvement in Spearman</p><p class="effect">{fmt(primary['delta'], True)}</p><p class="uncertainty">Conditional 95% interval [{fmt(primary['ci_low'], True)}, {fmt(primary['ci_high'], True)}]</p></div>
  <div class="score-pair"><p><span>Embedding selector</span><strong>{fmt(primary['candidate_score'])}</strong></p><p><span>Sequence selector</span><strong>{fmt(primary['baseline_score'])}</strong></p></div></div>
  <p>For unseen peptide clusters on known alleles, the prespecified embedding-or-original selector outperforms the competitive sequence selector across the 69 eligible alleles. The interval supports a ranking gain from Boltz-2 trunk representations, but its lower bound falls below the predeclared +0.05 threshold. A gain exceeding 0.05 Spearman units is therefore not established.</p>
  <p>Retaining position-aware token features improves accuracy over pooling the same underlying tokens, cleanly supporting positional indexing. While the full interface representation also outperforms contact-restricted pooled pairs, this comparison changes groove coverage and positional resolution jointly rather than isolating a positional effect. Separately, pooling over a crystal-derived peptide–HLA contact mask offers no established advantage over shuffled residue-pair membership ({interval(contact)}). The positive performance comparisons show that the learned representations carry predictive signal without establishing a specific physical contact mechanism behind the gain.</p>
  <p class="source">{path_link(main_path, 'Paired results')} · {path_link(BASE + 'results/main/pooled_scores.csv', 'All procedure scores')}</p>
</section>
<section id="transfer" aria-labelledby="transfer-title">
  <p class="eyebrow">02 · Held-out HLA groups</p>
  <h2 id="transfer-title">The advantage persists when target HLA groups are excluded from fitting</h2>
  <p>Generalizing across HLA groups requires predicting stability for alleles excluded from supervised training. Two comparisons test transfer under HLA-group holdout and under joint holdout of both target HLA groups and target peptide clusters: matched multilayer perceptrons (MLPs) comparing sequence against Boltz features, and a broader comparison between selected sequence and embedding procedures.</p>
  <div class="table-scroll" role="region" aria-label="Transfer results" tabindex="0"><table><caption>Median within-allele Spearman · 69 alleles · conditional crossed 95% intervals</caption><thead><tr><th scope="col">Held out / comparison</th><th scope="col">Sequence</th><th scope="col">Embedding</th><th scope="col">Difference [interval]</th></tr></thead><tbody>{transfer_html}</tbody></table></div>
  <p class="note">Matched nonlinear MLPs provide the headline architectural comparison because an additive sequence-ridge model cannot express allele-specific peptide preferences across alleles. The original 30-member sequence ensemble scores {fmt(transfer_original["hla"]["baseline_score"])} under HLA holdout and {fmt(transfer_original["joint"]["baseline_score"])} under joint holdout; expanding sequence selection does not consistently improve this reference. Because Boltz-2 brings broader molecular context and prior structural training, holding out HLA groups from supervised fitting does not establish structural novelty during pretraining.</p>
  <p class="source">{path_link(transfer_base + 'comparisons.csv', 'All transfer comparisons')}</p>
</section>
<section id="adaptation" aria-labelledby="adaptation-title">
  <p class="eyebrow">03 · Few-shot adaptation</p>
  <h2 id="adaptation-title">Twenty measurements per target allele improve transfer</h2>
  <p>When a target HLA allele has limited experimental data, few-shot adaptation can adjust source predictions using a small support set. Under strict joint holdout across 68 target alleles, adapting a Boltz-feature ridge model (R5) raises median within-allele Spearman correlation from <strong>{fmt(few_zero)}</strong> without target labels to <strong>{fmt(few_twenty)}</strong> with 20 labels per allele, a paired improvement of <strong>{fmt(few['delta'], True)}</strong> (98.33% three-primary Bonferroni interval <strong>[{fmt(few_family['ci_low'], True)}, {fmt(few_family['ci_high'], True)}]</strong>).</p>
  <figure><img src="figures/fewshot_curves.svg" width="1020" height="530" loading="lazy" alt="Learning curves for sequence and Boltz representations, with the positive family-adjusted interval for Boltz adaptation at twenty labels."><figcaption>Scores average three fixed support-set sampling designs across 68 target alleles. Support and query peptide families are excluded from source-model fitting. R0 is additive sequence ridge, and R5 uses Boltz features. The four-member sequence ensemble is a reduced reference comparator, not the fully tuned sequence selector. The interval at 20 labels conditions on these three fixed support designs. Curve uncertainty bands are not estimated.</figcaption></figure>
  <p class="source">{path_link(few_base + 'absolute_curves.csv', 'Exact learning curves')} · {path_link(few_base + 'primary_family3_interval.json', 'Primary family interval')} · <a href="figures/fewshot_curves.svg">Download figure</a></p>
</section>
<section id="extensions" aria-labelledby="extensions-title">
  <p class="eyebrow">04 · Additional descriptors and learners</p>
  <h2 id="extensions-title">Added complexity has no established benefit beyond Boltz</h2>
  <p>We next test whether supplementing frozen Boltz trunk embeddings with explicit inverse-folding likelihoods or physical energy terms adds predictive signal. On a matched 4,449-pair subset with 69 eligible alleles, MLPs evaluate two augmentations: adding complex-context ProteinMPNN scores to a reference of Boltz plus peptide-only ProteinMPNN scores, and adding repaired FoldX energy descriptors to Boltz features alone.</p>
  <div class="table-scroll" role="region" aria-label="Descriptor results" tabindex="0"><table><caption>Matched MLPs · 69 alleles · conditional 98.33% family intervals</caption><thead><tr><th scope="col">Added information</th><th scope="col">Reference</th><th scope="col">Augmented</th><th scope="col">Difference [interval]</th></tr></thead><tbody>{descriptor_html}</tbody></table></div>
  <p>A post-hoc extension of the ProteinMPNN comparison to all 19,716 development pairs, selected after the subset analysis, yields {interval(full['main'])} relative to Boltz alone (pointwise 95% interval). Alternative pairwise ranking losses ({interval(ranking)}) and hybrid learners ({interval(hybrid)}) also yield intervals spanning zero. Because such intervals remain compatible with small effects, these experiments establish neither added benefit nor equivalence.</p>
  <p class="source">{path_link('reports/followup3/inverse_execution_v2/comparisons.csv', 'ProteinMPNN')} · {path_link('reports/followup3/energy_execution_v2/comparisons.csv', 'FoldX')} · {path_link('reports/followup3/inverse_fulltrain_candidate/comparisons.csv', 'Full-cohort extension')} · {path_link(ranking_base + 'comparisons.csv', 'Ranking loss')} · {path_link(hybrid_base + 'comparisons.csv', 'Hybrid learners')}</p>
</section>
<section id="interpretation" class="discussion" aria-labelledby="interpretation-title">
  <p class="eyebrow">Interpretation</p>
  <h2 id="interpretation-title">A useful development result; independent measurements must test generalisation</h2>
  <p>The evidence supports improved within-allele stability ranking inside this development corpus, but does not establish calibrated physical half-lives, immunogenicity prediction, or experimental screening savings. The underlying assay measures β2-microglobulin dissociation as a stability proxy, and zero-valued summary records may reflect censoring or measurement limitations that aggregated labels cannot resolve.</p>
  <p>All confidence intervals resample saved out-of-fold predictions while holding fitted models and selection choices fixed, using peptide-cluster resampling for peptide holdout and crossed HLA-group by peptide-cluster resampling for transfer and adaptation. These conditional intervals omit variance from refitting models, collecting new measurements, and selecting follow-up studies adaptively; the three-primary correction covers its declared family only.</p>
  <p>The decisive next experiment is an independent, blinded kinetic-assay panel evaluated with frozen models and a prespecified analysis plan. Retaining raw dissociation curves and experimental replicates is needed to distinguish rapid decay from censoring or assay failure, providing the independent confirmation that development-split evaluation cannot supply.</p>
</section>
<footer><p>For related context, see the report sections on <a href="https://github.com/cheparukhin/peptide-hla-stability/blob/main/reports/REPORT.md#earlier-sequence-and-pretrained-feature-comparisons">earlier sequence baselines</a> and <a href="https://github.com/cheparukhin/peptide-hla-stability/blob/main/reports/REPORT.md#sequence-predictions-transfer-to-ligand-presentation">ligand presentation</a>.</p><p>{path_link('reports/REPORT.md', 'Full scientific report')} · {path_link('EVALUATION.md', 'Evaluation methods')} · {path_link('docs/DATASETS.md', 'Dataset and assay')} · {path_link('docs/ROUND2_INFERENCE.md', 'Use the released models')}</p><p><a href="round2-data.json">Source values and hashes</a> · <a href="intro/">Peptide–HLA biology animation</a> · <a href="https://github.com/cheparukhin/peptide-hla-stability">Code and evidence</a></p></footer>
'''
    payload = {'schema': 'scientific-submission.2', 'endpoint': 'median within-allele Spearman',
               'scope': 'conditional development assessment; no independent confirmation',
               'cohort': cohort, 'main': primary, 'contact_control': contact,
               'transfer': transfer_rows, 'transfer_original_reference': transfer_original, 'fewshot': {'primary': few, 'family_interval': few_family, 'curves': curves},
               'descriptor_comparisons': descriptor_rows, 'full_cohort': full,
               'ranking': ranking, 'hybrid': hybrid, 'source_sha256': dict(sorted(SOURCES.items()))}
    return body.strip(), json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check generated page, figures, evidence and links')
    parser.add_argument('--check-full-cohort', action='store_true', help='Check all saved full-cohort comparison and fixed-query rows')
    args = parser.parse_args()
    if args.check_full_cohort:
        full_cohort_evidence()
        print('Full-cohort evidence passed: 12 contrasts and 24 fixed-query entries.')
        return
    body, payload = build()
    index = ROOT / 'site/index.html'
    content = index.read_text()
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError('Expected one generated section boundary pair')
    before, remainder = content.split(START)
    _, after = remainder.split(END)
    updated = before + START + '\n' + body + '\n' + END + after
    outputs = {index: updated.encode(), ROOT / 'site/round2-data.json': payload.encode()}
    for name in ('fewshot_curves.svg', 'fewshot_curves.png'):
        outputs[ROOT / 'site/figures' / name] = (ROOT / 'reports/followup3/figures' / name).read_bytes()
    publication_link_check(outputs)
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, data in outputs.items() if not path.exists() or path.read_bytes() != data]
        if stale:
            raise SystemExit('Stale submission files: ' + ', '.join(stale))
        print('Submission matches saved results; arithmetic, audits, figures and links passed.')
    else:
        for path, data in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print('Built scientific submission and source-hashed evidence snapshot.')


if __name__ == '__main__':
    main()
