#!/usr/bin/env python3
"""Render audited balanced-panel MLP descriptors, never fit or resample.

Requires completed cloud + independent audits for BOTH studies. Default roots:
  reports/followup3/inverse_execution_v2
  reports/followup3/energy_execution_v2

Each needs pooled_scores.csv, comparisons.csv, completion_audit.json,
independent_audit.json, primary_family3_interval.json and the original
bootstrap_mlp_min20.csv. Missing/pending evidence stops before reading scores
or importing the renderer. --check-inputs validates without exporting.

Exports descriptor_comparison.svg/.png plus exact plotted rows/source hashes.
Use a Python runtime with Matplotlib; SVG embeds glyph paths for portability.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {}
TOL = 2e-13


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, kind='json'):
    if not path.is_file():
        raise ValueError(f'Completed audited evidence unavailable: {path}')
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    if kind == 'csv':
        return list(csv.DictReader(raw.decode().splitlines()))
    if kind == 'bytes':
        return raw
    return json.loads(raw)


def one(rows, **fields):
    found = [r for r in rows if all(str(r[k]) == str(v) for k, v in fields.items())]
    require(len(found) == 1, f'Expected one reviewed row: {fields}; found {len(found)}')
    return found[0]


def near(a, b):
    return math.isclose(float(a), float(b), abs_tol=TOL, rel_tol=0)


def validate_audits(directory, study):
    cloud = read(directory / 'completion_audit.json')
    independent = read(directory / 'independent_audit.json')
    require(cloud['status'] == independent['status'] == 'passed', f'{study}: both completed audits must pass')
    require(independent['study'] == study and independent['rows'] == 4449, f'{study}: wrong study/panel')
    expected = {'mlp': 600, 'ridge': 300} if study == 'inverse' else {'mlp': 200, 'ridge': 100}
    require(independent['fit_counts'] == expected, f'{study}: frozen fit graph incomplete')
    require(cloud['protocol_hash'] == independent['protocol_hash'], f'{study}: protocol ancestry mismatch')
    require(independent['saved_weight_replay'] is False, f'{study}: reviewed audit scope changed')
    if study == 'energy':
        require(cloud['baseline_fits_repeated'] == 0, 'Energy baselines must be reused without refitting')
    return cloud, independent


def collect(inverse, energy):
    # Both pass gates BEFORE any score/interval file is opened.
    accepted = {name: validate_audits(directory, name)
                for name, directory in [('inverse', inverse), ('energy', energy)]}
    amendment = ROOT / 'reports/followup3/THIRD_PRIMARY_AMENDMENT.md'
    read(amendment, 'bytes')
    audit_source = ROOT / 'scripts/followup3_representation_independent_audit.py'
    read(audit_source, 'bytes')
    records, scores, panels = [], {}, {}
    for study, directory in [('inverse', inverse), ('energy', energy)]:
        cloud, independent = accepted[study]
        require(independent['amendment_sha256'] == sha(amendment), f'{study}: stale family amendment')
        require(independent['audit_source_sha256'] == sha(audit_source), f'{study}: independent audit source changed')
        panels[study] = independent['panels']['20']
        pooled = read(directory / 'pooled_scores.csv', 'csv')
        comparisons = read(directory / 'comparisons.csv', 'csv')
        expected_hashes = cloud['outputs'] if study == 'inverse' else cloud['output_sha256']
        for name in ('pooled_scores.csv', 'comparisons.csv'):
            require(sha(directory / name) == expected_hashes[name], f'{study}: changed saved {name}')
        arms = ('R0', 'R0_P', 'R0_PC', 'R5', 'R5_P', 'R5_PC') if study == 'inverse' else ('R0', 'R5', 'R0_E', 'R5_E')
        selected = [one(pooled, learner='mlp', min_rows=20, arm=arm) for arm in arms]
        require(len([r for r in pooled if r['learner'] == 'mlp' and r['min_rows'] == '20']) == len(arms), f'{study}: unexpected plotted arms')
        scores[study] = {}
        for row in selected:
            value = float(row['median_spearman'])
            require(math.isfinite(value) and -1 <= value <= 1, f'{study}: invalid absolute score')
            count_key = 'eligible_alleles' if study == 'inverse' else 'n_alleles'
            require(int(row[count_key]) == len(panels[study]), f'{study}: eligible-panel count mismatch')
            scores[study][row['arm']] = value
        baseline, candidate = ('R5_P', 'R5_PC') if study == 'inverse' else ('R5', 'R5_E')
        point = one(comparisons, learner='mlp', panel_min_rows=20, baseline=baseline, candidate=candidate)
        family = read(directory / 'primary_family3_interval.json')
        audited_family = one(independent['three_primary_intervals'], learner='mlp', panel_min_rows=20, primary=True)
        require(family == audited_family, f'{study}: family artifact differs from independent audit')
        require(near(family['interval_level'], 59/60) and near(family['quantiles'][0], 1/120)
                and near(family['quantiles'][1], 119/120), f'{study}: three-primary interval required')
        bootstrap = directory / 'bootstrap_mlp_min20.csv'
        read(bootstrap, 'bytes')
        require(family['saved_draws_sha256'] == sha(bootstrap), f'{study}: family interval draw hashes differ')
        delta = scores[study][candidate] - scores[study][baseline]
        require(near(delta, point['delta']) and near(delta, family['delta']), f'{study}: primary arithmetic mismatch')
        require(all(math.isfinite(float(family[k])) for k in ('delta', 'ci_low', 'ci_high'))
                and family['ci_low'] <= family['ci_high'], f'{study}: invalid family interval')
        records.append({'study': study, 'baseline': baseline, 'candidate': candidate,
                        'pointwise_95': point, 'family_98_333333': family})
    require(panels['inverse'] == panels['energy'], 'Studies must assess the identical eligible balanced panel')
    for arm in ('R0', 'R5'):
        require(near(scores['inverse'][arm], scores['energy'][arm]), f'Exact reused baseline score differs: {arm}')
    merged = dict(scores['inverse'], **{arm: scores['energy'][arm] for arm in ('R0_E', 'R5_E')})
    ordered = [{'arm': f'{rep}{suffix}', 'representation': rep, 'descriptor': label,
                'median_spearman': merged[f'{rep}{suffix}']}
               for suffix, label in [('', 'Baseline'), ('_P', '+ P'), ('_PC', '+ P + C'), ('_E', '+ E')]
               for rep in ('R0', 'R5')]
    return ordered, records, panels['inverse']


def render(rows, records, panel, out):
    # Plotting starts only after collect() has accepted actual audited evidence.
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pepstab-figure-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#717B88', 'text.color': '#15191F',
        'axes.labelcolor': '#15191F', 'xtick.color': '#474F5A', 'ytick.color': '#474F5A',
        'svg.fonttype': 'path', 'svg.hashsalt': 'pepstab-balanced-descriptors-v1'})
    fig = plt.figure(figsize=(10.4, 5.6), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=.10, right=.96, top=.76, bottom=.32, wspace=.35)
    absolute = fig.add_subplot(grid[0, 0])
    color = {'R0': '#717B88', 'R5': '#0E7490'}
    marker = {'R0': '^', 'R5': 'o'}
    variants = ['Baseline', '+ P', '+ P + C', '+ E']
    for row in rows:
        index = variants.index(row['descriptor'])
        offset = -.10 if row['representation'] == 'R0' else .10
        absolute.plot(row['median_spearman'], index + offset,
            marker=marker[row['representation']], color=color[row['representation']],
            linestyle='none', markersize=7)
        absolute.annotate(f"{row['median_spearman']:.4f}", (row['median_spearman'], index + offset),
            xytext=(7, -4 if offset < 0 else 3), textcoords='offset points',
            color=color[row['representation']], fontsize=10, va='center')
    values = [r['median_spearman'] for r in rows]
    absolute.set(xlim=(min(0, min(values)-.035), min(1.08, max(.12, max(values)+.14))),
        ylim=(-.45, 3.45), xlabel='Median per-allele Spearman')
    absolute.set_yticks(range(4), variants)
    absolute.invert_yaxis()
    absolute.xaxis.set_major_locator(MaxNLocator(4))
    absolute.grid(axis='x', color='#E7EAEF', linewidth=.8)
    absolute.set_axisbelow(True)
    absolute.set_title('All eight matched MLP arms', loc='left', pad=13, weight='bold')
    handles = [plt.Line2D([], [], color=color[rep], marker=marker[rep], linestyle='none',
               label=label, markersize=7) for rep, label in [('R0', 'R0: sequence one-hot'), ('R5', 'R5: Boltz')]]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.31, .205), ncol=2, frameon=False, fontsize=10.5)
    paired = fig.add_subplot(grid[0, 1])
    paired.axvline(0, color='#717B88', linestyle='--', linewidth=1)
    descriptions = ['ProteinMPNN context: R5+P+C − R5+P', 'Repaired descriptors: R5+E − R5']
    for y, record, label in zip((.75, .25), records, descriptions):
        interval = record['family_98_333333']
        paired.hlines(y, interval['ci_low'], interval['ci_high'], color=color['R5'], linewidth=2)
        paired.vlines([interval['ci_low'], interval['ci_high']], y-.025, y+.025, color=color['R5'], linewidth=1.5)
        paired.plot(interval['delta'], y, 'o', color=color['R5'], markersize=7)
        paired.text(.5, y+.17, label, ha='center', transform=paired.transAxes, fontsize=10.5, weight='bold',
            bbox={'facecolor':'white', 'edgecolor':'none', 'pad':2.5})
        paired.text(.5, y-.13,
            f"{interval['delta']:+.4f} [{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]",
            ha='center', transform=paired.transAxes, fontsize=10.5, color=color['R5'])
    bounds = [0] + [float(r['family_98_333333'][k]) for r in records for k in ('delta', 'ci_low', 'ci_high')]
    padding = max(.012, (max(bounds)-min(bounds))*.14)
    paired.set(xlim=(min(bounds)-padding, max(bounds)+padding), ylim=(0, 1),
        xlabel='Difference in median Spearman', yticks=[])
    paired.spines['left'].set_visible(False)
    paired.xaxis.set_major_locator(MaxNLocator(4))
    paired.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if abs(x) < 1e-12 else f'{x:+.2f}'))
    paired.set_title('Two primary paired 98.333333% intervals', loc='left', pad=13, weight='bold')
    fig.suptitle('Balanced-panel descriptor comparison', x=.10, y=.97, ha='left', fontsize=16, weight='bold')
    fig.text(.10, .865, f'4,449 original-training pairs · {len(panel)} eligible alleles · two-member MLPs · clean nested selection',
        fontsize=11, color='#474F5A')
    fig.text(.10, .16, 'P: peptide-only compatibility · C: full-complex compatibility · E: 29 repaired energy/interface/clash/residue descriptors.',
        fontsize=9.5, color='#474F5A')
    fig.text(.10, .11, 'Conditional development evidence. These balanced-panel MLP recipes differ from the main selector and few-shot ridge assessment.',
        fontsize=9.5, color='#474F5A')
    fig.text(.10, .06, 'Absolute-score intervals are not shown. Primary intervals use shared saved peptide-cluster draws with three-question family control.',
        fontsize=9.5, color='#474F5A')
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'png'):
        metadata = {'Creator': 'source-backed descriptor comparison', 'Date': None} if extension == 'svg' else {'Software': 'source-backed descriptor comparison'}
        fig.savefig(out / f'descriptor_comparison.{extension}', dpi=200, metadata=metadata)
        if extension == 'svg':
            svg = out / 'descriptor_comparison.svg'
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    manifest = {'schema': 'pepstab-descriptor-figure.1', 'n_original_training_pairs': 4449,
        'eligible_alleles': panel, 'learner': 'matched two-member MLP; clean nested regularization selection',
        'panel_minimum_rows': 20, 'scope': 'conditional development; distinct from main and few-shot assessments',
        'displayed_absolute_rows': rows, 'primary_comparisons': records,
        'input_sha256': dict(sorted(SOURCES.items())),
        'renderer': {'matplotlib_version': matplotlib.__version__, 'script_sha256': sha(Path(__file__))},
        'output_sha256': {f'descriptor_comparison.{ext}': sha(out / f'descriptor_comparison.{ext}') for ext in ('svg', 'png')}}
    (out / 'descriptor_comparison.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print('Exported eight audited absolute MLP scores and two primary family intervals to SVG and PNG.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inverse', type=Path, default=ROOT/'reports/followup3/inverse_execution_v2')
    parser.add_argument('--energy', type=Path, default=ROOT/'reports/followup3/energy_execution_v2')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    try:
        rows, intervals, panel = collect(args.inverse.resolve(), args.energy.resolve())
    except (ValueError, KeyError) as exc:
        parser.exit(2, f'Figure not rendered: {exc}\n')
    if args.check_inputs:
        print(f'Accepted eight absolute scores, two primary family intervals and {len(panel)} eligible alleles.')
        return
    render(rows, intervals, panel, args.out)


if __name__ == '__main__':
    main()
