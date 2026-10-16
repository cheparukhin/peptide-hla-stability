#!/usr/bin/env python3
"""Export the reviewed few-shot curves and primary interval; never fit models."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pepstab-figure-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/followup3/fewshot'
INPUTS = [BASE / 'absolute_curves.csv', BASE / 'primary_family3_interval.json', BASE / 'independent_audit.json']
curves = list(csv.DictReader(INPUTS[0].open()))
family = json.loads(INPUTS[1].read_text())
audit = json.loads(INPUTS[2].read_text())
assert audit['status'] == 'passed' and audit['source_fits'] == 450 and audit['target_adapters'] == 9180
assert math.isclose(family['interval_level'], 59 / 60)
rows = [r for r in curves if r['panel_min_rows'] == '20']
assert len(rows) == 12 and len({(r['arm'], r['budget']) for r in rows}) == 12
budgets = [0, 5, 10, 20]
values = {arm: [float(next(r['score'] for r in rows if r['arm'] == arm and int(r['budget']) == k)) for k in budgets]
          for arm in ('R0', 'R5', 'sequence_anchor')}
assert math.isclose(values['R5'][-1] - values['R5'][0], family['point'], abs_tol=1e-12)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11.5, 'axes.labelsize': 11.5,
    'axes.titlesize': 12.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#717B88', 'axes.labelcolor': '#15191F', 'text.color': '#15191F',
    'xtick.color': '#474F5A', 'ytick.color': '#474F5A', 'svg.fonttype': 'path',
    'svg.hashsalt': 'pepstab-fewshot-reviewed-v1'})
fig = plt.figure(figsize=(10.2, 5.3), facecolor='white')
grid = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=.08, right=.97, bottom=.30, top=.76, wspace=.37)
ax = fig.add_subplot(grid[0, 0])
colors = {'R5': '#0E7490', 'sequence_anchor': '#8C5A0E', 'R0': '#717B88'}
labels = {'R5': 'R5 ridge (Boltz)', 'sequence_anchor': 'Sequence anchor (4 members)', 'R0': 'R0 ridge (one-hot)'}
markers = {'R5': 'o', 'sequence_anchor': 's', 'R0': '^'}
styles = {'R5': '-', 'sequence_anchor': '--', 'R0': ':'}
for arm in ('R5', 'sequence_anchor', 'R0'):
    ax.plot(budgets, values[arm], color=colors[arm], marker=markers[arm], linestyle=styles[arm],
            linewidth=2, markersize=6, label=labels[arm])
    ax.annotate(f'{values[arm][-1]:.3f}', (20, values[arm][-1]), xytext=(8, 0), textcoords='offset points',
                va='center', color=colors[arm], weight='bold')
ax.set(xlim=(-.6, 23.5), ylim=(0, .65), xlabel='Measured support labels per target allele', ylabel='Median per-allele Spearman')
ax.set_xticks(budgets)
ax.set_yticks([0, .2, .4, .6])
ax.grid(axis='y', color='#E7EAEF', linewidth=.8)
ax.set_axisbelow(True)
ax.set_title('Absolute ranking scores', loc='left', pad=13, weight='bold')
fig.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.53, .18), ncol=3, frameon=False, fontsize=10.5, handlelength=2.4)

interval = fig.add_subplot(grid[0, 1])
interval.axvline(0, color='#717B88', linestyle='--', linewidth=1)
interval.errorbar(family['point'], .53,
    xerr=[[family['point'] - family['ci_low']], [family['ci_high'] - family['point']]],
    fmt='o', color=colors['R5'], linewidth=2, markersize=8, capsize=6)
interval.set(xlim=(-.008, .135), ylim=(0, 1), xlabel='Change in median Spearman')
interval.set_xticks([0, .05, .10])
interval.set_xticklabels(['0', '+0.05', '+0.10'])
interval.set_yticks([])
interval.spines['left'].set_visible(False)
interval.set_title('Primary R5 gain: 20 vs 0 labels', loc='left', pad=13, weight='bold')
interval.text(.5, .86, f"+{family['point']:.6f}", transform=interval.transAxes, ha='center', va='center',
              color=colors['R5'], fontsize=21, weight='bold')
interval.text(.5, .18, '98.333333% family interval\n'
              f"[{family['ci_low']:+.6f}, {family['ci_high']:+.6f}]", transform=interval.transAxes,
              ha='center', va='center', fontsize=10.5, linespacing=1.5)
fig.suptitle('Few-shot adaptation under strict joint transfer', x=.08, y=.97, ha='left', fontsize=16, weight='bold')
fig.text(.08, .87, '68 target alleles · three fixed support draws · source models + residual adaptation', fontsize=11.5, color='#474F5A')
fig.text(.08, .14, 'Source fitting excludes target HLA groups and support/query peptide families. Curves average draw scores, not predictions.', fontsize=9.8, color='#474F5A')
fig.text(.08, .095, 'Sequence is a reduced historical four-member anchor, not the main competitive selector. Conditional development evidence.', fontsize=9.8, color='#474F5A')
fig.text(.08, .05, 'Pointwise curve intervals are not shown. The primary interval uses shared crossed peptide/HLA draws and three-question family control.', fontsize=9.5, color='#474F5A')
for ext in ('svg', 'png'):
    fig.savefig(OUT / f'fewshot_curves.{ext}', dpi=200, metadata={'Creator': 'source-backed few-shot figure', 'Date': None} if ext == 'svg' else {'Software': 'source-backed few-shot figure'})
    if ext == 'svg':
        svg = OUT / 'fewshot_curves.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
plt.close(fig)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'schema': 'pepstab-fewshot-figure.1', 'population': 'fixed 68 alleles; original training; strict joint transfer',
    'aggregation': 'mean of three draw-specific median per-allele Spearman scores',
    'displayed_curve_rows': rows, 'primary_family_interval': family,
    'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in INPUTS},
    'renderer': {'matplotlib_version': matplotlib.__version__, 'script_sha256': sha(Path(__file__))},
    'output_sha256': {f'fewshot_curves.{ext}': sha(OUT / f'fewshot_curves.{ext}') for ext in ('svg', 'png')}}
(OUT / 'fewshot_curves.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print('Exported the 12 reviewed curve points and the primary family interval to SVG and PNG.')
