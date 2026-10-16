#!/usr/bin/env python3
"""Bounded synthetic planning simulation; never reads labels or model predictions.

Planning intervals invert an independently simulated endpoint error distribution.
They know the assumed DGP and are NOT the prespecified confirmation bootstrap.
A small exact cluster-percentile bootstrap diagnostic quantifies the discrepancy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab.evaluation import UNRANKED_CONTRIBUTION, validate_finite


@dataclass(frozen=True)
class Panel:
    name: str
    counts: tuple[int, ...]
    group_size: int = 4

    @property
    def n_clusters(self):
        # Fully crossed overlap: every allele draws from the same ordered peptide
        # pool. Sparse alleles see its initial subset, not independent peptides.
        return max(self.counts)


@dataclass(frozen=True)
class Dependence:
    name: str
    peptide_fraction: float
    hla_group_fraction: float
    paired_error_correlation: float = 0.8

    def __post_init__(self):
        if not (0 <= self.peptide_fraction + self.hla_group_fraction <= 1):
            raise ValueError('dependence variance fractions must sum to <= 1')
        if min(self.peptide_fraction, self.hla_group_fraction) < 0:
            raise ValueError('dependence variance fractions must be nonnegative')
        if not 0 <= self.paired_error_correlation <= 1:
            raise ValueError('paired error correlation must lie in [0, 1]')


PANELS = (
    Panel('balanced_12x50', (50,) * 12),
    Panel('balanced_12x100', (100,) * 12),
    Panel('balanced_12x200', (200,) * 12),
    Panel('balanced_12x400', (400,) * 12),
    Panel('balanced_12x800', (800,) * 12),
    Panel('balanced_24x200', (200,) * 24),
    Panel('sparse_2400', (10,) * 6 + (390,) * 6),
    Panel('near_threshold_2400', (30,) * 6 + (370,) * 6),
    Panel('repaired_2400', (50,) * 6 + (350,) * 6),
)
DEPENDENCE = (
    Dependence('weak', 0.0, 0.0),
    Dependence('moderate', 0.25, 0.20),
    Dependence('strong', 0.60, 0.25),
    # Separate sensitivity to less-correlated paired model residuals.
    Dependence('moderate_pair05', 0.25, 0.20, 0.5),
)


def correlated_field(rng, panel, dep):
    a, n = len(panel.counts), panel.n_clusters
    groups = np.arange(a) // panel.group_size
    return (np.sqrt(dep.peptide_fraction) * rng.normal(size=(1, n))
            + np.sqrt(dep.hla_group_fraction) * rng.normal(size=(groups.max() + 1, n))[groups]
            + np.sqrt(1 - dep.peptide_fraction - dep.hla_group_fraction)
            * rng.normal(size=(a, n)))


def generate(rng, panel, dep, r_a, r_b, floor_fraction=0.20, measurement_sd=0.4):
    """Every marginal is identical; shared fields induce crossed row dependence.

    Measurement noise is synthetic, not estimated from the corpus. Floor censoring
    yields average-rank label ties. Predictions share residuals with correlation
    fixed in dep; candidate gain is achieved by changing latent-signal loading.
    """
    z = correlated_field(rng, panel, dep)
    measurement = correlated_field(rng, panel, dep)
    ea = correlated_field(rng, panel, dep)
    eb = (dep.paired_error_correlation * ea
          + np.sqrt(1 - dep.paired_error_correlation ** 2) * correlated_field(rng, panel, dep))
    floor = norm.ppf(floor_fraction) * np.sqrt(1 + measurement_sd ** 2)
    y = np.maximum(z + measurement_sd * measurement - floor, 0)
    pa = r_a * z + np.sqrt(1 - r_a ** 2) * ea
    pb = r_b * z + np.sqrt(1 - r_b ** 2) * eb
    return y, pa, pb


def rank_correlation(y, p):
    """Average-rank Pearson correlation; constant predictions contribute zero."""
    yr = rankdata(y, method='average', axis=-1)
    pr = rankdata(p, method='average', axis=-1)
    yr = yr - yr.mean(axis=-1, keepdims=True)
    pr = pr - pr.mean(axis=-1, keepdims=True)
    denom = np.sqrt((yr ** 2).sum(axis=-1) * (pr ** 2).sum(axis=-1))
    out = np.divide((yr * pr).sum(axis=-1), denom,
                    out=np.full(denom.shape, UNRANKED_CONTRIBUTION), where=denom > 0)
    return np.where((yr ** 2).sum(axis=-1) > 0, out, np.nan)


def endpoint(y, pa, pb, panel, min_rows=20):
    for name, values in [('labels', y), ('baseline predictions', pa), ('candidate predictions', pb)]:
        validate_finite(values, name)
    a, b = [], []
    # Group equal coverage for vectorised ranking; fixed label/count eligibility.
    for n in sorted(set(panel.counts)):
        if n < min_rows:
            continue
        idx = [i for i, count in enumerate(panel.counts) if count == n]
        a.extend(rank_correlation(y[idx, :n], pa[idx, :n]))
        b.extend(rank_correlation(y[idx, :n], pb[idx, :n]))
    if not a or np.isnan(a).all():
        return float('nan')
    return float(np.nanmedian(b) - np.nanmedian(a))


def weighted_ranks(values, weights):
    """Exact average ranks of a duplicated sample represented by integer weights."""
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    group_weights = np.add.reduceat(weights[:, order], starts, axis=1)
    ranks = np.cumsum(group_weights, axis=1) - (group_weights - 1) / 2
    groups = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(values)]))
    out = np.empty_like(weights, dtype=float)
    out[:, order] = ranks[:, groups]
    return out


def weighted_rho(y, p, weights):
    yr, pr = weighted_ranks(y, weights), weighted_ranks(p, weights)
    total = weights.sum(axis=1)
    centre = (total + 1) / 2
    yr -= centre[:, None]
    pr -= centre[:, None]
    yy, pp = (weights * yr ** 2).sum(axis=1), (weights * pr ** 2).sum(axis=1)
    denominator = np.sqrt(yy * pp)
    rho = np.divide((weights * yr * pr).sum(axis=1), denominator,
                    out=np.full(len(total), UNRANKED_CONTRIBUTION), where=denominator > 0)
    return np.where((total >= 2) & (yy > 0), rho, np.nan)


def bootstrap_deltas(y, pa, pb, panel, draws, min_rows=20):
    """Exact whole-cluster resampling, paired across arms and HLA alleles."""
    weights = np.array([np.bincount(draw, minlength=panel.n_clusters) for draw in draws])
    a, b = [], []
    for i, n in enumerate(panel.counts):
        if n >= min_rows and np.unique(y[i, :n]).size >= 2:
            a.append(weighted_rho(y[i, :n], pa[i, :n], weights[:, :n]))
            b.append(weighted_rho(y[i, :n], pb[i, :n], weights[:, :n]))
    return np.nanmedian(np.array(b), axis=0) - np.nanmedian(np.array(a), axis=0)


def wilson(successes, n):
    if n == 0:
        return (float('nan'), float('nan'))
    p, z = successes / n, 1.959963984540054
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def calibrate(seed=20261012, n=160000):
    """Finite synthetic population calibration, with independent validation draws."""
    rng = np.random.default_rng(seed)
    z, measurement, ea, extra = rng.normal(size=(4, n))
    y = np.maximum(z + .4 * measurement - norm.ppf(.2) * np.sqrt(1.16), 0)
    yr = rankdata(y)
    yr -= yr.mean()
    ynorm = np.linalg.norm(yr)

    def find_loading(target, error):
        lo, hi = 0., .995
        for _ in range(25):
            r = (lo + hi) / 2
            ranks = rankdata(r * z + np.sqrt(1 - r*r) * error)
            ranks -= ranks.mean()
            rho = (yr @ ranks) / (ynorm * np.linalg.norm(ranks))
            if rho < target:
                lo = r
            else:
                hi = r
        return (lo + hi) / 2

    result = []
    r_a = find_loading(.66, ea)
    for pair_corr in (.8, .5):
        error = pair_corr * ea + np.sqrt(1 - pair_corr**2) * extra
        for gain in (0., .02, .05):
            # Under null use identical marginal loadings, not a calibration tweak
            # that changes true baseline/candidate equality.
            r_b = r_a if gain == 0 else find_loading(.66 + gain, error)
            validation = Panel('population_check', (n,))
            dep = Dependence('population_check', 0., 0., pair_corr)
            yv, av, bv = generate(np.random.default_rng(seed + 100), validation, dep, r_a, r_b)
            result.append({'pair_correlation': pair_corr, 'target_gain': gain,
                           'baseline_loading': r_a, 'candidate_loading': r_b,
                           'independent_population_baseline': float(rank_correlation(yv, av)[0]),
                           'independent_population_gain': endpoint(yv, av, bv, validation),
                           'independent_floor_share': float((yv == 0).mean()),
                           'calibration_rows': n})
    return pd.DataFrame(result)


def run(out, seed, pilot, trials, diagnostic_trials, n_boot):
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(seed)
    calibration.to_csv(out / 'population_calibration.csv', index=False)
    rates, diagnostics, draw_rows = [], [], []
    cell = 0
    for panel in PANELS:
        for dep in DEPENDENCE:
            for gain in (0., .02, .05):
                cell += 1
                cal = calibration.query('pair_correlation == @dep.paired_error_correlation and target_gain == @gain').iloc[0]
                ra, rb = cal.baseline_loading, cal.candidate_loading
                pilot_rng = np.random.default_rng(np.random.SeedSequence([seed, cell, 0]))
                eval_rng = np.random.default_rng(np.random.SeedSequence([seed, cell, 1]))
                # Conditional on this DGP, error = sample endpoint - population
                # gain. Independent draws estimate 95% equal-tailed uncertainty.
                pilot_data = [generate(pilot_rng, panel, dep, ra, rb) for _ in range(pilot)]
                pilot_delta = np.array([endpoint(*data, panel) for data in pilot_data])
                changes_panel50 = any(20 <= n < 50 for n in panel.counts)
                pilot_delta50 = (np.array([endpoint(*data, panel, min_rows=50) for data in pilot_data])
                                 if changes_panel50 else pilot_delta.copy())
                del pilot_data
                q025, q975 = np.quantile(pilot_delta - gain, [.025, .975])
                q02550, q97550 = np.quantile(pilot_delta50 - gain, [.025, .975])
                eval_delta = []
                eval_delta50 = []
                boot_lowers = []
                for trial in range(trials):
                    y, pa, pb = generate(eval_rng, panel, dep, ra, rb)
                    delta = endpoint(y, pa, pb, panel)
                    delta50 = endpoint(y, pa, pb, panel, min_rows=50) if changes_panel50 else delta
                    eval_delta.append(delta)
                    eval_delta50.append(delta50)
                    lower, upper = delta - q975, delta - q025
                    draw_rows.append({'cell': cell, 'phase': 'evaluation', 'trial': trial,
                                      'delta': delta, 'planning_ci_lower': lower, 'planning_ci_upper': upper,
                                      'delta_50': delta50, 'planning_ci50_lower': delta50 - q97550,
                                      'planning_ci50_upper': delta50 - q02550})
                    if trial < diagnostic_trials:
                        boot_rng = np.random.default_rng(np.random.SeedSequence([seed, cell, trial, 2]))
                        draws = boot_rng.choice(panel.n_clusters, (n_boot, panel.n_clusters), replace=True)
                        boot = bootstrap_deltas(y, pa, pb, panel, draws)
                        bl, bu = np.nanquantile(boot, [.025, .975])
                        boot_lowers.append(bl)
                        diagnostics.append({'cell': cell, 'trial': trial, 'delta': delta,
                                            'bootstrap_lower': bl, 'bootstrap_upper': bu,
                                            'bootstrap_positive': bl > 0,
                                            'planning_lower': lower, 'planning_upper': upper,
                                            'planning_positive': lower > 0,
                                            'bootstrap_covers_gain': bl <= gain <= bu,
                                            'planning_covers_gain': lower <= gain <= upper,
                                            'bootstrap_undefined_draws': int(np.isnan(boot).sum()),
                                            'bootstrap_resamples': n_boot})
                for trial, delta in enumerate(pilot_delta):
                    draw_rows.append({'cell': cell, 'phase': 'calibration', 'trial': trial, 'delta': delta,
                                      'delta_50': pilot_delta50[trial]})
                eval_delta = np.array(eval_delta)
                eval_delta50 = np.array(eval_delta50)
                lower, upper = eval_delta - q975, eval_delta - q025
                positive = int((lower > 0).sum())
                clears_bar = int((lower > .05).sum())
                wl, wu = wilson(positive, trials)
                mc_rng = np.random.default_rng(np.random.SeedSequence([seed, cell, 3]))
                # Resampling the two independent Monte Carlo ensembles quantifies
                # rate uncertainty from estimating the error quantile as well.
                mc_quantiles = np.quantile(mc_rng.choice(pilot_delta - gain, (400, pilot)), .975, axis=1)
                mc_eval = mc_rng.choice(eval_delta, (400, trials))
                mc_rates = (mc_eval > mc_quantiles[:, None]).mean(axis=1)
                mc_low, mc_high = np.quantile(mc_rates, [.025, .975])
                rates.append({'cell': cell, 'panel': panel.name, 'dependence': dep.name,
                              'target_gain': gain, 'measurement_rows': sum(panel.counts),
                              'peptide_clusters': panel.n_clusters, 'alleles': len(panel.counts),
                              'eligible_alleles_20': sum(n >= 20 for n in panel.counts),
                              'eligible_alleles_50': sum(n >= 50 for n in panel.counts),
                              'hla_groups': int(np.ceil(len(panel.counts) / panel.group_size)),
                              'min_rows': min(panel.counts), 'max_rows': max(panel.counts),
                              'pilot_trials': pilot, 'evaluation_trials': trials,
                              'sampling_mean_delta': float(eval_delta.mean()),
                              'sampling_sd_delta': float(eval_delta.std(ddof=1)),
                              'sampling_bias_vs_target': float(eval_delta.mean() - gain),
                              'error_q025': q025, 'error_q975': q975,
                              'planning_positive_rate': positive / trials,
                              'positive_rate_mc95_lower': wl, 'positive_rate_mc95_upper': wu,
                              'positive_rate_joint_mc95_lower': mc_low,
                              'positive_rate_joint_mc95_upper': mc_high,
                              'planning_above_005_rate': clears_bar / trials,
                              'planning_ci_coverage': float(((lower <= gain) & (gain <= upper)).mean()),
                              'sampling_mean_delta_50': float(eval_delta50.mean()),
                              'sampling_sd_delta_50': float(eval_delta50.std(ddof=1)),
                              'planning_positive_rate_50': float((eval_delta50 - q97550 > 0).mean()),
                              'planning_ci_coverage_50': float(((eval_delta50 - q97550 <= gain)
                                                               & (gain <= eval_delta50 - q02550)).mean()),
                              'diagnostic_bootstrap_positive_rate': float(np.mean(np.array(boot_lowers) > 0))})
            print(f'Finished {panel.name} / {dep.name}', flush=True)
    pd.DataFrame(rates).to_csv(out / 'simulation_summary.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out / 'bootstrap_diagnostics.csv', index=False)
    pd.DataFrame(draw_rows).to_csv(out / 'simulation_draws.csv.gz', index=False,
                                 compression={'method': 'gzip', 'mtime': 0})
    manifest = {'seed': seed, 'pilot_trials_per_cell': pilot, 'evaluation_trials_per_cell': trials,
                'diagnostic_trials_per_cell': diagnostic_trials, 'diagnostic_bootstrap_resamples': n_boot,
                'cells': cell, 'elapsed_seconds': round(time.time() - started, 2),
                'panels': [asdict(p) for p in PANELS], 'dependence': [asdict(d) for d in DEPENDENCE],
                'baseline_population_spearman_assumption': .66, 'floor_fraction_assumption': .20,
                'measurement_sd_assumption': .4, 'gains': [0., .02, .05],
                'min_rows': 20, 'sensitivity_min_rows': 50,
                'planning_interval': 'independent parametric endpoint error quantile inversion; known synthetic DGP',
                'confirmatory_interval': 'not replaced; frozen evaluation paired whole-peptide-cluster 2000 percentile bootstrap',
                'external_labels_read': False, 'empirical_noise_ceiling_estimated': False,
                'scope': 'fixed chosen HLA panel; not uncertainty over unseen HLA population',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'evaluation_sha256': hashlib.sha256((ROOT / 'pepstab/evaluation.py').read_bytes()).hexdigest(),
                'audit_eligibility_sha256': hashlib.sha256((ROOT / 'reports/further_research/round2/independent_data/confirmation_eligibility.csv').read_bytes()).hexdigest(),
                'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/further_research/round2/independent_data/design')
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--pilot', type=int, default=1000)
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--diagnostic-trials', type=int, default=30)
    parser.add_argument('--bootstrap', type=int, default=399)
    args = parser.parse_args()
    if min(args.pilot, args.trials, args.diagnostic_trials, args.bootstrap) < 1:
        parser.error('simulation counts must be positive')
    if args.diagnostic_trials > args.trials:
        parser.error('diagnostic trials cannot exceed evaluation trials')
    run(args.out, args.seed, args.pilot, args.trials, args.diagnostic_trials, args.bootstrap)


if __name__ == '__main__':
    main()
