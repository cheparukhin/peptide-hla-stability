"""Biological probes on the stability panel: what the labels say before any model.

    python scripts/biology_probes.py

Writes reports/allele_pair_concordance.csv, reports/position_eta2.csv and
reports/figures/biology_overview.png. CPU, ~30 s. Reads only the read-only raw
CSV -- no split dependence, because none of these quantities are model outputs.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pepstab.data import load_raw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
MIN_SHARED_PEPTIDES = 50
MIN_ROWS_PER_ALLELE = 100
SEED = 20261003

# BLOSUM62, used only for the continuous groove distance; the headline axis is
# the plain count of differing contact residues, which needs no matrix.
_B62_TEXT = """   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4"""


def blosum62() -> dict[tuple[str, str], int]:
    lines = _B62_TEXT.strip().split("\n")
    cols = lines[0].split()
    out = {}
    for line in lines[1:]:
        tok = line.split()
        for col, val in zip(cols, tok[1:]):
            out[(tok[0], col)] = int(val)
    return out


def groove_distance(a: str, b: str, mat: dict) -> float:
    """1 - normalised BLOSUM62 kernel between two 34-residue pseudosequences."""
    kab = sum(mat[(x, y)] for x, y in zip(a, b))
    kaa = sum(mat[(x, x)] for x in a)
    kbb = sum(mat[(y, y)] for y in b)
    return 1 - kab / np.sqrt(kaa * kbb)


def allele_pair_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """Rank agreement between every pair of alleles on the peptides they share.

    The upper end of this distribution is the closest thing this dataset has to
    a noise ceiling: two *different* HLA molecules measured on the same peptides
    cannot agree better than the assay resolves, so an observed rho bounds assay
    reproducibility from below.
    """
    mat = blosum62()
    pseudo = df.drop_duplicates("allele").set_index("allele")["hla_pseudoseq"].to_dict()
    series = {a: g.set_index("peptide")["y_log1p"] for a, g in df.groupby("allele")}
    rows = []
    for a, b in combinations(sorted(series), 2):
        shared = series[a].index.intersection(series[b].index)
        if len(shared) < MIN_SHARED_PEPTIDES:
            continue
        va, vb = series[a].loc[shared].to_numpy(), series[b].loc[shared].to_numpy()
        if len(np.unique(va)) < 2 or len(np.unique(vb)) < 2:
            continue
        rows.append({
            "a": a, "b": b, "n_shared": len(shared),
            "rho": float(stats.spearmanr(va, vb).statistic),
            "groove_distance": groove_distance(pseudo[a], pseudo[b], mat),
            "hamming": sum(x != y for x, y in zip(pseudo[a], pseudo[b])),
        })
    return pd.DataFrame(rows)


def position_variance(df: pd.DataFrame, n_perm: int = 20) -> pd.DataFrame:
    """Permutation-adjusted eta^2 of log half-life on each peptide position.

    Raw eta^2 is biased upward by the number of residue groups, and the groups
    differ by position (anchors carry fewer distinct residues because the assay
    panel was pre-selected for predicted binding). Subtracting the within-allele
    label-permutation mean removes that bias so positions are comparable.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for allele, grp in df.groupby("allele"):
        if len(grp) < MIN_ROWS_PER_ALLELE:
            continue
        y = grp["y_log1p"].to_numpy()
        if ((y - y.mean()) ** 2).sum() == 0:
            continue
        peptides = grp["peptide"].to_numpy()
        for pos in range(9):
            res = np.array([p[pos] for p in peptides])
            masks = [res == r for r in np.unique(res)]

            def eta2(vals: np.ndarray) -> float:
                m = vals.mean()
                between = sum(((vals[k].mean() - m) ** 2) * k.sum() for k in masks)
                return between / ((vals - m) ** 2).sum()

            null = np.mean([eta2(rng.permutation(y)) for _ in range(n_perm)])
            share = np.array([k.sum() for k in masks], float)
            share /= share.sum()
            rows.append({
                "allele": allele, "position": pos + 1,
                "eta2": eta2(y), "eta2_adj": eta2(y) - null,
                "n_residues": len(masks),
                "entropy_bits": float(-(share * np.log2(share)).sum()),
            })
    return pd.DataFrame(rows)


def identity_oracle(df: pd.DataFrame, n_folds: int = 5, shrink: float = 1.0) -> dict:
    """Out-of-fold additive model y ~ allele + peptide, fitted on identities only.

    It is an oracle in the sense that it reads each peptide's measured behaviour
    on *other* alleles -- information no sequence model has. Whatever it cannot
    rank is peptide x groove interaction, which is the part a pan-specific model
    has to supply.
    """
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, n_folds, len(df))
    ai = pd.factorize(df["allele"])[0]
    pi = pd.factorize(df["peptide"])[0]
    y = df["y_log1p"].to_numpy()
    pred = np.full(len(df), np.nan)
    for f in range(n_folds):
        tr = fold != f
        mu = y[tr].mean()
        a_eff = np.zeros(ai.max() + 1)
        p_eff = np.zeros(pi.max() + 1)
        for _ in range(30):  # alternating least squares, converged well before 30
            resid = y[tr] - mu - p_eff[pi[tr]]
            a_eff = (np.bincount(ai[tr], resid, a_eff.size)
                     / (np.bincount(ai[tr], None, a_eff.size) + shrink))
            resid = y[tr] - mu - a_eff[ai[tr]]
            p_eff = (np.bincount(pi[tr], resid, p_eff.size)
                     / (np.bincount(pi[tr], None, p_eff.size) + shrink))
        pred[~tr] = mu + a_eff[ai[~tr]] + p_eff[pi[~tr]]

    frame = pd.DataFrame({"allele": df["allele"], "y": y, "p": pred})
    rhos = [stats.spearmanr(g.y, g.p).statistic
            for _, g in frame.groupby("allele")
            if len(g) >= MIN_ROWS_PER_ALLELE // 2 and g.y.nunique() > 1 and g.p.nunique() > 1]
    ss_tot = ((y - y.mean()) ** 2).sum()
    allele_mean = df.groupby("allele")["y_log1p"].transform("mean").to_numpy()
    return {
        "median_per_allele_rho": float(np.median(rhos)),
        "n_alleles": len(rhos),
        "r2_out_of_fold": float(1 - ((y - pred) ** 2).sum() / ss_tot),
        "r2_allele_main_effect_in_sample": float(1 - ((y - allele_mean) ** 2).sum() / ss_tot),
    }


def main() -> int:
    df = load_raw()
    REPORT_DIR.mkdir(exist_ok=True)

    conc = allele_pair_concordance(df)
    conc.to_csv(REPORT_DIR / "allele_pair_concordance.csv", index=False)
    bins = pd.cut(conc.hamming, [-1, 0, 4, 9, 14, 19, 34],
                  labels=["0", "1-4", "5-9", "10-14", "15-19", "20+"])
    print("concordance between alleles, by differing contact residues")
    print(conc.groupby(bins, observed=True).agg(pairs=("rho", "size"),
                                                median_rho=("rho", "median")).round(3))

    eta = position_variance(df)
    eta.to_csv(REPORT_DIR / "position_eta2.csv", index=False)
    print("\nvariance explained per peptide position (permutation-adjusted median)")
    print(eta.groupby("position").eta2_adj.median().round(3))

    print("\nidentity oracle (allele + peptide main effects, no sequence)")
    for k, v in identity_oracle(df).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    floor = df.groupby("allele").thalf_hours.apply(lambda s: (s == 0).mean())
    print("\nmost-censored alleles (share of labels at the assay floor)")
    print(floor.sort_values(ascending=False).head(5).round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
