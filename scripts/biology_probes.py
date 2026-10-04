"""Label-side probes on the stability panel: what the data says before any model.

    .venv/bin/python scripts/biology_probes.py

Writes ``reports/allele_pair_concordance.csv``, ``reports/position_eta2.csv``
and ``reports/biology_probes.json``. CPU, well under a minute. Reads only the
read-only raw CSV: none of these quantities are model outputs, so none of them
depend on the splits, and nothing here can leak.

Adapted from the ``stage3-esm`` branch, which ran these and reported them in
``reports/BIOLOGY_NOTES.md``. Three changes:

1. **The BLOSUM62 matrix is the repo's**, via
   :func:`pepstab.features.residue_rows`, instead of a second copy pasted into
   this file. A duplicated substitution matrix is a thing that can silently
   diverge from the one the models use. The groove distance is a normalised
   kernel, so the repo's 1/5 scaling cancels and the numbers are unchanged --
   asserted against the branch's published values in ``tests/test_biology.py``.
2. **The noise-ceiling probe reports every near-identical pair, not the best
   four.** The branch's §4 quotes four pairs at rho 0.901-0.921 and reads as
   though those are the only pairs differing at one contact residue. There are
   twelve, and the other eight run 0.657-0.838. The lower bound survives --
   reproducibility is at least the *maximum* observed concordance, since
   observed = reproducibility x true similarity and true similarity is below 1
   -- but it has to be stated as the best of twelve, with the count visible.
3. **The hamming-0 pair is separated out.** Two alleles identical across all 34
   contact residues are a strictly better reproducibility probe than two that
   differ at one, because true similarity on the contact surface is 1 by
   construction. The branch's table binned it away.
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import AA_INDEX, load_raw  # noqa: E402
from pepstab.features import residue_rows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"

#: A pair needs this many shared peptides before its rho means anything.
MIN_SHARED_PEPTIDES = 50
#: An allele needs this many rows before it enters the position-variance probe.
MIN_ROWS_PER_ALLELE = 100
SEED = 20261003
PEPTIDE_LENGTH = 9


def groove_distance(a: str, b: str, b62: np.ndarray) -> float:
    """``1 - `` normalised BLOSUM62 kernel between two 34-residue pseudosequences.

    Scale-invariant in ``b62``: the normalisation divides by
    ``sqrt(k(a,a) k(b,b))``, so the repo's 1/5 scaling cancels exactly.
    """
    ia = np.fromiter((AA_INDEX[c] for c in a), dtype=np.int64, count=len(a))
    ib = np.fromiter((AA_INDEX[c] for c in b), dtype=np.int64, count=len(b))
    kab = float(b62[ia, ib].sum())
    kaa = float(b62[ia, ia].sum())
    kbb = float(b62[ib, ib].sum())
    return 1 - kab / np.sqrt(kaa * kbb)


def allele_pair_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """Rank agreement between every pair of alleles on the peptides they share.

    The upper end of this distribution is the closest thing this dataset has to
    a noise ceiling. Two *different* HLA molecules measured on the same peptides
    cannot agree better than the assay resolves, so an observed rho is a lower
    bound on assay reproducibility -- and the bound is tightest for the pair
    whose binding surfaces differ least.
    """
    b62 = residue_rows("blosum")
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
            "groove_distance": groove_distance(pseudo[a], pseudo[b], b62),
            "hamming": int(sum(x != y for x, y in zip(pseudo[a], pseudo[b]))),
            "floor_share_max": float(max(
                (df[df.allele == a].thalf_hours == 0).mean(),
                (df[df.allele == b].thalf_hours == 0).mean())),
        })
    return pd.DataFrame(rows).sort_values(["hamming", "rho"],
                                          ascending=[True, False])


def noise_ceiling(conc: pd.DataFrame) -> dict:
    """A lower bound on assay reproducibility, from the most similar allele pairs.

    Reported at Hamming 0 and Hamming <= 1 separately, each with the number of
    pairs the bound is drawn from, because "the best of twelve" and "all four
    agree" are very different evidential claims and the distinction is the one
    the branch's write-up lost.
    """
    out: dict = {}
    for label, mask in (("hamming_0", conc.hamming == 0),
                        ("hamming_le_1", conc.hamming <= 1)):
        sub = conc[mask]
        if sub.empty:
            out[label] = None
            continue
        out[label] = {
            "n_pairs": int(len(sub)),
            "max_rho": round(float(sub.rho.max()), 4),
            "median_rho": round(float(sub.rho.median()), 4),
            "min_rho": round(float(sub.rho.min()), 4),
            "best_pair": f"{sub.loc[sub.rho.idxmax(), 'a']} / "
                         f"{sub.loc[sub.rho.idxmax(), 'b']}",
            "best_pair_n_shared": int(sub.loc[sub.rho.idxmax(), "n_shared"]),
        }
    # Concordance degrades on floor-heavy alleles, so the bound is drawn from
    # lightly-censored panels and does not transfer to the censored ones.
    sub = conc[conc.hamming <= 4]
    if len(sub) > 3:
        out["rho_vs_censoring_spearman"] = round(float(
            stats.spearmanr(sub.rho, sub.floor_share_max).statistic), 4)
    return out


def position_variance(df: pd.DataFrame, n_perm: int = 20) -> pd.DataFrame:
    """Permutation-adjusted eta^2 of log half-life on each peptide position.

    Raw eta^2 is biased upward by the number of residue groups, and the groups
    differ by position: anchor positions carry fewer distinct residues, because
    the assay panel was pre-selected for predicted binding affinity. Subtracting
    the within-allele label-permutation mean removes that bias, so positions
    become comparable to each other.
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
        for pos in range(PEPTIDE_LENGTH):
            res = np.array([p[pos] for p in peptides])
            masks = [res == r for r in np.unique(res)]

            def eta2(vals: np.ndarray) -> float:
                m = vals.mean()
                between = sum(((vals[k].mean() - m) ** 2) * k.sum() for k in masks)
                return between / ((vals - m) ** 2).sum()

            null = float(np.mean([eta2(rng.permutation(y)) for _ in range(n_perm)]))
            share = np.array([k.sum() for k in masks], dtype=float)
            share /= share.sum()
            rows.append({
                "allele": allele, "position": pos + 1,
                "eta2": eta2(y), "eta2_adj": eta2(y) - null,
                "n_residues": len(masks),
                "entropy_bits": float(-(share * np.log2(share)).sum()),
            })
    return pd.DataFrame(rows)


def identity_oracle(df: pd.DataFrame, n_folds: int = 5, shrink: float = 1.0) -> dict:
    """Out-of-fold additive ``y ~ allele + peptide``, fitted on identities only.

    An oracle in the strict sense: folds are assigned by row, so a held-out row's
    peptide effect was fitted on that same peptide's measurements against *other*
    alleles -- information no sequence model is given. Whatever it still cannot
    rank is peptide x groove interaction, which is the part a pan-specific model
    has to supply from sequence. That makes it an upper bound on how far
    peptide-intrinsic and allele-intrinsic effects can take you.
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
        for _ in range(30):      # alternating least squares; converged well before 30
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
            if len(g) >= MIN_ROWS_PER_ALLELE // 2
            and g.y.nunique() > 1 and g.p.nunique() > 1]
    ss_tot = ((y - y.mean()) ** 2).sum()
    allele_mean = df.groupby("allele")["y_log1p"].transform("mean").to_numpy()
    return {
        "median_per_allele_rho": round(float(np.median(rhos)), 4),
        "n_alleles": len(rhos),
        "r2_out_of_fold": round(float(1 - ((y - pred) ** 2).sum() / ss_tot), 4),
        "r2_allele_main_effect_in_sample": round(
            float(1 - ((y - allele_mean) ** 2).sum() / ss_tot), 4),
    }


def main() -> int:
    df = load_raw()
    REPORT_DIR.mkdir(exist_ok=True)
    out: dict = {"n_rows": int(len(df)), "n_alleles": int(df.allele.nunique())}

    conc = allele_pair_concordance(df)
    conc.to_csv(REPORT_DIR / "allele_pair_concordance.csv", index=False)
    bins = pd.cut(conc.hamming, [-1, 0, 4, 9, 14, 19, 34],
                  labels=["0", "1-4", "5-9", "10-14", "15-19", "20+"])
    print("concordance between alleles, by differing contact residues")
    print(conc.groupby(bins, observed=True).agg(
        pairs=("rho", "size"), median_rho=("rho", "median"),
        max_rho=("rho", "max")).round(3))
    out["n_pairs_scored"] = int(len(conc))
    out["concordance_by_hamming_bin"] = {
        str(k): {"pairs": int(v["pairs"]), "median_rho": round(float(v["median_rho"]), 4)}
        for k, v in conc.groupby(bins, observed=True).agg(
            pairs=("rho", "size"), median_rho=("rho", "median")).iterrows()}

    print("\nevery pair differing at <= 1 of the 34 contact residues")
    near = conc[conc.hamming <= 1][["a", "b", "n_shared", "rho", "hamming",
                                    "floor_share_max"]]
    print(near.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ceiling = noise_ceiling(conc)
    out["noise_ceiling"] = ceiling
    print("\nnoise ceiling (lower bound on assay reproducibility)")
    for k, v in ceiling.items():
        print(f"  {k}: {v}")

    eta = position_variance(df)
    eta.to_csv(REPORT_DIR / "position_eta2.csv", index=False)
    med = eta.groupby("position").eta2_adj.median()
    print("\nvariance explained per peptide position "
          "(permutation-adjusted, median over alleles)")
    print(med.round(3))
    out["position_eta2_adj_median"] = {int(k): round(float(v), 4)
                                       for k, v in med.items()}
    out["position_entropy_bits_median"] = {
        int(k): round(float(v), 3)
        for k, v in eta.groupby("position").entropy_bits.median().items()}

    oracle = identity_oracle(df)
    out["identity_oracle"] = oracle
    print("\nidentity oracle (allele + peptide main effects, no sequence)")
    for k, v in oracle.items():
        print(f"  {k}: {v}")

    floor = df.groupby("allele").thalf_hours.apply(lambda s: float((s == 0).mean()))
    print("\nmost-censored alleles (share of labels at the assay floor)")
    print(floor.sort_values(ascending=False).head(5).round(3))
    out["most_censored_alleles"] = {str(k): round(float(v), 4) for k, v
                                    in floor.sort_values(ascending=False).head(5).items()}

    (REPORT_DIR / "biology_probes.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/allele_pair_concordance.csv, reports/position_eta2.csv, "
          "reports/biology_probes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
