"""Test whether MS-eluted ligands have higher measured pMHC stability.

Joins the MHC Motif Atlas class I ligand list to the Rasmussen half-life dataset on
(allele, peptide) and compares the half-life distribution of peptides that were
observed by immunopeptidomics against those that were not, on the alleles the two
sources share.

    python 04_elution_stability_test.py \
        --stability rasmussen_et_al_dataset.csv \
        --atlas data_classI_all_peptides.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ATLAS_ALLELE = re.compile(r"^([ABC])(\d{2})(\d{2,3})$")
MIN_ELUTED_PER_ALLELE = 3


def atlas_allele_to_standard(name: str) -> str | None:
    """A0201 -> HLA-A*02:01; returns None for non-human / unparseable entries."""
    m = ATLAS_ALLELE.match(str(name).strip())
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}" if m else None


def common_language_effect(a: np.ndarray, b: np.ndarray) -> float:
    """P(random a > random b), ties counted as half."""
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability", default="rasmussen_et_al_dataset.csv")
    parser.add_argument("--atlas", default="data_classI_all_peptides.txt")
    parser.add_argument("--out", default="elution_stability_test.json")
    args = parser.parse_args()

    stability = pd.read_csv(args.stability)
    stability["allele_std"] = stability.allele.str.replace("(C67S)", "", regex=False)

    atlas = pd.read_csv(args.atlas, sep="\t")
    atlas["allele_std"] = atlas.Allele.map(atlas_allele_to_standard)
    atlas9 = atlas[(atlas.Peptide.str.len() == 9) & atlas.allele_std.notna()]
    atlas9 = atlas9[atlas9.allele_std.isin(set(stability.allele_std))]

    shared = set(atlas9.allele_std)
    d = stability[stability.allele_std.isin(shared)].copy()
    eluted_pairs = set(zip(atlas9.allele_std, atlas9.Peptide))
    d["eluted"] = [k in eluted_pairs for k in zip(d.allele_std, d.peptide)]

    hit, miss = d.loc[d.eluted, "thalf_hours"], d.loc[~d.eluted, "thalf_hours"]
    test = mannwhitneyu(hit, miss, alternative="greater")

    def describe(s: pd.Series) -> dict:
        return {"n": int(len(s)), "median_thalf_h": round(float(s.median()), 2),
                "mean_thalf_h": round(float(s.mean()), 2),
                "frac_censored": round(float((s == 0).mean()), 3),
                "frac_ge_1h": round(float((s >= 1).mean()), 3)}

    per = (d.groupby("allele_std")
             .apply(lambda g: pd.Series({
                 "n_eluted": int(g.eluted.sum()),
                 "median_eluted": g.loc[g.eluted, "thalf_hours"].median(),
                 "median_other": g.loc[~g.eluted, "thalf_hours"].median()}),
                    include_groups=False)
             .dropna(subset=["median_eluted"]))
    per = per[per.n_eluted >= MIN_ELUTED_PER_ALLELE]

    counter = (d[d.eluted & (d.thalf_hours < 0.5)][["allele_std", "peptide", "thalf_hours"]]
                 .sort_values("thalf_hours"))

    result = {
        "atlas_rows_total": int(len(atlas)),
        "atlas_9mers_on_stability_alleles": int(len(atlas9)),
        "shared_alleles": len(shared),
        "stability_rows_on_shared_alleles": int(len(d)),
        "eluted": describe(hit),
        "not_eluted": describe(miss),
        "mannwhitney_u": float(test.statistic),
        "mannwhitney_p_greater": float(test.pvalue),
        "common_language_effect": round(
            common_language_effect(hit.to_numpy(), miss.to_numpy()), 3),
        "per_allele_min_eluted": MIN_ELUTED_PER_ALLELE,
        "per_allele_alleles_tested": int(len(per)),
        "per_allele_eluted_median_higher": int((per.median_eluted > per.median_other).sum()),
        "counterexamples": counter.to_dict(orient="records"),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    per.round(2).to_csv(Path(args.out).with_name("elution_stability_per_allele.csv"))
    print(json.dumps({k: v for k, v in result.items() if k != "counterexamples"}, indent=2))


if __name__ == "__main__":
    main()
