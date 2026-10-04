"""Stage 3c: the MHC Motif Atlas as an external scoring target.

The atlas is a mass-spectrometry ligand list -- allele and peptide, nothing
else. It is **never training data** (four reasons in HACKATHON_PLAN.md 3c and
``elution_stability_finding.md``). This module supplies the pieces of a
scoring-only pass: parse the atlas, build the human-proteome decoy universe,
draw length- and allele-matched decoys under a predeclared ratio and seed, and
score the ligand-vs-decoy ranking.

Nothing here fits a parameter. The only randomness is decoy sampling and the
ligand cap, both from one seeded generator whose seed is a module constant, so
the scoring set is byte-identical across runs and can be handed to another arm.

**Symmetric filtering.** Positives and decoys pass through the *same* exclusion
list: any peptide that appears anywhere in the Rasmussen half-life CSV is
dropped from both. Filtering only the decoys would make "absent from the
training data" a property of the negative class, and the model could then score
well on the filter rather than on the biology.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import AMINO_ACIDS, PEPTIDE_LENGTH

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = REPO_ROOT / "external"

#: Downloaded from http://mhcmotifatlas.org/data/classI/all_peptides.txt
#: (the "Download all peptides" link on the Class I Alleles page; a browser
#: saves it under the flattened name used here).
ATLAS_TXT = EXTERNAL_DIR / "data_classI_all_peptides.txt"
ATLAS_URL = "http://mhcmotifatlas.org/data/classI/all_peptides.txt"

#: UniProt reviewed (Swiss-Prot) human proteome, already present locally.
PROTEOME_FASTA = EXTERNAL_DIR / "uniprot_human_reviewed.fasta.gz"

# --- predeclared decoy protocol -------------------------------------------
# Fixed before any score was computed. AUROC moves with the decoy ratio, so a
# ratio picked after seeing results is not a result.

#: Decoys drawn per atlas ligand, within each allele.
DECOY_RATIO = 10

#: The one seed. Decoy draws and the ligand cap both come from it.
SEED = 20261004

#: Alleles need at least this many distinct atlas 9-mer ligands (after the
#: shared exclusion filter) to be scored. Chosen from the count distribution
#: alone: the smallest shared allele has 101 ligands, so this bar is set where
#: it barely binds and no allele is dropped for scoring badly.
MIN_LIGANDS_PER_ALLELE = 100

#: Ligands scored per allele. Alleles above this are subsampled uniformly at
#: :data:`SEED`. A CPU-budget cap, not a statistical one -- at 2,000 positives
#: against 20,000 decoys the standard error on AUROC is under 0.01, so the cap
#: costs no resolution while keeping the pass to a few minutes on one laptop.
MAX_LIGANDS_PER_ALLELE = 2000

#: Top-ranked fractions reported as enrichment.
ENRICHMENT_FRACTIONS = (0.01, 0.10)

#: ``A0201`` -> ``HLA-A*02:01``. Same convention as
#: ``04_elution_stability_test.py``; non-human and non-ABC entries (``H2-Kb``,
#: ``G0101``, ...) do not match and are dropped.
ATLAS_ALLELE_RE = re.compile(r"^([ABC])(\d{2})(\d{2,3})$")

_STANDARD_AA = set(AMINO_ACIDS)


def atlas_allele_to_standard(name: str) -> str | None:
    """``A0201`` -> ``HLA-A*02:01``; ``None`` for anything unparseable."""
    m = ATLAS_ALLELE_RE.match(str(name).strip())
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}" if m else None


def strip_c67s(allele: pd.Series) -> pd.Series:
    """``HLA-B*14:01(C67S)`` -> ``HLA-B*14:01``."""
    return allele.str.replace("(C67S)", "", regex=False)


def load_atlas(path: Path = ATLAS_TXT) -> pd.DataFrame:
    """The raw atlas table with a standardised allele column.

    Two columns in the file, ``Allele`` and ``Peptide``. Returns every row,
    including unparseable alleles and every peptide length -- filtering is the
    caller's job so the counts it drops are visible.
    """
    atlas = pd.read_csv(path, sep="\t", dtype={"Allele": str, "Peptide": str})
    atlas = atlas.rename(columns={"Allele": "atlas_allele", "Peptide": "peptide"})
    atlas["allele"] = atlas.atlas_allele.map(atlas_allele_to_standard)
    return atlas


#: A peptide the model can encode: nine standard, unmodified residues. The
#: atlas marks phosphorylated residues in lower case (``LSsRtNLQY``); 2,269 of
#: its 286,974 9-mer rows carry one. They are dropped from the ligand set --
#: the stability assay has no phosphopeptides and the encoder has no column
#: for a modified residue -- but their unmodified backbone still blocks the
#: corresponding proteome 9-mer from being drawn as a decoy.
STANDARD_9MER = re.compile(rf"^[{AMINO_ACIDS}]{{{PEPTIDE_LENGTH}}}$")


def is_standard_9mer(peptide: pd.Series) -> pd.Series:
    return peptide.str.fullmatch(STANDARD_9MER).fillna(False)


def atlas_nonamers(atlas: pd.DataFrame) -> np.ndarray:
    """Every distinct 9-mer in the atlas, on any allele, human or not.

    The decoy exclusion list. Broader than "a ligand on *this* allele" on
    purpose: a peptide eluted from any MHC molecule is a presented peptide, and
    calling it a decoy anywhere would be scoring noise. Phospho-marked
    sequences are upper-cased first, so the backbone they were eluted as is
    blocked too.
    """
    pep = atlas.peptide[atlas.peptide.str.len() == PEPTIDE_LENGTH]
    return np.unique(pep.str.upper().to_numpy(dtype="S9"))


def proteome_nonamers(path: Path = PROTEOME_FASTA) -> np.ndarray:
    """Every distinct 9-mer window of the reviewed human proteome.

    Overlapping windows, one per start position. Windows containing a
    non-standard residue (``U``, ``X``, ``B``, ``Z``, ``O``) are dropped: the
    model's encoder rejects them, and they are not peptides anyone would
    propose as a decoy.
    """
    windows: list[bytes] = []
    with gzip.open(path, "rt") as fh:
        chunks: list[str] = []
        for line in fh:
            if line.startswith(">"):
                if chunks:
                    windows.extend(_windows("".join(chunks)))
                chunks = []
            else:
                chunks.append(line.strip())
        if chunks:
            windows.extend(_windows("".join(chunks)))
    return np.unique(np.array(windows, dtype="S9"))


def _windows(seq: str) -> list[bytes]:
    n = PEPTIDE_LENGTH
    return [w.encode() for i in range(len(seq) - n + 1)
            if _STANDARD_AA.issuperset(w := seq[i:i + n])]


def build_scoring_set(
    atlas: pd.DataFrame,
    stability_peptides: set[str],
    allele_pseudoseq: dict[str, str],
    *,
    decoy_ratio: int = DECOY_RATIO,
    seed: int = SEED,
    min_ligands: int = MIN_LIGANDS_PER_ALLELE,
    max_ligands: int = MAX_LIGANDS_PER_ALLELE,
    proteome: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """The predeclared ligand/decoy table, plus a provenance dict.

    ``allele_pseudoseq`` fixes the allele panel: only alleles the stability
    dataset carries a pseudosequence for can be scored, because the model eats
    one. Returns a frame with ``allele``, ``peptide``, ``label`` (1 ligand,
    0 decoy) and ``hla_pseudoseq``.
    """
    rng = np.random.default_rng(seed)
    audit: dict = {"decoy_ratio": decoy_ratio, "seed": seed,
                   "min_ligands_per_allele": min_ligands,
                   "max_ligands_per_allele": max_ligands,
                   "atlas_rows": int(len(atlas))}

    excluded = np.unique(np.array(sorted(stability_peptides), dtype="S9"))
    audit["stability_peptides_excluded_from_both_classes"] = int(len(excluded))

    nine = atlas[(atlas.peptide.str.len() == PEPTIDE_LENGTH) & atlas.allele.notna()]
    audit["atlas_9mers_parsed_allele"] = int(len(nine))
    standard = is_standard_9mer(nine.peptide)
    audit["atlas_9mers_dropped_modified_residue"] = int((~standard).sum())
    nine = nine[standard]
    nine = nine[nine.allele.isin(allele_pseudoseq)]
    nine = nine.drop_duplicates(["allele", "peptide"])
    audit["atlas_9mer_pairs_on_panel_alleles"] = int(len(nine))

    keep = ~nine.peptide.isin(stability_peptides)
    audit["ligands_dropped_measured_in_stability_csv"] = int((~keep).sum())
    nine = nine[keep]

    counts = nine.allele.value_counts()
    eligible = sorted(counts[counts >= min_ligands].index)
    audit["alleles_before_min_ligands"] = int(len(counts))
    audit["alleles_scored"] = len(eligible)
    audit["alleles_dropped_too_few_ligands"] = sorted(
        counts[counts < min_ligands].index)

    # --- decoy universe: proteome minus every atlas ligand and every
    #     measured peptide, so a decoy is a peptide neither assay has seen.
    if proteome is None:
        proteome = proteome_nonamers()
    audit["proteome_distinct_9mers"] = int(len(proteome))
    atlas9 = atlas_nonamers(atlas)
    audit["atlas_distinct_9mers_any_allele"] = int(len(atlas9))
    blocked = np.union1d(atlas9, excluded)
    allowed = proteome[~np.isin(proteome, blocked, assume_unique=True)]
    audit["decoy_universe"] = int(len(allowed))
    audit["decoy_universe_removed"] = int(len(proteome) - len(allowed))

    rows = []
    per_allele = []
    for allele in eligible:
        ligands = np.sort(nine.loc[nine.allele == allele, "peptide"].to_numpy(dtype=str))
        if len(ligands) > max_ligands:
            ligands = ligands[rng.choice(len(ligands), max_ligands, replace=False)]
        n_dec = decoy_ratio * len(ligands)
        decoys = allowed[rng.choice(len(allowed), n_dec, replace=False)].astype(str)
        rows.append(pd.DataFrame({"allele": allele, "peptide": ligands, "label": 1}))
        rows.append(pd.DataFrame({"allele": allele, "peptide": decoys, "label": 0}))
        per_allele.append({"allele": allele,
                           "n_ligands_available": int(counts[allele]),
                           "n_ligands_scored": int(len(ligands)),
                           "n_decoys": int(n_dec)})

    scoring = pd.concat(rows, ignore_index=True)
    scoring["hla_pseudoseq"] = scoring.allele.map(allele_pseudoseq)
    audit["rows_scored"] = int(len(scoring))
    audit["ligands_scored"] = int(scoring.label.sum())
    audit["per_allele"] = per_allele
    return scoring, audit


def eligible_atlas_ligands(
    atlas: pd.DataFrame,
    stability_peptides: set[str],
    allele_pseudoseq: dict[str, str],
    *,
    min_ligands: int = MIN_LIGANDS_PER_ALLELE,
) -> pd.DataFrame:
    """The filtered ``(allele, peptide)`` ligand table the panel is drawn from.

    Same filters as :func:`build_scoring_set`, exposed separately because the
    allele-swapped control needs the *uncapped* ligand list of every allele to
    decide which peptides are not negatives for a given target.
    """
    nine = atlas[(atlas.peptide.str.len() == PEPTIDE_LENGTH) & atlas.allele.notna()]
    nine = nine[is_standard_9mer(nine.peptide)]
    nine = nine[nine.allele.isin(allele_pseudoseq)]
    nine = nine.drop_duplicates(["allele", "peptide"])
    nine = nine[~nine.peptide.isin(stability_peptides)]
    counts = nine.allele.value_counts()
    return nine[nine.allele.isin(counts[counts >= min_ligands].index)]


def pseudoseq_distances(allele_pseudoseq: dict[str, str],
                        alleles: list[str]) -> np.ndarray:
    """``(n, n)`` Hamming distance between allele pseudosequences."""
    seqs = np.array([list(allele_pseudoseq[a]) for a in alleles])
    return (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)


def build_swapped_scoring_set(
    ligands: pd.DataFrame,
    primary: pd.DataFrame,
    allele_pseudoseq: dict[str, str],
    *,
    decoy_ratio: int = DECOY_RATIO,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """Allele-swapped control: negatives are other alleles' eluted ligands.

    Proteome decoys are negatives on every axis at once -- not cleaved, not
    transported, not presented, not ionised *and* not stable -- so part of a
    high AUROC is "does this look like a presentable peptide at all". Swapping
    in real eluted ligands from other alleles holds source abundance,
    proteasomal cleavage, TAP transport and MS ionisation roughly constant and
    leaves allele specificity.

    Positives are the *same* capped ligand sample as ``primary``, so the two
    AUROCs differ only in their negatives. A peptide that the target allele
    also presents is never a negative for it. Each decoy carries
    ``donor_distance``: the pseudosequence Hamming distance from the target to
    the nearest allele known to present that peptide -- a small distance means
    the negative came from a similar groove, and the gradient over it says
    whether the model is reading allele specificity or peptide generality.
    """
    rng = np.random.default_rng(seed)
    alleles = sorted(primary.allele.unique())
    index = {a: i for i, a in enumerate(alleles)}
    dist = pseudoseq_distances(allele_pseudoseq, alleles)

    pool = ligands[ligands.allele.isin(alleles)]
    peptides = np.sort(pool.peptide.unique())
    pep_index = {p: i for i, p in enumerate(peptides)}
    presented = np.zeros((len(peptides), len(alleles)), dtype=bool)
    presented[pool.peptide.map(pep_index).to_numpy(),
              pool.allele.map(index).to_numpy()] = True

    rows, audit_rows = [], []
    for allele in alleles:
        t = index[allele]
        lig = primary.loc[(primary.allele == allele) & (primary.label == 1),
                          "peptide"].to_numpy(dtype=str)
        candidate = np.flatnonzero(~presented[:, t])
        n_dec = decoy_ratio * len(lig)
        if n_dec > len(candidate):
            raise ValueError(f"{allele}: only {len(candidate)} swapped decoys "
                             f"available, need {n_dec}")
        pick = candidate[rng.choice(len(candidate), n_dec, replace=False)]
        donor = np.where(presented[pick], dist[t][None, :], np.iinfo(np.int32).max
                         ).min(axis=1)
        rows.append(pd.DataFrame({"allele": allele, "peptide": lig, "label": 1,
                                  "donor_distance": -1}))
        rows.append(pd.DataFrame({"allele": allele, "peptide": peptides[pick],
                                  "label": 0, "donor_distance": donor}))
        audit_rows.append({"allele": allele, "n_ligands": int(len(lig)),
                           "n_decoys": int(n_dec),
                           "n_candidate_decoys": int(len(candidate)),
                           "median_donor_distance": float(np.median(donor))})

    swapped = pd.concat(rows, ignore_index=True)
    swapped["hla_pseudoseq"] = swapped.allele.map(allele_pseudoseq)
    audit = {"decoy_ratio": decoy_ratio, "seed": seed,
             "alleles_scored": len(alleles),
             "decoy_pool_distinct_peptides": int(len(peptides)),
             "rows_scored": int(len(swapped)),
             "ligands_scored": int(swapped.label.sum()),
             "per_allele": audit_rows}
    return swapped, audit


def score_by_donor_distance(swapped: pd.DataFrame, y_pred: np.ndarray,
                            n_bins: int = 2) -> pd.DataFrame:
    """Per-allele AUROC against near vs far donor alleles.

    The same ligands each time; only the negatives change. Bin edges are the
    quantiles of ``donor_distance`` over *all* decoys, so every allele is cut at
    the same distances.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    dec = swapped.donor_distance.to_numpy()
    edges = np.quantile(dec[swapped.label.to_numpy() == 0],
                        np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for allele, idx in swapped.groupby("allele", sort=True).indices.items():
        lab = swapped.label.to_numpy()[idx]
        for b in range(n_bins):
            keep = (lab == 1) | ((dec[idx] > edges[b]) & (dec[idx] <= edges[b + 1]))
            sub = idx[keep]
            if (swapped.label.to_numpy()[sub] == 0).sum() < 20:
                continue
            out.append({"allele": allele, "bin": b,
                        "donor_distance_low": float(edges[b]),
                        "donor_distance_high": float(edges[b + 1]),
                        "n_decoys": int((swapped.label.to_numpy()[sub] == 0).sum()),
                        "auroc": round(roc_auc(swapped.label.to_numpy()[sub],
                                               y_pred[sub]), 4)})
    return pd.DataFrame(out)


# --- metrics ---------------------------------------------------------------


@dataclass(frozen=True)
class AlleleResult:
    allele: str
    n_ligands: int
    n_decoys: int
    auroc: float
    auprc: float
    auprc_baseline: float
    enrichment: dict[float, tuple[float, float]]

    def as_row(self) -> dict:
        row = {"allele": self.allele, "n_ligands": self.n_ligands,
               "n_decoys": self.n_decoys, "auroc": round(self.auroc, 4),
               "auprc": round(self.auprc, 4),
               "auprc_baseline": round(self.auprc_baseline, 4)}
        for frac, (precision, ef) in self.enrichment.items():
            tag = f"{frac:.0%}".replace("%", "pct")
            row[f"precision_top_{tag}"] = round(precision, 4)
            row[f"enrichment_top_{tag}"] = round(ef, 3)
        return row


def roc_auc(label: np.ndarray, score: np.ndarray) -> float:
    """AUROC via the rank identity, ties given average ranks."""
    label = np.asarray(label, dtype=bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within tied score groups
    s = score[order]
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i + 1) / 2
            start = i
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(label: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve, the step-function estimator."""
    label = np.asarray(label, dtype=bool)
    n_pos = int(label.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hits = label[order]
    cum = np.cumsum(hits)
    precision = cum / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)


def enrichment_at(label: np.ndarray, score: np.ndarray,
                  fraction: float) -> tuple[float, float]:
    """``(precision, enrichment factor)`` in the top ``fraction`` of the ranking."""
    label = np.asarray(label, dtype=bool)
    k = max(1, int(round(fraction * len(label))))
    top = label[np.argsort(-score, kind="mergesort")[:k]]
    precision = float(top.mean())
    base = float(label.mean())
    return precision, precision / base if base else float("nan")


def score_allele(allele: str, label: np.ndarray, score: np.ndarray,
                 fractions=ENRICHMENT_FRACTIONS) -> AlleleResult:
    label = np.asarray(label, dtype=int)
    return AlleleResult(
        allele=allele,
        n_ligands=int(label.sum()),
        n_decoys=int((label == 0).sum()),
        auroc=roc_auc(label, score),
        auprc=average_precision(label, score),
        auprc_baseline=float(label.mean()),
        enrichment={f: enrichment_at(label, score, f) for f in fractions},
    )


def score_panel(scoring: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-allele table for one arm's predictions, ordered by allele."""
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.isfinite(y_pred).all():
        raise ValueError("predictions contain NaN or inf")
    out = []
    for allele, idx in scoring.groupby("allele", sort=True).indices.items():
        out.append(score_allele(allele, scoring.label.to_numpy()[idx],
                                y_pred[idx]).as_row())
    return pd.DataFrame(out)


def summarise_panel(per_allele: pd.DataFrame) -> dict:
    """Median and IQR across alleles for every metric column."""
    summary = {"n_alleles": int(len(per_allele))}
    for col in per_allele.columns:
        if col in ("allele", "n_ligands", "n_decoys"):
            continue
        s = per_allele[col]
        summary[col] = {"median": round(float(s.median()), 4),
                        "iqr_low": round(float(s.quantile(0.25)), 4),
                        "iqr_high": round(float(s.quantile(0.75)), 4),
                        "min": round(float(s.min()), 4),
                        "max": round(float(s.max()), 4)}
    return summary


def derangement(items: list[str], seed: int = SEED) -> dict[str, str]:
    """A permutation with no fixed point, for the wrong-allele control.

    Scoring each allele's rows against another allele's pseudosequence asks
    whether the ranking is allele-specific or just "this looks like a peptide".
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(items)
    for _ in range(1000):
        perm = rng.permutation(len(arr))
        if not (perm == np.arange(len(arr))).any():
            return dict(zip(arr, arr[perm]))
    raise RuntimeError("no derangement found")
