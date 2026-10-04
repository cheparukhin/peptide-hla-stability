# Stage 7b — leave-allele-out evaluation, stratified by pseudosequence distance

Status: **protocol predeclared, results pending.** Sections 1–5 were written to
disk before the first fold was fitted.

> ## This is not the project's headline evaluation
>
> `EVALUATION.md` is the frozen contract and is untouched by this document.
> `data/splits.csv` is untouched. Nothing here reinterprets either: this is a
> **second, separate contract** answering a **different question**, and its
> numbers are not comparable to a number from the frozen peptide-grouped split.
>
> `HACKATHON_PLAN.md` gives four reasons the allele axis is not the headline.
> **All four still stand** and are restated in §6 next to every number. Reason 1
> is a genuine confound, not a caveat — though §4 shows it bites through a
> different mechanism than the plan states.

Scope: `split in {train, val}` only. The frozen **test split is never read**, so
this evaluation cannot and does not consume the single-use test budget. Cost:
each held-out allele is evaluated on ~80% of its rows rather than all of them.

---

## 1. The question

The frozen split holds out **peptides** and keeps every allele in training. It
answers: *given an allele we have assayed, how well do we rank peptides we have
never seen on it?*

Leave-allele-out asks the opposite: *given a peptide panel we have largely seen,
how well do we transfer to an HLA allotype we have never assayed?* Pan-allele
generalisation is the strongest prior for pretrained representations, so if an
ESM-2 arm beats the sequence baseline anywhere, this stratum is where. That is
why the plan records it as worth doing even though it is not the headline.

## 2. The contract

| Item | Value |
|---|---|
| Rows in scope | `split in {train, val}` — 22,533 of 28,166 |
| Folds | one per eligible allele; fit on every **other** allele's rows |
| Eligibility | allele holds **>= 50 rows** in scope and **>= 2 distinct labels** |
| Eligible alleles | **68 of 75** |
| Inner dev fold | 10% of the fit rows, cut along whole Hamming <= 3 peptide clusters |
| Ensemble | **6 networks per fold** — 2 encodings x 3 seeds `(0, 1, 2)`, mean-averaged |
| Architecture | `(256, 64)`, per-encoding L2 from `baseline_ensemble.SELECTED` |
| Tuning | **none.** Configs inherited from stage 2; nothing is selected on this axis |
| Per-allele metric | Spearman rho between predicted and measured `log1p` half-life on that allele's in-scope rows, average ranks for ties |
| Stratum metric | **median** of the per-allele rho within the stratum |

The 50-row bar is not a judgement call here: in-scope allele counts jump from
**30 to 177** with nothing in between, so every bar between 31 and 177 selects
the same 68 alleles.

| In-scope rows | Alleles |
|---|---:|
| 6, 12, 13, 13, 17, 21, 30 | 7 (excluded) |
| 177 – 848 | 68 (eligible) |

## 3. Distance strata — chosen from the distance distribution, not from results

Distance is the Hamming distance between the held-out allele's 34-residue
contact pseudosequence and the **nearest allele remaining in the fit set** (all
74 others, since exactly one allele is held out). Computed from sequences
alone.

Distribution over the 68 eligible alleles:

| d | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| alleles | 2 | 23 | 16 | 7 | 11 | 1 | 4 | 1 | 2 | 1 |

Three strata are supported here, and the plan's reason 4 ("allele coverage is
too skewed to stratify") does **not** bind on this axis — it was written about
*test-row* counts under the frozen peptide split, which is a different quantity:

| Stratum | Rule | Alleles | Rows | Median rows/allele |
|---|---|---:|---:|---:|
| near | d <= 1 | 25 | 8,547 | 296 |
| intermediate | d = 2–3 | 23 | 7,279 | 298 |
| distant | d >= 4 | 20 | 6,595 | 295 |

A finer split is not supported: d >= 5 holds only 9 alleles. The bins are
near-tertiles of the distance distribution and were fixed before any fold ran.

**The two d = 0 alleles are a known degenerate case.** `HLA-B*14:01(C67S)` and
`HLA-B*14:02(C67S)` share one pseudosequence (`audit_summary.md` §1), so when
either is held out a pseudosequence-only model sees its exact input in training.
They are kept — dropping them would be a post-hoc panel change — but flagged in
the per-allele table, and the near stratum is reported with and without them.

## 4. The confound, measured

Reason 1 of the plan: *"Each allele was assayed on its own peptide panel …
holding out an allele simultaneously holds out its peptide panel, so an apparent
allele-distance effect is inseparable from a peptide-panel effect."*

The confound is real, but the mechanism in the data is **not** panel hold-out.
Measured on the in-scope rows:

| Quantity | Value |
|---|---:|
| Median alleles per peptide | 4 |
| Peptides appearing on exactly one allele (all splits) | 1,692 / 5,633 (30.0%) |
| Rows carrying such a peptide (all splits) | 1,692 / 28,166 (**6.0%**) |
| Median over eligible alleles of "share of its rows whose peptide is also assayed on another allele" | **1.000** |
| Minimum of that share | 0.565 (`HLA-A*02:01`) |

So for the median eligible allele, holding it out holds out **none** of its
peptides. Leave-allele-out here is largely *seen peptide, unseen allotype* —
which makes it an easier question than the frozen split, not a harder one, and
means its absolute numbers must never be compared to a frozen-split number.

What survives, and what §6 reports next to every number, is **panel
composition**: the panel assayed on each allele was partly selected by predicted
binding affinity for that allele (`HACKATHON_PLAN.md`, stage 1 caveats), so the
held-out allele's rows differ systematically in difficulty. That is confounded
with distance here:

| Stratum | Median per-allele zero share | Median per-allele median `log1p` label |
|---|---:|---:|
| near (d <= 1) | 0.063 | 0.875 |
| intermediate (d = 2–3) | 0.219 | 0.718 |
| distant (d >= 4) | 0.189 | 0.500 |

Spearman(distance, per-allele zero share) = **+0.251, p = 0.039** over the 68
alleles. Distant alleles carry weaker-binding, more heavily censored panels.
Peptide overlap, by contrast, is **not** confounded with distance: Spearman
(distance, peptide-seen share) = **−0.067, p = 0.59**.

**Therefore: a stratum difference measured here cannot be attributed to
pseudosequence distance alone.** It is a joint effect of distance and panel
difficulty, and this evaluation cannot separate them.

## 5. Uncertainty

The resampling unit is the **allele**, because each allele is one fold and the
folds are the independent replicates of this design. Per stratum: 2,000
bootstrap resamples of that stratum's alleles with replacement, reporting the
median per-allele rho and its 2.5/97.5 percentiles. Seed `20261003`, matching
`EVALUATION.md`'s bootstrap seed.

Between-arm comparisons (e.g. a later ESM-2 arm) use a **paired** resample —
the same drawn alleles scored for both arms — and the predeclared minimum
worthwhile gain of **Δ median per-allele Spearman = 0.05** applies, with the
same six verdicts as `EVALUATION.md`. An interval crossing zero is
**inconclusive**, not negative.

With 20–25 alleles per stratum, this design has limited power. The interval
width is reported, not hidden.

---

## 6. Results

*Pending — filled in after the run.*
