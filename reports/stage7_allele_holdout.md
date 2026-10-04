# Stage 7b — leave-allele-out evaluation, stratified by pseudosequence distance

Status: **complete.** The protocol in sections 1–5 was predeclared and is preserved verbatim below; the results follow it. Headline: near (d≤1) **0.741**, intermediate **0.576**, distant (d≥4) **0.339**; near−distant +0.403 [+0.223, +0.478], but near−intermediate crosses zero.

Original pre-registration header: Sections 1–5 were written to
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

Run: `scripts/stage7_allele_holdout.py`, 68 folds x 6 networks = **408
networks**, 36.2 min on 2 workers. Artifacts:
`reports/stage7_allele_holdout_{per_allele,strata,runs,headline}_seq_pep_pseudo.{csv,json}`,
predictions in `preds/stage7_allele_holdout_seq_pep_pseudo.csv`.

Arm: the stage 2 sequence baseline, `pep_pseudo`, configs inherited from
`baseline_ensemble.SELECTED` — **onehot (256, 64) L2 1e-5, blosum (256, 64) L2
1e-3**, seeds (0, 1, 2). Recorded explicitly because stage 2's L2 ladder is
being extended downward; if `SELECTED` changes, this run predates the change
and the arm must be re-run before any delta against a new baseline is quoted.

### 6.0 The four reasons, restated — read these before the numbers

`HACKATHON_PLAN.md`, "Why the allele-axis hold-out is not the headline". **All
four still stand.** Status after measuring:

| # | Reason (plan) | Status |
|---|---|---|
| **1** | **"It is confounded by design in this dataset.** Each allele was assayed on its own peptide panel … holding out an allele simultaneously holds out its peptide panel, so an apparent allele-distance effect is inseparable from a peptide-panel effect." | **Stands as a confound — but the mechanism is different from the one stated, and the measured confound is weaker than "inseparable". See §4 and §6.3.** |
| **2** | "It answers a different question from our headline claim." | **Stands, unchanged.** The headline claim is unseen peptides on trained alleles; this is unseen allotype on largely-seen peptides. The numbers below are **not comparable** to any frozen-split number and must never be quoted beside one. |
| **3** | "The frozen split is peptide-grouped, not allele-grouped — a second split, a second contract, every number twice." | **Stands, and is the cost paid here.** This document is that second contract. `EVALUATION.md` and `data/splits.csv` are unmodified (`tests/test_allele_holdout.py` guards both). |
| **4** | "Allele coverage is too skewed to stratify. 8 of 75 alleles hold fewer than 50 test rows, 7 of them ≤32 pairs in the entire dataset." | **The counts are correct but describe a different quantity.** Verified: exactly 8 alleles hold <50 **test-split** rows, 7 of them ≤32 pairs total. On *this* axis the relevant count is in-scope rows per allele, which jumps 30 → 177 with nothing between, so 68 alleles are comfortably stratifiable into 25/23/20. Reason 4 does not bind here. |

### 6.1 Per-stratum result

Median per-allele Spearman, allele-level bootstrap (2,000 resamples, seed
20261003). **The allele is the resampling unit** — each allele is one fold.

| Stratum | Alleles | Rows | **Median rho** | 95% CI | IQR | min–max | Median MAE | Median p@10 |
|---|---:|---:|---:|---|---|---|---:|---:|
| near (d ≤ 1) | 25 | 8,547 | **0.741** | [0.642, 0.803] | [0.623, 0.835] | 0.419–0.908 | 0.515 | 1.00 |
| intermediate (d = 2–3) | 23 | 7,279 | **0.576** | [0.458, 0.726] | [0.375, 0.726] | 0.119–0.802 | 0.541 | 0.80 |
| distant (d ≥ 4) | 20 | 6,595 | **0.339** | [0.298, 0.491] | [0.272, 0.516] | 0.151–0.753 | 0.756 | 0.60 |
| *near, excl. the shared-pseudosequence pair* | 23 | 7,932 | 0.750 | [0.647, 0.834] | [0.644, 0.835] | 0.419–0.908 | 0.534 | 1.00 |
| **all strata pooled** | 68 | 22,421 | **0.614** | [0.493, 0.694] | [0.376, 0.743] | 0.119–0.908 | 0.581 | 0.90 |

The strata overlap heavily: the distant bin's best allele (0.753) beats the
near bin's worst (0.419). The medians separate; the distributions do not.

Dropping the two `C67S` alleles that share a pseudosequence *raises* the near
stratum slightly (0.741 → 0.750), so the degenerate pair is not propping up
that bin.

**Contrasts** (two-sample bootstrap of the difference in medians, 10,000
resamples):

| Contrast | Difference | 95% CI | Reading |
|---|---:|---|---|
| near − distant | **+0.403** | [+0.223, +0.478] | large, excludes 0 |
| intermediate − distant | +0.237 | [+0.002, +0.386] | barely excludes 0 |
| near − intermediate | +0.166 | [−0.010, +0.327] | **crosses 0** |

So: **the extreme contrast is solid; the adjacent ones are not.** The monotone
three-bin ordering is suggestive, but this design separates near from distant
and does not reliably separate either from the middle. Reporting the three-bin
ordering as an established monotone trend would overstate it.

Per-allele Spearman(distance, rho) over all 68 alleles = **−0.636, p = 5.6e−9**.

### 6.2 What this says about pan-allele transfer

The sequence baseline **degrades sharply on allotypes unlike anything it
trained on**: median rho 0.741 at Hamming ≤ 1 against 0.339 at ≥ 4.

The eight worst folds are mostly, but not only, the isolated alleles:

| Allele | d | rho | zero share |
|---|---:|---:|---:|
| `HLA-B*39:06(C67S)` | 3 | 0.119 | **0.918** |
| `HLA-B*46:01` | 6 | 0.151 | 0.419 |
| `HLA-B*15:17` | 6 | 0.181 | 0.064 |
| `HLA-B*44:05` | 7 | 0.187 | 0.052 |
| `HLA-A*01:01` | 8 | 0.195 | 0.040 |
| `HLA-B*27:20` | 4 | 0.220 | 0.300 |
| `HLA-B*15:01` | 3 | 0.231 | 0.054 |
| `HLA-B*40:01` | 3 | 0.288 | 0.054 |

Five of the eight sit at d ≥ 4, but the single worst fold is
`HLA-B*39:06(C67S)` at **d = 3**, whose panel is 91.8% floor rows — there is
very little label spread left to rank. That is the confound of §6.3 in one
allele: it scores badly because of its panel, not its distance.

This is the headroom an ESM-2 arm would have to capture, and it is where the
plan says pretrained features have their strongest prior. **Nothing here says
ESM-2 will capture it** — only that there is a large, measurable deficit in
exactly that stratum, which is a precondition for the claim, not evidence for
it.

### 6.3 The confound, quantified — reason 1 in detail

The plan's stated mechanism is that holding out an allele holds out its peptide
panel. **Measured, that is not what happens** (§4): for the median eligible
allele, **100%** of its rows carry a peptide also assayed on another allele
(minimum 0.565), and only 6.0% of all rows carry a peptide unique to one
allele. Leave-allele-out here is largely *seen peptide, unseen allotype*.

Nor does residual peptide overlap explain per-allele performance:

| Correlate of per-allele rho | Spearman | p |
|---|---:|---:|
| pseudosequence distance | **−0.636** | 5.6e−9 |
| per-allele zero share (panel difficulty) | −0.410 | 5.1e−4 |
| per-allele median label | +0.457 | 8.8e−5 |
| **peptide-seen share** | **+0.034** | **0.79** |
| in-scope row count | +0.066 | 0.59 |

What *is* confounded with distance is **panel composition**: each allele's
panel was partly selected by predicted binding affinity for that allele, and
distant alleles carry weaker-binding, more heavily censored panels
(Spearman(distance, zero share) = +0.251, p = 0.039). Partialling the two
apart:

- partial(distance, rho | zero share) = **−0.604**, p = 4.9e−8
- partial(zero share, rho | distance) = **−0.336**, p = 5.1e−3

**Both carry independent signal.** The distance effect is not merely panel
difficulty wearing a costume — it survives at nearly full strength when zero
share is controlled. But zero share is only *one* proxy for panel composition;
anchor-motif composition, peptide diversity and the affinity-prediction step
that chose each panel are unmeasured here. **The confound is attenuated, not
eliminated, and the +0.403 near−distant gap remains a joint effect of
pseudosequence distance and unmeasured panel properties.** No number in §6.1
may be quoted without this paragraph.

### 6.4 Power — what this design can and cannot detect

Measured rather than assumed, following the precedent set tonight by
`eval-harness`. Minimum detectable effect for a **paired** between-arm
comparison (arm B = arm A + δ + per-allele noise with SD *s*; smallest δ whose
paired 95% CI excludes 0; median over 200 simulations):

| Stratum | Alleles | MDE at s = 0.02 | s = 0.05 | s = 0.10 |
|---|---:|---:|---:|---:|
| near (d ≤ 1) | 25 | 0.030 | 0.055 | 0.085 |
| intermediate | 23 | 0.030 | 0.070 | 0.115 |
| distant (d ≥ 4) | 20 | 0.025 | 0.050 | 0.075 |
| all pooled | 68 | 0.025 | 0.050 | 0.075 |

Read: if a second arm's per-allele deltas are tight (*s* ≲ 0.05), each stratum
can resolve the predeclared **0.05** worthwhile-gain bar. If they are loose
(*s* ≈ 0.10) the intermediate stratum needs ≈ 0.115 and cannot. This is an MDE
for *excluding zero*, which is a weaker test than clearing the 0.05 bar.

**A per-stratum ESM-2 comparison is therefore worth running but is not
guaranteed to conclude.** An inconclusive stratum must be reported as
inconclusive, not as a null.

### 6.5 Adding an arm

Three arms were flagged as coming: `preds/esm_ensemble.csv`,
`preds/esm_plus_seq_ensemble.csv`, `preds/boltz_structural.csv`.

**A prediction file cannot be run through this contract.** Those files hold the
output of a model fitted on **every** allele; scoring one here would score a
model on alleles it trained on and report the leak as pan-allele
generalisation — a number that would look like a win. Leave-allele-out refits
68 times, so an arm must supply **features**, not predictions.
`scripts/stage7_allele_holdout.py` now rejects a `preds/*.csv` with that
message, and `tests/test_allele_holdout.py` guards it.

What is needed per arm is a feature matrix plus its `pair_id` index, aligned by
`pair_id` and never positionally:

    .venv/bin/python scripts/stage7_allele_holdout.py \
        --arm esm --npy esm=features/esm2.npy:features/esm2_pair_ids.npy \
        --seeds 0 1 2 3 4 5 --workers 1

The ensemble is fixed at **6 networks per fold** and `Arm.validate()` refuses
anything else, so no arm can win on ensemble budget — stage 2 measured
ensembling alone at +0.074 mean SCC against the deployed single network,
about 1.5x the worthwhile-gain bar (the +0.090 sometimes quoted is against the
mean of the ensemble's own 30 members — a different reference; see
`reports/stage2_baselines.md`). For a
single-representation arm that means 6 seeds; for the two-encoding sequence arm
it is 2 x 3. Compare arms with `paired_allele_bootstrap`, optionally restricted
to one stratum.

### 6.6 Conclusion, bounded to what was tested

1. **The sequence baseline transfers poorly to distant allotypes.** Median
   per-allele Spearman falls from 0.741 (d ≤ 1) to 0.339 (d ≥ 4), a gap of
   +0.403 [+0.223, +0.478]. The near-vs-distant contrast is solid; adjacent
   strata are not reliably separated.
2. **This is confounded and the confound is not removable here.** Distant
   alleles carry systematically harder panels. Controlling for the one
   measurable proxy leaves the distance effect nearly intact
   (−0.636 → −0.604), which weakens but does not retire reason 1. Other panel
   properties are unmeasured.
3. **The plan's stated mechanism for reason 1 is not what the data shows.**
   Holding out an allele does not hold out its peptide panel — median peptide
   overlap is 100%, and overlap is uncorrelated with both distance (−0.067)
   and per-allele performance (+0.034). Reason 1 survives through panel
   *composition*, not panel *hold-out*. Reported here, not edited into the
   plan.
4. **This is not the headline and its numbers are not comparable to one.**
   Different contract, different question, easier in one respect (peptides
   largely seen) and harder in another (allotype unseen).
5. **The design can probably resolve a 0.05 between-arm difference, but not
   certainly.**

**Bounds.** One arm, one input representation, one architecture, 6 networks per
fold, `split in {train, val}` only, 68 of 75 alleles, pseudosequence Hamming as
the distance metric, and medians over 20–25 alleles per bin. The frozen test
split was not read. Nothing here licenses a claim about the test split or about
any arm that has not been run through this same contract.
