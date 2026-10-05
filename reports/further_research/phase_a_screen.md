# Phase A screen: assembly, scoring, selection, comparison

Written 2026-10-04T22:07:36+00:00 by `scripts/phase_a_select.py`. Binding protocol: [phase_a_prespecification.md](phase_a_prespecification.md) §2 (metrics and the four calibration rules), §3 (one-standard-error parsimony, the grid-boundary trigger, the same-geometry rule), §4 (arms), §6a (amendments).

Screen status: **complete**. BLAS threads pinned to 1 (`OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS`). Bootstrap: 2000 resamples at seed 20261003. Wall time 2249 s, no model fitted — every prediction read here was produced on Modal by `pepstab.phase_a.run_unit`.

## What a screen number is, and is not

One assembled prediction averages **12 members** (3 inner folds × 2 seeds × 2 encodings). A0 and every assessment arm carry **30**. `n_members` is written on every assembled row, every score row and every comparison row. **The honest output of this screen is a ranking of configurations, not a verdict against A0.**

**How big is that gap? Measured here, +0.0156 — not the +0.07 to +0.09 the prespecification quotes.** The matched 12-member control (A3-a: MSE loss, natural sampling, MSE stopping, reference hyperparameters — A0's own member recipe run inside this screen) scores 0.6337 against A0's 0.6493. That 0.0156 covers arity (12 members against 30), inner-fold count (3 against 5, so each member fits on fewer rows too) and platform together. The +0.07 to +0.09 figure in §1, §A0 and the README is a *single network* against an ensemble, which is a different comparison; applied to 12-against-30 it overstates the gap about fivefold. The caveat is real and worth roughly a third of the 0.05 bar, not more than the bar.

**Platform.** A0 was fitted on macOS/Accelerate and every screen unit on Linux/OpenBLAS; Phase B measured max |diff| 0.18 per row and 0.0020 on median-of-medians across the two. Deltas against A0 carry that term; deltas within the screen do not. This is why the matched control, not A0, is the headline reference below: an arm-against-control delta is free of the arity, inner-fold-count and platform terms, while an arm-against-A0 delta carries all three.

The inner out-of-fold role is named `inner_oof_4` rather than reusing A0's `inner_oof_6`: a screen row sits in one inner fold, so its inner mean has 4 members (2 encodings × 2 seeds), and the two objects must not share a name.

No calibration map exists anywhere in this script. §2's four rules are binding and isotonic regression moved per-allele Spearman between −0.0431 and +0.1109 on this project's own predictions — both signs inside one fit — purely by collapsing distinct predictions into ties.

## Assembly

Journal: `phase_a_runs_a-cheparukhin.partial.jsonl`, 1920 lines, 1920 usable units of 1920.

| arm | configs seen | folds complete (of 5) |
|---|---:|---|
| A1 | 12 | min 5, max 5 |
| A2 | 4 | min 5, max 5 |
| A3 | 16 | min 5, max 5 |

Asserted before anything was written:
- outer folds identical to phase_a_outer_folds.csv for all 19716 training rows, fold sizes [3944, 3943, 3943, 3943, 3943]
- A0 reproduces its declared per-fold medians [0.6765, 0.6281, 0.6418, 0.6493, 0.6515] and median of medians 0.6493 to within 0.0005
- every assembled outer-held row averaged exactly 12 members, by member-axis identity against `phase_a.enumerate_units`, not by counting to 12
- every assembled inner out-of-fold row averaged exactly 4 members, and no outer-held row received one

## Scores — five-fold median of medians and fold-to-fold span

A0's own span is 0.0484 with the model held fixed, which is essentially the whole 0.05 worthwhile bar. **A difference smaller than an arm's own span is not a result**, so the span sits next to every median rather than in a footnote.

### ≥20 rows / ≥2 distinct labels (primary panel)

| arm | config | members | median of medians | fold min | fold max | span | alleles (median) | row coverage (mean) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | `A0` | 30 | 0.6493 | 0.6281 | 0.6765 | 0.0484 | 68 | 0.995 |
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.06` | 12 | 0.6611 | 0.6494 | 0.6857 | 0.0363 | 68 | 0.995 |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.06` | 12 | 0.6583 | 0.6468 | 0.6829 | 0.0360 | 68 | 0.995 |
| A1 | `depth=6,l2_leaf_reg=3,learning_rate=0.06` | 12 | 0.6540 | 0.6439 | 0.6634 | 0.0194 | 68 | 0.995 |
| A1 | `depth=6,l2_leaf_reg=10,learning_rate=0.06` | 12 | 0.6487 | 0.6406 | 0.6654 | 0.0248 | 68 | 0.995 |
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.03` | 12 | 0.6433 | 0.6328 | 0.6502 | 0.0174 | 68 | 0.995 |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.03` | 12 | 0.6421 | 0.6362 | 0.6623 | 0.0261 | 68 | 0.995 |
| A1 | `depth=6,l2_leaf_reg=3,learning_rate=0.03` | 12 | 0.6250 | 0.6189 | 0.6274 | 0.0085 | 68 | 0.995 |
| A1 | `depth=4,l2_leaf_reg=3,learning_rate=0.06` | 12 | 0.6157 | 0.6053 | 0.6211 | 0.0158 | 68 | 0.995 |
| A1 | `depth=6,l2_leaf_reg=10,learning_rate=0.03` | 12 | 0.6118 | 0.6082 | 0.6190 | 0.0108 | 68 | 0.995 |
| A1 | `depth=4,l2_leaf_reg=10,learning_rate=0.06` | 12 | 0.6032 | 0.5989 | 0.6114 | 0.0125 | 68 | 0.995 |
| A1 | `depth=4,l2_leaf_reg=3,learning_rate=0.03` | 12 | 0.5652 | 0.5584 | 0.5765 | 0.0181 | 68 | 0.995 |
| A1 | `depth=4,l2_leaf_reg=10,learning_rate=0.03` | 12 | 0.5621 | 0.5413 | 0.5697 | 0.0284 | 68 | 0.995 |
| A2 | `k=20` | 12 | 0.6298 | 0.6045 | 0.6674 | 0.0630 | 68 | 0.995 |
| A2 | `k=5` | 12 | 0.6296 | 0.6078 | 0.6724 | 0.0646 | 68 | 0.995 |
| A2 | `k=1` | 12 | 0.6278 | 0.6147 | 0.6722 | 0.0574 | 68 | 0.995 |
| A2 | `k=100` | 12 | 0.6252 | 0.6074 | 0.6660 | 0.0586 | 68 | 0.995 |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank` | 12 | 0.6590 | 0.6256 | 0.6974 | 0.0718 | 68 | 0.995 |
| A3 | `cell=A3-d,lam=1.0,loss=mse+rank,sampling=balanced,stopping=rank` | 12 | 0.6512 | 0.6217 | 0.6911 | 0.0694 | 68 | 0.995 |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=mse` | 12 | 0.6475 | 0.6129 | 0.6597 | 0.0468 | 68 | 0.995 |
| A3 | `cell=A3-c,lam=0.3,loss=mse+rank,sampling=natural,stopping=rank` | 12 | 0.6475 | 0.6069 | 0.6760 | 0.0690 | 68 | 0.995 |
| A3 | `cell=A3-d,lam=0.3,loss=mse+rank,sampling=balanced,stopping=rank` | 12 | 0.6459 | 0.6101 | 0.6769 | 0.0668 | 68 | 0.995 |
| A3 | `cell=A3-a,lam=,loss=mse,sampling=natural,stopping=rank` | 12 | 0.6422 | 0.6191 | 0.6798 | 0.0607 | 68 | 0.995 |
| A3 | `cell=A3-c,lam=0.3,loss=mse+rank,sampling=natural,stopping=mse` | 12 | 0.6420 | 0.5960 | 0.6755 | 0.0795 | 68 | 0.995 |
| A3 | `cell=A3-b,lam=,loss=mse,sampling=balanced,stopping=rank` | 12 | 0.6406 | 0.6023 | 0.6832 | 0.0808 | 68 | 0.995 |
| A3 | `cell=A3-d,lam=1.0,loss=mse+rank,sampling=balanced,stopping=mse` | 12 | 0.6383 | 0.5988 | 0.6775 | 0.0787 | 68 | 0.995 |
| A3 | `cell=A3-c,lam=0.1,loss=mse+rank,sampling=natural,stopping=rank` | 12 | 0.6368 | 0.6157 | 0.6764 | 0.0607 | 68 | 0.995 |
| A3 | `cell=A3-d,lam=0.1,loss=mse+rank,sampling=balanced,stopping=rank` | 12 | 0.6338 | 0.6036 | 0.6779 | 0.0743 | 68 | 0.995 |
| A3 | `cell=A3-a,lam=,loss=mse,sampling=natural,stopping=mse` | 12 | 0.6337 | 0.6154 | 0.6617 | 0.0463 | 68 | 0.995 |
| A3 | `cell=A3-c,lam=0.1,loss=mse+rank,sampling=natural,stopping=mse` | 12 | 0.6309 | 0.6027 | 0.6704 | 0.0677 | 68 | 0.995 |
| A3 | `cell=A3-d,lam=0.3,loss=mse+rank,sampling=balanced,stopping=mse` | 12 | 0.6296 | 0.6051 | 0.6733 | 0.0682 | 68 | 0.995 |
| A3 | `cell=A3-b,lam=,loss=mse,sampling=balanced,stopping=mse` | 12 | 0.6234 | 0.6042 | 0.6704 | 0.0662 | 68 | 0.995 |
| A3 | `cell=A3-d,lam=0.1,loss=mse+rank,sampling=balanced,stopping=mse` | 12 | 0.6234 | 0.5970 | 0.6705 | 0.0734 | 68 | 0.995 |

### ≥50 rows / ≥2 distinct labels (declared sensitivity check)

| arm | config | members | median of medians | fold min | fold max | span | alleles (median) | row coverage (mean) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | `A0` | 30 | 0.6737 | 0.6123 | 0.7032 | 0.0910 | 48 | 0.751 |
| A1 | `depth=6,l2_leaf_reg=3,learning_rate=0.06` | 12 | 0.6738 | 0.6258 | 0.6970 | 0.0712 | 48 | 0.751 |
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.06` | 12 | 0.6725 | 0.6475 | 0.7134 | 0.0659 | 48 | 0.751 |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.06` | 12 | 0.6684 | 0.6468 | 0.7083 | 0.0614 | 48 | 0.751 |
| A1 | `depth=6,l2_leaf_reg=10,learning_rate=0.06` | 12 | 0.6665 | 0.6106 | 0.6905 | 0.0799 | 48 | 0.751 |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.03` | 12 | 0.6588 | 0.6205 | 0.6897 | 0.0691 | 48 | 0.751 |
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.03` | 12 | 0.6565 | 0.6104 | 0.6797 | 0.0693 | 48 | 0.751 |
| A1 | `depth=6,l2_leaf_reg=3,learning_rate=0.03` | 12 | 0.6439 | 0.5979 | 0.6749 | 0.0770 | 48 | 0.751 |
| A1 | `depth=4,l2_leaf_reg=3,learning_rate=0.06` | 12 | 0.6324 | 0.5772 | 0.6608 | 0.0837 | 48 | 0.751 |
| A1 | `depth=4,l2_leaf_reg=10,learning_rate=0.06` | 12 | 0.6294 | 0.5731 | 0.6502 | 0.0771 | 48 | 0.751 |
| A1 | `depth=6,l2_leaf_reg=10,learning_rate=0.03` | 12 | 0.6216 | 0.5855 | 0.6654 | 0.0799 | 48 | 0.751 |
| A1 | `depth=4,l2_leaf_reg=3,learning_rate=0.03` | 12 | 0.5863 | 0.5501 | 0.6181 | 0.0680 | 48 | 0.751 |
| A1 | `depth=4,l2_leaf_reg=10,learning_rate=0.03` | 12 | 0.5791 | 0.5276 | 0.6159 | 0.0883 | 48 | 0.751 |
| A2 | `k=5` | 12 | 0.6766 | 0.6082 | 0.6784 | 0.0702 | 48 | 0.751 |
| A2 | `k=1` | 12 | 0.6749 | 0.6147 | 0.6862 | 0.0715 | 48 | 0.751 |
| A2 | `k=100` | 12 | 0.6722 | 0.6150 | 0.6933 | 0.0783 | 48 | 0.751 |
| A2 | `k=20` | 12 | 0.6690 | 0.6043 | 0.6802 | 0.0759 | 48 | 0.751 |
| A3 | `cell=A3-c,lam=0.3,loss=mse+rank,sampling=natural,stopping=rank` | 12 | 0.6815 | 0.5975 | 0.6979 | 0.1004 | 48 | 0.751 |
| A3 | `cell=A3-d,lam=1.0,loss=mse+rank,sampling=balanced,stopping=mse` | 12 | 0.6788 | 0.6012 | 0.6857 | 0.0845 | 48 | 0.751 |
| A3 | `cell=A3-d,lam=1.0,loss=mse+rank,sampling=balanced,stopping=rank` | 12 | 0.6784 | 0.6234 | 0.6935 | 0.0701 | 48 | 0.751 |
| A3 | `cell=A3-c,lam=0.3,loss=mse+rank,sampling=natural,stopping=mse` | 12 | 0.6765 | 0.6067 | 0.6892 | 0.0825 | 48 | 0.751 |
| A3 | `cell=A3-c,lam=0.1,loss=mse+rank,sampling=natural,stopping=rank` | 12 | 0.6748 | 0.6070 | 0.7039 | 0.0969 | 48 | 0.751 |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=mse` | 12 | 0.6748 | 0.6180 | 0.6908 | 0.0728 | 48 | 0.751 |
| A3 | `cell=A3-d,lam=0.1,loss=mse+rank,sampling=balanced,stopping=mse` | 12 | 0.6740 | 0.5970 | 0.6806 | 0.0836 | 48 | 0.751 |
| A3 | `cell=A3-c,lam=0.1,loss=mse+rank,sampling=natural,stopping=mse` | 12 | 0.6726 | 0.6007 | 0.6999 | 0.0992 | 48 | 0.751 |
| A3 | `cell=A3-b,lam=,loss=mse,sampling=balanced,stopping=mse` | 12 | 0.6717 | 0.5943 | 0.6865 | 0.0922 | 48 | 0.751 |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank` | 12 | 0.6703 | 0.6355 | 0.7050 | 0.0695 | 48 | 0.751 |
| A3 | `cell=A3-d,lam=0.3,loss=mse+rank,sampling=balanced,stopping=rank` | 12 | 0.6660 | 0.6015 | 0.6834 | 0.0819 | 48 | 0.751 |
| A3 | `cell=A3-d,lam=0.1,loss=mse+rank,sampling=balanced,stopping=rank` | 12 | 0.6635 | 0.6036 | 0.6851 | 0.0815 | 48 | 0.751 |
| A3 | `cell=A3-a,lam=,loss=mse,sampling=natural,stopping=rank` | 12 | 0.6635 | 0.6225 | 0.7050 | 0.0825 | 48 | 0.751 |
| A3 | `cell=A3-b,lam=,loss=mse,sampling=balanced,stopping=rank` | 12 | 0.6627 | 0.6023 | 0.6896 | 0.0873 | 48 | 0.751 |
| A3 | `cell=A3-d,lam=0.3,loss=mse+rank,sampling=balanced,stopping=mse` | 12 | 0.6610 | 0.5987 | 0.6795 | 0.0808 | 48 | 0.751 |
| A3 | `cell=A3-a,lam=,loss=mse,sampling=natural,stopping=mse` | 12 | 0.6607 | 0.6199 | 0.7104 | 0.0905 | 48 | 0.751 |

The ≥50 panel is a **weaker instrument at fold scope than §2's own figures suggest.** §2 quotes ≥50 as 68/75 alleles at 99.51% row coverage, which is the *pooled* training split. On a ~3,943-row outer fold the same bar keeps a median of 48 alleles at 75.1% coverage, against 68 at 99.5% on the ≥20 panel, and A0's own fold-to-fold span widens from 0.0484 to 0.0910. It is reported for every arm as §2 requires, but it is noisier than the primary panel, not stricter.

### Does any arm beat A0?

**5 configuration(s) sit above A0 on the five-fold median of medians**, the best by +0.0118 — `A1|depth=8,l2_leaf_reg=10,learning_rate=0.06` at 0.6611 against A0's 0.6493. Read that with three things in view, in this order:

1. **It is smaller than A0's own fold-to-fold span** (0.0484, with the model held fixed), so it is below the resolution at which this design calls anything a difference.
2. **The arity runs the wrong way for it.** A 12-member arm above a 30-member reference is the opposite of what ensembling predicts, so the first hypothesis is an assembly error rather than a finding. The member-identity assertions in the assembly section are what rule that out, and they passed. The gap it has to overcome is +0.0156, measured against the matched 12-member control, not the +0.07 to +0.09 the prespecification quotes for a single network against an ensemble.
3. **The paired interval is the arbiter, and the matched control is the reference to read.** Median-of-medians and the pooled panel median are different aggregations and can disagree in sign; the comparison table below reports the pooled delta, which is what the bootstrap puts an interval on, against the matched control as well as against A0.

## Selection

`pepstab.phase_a.parsimonious_select` is applied per family and not reimplemented; A3 goes through `common_hyperparameters`, which refuses a lambda chosen on a partial pool. Read `boundary_axes`, not `interior`: §6a records that `interior` is structurally unreachable for the tree grids, where `learning_rate` and `l2_leaf_reg` carry two values each and every selection is at an endpoint on both axes by construction.

| arm | ladder | argmax | selected | score (argmax → selected) | seed spread (argmax / selected) | SE | within 1 SE | boundary axes | trigger |
|---|---|---|---|---|---|---:|---:|---|---|
| A1 | declared | `{"depth": 8.0, "l2_leaf_reg": 3.0, "learning_rate": 0.06}` | `{"depth": 8.0, "l2_leaf_reg": 10.0, "learning_rate": 0.06}` | 0.6035 → 0.6035 | 0.0009 / 0.0004 | 0.0038 | 2/12 | l2_leaf_reg, depth, learning_rate | strong-end boundary |
| A2 | declared | `{"k": 100.0}` | `{"k": 100.0}` | 0.5282 → 0.5282 | 0.0012 / 0.0012 | 0.0032 | 1/4 | k | strong-end boundary |
| A3 | declared | `{"lam": 1.0}` | `{"lam": 1.0}` | 0.5473 → 0.5473 | 0.0069 / 0.0069 | 0.0015 | 1/3 | lam | **fired** — lam -> {0.03, 0.1, 0.3, 1.0} |
| A3 | diagnostic:stopping=mse | `{"lam": 1.0}` | `{"lam": 1.0}` | 0.5417 → 0.5417 | 0.0093 / 0.0093 | 0.0023 | 1/3 | lam | **fired** — lam -> {0.03, 0.1, 0.3, 1.0} |
| A3 | diagnostic:stopping=rank | `{"lam": 1.0}` | `{"lam": 1.0}` | 0.5530 → 0.5530 | 0.0045 / 0.0045 | 0.0019 | 1/3 | lam | **fired** — lam -> {0.03, 0.1, 0.3, 1.0} |

### The grid-boundary trigger fired — reported, not run

§3 fixes both the trigger and its remedy in advance: the axis extends **once**, by one declared step, and **every arm in the family is then re-selected on the extended ladder**, with both ladders reported. That re-selects arms this screen has already chosen for, so it is a decision to surface rather than a post-processing step. This script stops here.

- **A3** (A3, ladder `declared`): selected `{"lam": 1.0}`, at the weakest-regularisation end of `lam`. Declared one-step extension: `lam -> {0.03, 0.1, 0.3, 1.0}`. Not run.
- **A3** (A3, ladder `diagnostic:stopping=mse`): selected `{"lam": 1.0}`, at the weakest-regularisation end of `lam`. Declared one-step extension: `lam -> {0.03, 0.1, 0.3, 1.0}`. Not run.
- **A3** (A3, ladder `diagnostic:stopping=rank`): selected `{"lam": 1.0}`, at the weakest-regularisation end of `lam`. Declared one-step extension: `lam -> {0.03, 0.1, 0.3, 1.0}`. Not run.

### Two gaps in §3's declared remedy, found by applying it

Reported rather than patched: §1–§6 of the prespecification are not edited after the first fit, and inventing an extension step now is exactly the post-hoc budget expansion §3.3 exists to prevent.

- **A2's selection has no declared step past it.** The selection sits at the strongest-regularisation end of `k` (`{"k": 100.0}`), but the declared extension `k -> {0.2, 1, 5, 20, 100}` adds values on the *other* end only. §3 declares the trigger and the remedy as a pair and says the axis "extends once, by one declared step **in that direction**" — for this family the declared set does not go in that direction, so the remedy as written does not answer the trigger it is attached to. The tree ladder `{1, 3, 10, 30}` is the only one of the three that reaches both ends.
- **A3's selection has no declared step past it.** The selection sits at the weakest-regularisation end of `lam` (`{"lam": 1.0}`), but the declared extension `lam -> {0.03, 0.1, 0.3, 1.0}` adds values on the *other* end only. §3 declares the trigger and the remedy as a pair and says the axis "extends once, by one declared step **in that direction**" — for this family the declared set does not go in that direction, so the remedy as written does not answer the trigger it is attached to. The tree ladder `{1, 3, 10, 30}` is the only one of the three that reaches both ends.
- **A1: axes at the weak end that §3's trigger does not name** — `depth`, `learning_rate` (selected `{"depth": 8.0, "l2_leaf_reg": 10.0, "learning_rate": 0.06}`). §3 names only `l2_leaf_reg` for this family, so no remedy attaches to these, although the argument for the trigger — a truncated space at the weak-regularisation end cost this project 0.093 median Spearman once — applies to them identically. An axis with three or more tried values could have been interior, so its endpoint is a genuine edge rather than §6a's unextendable-by-construction case; `depth` (3 values), `learning_rate` (2 values).

A3's per-stopping-rule rows are **diagnostics, not selections**. §A3 makes the stopping rule its own axis and the declared lambda is the one chosen on the pooled ladder; these show whether each stopping surface would have chosen the same lambda.

## A3's stopping rule, as its own axis

§A3 runs every cell under MSE-based and ranking-based early stopping and requires the two be compared explicitly, because stage 7's apparent ranking recovery traced to the stopping rule rather than to the gradient. Five-fold median of medians at the ≥20 panel:

| cell | loss | sampling | λ | MSE stopping | rank stopping | rank − MSE |
|---|---|---|---|---:|---:|---:|
| A3-c | mse+rank | natural | 0.1 | 0.6309 | 0.6368 | +0.0059 |
| A3-c | mse+rank | natural | 0.3 | 0.6420 | 0.6475 | +0.0055 |
| A3-c | mse+rank | natural | 1.0 | 0.6475 | 0.6590 | +0.0114 |
| A3-d | mse+rank | balanced | 0.1 | 0.6234 | 0.6338 | +0.0105 |
| A3-d | mse+rank | balanced | 0.3 | 0.6296 | 0.6459 | +0.0163 |
| A3-d | mse+rank | balanced | 1.0 | 0.6383 | 0.6512 | +0.0129 |

**Ranking-based stopping is ahead in 6 of 6 matched cells, by +0.0104 on average** — and, decisively for the confound, it is ahead in the **pure-MSE-loss** cells too (), where there is no ranking gradient at all. So part of what a naive loss comparison would credit to the pairwise objective is the stopping surface. This is the stage 7 confound reproducing itself on fresh data, caught because §A3 made the stopping rule an axis instead of folding it into the loss.

The ranking gradient is **not** fully explained by it: at rank stopping, the best ranking cell still sits above the matched MSE-loss cell. Both terms are present, and neither is resolvable at this design — A3's own fold-to-fold spans run to 0.0808, several times every difference in the table.

## Comparison against three references

§2 requires two: A0's nested sequence reference, and the strongest *new* sequence-only control among A2/A3, so a better learner is not mistaken for a benefit from embeddings or structure. A1 is not a candidate control — it is the learner-family arm under test. A third is added here and is the one to read first:

- **matched control** — A3-a (MSE loss, natural sampling, MSE stopping, reference hyperparameters), which is A0's own member recipe run inside this screen. Same arity, same inner-fold count, same containers, so the delta isolates what the arm changed and nothing else.
- **A0** — §2's first reference, and the only one that connects this screen to the published work. Carries arity (12 members against 30), inner-fold count (3 against 5, so each member fits on fewer rows too) and platform, worth about +0.0156 together, plus the cross-platform term.
- **strongest new sequence-only control** — §2's second reference.

`pepstab.evaluation.paired_cluster_bootstrap` is used verbatim with the reference as `pred_a`, so every interval is on `arm − reference`, and `describe_delta` supplies the verdict from EVALUATION.md's six. Intervals are pooled over all five outer folds (one prediction per training row), the aggregation Phase D used; the pooled panel median is not the five-fold median of medians, and both are given.

| arm | config | reference | members (arm vs ref) | pooled arm | pooled ref | Δ | 95% CI | verdict |
|---|---|---|---|---:|---:|---:|---|---|
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.06` | A3|cell=A3-a,lam=,loss=mse,sampling=natural,stopping=mse | 12 vs 12 | 0.6687 | 0.6510 | +0.0178 | [+0.0033, +0.0459] | real, but rules out a 0.05 gain |
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.06` | A0 | 12 vs 30 | 0.6687 | 0.6615 | +0.0072 | [-0.0113, +0.0304] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.06` | A3|cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank | 12 vs 12 | 0.6687 | 0.6719 | -0.0031 | [-0.0150, +0.0240] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.06` | A3|cell=A3-a,lam=,loss=mse,sampling=natural,stopping=mse | 12 vs 12 | 0.6731 | 0.6510 | +0.0222 | [+0.0043, +0.0482] | real, but rules out a 0.05 gain |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.06` | A0 | 12 vs 30 | 0.6731 | 0.6615 | +0.0116 | [-0.0099, +0.0331] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.06` | A3|cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank | 12 vs 12 | 0.6731 | 0.6719 | +0.0013 | [-0.0142, +0.0267] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A2 | `k=100` | A3|cell=A3-a,lam=,loss=mse,sampling=natural,stopping=mse | 12 vs 12 | 0.6483 | 0.6510 | -0.0026 | [-0.0153, +0.0139] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A2 | `k=100` | A0 | 12 vs 30 | 0.6483 | 0.6615 | -0.0132 | [-0.0310, -0.0003] | worse: CI entirely below 0 |
| A2 | `k=100` | A3|cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank | 12 vs 12 | 0.6483 | 0.6719 | -0.0235 | [-0.0379, -0.0037] | worse: CI entirely below 0 |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank` | A3|cell=A3-a,lam=,loss=mse,sampling=natural,stopping=mse | 12 vs 12 | 0.6719 | 0.6510 | +0.0209 | [+0.0040, +0.0354] | real, but rules out a 0.05 gain |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank` | A0 | 12 vs 30 | 0.6719 | 0.6615 | +0.0103 | [-0.0109, +0.0216] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |

**On every A0 row above:** cross-reference delta: a 12-member screen arm against the 30-member A0. The gap between those two setups -- arity (12 members against 30), inner-fold count (3 against 5, so each member fits on fewer rows too) and platform -- measures about +0.0156 median Spearman on this screen (matched control 0.6337 against A0's 0.6493), so the delta carries that term. A0 was additionally fitted on macOS/Accelerate while every screen unit ran on Linux/OpenBLAS; Phase B measured that at up to 0.18 per row and 0.0020 on median-of-medians, so this delta is also cross-platform. Prefer the matched-control comparison, which carries none of it.

### Intervals on the stopping axis — the matched pair

Both sides share arity, platform, encodings, seeds, hyperparameters, loss and sampling, and differ only in which curve chose the epoch. Nothing else in this screen is this clean.

| cell at λ | rank-stopping | MSE-stopping | Δ (rank − MSE) | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| cell=A3-a,lam=None | 0.6478 | 0.6510 | -0.0032 | [-0.0073, +0.0175] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| cell=A3-b,lam=None | 0.6480 | 0.6422 | +0.0058 | [-0.0048, +0.0204] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| cell=A3-c,lam=1.0 | 0.6719 | 0.6430 | +0.0288 | [+0.0012, +0.0279] | real, but rules out a 0.05 gain |
| cell=A3-d,lam=1.0 | 0.6635 | 0.6550 | +0.0085 | [+0.0004, +0.0254] | real, but rules out a 0.05 gain |

## A5 groundwork

The machinery: non-negative weights summing to one, fitted by grid search over the simplex at step 0.01, on **inner** out-of-fold rows only, applied to outer-held predictions, per outer fold. An outer-held label never chooses a weight.

Which inner surrogate to fit on is **not** inherited from §A0's residual-scale argument, which is about a ridge correction's residuals: a weight vector constrained to the simplex is a different functional and may be far less sensitive to a uniform scale shift. Both are fitted and the weights compared directly.

### Inner surrogate scale, scored as a stacker sees it

One row per `(pair_id, outer fold)` replicate. Each `pair_id` carries four inner replicates, one per outer fold in whose fitting partition it sits, and a stacking model inside one outer fold gets exactly **one** of them.

| role | aggregation | rows | RMSE (all) | RMSE (measured) | ratio to `outer_held` |
|---|---|---:|---:|---:|---:|
| `outer_held` | per replicate row | 19716 | 0.6678 | 0.6944 | 1.0000 |
| `inner_oof_6` | per replicate row | 78864 | 0.6805 | 0.7086 | 1.0204 |
| `inner_oof_6` | replicate-averaged (WRONG -- recorded as the error it is; no stacker sees this) | 19716 | 0.6582 | 0.6845 | 0.9857 |
| `inner_oof_30` | per replicate row | 78864 | 0.6904 | 0.7176 | 1.0334 |
| `inner_oof_30` | replicate-averaged (WRONG -- recorded as the error it is; no stacker sees this) | 19716 | 0.6746 | 0.7005 | 1.0087 |

**Both surrogates are pessimistic.** They bound the out-of-sample residual scale from one side only and do **not** bracket it; `inner_oof_6` is the tighter bound. §A0's early-stopping-optimism mechanism — `inner_oof_6`'s six members chose their stopping epoch on the rows they predict — is real, but it is swamped by the fit-size effect: 10,094 or 12,618 training rows against 15,773.

**Averaging the four replicates first reverses the sign** and fabricates a bracket, putting `inner_oof_6` *below* `outer_held`. The wrong aggregation is in the table above, labelled, because this measurement is easy to re-derive incorrectly — it was, once, during this screen's own review.

| arm | config | surrogate | mean A0 weight | max |Δw| between surrogates | agree | blend Δ vs A0 | 95% CI | verdict |
|---|---|---|---:|---:|---|---:|---|---|
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.06` | inner_oof_6 | 0.6860 | 0.0700 | no | +0.0128 | [+0.0016, +0.0246] | real, but rules out a 0.05 gain |
| A1 | `depth=8,l2_leaf_reg=10,learning_rate=0.06` | inner_oof_30 | 0.6480 | 0.0700 | no | +0.0132 | [+0.0014, +0.0258] | real, but rules out a 0.05 gain |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.06` | inner_oof_6 | 0.6740 | 0.0700 | no | +0.0141 | [+0.0023, +0.0265] | real, but rules out a 0.05 gain |
| A1 | `depth=8,l2_leaf_reg=3,learning_rate=0.06` | inner_oof_30 | 0.6340 | 0.0700 | no | +0.0147 | [+0.0026, +0.0285] | real, but rules out a 0.05 gain |
| A2 | `k=100` | inner_oof_6 | 0.7120 | 0.1100 | no | +0.0018 | [-0.0088, +0.0063] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A2 | `k=100` | inner_oof_30 | 0.6500 | 0.1100 | no | +0.0025 | [-0.0104, +0.0065] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank` | inner_oof_6 | 0.6700 | 0.0900 | no | +0.0066 | [-0.0032, +0.0156] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |
| A3 | `cell=A3-c,lam=1.0,loss=mse+rank,sampling=natural,stopping=rank` | inner_oof_30 | 0.6180 | 0.0900 | no | +0.0065 | [-0.0039, +0.0162] | inconclusive at 0, and rules out a 0.05 gain -- not a negative result |

**A0 + ESM is not computable.** covers outer folds [0] only, so an A0+ESM blend cannot be pooled over five folds ESM outer-held rows present: 3944.

**None of this is A5's verdict.** A5 needs the ESM arm and an A1 assessment at 30 members; these rows are the machinery exercised on what exists, with a 4-member screen inner surrogate and a 12-member arm component.

## Provenance

- training split only, via `pepstab.data.load_with_splits()` filtered to `split == "train"`; the frozen validation and test splits are never read and never scored
- the primary metric runs through `pepstab.nested.score_fold`, which delegates to the frozen `pepstab.evaluation` — average-rank ties, `UNRANKED_CONTRIBUTION` for a constant prediction on an eligible allele, `validate_finite` before scoring
- every unit reported `thread_pin_ok`; thread counts present: 2. Members at different thread counts are refused inside one average rather than pooled
- stopping metrics CatBoost resolved, carried per configuration rather than inherited: RMSE. §6a withdrew A4/YetiRank partly because its resolved metric was NDCG, which nothing had declared
- one payload digest across the journal: `6dba6fa6a61ff06e`
- the journal's `outer_fold` field is null on every row; the outer fold is read from `fold` and agreed against the `of=` field in each unit key
- `data/rasmussen_et_al_dataset.csv` read-only check: PASS — rasmussen_et_al_dataset.csv: OK

