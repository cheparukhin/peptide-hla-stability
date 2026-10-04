# Submission film: shot-by-shot scenario

A ~2-minute film about predicting peptide-HLA dissociation half-life, made for
the London AI x Science protein engineering track (3-4 October 2026).

This document is self-contained. Every number it asks you to put on screen is
written out here. A voiceover artist, an editor or a developer can work from
this file alone.

## The thesis

The film is a bet that loses, with the receipt attached.

A decision threshold is announced before any result exists. An expensive
structural-prediction arm is then run at real cost. It loses to the cheap
sequence model. The negative result is only credible because the rule was
declared first, so the film's job is to make the audience watch the rule being
set, watch the money being spent, and then watch the rule being honoured.

Three consequences for the edit:

- The threshold card is declared in Act III and stays on screen until the
  verdict resolves against it. It is never re-introduced, because
  re-introducing it would imply it was chosen after the fact.
- The fold storm in Act IV must feel like real spend, not a progress bar. The
  money counter is the emotional beat.
- Act V does not gloat. The losing arm is presented with the same bar, the same
  interval and the same axis as the winner.

## Running order

| Act | Start | Dur | Name |
|---|---|---|---|
| 0 | 0:00 | 8s | Cold open: "17.5 hours" |
| I | 0:08 | 38s | The existing fragment: how a peptide is presented |
| II | 0:46 | 12s | The question |
| III | 0:58 | 18s | The data, and the firewall |
| IV | 1:16 | 24s | The race, and the burn |
| V | 1:40 | 18s | The verdict |
| VI | 1:58 | 10s | What survived |

Total 2:08.

**Cutting to a hard 2:00.** Trim Act IV from 24s to 20s and Act III from 18s to
14s. In Act IV, take the 4s out of the fold storm hold (1:22-1:34), not out of
the three-lane setup and not out of the zero-failures hold. In Act III, take
the 4s out of the dataset counters (1:02 of screen time becomes 0:02) and the
histogram; do not shorten the firewall beat or the threshold declaration, which
are the acts the thesis rests on. Everything after the trimmed act shifts
earlier by the amount removed.

## Two visual languages

The film changes its visual register once, at 0:58, and never changes back.

**Acts 0 to II are photographic.** Pre-rendered molecular stills, multiply-
blended over a warm grey radial wash (`#F5F6F9` centre to `#D7DAE0` edge), with
a soft vignette. Camera moves are scale-and-offset pushes on the still, not 3D
camera moves. Depth, softness, light.

**Acts III to VI are an instrument.** Hairline rules, flat fills, no gradients,
no drop shadows, no bevels, no easing that overshoots. Axes have ticks and
units. The reference is laboratory equipment and a printed results table, not a
pitch deck. Nothing glows after 0:58 except the one crimson accent.

The handover at 0:58 is a dissolve, not a cut. The wash stays; the molecules
leave; the rules draw on.

## Colour and type

| Role | Hex |
|---|---|
| Ink (all text, all rules) | `#2B2D42` |
| Peptide crimson (the one accent) | `#B22222` |
| HLA helices | `#8A8B90` |
| HLA sheet floor and loops | `#DCDDE2` |
| HLA surface | `#BABBC0` |
| TCR alpha chain | `#00B4D8` |
| TCR beta chain | `#1D3557` |
| Contact glow | `#E0FAFF` |

**The convention, inherited from the existing fragment: the HLA stays grey so
that the peptide is the only element that changes colour.** Hold this in the
data acts too. Crimson marks the thing under test and nothing else: the decaying
peptide in Act 0, the blocked peptide at the firewall in Act III, the `+0.05`
threshold line in Act V, the wrong-HLA collapse in Act VI. If a chart needs a
second category colour, use a grey from the HLA set.

Typeface IBM Plex Mono throughout, weights 400/500/600. Stage is 1920x1080.
Reference frame rate 30 fps; the one hard cut is specified in frames, so a
different rate needs it restated.

## Act 0 - Cold open: "17.5 hours" (0:00-0:08, 8s)

Black, then a single measurement, stated before anything is explained.

| Time | Shot |
|---|---|
| 0:00.0-0:00.6 | Full black. Silence. |
| 0:00.6-0:02.0 | The grey HLA groove fades up from black onto the warm wash. A crimson peptide sits in the cleft at full opacity. No labels yet. |
| 0:02.0-0:05.5 | A timer in the lower third counts hours, compressed to seconds: it sweeps 0 h to 35 h across these 3.5s, so one screen second is ten hours. The crimson peptide fades on an exponential decay over the same window. At 0:03.75 the timer reads 17.5 h and the peptide is at exactly half its starting opacity; hold a single beat there and let the timer run on. |
| 0:05.5-0:08.0 | The groove recedes. Title lands centred. |

**On-screen text.**

- Timer, lower third, monospace, ink: `0.0 h` counting to `35.0 h`.
- Pinned under the timer from 0:02.0: `SLLMWITQV on HLA-A*02:01 -- measured
  half-life 17.5 h`
- Title, 0:05.5: `HOW LONG DOES IT STAY?` with the subtitle
  `peptide-HLA dissociation half-life` on the line below, in ink, no crimson.
- Honesty caption, bottom edge, from 0:02.0 to 0:05.5, 60% ink:
  `Illustrative decay at the measured half-life. Not a simulated trajectory.`

That caption is mandatory. The curve is drawn to pass through the measured
half-life; it is not a physical simulation and no frame of it is data.

**Caption/VO.** "One peptide. One HLA molecule. Seventeen and a half hours
before half of them have come apart."

**Structure.** PDB 2BNQ at 1.70 Angstrom resolution, the same file used
throughout Act I.

## Act I - The existing fragment (0:08-0:46, 38s)

**This act is already shot. Do not re-storyboard it.** It is the four-beat pMHC
intro, mounted unchanged on the film's timeline. The description below is so an
editor knows what arrives and when, and can trim around it.

Source: fourteen pre-rendered molecular stills at 1280x720, composited in 2D
and scaled 1.5x to the 1920x1080 stage.

**Provenance.** Hero structure is PDB 2BNQ at 1.70 Angstrom resolution: the 1G4
T-cell receptor bound to the NY-ESO-1 peptide `SLLMWITQV` on HLA-A*02:01.
Chains are A heavy chain, B beta-2 microglobulin, C peptide, D TCR alpha,
E TCR beta. One file covers the HLA-alone, groove and receptor shots by hiding
and unhiding chains D and E, so the docking geometry is crystallographic in
every frame and no pose is interpolated. The two alternative candidate peptides
are superposed from PDB 9SL0 and 1TVB, at 0.42 and 0.28 Angstrom RMSD on the
alpha-1/alpha-2 helices.

**A standing constraint.** Peptide instability in beat 3 is animated in 2D, as
screen-space offset and rotation of a separately rendered peptide layer over an
empty-groove layer shot from an identical camera. No molecular coordinate is
distorted, so no frame implies a conformation that was not observed.

**The four beats, as twelve cues.**

| Beat | Cue | Start | Dur | What is on screen |
|---|---|---|---|---|
| 1 | Groove | 0:08.0 | 2.0s | HLA surface, camera pushes toward the groove |
| 1 | Anatomy | 0:10.0 | 4.6s | Groove anatomy: two helices, the sheet floor, nine residues in the cleft |
| 2 | Candidates | 0:14.6 | 3.6s | Four candidate peptides cycle through the groove, changing colour and shape |
| 2 | Lock | 0:18.2 | 1.6s | One candidate locks; the pair is stable |
| 2 | Approach | 0:19.8 | 3.2s | The T-cell receptor descends as a surface |
| 2 | Interface | 0:23.0 | 2.6s | The receptor resolves to V-domain ribbons, close but not touching |
| 3 | Wobble | 0:25.6 | 5.0s | A loose peptide wobbles and escapes the groove |
| 3 | Anchors | 0:30.6 | 2.6s | A second peptide settles, anchor residues seated in their pockets |
| 3 | Contact | 0:33.2 | 3.6s | A receptor loop contacts the peptide; the contact glows |
| 4 | Activation | 0:36.8 | 2.0s | The T cell activates |
| 4 | Expansion | 0:38.8 | 4.2s | One clone expands out of the naive repertoire |
| 4 | Body | 0:43.0 | 3.0s | Pull out to the whole-body response |

The cue durations sum to exactly 38.0s. The beats are fixed and must not be
reordered or expanded.

**Caption/VO** (four lines, one per beat, timed to the beat starts):

- 0:10 "An HLA molecule holds a short protein fragment in a groove on the cell
  surface, so passing T cells can read it."
- 0:15 "Many fragments compete for the groove. One fits."
- 0:26 "A loose fragment falls out before anyone reads it. A stable one stays
  seated."
- 0:37 "If a T cell reads it, that clone multiplies. How long the fragment
  stays is part of whether that happens at all."

Gloss on first use, in caption or VO: *HLA* is the molecule that displays the
fragment; *peptide* is the fragment itself, nine amino acids long here;
*T-cell receptor* is the protein that reads the pair.

## Act II - The question (0:46-0:58, 12s)

Register is still photographic for the first beat, then begins to flatten.

| Time | Shot |
|---|---|
| 0:46.0-0:49.0 | Hold on the single body figure from the end of Act I, then pull back: the figure repeats into a field of identical small grey figures filling frame. No crimson. |
| 0:49.0-0:53.5 | The field resolves into a pair of axes. Two curves draw on, sharing the same x axis. The first is labelled `affinity -- how tightly`. The second, in crimson, is labelled `half-life -- how long`. The two curves are deliberately not parallel: they are drawn to show that the two quantities are different questions about the same complex, not two views of one. Label the axes or leave them unnumbered -- do not invent tick values. |
| 0:53.5-0:58.0 | Curves fade to 20%. The research question types in, two lines, left-aligned, centred block. The second line gets a crimson underline drawn left to right over 0.5s and held. |

**On-screen text, 0:53.5:**

```
Does a pretrained protein model beat a plain supervised one?
And is the gain worth the compute?
```

The second line carries the crimson underline. That underline is the film's
promise, and Act V is the payment.

**Caption/VO.** 0:49: "Binding affinity says how tightly. Half-life says how
long. They are not the same measurement." 0:54: "So: does a pretrained protein
model beat a plain supervised one -- and is the gain worth the compute?"

## Act III - The data, and the firewall (0:58-1:16, 18s)

The register switches here. Hairlines, flat fills, no gradients from this point
to the end of the film.

| Time | Shot |
|---|---|
| 0:58.0-1:02.0 | Four counters roll up in a row, each with its label beneath, ink on wash. They settle in order, 0.4s apart. |
| 1:02.0-1:05.5 | The counters slide to a header strip. A histogram of half-life draws beneath them. Mass piles hard against the left edge: the leftmost bar is the assay floor and is drawn in crimson. A callout points at it. |
| 1:05.5-1:11.5 | **The firewall.** Two labelled columns, `TRAIN` left and `TEST` right, with a vertical hairline between them. A peptide sequence slides from the train column toward the test column. At the line a crimson rule snaps across and stops it; a badge reads `2 substitutions -- blocked`. The blocked peptide returns to the train column. A second peptide follows, badge `4 substitutions`, and crosses without resistance. Split counts appear under the two columns. |
| 1:11.5-1:16.0 | Everything dims to 30%. The threshold card draws on, centre frame, then docks to the top right corner at 1:15.5 and **stays there, undimmed, through Act V**. |

**On-screen text.**

Counters (1:02 values): `28,166` pairs / `5,633` unique peptides / `75` HLA
alleles / `9` residues per peptide.

Histogram callout: `5,679 rows (20.16%) sit at the assay floor of 0 hours`,
with the sub-line `ties, not zeros`.

Split counts: `train 19,716` / `validation 2,817` / `test 5,633`.

Firewall rule text, under the hairline:
`peptides within three substitutions cannot appear in different splits`

Threshold card, verbatim:

```
DECLARED BEFORE SCORING
minimum worthwhile gain
+0.05 median per-allele Spearman
paired 95% intervals, 2,000 whole-cluster resamples
```

**Caption/VO.** 1:02: "Twenty-eight thousand peptide-HLA pairs, seventy-five
alleles, every peptide nine residues long." 1:03: "One row in five sits at the
floor of the assay -- those are ties, not zeros." 1:06: "Near-duplicate
peptides cannot straddle the split. Within three substitutions, they stay on
the same side." 1:12: "And before any model was scored, we fixed what would
count as a win: five hundredths of a point of median per-allele Spearman."

Gloss *Spearman* once, in caption: `Spearman: rank agreement, 1.0 = perfect
ordering`.

## Act IV - The race, and the burn (1:16-1:40, 24s)

| Time | Shot |
|---|---|
| 1:16.0-1:19.0 | Three horizontal lanes draw in, stacked, each with a name plate at the left: `SEQUENCE`, `ESM-2`, `BOLTZ-2`. A cost gutter runs along the right edge, empty. |
| 1:19.0-1:22.0 | The sequence lane fills end to end almost instantly; its cost chip reads `$9.6e-07 per 1,000 pairs`. The ESM-2 lane follows a beat later; chip reads `$0.00047 per 1,000 pairs`. Both lanes go quiet. The Boltz-2 lane has not moved. |
| 1:22.0-1:34.0 | **The fold storm.** The Boltz-2 lane expands to fill frame as a dense grid of small folding complexes, hundreds of cells, each resolving from noise to a shape and then greying out as it completes. A fold counter climbs from `0` to `28,166`. A money counter climbs beside it from `$0.00` to `$212.71`. The two counters are visually locked: same baseline, same size, same weight. Shard, GPU-hour and wall-clock readouts tick in a side rail. The grid completes at 1:32.5; the counters land together at 1:33.0. |
| 1:34.0-1:40.0 | The grid stills and desaturates to the HLA greys. One line of ink text resolves centre frame and is held for the full six seconds. |

**On-screen text.**

Side rail during the storm:
`282 shards` / `143.61 A10G GPU-hours` / `7 h 32 min wall clock`

Money counter, with its honesty sub-line locked beneath it for the whole storm,
60% ink:
`$212.71 -- derived from recorded container-hours x a measured rate. Not a
reconciled provider bill.`

Hold card, 1:34:

```
28,166 folds.  0 execution failures.
```

with the sub-line `The pipeline did not fail. The features did not help.`

**Caption/VO.** 1:19: "Two of the three arms finish before you can read their
cost." 1:23: "The third folds every complex in the dataset. Twenty-eight
thousand structures, two hundred and eighty-two shards, a hundred and
forty-three GPU-hours, seven and a half hours of wall clock." 1:30: "Two
hundred and twelve dollars and seventy-one cents, derived from recorded
container-hours -- not an invoice." 1:35: "Zero execution failures. The
pipeline did not fail. The features did not help."

## Act V - The verdict (1:40-1:58, 18s)

**The film's only hard cut.** Four frames of black, 1:39.867 to 1:40.000 at
30 fps, immediately before this act. No dissolve, no sound tail. The threshold
card from Act III is still parked top right when the picture returns.

| Time | Shot |
|---|---|
| 1:40.0-1:46.0 | Six horizontal bars draw left to right, longest at the top, each with its paired 95% interval as a whisker. The baseline row's bar is ink; the Boltz-2 row's bar is the only crimson one. A header strip states the scoring scope. |
| 1:46.0-1:52.0 | The bars re-anchor: the baseline collapses to a zero line and the other five become deltas against it. A crimson vertical line draws at `+0.05` and is labelled from the parked threshold card, which brightens to full for this beat. The two ESM whiskers visibly cross zero and visibly stop short of the crimson line. The Boltz-2 whisker sits entirely left of zero. |
| 1:52.0-1:58.0 | The axes rotate: accuracy becomes the y axis and cost per 1,000 pairs becomes a log-scale x axis spanning nine decades. The six arms land as points. The trend runs down and to the right. Hold. |

**The six arms.** Test scoring, 5,565 rows across 67 eligible alleles. Median
per-allele Spearman, paired delta against the sequence-ensemble baseline with
95% interval, and USD per 1,000 pairs.

| Arm | Median Spearman | Delta vs. baseline [95% CI] | USD / 1,000 pairs |
|---|---|---|---|
| Sequence ensemble, 30 nets | 0.7064 | baseline | $9.6e-07 |
| ESM-2 ensemble (ESM only) | 0.6979 | -0.0085 [-0.0291, +0.0233] | $0.00047 |
| Sequence ensemble, full domain | 0.6904 | -0.0160 [-0.0388, +0.0043] | $3.8e-06 |
| Sequence + ESM-2 ensemble | 0.6816 | -0.0248 [-0.0460, +0.0073] | $0.00047 |
| MLP, peptide+pseudoseq (single net) | 0.6181 | -0.0883 [-0.1201, -0.0555] | $6.6e-09 |
| Boltz-2 structural ensemble | 0.5947 | -0.1117 [-0.1339, -0.0575] | $6.90 |

Header strip text: `test set, scored once -- 5,565 rows, 67 alleles`.

**What the delta view must make legible, in this order.** The two ESM intervals
cross zero, so their direction is unresolved; and their upper bounds, `+0.0233`
and `+0.0073`, both fall short of `+0.05`, so a worthwhile gain is excluded.
The Boltz-2 interval lies entirely below zero, so that arm is worse. Do not let
the animation imply ESM is "proven equal" -- the whisker crossing zero is the
point, and it must stay visible.

**Caption/VO.** 1:41: "Six arms, scored once on the test set." 1:47: "The
pretrained-embedding arms land either side of zero -- direction unresolved. But
both upper bounds stop short of the line we drew first, so the worthwhile gain
is ruled out." 1:51: "The structural arm's whole interval sits below zero. It
is worse." 1:53: "Plot accuracy against cost and the line runs the wrong way.
The cheapest arm is the best one."

## Act VI - What survived (1:58-2:08, 10s)

| Time | Shot |
|---|---|
| 1:58.0-2:02.0 | The scatter wipes. A single ROC curve sweeps up from the origin and holds hard against the top-left corner. Area under it fills in flat ink at 15% opacity. The AUROC figure and its spread resolve beside it. |
| 2:02.0-2:05.5 | Three control markers drop onto the same axis in sequence, 0.7s apart: random, other-allele decoys, wrong HLA. The third is animated -- the curve visibly slumps from the 0.9656 position toward 0.6965 over 0.8s, in crimson, and stays slumped. |
| 2:05.5-2:08.0 | Everything clears to the plain wash. The final card types on, two lines, with the credit beneath it. |

**On-screen text.**

ROC readout:
`AUROC 0.9656 (IQR 0.9435-0.9779, min 0.8046, max 0.9907)`
`81,600 naturally presented peptides vs. 816,000 proteome decoys, 51 alleles`
`AUPRC 0.7804 against a chance baseline of 0.0909`

Controls, as a stacked list that builds:

| Control | AUROC |
|---|---|
| Random scores | 0.4974 |
| Other-allele ligands as decoys | 0.9157 |
| Wrong HLA pseudosequence | 0.6965 |

Honesty caption, held for the whole act, 60% ink:
`Elution is a different assay, not a second measurement of stability accuracy.
No retraining.`

Final card, 2:05.5:

```
The expensive arm lost.
The threshold we declared first is why we can say so.
```

Credit beneath, small, 60% ink:
`Limitations and an earlier test exposure are disclosed in REPORT.md,
sections 2 and 8.`

**Caption/VO.** 1:59: "The cheap model does transfer. Without retraining, it
separates naturally presented peptides from proteome decoys with a median AUROC
of 0.9656." 2:03: "Give it the wrong HLA and it collapses to 0.6965 -- so it is
reading the pairing, not memorising peptides." 2:06: "The expensive arm lost.
The threshold we declared first is why we can say so."

Gloss *elution* on first use, in caption: `elution: peptides recovered from
real cells, a different experiment from the half-life assay`.

## Honesty register

Four claims must appear on screen, in the act listed, legibly, for at least
2.5 seconds each. They are not optional and they are not footnotes.

| # | Act | Claim as it appears |
|---|---|---|
| 1 | 0 | `Illustrative decay at the measured half-life. Not a simulated trajectory.` |
| 2 | IV | `$212.71 -- derived from recorded container-hours x a measured rate. Not a reconciled provider bill.` |
| 3 | VI | `Elution is a different assay, not a second measurement of stability accuracy.` |
| 4 | VI | `Limitations and an earlier test exposure are disclosed in REPORT.md, sections 2 and 8.` |

Wherever the `$212.71` figure appears -- counter, caption, VO, thumbnail,
anywhere -- the derived qualifier travels with it.

One further standing constraint, inherited rather than stated on screen: in
Act I no frame distorts a molecular coordinate (see the Act I note). Nothing in
Acts III to VI is a reconstruction; every mark is plotted from a measured
value listed in this document.

## Caption and VO script

Collected for recording. Timecodes are the line's in-point on the 2:08 cut.
Lines are sized for roughly 2.5 words per second, which is a production
assumption, not a measurement -- time the read and adjust the in-points, not
the numbers.

| In | Act | Line |
|---|---|---|
| 0:02 | 0 | One peptide. One HLA molecule. Seventeen and a half hours before half of them have come apart. |
| 0:10 | I | An HLA molecule holds a short protein fragment in a groove on the cell surface, so passing T cells can read it. |
| 0:15 | I | Many fragments compete for the groove. One fits. |
| 0:26 | I | A loose fragment falls out before anyone reads it. A stable one stays seated. |
| 0:37 | I | If a T cell reads it, that clone multiplies. How long the fragment stays is part of whether that happens at all. |
| 0:49 | II | Binding affinity says how tightly. Half-life says how long. They are not the same measurement. |
| 0:54 | II | So: does a pretrained protein model beat a plain supervised one -- and is the gain worth the compute? |
| 1:02 | III | Twenty-eight thousand peptide-HLA pairs, seventy-five alleles, every peptide nine residues long. |
| 1:03 | III | One row in five sits at the floor of the assay -- those are ties, not zeros. |
| 1:06 | III | Near-duplicate peptides cannot straddle the split. Within three substitutions, they stay on the same side. |
| 1:12 | III | And before any model was scored, we fixed what would count as a win: five hundredths of a point of median per-allele Spearman. |
| 1:19 | IV | Two of the three arms finish before you can read their cost. |
| 1:23 | IV | The third folds every complex in the dataset. Twenty-eight thousand structures, two hundred and eighty-two shards, a hundred and forty-three GPU-hours, seven and a half hours of wall clock. |
| 1:30 | IV | Two hundred and twelve dollars and seventy-one cents, derived from recorded container-hours -- not an invoice. |
| 1:35 | IV | Zero execution failures. The pipeline did not fail. The features did not help. |
| 1:41 | V | Six arms, scored once on the test set. |
| 1:47 | V | The pretrained-embedding arms land either side of zero -- direction unresolved. But both upper bounds stop short of the line we drew first, so the worthwhile gain is ruled out. |
| 1:51 | V | The structural arm's whole interval sits below zero. It is worse. |
| 1:53 | V | Plot accuracy against cost and the line runs the wrong way. The cheapest arm is the best one. |
| 1:59 | VI | The cheap model does transfer. Without retraining, it separates naturally presented peptides from proteome decoys with a median AUROC of 0.9656. |
| 2:03 | VI | Give it the wrong HLA and it collapses to 0.6965 -- so it is reading the pairing, not memorising peptides. |
| 2:06 | VI | The expensive arm lost. The threshold we declared first is why we can say so. |

If the film ships caption-only, every line above is also the caption text, set
in IBM Plex Mono 400 in ink at the lower third, two lines maximum, with the
honesty captions from the register rendered separately at 60% ink so they are
never mistaken for narration.

## Where each act lives in code

The film is assembled in `src/animations/submission-film/`. Acts are plain
components taking `{T, t0, L}` and animating on local time `T - t0`, mounted on
one continuous timeline by `film-scene.jsx`.

| Act | File | Status |
|---|---|---|
| 0, II | `act0-coldopen.jsx` | written |
| I | `pmhc-scene-v2.jsx` (symlink to `../pmhc-intro/`) | written, do not edit |
| III | `act3-data.jsx` | not yet written |
| IV | `act4-race.jsx` | not yet written |
| V | `act5-verdict.jsx` | not yet written |
| VI | `act6-transfer.jsx` | not yet written |

The act durations in this document must match three places or the timeline
drifts: the `ACTS` array in `film-scene.jsx`, the `OM_SCENES` list in
`index.html`, and the Act I cue sum of 38.0s. The measured figures used on
screen are generated into `film_data.js` by `build_film_data.py`; the values in
this document and the values in that file must agree.
