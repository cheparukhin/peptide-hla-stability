# site

Local static submission view for the **audited development assessment and
completed finalization studies, updated 13 October 2026**. Main, stronger
transfer, complementarity, ranking, hybrid, label-budget and portable-inference
results are backed by the [combined readout](../reports/finalization/FINAL_REPORT.md)
and linked saved artifacts. The subsequently authorized ProteinMPNN/few-shot
tranche has a [frozen protocol](../reports/followup3/PROTOCOL.md). Its
[pre-efficacy amendment](../reports/followup3/THIRD_PRIMARY_AMENDMENT.md) adds
conditional repaired-energy complementarity and three-primary family control.
Few-shot fitting, exact saved-state replay and independent numerical/ancestry
audit are complete. Its gain is positive under both individual 95% and
three-primary 98.333333% intervals; see the generated absolute learning curves.
ProteinMPNN extraction and all 900 head fits are complete and independently
audited. Context augmentation remains inconclusive beyond Boltz. Repaired-
descriptor extraction, 300 new head fits and independent recovery/matrix/
assessment audits are complete; its R5 increment remains inconclusive.
The balanced-panel figure shows all eight MLP arms and both descriptor family
intervals. Few-shot is the sole positive original-family primary. One selected
19,716-pair ProteinMPNN extension is complete and independently audited,
with no established additional R5 value. All three arms, 12 contrasts and
24 fixed-query descriptive entries are preserved. Computational search stops;
independent kinetic-assay confirmation is next.
The [completed independent-data metadata/methods audit](../reports/followup3/independent_data/PROVENANCE.md)
preserves all 129 candidates and their external candidate outcome reservation,
while exposing
heterogeneous endpoints and unresolved physical constructs/measurement lineage.
The original October 3–4 film, approaches and held-out test are retained in a
clearly dated, collapsed archive. Their original numeric results are unchanged.

All later performance assessments are conditional development evidence on the
original training rows. The main gain is positive but its lower bound is below
the prespecified +0.05 worthwhile-gain criterion. Ranking and hybrid primary
intervals span zero. Matched transfer heads support representation utility;
they do not establish structural causality or independent confirmation.
Historical round-two and finalization cost records are separate from the new
tranche's $139 inclusive ceiling. The new tranche is closed at $67.302731
conservatively, with $0 reserved. Scientific outputs are preserved locally,
temporary tranche storage is deleted and historical caches remain. See the
[final closeout](../reports/followup3/resources/cloud_closeout.json). These are accounting records,
not final invoices or a live credit balance.

The page uses static generated HTML, system-font fallbacks, and local film assets.
The current results work without JavaScript, a backend, or network access. Evidence
links outside `site/` point to tracked files on GitHub `main`; the downloadable
numeric/hash snapshot and media remain local to the Pages artifact. Pages uploads
only `site/`, so `../reports` and `../docs` URLs would not resolve after deployment.
Local source changes do not update the published site until deployment runs.

**GitHub Pages deployment target:
<https://cheparukhin.github.io/peptide-hla-stability/>.** The current release is
validated through a push and successful Pages workflow; the workflow result
confirms publication. Deployment uses
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml), which validates and uploads
this folder on matching pushes to `main` (site, reports, docs, scripts or workflow changes). Pages cannot serve an
arbitrary folder from a branch and `docs/` is already the project's data
documentation, so the workflow uploads `site/` as an artifact rather than using
the built-in "/docs folder" source.

| File | What it is |
|---|---|
| `index.html` | The page. Inline CSS, system fonts, offline-readable result sections. |
| `round2-build.py` | Standard-library evidence builder/checker; writes the marked current section/payload and copies verified figure assets. |
| `round2-data.json` | Selected saved comparisons, procedure scores, cohort, conservative accounting and SHA-256 source hashes. |
| `figures/fewshot_curves.svg` / `.png` | Copied source-backed few-shot curves and primary family interval; scientific exports originate in `reports/followup3/figures/`. |
| `figures/descriptor_comparison.svg` / `.png` | All eight balanced-panel MLP scores and both descriptor primary family intervals, with portable glyph paths and a source-hash manifest. |
| `film/` | Generated — the submission film as a static page. Do not hand-edit. |
| `intro/` | Generated — the original 30-second animation, live. Do not hand-edit. |
| `intro.mp4` | The intro animation, re-encoded for web: 1280×720, **H.264 Constrained Baseline, level 3.0**, yuv420p, faststart, no audio track (2.9 MB). The 20 MB source is `pMHC Intro v2.mp4` in the repo root. |
| `intro.webm` | VP9 fallback for browsers that refuse the MP4 (2.0 MB). |
| `poster.png` | Video poster frame. |

The first encode used H.264 **High** profile and did not play in the artifact
viewer. Baseline is the widest-compatibility profile; keep it. The `<video>`
element carries both sources plus an explicit codec string
(`avc1.42E01E`) and falls back to a download link.

## Current evidence and validation

From the repository root:

```bash
python3 site/round2-build.py
python3 site/round2-build.py --check
python3 -m http.server 8000 --directory site
```

Run the checks from a clean checkout containing the intended release commit.
The builder uses Python standard-library modules and requires the Git CLI in
a real Git checkout; a plain source archive cannot run its publication guard.
Run release checks from a clean clone. Every static HTML `href`, `src` and
`poster` is validated: relative dependencies must stay inside `site/`, and local
assets and GitHub evidence targets must exist and be Git-tracked. Untracked
local files cannot conceal dependencies that would be absent from the release. Evidence URLs resolve on
GitHub after the corresponding commit reaches `main`.

Open `http://localhost:8000` at desktop and mobile widths. Check the primary
paired gain and criterion statement, expand the procedure/control tables, check
the transfer table scrolls on a narrow screen, open the source payload, and expand
the dated archive/film. All current content remains readable with JavaScript disabled.
The historical live film requires JavaScript; the MP4 is its fallback.

The builder validates unique selected CSV rows, finite scores, delta consistency,
point/interval agreement, the authored primary-criterion interpretation, the budget
ceiling, zero locked-repeat differences, completed ranking audit and MSE fallback,
ranking interval interpretation, and exact model-export completion. Few-shot
checks require passed cloud/independent audits, frozen fit coverage, primary
arithmetic, shared-panel identity, target/source budgets, exact plotted figure rows/asset hashes, positive group-deletion
points and the family interval's saved-draw/amendment hashes. `--check` compares the generated HTML
and JSON against current saved inputs and fails on stale content or source hashes.
It never executes models, reads the original test for selection, edits saved research
evidence, accesses the network or launches paid compute.

Inputs: round-two main pooled scores/paired comparisons, complementarity paired
comparisons, crossed transfer comparisons, runtime/cost accounting, cache and
embedding-repeat audits. The canonical assessment and audit/independent-data
narratives are hash-bound too. New results require reviewing the authored prose and
regenerating; the builder does not infer new scientific claims from a higher score.
The current follow-up also binds ranking comparisons/scores/selections and audit,
the portable distribution manifest, the unsent assay brief, the combined report,
and the historical transfer specification. It also binds the independent-data
metadata manifest, publication metadata, method-status inventory and provenance
report. The new follow-up sources bind the shared protocol, pre-efficacy
third-primary amendment, FoldX technical receipt, few-shot curves/comparisons,
source/target budgets, independent audit and reproducible audit script. The
completed inverse study also binds all 24 score rows/28 contrasts, its current
three-primary interval, 900-fit independent audit, parser-recovery receipt and
full-cohort decision framework. Completed energy tables, recovery/matrix and
assessment audits, the descriptor figure and selected full-cohort decision are
also bound and validated. The
subordinate offset diagnostic is explicitly post-hoc; its nominal intervals,
rank/arithmetic audit, source ancestry and unchanged primary family are checked
and hash-bound too. Any
combined-report, submission-brief or saved-source change requires regeneration.

The completed full-cohort presentation is gated by saved-output audits:

```bash
python3 site/round2-build.py --check-full-cohort
```

It requires complete and passed independent audits before reading outcomes,
checks all three absolute arms for both learners/panels, the 12 conditional
95% contrasts, and all 24 fixed-query descriptive entries. Missing or unaudited
results are rejected. This command checks the tables without writing a snapshot;
the default builder integrates the completed, audited readout.
No full-cohort metric, fit or bootstrap is generated by this presentation path.

The generated section lies between `ROUND2 GENERATED START` and `ROUND2 GENERATED END`
in `index.html`. Change its template in `round2-build.py`, then rebuild. Other parts
of the page, including the dated archive, are maintained in `index.html`. The snapshot
schema is `round2-site-snapshot.1`; original source paths/hashes accompany selected
rows so the payload can be traced back to saved evidence.

## Historical numbers

The archive table is the **original held-out test split** (5,565 rows, 67 alleles)
from [`../reports/stage6_test_summary.csv`](../reports/stage6_test_summary.csv) and
[`../reports/stage6_test_paired_ci.csv`](../reports/stage6_test_paired_ci.csv).
The external presentation AUROC and secondary experiments are original validation
results. These scores do not establish independent round-two confirmation and
cannot be subtracted from round-two scores across assessment regimes.

The original test pass scored six arms. FoldX and ESM-2 150M were not among them,
so they appear in the historical approach list but not its test table.

## The film

`film/` is **generated**. Rebuild it with:

```bash
node src/animations/submission-film/build_site_bundle.mjs
```

It runs 1:34 and plays live in the browser rather than as a recording — the
same composition the repo renders, embedded in `index.html` through a sandboxed
iframe so its globals cannot collide with the page. Every number on screen is
read at runtime from `film/film_data.js`, which
`src/animations/submission-film/build_film_data.py` generates from the frozen
result tables; nothing is typed into a scene.

The build concatenates the scene sources **into one script** and transpiles
them with the vendored Babel at build time. Both matter:

- The sources share one scope and have no `import`/`export`. A `const` declared
  inside an `eval` is scoped to that `eval`, so evaluating them file-by-file
  silently loses every shared constant while appearing to work — only
  `function` declarations leak. One combined script avoids this.
- Transpiling at build time drops the 3 MB Babel payload and the transform
  pause the development page pays on every load.

`src/animations/submission-film/index.html` is the development page: it fetches
each `.jsx` and transforms in-browser, and carries the tweaks panel and a `?t=`
seek used for frame review. The published page keeps `?t=` and stubs the panel.

## The original animation

The same build also emits `intro/` — the original pMHC interface animation
running live rather than as `intro.mp4`. Both are kept: the MP4 is the
no-JavaScript fallback and the download.

Its cue sheet is **not** in the repo (it lived in the artifact host page), so
the build derives each cue's duration from the lower bound the beat code
itself implies — the frame at which that beat's last callout, residue pop or
ripple finishes. Those bounds sum to **29.9 s, exactly the duration of
`intro.mp4`**, which is the evidence that the reconstruction reproduces the
original pacing rather than approximating it. If a beat's choreography is ever
retimed, re-derive the bound rather than nudging the number until it looks
right.

## Regenerating the intro video

```bash
# MP4 — Constrained Baseline so every browser accepts it
ffmpeg -y -i "pMHC Intro v2.mp4" -vf "scale=1280:-2,format=yuv420p" \
  -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p \
  -crf 28 -preset slow -g 60 -movflags +faststart -an site/intro.mp4

# WebM fallback
ffmpeg -y -i "pMHC Intro v2.mp4" -vf "scale=1280:-2" \
  -c:v libvpx-vp9 -crf 36 -b:v 0 -row-mt 1 -cpu-used 4 -an site/intro.webm

# Poster frame
ffmpeg -y -i site/intro.mp4 -vf "select=eq(n\,60)" -vframes 1 site/poster.png
```

Verify before publishing — `moov` must precede `mdat` for progressive playback:

```bash
ffprobe -v error -show_entries stream=profile,level,pix_fmt site/intro.mp4
ffmpeg -v error -i site/intro.mp4 -f null -    # must print nothing
```

## Publishing

Deployment target:
<https://cheparukhin.github.io/peptide-hla-stability/>. Pushing to `main` with
matching site or evidence changes redeploys it; there is nothing to upload by hand.

Rebuild the current evidence snapshot before pushing if its saved inputs changed.
Run both `python3 site/round2-build.py --check` and
`python3 site/round2-build.py --check-full-cohort` in the clean release checkout.
After the Pages workflow succeeds, verify the main page, an evidence link on
GitHub and the local figure/media assets at the deployed URL. A local pass or
push alone does not confirm publication.

Rebuild `film/` before pushing if any scene source or `film_data.js` changed —
the workflow publishes this folder as-is and does not run the build.

This replaces the earlier private Claude Artifact, which is no longer the
canonical link.
