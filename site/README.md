# site

Local static submission view for the **audited round-two development assessment,
12 October 2026**. The primary finding, methods, bounded transfer, complementarity,
execution repeats, provenance gap, cost accounting and open independent confirmation
are backed by [the current round-two assessment](../reports/further_research/round2/REPORT.md).
The completed [matched ranking-loss follow-up](../reports/finalization/ranking/REPORT.md)
and [portable inference package](../reports/finalization/inference/README.md) now
have their own source-backed sections. Ranking intervals cross zero and the R5
deployment selector retained MSE in all five folds. Stronger transfer controls
remain explicitly pending in the [combined readout](../reports/finalization/FINAL_REPORT.md).
The original October 3–4 film, approaches and held-out test are retained in a clearly
dated, collapsed archive. Their original numeric results are unchanged.

The current result is development evidence on the original training rows, not a
new independent confirmation. The lower bound of the peptide-held-out selector gain
is positive but below the prespecified 0.05 criterion. Transfer comparisons concern
complete procedures against narrow sequence comparators. Budget accounting is
**$76.8388 conservatively allocated against $100**, including the full retained $2
engineering allowance; it is not a final invoice or live credit balance.

The page uses static generated HTML, system-font fallbacks, and local film assets.
The current results work without JavaScript, a backend, or network access. Evidence
links point to the repository; the downloadable numeric/hash snapshot is local.
Local source changes do not update the published site until deployment runs.

**Published to GitHub Pages at
<https://cheparukhin.github.io/peptide-hla-stability/>** by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml), which uploads
this folder on every push to `main` that touches it. Pages cannot serve an
arbitrary folder from a branch and `docs/` is already the project's data
documentation, so the workflow uploads `site/` as an artifact rather than using
the built-in "/docs folder" source.

| File | What it is |
|---|---|
| `index.html` | The page. Inline CSS, system fonts, offline-readable result sections. |
| `round2-build.py` | Standard-library evidence builder/checker; writes only the marked current section and payload. |
| `round2-data.json` | Selected saved comparisons, procedure scores, cohort, conservative accounting and SHA-256 source hashes. |
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

Open `http://localhost:8000` at desktop and mobile widths. Check the primary
paired gain and criterion statement, expand the procedure/control tables, check
the transfer table scrolls on a narrow screen, open the source payload, and expand
the dated archive/film. All current content remains readable with JavaScript disabled.
The historical live film requires JavaScript; the MP4 is its fallback.

The builder validates unique selected CSV rows, finite scores, delta consistency,
point/interval agreement, the authored primary-criterion interpretation, the budget
ceiling, zero locked-repeat differences, completed ranking audit and MSE fallback,
ranking interval interpretation, and exact model-export completion. `--check` compares the generated HTML
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
and the pending transfer specification. Updating the combined report after transfer
completion requires regenerating the snapshot and reviewing its pending-state copy.

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

Published to GitHub Pages at
<https://cheparukhin.github.io/peptide-hla-stability/>. Pushing to `main` with
changes under `site/` redeploys it; there is nothing to upload by hand.

Rebuild the current evidence snapshot before pushing if its saved inputs changed.
Run `python3 site/round2-build.py --check` before release.

Rebuild `film/` before pushing if any scene source or `film_data.js` changed —
the workflow publishes this folder as-is and does not run the build.

This replaces the earlier private Claude Artifact, which is no longer the
canonical link.
