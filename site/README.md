# site

One-page project site: the task, the two-minute film, the approaches tested,
the held-out test results, next steps and methods.

**Published to GitHub Pages at
<https://cheparukhin.github.io/peptide-hla-stability/>** by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml), which uploads
this folder on every push to `main` that touches it. Pages cannot serve an
arbitrary folder from a branch and `docs/` is already the project's data
documentation, so the workflow uploads `site/` as an artifact rather than using
the built-in "/docs folder" source.

| File | What it is |
|---|---|
| `index.html` | The page. Self-contained — all CSS inline, fonts from Google Fonts. |
| `film/` | Generated — the two-minute film as a static page. Do not hand-edit. |
| `intro.mp4` | The intro animation, re-encoded for web: 1280×720, **H.264 Constrained Baseline, level 3.0**, yuv420p, faststart, no audio track (2.9 MB). The 20 MB source is `pMHC Intro v2.mp4` in the repo root. |
| `intro.webm` | VP9 fallback for browsers that refuse the MP4 (2.0 MB). |
| `poster.png` | Video poster frame. |

The first encode used H.264 **High** profile and did not play in the artifact
viewer. Baseline is the widest-compatibility profile; keep it. The `<video>`
element carries both sources plus an explicit codec string
(`avc1.42E01E`) and falls back to a download link.

## Numbers on the page

The headline table is the **held-out test split** (5,565 rows, 67 alleles) from
[`../reports/stage6_test_summary.csv`](../reports/stage6_test_summary.csv) and
[`../reports/stage6_test_paired_ci.csv`](../reports/stage6_test_paired_ci.csv).
The external-validation AUROC and the secondary experiments are validation
numbers, labelled as such in the footer.

The test pass scored six arms. FoldX and ESM-2 150M were not among them, so
they appear in the approach list but not in the results table.

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

Rebuild `film/` before pushing if any scene source or `film_data.js` changed —
the workflow publishes this folder as-is and does not run the build.

This replaces the earlier private Claude Artifact, which is no longer the
canonical link.
