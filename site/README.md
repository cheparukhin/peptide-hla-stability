# site

One-page project site: the task, the intro video, the approaches tested, the
held-out test results, next steps and methods.

| File | What it is |
|---|---|
| `index.html` | The page. Self-contained — all CSS inline, fonts from Google Fonts. |
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

## Regenerating the video

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

Published as a private Claude Artifact. To update it, republish `index.html`
with `intro.mp4` and `poster.png` as supporting files to the same URL.
