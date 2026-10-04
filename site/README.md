# site

One-page project site: the task, the intro video, the approaches tested, the
held-out test results, next steps and methods.

| File | What it is |
|---|---|
| `index.html` | The page. Self-contained — all CSS inline, fonts from Google Fonts. |
| `intro.mp4` | The intro animation, re-encoded for web (1280×720, CRF 30, 1.2 MB). The 20 MB source is `pMHC Intro v2.mp4` in the repo root. |
| `poster.png` | Video poster frame. |

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
ffmpeg -i "pMHC Intro v2.mp4" -vf scale=1280:-2 -c:v libx264 -crf 30 \
  -preset slow -movflags +faststart -c:a aac -b:a 96k site/intro.mp4
ffmpeg -i site/intro.mp4 -vf "select=eq(n\,60)" -vframes 1 site/poster.png
```

## Publishing

Published as a private Claude Artifact. To update it, republish `index.html`
with `intro.mp4` and `poster.png` as supporting files to the same URL.
