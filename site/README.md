# Research results page

The [public results page](https://cheparukhin.github.io/peptide-hla-stability/)
is a concise visual overview of the [manuscript](../reports/MANUSCRIPT.md),
covering development ranking results, transfer, few-shot adaptation, and the
limits of those findings. The manuscript is the main scientific narrative, the
[supplementary report](../reports/REPORT.md) provides supplementary scores,
sensitivities, and historical benchmarks, and saved CSV and JSON files provide
the numerical evidence.

## Build and verify

From the repository root:

```bash
python3 site/round2-build.py
python3 site/round2-build.py --check
python3 site/round2-build.py --check-full-cohort
python3 -m http.server 8000 --directory site
```

Edit the narrative in `round2-build.py` and the page styles in `index.html`.
The builder reads saved results, verifies arithmetic and audit records, copies
the verified learning-curve figure, and writes `round2-data.json` with selected
values and source hashes; it never runs models or changes research evidence.
Run publication checks in a Git checkout, as they also verify that local assets
and repository evidence links are tracked and present.

The page is readable without JavaScript; review its tables at desktop and
mobile widths. The optional `intro/` animation explains peptide–HLA biology—it
is not a model prediction or a stability measurement—and the `film/` route
redirects to the results page.

The [Pages workflow](../.github/workflows/pages.yml) validates and uploads this
folder after relevant changes reach `main`. Repository evidence links therefore
use GitHub URLs rather than paths outside `site/`. A successful deployment,
not a local build, confirms that the public page has changed.
