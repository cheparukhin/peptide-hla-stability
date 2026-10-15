# Static Project Page (`site/`)

This directory contains the static project page deployed to [https://cheparukhin.github.io/peptide-hla-stability/](https://cheparukhin.github.io/peptide-hla-stability/):

- **[`index.html`](index.html)**: Main web presentation of the study's background, methods, primary experimental results, zero-shot immunopeptidomics transfer, cross-allele transfer, 20-shot adaptation, and secondary extensions.
- **[`round2-data.json`](round2-data.json)**: Snapshot of key evaluation metrics, confidence intervals, and SHA-256 hashes of the source CSV/JSON tables in `reports/`.
- **[`figures/fewshot_curves.svg`](figures/fewshot_curves.svg)**: Vector figure of the 20-shot target-allele adaptation learning curves across 68 alleles.
- **[`intro/index.html`](intro/index.html)**: Illustrated biological introduction to peptide–HLA class I presentation and T-cell recognition (built from [`src/animations/pmhc-intro/`](../src/animations/pmhc-intro/)).

## Local Preview

```bash
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000/` for the main project page or `http://localhost:8000/intro/` for the biology animation.
