# Peptide–HLA Class I Biology Introduction (`src/animations/pmhc-intro/`)

This directory contains the source assets and build script for the illustrated visual introduction to MHC class I antigen presentation and T-cell recognition deployed at `site/intro/`.

The animation provides conceptual biological context illustrating peptide loading, cell-surface persistence, dissociation, and T-cell receptor engagement using rendered stills of the [PDB 2BNQ](https://www.rcsb.org/structure/2BNQ) crystal structure ($1.70\,\text{\AA}$ resolution).

## Rebuilding the Bundle

```bash
node src/animations/pmhc-intro/build_site_bundle.mjs
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000/intro/` to view the compiled scene ($1920 \times 1080$ stage with local assets bundled under `site/intro/`).
