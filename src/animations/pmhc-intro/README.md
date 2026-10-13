# Peptide–HLA biology animation

An illustrated introduction to peptide presentation and T-cell recognition.
It explains biological context; it does not depict predicted kinetics or
establish that a stable complex will activate a T cell. Dissociation, docking,
signalling and cellular expansion are illustrative motion.

The main molecular stills use [PDB 2BNQ](https://www.rcsb.org/structure/2BNQ),
a peptide–HLA–TCR crystal structure at 1.70 Å resolution. Alternative peptide
images and the body illustration are presentation assets. Screen-space movement
of these stills is not a molecular simulation.

## Build

```bash
node src/animations/pmhc-intro/build_site_bundle.mjs
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000/intro/`. The builder combines `animations-v3.jsx`
and `pmhc-scene-v2.jsx`, transpiles with vendored Babel, and copies the runtime
and molecular stills into `site/intro/`. The page uses local assets only.
Changes to captions or scene code require rebuilding before publication.

The stage is 1920 × 1080. Molecular stills are 1280 × 720; `shots/` paths resolve
relative to the generated page. Timing controls the illustration only.
