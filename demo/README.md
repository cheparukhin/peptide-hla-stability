# Demo — peptide groove explorer

The two-minute demo page for the submission. Two steps only: **what we found**
and **what's next**.

The centrepiece is a triptych: pick an **HLA allele**, then one of the
**peptides** it was measured against, and see the **groove** that holds it,
alongside the crystallographic entry for that allele.

## Run it

```bash
python3 demo/build_payload.py          # regenerate demo/payload.js from frozen data
python3 -m http.server 8777 --directory demo
# open http://127.0.0.1:8777/
```

`index.html` is a single static page with no build step, no framework and no
runtime network calls. Fonts come from Google Fonts and degrade to a system
stack offline. The file is deliberately pure ASCII (HTML entities in markup,
`\uXXXX` escapes in script), so it renders correctly whether it is served with a
charset header, served without one, or opened over `file://`.

## Files

| File | What it is |
|---|---|
| `index.html` | the page: markup, styles and behaviour in one file |
| `build_payload.py` | regenerates `payload.js` from the frozen project data |
| `payload.js` | generated — a single `window.__DATA__` assignment, ~79 KB |
| `groove_standalone.html` | the 3D companion page — one self-contained 1.9 MB file, see below |

`payload.js` is generated. Do not hand-edit it; change `build_payload.py` and
re-run.

## Where the numbers come from

Everything on the page is measured. Nothing is predicted, estimated or
illustrative.

| Shown | Source |
|---|---|
| 28,166 pairs, peptides, half-lives | `data/rasmussen_et_al_dataset.csv` |
| 34-residue contact pseudosequence | same file, `hla_pseudoseq` |
| Per-allele validation Spearman | `reports/stage6_val_esm_per_allele.csv`, arm `seq_ensemble_pep_pseudo` |
| Distance stratum badge (near / intermediate / distant) | `reports/stage7_allele_holdout_per_allele_seq_pep_pseudo.csv` |
| Crystallographic template per allele | `data/allele_pdb_templates.csv` |
| "solved" pairs — measured *and* in the PDB | `data/pdb_rasmussen_overlap.csv` |
| Headline table | `reports/REPORT.md` §7 (test), §3 (validation) |

The per-allele score shown is the **30-network sequence baseline** — the arm the
whole study is measured against — not an ESM arm. The per-allele values above are
**validation** numbers; the test set was scored once at stage 6 (see
`reports/REPORT.md` §7).

## The groove map is a schematic, not a structure

The groove panel draws the allele's real 34 contact residues around the selected
peptide's real sequence. **It is a contact schematic, not predicted
coordinates.** It is not a Boltz-2 output and must not be presented as one.

For an actual 3D structure the page links out to the RCSB entry — the exact
complex where that peptide–HLA pair has been solved (32 pairs), otherwise the
allele's template entry.

To render real 3D in-page, the Boltz-2 mmCIF outputs would have to be pulled
from the `pepstab-structures` Modal volume and served next to `index.html`;
they are not committed to this repo and are not reachable from a sandboxed
environment.

## `groove_standalone.html` — the 3D companion page

`index.html` is unchanged and stays the schematic page. The companion page is a
**separate** single file that does render real 3D, so the two can be shown
side by side without the submission page depending on a WebGL context.

Everything is inlined — 3Dmol.js 2.4.2, the same `window.__DATA__` payload as
`payload.js` (byte-identical), and four crystal structures as
`window.__PDB_STRUCTURES__`:

| PDB | Allele | Peptide | Å |
|---|---|---|---|
| 1HHK | A\*02:01 | LLFGYPVYV (HTLV-1 Tax) | 2.5 |
| 1DUZ | A\*02:01 | LLFGYPVYV | 1.8 |
| 3MRE | A\*02:01 | GLCTLVAML (EBV EB2) | 1.1 |
| 5HHO | A\*02:01 | GILEFVFTL + JM22 TCR | 2.95 |

Coordinates are RCSB depositions with waters and alternate conformers
stripped; allele, peptide and resolution were checked against
`data.rcsb.org` entry metadata rather than taken from a label.

```bash
open demo/groove_standalone.html        # no server, no network needed
```

It follows the same conventions as `index.html`: pure ASCII (`\uXXXX` escapes
in script), no runtime data fetches, Google Fonts degrading to a system stack
offline. Links out to RCSB are the only outbound requests, and they are
user-initiated.

**Structure precedence per selected pair** — crystal structure where the exact
complex is bundled, otherwise the Boltz-2 Cα trace from the payload, otherwise
the allele's template entry, each labelled in the panel. The Cα trace is a
**prediction**, not a deposition, and is labelled as such; the same rule as the
schematic panel applies — it must not be presented as experimental.

**Not covered by automated checks**: the page boots cleanly under a stubbed DOM
(`node` harness, no browser available in the build environment), which verifies
parsing, payload integrity and the script's boot path — but not the WebGL
render itself. Look at it in a browser before demoing.

## Reading the two badges

- **Distance stratum** — how far the allele sits from its nearest training
  allele under leave-one-allele-out. Median per-allele Spearman falls from
  **0.741** (near, d ≤ 1) through **0.576** (intermediate, d = 2–3) to **0.339**
  (distant, d ≥ 4). That gradient is the cliff the headline refers to.
- **Template tier** — `A` means the template's α1/α2 groove is 100% identical to
  the allele and can be used directly; `B` and `C` mean substitutions, with the
  count inside the 34 contact positions reported.
