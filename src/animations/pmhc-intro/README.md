# pMHC intro animation

A four-beat explainer of peptide-HLA pairing and T-cell recognition, built as a continuous
composition over pre-rendered molecular stills.

## Files

| File | What it is |
|---|---|
| `pmhc-scene-v2.jsx` | The scene: timeline, motion constants, layout, 2D compositing |
| `animations-v3.jsx` | Runtime library the scene depends on (`useComposition`, `useTweaks`, timeline/layer hooks) |
| `shots/` | 14 pre-rendered PNGs loaded by the scene |

## How to run it

**These are not ES modules.** Neither file has `import` or `export`; they are written for an
artifact/canvas runtime that evaluates both in one scope. `pmhc-scene-v2.jsx` calls
`useComposition` and `useTweaks`, which are defined in `animations-v3.jsx`, so `animations-v3.jsx`
must be evaluated first. Dropping the scene alone into a React app will fail.

To embed in a normal bundler-based app you need a small wrapper: evaluate both files in one
module scope, add an `export default` to the scene's root component, and serve `shots/` at a path
where `shots/<name>.png` resolves from the hosting page.

Stage is 1920x1080. The molecular stills are 1280x720 and the scene scales them by 1.5
(`const W = 1920, H = 1080, S = 1.5`). `body_figure.png` is 344x778 and is placed separately.

Image paths are hardcoded as `shots/' + name + '.png'` **relative to the page**, not to this
directory - so if the host page lives elsewhere, either copy `shots/` next to it or change the
`IMG` helper at the top of the scene.

No video is rendered here. If you add one, `public/videos/pmhc-intro.mp4` is the natural home.

## The four beats

The narrative is fixed at four voiceover beats and should not be reordered or expanded:

1. HLA surface, zoom into the groove, peptide visible in the cleft
   (`shot1a_hla_surface`, `shot1b_groove_surface`, `shot1c_groove_cartoon`)
2. Candidate peptides change colour and shape in the groove, one locks; TCR approaches as a
   surface then resolves to V-domain ribbons, close but not touching
   (`pep_cand1`-`pep_cand4`, `shot2d_tcr_surface_approach`, `shot2e_vdomain_ribbons`)
3. The peptide wobbles, settles, and the TCR contact glows
   (`layer_groove_empty` + `layer_peptide_alpha` composited, `shot3d_settled_anchors_grey`,
   `shot3e_contact_glow`)
4. T-cell activation, clonal expansion, whole body (`body_figure`, plus drawn graphics)

## Provenance of the stills

Hero structure is **PDB 2BNQ** (1.70 A) - the 1G4 TCR bound to NY-ESO-1 `SLLMWITQV` on
HLA-A*02:01. Chains: A heavy chain, B beta2m, C peptide, D TCR-alpha, E TCR-beta. One file covers
the HLA-alone, groove and TCR shots by hiding and unhiding chains D and E, so the docking geometry
is crystallographic throughout, with no interpolated poses. That peptide has a measured half-life
of 17.5 h in the Rasmussen set. The two alternative candidate peptides are superposed from 9SL0
and 1TVB (0.42 A and 0.28 A RMSD on alpha1/alpha2).

Colour convention: the HLA is grey so that the peptide is the only element that changes colour.
Helices `#8A8B90`, sheet floor and loops `#DCDDE2`, surface `#BABBC0`; peptide crimson `#B22222`,
candidates `#F4D06F` and `#C44D96`; TCR alpha `#00B4D8`, TCR beta `#1D3557`; contact glow
`#E0FAFF`.

**Peptide instability is animated in 2D, never as a 3D coordinate perturbation.** Beat 3
composites `layer_peptide_alpha.png` over `layer_groove_empty.png` - both rendered from an
identical camera - and applies screen-space offsets and rotations. Nothing in the molecular
coordinates is distorted, so no frame implies a conformation that was not observed.
