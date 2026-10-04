# Stage 4b.1: the Boltz-2 MSA cache

This completed cache contains **182-residue groove** alignments. It supplies
arm A of the new [stage 4c pilot](../HACKATHON_PLAN.md#4c-ectodomain--beta-2-microglobulin-folding);
the [ectodomain checklist](../docs/ECTODOMAIN_FOLDING_PLAN.md) specifies the
remaining 275-residue MSAs and cropped-MSA control. Counts and measurements
below describe the original groove cache.

**Status: done.** 75 MSAs — one per unique HLA domain sequence, the whole
dataset, not just the panel. CPU only, no GPU booked, **$0 spent**: 137 s of
wall clock and 141.3 MB on disk.

Regenerate: `<boltz-env>/bin/python scripts/make_msas.py --all-alleles`
(see *Environment* below). Manifest: [`msa_manifest.csv`](msa_manifest.csv).
Parse benchmark: [`msa_parse_benchmark.json`](msa_parse_benchmark.json).

The MSA files themselves live in `structures/msa/<stem>.csv` and are
**gitignored** — 141 MB of regenerable derived data. The committed record is
the manifest, which carries `sha256` of each HLA sequence and of each MSA file,
so the cache can be verified or rebuilt without storing it in git.

## What was generated

Alleles and HLA domain sequences are **1:1 in this dataset** — 75 alleles,
75 distinct domain sequences (asserted before any server call). One MSA per
allele is reused across every complex on that allele: the ~2,000 panel folds
need six MSA files, and the cache now covers any panel we pick.

| | |
|---|---:|
| Alleles cached | 75 / 75 |
| Distinct HLA sequences (sha256) | 75 |
| Distinct MSA files (sha256) | 75 |
| HLA domain length | 182 aa (all) |
| MSA rows: min / median / max | 9,505 / 9,991 / 10,454 |
| Unique ungapped sequences: min / median / max | 9,087 / 9,554 / 10,025 |
| Total size | 141.3 MB |
| Generation time, 75 alleles | 137 s (median 2.0 s each) |
| Cost | $0 — public ColabFold server, local CPU |

The six structural-panel alleles:

| Allele | Stem | MSA rows | Unique ungapped |
|---|---|---:|---:|
| `HLA-B*15:01` | `HLA-B_15_01` | 10,305 | 9,883 |
| `HLA-A*02:01` | `HLA-A_02_01` | 9,771 | 9,347 |
| `HLA-A*03:01` | `HLA-A_03_01` | 9,872 | 9,458 |
| `HLA-B*39:01` | `HLA-B_39_01` | 10,187 | 9,752 |
| `HLA-B*35:01` | `HLA-B_35_01` | 10,252 | 9,826 |
| `HLA-B*07:02` | `HLA-B_07_02` | 10,216 | 9,781 |

**Why all 75 and not the planned 6.** `data/structural_panel.csv` (stage 4b.4)
is not frozen yet, and the hour-5 call may move the panel. At ~2 s and $0 per
allele the full cache costs 137 s, removing the ordering dependency between
4b.1 and 4b.4 entirely. A tier change costs no new MSA work.

Every MSA row carries key `-1` (unpaired) — the single-chain branch that
`msa: empty` on the peptide forces. The first row of each file is the query
HLA sequence; the script asserts both.

## Route: `compute_msa` directly, not the `boltz predict` harvest

The plan prescribed writing a pilot YAML per allele, running
`boltz predict ... --use_msa_server --accelerator cpu`, and harvesting
`msa_gen/boltz_results_<stem>/msa/<stem>_0.csv`.

**The model load is not the problem; the download is.** In boltz 2.2.1,
`download_boltz2(cache)` runs at `main.py:1141` and `process_inputs` (which
calls the MSA server) at `main.py:1162`, each gated only on `path.exists()`.
The harvest route pulls **all 6.2 GB** (`boltz2_conf.ckpt` 2.29 GB,
`boltz2_aff.ckpt` 2.06 GB, `mols.tar` 1.86 GB) *before* generating a single
MSA, then requires the process to be killed before the model loads.

`boltz.main.compute_msa` is the function that route eventually reaches. It is
importable and needs neither checkpoint nor CCD — only `run_mmseqs2` and
`boltz.data.const`. Calling it directly gives the same code path, so the output
format is correct by construction: no download, no YAML, no kill hack.

With one sequence in `data`, `compute_msa` takes the `len(data) > 1` false
branch (unpaired only) and writes `msa_dir/<name>.csv` — exactly the file the
plan wanted to harvest, without the `_0` suffix.

## The generation cap is 16384, not the CLI's 8192

Two different caps, easy to confuse:

- **At generation**, `compute_msa` truncates to `const.max_msa_seqs = 16384`.
  Real depth came in at 9.5k–10.5k rows, so the cap never bound.
- **At parse**, the batch run's `--max_msa_seqs` (CLI default **8192**) trims
  what the model actually sees, via `parse_csv(msa_path, max_seqs=...)`
  (`main.py:622`).

So the cached files hold full depth and the batch flag decides how much is
used. No regeneration is needed to change depth.

## Parse cost: measured, and it does not justify trimming

The plan flagged that Boltz re-parses the MSA once per complex — ~2,000 times
on the GPU worker — and prescribed capping `--max_msa_seqs`. The re-parse is
real: `processed/msa/<target_id>_<idx>.npz` is keyed by **target**, so there
is no reuse across complexes on the same allele.

Measured with boltz's own `parse_csv` on the deepest file in the cache
(`HLA-B*35:08`, 10,454 rows), Apple M-series CPU:

| `--max_msa_seqs` | Sequences parsed | Parse time | × 2,000 complexes |
|---:|---:|---:|---:|
| 256 | 256 | 0.019 s | 38 s |
| 1,024 | 1,024 | 0.035 s | 70 s |
| 4,096 | 4,096 | 0.095 s | 190 s |
| **8,192 (default)** | **8,192** | **0.176 s** | **5.9 min** |
| 16,384 | 10,025 (depth-limited) | 0.219 s | 7.3 min |

**At the default, the full panel costs ~5.9 minutes of parse across 2,000
complexes — about $0.25 of L40S worker time at $2.40/hour.** That is under 0.2%
of arm B's $240 ceiling. **Do not trim `--max_msa_seqs` for cost**; trim it
only if a depth-versus-accuracy benchmark at 4b.3 says to. This overturns the
plan's concern rather than confirming it.

Caveat: measured on Apple silicon, not a Modal worker. The loop is
single-threaded pure Python, so it tracks single-core speed. Re-measure at
4b.3 if it ever looks material — but two orders of magnitude below the fold
itself, the conclusion is not close.

## `--subsample_msa` is a broken flag, not just a documentation mismatch

The plan noted that its "CLI default and its help text disagree". It is worse
than that:

```python
@click.option("--subsample_msa", is_flag=True,
              help="Whether to subsample the MSA. Default is True.")
```

`is_flag=True` with no `default=` means click defaults it to **False**, while
the help text says True and the `predict()` signature default is also `True`.
So:

- **omit the flag** → no subsampling; the model sees up to `--max_msa_seqs` rows
- **pass the flag** → subsampled to `--num_subsampled_msa` (default 1,024)

These are very different compute profiles, and the help text points the wrong
way. **Pin it explicitly in the 4b.3 benchmark and in the batch command.**

## The four plan constraints, verified in source

| Constraint | Verified | Where |
|---|---|---|
| `msa: empty` on the peptide is mandatory | Confirmed — a missing key defaults to `0` (auto-generate); `"empty"` maps to `-1` (single-sequence) | `schema.py:1111,1127` |
| Only `.a3m` and `.csv` accepted | Confirmed — `.a3m.gz` fails because `Path.suffix` is `.gz` | `main.py:615-624` |
| Omit `--use_msa_server` on the batch run | Confirmed — any chain still needing an MSA raises `"Missing MSA's..."` | `main.py:581-582` |
| Trim the MSA | Mechanism confirmed (`parse_csv(..., max_seqs=...)`); **cost conclusion reversed** — see above | `main.py:622` |

Two further facts the batch YAML depends on:

- **`msa:` paths must be absolute.** Boltz resolves `Path(msa_id)` verbatim
  (`main.py:605-608`), so a relative path resolves against the worker's CWD.
  The plan's `/abs/path/...` is a requirement, not a style choice.
- **Boltz refuses to mix custom and auto-generated MSAs** in one input
  (`schema.py:1311`). This is *why* `msa: empty` is needed on the peptide
  rather than simply omitting the key.

## The C67S pseudosequence collision does not reach the structural arm

Stage 1 found that `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share one
contact pseudosequence (74 unique pseudosequences for 75 alleles), capping the
stage 2 pseudosequence arm.

Their **182-residue domains differ** — a single substitution at position 10
(`A` vs `S`), outside the 34 contact positions — so they received distinct MSAs
with distinct checksums. Boltz-2 takes the full domain, so the structural arm
sees two different inputs. The collision caps the pseudosequence arm only.

All three C67S constructs are cached: `HLA-B*14:01(C67S)` (10,112 rows),
`HLA-B*14:02(C67S)` (10,246), `HLA-B*39:06(C67S)` (9,990). Filenames sanitise
to `HLA-B_14_01_C67S` and so on.

## Environment

boltz **2.2.1** is a dependency of this step only and depends on torch, so it
is **not** installed into the project `.venv`. It lives in a throwaway
environment:

```bash
uv venv /tmp/boltzenv --python 3.12
uv pip install --python /tmp/boltzenv/bin/python boltz
/tmp/boltzenv/bin/python scripts/make_msas.py --all-alleles
```

`boltz_version` is recorded per row in the manifest. The MSA server was the
public `https://api.colabfold.com` with pairing strategy `greedy`; the script
sleeps 1 s between calls.

## What this unblocks, and what it does not

Arm B's MSA dependency is **cleared for every allele in the dataset**. Both
4b.2's crystal pilot and 4b.5's batch can run without further CPU prework,
under any panel the hour-5 decision picks.

Still open (not part of this step):

- **No YAML has been parsed by Boltz yet.** `load_canonicals` needs 21 CCD
  residue files from `mols.tar`, which this route deliberately never downloaded.
  The batch YAML shape is verified by source reading, not execution. **First
  item in the 4b.2 pilot: parse one YAML with a custom `msa:` path and
  `msa: empty` peptide.**
- `data/structural_panel.csv` (4b.4) is still unfrozen.
- Nothing here measures fold throughput or cost (4b.3).
