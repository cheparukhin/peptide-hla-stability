# Frozen FoldX technical pilot

This pilot checks whether the installed FoldX build produces interpretable,
finite paired interface energies after repair across diverse original-training
structures, and measures current runtime. It reads no assay outcomes and
performs no half-life fitting, correlation, ranking or model selection.

Select one pair per allele from the frozen inverse-folding panel by the minimum
SHA256 of `["followup3-foldx-technical-20261013", allele, peptide]`, using compact
sorted-key JSON. Order alleles by the SHA256 of `[seed, allele]` and retain the
first 32 distinct alleles. Original train membership is verified against the
unchanged splits using `(allele, peptide)`. No structures are newly predicted.
Freeze this panel and implementation before reading any new energy.

Use the installed Linux FoldX binary with SHA256
`faed5e54e47744ab6ab75f8f9ad1a1f2e2cdbeac4f0f98d4e36c97049eb14268`.
User authorization and existing account/workspace context permit this bounded
research pilot. Executable presence and task authorization do not verify a
license. An executable license/access rejection stops further command scheduling;
never change account, bypass restrictions, patch the binary or purchase access.
Do not commit or redistribute the executable, support distribution or keys.

Reuse the established verified mmCIF-to-PDB conversion and header-driven parser.
Each case is the saved three-chain Boltz production construct, arm B, seed 0.
Verify input allele/native peptide/training membership/declared chains, preserve
beta2m, and analyse exactly the verified peptide chain versus HLA chain. First
run AnalyseComplex on the unmodified converted PDB. Then RepairPDB with its
default ALL scope, followed by AnalyseComplex on the repaired PDB. Do not use
interface-only repair. Verify repaired sequences/chain identities, finite
coordinates and expected interaction-group selection. Compare matched heavy
atom coordinates and backbone displacements; retain atom-count changes and
the complete parsed numeric energy/clash/interface terms. Repair-induced
ranking or plausible energy ranges do not establish half-life predictive value.

The first predetermined paired case runs serially to gate executable access,
conversion and parsing. Only a successful first case starts the remaining
bounded fanout. Run at most eight concurrent single-threaded FoldX cases.
Every whole raw-plus-repaired case has a cumulative 1,200-second deadline,
passed as remaining time to each subprocess. Root supplies an additional
absolute cloud lifetime/cost cap. No retries or failed-case replacements.
Any technical failure is retained. Any access failure stops new commands;
already running commands remain under their original per-case deadlines.

Preserve one per-case JSON with natural keys, original structure/metadata/PDB
hashes, binary identity, validated conversion/native chains, phase timings,
repaired-coordinate checks, parsed raw/repaired terms and term differences.
Never retain executable stdout/stderr or license-bearing parser excerpts.
Delete temporary working PDBs after deriving the technical record. Aggregate
paired terms and exact case-artifact hashes. Primary technical completion is
all 32 cases passing; retain attempted/unattempted counts if the gate fails.
No outcome-based expansion follows from this pilot.

Historical timing is a forecast input only: the old repaired five-case samples
took 235–257 seconds per process, roughly 69–71 times raw scoring, and did not
measure repaired contention across diverse alleles. At 32 cases and eight
workers this suggests about 17 minutes before startup; a full worst-case
four-wave timeout is 80 minutes, plus the serial first-case gate up to 20 minutes.
Root must reserve the full provider deadline before launch. Report current
measured throughput separately from this historical forecast. Any later energy
increment study requires a separate frozen scientifically adequate design and
budget gate; this technical pilot does not authorize full production.
