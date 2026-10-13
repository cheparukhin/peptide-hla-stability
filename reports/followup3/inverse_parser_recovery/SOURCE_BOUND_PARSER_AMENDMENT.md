# Exact native-peptide chain selection recovery

The stopped original extraction checkpoint contains 4,164 saved records:
4,163 successful and one failure, HLA-B*56:01/YPAEITLTW. No supervised inverse
head has been fitted or used to choose this recovery.

The original loader calls the generic first-containing-chain substring finder,
then rejects a hit whose chain length differs from the native peptide length.
The actual failed mmCIF and metadata show HLA chain A has275 residues and
contains YPAEITLTW at zero-based offset208, whereas chain C is the unique
entire nine-residue native peptide. Beta2m chain B has99 residues. Every
coordinate is finite, all383 residues have N/CA/C/O, and all CA-derived chain
sequences match metadata. FAILED_STRUCTURE_INSPECTION.json binds exact source
hashes and this label-free inspection. The old finder chooses A,208 and the
length guard raises before any ProteinMPNN forward pass. The remaining panel
also includes HLA-B*57:02/YPAEITLTW with this native-HLA sequence coincidence.

The repair selects exactly one chain whose ENTIRE CA-derived sequence equals
the native peptide, returning offset0. Zero or multiple exact whole-chain
matches remain errors. A new process-local helper provider returns this finder
to the unchanged original loader. The shared pose-check substring function is
not modified. Every original metadata/native-chain/backbone-finiteness guard,
checkpoint, coordinate, 16 decoding orders, numerical setting and scoring body
remains. The spawnable worker explicitly installs the selector in each child;
it never relies on parent monkeypatch inheritance.

Every old successful record passed the old whole-chain-length/offset guard.
Thus the selected chain already equalled the complete native peptide, rather
than an HLA substring, and its validated tensor/coordinates are unchanged by
the new selector. Production metadata declares only one nine-residue peptide
chain. All 4,163 successful JSON records are reused byte-identically and their
hashes are checked before and after continuation; none is recomputed. This is
a parser recovery, not an outcome-based repeat or tolerance change.

Preserve the failed JSON under original_failed_attempt.json through an atomic
same-volume move, only when its exact original hash matches the frozen
manifest. Never overwrite or silently delete a failure. The demonstrated
failure hash is
`1972c68c52a91a06d846df8910fd0ebdf240fc1320fb0959efa01a704b56bb79`.
The repaired attempt is explicitly recorded. A problem-case parse-only check
against the downloaded source hashes runs before archival and any new scoring.

Root may admit one ≤1-hour, 16-CPU/32-GiB continuation with eight spawned
workers and two numerical threads each, under its new$4.50 maximum reservation.
There are exactly286 corrected/missing cases in the unchanged4,449 panel.
The new wrapper calls the original extraction/aggregate acceptance functions
through a documented replacement task mapper. Original frozen code/protocol
bytes, active FoldX code and energy source bindings remain untouched. No cohort
expansion, complete-case assessment or scientific head change is authorized.

The final inverse_execution_v2/parser_recovery_receipt.json binds wrapper/source
digests, parent scientific protocol, inherited successful row hashes, original
failure hash, corrected/missing identities, complete row hashes and aggregate
score/receipt hashes. Independent assessment auditing must include this receipt
to certify the actual execution ancestry beyond the original source manifest.

```
python scripts/followup3_inverse_parser_recovery.py prepare
python scripts/followup3_inverse_parser_recovery.py extract --volume-root /structures --workers 8
```

Preparation is lightweight; only root launches the cloud continuation after
source review, exact cached-record validation, fresh budget and hard deadline.
