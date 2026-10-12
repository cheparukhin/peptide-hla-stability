# Correct ensemble arithmetic during saved-model replay

The second report stopped on 12 October at 23:39 UTC, before any statistics.
The real fitting worker validates each member with `evaluation.validate_finite`,
which converts predictions to float64, then averages. The report omitted that
conversion and averaged float32 MLP outputs. This was a reporting bug, not a
change in any stored model or scientific prediction.

The cloud report now applies the same conversion before its existing mean.
All saved member predictions, aggregate CSVs, original scientific sources,
selection formulas and comparison tolerances remain unchanged. Twelve
lightweight saved-vector checks reproduced stored aggregates exactly with the
worker arithmetic; eight MLP aggregates differed with the old report arithmetic.
The regression test uses float32 synthetic members through the actual worker
and report paths, and still rejects an altered aggregate after its hash is
re-signed. The regression test plus existing scalar-replay tests passed 25 checks.
No real inference, model fitting or bootstrap ran locally.

The previously disclosed one-ULP recomputed stopping-MSE rule is unchanged.
Another report attempt has the same $5 cap and original 13 October 06:45 UTC
work stop / 06:47 UTC provider stop. The failed attempt returned 1, cleaned its
child, and was closed with its evidence intact; metered compute was $0.12941363
and $0.20 is carried conservatively. There are no scored hybrid results yet.
