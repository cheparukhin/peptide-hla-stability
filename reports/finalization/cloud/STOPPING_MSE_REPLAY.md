# Report-only stopping-MSE replay amendment

The first hybrid report stopped before main-model auditing or statistics.
During pilot replay, the leaf MLP's recomputed float32 stopping MSE was
0.8145773410797119, versus the saved curve value 0.8145772814750671.
The difference is exactly one float32 ULP: 5.960464477539063e-8. The stored
best-MSE field still matched the selected saved curve value.

The fit and replay both evaluate float32 predictions, squared residuals and
mean. CPU/BLAS dispatch or memory layout can plausibly affect rounding, but
the precise historical cause is unproven: stopping prediction vectors were
not saved. Matching Python and package versions does not establish identical
floating-point execution on different containers.

Independent scientific review approved a narrowly bounded implementation
amendment. A source-bound report adapter copies the frozen audit function and
changes exactly one assertion. Recomputed stopping MSE may differ by at most
one float32 ULP. The stored best MSE must equal the stored selected curve value
exactly. Every other assertion still executes, including exact held-prediction
equality, feature/weight/transform hashes, cohort and partition ancestry,
curve/epoch/patience checks, tree reuse, and downstream selection/statistics.
No global NumPy assertion is modified. Fitting sources and saved artifacts are
unchanged. The cloud entry activates the adapter only for the hybrid report.

The next report must complete pilot replay before main auditing. On full
success it records every scalar comparison (328 MLP checks across 82 bundles)
and the maximum ULP difference in audit.json, with a disclosure in REPORT.md.
Any difference above one ULP or failed held-prediction equality stops the run;
there is no automatic tolerance widening or refitting. The existing failure
alone does not establish that the leaf model's held predictions match exactly.

The failed attempt is closed, with no new fits or statistics, a provider
snapshot of $0.06052292 and conservative carry of $0.10. Its authentic error
and receipts are preserved. The separately authorized report recovery keeps
2 CPU / 8 GiB, a $5 cap, and the original work/provider deadlines of 13 October
06:45/06:47 UTC. Source-bound metadata is in
[stopping_mse_replay_amendment.json](stopping_mse_replay_amendment.json).
The repair passed 69 lightweight checks, including scalar boundary tests and
unchanged scientific-worker/report source checks; no local model fits were run.
