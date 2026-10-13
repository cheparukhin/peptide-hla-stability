# Match replay to the recorded numerical backend

The 23:46 report attempt failed on exact held-prediction replay for the R0
pilot leaf MLP, before main statistics: 3,886 of 5,258 values differed, with
maximum absolute difference 0.0000019073486. Its input/model/transform hashes
passed. The failed container did not record its BLAS architecture, so the cause
is not established. The failed attempt, actual exit and original arrays remain
preserved; no result has been declared audited.

All twelve authentic fitting runtime files record the same two-thread SkylakeX
OpenBLAS kernels and library versions. One new report attempt will reproduce
that recorded backend, checking exposed CPU capabilities before forcing the
kernel and checking actual library versions, architecture and threads after
import. This follows OpenBLAS's documented [kernel selection implementation](https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/develop/driver/others/dynamic.c).
It is a provenance-based hypothesis test, not a search among kernels.

The existing unscored pilot replay must pass before main audit. Held predictions
still require exact equality; the stopping-MSE allowance remains at most one
float32 ULP. No fitting, saved prediction, selection, statistic or tolerance
changes. Actual backend identities are recorded in stdout and the successful
audit. If this matched-backend attempt fails, stop repeated report attempts and
preserve the evidence instead of automatically widening tolerances.

Independent Astra scientific review approved this bounded approach. Thirty-one
lightweight checks passed, including fresh-process pre-import ordering; all twelve runtime bindings were verified. The
original 13 October 06:45 UTC work stop, 06:47 UTC provider stop and $5 cap remain.
The closed failed attempt metered $0.02883595 and carries $0.05 conservatively.
