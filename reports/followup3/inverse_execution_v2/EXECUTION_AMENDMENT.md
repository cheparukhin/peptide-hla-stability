# Inverse extraction worker-start amendment

The first source-bound inverse pilot in Sandbox
`sb-01M4TJ1RDW7JJZAG1BPFBQX9TN` persisted zero score rows after more than eight
minutes and was manually stopped. A direct fresh-process torch import later
completed in 3.95 seconds. This makes inherited native-thread state under the
default fork process pool a plausible explanation. It does not prove the cause,
establish a filesystem fault, or provide any scientific performance evidence.
The original stopped attempt and its resource receipts remain preserved.

The original frozen protocol file remains unchanged:

- File SHA256: `1953b6853882e06abf01c7b500832fe5243ca1cafb03a9a5c9e88f937fab9987`
- Scientific specification digest: `2a97d0d380c3672633e5ee75913e5841ce1eca29964a59f24a043f0ddd0c073c`
- Panel SHA256: `0e78d19b306269865fad388ff45dcb53605746ad91c6cb5e6c3bae1df9f5d8a5`
- Original CLI SHA256: `5a6b8cd0c0b5f1f7854ae457d7358876d074b57d29744ede669b3886e7a014aa`

Only worker startup and diagnostic telemetry change. One-worker extraction
runs directly in the fresh CLI interpreter, without a process pool. Parallel
extraction explicitly uses Python's spawn context, so each worker starts a new
interpreter instead of inheriting initialized numpy/pandas/native-thread state.
The five-row engineering pilot emits flushed progress at torch import,
thread configuration, native structure/model loading and scoring, and row
persistence. Every new score also records total worker elapsed time separately
from the existing scientific scoring timer. Extraction receipts record serial
versus spawn startup and summed worker time. This instrumentation does not
alter model inputs, checkpoint, decoding orders, batch size, precision, CPU
thread count, feature definitions, panel, fitting grid or assessment design.

Use the separate `inverse_execution_v2` output namespace. Preparation reproduced
the original 4,449-pair, 75-allele, 2,082-cluster panel and scientific
specification exactly, while binding the revised CLI bytes in its source map.
The old protocol is not rewritten to bless revised source code. All other
frozen scientific source and input digests must still agree.

```
python scripts/followup3_inverse.py prepare --out reports/followup3/inverse_execution_v2
python scripts/followup3_inverse.py pilot --out reports/followup3/inverse_execution_v2 --volume-root /structures --workers 1
```

The preparation above is local and lightweight. Root must bind a fresh cloud
plan/source snapshot, reserve its bounded cost/lifetime, and review actual pilot
output before admitting any main extraction. A successful serial pilot resolves
the immediate single-worker startup gate; it does not by itself measure
eight-worker spawn throughput or establish the hypothesized original cause.
The parallel resource forecast must therefore remain conservative or use a
separately bounded engineering check without changing the scientific scope.
