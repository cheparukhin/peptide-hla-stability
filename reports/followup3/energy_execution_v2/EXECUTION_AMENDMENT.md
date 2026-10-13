# Repaired-energy production-path execution amendment

Original app `ap-wIoBs2UraLYipTAxsIkEOl` failed at Volume.reload before new
scoring. Root confirmed its terminal provider state: zero tasks, zero started
markers and only the 32 previously uploaded technical pilot rows. Original
protocol, scientific source bytes, failed-run handoff and resource receipts
remain preserved. This namespace additionally saves original hash-bound source
snapshots and the original protocol without rewriting them.

Original protocol digest:
`ea9c5891c6f3fcda28e3f207951e78ac54b5152258469f5f90e0954a5a5a8361`.
Original protocol file SHA256:
`2124b7bb695d6c2143b9cdb78aca4f88e3ec2da60b4929707eea2ef0b849598f`.
Original controller SHA256:
`fe9d98608f139cab79c7e9935963c0d277e384a890ac1578d3868d84d092bc87`.

Modal's authoritative Volume guide still requires reload to observe another
container's committed changes, including v2. It also documents that reload fails
while files remain open on the mount. The original controller imported native
shared libraries from the mounted dependency directory, so reload encountered
that restriction. V2 supports `sync <mountpoint>` for persistence, but the guide
does not certify immediate cross-container visibility or fsync-alone replacement
of commit. Neither assumption is used in this amendment.

Authoritative reference, checked 13 October 2026:
[Modal Volumes guide](https://modal.com/docs/guide/volumes), sections “Volume
commits and reloads”, “Busy Volume errors” and “Volumes v2 overview”.

The new controller uses the existing base image
`im-Epc3H2RdNAgJSwtXBYMCfy` with numpy2.5.3, pandas3.0.6, scipy1.18.1 and
biotite1.7.1 installed in the image. Native packages must resolve outside the
mounted repository; the worker checks package paths. The mounted dependency
directory is removed from PYTHONPATH/sys.path. Frozen Python source and the
existing authorized FoldX distribution remain on the shared volume and are
hash-checked before use. Documented reload/commit operations remain, outside
native subprocess execution. The small cloud image build is covered by root's
fresh resource authorization and overhead allowance.

The scientific specification, all 4,449 identities, 29 finite energy terms,
32 technical pilot rows, exact 200 new MLP plus100 new ridge fits, baseline
reuse and three-primary family rule are unchanged. This is an operational
amendment after a pre-scoring failure, not a new experimental choice.

Before any fanout, run exactly one predetermined uncached panel case:
the first in original panel order excluding all 32 reused technical cases.
Its identity is frozen in the v2 protocol. It calls the ordinary production
Function synchronously with the same source/construct checks, hard CPU/memory
limits, attempt marker, reload/commit sequence, repaired scoring and deadlines.
Require status=ok and cached=false. Persist this feature through the ordinary
completion path, then exclude that identity from subsequent fanout so FoldX
never repeats it. Any failed gate stops fanout. The 32 pilot cases receive only
cached source-hash verification calls.

Root must provide a fresh source-bound $40 maximum admission, fixed UTC
deadline, independent guard and application handoff before launch. The image
build, production-path pilot and all remaining work share that single global
five-hour deadline and resource envelope; no deadline extension or blind retry.
The new app is `pepstab-followup3-repaired-energy-v2` and writes only under
`reports/followup3/energy_execution_v2/energy_scoring/<protocol_hash>/`.
Root's guard must bind this app description and its newly created app ID.

```
python scripts/followup3_energy_v2.py freeze
MODAL_PROFILE=a-cheparukhin modal run modal_app/followup3_energy_v2.py::run --protocol-path reports/followup3/energy_execution_v2/protocol.json --authorization-path <new-root-authorization.json>
```

Only root launches paid work. The first production-path success is engineering
evidence; it does not establish energy increment efficacy or full-panel runtime.
