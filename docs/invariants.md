# System invariants

Load-bearing properties that callers depend on. Changing a module listed here
requires re-verifying its invariant. Referenced by the SOP router (`CLAUDE.md`):
tier T3 fires on any change to a module named below.

## Encoder adapter parity

The two `Encoder` adapters — the synchronous TS `NeedleModel` and the WebGPU
`GpuEncoder` — must produce numerically equivalent output for the same input:
`encode()` outputs and the `prepareCross()` cross-attention K/V agree within a
GPU-vs-TS max absolute difference of ~5e-6. This is what lets the worker demote
from GPU to the TS path mid-flight without changing generated text.

Verified by: the Parity eval (`bun scripts/parity-eval.ts`), the worker
`selfcheck` message (GPU vs TS encoder on a fixed input), and the dev parity
page.

**Modules:** lib/weave/needle/model.ts, lib/weave/needle/gpu/encoder.ts, lib/weave/needle/encoder.ts
