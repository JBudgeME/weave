# Brief: Web Worker inference (prd:worker-inference, T1)

## SOP exception

Local brief instead of GH issue — same no-remote authorization as
`wasm-needle.md` (grilling Q7, 2026-07-13).

## Problem

In-browser inference blocks the main thread (~9s/query of matmuls): typing,
scrolling, and the progress line all freeze while the model thinks.

## Change

Host `NeedleInference` (bundle fetch + generate) in a dedicated Web Worker.
`lib/weave/wasm-model.ts` keeps its public shape (`wasmNeedle: UIModel`,
`loadNeedle(onProgress)`) but proxies via `postMessage`; progress events
stream from the worker. No inference-code changes — same engine, new thread.

Noted deviation: `loadNeedle` now resolves to a thin async `NeedleClient`
(`infer(query, tools)`) instead of the raw `NeedleInference` — the engine
lives in the worker and can't cross the boundary. Both callers updated.

## Slice checklist

- [x] Worker module + postMessage protocol (load-with-progress, infer)
- [x] wasm-model.ts proxies to the worker; mock fallback unchanged
- [x] Verify: UI stays interactive (typing works mid-inference), `· wasm`
      answer still correct in browser; lint/tests green; static export builds

## Success criteria (observable)

While a command is inferring, the input accepts keystrokes and the busy
button animates; result identical to pre-worker behavior (same tool call,
same source tag). `/dev/parity` page still works (it may stay main-thread).

## Out of scope

- Speeding up inference itself (WebGPU/SIMD) — this only unblocks the UI
- Parallel inference / request queueing beyond the existing `busy` gate
