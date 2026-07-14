# PRD: WebGPU encoder backend (prd:webgpu-encoder, T2)

## SOP exception

Local PRD instead of GH issue — standing no-remote authorization
(wasm-needle grilling Q7, 2026-07-13).

## Goal

Cut in-browser command latency from ~2.8s to ≤700ms median by running the
Needle **encoder** forward pass in WebGPU compute shaders. The decoder
(0.33s TS, KV-cached) and all sequencing stay in TypeScript.

## Decisions (from grilling)

| #   | Decision                                                                                                                                                                                                                                                                                                   |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | **WebGPU** (not WASM SIMD) — the shipped TS path is already the universal fallback                                                                                                                                                                                                                         |
| Q2  | **Encoder only** on GPU; TS decoder unchanged. _Implementation note: `prepareCross` (one-shot encoder-output projection, ~470ms in TS — the real bottleneck) runs on GPU too; the "TS decoder" that stays untouched is the KV-cached autoregressive loop. Without this, the ≤700ms target is unreachable._ |
| Q3  | Acceptance = 100%/100% on `/dev/parity` **with WebGPU active** (268 examples) + encoder-activation fixture within ~1e-2                                                                                                                                                                                    |
| Q4  | Auto-detect in the worker (`navigator.gpu` + adapter); source tag `· webgpu` when GPU answered, `· wasm` on TS fallback                                                                                                                                                                                    |
| Q5  | **Median command ≤700ms** in Chrome on the dev machine (RTX 4070 Ti Super); per-phase timings surfaced on `/dev/parity`                                                                                                                                                                                    |

## Modules expected to change

- `lib/weave/needle/` — new `gpu/` (or similar): device init, weight buffer
  upload, WGSL shaders (matmul, ZCRMSNorm, RoPE, GQA attention, gated
  residuals), encoder dispatch returning the same `encode()` output shape
- `lib/weave/needle-worker.ts` — backend detection, source tag plumbing
- `lib/weave/wasm-model.ts` — pass the backend tag through to `ToolCall.source`
- `app/dev/parity/page.tsx` — per-phase timing display
- README — backend story

## Success criteria (observable)

1. `/dev/parity` in Chrome with WebGPU: 100% routing / 100% args, both splits.
2. Median end-to-end command ≤700ms in Chrome on the dev machine (timing
   visible on the parity page).
3. In a browser without WebGPU (or adapter failure), commands still work via
   the TS path, tagged `· wasm`.
4. `bun test`, lint, `tsc`, static export build all green; Bun-side
   parity eval still 100%/100% (TS path untouched).

## Out of scope

- Decoder on GPU (future increment)
- f16 shaders (f32 compute first)
- WASM SIMD middle tier
- UI changes beyond the source tag

## Slices

- [ ] S1: GPU scaffold — device/adapter init in the worker, weight upload,
      WGSL matmul; fixture test: one encoder layer's output matches TS
      within tolerance (dev harness, not bun-testable — verified via
      parity page / browser harness)
- [ ] S2: Full encoder on GPU — all 12 layers + norms + RoPE + GQA;
      encoder-activation fixture within 1e-2 in browser; wire auto-detect +
      `· webgpu` tag + TS fallback
- [ ] S3: Verify & polish — parity page timings, 268-example GPU parity run,
      ≤700ms median confirmed, README, Bun-side suite green
