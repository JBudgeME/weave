import type { CrossKV } from "./model";

/**
 * The encoder seam: the structural `{ encode, prepareCross }` shape shared by
 * the synchronous TS NeedleModel (./model.ts) and the async WebGPU GpuEncoder
 * (./gpu/encoder.ts). Either satisfies this interface, so NeedleInference.
 * generateAsync (./infer.ts) and the worker's backend selector (../needle-
 * worker.ts) can drive whichever backend is live. Return types are sync|async
 * unions because the TS model returns synchronously and the GPU adapter returns
 * Promises — `await` on the consumer side handles both.
 *
 * Invariant "Encoder adapter parity" (docs/invariants.md): both adapters must
 * produce numerically equivalent output (GPU-vs-TS max abs diff ~5e-6).
 */
export interface Encoder {
  encode(
    tokens: number[],
  ):
    | { out: Float32Array; T: number }
    | Promise<{ out: Float32Array; T: number }>;
  prepareCross(out: Float32Array, T: number): CrossKV[] | Promise<CrossKV[]>;
}
