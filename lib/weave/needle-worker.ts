/**
 * Web Worker host for NeedleInference: bundle fetch + generate run here so
 * the ~seconds of matmuls never block the UI thread. Auto-detects WebGPU and
 * runs the encoder on the GPU when available, falling back to the TS encoder
 * otherwise. Protocol (see wasm-model.ts for the client side):
 *   in:  { type: "load" } | { type: "infer", id, query, tools }
 *        | { type: "selfcheck", id }
 *   out: { type: "progress", loaded, total } | { type: "ready", backend }
 *        | { type: "result", id, text, backend, encodeMs, decodeMs }
 *        | { type: "selfcheck", id, maxDiff, meanDiff, backend }
 *        | { type: "error", id?, message }
 */

import { GpuEncoder } from "./needle/gpu/encoder";
import { initGpu } from "./needle/gpu/device";
import { NeedleInference } from "./needle/infer";
import { NeedleModel } from "./needle/model";
import { NeedleTokenizer, type TokenizerData } from "./needle/tokenizer";
import { Weights, type Manifest } from "./needle/weights";

export type Backend = "webgpu" | "wasm";

export type WorkerRequest =
  | { type: "load" }
  | { type: "infer"; id: number; query: string; tools: string }
  | { type: "selfcheck"; id: number };

export type WorkerResponse =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; backend: Backend }
  | {
      type: "result";
      id: number;
      text: string;
      backend: Backend;
      encodeMs: number;
      decodeMs: number;
    }
  | {
      type: "selfcheck";
      id: number;
      maxDiff: number;
      meanDiff: number;
      backend: Backend;
    }
  | { type: "error"; id?: number; message: string };

const post = (msg: WorkerResponse) => self.postMessage(msg);

async function fetchWithProgress(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("Content-Length") ?? 0);
  if (!res.body || !total) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    post({ type: "progress", loaded, total });
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out.buffer;
}

interface Engine {
  inference: NeedleInference;
  model: NeedleModel;
  gpu: GpuEncoder | null;
  backend: Backend;
}

let enginePromise: Promise<Engine> | null = null;

function loadEngine(): Promise<Engine> {
  enginePromise ??= (async () => {
    const [manifest, tokData, weightBuf, device] = await Promise.all([
      fetch("/needle/manifest.json").then((r) => r.json() as Promise<Manifest>),
      fetch("/needle/tokenizer.json").then(
        (r) => r.json() as Promise<TokenizerData>,
      ),
      fetchWithProgress("/needle/weights.bin"),
      initGpu(),
    ]);
    const weights = new Weights(manifest, weightBuf);
    const model = new NeedleModel(weights);
    let gpu: GpuEncoder | null = null;
    if (device) {
      try {
        gpu = new GpuEncoder(device, weights);
        gpu.init();
      } catch {
        gpu = null; // any GPU setup failure -> TS fallback
      }
    }
    const engine: Engine = {
      inference: new NeedleInference(
        model,
        new NeedleTokenizer(tokData),
        manifest,
      ),
      model,
      gpu,
      backend: gpu ? "webgpu" : "wasm",
    };
    // Driver reset / context loss after successful init: demote to TS.
    if (device && gpu) {
      void device.lost.then((info) => {
        engine.gpu = null;
        engine.backend = "wasm";
        console.warn(
          "needle: GPU device lost, falling back to TS",
          info.message,
        );
      });
    }
    return engine;
  })();
  enginePromise.catch(() => {
    enginePromise = null;
  });
  return enginePromise;
}

/** Reject a hung GPU op (e.g. device lost mid-flight: mapAsync never
 * resolves, per spec) so the caller can fall back instead of waiting forever. */
function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<never>((_, rej) =>
      setTimeout(() => rej(new Error(`${what} timed out (${ms}ms)`)), ms),
    ),
  ]);
}

/** Encode + prepareCross for the active backend. A GPU failure (crash,
 * device lost, timeout) demotes the engine to the TS path for this and all
 * future requests — never to the mock. */
function backendFor(engine: Engine) {
  const demote = (err: unknown) => {
    engine.gpu = null;
    engine.backend = "wasm";
    console.warn("needle: GPU path failed, falling back to TS", err);
  };
  return {
    encode: async (tokens: number[]) => {
      if (engine.gpu) {
        try {
          return await withTimeout(
            engine.gpu.encode(tokens),
            10_000,
            "gpu encode",
          );
        } catch (err) {
          demote(err);
        }
      }
      return engine.model.encode(tokens);
    },
    prepareCross: async (out: Float32Array, T: number) => {
      if (engine.gpu) {
        try {
          return await withTimeout(
            Promise.resolve(engine.gpu.prepareCross(out, T)),
            10_000,
            "gpu prepareCross",
          );
        } catch (err) {
          demote(err);
          // GPU held the encoder output; recompute the TS pair from scratch.
          throw err instanceof Error ? err : new Error(String(err));
        }
      }
      return engine.model.prepareCross(out, T);
    },
  };
}

// GPU buffers (bufX, readback, …) are single-instance: overlapping requests
// would stomp each other and double-map the readback buffer. FIFO-serialize.
let queue: Promise<void> = Promise.resolve();

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  queue = queue.then(() => handle(e.data)).catch(() => {});
};

async function handle(msg: WorkerRequest): Promise<void> {
  try {
    if (msg.type === "load") {
      const engine = await loadEngine();
      post({ type: "ready", backend: engine.backend });
    } else if (msg.type === "infer") {
      const engine = await loadEngine();
      let out;
      try {
        out = await engine.inference.generateAsync(
          msg.query,
          msg.tools,
          backendFor(engine),
        );
      } catch (err) {
        // A mid-request GPU failure demoted the engine to TS — retry once
        // on the TS path rather than surfacing an error (which would drop
        // the client to the regex mock).
        if (engine.gpu !== null) throw err;
        out = await engine.inference.generateAsync(
          msg.query,
          msg.tools,
          backendFor(engine),
        );
      }
      const { text, encodeMs, decodeMs } = out;
      post({
        type: "result",
        id: msg.id,
        text,
        backend: engine.backend,
        encodeMs,
        decodeMs,
      });
    } else {
      // selfcheck: GPU encoder vs TS encoder on a fixed input (dev harness for
      // S1/S2 — proves the WGSL reproduces encode() within tolerance).
      const engine = await loadEngine();
      const tokens = engine.inference.encInput(
        "Remove practice guitar",
        '[{"name":"show_tasks"}]',
      );
      const ts = engine.model.encode(tokens);
      const gpu = engine.gpu
        ? await engine.gpu.encode(tokens)
        : { out: ts.out, T: ts.T };
      let maxDiff = 0;
      let sum = 0;
      for (let i = 0; i < ts.out.length; i++) {
        const dd = Math.abs(ts.out[i] - gpu.out[i]);
        if (dd > maxDiff) maxDiff = dd;
        sum += dd;
      }
      const meanDiff = sum / ts.out.length;
      post({
        type: "selfcheck",
        id: msg.id,
        maxDiff,
        meanDiff,
        backend: engine.backend,
      });
    }
  } catch (err) {
    post({
      type: "error",
      id: msg.type === "infer" || msg.type === "selfcheck" ? msg.id : undefined,
      message: String(err),
    });
  }
}
