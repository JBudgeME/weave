/**
 * In-browser Needle behind a Web Worker: the fp16 bundle download and the
 * seconds-long generate() run off the main thread (lib/weave/needle-worker.ts)
 * so the UI never freezes. Falls back to the regex mock only if the bundle
 * fails to load. See docs/prd/wasm-needle.md + docs/prd/worker-inference.md.
 */

import { mockNeedle, type ToolCall, type UIModel } from "./model";
import type { Backend, WorkerResponse } from "./needle-worker";
import { TOOLS } from "./tools";

const TOOLS_JSON = JSON.stringify(TOOLS);

export type LoadProgress = { loaded: number; total: number };

export interface InferResult {
  text: string;
  backend: Backend;
  encodeMs: number;
  decodeMs: number;
}

export interface SelfCheck {
  maxDiff: number;
  meanDiff: number;
  backend: Backend;
}

/** Parse the model's generated text into a ToolCall (show_help on no/bad call).
 * `source` is the backend that produced the text (`webgpu`/`wasm`). */
export function toToolCall(text: string, source: Backend = "wasm"): ToolCall {
  try {
    const calls = JSON.parse(text) as Array<{
      name: string;
      arguments?: Record<string, unknown>;
    }>;
    if (calls.length === 0) return { tool: "show_help", args: {}, source };
    const args: Record<string, string> = {};
    for (const [k, v] of Object.entries(calls[0].arguments ?? {})) {
      args[k] = String(v);
    }
    return { tool: calls[0].name, args, source };
  } catch {
    return { tool: "show_help", args: {}, source };
  }
}

export interface NeedleClient {
  infer(query: string, tools: string): Promise<InferResult>;
  selfcheck(): Promise<SelfCheck>;
}

let worker: Worker | null = null;
let clientPromise: Promise<NeedleClient> | null = null;
let inferId = 0;
const pending = new Map<
  number,
  {
    resolve: (v: InferResult | SelfCheck) => void;
    reject: (e: Error) => void;
  }
>();
const progressListeners = new Set<(p: LoadProgress) => void>();
let loadReject: ((e: Error) => void) | null = null;

/** A crashed worker rejects everything in flight and is discarded so the
 * next command spawns a fresh one (and wasmNeedle falls back to the mock). */
function crash(message: string) {
  const err = new Error(message);
  loadReject?.(err);
  loadReject = null;
  for (const p of pending.values()) p.reject(err);
  pending.clear();
  progressListeners.clear();
  worker?.terminate();
  worker = null;
}

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./needle-worker.ts", import.meta.url));
    worker.onerror = (e) => crash(e.message || "needle worker crashed");
    worker.onmessageerror = () => crash("needle worker message error");
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === "progress") {
        for (const cb of progressListeners) cb(msg);
      } else if (msg.type === "result") {
        pending.get(msg.id)?.resolve({
          text: msg.text,
          backend: msg.backend,
          encodeMs: msg.encodeMs,
          decodeMs: msg.decodeMs,
        });
        pending.delete(msg.id);
      } else if (msg.type === "selfcheck") {
        pending.get(msg.id)?.resolve({
          maxDiff: msg.maxDiff,
          meanDiff: msg.meanDiff,
          backend: msg.backend,
        });
        pending.delete(msg.id);
      } else if (msg.type === "error" && msg.id !== undefined) {
        pending.get(msg.id)?.reject(new Error(msg.message));
        pending.delete(msg.id);
      }
    };
  }
  return worker;
}

/** Idempotent: kicks off (or joins) the one-time bundle download + build. */
export function loadNeedle(
  onProgress?: (p: LoadProgress) => void,
): Promise<NeedleClient> {
  if (onProgress) progressListeners.add(onProgress);
  clientPromise ??= new Promise<NeedleClient>((resolve, reject) => {
    const w = getWorker();
    loadReject = reject;
    const onReady = (e: MessageEvent<WorkerResponse>) => {
      if (e.data.type === "ready") {
        w.removeEventListener("message", onReady);
        loadReject = null;
        // Progress only happens during this one-time load; dropping the
        // listeners here also stops per-command closures accumulating.
        progressListeners.clear();
        resolve({
          infer(query, tools) {
            return new Promise<InferResult>((res, rej) => {
              const id = ++inferId;
              pending.set(id, {
                resolve: (v) => res(v as InferResult),
                reject: rej,
              });
              w.postMessage({ type: "infer", id, query, tools });
            });
          },
          selfcheck() {
            return new Promise<SelfCheck>((res, rej) => {
              const id = ++inferId;
              pending.set(id, {
                resolve: (v) => res(v as SelfCheck),
                reject: rej,
              });
              w.postMessage({ type: "selfcheck", id });
            });
          },
        });
      } else if (e.data.type === "error" && e.data.id === undefined) {
        w.removeEventListener("message", onReady);
        loadReject = null;
        reject(new Error(e.data.message));
      }
    };
    w.addEventListener("message", onReady);
    w.postMessage({ type: "load" });
  });
  // Failed load clears the cache so the next command retries the fetch —
  // otherwise one network blip pins the whole session to the regex mock.
  clientPromise.catch(() => {
    clientPromise = null;
  });
  return clientPromise;
}

/** wasm → mock. The mock only answers if the bundle can't load at all. */
export const wasmNeedle: UIModel = {
  async infer(query: string): Promise<ToolCall> {
    try {
      const client = await loadNeedle();
      const r = await client.infer(query, TOOLS_JSON);
      return toToolCall(r.text, r.backend);
    } catch {
      return mockNeedle.infer(query);
    }
  },
};
