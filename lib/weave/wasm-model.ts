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

/** Per-request resolver, stored id-keyed. `resolve` is widened to `unknown` so
 * infer/selfcheck share one map without an unsound union cast at the call site;
 * the message handler always feeds it a concrete InferResult/SelfCheck. */
type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };

/** Outbound request minus its correlation id (assigned per send in `request`). */
type RequestBody =
  | { type: "infer"; query: string; tools: string }
  | { type: "selfcheck" };

/** Owns one worker's lifecycle: spawn, request correlation, crash/reset, and
 * the one-time load (with retry-on-failure). One default instance backs the
 * module exports; tests construct their own with an injected `spawn`. */
export class WorkerClient {
  private worker: Worker | null = null;
  private clientPromise: Promise<NeedleClient> | null = null;
  private inferId = 0;
  private readonly pending = new Map<number, Pending>();
  private readonly progressListeners = new Set<(p: LoadProgress) => void>();
  private loadReject: ((e: Error) => void) | null = null;

  constructor(
    private readonly spawn: () => Worker = () =>
      new Worker(new URL("./needle-worker.ts", import.meta.url)),
  ) {}

  /** A crashed worker rejects everything in flight and is discarded so the
   * next command spawns a fresh one (and wasmNeedle falls back to the mock). */
  private crash(message: string) {
    const err = new Error(message);
    this.loadReject?.(err);
    this.loadReject = null;
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
    this.progressListeners.clear();
    this.worker?.terminate();
    this.worker = null;
  }

  private getWorker(): Worker {
    if (!this.worker) {
      const worker = this.spawn();
      worker.onerror = (e) => this.crash(e.message || "needle worker crashed");
      worker.onmessageerror = () => this.crash("needle worker message error");
      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data;
        if (msg.type === "progress") {
          for (const cb of this.progressListeners) cb(msg);
        } else if (msg.type === "result") {
          this.pending.get(msg.id)?.resolve({
            text: msg.text,
            backend: msg.backend,
            encodeMs: msg.encodeMs,
            decodeMs: msg.decodeMs,
          } satisfies InferResult);
          this.pending.delete(msg.id);
        } else if (msg.type === "selfcheck") {
          this.pending.get(msg.id)?.resolve({
            maxDiff: msg.maxDiff,
            meanDiff: msg.meanDiff,
            backend: msg.backend,
          } satisfies SelfCheck);
          this.pending.delete(msg.id);
        } else if (msg.type === "error" && msg.id !== undefined) {
          this.pending.get(msg.id)?.reject(new Error(msg.message));
          this.pending.delete(msg.id);
        }
      };
      this.worker = worker;
    }
    return this.worker;
  }

  /** Correlate one request/response round trip. `w` is captured at load time
   * (not re-read from `this.worker`) so a post-ready crash leaves this posting
   * to the terminated worker — matching the pre-refactor hang, not a throw. */
  private request<T>(w: Worker, body: RequestBody): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const id = ++this.inferId;
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      w.postMessage({ ...body, id });
    });
  }

  /** Idempotent: kicks off (or joins) the one-time bundle download + build. */
  load(onProgress?: (p: LoadProgress) => void): Promise<NeedleClient> {
    if (onProgress) this.progressListeners.add(onProgress);
    this.clientPromise ??= new Promise<NeedleClient>((resolve, reject) => {
      const w = this.getWorker();
      this.loadReject = reject;
      const onReady = (e: MessageEvent<WorkerResponse>) => {
        if (e.data.type === "ready") {
          w.removeEventListener("message", onReady);
          this.loadReject = null;
          // Progress only happens during this one-time load; dropping the
          // listeners here also stops per-command closures accumulating.
          this.progressListeners.clear();
          resolve({
            infer: (query, tools) =>
              this.request<InferResult>(w, { type: "infer", query, tools }),
            selfcheck: () => this.request<SelfCheck>(w, { type: "selfcheck" }),
          });
        } else if (e.data.type === "error" && e.data.id === undefined) {
          w.removeEventListener("message", onReady);
          this.loadReject = null;
          reject(new Error(e.data.message));
        }
      };
      w.addEventListener("message", onReady);
      w.postMessage({ type: "load" });
    });
    // Failed load clears the cache so the next command retries the fetch —
    // otherwise one network blip pins the whole session to the regex mock.
    this.clientPromise.catch(() => {
      this.clientPromise = null;
    });
    return this.clientPromise;
  }
}

/** The one default instance; module exports are thin delegates to it. */
const defaultClient = new WorkerClient();

/** Idempotent: kicks off (or joins) the one-time bundle download + build. */
export function loadNeedle(
  onProgress?: (p: LoadProgress) => void,
): Promise<NeedleClient> {
  return defaultClient.load(onProgress);
}

/** wasm → mock. The mock only answers if the bundle can't load at all.
 * Factored so tests can bind a UIModel to an injected-spawn client. */
export function makeWasmNeedle(client: WorkerClient): UIModel {
  return {
    async infer(query: string): Promise<ToolCall> {
      try {
        const c = await client.load();
        const r = await c.infer(query, TOOLS_JSON);
        return toToolCall(r.text, r.backend);
      } catch {
        return mockNeedle.infer(query);
      }
    },
  };
}

export const wasmNeedle: UIModel = makeWasmNeedle(defaultClient);
