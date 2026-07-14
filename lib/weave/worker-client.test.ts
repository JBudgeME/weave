import { describe, expect, test } from "bun:test";

import { makeWasmNeedle, WorkerClient } from "./wasm-model";
import type { WorkerRequest, WorkerResponse } from "./needle-worker";

/**
 * Scripted stand-in for a Web Worker. Records posted messages, supports both
 * the `onmessage` property and `addEventListener("message", …)` (the real
 * worker fires both, and WorkerClient uses each for a different purpose), and
 * exposes `emit`/`emitError` so a test drives the response protocol by hand.
 */
class FakeWorker {
  onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((e: { message: string }) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  readonly posted: WorkerRequest[] = [];
  terminated = false;
  private readonly listeners = new Set<
    (e: MessageEvent<WorkerResponse>) => void
  >();

  postMessage(msg: WorkerRequest) {
    this.posted.push(msg);
  }
  addEventListener(
    _type: "message",
    fn: (e: MessageEvent<WorkerResponse>) => void,
  ) {
    this.listeners.add(fn);
  }
  removeEventListener(
    _type: "message",
    fn: (e: MessageEvent<WorkerResponse>) => void,
  ) {
    this.listeners.delete(fn);
  }
  terminate() {
    this.terminated = true;
  }

  /** Deliver a worker→client message to both delivery mechanisms. */
  emit(data: WorkerResponse) {
    const e = { data } as MessageEvent<WorkerResponse>;
    this.onmessage?.(e);
    for (const fn of [...this.listeners]) fn(e);
  }
  emitError(message: string) {
    this.onerror?.({ message });
  }
  /** id of the Nth (0-based) infer/selfcheck request this worker received. */
  reqId(n: number): number {
    const req = this.posted.filter(
      (m) => m.type === "infer" || m.type === "selfcheck",
    )[n] as { id: number };
    return req.id;
  }
}

const asWorker = (f: FakeWorker) => f as unknown as Worker;

/** Flush pending microtasks/macrotasks between manual protocol steps. */
const tick = () => new Promise((r) => setTimeout(r, 0));

/** Drive a WorkerClient to a resolved NeedleClient over a fresh FakeWorker. */
async function loaded() {
  const fake = new FakeWorker();
  const client = new WorkerClient(() => asWorker(fake));
  const ready = client.load();
  fake.emit({ type: "ready", backend: "wasm" });
  return { fake, client, needle: await ready };
}

const inferResult = (id: number, text: string): WorkerResponse => ({
  type: "result",
  id,
  text,
  backend: "wasm",
  encodeMs: 1,
  decodeMs: 2,
});

describe("WorkerClient", () => {
  // Scenario 1 — correlation: two overlapping infers resolve to their own callers.
  test("correlates overlapping requests to the right callers", async () => {
    const { fake, needle } = await loaded();
    const a = needle.infer("first", "[]");
    const b = needle.infer("second", "[]");
    const idA = fake.reqId(0);
    const idB = fake.reqId(1);
    // Reply out of order to prove it's keyed by id, not arrival order.
    fake.emit(inferResult(idB, "B"));
    fake.emit(inferResult(idA, "A"));
    expect((await a).text).toBe("A");
    expect((await b).text).toBe("B");
  });

  // Scenario 2 — crash mid-flight: all pending reject; wasmNeedle → mock.
  test("crash rejects in-flight requests and wasmNeedle falls back to mock", async () => {
    const fake = new FakeWorker();
    const client = new WorkerClient(() => asWorker(fake));
    const wasmNeedle = makeWasmNeedle(client);

    const ready = client.load();
    fake.emit({ type: "ready", backend: "wasm" });
    await ready;

    const inflight = wasmNeedle.infer("remove practice guitar"); // posts an infer, awaits
    await tick(); // let the infer promise register in `pending`
    fake.emitError("boom");

    // The crashed request rejects internally; makeWasmNeedle catches it and
    // returns the regex mock's answer instead of throwing.
    const call = await inflight;
    expect(call.source).toBe("mock");
    expect(fake.terminated).toBe(true);
  });

  // Scenario 3 — retry after a failed load: next call re-attempts (not mock-pinned).
  test("re-attempts the load after a load failure", async () => {
    let spawns = 0;
    const fakes: FakeWorker[] = [];
    const client = new WorkerClient(() => {
      spawns++;
      const f = new FakeWorker();
      fakes.push(f);
      return asWorker(f);
    });

    const first = client.load();
    fakes[0].emit({ type: "error", message: "load failed" }); // untargeted → rejects load
    await expect(first).rejects.toThrow("load failed");
    await tick(); // let load's .catch null clientPromise

    // A failed load clears clientPromise (not the worker), so the next call
    // re-attempts the load on the same worker — never pinned to the mock.
    const second = client.load();
    fakes[0].emit({ type: "ready", backend: "wasm" });
    await expect(second).resolves.toBeDefined();
    expect(fakes[0].posted.filter((m) => m.type === "load").length).toBe(2);
    expect(spawns).toBe(1); // reused, not respawned — respawn needs a crash
  });

  // Scenario 4 — progress lifecycle: listeners fire during load, dropped after ready.
  test("fires progress listeners during load and drops them after ready", async () => {
    const fake = new FakeWorker();
    const client = new WorkerClient(() => asWorker(fake));
    const seen: number[] = [];
    const ready = client.load((p) => seen.push(p.loaded));

    fake.emit({ type: "progress", loaded: 10, total: 100 });
    fake.emit({ type: "progress", loaded: 50, total: 100 });
    fake.emit({ type: "ready", backend: "wasm" });
    await ready;

    // Post-ready progress must not reach the (now-cleared) listener.
    fake.emit({ type: "progress", loaded: 90, total: 100 });
    expect(seen).toEqual([10, 50]);
  });

  // Scenario 5 — respawn: a during-load crash lets the next load spawn afresh.
  test("spawns a fresh worker after a during-load crash", async () => {
    let spawns = 0;
    const fakes: FakeWorker[] = [];
    const client = new WorkerClient(() => {
      spawns++;
      const f = new FakeWorker();
      fakes.push(f);
      return asWorker(f);
    });

    const first = client.load();
    fakes[0].emitError("crashed before ready");
    await expect(first).rejects.toThrow("crashed before ready");
    expect(fakes[0].terminated).toBe(true);
    await tick(); // let load's .catch null clientPromise

    const second = client.load();
    fakes[1].emit({ type: "ready", backend: "wasm" });
    await expect(second).resolves.toBeDefined();
    expect(spawns).toBe(2);
  });
});
