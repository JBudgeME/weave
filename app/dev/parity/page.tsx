"use client";

/**
 * Dev-only browser-side parity eval (PRD success criteria): runs finetune eval +
 * regression splits through the in-browser engine (WebGPU encoder when active,
 * else TS) and reports routing / exact-args accuracy plus per-phase timings
 * (load / encode / decode, with the end-to-end median). Data comes from
 * /needle/{eval,regression}.jsonl (copied by scripts/export-weights.py).
 */

import { useState } from "react";

import { loadNeedle, toToolCall } from "@/lib/weave/wasm-model";

interface Row {
  query: string;
  tools: string;
  answers: string;
}

interface SplitResult {
  file: string;
  n: number;
  routing: number;
  full: number;
  misses: string[];
  backend: string;
  encodeMs: number[];
  decodeMs: number[];
}

const median = (xs: number[]) => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

async function evalSplit(
  file: string,
  onTick: (done: number, n: number) => void,
): Promise<SplitResult> {
  const text = await fetch(`/needle/${file}`).then((r) => {
    if (!r.ok) throw new Error(`${file}: ${r.status}`);
    return r.text();
  });
  const rows: Row[] = text
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Row);
  const engine = await loadNeedle();
  const res: SplitResult = {
    file,
    n: rows.length,
    routing: 0,
    full: 0,
    misses: [],
    backend: "",
    encodeMs: [],
    decodeMs: [],
  };
  for (let i = 0; i < rows.length; i++) {
    const want = (
      JSON.parse(rows[i].answers) as Array<{
        name: string;
        arguments: Record<string, unknown>;
      }>
    )[0];
    const wantArgs: Record<string, string> = {};
    for (const [k, v] of Object.entries(want.arguments))
      wantArgs[k] = String(v);
    const r = await engine.infer(rows[i].query, rows[i].tools);
    res.backend = r.backend;
    res.encodeMs.push(r.encodeMs);
    res.decodeMs.push(r.decodeMs);
    const got = toToolCall(r.text, r.backend);
    if (got.tool === want.name) {
      res.routing++;
      const keys = new Set([
        ...Object.keys(wantArgs),
        ...Object.keys(got.args),
      ]);
      if ([...keys].every((k) => wantArgs[k] === got.args[k])) res.full++;
      else res.misses.push(rows[i].query);
    } else {
      res.misses.push(rows[i].query);
    }
    onTick(i + 1, rows.length);
    // let the UI paint
    await new Promise((r) => setTimeout(r, 0));
  }
  return res;
}

// PRD acceptance bars (docs/prd/webgpu-encoder.md Q3/Q5) — asserted, not
// just displayed.
const TOLERANCE = 1e-2;
const TARGET_MS = 700;

export default function ParityPage() {
  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState<SplitResult[]>([]);
  const [check, setCheck] = useState("");
  const [verdict, setVerdict] = useState("");
  const [busy, setBusy] = useState(false);

  async function run() {
    if (busy) return;
    setBusy(true);
    setResults([]);
    setCheck("");
    setVerdict("");
    try {
      const t0 = performance.now();
      const engine = await loadNeedle();
      const loadMs = performance.now() - t0;
      const sc = await engine.selfcheck();
      setCheck(
        `backend: ${sc.backend} · load ${loadMs.toFixed(0)}ms · ` +
          `GPU-vs-TS encoder maxDiff ${sc.maxDiff.toExponential(2)} ` +
          `meanDiff ${sc.meanDiff.toExponential(2)}`,
      );
      const all: SplitResult[] = [];
      for (const file of ["eval.jsonl", "regression.jsonl"]) {
        const r = await evalSplit(file, (done, n) =>
          setStatus(`${file}: ${done}/${n}`),
        );
        all.push(r);
        setResults((prev) => [...prev, r]);
      }
      const enc = all.flatMap((r) => r.encodeMs);
      const dec = all.flatMap((r) => r.decodeMs);
      const e2e = all.flatMap((r) =>
        r.encodeMs.map((v, i) => v + r.decodeMs[i]),
      );
      setStatus(
        `done · median encode ${median(enc).toFixed(0)}ms · ` +
          `decode ${median(dec).toFixed(0)}ms · ` +
          `end-to-end ${median(e2e).toFixed(0)}ms`,
      );
      const accuracyOk = all.every((r) => r.routing === r.n && r.full === r.n);
      const diffOk = sc.maxDiff <= TOLERANCE;
      const speedOk = median(e2e) <= TARGET_MS;
      setVerdict(
        [
          accuracyOk ? "accuracy PASS" : "accuracy FAIL",
          `maxDiff ${diffOk ? "PASS" : "FAIL"} (≤${TOLERANCE})`,
          `median ${speedOk ? "PASS" : "FAIL"} (≤${TARGET_MS}ms)`,
        ].join(" · "),
      );
    } catch (e) {
      setStatus(`error: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-2xl px-4 py-10">
      <h1 className="text-2xl font-semibold tracking-tight">
        Browser parity eval
      </h1>
      <p className="text-muted-foreground mt-2 text-sm leading-relaxed">
        Re-runs the full held-out eval + regression sets (442 examples) through
        the in-browser engine on your hardware and checks routing / exact-args
        accuracy, GPU-vs-TS numeric drift, and latency against the
        project&apos;s acceptance bars. Takes a few minutes — longer on the
        TypeScript fallback, and background tabs run slower.
      </p>
      <button
        type="button"
        onClick={() => void run()}
        disabled={busy}
        className="mt-5 inline-flex cursor-pointer items-center rounded-full bg-violet-600 px-5 py-2 text-sm font-medium text-white transition-colors duration-200 hover:bg-violet-500 disabled:opacity-50"
      >
        {busy ? "running…" : "run the eval"}
      </button>

      <div className="border-border bg-card mt-6 space-y-3 rounded-2xl border p-5 font-mono text-sm">
        {check && (
          <p data-testid="selfcheck" className="text-muted-foreground">
            {check}
          </p>
        )}
        <p data-testid="status">{status}</p>
        {results.map((r) => (
          <p key={r.file} data-testid={`result-${r.file}`}>
            {r.file}: routing {r.routing}/{r.n} full {r.full}/{r.n}
            {r.misses.length > 0 &&
              ` — misses: ${r.misses.slice(0, 5).join("; ")}`}
          </p>
        ))}
        {verdict && (
          <p data-testid="verdict" className="border-border border-t pt-3">
            {verdict.split(" · ").map((part) => (
              <span
                key={part}
                className={`mr-2 inline-block rounded-full px-2.5 py-0.5 text-xs ${
                  part.includes("PASS")
                    ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"
                    : "bg-red-500/15 text-red-700 dark:text-red-300"
                }`}
              >
                {part}
              </span>
            ))}
          </p>
        )}
      </div>
    </main>
  );
}
