/**
 * Parity eval: run the pure-TS Needle inference over finetune/eval.jsonl and
 * finetune/regression.jsonl, report routing/full accuracy in the same terms as
 * scripts/eval-needle.py. Gate: 100% / 100% on both files.
 *
 *   bun scripts/parity-eval.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { NeedleModel } from "../lib/weave/needle/model";
import {
  NeedleTokenizer,
  type TokenizerData,
} from "../lib/weave/needle/tokenizer";
import { Weights, type Manifest } from "../lib/weave/needle/weights";
import { NeedleInference } from "../lib/weave/needle/infer";

const ROOT = join(import.meta.dir, "..");
const P = join(ROOT, "public", "needle");

function loadInference(): NeedleInference {
  const manifest = JSON.parse(
    readFileSync(join(P, "manifest.json"), "utf-8"),
  ) as Manifest;
  const buf = readFileSync(join(P, "weights.bin"));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const w = new Weights(manifest, ab);
  const tok = new NeedleTokenizer(
    JSON.parse(
      readFileSync(join(P, "tokenizer.json"), "utf-8"),
    ) as TokenizerData,
  );
  return new NeedleInference(new NeedleModel(w), tok, manifest);
}

interface Row {
  query: string;
  tools: string;
  answers: string;
}

function parseCall(text: string): {
  name: string | null;
  args: Record<string, string> | null;
} {
  try {
    const c = JSON.parse(text)[0];
    const args: Record<string, string> = {};
    for (const [k, v] of Object.entries(c.arguments ?? {})) args[k] = String(v);
    return { name: c.name, args };
  } catch {
    return { name: null, args: null };
  }
}

/** Order-insensitive dict equality (matches Python `args == want_args`). */
function argsEqual(
  a: Record<string, string>,
  b: Record<string, string>,
): boolean {
  const ak = Object.keys(a);
  const bk = Object.keys(b);
  if (ak.length !== bk.length) return false;
  for (const k of ak) if (a[k] !== b[k]) return false;
  return true;
}

function evalFile(
  inf: NeedleInference,
  path: string,
): {
  route: number;
  full: number;
  n: number;
  misses: string[];
  nonCompact: number;
} {
  const rows = readFileSync(path, "utf-8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as Row);
  let route = 0;
  let full = 0;
  let nonCompact = 0; // raw outputs that restore_tool_names would re-serialize
  const misses: string[] = [];
  for (const r of rows) {
    const out = inf.generate(r.query, r.tools);
    try {
      if (JSON.stringify(JSON.parse(out)) !== out) nonCompact++;
    } catch {
      nonCompact++;
    }
    const want = JSON.parse(r.answers)[0];
    const wantArgs: Record<string, string> = {};
    for (const [k, v] of Object.entries(want.arguments ?? {}))
      wantArgs[k] = String(v);
    const { name, args } = parseCall(out);
    if (name === want.name) {
      route++;
      if (args && argsEqual(args, wantArgs)) full++;
      else
        misses.push(
          `  MISS args ${JSON.stringify(r.query)}: want ${JSON.stringify(wantArgs)} got ${JSON.stringify(args)}`,
        );
    } else {
      misses.push(
        `  MISS route ${JSON.stringify(r.query)}: want ${want.name} got ${name} (raw: ${out})`,
      );
    }
  }
  return { route, full, n: rows.length, misses, nonCompact };
}

function main() {
  const inf = loadInference();
  let allPass = true;
  for (const [tag, file] of [
    ["eval", join(ROOT, "finetune", "eval.jsonl")],
    ["regression", join(ROOT, "finetune", "regression.jsonl")],
  ] as const) {
    const { route, full, n, misses, nonCompact } = evalFile(inf, file);
    const pct = (x: number) => `${x}/${n} (${((100 * x) / n).toFixed(1)}%)`;
    console.log(
      `${tag}: routing ${pct(route)}  full ${pct(full)}  [raw non-compact: ${nonCompact}]`,
    );
    for (const m of misses.slice(0, 10)) console.log(m);
    if (route !== n || full !== n) allPass = false;
  }
  process.exit(allPass ? 0 : 1);
}

main();
