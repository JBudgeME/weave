/**
 * Model-level parity tests. Require public/needle/weights.bin (52MB, gitignored;
 * regenerate with `python scripts/export-weights.py <ckpt.pkl>`), so they skip
 * when weights are absent. The authoritative gate is `bun scripts/parity-eval.ts`.
 */
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { NeedleModel } from "./model";
import { NeedleTokenizer, type TokenizerData } from "./tokenizer";
import { Weights, type Manifest } from "./weights";
import { NeedleInference } from "./infer";

const ROOT = join(import.meta.dir, "..", "..", "..");
const P = join(ROOT, "public", "needle");
const FIX = join(import.meta.dir, "__fixtures__");
const hasWeights = existsSync(join(P, "weights.bin"));
const load = <T>(p: string): T => JSON.parse(readFileSync(p, "utf-8")) as T;

// describe.if(false) still executes this body to register tests, so the
// 52MB gitignored weight load must be lazy — eager reads crash CI.
let cached: {
  manifest: Manifest;
  w: Weights;
  model: NeedleModel;
  tok: NeedleTokenizer;
} | null = null;
function engine() {
  if (!cached) {
    const manifest = load<Manifest>(join(P, "manifest.json"));
    const buf = readFileSync(join(P, "weights.bin"));
    const ab = buf.buffer.slice(
      buf.byteOffset,
      buf.byteOffset + buf.byteLength,
    );
    const w = new Weights(manifest, ab);
    cached = {
      manifest,
      w,
      model: new NeedleModel(w),
      tok: new NeedleTokenizer(load<TokenizerData>(join(P, "tokenizer.json"))),
    };
  }
  return cached;
}

describe.if(hasWeights)("Needle model parity", () => {
  test("encoder output matches Python (bf16 ref): mean abs err < 1e-2", () => {
    const act = load<{
      enc_tokens: number[];
      encoder_out: number[][];
      first_step_logits_top5: number[];
    }>(join(FIX, "activation.json"));
    const { model, w, manifest } = engine();
    const { out, T } = model.encode(act.enc_tokens);
    const d = w.config.d_model;
    let sum = 0;
    let cnt = 0;
    for (let t = 0; t < T; t++)
      for (let i = 0; i < d; i++) {
        sum += Math.abs(out[t * d + i] - act.encoder_out[t][i]);
        cnt++;
      }
    expect(sum / cnt).toBeLessThan(1e-2);

    // Cross-attn + tied head: first-step decoder argmax must match reference.
    const session = model.startDecode(model.prepareCross(out, T), T);
    const logits = session.step(manifest.special.eos);
    let best = 0;
    for (let i = 1; i < logits.length; i++)
      if (logits[i] > logits[best]) best = i;
    expect(best).toBe(act.first_step_logits_top5[0]);
  }, 60000);

  test("end-to-end: known query -> expected tool-call JSON", () => {
    const e2e = load<Array<{ query: string; tools: string; text: string }>>(
      join(FIX, "e2e.json"),
    );
    const { model, tok, manifest } = engine();
    const inf = new NeedleInference(model, tok, manifest);
    // Spot-check a handful of diverse fixtures (full 268 is parity-eval.ts).
    // Compare parsed JSON (restore_tool_names re-serialization is whitespace-only
    // and name-identity here, so exact-string could differ on spacing).
    for (const c of [e2e[0], e2e[5], e2e[42], e2e[260], e2e[265]]) {
      expect(JSON.parse(inf.generate(c.query, c.tools))).toEqual(
        JSON.parse(c.text),
      );
    }
  }, 120000);
});

test.skipIf(hasWeights)(
  "model parity requires public/needle/weights.bin (run scripts/export-weights.py)",
  () => {},
);
