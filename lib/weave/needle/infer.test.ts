/**
 * Seam test for the Encoder abstraction (./encoder.ts). Proves
 * NeedleInference.generateAsync drives the encode -> prepareCross -> decode
 * pipeline against ANY object satisfying Encoder — no real weights, no GPU, no
 * NeedleModel. The encoder AND the decode-side model/session are scripted, so
 * this runs in CI (unlike model.test.ts, which needs the gitignored weights).
 *
 * What the seam makes testable here: that generateAsync (a) awaits encode,
 * (b) threads its {out, T} unchanged into prepareCross, (c) awaits prepareCross
 * whether it returns sync or async (the Encoder return-type union), and (d)
 * feeds that K/V straight into the decode loop. The encoder's numerical
 * *correctness* is a separate concern (parity-eval.ts / the invariant).
 */
import { describe, expect, test } from "bun:test";

import type { Encoder } from "./encoder";
import { NeedleInference } from "./infer";
import type { CrossKV, NeedleModel } from "./model";
import type { NeedleTokenizer } from "./tokenizer";
import type { Manifest } from "./weights";

const EOS = 0;
const TOOLS_SEP = 99;
const SCRIPTED_TEXT = '<tool_call>[{"name": "show_tasks", "arguments": {}}]';

/** A fake decode session whose argmax walks a fixed token sequence then stops. */
function scriptedSession(emit: number[], onCross: (c: CrossKV[]) => void) {
  return (cross: CrossKV[]) => {
    onCross(cross);
    const seq = [...emit, EOS];
    let i = 0;
    return {
      step(): Float32Array {
        const id = seq[Math.min(i++, seq.length - 1)];
        const logits = new Float32Array(128);
        logits[id] = 1; // argmax -> id
        return logits;
      },
    };
  };
}

/** NeedleInference wired to a scripted model + tokenizer (no weights). */
function fakeInference(onCross: (c: CrossKV[]) => void) {
  const model = {
    cfg: { max_enc_len: 64, max_gen_len: 16 },
    startDecode: scriptedSession([1, 2, 3], onCross),
  } as unknown as NeedleModel;
  const tok = {
    encode: (s: string): number[] => s.split(" ").map((_, i) => i + 1),
    decode: (): string => SCRIPTED_TEXT,
  } as unknown as NeedleTokenizer;
  const manifest = {
    special: { pad: 1, eos: EOS, tools: TOOLS_SEP },
  } as unknown as Manifest;
  return new NeedleInference(model, tok, manifest);
}

/** Records how the backend was used; `cross` is what reached the decoder. */
function makeBackend(asyncPrepare: boolean) {
  const seen = {
    encodeTokens: null as number[] | null,
    encodeOut: null as Float32Array | null,
    prepareArgs: null as { out: Float32Array; T: number } | null,
    producedCross: null as CrossKV[] | null,
  };
  const backend: Encoder = {
    encode(tokens: number[]) {
      seen.encodeTokens = tokens;
      const out = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      seen.encodeOut = out;
      return { out, T: 2 };
    },
    prepareCross(out: Float32Array, T: number) {
      seen.prepareArgs = { out, T };
      const cross: CrossKV[] = [
        { k: new Float32Array([9]), v: new Float32Array([8]) },
      ];
      seen.producedCross = cross;
      return asyncPrepare ? Promise.resolve(cross) : cross;
    },
  };
  return { backend, seen };
}

describe("generateAsync drives the decode loop through the Encoder seam", () => {
  test("threads a fake backend end-to-end and decodes scripted tokens", async () => {
    let decoderCross: CrossKV[] | null = null;
    const inf = fakeInference((c) => (decoderCross = c));
    const { backend, seen } = makeBackend(false);

    const { text, encodeMs, decodeMs } = await inf.generateAsync(
      "remove practice guitar",
      "[]",
      backend,
    );

    // Decode ran off the fake backend: argmax walked [1,2,3] -> eos, and the
    // text is restore_tool_names(compact) of the scripted tool call.
    expect(text).toBe('[{"name":"show_tasks","arguments":{}}]');
    // encode was called with the built encoder input (non-empty token ids).
    expect(seen.encodeTokens?.length).toBeGreaterThan(0);
    // prepareCross received encode()'s exact out ref and T — threaded unchanged.
    expect(seen.prepareArgs!.out).toBe(seen.encodeOut!);
    expect(seen.prepareArgs!.T).toBe(2);
    // The exact CrossKV[] prepareCross produced is what the decoder consumed.
    expect(decoderCross!).toBe(seen.producedCross!);
    expect(typeof encodeMs).toBe("number");
    expect(decodeMs).toBeGreaterThanOrEqual(0);
  });

  test("awaits a Promise-returning prepareCross (sync|async union)", async () => {
    const inf = fakeInference(() => {});
    const { backend } = makeBackend(true); // prepareCross returns a Promise
    const { text } = await inf.generateAsync("show my tasks", "[]", backend);
    expect(text).toBe('[{"name":"show_tasks","arguments":{}}]');
  });
});
