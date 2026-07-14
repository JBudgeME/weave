/**
 * buildEncoderInput + greedy decode, mirroring needle.model.run.generate
 * (constrained=False, normalize=True). normalize is a no-op for the fixed
 * snake_case 4-tool catalog (verified: json round-trip is identity, so the
 * tools string is fed verbatim), so it is intentionally not ported.
 */

import type { Encoder } from "./encoder";
import { type CrossKV, NeedleModel } from "./model";
import type { NeedleTokenizer } from "./tokenizer";
import type { Manifest } from "./weights";

export interface Special {
  pad: number;
  eos: number;
  tools: number;
}

/** [query..., <tools>, tools...] truncated to maxEncLen. */
export function buildEncoderInput(
  tok: NeedleTokenizer,
  query: string,
  tools: string,
  toolsSepId: number,
  maxEncLen: number,
): number[] {
  let q = tok.encode(query);
  const t = tok.encode(tools);
  const maxQuery = maxEncLen - 2;
  if (q.length > maxQuery) q = q.slice(0, maxQuery);
  const remaining = maxEncLen - q.length - 1;
  return [...q, toolsSepId, ...t.slice(0, Math.max(0, remaining))];
}

function argmax(a: Float32Array): number {
  let best = 0;
  let bestV = a[0];
  for (let i = 1; i < a.length; i++) {
    if (a[i] > bestV) {
      bestV = a[i];
      best = i;
    }
  }
  return best;
}

export class NeedleInference {
  constructor(
    private model: NeedleModel,
    private tok: NeedleTokenizer,
    private manifest: Manifest,
  ) {}

  /** Token ids for a query+tools pair (shared by the sync and async paths). */
  encInput(query: string, tools: string): number[] {
    const cfg = this.model.cfg;
    return buildEncoderInput(
      this.tok,
      query,
      tools,
      this.manifest.special.tools,
      cfg.max_enc_len,
    );
  }

  /** Returns the raw generated tool-call text (with <tool_call> prefix stripped). */
  generate(query: string, tools: string): string {
    const { out, T } = this.model.encode(this.encInput(query, tools));
    return this.runDecode(this.model.prepareCross(out, T), T);
  }

  /**
   * Async variant used by the WebGPU worker. `backend.encode` runs the encoder
   * forward pass and `backend.prepareCross` projects encoder output into the
   * decoder cross-attention K/V — both on whatever backend is active (GPU or the
   * TS fallback). The KV-cached autoregressive decode loop always runs in TS.
   * `encodeMs` covers the encoder forward; `decodeMs` covers prepareCross + loop.
   * Same text as generate().
   */
  async generateAsync(
    query: string,
    tools: string,
    backend: Encoder,
  ): Promise<{ text: string; encodeMs: number; decodeMs: number }> {
    const input = this.encInput(query, tools);
    const t0 = performance.now();
    const { out, T } = await backend.encode(input);
    const t1 = performance.now();
    const cross = await backend.prepareCross(out, T);
    const text = this.runDecode(cross, T);
    const t2 = performance.now();
    return { text, encodeMs: t1 - t0, decodeMs: t2 - t1 };
  }

  /** KV-cached greedy decode from precomputed cross K/V -> tool-call text. */
  private runDecode(cross: CrossKV[], T: number): string {
    const cfg = this.model.cfg;
    const sp = this.manifest.special;

    const session = this.model.startDecode(cross, T);
    const gen: number[] = [];
    let next = sp.eos;
    for (let i = 0; i < cfg.max_gen_len - 1; i++) {
      const logits = session.step(next);
      next = argmax(logits);
      if (next === sp.eos) break;
      gen.push(next);
    }
    let text = this.tok.decode(gen);
    if (text.startsWith("<tool_call>")) text = text.slice("<tool_call>".length);
    // restore_tool_names: re-serialize compact (JSON.parse -> stringify). Tool
    // names are already snake_case so the name map is identity; the only real
    // effect is compaction (matches Python generate). Malformed -> return as-is.
    try {
      text = JSON.stringify(JSON.parse(text));
    } catch {
      /* leave raw text (Python falls back to identity string replacement here) */
    }
    return text;
  }
}
