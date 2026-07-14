/**
 * Needle forward pass in pure fp32. Encoder-decoder transformer with:
 *  - shared/tied embeddings (scaled by sqrt(d_model)), tied output head
 *  - ZCRMSNorm ((1+scale)*x / sqrt(mean(x^2)+eps)), pre-norm blocks
 *  - RoPE (half-split) on q,k; per-head q/k ZCRMSNorm over head_dim
 *  - GQA (num_kv_heads < num_heads, k/v repeated)
 *  - sigmoid-gated residuals; NO feedforward (no_feedforward=true)
 *  - decoder self-attn causal, cross-attn over encoder output
 *
 * Batch=1, unpadded. Mirrors architecture.py exactly.
 */

import type { ModelConfig, Weights } from "./weights";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** y[T,out] = x[T,in] @ W[in,out] (Flax Dense kernel layout, no bias). */
function matmul(
  x: Float32Array,
  T: number,
  inD: number,
  W: Float32Array,
  outD: number,
): Float32Array {
  const y = new Float32Array(T * outD);
  for (let t = 0; t < T; t++) {
    const xoff = t * inD;
    const yoff = t * outD;
    for (let i = 0; i < inD; i++) {
      const xv = x[xoff + i];
      if (xv === 0) continue;
      const woff = i * outD;
      for (let o = 0; o < outD; o++) y[yoff + o] += xv * W[woff + o];
    }
  }
  return y;
}

/** ZCRMSNorm over rows of length `dim`: (1+scale)*x / sqrt(mean(x^2)+eps). */
function zcrms(
  x: Float32Array,
  T: number,
  dim: number,
  scale: Float32Array,
  eps: number,
): Float32Array {
  const y = new Float32Array(T * dim);
  for (let t = 0; t < T; t++) {
    const off = t * dim;
    let ss = 0;
    for (let d = 0; d < dim; d++) ss += x[off + d] * x[off + d];
    const rms = Math.sqrt(ss / dim + eps);
    for (let d = 0; d < dim; d++)
      y[off + d] = ((1 + scale[d]) * x[off + d]) / rms;
  }
  return y;
}

function ropeTable(theta: number, headDim: number, seqLen: number) {
  const half = headDim / 2;
  const cos = new Float32Array(seqLen * half);
  const sin = new Float32Array(seqLen * half);
  for (let t = 0; t < seqLen; t++) {
    for (let f = 0; f < half; f++) {
      const freq = Math.pow(theta, -(2 * f) / headDim);
      const a = t * freq;
      cos[t * half + f] = Math.cos(a);
      sin[t * half + f] = Math.sin(a);
    }
  }
  return { cos, sin, half };
}

/** In-place RoPE on a [T,H,hd] tensor (flat), applied per head. */
function applyRope(
  x: Float32Array,
  T: number,
  H: number,
  hd: number,
  cos: Float32Array,
  sin: Float32Array,
  half: number,
) {
  for (let t = 0; t < T; t++) {
    const coff = t * half;
    for (let h = 0; h < H; h++) {
      const base = (t * H + h) * hd;
      for (let f = 0; f < half; f++) {
        const x1 = x[base + f];
        const x2 = x[base + half + f];
        const c = cos[coff + f];
        const s = sin[coff + f];
        x[base + f] = x1 * c - x2 * s;
        x[base + half + f] = x2 * c + x1 * s;
      }
    }
  }
}

interface Rope {
  cos: Float32Array;
  sin: Float32Array;
  half: number;
}

/**
 * Multi-head attention. q_input [Tq,d], kv_input [Tk,d].
 * `causal` gates decoder self-attn (key j visible to query i iff j<=i).
 * `rope` applied to q (unless null) and k. Returns [Tq,d].
 */
function attention(
  cfg: ModelConfig,
  q_input: Float32Array,
  Tq: number,
  kv_input: Float32Array,
  Tk: number,
  Wq: Float32Array,
  Wk: Float32Array,
  Wv: Float32Array,
  Wout: Float32Array,
  qNorm: Float32Array,
  kNorm: Float32Array,
  rope: Rope | null,
  causal: boolean,
): Float32Array {
  const d = cfg.d_model;
  const H = cfg.num_heads;
  const Hkv = cfg.num_kv_heads;
  const hd = cfg.head_dim;
  const kvDim = Hkv * hd;
  const R = H / Hkv;
  const eps = cfg.eps;

  const q = matmul(q_input, Tq, d, Wq, d); // [Tq, H*hd]
  const k = matmul(kv_input, Tk, d, Wk, kvDim); // [Tk, Hkv*hd]
  const v = matmul(kv_input, Tk, d, Wv, kvDim);

  // Per-head q/k ZCRMSNorm over head_dim. Layout [T, H, hd] already contiguous.
  const qn = zcrms(q, Tq * H, hd, qNorm, eps);
  const kn = zcrms(k, Tk * Hkv, hd, kNorm, eps);

  if (rope) {
    // rope_keys_only is false in this model: rotate both q and k.
    applyRope(qn, Tq, H, hd, rope.cos, rope.sin, rope.half);
    applyRope(kn, Tk, Hkv, hd, rope.cos, rope.sin, rope.half);
  }

  const scale = Math.sqrt(hd);
  const out = new Float32Array(Tq * d); // [Tq, H*hd]
  const scores = new Float32Array(Tk);

  for (let h = 0; h < H; h++) {
    const kvh = Math.floor(h / R); // GQA mapping (jnp.repeat interleaves)
    for (let i = 0; i < Tq; i++) {
      const qoff = (i * H + h) * hd;
      const jmax = causal ? i : Tk - 1;
      let maxs = -Infinity;
      for (let j = 0; j <= jmax; j++) {
        const koff = (j * Hkv + kvh) * hd;
        let dot = 0;
        for (let e = 0; e < hd; e++) dot += qn[qoff + e] * kn[koff + e];
        const s = dot / scale;
        scores[j] = s;
        if (s > maxs) maxs = s;
      }
      let sum = 0;
      for (let j = 0; j <= jmax; j++) {
        const e = Math.exp(scores[j] - maxs);
        scores[j] = e;
        sum += e;
      }
      const ooff = (i * H + h) * hd;
      for (let j = 0; j <= jmax; j++) {
        const p = scores[j] / sum;
        const voff = (j * Hkv + kvh) * hd;
        for (let e = 0; e < hd; e++) out[ooff + e] += p * v[voff + e];
      }
    }
  }

  return matmul(out, Tq, d, Wout, d);
}

/**
 * Cross-attention with precomputed (encoder-derived) key/value tensors.
 * kCached/vCached are [Tk, Hkv, hd] flat; kCached already ZCRMSNorm'd, no RoPE.
 */
function crossAttention(
  cfg: ModelConfig,
  q_input: Float32Array,
  Tq: number,
  Tk: number,
  Wq: Float32Array,
  Wout: Float32Array,
  qNorm: Float32Array,
  kCached: Float32Array,
  vCached: Float32Array,
): Float32Array {
  const d = cfg.d_model;
  const H = cfg.num_heads;
  const Hkv = cfg.num_kv_heads;
  const hd = cfg.head_dim;
  const R = H / Hkv;

  const q = matmul(q_input, Tq, d, Wq, d);
  const qn = zcrms(q, Tq * H, hd, qNorm, cfg.eps);

  const scale = Math.sqrt(hd);
  const out = new Float32Array(Tq * d);
  const scores = new Float32Array(Tk);

  for (let h = 0; h < H; h++) {
    const kvh = Math.floor(h / R);
    for (let i = 0; i < Tq; i++) {
      const qoff = (i * H + h) * hd;
      let maxs = -Infinity;
      for (let j = 0; j < Tk; j++) {
        const koff = (j * Hkv + kvh) * hd;
        let dot = 0;
        for (let e = 0; e < hd; e++) dot += qn[qoff + e] * kCached[koff + e];
        const s = dot / scale;
        scores[j] = s;
        if (s > maxs) maxs = s;
      }
      let sum = 0;
      for (let j = 0; j < Tk; j++) {
        const e = Math.exp(scores[j] - maxs);
        scores[j] = e;
        sum += e;
      }
      const ooff = (i * H + h) * hd;
      for (let j = 0; j < Tk; j++) {
        const p = scores[j] / sum;
        const voff = (j * Hkv + kvh) * hd;
        for (let e = 0; e < hd; e++) out[ooff + e] += p * vCached[voff + e];
      }
    }
  }
  return matmul(out, Tq, d, Wout, d);
}

export interface CrossKV {
  k: Float32Array; // [T, Hkv, hd], ZCRMSNorm'd
  v: Float32Array; // [T, Hkv, hd]
}

export class NeedleModel {
  readonly cfg: ModelConfig;
  private w: Weights;
  private emb: Float32Array;
  private ropeCache = new Map<number, Rope>();

  constructor(w: Weights) {
    this.w = w;
    this.cfg = w.config;
    this.emb = w.get("embedding.embedding").data;
  }

  private rope(seqLen: number): Rope {
    let r = this.ropeCache.get(seqLen);
    if (!r) {
      r = ropeTable(this.cfg.rope_theta, this.cfg.head_dim, seqLen);
      this.ropeCache.set(seqLen, r);
    }
    return r;
  }

  private embed(tokens: number[]): Float32Array {
    const d = this.cfg.d_model;
    const scale = Math.sqrt(d);
    const x = new Float32Array(tokens.length * d);
    for (let t = 0; t < tokens.length; t++) {
      const eoff = tokens[t] * d;
      const xoff = t * d;
      for (let i = 0; i < d; i++) x[xoff + i] = this.emb[eoff + i] * scale;
    }
    return x;
  }

  /** Encode token ids -> encoder output [T, d].
   * Invariant "Encoder adapter parity" (docs/invariants.md): this output must
   * match GpuEncoder.encode within ~5e-6; both satisfy the Encoder seam. */
  encode(tokens: number[]): { out: Float32Array; T: number } {
    const cfg = this.cfg;
    const d = cfg.d_model;
    const T = tokens.length;
    let x = this.embed(tokens);
    const rope = this.rope(T);
    const eps = cfg.eps;

    for (let l = 0; l < cfg.num_encoder_layers; l++) {
      const pre = `encoder.layers.EncoderBlock_0`;
      const gate = sigmoid(this.w.layer(`${pre}.attn_gate`, l).data[0]);
      const xn = zcrms(
        x,
        T,
        d,
        this.w.layer(`${pre}.ZCRMSNorm_0.scale`, l).data,
        eps,
      );
      const a = attention(
        cfg,
        xn,
        T,
        xn,
        T,
        this.w.layer(`${pre}.self_attn.q_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.k_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.v_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.out_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.q_norm.scale`, l).data,
        this.w.layer(`${pre}.self_attn.k_norm.scale`, l).data,
        rope,
        false,
      );
      const nx = new Float32Array(T * d);
      for (let i = 0; i < T * d; i++) nx[i] = x[i] + gate * a[i];
      x = nx;
    }
    x = zcrms(x, T, d, this.w.get("encoder.final_norm.scale").data, eps);
    return { out: x, T };
  }

  /**
   * Precompute cross-attention K/V per decoder layer from encoder output
   * (fixed across decode steps): K is projected + ZCRMSNorm'd, V projected.
   */
  prepareCross(encOut: Float32Array, encT: number): CrossKV[] {
    const cfg = this.cfg;
    const d = cfg.d_model;
    const kvDim = cfg.num_kv_heads * cfg.head_dim;
    const out: CrossKV[] = [];
    for (let l = 0; l < cfg.num_decoder_layers; l++) {
      const pre = `decoder.layers.DecoderBlock_0`;
      const k = matmul(
        encOut,
        encT,
        d,
        this.w.layer(`${pre}.cross_attn.k_proj.kernel`, l).data,
        kvDim,
      );
      const kn = zcrms(
        k,
        encT * cfg.num_kv_heads,
        cfg.head_dim,
        this.w.layer(`${pre}.cross_attn.k_norm.scale`, l).data,
        cfg.eps,
      );
      const v = matmul(
        encOut,
        encT,
        d,
        this.w.layer(`${pre}.cross_attn.v_proj.kernel`, l).data,
        kvDim,
      );
      out.push({ k: kn, v });
    }
    return out;
  }

  /**
   * Begin a KV-cached greedy decode. Returns a stateful DecodeSession whose
   * step(token) appends one position and returns its logits. The model stays
   * stateless — all mutable decode state lives in the session.
   */
  startDecode(cross: CrossKV[], encT: number): DecodeSession {
    return new DecodeSession(
      this.cfg,
      this.w,
      this.emb,
      this.rope(this.cfg.max_gen_len),
      cross,
      encT,
    );
  }
}

/**
 * Single-step decoder self-attention against a growing K/V cache. `xn` is the
 * normed [1,d] block input; q/k/v project from it (self-attn: q_input==kv_input),
 * q/k get per-head ZCRMSNorm then RoPE at absolute position `pos`. The new k/v
 * are written into kCache/vCache ([maxLen, Hkv, hd] flat) at `pos`, then the
 * single query attends causally over cache rows 0..pos. Returns [1,d].
 *
 * Bit-identical to attention(..., causal=true) evaluated at the last position:
 * causal masking makes each position's k/v depend only on tokens <= it, so the
 * k/v cached when a token was stepped equal what a full re-run would recompute.
 */
function selfAttnStep(
  cfg: ModelConfig,
  xn: Float32Array,
  Wq: Float32Array,
  Wk: Float32Array,
  Wv: Float32Array,
  Wout: Float32Array,
  qNorm: Float32Array,
  kNorm: Float32Array,
  rope: Rope,
  pos: number,
  kCache: Float32Array,
  vCache: Float32Array,
): Float32Array {
  const d = cfg.d_model;
  const H = cfg.num_heads;
  const Hkv = cfg.num_kv_heads;
  const hd = cfg.head_dim;
  const kvDim = Hkv * hd;
  const R = H / Hkv;
  const eps = cfg.eps;

  const q = matmul(xn, 1, d, Wq, d); // [1, H*hd]
  const k = matmul(xn, 1, d, Wk, kvDim); // [1, Hkv*hd]
  const v = matmul(xn, 1, d, Wv, kvDim);

  const qn = zcrms(q, H, hd, qNorm, eps);
  const kn = zcrms(k, Hkv, hd, kNorm, eps);

  // RoPE at absolute position `pos` (single row of the precomputed table).
  const half = rope.half;
  const cosRow = rope.cos.subarray(pos * half, (pos + 1) * half);
  const sinRow = rope.sin.subarray(pos * half, (pos + 1) * half);
  applyRope(qn, 1, H, hd, cosRow, sinRow, half);
  applyRope(kn, 1, Hkv, hd, cosRow, sinRow, half);

  // Append this position's k/v to the cache.
  kCache.set(kn, pos * kvDim);
  vCache.set(v, pos * kvDim);

  const Tk = pos + 1;
  const scale = Math.sqrt(hd);
  const out = new Float32Array(d); // [1, H*hd]
  const scores = new Float32Array(Tk);

  for (let h = 0; h < H; h++) {
    const kvh = Math.floor(h / R); // GQA mapping (jnp.repeat interleaves)
    const qoff = h * hd;
    let maxs = -Infinity;
    for (let j = 0; j < Tk; j++) {
      const koff = (j * Hkv + kvh) * hd;
      let dot = 0;
      for (let e = 0; e < hd; e++) dot += qn[qoff + e] * kCache[koff + e];
      const s = dot / scale;
      scores[j] = s;
      if (s > maxs) maxs = s;
    }
    let sum = 0;
    for (let j = 0; j < Tk; j++) {
      const e = Math.exp(scores[j] - maxs);
      scores[j] = e;
      sum += e;
    }
    const ooff = h * hd;
    for (let j = 0; j < Tk; j++) {
      const p = scores[j] / sum;
      const voff = (j * Hkv + kvh) * hd;
      for (let e = 0; e < hd; e++) out[ooff + e] += p * vCache[voff + e];
    }
  }
  return matmul(out, 1, d, Wout, d);
}

/**
 * Stateful KV-cached greedy decode. Holds per-layer self-attention K/V caches
 * (cross-attention K/V is fixed and lives in the passed CrossKV[]). step(token)
 * appends the token at the next position, attends over the cache, and returns
 * logits for that new position only. Replaces the old O(steps^2) full-buffer
 * decodeStep; produces bit-identical logits (see selfAttnStep).
 */
export class DecodeSession {
  private len = 0; // tokens consumed == absolute position of the next token
  private kCache: Float32Array[]; // per layer, [maxLen, Hkv, hd] flat
  private vCache: Float32Array[];

  constructor(
    private cfg: ModelConfig,
    private w: Weights,
    private emb: Float32Array,
    private rope: Rope,
    private cross: CrossKV[],
    private encT: number,
  ) {
    const kvDim = cfg.num_kv_heads * cfg.head_dim;
    const maxLen = cfg.max_gen_len;
    this.kCache = Array.from(
      { length: cfg.num_decoder_layers },
      () => new Float32Array(maxLen * kvDim),
    );
    this.vCache = Array.from(
      { length: cfg.num_decoder_layers },
      () => new Float32Array(maxLen * kvDim),
    );
  }

  /** Append `token` at its position; return logits [vocab_size] for it. */
  step(token: number): Float32Array {
    const cfg = this.cfg;
    const d = cfg.d_model;
    const eps = cfg.eps;
    const pos = this.len;
    if (pos >= cfg.max_gen_len)
      throw new Error(
        `DecodeSession.step past max_gen_len (${cfg.max_gen_len})`,
      );

    // Embed the single token, scaled by sqrt(d).
    const scale = Math.sqrt(d);
    let x: Float32Array = new Float32Array(d);
    const eoff = token * d;
    for (let i = 0; i < d; i++) x[i] = this.emb[eoff + i] * scale;

    for (let l = 0; l < cfg.num_decoder_layers; l++) {
      const pre = `decoder.layers.DecoderBlock_0`;
      const selfGate = sigmoid(
        this.w.layer(`${pre}.self_attn_gate`, l).data[0],
      );
      const crossGate = sigmoid(
        this.w.layer(`${pre}.cross_attn_gate`, l).data[0],
      );

      let xn = zcrms(
        x,
        1,
        d,
        this.w.layer(`${pre}.ZCRMSNorm_0.scale`, l).data,
        eps,
      );
      const sa = selfAttnStep(
        cfg,
        xn,
        this.w.layer(`${pre}.self_attn.q_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.k_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.v_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.out_proj.kernel`, l).data,
        this.w.layer(`${pre}.self_attn.q_norm.scale`, l).data,
        this.w.layer(`${pre}.self_attn.k_norm.scale`, l).data,
        this.rope,
        pos,
        this.kCache[l],
        this.vCache[l],
      );
      const x1 = new Float32Array(d);
      for (let i = 0; i < d; i++) x1[i] = x[i] + selfGate * sa[i];

      xn = zcrms(
        x1,
        1,
        d,
        this.w.layer(`${pre}.ZCRMSNorm_1.scale`, l).data,
        eps,
      );
      const ca = crossAttention(
        cfg,
        xn,
        1,
        this.encT,
        this.w.layer(`${pre}.cross_attn.q_proj.kernel`, l).data,
        this.w.layer(`${pre}.cross_attn.out_proj.kernel`, l).data,
        this.w.layer(`${pre}.cross_attn.q_norm.scale`, l).data,
        this.cross[l].k,
        this.cross[l].v,
      );
      const x2 = new Float32Array(d);
      for (let i = 0; i < d; i++) x2[i] = x1[i] + crossGate * ca[i];
      x = x2;
    }
    x = zcrms(x, 1, d, this.w.get("decoder.ZCRMSNorm_0.scale").data, eps);

    // Logits for this position: out_row @ embedding.T
    const V = cfg.vocab_size;
    const logits = new Float32Array(V);
    for (let vtok = 0; vtok < V; vtok++) {
      const veoff = vtok * d;
      let acc = 0;
      for (let i = 0; i < d; i++) acc += x[i] * this.emb[veoff + i];
      logits[vtok] = acc;
    }

    this.len++;
    return logits;
  }
}
