/**
 * Weight blob loader. Decodes the fp16 little-endian `weights.bin` into a
 * single fp32 Float32Array on construction and exposes named tensors (with
 * their shapes) as zero-copy subarray views. Runtime-agnostic: takes an
 * ArrayBuffer, no fs/DOM.
 */

export interface TensorMeta {
  offset: number; // byte offset into weights.bin
  shape: number[];
  dtype: "float16";
}

export interface ModelConfig {
  vocab_size: number;
  d_model: number;
  num_heads: number;
  num_kv_heads: number;
  num_encoder_layers: number;
  num_decoder_layers: number;
  d_ff: number;
  head_dim: number;
  rope_theta: number;
  activation: string;
  no_feedforward: boolean;
  pad_token_id: number;
  eps: number;
  max_enc_len: number;
  max_gen_len: number;
}

export interface Manifest {
  config: ModelConfig;
  rope: { theta: number; head_dim: number };
  special: {
    pad: number;
    eos: number;
    bos: number;
    unk: number;
    tool_call: number;
    tools: number;
  };
  total_bytes: number;
  tensors: Record<string, TensorMeta>;
}

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

/** Decode a little-endian fp16 buffer to a fresh Float32Array. */
function decodeFp16(buf: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(buf);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  let val: number;
  if (exp === 0) {
    val = frac * Math.pow(2, -24); // subnormal
  } else if (exp === 0x1f) {
    val = frac === 0 ? Infinity : NaN;
  } else {
    val = (1 + frac / 1024) * Math.pow(2, exp - 15);
  }
  return sign ? -val : val;
}

export class Weights {
  readonly config: ModelConfig;
  readonly manifest: Manifest;
  private data: Float32Array;

  constructor(manifest: Manifest, buffer: ArrayBuffer) {
    this.manifest = manifest;
    this.config = manifest.config;
    this.data = decodeFp16(buffer);
  }

  get(name: string): Tensor {
    const meta = this.manifest.tensors[name];
    if (!meta) throw new Error(`missing tensor: ${name}`);
    const n = meta.shape.reduce((a, b) => a * b, 1);
    const start = meta.offset / 2; // fp16 = 2 bytes
    return { data: this.data.subarray(start, start + n), shape: meta.shape };
  }

  /**
   * Layer slice of a scan-stacked tensor whose leading axis is the layer index.
   * Returns the flat data for `layer` plus the remaining (per-layer) shape.
   */
  layer(name: string, layer: number): Tensor {
    const t = this.get(name);
    const perLayer = t.shape.slice(1);
    const stride = perLayer.reduce((a, b) => a * b, 1);
    const start = layer * stride;
    return { data: t.data.subarray(start, start + stride), shape: perLayer };
  }
}
