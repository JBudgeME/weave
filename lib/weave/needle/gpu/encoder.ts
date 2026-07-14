/// <reference types="@webgpu/types" />

/**
 * WebGPU Needle encoder. Reproduces NeedleModel.encode() (../model.ts) on the
 * GPU in f32: embedding + RoPE table on the CPU, then 12 encoder blocks
 * (ZCRMSNorm -> q/k/v proj -> per-head norm -> RoPE -> GQA attention -> out
 * proj -> sigmoid-gated residual) and the final norm as compute dispatches.
 * Output shape matches encode(): { out: Float32Array [T*d], T } so the existing
 * TS prepareCross/DecodeSession consume it unchanged.
 *
 * Invariant "Encoder adapter parity" (docs/invariants.md): this adapter's
 * encode/prepareCross output must match NeedleModel's within ~5e-6; both satisfy
 * the Encoder seam (../encoder.ts).
 */

import type { CrossKV } from "../model";
import type { Weights } from "../weights";
import {
  attnWGSL,
  type Dims,
  gatedAddWGSL,
  matmulWGSL,
  ropeWGSL,
  zcrmsWGSL,
} from "./shaders";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

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
  return { cos, sin };
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export class GpuEncoder {
  private dims: Dims;
  private L: number;
  private d: number;
  private kvDim: number;
  private maxT: number;
  private emb: Float32Array;

  // pipelines
  private pMM512!: GPUComputePipeline; // 512 -> 512
  private pMM256!: GPUComputePipeline; // 512 -> 256
  private pNormFull!: GPUComputePipeline;
  private pNormHeadQ!: GPUComputePipeline;
  private pNormHeadK!: GPUComputePipeline;
  private pRopeQ!: GPUComputePipeline;
  private pRopeK!: GPUComputePipeline;
  private pAttn!: GPUComputePipeline;
  private pGate!: GPUComputePipeline;

  // working buffers
  private bufX!: GPUBuffer;
  private bufXn!: GPUBuffer;
  private bufQ!: GPUBuffer;
  private bufK!: GPUBuffer;
  private bufV!: GPUBuffer;
  private bufAttn!: GPUBuffer;
  private bufO!: GPUBuffer;
  private bufOut!: GPUBuffer;
  private cosBuf!: GPUBuffer;
  private sinBuf!: GPUBuffer;
  private readback!: GPUBuffer;
  private tParam!: GPUBuffer;
  private layerParams!: GPUBuffer[];

  // per-layer bind groups
  private bg!: {
    norm0: GPUBindGroup;
    q: GPUBindGroup;
    k: GPUBindGroup;
    v: GPUBindGroup;
    qNorm: GPUBindGroup;
    kNorm: GPUBindGroup;
    ropeQ: GPUBindGroup;
    ropeK: GPUBindGroup;
    attn: GPUBindGroup;
    out: GPUBindGroup;
    gate: GPUBindGroup;
  }[];
  private bgFinal!: GPUBindGroup;
  private gates!: number[];

  // decoder cross-attention K/V projection (prepareCross on GPU)
  private Ldec!: number;
  private bufCross!: GPUBuffer; // [Ldec][2][maxT*kvDim] f32, matmul targets
  private crossReadback!: GPUBuffer;
  private bgCross!: { k: GPUBindGroup; kNorm: GPUBindGroup; v: GPUBindGroup }[];

  constructor(
    private device: GPUDevice,
    private w: Weights,
  ) {
    const c = w.config;
    this.dims = {
      d: c.d_model,
      H: c.num_heads,
      Hkv: c.num_kv_heads,
      hd: c.head_dim,
      half: c.head_dim / 2,
      eps: c.eps,
    };
    this.L = c.num_encoder_layers;
    this.d = c.d_model;
    this.kvDim = c.num_kv_heads * c.head_dim;
    this.maxT = c.max_enc_len;
    this.emb = w.get("embedding.embedding").data;
  }

  private pipe(code: string): GPUComputePipeline {
    const shader = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({
      layout: "auto",
      compute: { module: shader, entryPoint: "main" },
    });
  }

  private storage(floats: number): GPUBuffer {
    return this.device.createBuffer({ size: floats * 4, usage: STORAGE });
  }

  private weightBuf(data: Float32Array): GPUBuffer {
    // Copy the subarray into a compact, zero-offset Float32Array: queue.writeBuffer
    // reads from the view's underlying ArrayBuffer and byteOffset handling on
    // subarrays is a footgun across implementations.
    const compact = new Float32Array(data);
    const b = this.device.createBuffer({
      size: compact.byteLength,
      usage: STORAGE,
    });
    this.device.queue.writeBuffer(b, 0, compact);
    return b;
  }

  /** Upload weights + allocate buffers + pipelines. Call once. */
  init(): void {
    const { dims, d, kvDim, maxT, device } = this;
    this.pMM512 = this.pipe(matmulWGSL(d, d));
    this.pMM256 = this.pipe(matmulWGSL(d, kvDim));
    this.pNormFull = this.pipe(zcrmsWGSL(dims, d, 1, false));
    this.pNormHeadQ = this.pipe(zcrmsWGSL(dims, dims.hd, dims.H, true));
    this.pNormHeadK = this.pipe(zcrmsWGSL(dims, dims.hd, dims.Hkv, true));
    this.pRopeQ = this.pipe(ropeWGSL(dims, dims.H));
    this.pRopeK = this.pipe(ropeWGSL(dims, dims.Hkv));
    this.pAttn = this.pipe(attnWGSL(dims));
    this.pGate = this.pipe(gatedAddWGSL(dims));

    this.bufX = this.storage(maxT * d);
    this.bufXn = this.storage(maxT * d);
    this.bufQ = this.storage(maxT * d);
    this.bufK = this.storage(maxT * kvDim);
    this.bufV = this.storage(maxT * kvDim);
    this.bufAttn = this.storage(maxT * d);
    this.bufO = this.storage(maxT * d);
    // bufOut is the readback source — needs COPY_SRC on top of STORAGE.
    this.bufOut = device.createBuffer({
      size: maxT * d * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.cosBuf = this.storage(maxT * dims.half);
    this.sinBuf = this.storage(maxT * dims.half);
    this.readback = device.createBuffer({
      size: maxT * d * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.tParam = device.createBuffer({ size: 16, usage: UNIFORM });
    this.layerParams = Array.from({ length: this.L }, () =>
      device.createBuffer({ size: 16, usage: UNIFORM }),
    );

    const pre = "encoder.layers.EncoderBlock_0";
    const layer = (name: string, l: number) => this.w.layer(name, l).data;
    this.gates = Array.from({ length: this.L }, (_, l) =>
      sigmoid(layer(`${pre}.attn_gate`, l)[0]),
    );

    const bind = (
      pipe: GPUComputePipeline,
      entries: { binding: number; buf: GPUBuffer }[],
    ) =>
      device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: entries.map((e) => ({
          binding: e.binding,
          resource: { buffer: e.buf },
        })),
      });

    this.bg = [];
    for (let l = 0; l < this.L; l++) {
      const norm0 = this.weightBuf(layer(`${pre}.ZCRMSNorm_0.scale`, l));
      const qNorm = this.weightBuf(layer(`${pre}.self_attn.q_norm.scale`, l));
      const kNorm = this.weightBuf(layer(`${pre}.self_attn.k_norm.scale`, l));
      const Wq = this.weightBuf(layer(`${pre}.self_attn.q_proj.kernel`, l));
      const Wk = this.weightBuf(layer(`${pre}.self_attn.k_proj.kernel`, l));
      const Wv = this.weightBuf(layer(`${pre}.self_attn.v_proj.kernel`, l));
      const Wout = this.weightBuf(layer(`${pre}.self_attn.out_proj.kernel`, l));
      this.bg.push({
        norm0: bind(this.pNormFull, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: norm0 },
          { binding: 2, buf: this.bufX },
          { binding: 3, buf: this.bufXn },
        ]),
        q: bind(this.pMM512, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.bufXn },
          { binding: 2, buf: Wq },
          { binding: 3, buf: this.bufQ },
        ]),
        k: bind(this.pMM256, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.bufXn },
          { binding: 2, buf: Wk },
          { binding: 3, buf: this.bufK },
        ]),
        v: bind(this.pMM256, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.bufXn },
          { binding: 2, buf: Wv },
          { binding: 3, buf: this.bufV },
        ]),
        qNorm: bind(this.pNormHeadQ, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: qNorm },
          { binding: 2, buf: this.bufQ },
        ]),
        kNorm: bind(this.pNormHeadK, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: kNorm },
          { binding: 2, buf: this.bufK },
        ]),
        ropeQ: bind(this.pRopeQ, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.cosBuf },
          { binding: 2, buf: this.sinBuf },
          { binding: 3, buf: this.bufQ },
        ]),
        ropeK: bind(this.pRopeK, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.cosBuf },
          { binding: 2, buf: this.sinBuf },
          { binding: 3, buf: this.bufK },
        ]),
        attn: bind(this.pAttn, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.bufQ },
          { binding: 2, buf: this.bufK },
          { binding: 3, buf: this.bufV },
          { binding: 4, buf: this.bufAttn },
        ]),
        out: bind(this.pMM512, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.bufAttn },
          { binding: 2, buf: Wout },
          { binding: 3, buf: this.bufO },
        ]),
        gate: bind(this.pGate, [
          { binding: 0, buf: this.layerParams[l] },
          { binding: 1, buf: this.bufO },
          { binding: 2, buf: this.bufX },
        ]),
      });
    }

    const finalNorm = this.weightBuf(
      this.w.get("encoder.final_norm.scale").data,
    );
    this.bgFinal = bind(this.pNormFull, [
      { binding: 0, buf: this.tParam },
      { binding: 1, buf: finalNorm },
      { binding: 2, buf: this.bufX },
      { binding: 3, buf: this.bufOut },
    ]);

    // --- Decoder cross-attention K/V projection (prepareCross on GPU) ---
    // Reuses bufOut (encoder output) as input. Per decoder layer: k = encOut @
    // Wk (+ per-head ZCRMSNorm), v = encOut @ Wv. Results land in bufCross, laid
    // out [Ldec][2][maxT*kvDim] (k then v), read back in one shot.
    this.Ldec = this.w.config.num_decoder_layers;
    const slice = maxT * kvDim;
    // bgCross binds bufCross at per-layer byte offsets; WebGPU requires
    // storage-buffer offsets aligned to 256. Holds today because
    // maxT*kvDim*4 is 256-divisible — fail loudly if a config change breaks it.
    if ((slice * 4) % 256 !== 0) {
      throw new Error(`cross buffer slice ${slice * 4}B not 256-aligned`);
    }
    this.bufCross = device.createBuffer({
      size: this.Ldec * 2 * slice * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.crossReadback = device.createBuffer({
      size: this.Ldec * 2 * slice * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const bindOff = (
      pipe: GPUComputePipeline,
      entries: {
        binding: number;
        buf: GPUBuffer;
        offset?: number;
        size?: number;
      }[],
    ) =>
      device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: entries.map((e) => ({
          binding: e.binding,
          resource:
            e.offset !== undefined
              ? { buffer: e.buf, offset: e.offset, size: e.size }
              : { buffer: e.buf },
        })),
      });
    const dpre = "decoder.layers.DecoderBlock_0";
    this.bgCross = [];
    for (let l = 0; l < this.Ldec; l++) {
      const Wk = this.weightBuf(layer(`${dpre}.cross_attn.k_proj.kernel`, l));
      const Wv = this.weightBuf(layer(`${dpre}.cross_attn.v_proj.kernel`, l));
      const kN = this.weightBuf(layer(`${dpre}.cross_attn.k_norm.scale`, l));
      const kOff = 2 * l * slice * 4;
      const vOff = (2 * l + 1) * slice * 4;
      this.bgCross.push({
        k: bindOff(this.pMM256, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.bufOut },
          { binding: 2, buf: Wk },
          { binding: 3, buf: this.bufCross, offset: kOff, size: slice * 4 },
        ]),
        kNorm: bindOff(this.pNormHeadK, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: kN },
          { binding: 2, buf: this.bufCross, offset: kOff, size: slice * 4 },
        ]),
        v: bindOff(this.pMM256, [
          { binding: 0, buf: this.tParam },
          { binding: 1, buf: this.bufOut },
          { binding: 2, buf: Wv },
          { binding: 3, buf: this.bufCross, offset: vOff, size: slice * 4 },
        ]),
      });
    }
  }

  /**
   * Project the encoder output (still resident in bufOut from the preceding
   * encode()) into per-decoder-layer cross-attention K/V, matching
   * NeedleModel.prepareCross: k is ZCRMSNorm'd per head, v is raw. Returns
   * CrossKV[] consumed unchanged by the TS DecodeSession.
   */
  async prepareCross(_encOut: Float32Array, encT: number): Promise<CrossKV[]> {
    const { device, kvDim } = this;
    // bufOut must hold the output of the immediately-preceding encode() for
    // this same input — a mismatch means silently wrong K/V. Fail loud.
    if (encT !== this.lastEncT) {
      throw new Error(
        `GpuEncoder.prepareCross: encT ${encT} != last encode T ${this.lastEncT}`,
      );
    }
    device.queue.writeBuffer(this.tParam, 0, new Uint32Array([encT]));
    const enc = device.createCommandEncoder();
    const run = (pipe: GPUComputePipeline, bg: GPUBindGroup, count: number) => {
      const p = enc.beginComputePass();
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(Math.ceil(count / 64));
      p.end();
    };
    for (let l = 0; l < this.Ldec; l++) {
      const b = this.bgCross[l];
      run(this.pMM256, b.k, encT * kvDim);
      run(this.pNormHeadK, b.kNorm, encT * this.dims.Hkv);
      run(this.pMM256, b.v, encT * kvDim);
    }
    enc.copyBufferToBuffer(
      this.bufCross,
      0,
      this.crossReadback,
      0,
      this.bufCross.size,
    );
    device.queue.submit([enc.finish()]);

    await this.crossReadback.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(this.crossReadback.getMappedRange().slice(0));
    this.crossReadback.unmap();

    const slice = this.maxT * kvDim;
    const n = encT * kvDim;
    const out: CrossKV[] = [];
    for (let l = 0; l < this.Ldec; l++) {
      const kStart = 2 * l * slice;
      const vStart = (2 * l + 1) * slice;
      out.push({
        k: all.slice(kStart, kStart + n),
        v: all.slice(vStart, vStart + n),
      });
    }
    return out;
  }

  /** Encode token ids -> encoder output [T*d] flat. Mirrors NeedleModel.encode. */
  /** T of the most recent encode(); guards prepareCross pairing. */
  private lastEncT = -1;

  async encode(tokens: number[]): Promise<{ out: Float32Array; T: number }> {
    const { device, d, dims } = this;
    const T = tokens.length;
    if (T > this.maxT)
      throw new Error(`GpuEncoder: T ${T} > maxT ${this.maxT}`);
    this.lastEncT = T;

    // Embedding (scaled by sqrt(d)) + RoPE table on the CPU.
    const x0 = new Float32Array(T * d);
    const s = Math.sqrt(d);
    for (let t = 0; t < T; t++) {
      const eoff = tokens[t] * d;
      const xoff = t * d;
      for (let i = 0; i < d; i++) x0[xoff + i] = this.emb[eoff + i] * s;
    }
    const { cos, sin } = ropeTable(this.w.config.rope_theta, dims.hd, T);

    device.queue.writeBuffer(this.bufX, 0, x0);
    device.queue.writeBuffer(this.cosBuf, 0, cos);
    device.queue.writeBuffer(this.sinBuf, 0, sin);
    device.queue.writeBuffer(this.tParam, 0, new Uint32Array([T]));
    for (let l = 0; l < this.L; l++) {
      const lp = new ArrayBuffer(16);
      new Uint32Array(lp, 0, 1)[0] = T;
      new Float32Array(lp, 4, 1)[0] = this.gates[l];
      device.queue.writeBuffer(this.layerParams[l], 0, lp);
    }

    const enc = device.createCommandEncoder();
    const run = (pipe: GPUComputePipeline, bg: GPUBindGroup, count: number) => {
      const p = enc.beginComputePass();
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(Math.ceil(count / 64));
      p.end();
    };

    for (let l = 0; l < this.L; l++) {
      const b = this.bg[l];
      run(this.pNormFull, b.norm0, T);
      run(this.pMM512, b.q, T * d);
      run(this.pMM256, b.k, T * this.kvDim);
      run(this.pMM256, b.v, T * this.kvDim);
      run(this.pNormHeadQ, b.qNorm, T * dims.H);
      run(this.pNormHeadK, b.kNorm, T * dims.Hkv);
      run(this.pRopeQ, b.ropeQ, T * dims.H);
      run(this.pRopeK, b.ropeK, T * dims.Hkv);
      run(this.pAttn, b.attn, T * dims.H);
      run(this.pMM512, b.out, T * d);
      run(this.pGate, b.gate, T * d);
    }
    run(this.pNormFull, this.bgFinal, T);
    enc.copyBufferToBuffer(this.bufOut, 0, this.readback, 0, T * d * 4);
    device.queue.submit([enc.finish()]);

    await this.readback.mapAsync(GPUMapMode.READ, 0, T * d * 4);
    const out = new Float32Array(
      this.readback.getMappedRange(0, T * d * 4).slice(0),
    );
    this.readback.unmap();
    return { out, T };
  }
}
