/// <reference types="@webgpu/types" />

/**
 * WGSL for the Needle encoder forward pass, f32 compute. Each shader mirrors the
 * matching TS routine in ../model.ts (encode()/attention()/zcrms()/applyRope()).
 * Model dims are baked in as literals at build time; only T (token count) and the
 * per-layer sigmoid gate flow in via uniforms. One naive invocation per output
 * element — encode is a few GFLOP, so dispatch/readback dominate, not compute.
 */

export interface Dims {
  d: number; // d_model
  H: number; // num_heads
  Hkv: number; // num_kv_heads
  hd: number; // head_dim
  half: number; // hd/2
  eps: number;
}

/** WGSL float literal (avoids locale/precision surprises from String()). */
const f = (x: number) => {
  const s = String(x);
  return s.includes(".") || s.includes("e") || s.includes("E") ? s : s + ".0";
};

/** y[T,OUT] = x[T,IN] @ w[IN,OUT] (Flax kernel layout, no bias). */
export const matmulWGSL = (IN: number, OUT: number) => `
struct P { T: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.T * ${OUT}u;
  if (idx >= total) { return; }
  let t = idx / ${OUT}u;
  let o = idx % ${OUT}u;
  var acc = 0.0;
  let xoff = t * ${IN}u;
  for (var i = 0u; i < ${IN}u; i = i + 1u) {
    acc = acc + x[xoff + i] * w[i * ${OUT}u + o];
  }
  y[idx] = acc;
}`;

/** ZCRMSNorm over rows of length DIM: y = (1+scale[d]) * x / sqrt(mean(x^2)+eps).
 * `rows` = T*ROWMUL (ROWMUL=1 for d_model rows, or H/Hkv for per-head norm).
 * In-place when x and y bind the same buffer is avoided; caller uses distinct
 * buffers for the full-dim norm and passes IN_PLACE=true for the per-head norm. */
export const zcrmsWGSL = (
  dims: Dims,
  DIM: number,
  rowMul: number,
  inPlace: boolean,
) => `
struct P { T: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, ${inPlace ? "read_write" : "read"}> x: array<f32>;
${inPlace ? "" : "@group(0) @binding(3) var<storage, read_write> y: array<f32>;"}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  let rows = p.T * ${rowMul}u;
  if (r >= rows) { return; }
  let off = r * ${DIM}u;
  var ss = 0.0;
  for (var d = 0u; d < ${DIM}u; d = d + 1u) { let v = x[off + d]; ss = ss + v * v; }
  let rms = sqrt(ss / ${f(DIM)} + ${f(dims.eps)});
  for (var d = 0u; d < ${DIM}u; d = d + 1u) {
    ${inPlace ? "x" : "y"}[off + d] = (1.0 + scale[d]) * x[off + d] / rms;
  }
}`;

/** Half-split RoPE in place over a [T, HC, hd] tensor, per head. */
export const ropeWGSL = (dims: Dims, HC: number) => `
struct P { T: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> cosb: array<f32>;
@group(0) @binding(2) var<storage, read> sinb: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.T * ${HC}u;
  if (idx >= total) { return; }
  let t = idx / ${HC}u;
  let base = idx * ${dims.hd}u;
  let coff = t * ${dims.half}u;
  for (var fr = 0u; fr < ${dims.half}u; fr = fr + 1u) {
    let x1 = x[base + fr];
    let x2 = x[base + ${dims.half}u + fr];
    let c = cosb[coff + fr];
    let s = sinb[coff + fr];
    x[base + fr] = x1 * c - x2 * s;
    x[base + ${dims.half}u + fr] = x2 * c + x1 * s;
  }
}`;

/** GQA attention, no mask (encoder is bidirectional over the exact T tokens).
 * One invocation per (query i, head h); softmax computed with a running max and
 * two extra passes over keys (T is small, registers can't hold all scores). */
export const attnWGSL = (dims: Dims) => {
  const R = dims.H / dims.Hkv;
  const scale = Math.sqrt(dims.hd);
  return `
struct P { T: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.T * ${dims.H}u;
  if (idx >= total) { return; }
  let i = idx / ${dims.H}u;
  let h = idx % ${dims.H}u;
  let kvh = h / ${R}u;
  let qbase = idx * ${dims.hd}u;
  var maxs = -1.0e30;
  for (var j = 0u; j < p.T; j = j + 1u) {
    let kbase = (j * ${dims.Hkv}u + kvh) * ${dims.hd}u;
    var dot = 0.0;
    for (var e = 0u; e < ${dims.hd}u; e = e + 1u) { dot = dot + q[qbase + e] * k[kbase + e]; }
    let sc = dot / ${f(scale)};
    if (sc > maxs) { maxs = sc; }
  }
  var sum = 0.0;
  for (var j = 0u; j < p.T; j = j + 1u) {
    let kbase = (j * ${dims.Hkv}u + kvh) * ${dims.hd}u;
    var dot = 0.0;
    for (var e = 0u; e < ${dims.hd}u; e = e + 1u) { dot = dot + q[qbase + e] * k[kbase + e]; }
    sum = sum + exp(dot / ${f(scale)} - maxs);
  }
  let obase = idx * ${dims.hd}u;
  for (var e = 0u; e < ${dims.hd}u; e = e + 1u) { outp[obase + e] = 0.0; }
  for (var j = 0u; j < p.T; j = j + 1u) {
    let kbase = (j * ${dims.Hkv}u + kvh) * ${dims.hd}u;
    var dot = 0.0;
    for (var e = 0u; e < ${dims.hd}u; e = e + 1u) { dot = dot + q[qbase + e] * k[kbase + e]; }
    let pw = exp(dot / ${f(scale)} - maxs) / sum;
    let vbase = (j * ${dims.Hkv}u + kvh) * ${dims.hd}u;
    for (var e = 0u; e < ${dims.hd}u; e = e + 1u) { outp[obase + e] = outp[obase + e] + pw * v[vbase + e]; }
  }
}`;
};

/** Sigmoid-gated residual, in place: x[i] = x[i] + gate * o[i], over T*d. */
export const gatedAddWGSL = (dims: Dims) => `
struct P { T: u32, gate: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> o: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.T * ${dims.d}u) { return; }
  x[idx] = x[idx] + p.gate * o[idx];
}`;
