/// <reference types="@webgpu/types" />

/**
 * WebGPU device acquisition. Returns a live GPUDevice or null when WebGPU is
 * unavailable (no `navigator.gpu`, no adapter, or device request throws) so the
 * caller can fall back to the pure-TS encoder. Never throws.
 */
export async function initGpu(): Promise<GPUDevice | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return null;
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    return await adapter.requestDevice();
  } catch {
    return null;
  }
}
