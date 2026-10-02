// WebGPU engine for MotilityDDM: resource lifetime, dispatch and readback.
//
// Lifetime rules (each one fixes a failure mode seen on real drivers):
//  * Readback goes through ONE persistent MAP_READ staging buffer, reused for
//    every read and never destroyed per call. Creating and destroying a staging
//    buffer per read raced mapAsync on some drivers ("external Instance
//    reference no longer exists").
//  * A VideoFrame imported with importExternalTexture stays open until the
//    queue.onSubmittedWorkDone() promise of the submit that sampled it
//    resolves; only then is close() called.
//  * Frames are dropped (not queued) while more than MAX_PENDING submits are
//    unresolved, so the camera/compositor is never starved by our backlog.
//  * The temporal pass is split into 8 dispatches of 4128 workgroups, each its
//    own submit, and the engine waits for each chunk and yields the main thread
//    (setTimeout 0) before the next, leaving gaps for the compositor.

import { DDM, makeBins, temporalMeta, gpuTwiddles, radialFromGpu, TEMPORAL_META } from './ddm_ref.js';
import { blackmanHarris } from './fft.js';
import type { Radial, Bins } from './ddm_ref.js';

export type FrameSource = VideoFrame | HTMLVideoElement;

export interface EngineOptions {
  /** WGSL source of src/ddm.wgsl (Vite: `?raw` import; harness: fetched). */
  shaderCode: string;
  /** Called once if the GPU device is lost. */
  onLost?: (message: string) => void;
  /** Called for uncaptured GPU validation/OOM errors. */
  onError?: (message: string) => void;
  powerPreference?: GPUPowerPreference;
}

/** Byte sizes of the large buffers, exported for docs and tests. */
export const SIZES = Object.freeze({
  frame: DDM.N * DDM.N * 4,
  rows: DDM.N * DDM.NH * 8,
  hist: DDM.MODES * DDM.T * 8, // 67,633,152 B
  outD: DDM.MODES * DDM.L * 4,
  outC: DDM.MODES * DDM.L * 8,
  outP: DDM.MODES * 4,
  radOut: (DDM.NB * DDM.L + DDM.NB) * 4,
  radDrift: DDM.NB * DDM.L * DDM.DRIFT_STRIDE * 4,
  staging: 8 * 1024 * 1024,
});

const MAX_PENDING = 2;
const WORKGROUP_BYTES = 10240;

/** Limits the kernels need; all are at or below the WebGPU defaults. */
const REQUIRED_LIMITS: Array<[keyof GPUSupportedLimits, number]> = [
  ['maxStorageBufferBindingSize', SIZES.hist],
  ['maxBufferSize', SIZES.hist],
  ['maxComputeWorkgroupStorageSize', WORKGROUP_BYTES],
  ['maxComputeInvocationsPerWorkgroup', 256],
  ['maxComputeWorkgroupSizeX', 256],
  ['maxStorageBuffersPerShaderStage', 7],
  ['maxComputeWorkgroupsPerDimension', DDM.MODES / DDM.TEMPORAL_CHUNKS],
];

const nextTask = () => new Promise<void>((r) => setTimeout(r, 0));

export class DdmEngine {
  readonly device: GPUDevice;
  /** Held for the engine's lifetime: if the adapter is garbage collected,
   * Chromium can tear down the Dawn instance under pending futures and reject
   * them with "A valid external Instance reference no longer exists". */
  readonly adapter: GPUAdapter;
  readonly adapterInfo: { vendor: string; architecture: string; description: string };
  readonly bins: Bins;

  private pIngest!: GPUComputePipeline;
  private pReduce!: GPUComputePipeline;
  private pRows!: GPUComputePipeline;
  private pCols!: GPUComputePipeline;
  private pTemporal!: GPUComputePipeline;
  private pRadial!: GPUComputePipeline;

  private bFrame!: GPUBuffer;
  private bStats!: GPUBuffer;
  private bWin!: GPUBuffer;
  private bTw!: GPUBuffer;
  private bRows!: GPUBuffer;
  private bHist!: GPUBuffer;
  private bPf!: GPUBuffer;
  private bPt!: GPUBuffer;
  private bMeta!: GPUBuffer;
  private bOutD!: GPUBuffer;
  private bOutC!: GPUBuffer;
  private bOutP!: GPUBuffer;
  private bBinOff!: GPUBuffer;
  private bBinModes!: GPUBuffer;
  private bRadOut!: GPUBuffer;
  private bRadDrift!: GPUBuffer;
  private bStaging!: GPUBuffer;

  private gReduce!: GPUBindGroup;
  private gRows!: GPUBindGroup;
  private gCols!: GPUBindGroup;
  private gTemporal!: GPUBindGroup;
  private gRadial!: GPUBindGroup;

  private pending = 0;
  private readChain: Promise<unknown> = Promise.resolve();
  private temporalBusy = false;
  private destroyed = false;
  /** Frames dropped because the GPU queue was saturated. */
  throttled = 0;

  private constructor(adapter: GPUAdapter, device: GPUDevice, info: GPUAdapterInfo | undefined, bins: Bins) {
    this.adapter = adapter;
    this.device = device;
    this.bins = bins;
    this.adapterInfo = {
      vendor: info?.vendor ?? '',
      architecture: info?.architecture ?? '',
      description: info?.description ?? '',
    };
  }

  /** Number of submits whose onSubmittedWorkDone has not resolved yet. */
  get pendingSubmits(): number {
    return this.pending;
  }

  static async create(opts: EngineOptions): Promise<DdmEngine> {
    if (!('gpu' in navigator) || !navigator.gpu) {
      throw new Error('WebGPU is not available in this browser (navigator.gpu missing). Use Chrome 121+ on Android 12+, Windows or macOS.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: opts.powerPreference ?? 'high-performance' });
    if (!adapter) throw new Error('WebGPU is not available: requestAdapter() returned null.');
    for (const [name, need] of REQUIRED_LIMITS) {
      const have = adapter.limits[name] as number;
      if (have < need) throw new Error(`GPU limit ${name} = ${have}, needs ${need}`);
    }
    const requiredLimits: Record<string, number> = {};
    for (const [name, need] of REQUIRED_LIMITS) requiredLimits[name] = need;
    const device = await adapter.requestDevice({ requiredLimits });
    const engine = new DdmEngine(adapter, device, adapter.info, makeBins());
    device.lost.then((info) => {
      if (!engine.destroyed) opts.onLost?.(`GPU device lost (${info.reason}): ${info.message}`);
    });
    device.addEventListener('uncapturederror', (ev) => {
      const msg = (ev as GPUUncapturedErrorEvent).error.message;
      console.error('[ddm-engine] uncaptured GPU error:', msg);
      opts.onError?.(msg);
    });
    await engine.init(opts.shaderCode);
    return engine;
  }

  private async init(code: string): Promise<void> {
    const d = this.device;
    const module = d.createShaderModule({ label: 'ddm.wgsl', code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) {
      const m = errors[0];
      throw new Error(`WGSL compilation failed: line ${m.lineNum}:${m.linePos} ${m.message}`);
    }
    const mk = (entryPoint: string) =>
      d.createComputePipelineAsync({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } }).catch((e: unknown) => {
        throw new Error(`Pipeline creation failed for ${entryPoint}: ${(e as Error).message ?? e}`);
      });
    [this.pIngest, this.pReduce, this.pRows, this.pCols, this.pTemporal, this.pRadial] = await Promise.all([
      mk('ingest'),
      mk('reduce_mean'),
      mk('fft_rows'),
      mk('fft_cols'),
      mk('temporal'),
      mk('radial'),
    ]);

    const S = GPUBufferUsage.STORAGE;
    const buf = (label: string, size: number, usage: number) => d.createBuffer({ label, size, usage });
    this.bFrame = buf('frame', SIZES.frame, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.bStats = buf('stats', 16, S | GPUBufferUsage.COPY_SRC);
    this.bWin = buf('win', DDM.N * 4, S | GPUBufferUsage.COPY_DST);
    this.bTw = buf('tw', 256 * 8, S | GPUBufferUsage.COPY_DST);
    this.bRows = buf('rows', SIZES.rows, S | GPUBufferUsage.COPY_SRC);
    this.bHist = buf('hist', SIZES.hist, S | GPUBufferUsage.COPY_SRC);
    this.bPf = buf('frameParams', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.bPt = buf('temporalParams', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.bMeta = buf('temporalMeta', TEMPORAL_META.BYTES, S | GPUBufferUsage.COPY_DST);
    this.bOutD = buf('outD', SIZES.outD, S | GPUBufferUsage.COPY_SRC);
    this.bOutC = buf('outC', SIZES.outC, S | GPUBufferUsage.COPY_SRC);
    this.bOutP = buf('outP', SIZES.outP, S | GPUBufferUsage.COPY_SRC);
    this.bBinOff = buf('binOff', this.bins.binOffsets.byteLength, S | GPUBufferUsage.COPY_DST);
    this.bBinModes = buf('binModes', Math.max(16, this.bins.binModes.byteLength), S | GPUBufferUsage.COPY_DST);
    this.bRadOut = buf('radOut', SIZES.radOut, S | GPUBufferUsage.COPY_SRC);
    this.bRadDrift = buf('radDrift', SIZES.radDrift, S | GPUBufferUsage.COPY_SRC);
    this.bStaging = buf('staging (persistent)', SIZES.staging, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    const q = d.queue;
    q.writeBuffer(this.bWin, 0, Float32Array.from(blackmanHarris(DDM.N)));
    q.writeBuffer(this.bTw, 0, gpuTwiddles());
    q.writeBuffer(this.bBinOff, 0, this.bins.binOffsets);
    q.writeBuffer(this.bBinModes, 0, this.bins.binModes);

    const bg = (p: GPUComputePipeline, entries: Array<[number, GPUBuffer]>) =>
      d.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      });
    this.gReduce = bg(this.pReduce, [[1, this.bFrame], [2, this.bStats]]);
    this.gRows = bg(this.pRows, [[1, this.bFrame], [2, this.bStats], [3, this.bWin], [4, this.bTw], [5, this.bRows]]);
    this.gCols = bg(this.pCols, [[4, this.bTw], [5, this.bRows], [6, this.bHist], [7, this.bPf]]);
    this.gTemporal = bg(this.pTemporal, [
      [4, this.bTw], [6, this.bHist], [8, this.bPt], [9, this.bMeta], [10, this.bOutD], [11, this.bOutC], [12, this.bOutP],
    ]);
    this.gRadial = bg(this.pRadial, [
      [10, this.bOutD], [11, this.bOutC], [12, this.bOutP], [13, this.bBinOff], [14, this.bBinModes], [15, this.bRadOut], [16, this.bRadDrift],
    ]);
    await q.onSubmittedWorkDone();
  }

  /** True while the queue is saturated; frames offered now would be dropped. */
  get saturated(): boolean {
    return this.pending > MAX_PENDING;
  }

  /** Resolve once the number of unresolved submits is at most MAX_PENDING. */
  async waitForCapacity(): Promise<void> {
    while (this.pending > MAX_PENDING) await this.device.queue.onSubmittedWorkDone();
  }

  private track(onDone?: () => void): void {
    this.pending++;
    this.device.queue.onSubmittedWorkDone().then(
      () => {
        this.pending--;
        onDone?.();
      },
      () => {
        this.pending--;
        onDone?.();
      },
    );
  }

  private encodeSpatial(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.pReduce);
    pass.setBindGroup(0, this.gReduce);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pRows);
    pass.setBindGroup(0, this.gRows);
    pass.dispatchWorkgroups(DDM.N);
    pass.setPipeline(this.pCols);
    pass.setBindGroup(0, this.gCols);
    pass.dispatchWorkgroups(DDM.NH);
  }

  /**
   * Transform one camera frame into history slot `slot` (0..255). The centred
   * 256x256 ROI is read at native resolution. A VideoFrame is closed by the
   * engine after the GPU has finished with it, also when the frame is dropped.
   * @returns false if the frame was dropped because the queue is saturated.
   */
  ingestFrame(source: FrameSource, slot: number): boolean {
    const isVF = typeof VideoFrame !== 'undefined' && source instanceof VideoFrame;
    const close = () => {
      if (isVF) (source as VideoFrame).close();
    };
    if (this.destroyed || this.pending > MAX_PENDING) {
      this.throttled++;
      close();
      return false;
    }
    const w = isVF ? (source as VideoFrame).displayWidth : (source as HTMLVideoElement).videoWidth;
    const h = isVF ? (source as VideoFrame).displayHeight : (source as HTMLVideoElement).videoHeight;
    if (w < DDM.N || h < DDM.N) {
      close();
      throw new Error(`Frame ${w}x${h} is smaller than the ${DDM.N}x${DDM.N} ROI`);
    }
    let ext: GPUExternalTexture;
    try {
      ext = this.device.importExternalTexture({ source });
    } catch (e) {
      close();
      throw e;
    }
    const x0 = Math.floor((w - DDM.N) / 2), y0 = Math.floor((h - DDM.N) / 2);
    this.device.queue.writeBuffer(this.bPf, 0, new Uint32Array([x0, y0, slot % DDM.T, 0]));
    const gIngest = this.device.createBindGroup({
      layout: this.pIngest.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: ext },
        { binding: 1, resource: { buffer: this.bFrame } },
        { binding: 7, resource: { buffer: this.bPf } },
      ],
    });
    const enc = this.device.createCommandEncoder({ label: `frame slot ${slot}` });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pIngest);
    pass.setBindGroup(0, gIngest);
    pass.dispatchWorkgroups(DDM.N / 16, DDM.N / 16);
    this.encodeSpatial(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    // The VideoFrame must outlive the GPU work that samples it.
    this.track(close);
    return true;
  }

  /**
   * Transform a CPU-side luma frame (N*N floats, row-major) into slot `slot`.
   * Used by the synthetic demo and as the harness fallback (?src=buffer).
   */
  ingestLuma(luma: Float32Array, slot: number): boolean {
    if (this.destroyed || this.pending > MAX_PENDING) {
      this.throttled++;
      return false;
    }
    if (luma.length !== DDM.N * DDM.N) throw new Error(`ingestLuma expects ${DDM.N * DDM.N} values`);
    const q = this.device.queue;
    q.writeBuffer(this.bFrame, 0, luma);
    q.writeBuffer(this.bPf, 0, new Uint32Array([0, 0, slot % DDM.T, 0]));
    const enc = this.device.createCommandEncoder({ label: `luma slot ${slot}` });
    const pass = enc.beginComputePass();
    this.encodeSpatial(pass);
    pass.end();
    q.submit([enc.finish()]);
    this.track();
    return true;
  }

  /**
   * Temporal + radial analysis of the current history.
   * @param order order[j] = ring slot holding time index j (length T)
   * @param mask  mask[j] = 1 if time index j holds a valid frame
   */
  async computeTemporal(order: Uint32Array, mask: Float32Array, opts: { yieldBetweenChunks?: boolean } = {}): Promise<void> {
    if (this.temporalBusy) throw new Error('computeTemporal is already running');
    this.temporalBusy = true;
    try {
      const q = this.device.queue;
      q.writeBuffer(this.bMeta, 0, temporalMeta(order, mask));
      const per = DDM.MODES / DDM.TEMPORAL_CHUNKS; // 4128
      for (let c = 0; c < DDM.TEMPORAL_CHUNKS; c++) {
        q.writeBuffer(this.bPt, 0, new Uint32Array([c * per, 0, 0, 0]));
        const enc = this.device.createCommandEncoder({ label: `temporal chunk ${c}` });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pTemporal);
        pass.setBindGroup(0, this.gTemporal);
        pass.dispatchWorkgroups(per);
        pass.end();
        q.submit([enc.finish()]);
        this.track();
        if (opts.yieldBetweenChunks !== false) {
          await q.onSubmittedWorkDone();
          await nextTask();
        }
      }
      const enc = this.device.createCommandEncoder({ label: 'radial' });
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pRadial);
      pass.setBindGroup(0, this.gRadial);
      pass.dispatchWorkgroups(DDM.NB);
      pass.end();
      q.submit([enc.finish()]);
      this.track();
      await q.onSubmittedWorkDone();
    } finally {
      this.temporalBusy = false;
    }
  }

  /**
   * Read `size` bytes of `src` at `offset` through the persistent staging
   * buffer (chunked, serialized). Never creates or destroys buffers.
   */
  read(src: GPUBuffer, offset: number, size: number): Promise<ArrayBuffer> {
    const job = this.readChain.then(async () => {
      const out = new Uint8Array(size);
      for (let done = 0; done < size; ) {
        const n = Math.min(SIZES.staging, size - done);
        const enc = this.device.createCommandEncoder({ label: 'readback' });
        enc.copyBufferToBuffer(src, offset + done, this.bStaging, 0, n);
        this.device.queue.submit([enc.finish()]);
        await this.bStaging.mapAsync(GPUMapMode.READ, 0, n);
        out.set(new Uint8Array(this.bStaging.getMappedRange(0, n)), done);
        this.bStaging.unmap();
        done += n;
      }
      return out.buffer;
    });
    this.readChain = job.catch(() => undefined);
    return job;
  }

  /** Radial reduction of the last computeTemporal, as a motility_fit Radial. */
  async readRadial(): Promise<Radial> {
    const radOut = new Float32Array(await this.read(this.bRadOut, 0, SIZES.radOut));
    const radDrift = new Float32Array(await this.read(this.bRadDrift, 0, SIZES.radDrift));
    return radialFromGpu(this.bins, radOut, radDrift);
  }

  /** Full spectral history (harness): hist[(mode*T + slot)*2 + {0,1}]. */
  async readHistory(): Promise<Float32Array> {
    return new Float32Array(await this.read(this.bHist, 0, SIZES.hist));
  }

  /** Full per-mode temporal outputs (harness). */
  async readTemporalFull(): Promise<{ D: Float32Array; C: Float32Array; P: Float32Array }> {
    const D = new Float32Array(await this.read(this.bOutD, 0, SIZES.outD));
    const C = new Float32Array(await this.read(this.bOutC, 0, SIZES.outC));
    const P = new Float32Array(await this.read(this.bOutP, 0, SIZES.outP));
    return { D, C, P };
  }

  /** Last frame's mean luma (debug): hi + lo parts. */
  async readMean(): Promise<number> {
    const s = new Float32Array(await this.read(this.bStats, 0, 16));
    return s[0] + s[1];
  }

  /** Resolve when every submitted command buffer has finished. */
  idle(): Promise<undefined> {
    return this.device.queue.onSubmittedWorkDone();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.device.destroy();
  }
}
