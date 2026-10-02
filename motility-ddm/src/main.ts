// MotilityDDM dashboard: camera -> WebGPU DDM engine -> worker fit -> plots.

import shaderCode from './ddm.wgsl?raw';
import { DdmEngine } from './ddm-engine';
import { CameraManager, FrameLoop, FrameClock } from './camera';
import type { FrameEvent, LockReport } from './camera';
import { DDM } from './ddm_ref.js';
import type { Radial } from './ddm_ref.js';
import { analyzeMotility, prepare, modelColumn, swimmerKernel, GATES } from './motility_fit.js';
import type { MotilityResult, AnalysisOptions } from './motility_fit.js';
import { SCENARIOS, simulate } from '../test/sim.mjs';

// ---------------------------------------------------------------- DOM

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video');
const overlay = $<HTMLCanvasElement>('overlay');
const videoEmpty = $('videoEmpty');
const btnStart = $<HTMLButtonElement>('btnStart');
const btnLock = $<HTMLButtonElement>('btnLock');
const btnDemo = $<HTMLButtonElement>('btnDemo');
const pitchIn = $<HTMLInputElement>('pitch');
const geometryIn = $<HTMLSelectElement>('geometry');
const shuffleIn = $<HTMLInputElement>('shuffle');
const demoSel = $<HTMLSelectElement>('demoScenario');
const banner = $('banner');
const plotD = $<HTMLCanvasElement>('plotD');
const plotS = $<HTMLCanvasElement>('plotS');

// Lags reach 127 frames; below ~1.5x that the long-lag estimates rest on too
// few frame pairs and an early verdict could flip on screen.
const MIN_FRAMES = 192;
const ANALYSIS_PERIOD_MS = 1500;

type Mode = 'idle' | 'camera' | 'demo';
type IngestMode = 'external' | 'canvas';

const state = {
  mode: 'idle' as Mode,
  engine: null as DdmEngine | null,
  enginePromise: null as Promise<DdmEngine> | null,
  camera: new CameraManager(),
  loop: null as FrameLoop | null,
  clock: null as FrameClock | null,
  fps: 30,
  ingest: (new URLSearchParams(location.search).get('ingest') === 'canvas' ? 'canvas' : 'external') as IngestMode,
  lock: null as LockReport | null,
  busy: false,
  guard: 32,
  lastMs: 0,
  lastResult: null as MotilityResult | null,
  lastRadial: null as Radial | null,
  lastOpt: null as AnalysisOptions | null,
  stale: 0,
  demoTimer: 0 as ReturnType<typeof setTimeout> | 0,
  demoFrame: null as Uint8Array | null,
  demoDone: false,
  roiCanvas: null as OffscreenCanvas | null,
};

function showBanner(msg: string, kind: 'error' | 'info' = 'error'): void {
  banner.textContent = msg;
  banner.className = `banner show${kind === 'info' ? ' info' : ''}`;
}
function hideBanner(): void {
  banner.className = 'banner';
}

// ---------------------------------------------------------------- engine

async function ensureEngine(): Promise<DdmEngine> {
  if (state.engine) return state.engine;
  if (!state.enginePromise) {
    state.enginePromise = DdmEngine.create({
      shaderCode,
      onLost: (m) => void onDeviceLost(m),
      onError: (m) => console.warn('[gpu]', m),
    }).then(
      (e) => {
        state.engine = e;
        const a = e.adapterInfo;
        $('chipGpu').textContent = [a.vendor, a.architecture].filter(Boolean).join(' ') || 'ok';
        return e;
      },
      (err) => {
        state.enginePromise = null;
        throw err;
      },
    );
  }
  return state.enginePromise;
}

/**
 * A device lost right after video import means the driver's external-texture
 * path is broken. Fall back to canvas readback ingest (2D drawImage of the ROI
 * + ingestLuma) on a fresh device; the history restarts.
 */
async function onDeviceLost(msg: string): Promise<void> {
  state.engine = null;
  state.enginePromise = null;
  console.error(msg);
  if (state.mode === 'camera' && state.ingest === 'external') {
    state.ingest = 'canvas';
    showBanner(`${msg}. Switched to canvas-readback ingest (?ingest=canvas); history restarted.`, 'info');
    state.clock = new FrameClock(state.fps);
    try {
      await ensureEngine();
    } catch (e) {
      showBanner(`GPU unavailable after device loss: ${(e as Error).message}`);
    }
  } else if (state.mode !== 'idle') {
    showBanner(msg);
  }
}

function ingestViaCanvas(engine: DdmEngine, src: HTMLVideoElement, slot: number): boolean {
  const N = DDM.N;
  if (!state.roiCanvas) state.roiCanvas = new OffscreenCanvas(N, N);
  const ctx = state.roiCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return false;
  const x0 = Math.floor((src.videoWidth - N) / 2), y0 = Math.floor((src.videoHeight - N) / 2);
  ctx.drawImage(src, x0, y0, N, N, 0, 0, N, N); // 1:1, no resampling
  const px = ctx.getImageData(0, 0, N, N).data;
  const luma = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) luma[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
  return engine.ingestLuma(luma, slot);
}

// ---------------------------------------------------------------- camera

async function startCamera(): Promise<void> {
  stopDemo();
  hideBanner();
  btnStart.disabled = true;
  try {
    await ensureEngine();
    const settings = await state.camera.start(video);
    state.fps = settings.frameRate || 30;
    state.clock = new FrameClock(state.fps);
    state.mode = 'camera';
    state.lock = null;
    setLockBadge(null);
    videoEmpty.style.display = 'none';
    $('roiLabel').textContent = `ROI 256 × 256 of ${settings.width}×${settings.height}, native pixels`;
    state.loop?.stop();
    state.loop = new FrameLoop(video, onCameraFrame);
    state.loop.start();
    btnLock.disabled = false;
    btnStart.textContent = 'Restart Camera';
    drawOverlay();
  } catch (e) {
    showBanner(`Camera/GPU start failed: ${(e as Error).message}`);
  } finally {
    btnStart.disabled = false;
  }
}

function onCameraFrame(e: FrameEvent): void {
  const engine = state.engine, clock = state.clock;
  if (!engine || !clock || state.mode !== 'camera') return;
  const placed = clock.place(e.timestampMs, e.timeSource);
  if (!placed) return;
  let ok = false;
  try {
    if (state.ingest === 'external') {
      let src: VideoFrame | HTMLVideoElement = e.video;
      if (typeof VideoFrame !== 'undefined') {
        try {
          src = new VideoFrame(e.video, { timestamp: Math.round(e.timestampMs * 1000) });
        } catch {
          src = e.video;
        }
      }
      ok = engine.ingestFrame(src, placed.slot);
    } else {
      ok = ingestViaCanvas(engine, e.video, placed.slot);
    }
  } catch (err) {
    console.warn('ingest failed', err);
    ok = false;
  }
  if (!ok) clock.drop(placed.n);
}

async function lockCamera(): Promise<void> {
  btnLock.disabled = true;
  try {
    const r = await state.camera.lock();
    state.lock = r;
    setLockBadge(r);
    renderLockReport(r);
    // Locking changes the image statistics; restart the history.
    state.clock = new FrameClock(state.fps);
  } catch (e) {
    showBanner(`Lock failed: ${(e as Error).message}`);
  } finally {
    btnLock.disabled = false;
  }
}

function setLockBadge(r: LockReport | null): void {
  const el = $('lockBadge');
  if (state.mode === 'demo') {
    el.className = 'badge b-NONE';
    el.textContent = 'SYNTHETIC';
    return;
  }
  el.className = `badge b-${r ? r.badge : 'NONE'}`;
  el.textContent = r ? r.badge : 'NOT LOCKED';
}

function renderLockReport(r: LockReport): void {
  const rows: Array<[string, string]> = [
    ['exposure', r.exposure],
    ['iso', r.iso],
    ['focus', r.focus],
    ['white bal.', r.whiteBalance],
  ];
  const s = r.settings as Record<string, unknown>;
  const el = $('lockReport');
  el.innerHTML = '';
  for (const [k, v] of rows) {
    el.insertAdjacentHTML('beforeend', `<span>${k}</span><span class="v st-${v}">${v}</span>`);
  }
  const fmt = (x: unknown) => (typeof x === 'number' ? +x.toFixed(3) : String(x ?? '—'));
  el.insertAdjacentHTML('beforeend', `<span>readback</span><span class="v">t=${fmt(s.exposureTime)} iso=${fmt(s.iso)} f=${fmt(s.focusDistance)} K=${fmt(s.colorTemperature)}</span>`);
  if (r.errors.length) el.insertAdjacentHTML('beforeend', `<span>notes</span><span class="v">${r.errors.map(escapeHtml).join('; ')}</span>`);
  if (r.badge !== 'LOCKED') {
    showBanner('Exposure is not locked: measurements continue, but every verdict is reported as UNLOCKED_EXPOSURE (auto-exposure steps decorrelate the image and mimic motion).', 'info');
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] as string);
}

// ---------------------------------------------------------------- synthetic demo

function setupDemoSelect(): void {
  for (const sc of SCENARIOS) {
    const o = document.createElement('option');
    o.value = sc.name;
    o.textContent = `${sc.name} (expect ${sc.expect})`;
    demoSel.appendChild(o);
  }
}

function stopDemo(): void {
  if (state.demoTimer) clearTimeout(state.demoTimer);
  state.demoTimer = 0;
  state.demoFrame = null;
  if (state.mode === 'demo') state.mode = 'idle';
}

/**
 * Play a synthetic scenario from test/sim.mjs through the real GPU pipeline at
 * 30 fps (ingestLuma path). Pixel pitch, geometry and shuffle are set from the
 * scenario so the demo is reproducible on camera.
 */
async function runDemo(): Promise<void> {
  const sc = SCENARIOS.find((s) => s.name === demoSel.value) ?? SCENARIOS[0];
  btnDemo.disabled = true;
  hideBanner();
  try {
    const engine = await ensureEngine();
    state.loop?.stop();
    state.camera.stop();
    stopDemo();
    videoEmpty.textContent = `Rendering synthetic movie “${sc.name}”…`;
    videoEmpty.style.display = '';
    await new Promise((r) => setTimeout(r, 30));
    const movie = simulate(sc.sim);
    pitchIn.value = String(movie.dxUm);
    geometryIn.value = sc.geometry;
    shuffleIn.checked = !!sc.shuffle;
    state.fps = movie.fps;
    state.clock = new FrameClock(movie.fps);
    state.mode = 'demo';
    state.lastResult = null;
    setLockBadge(null);
    $('lockReport').innerHTML = `<span>scenario</span><span class="v">${escapeHtml(sc.description)}</span><span>expected</span><span class="v">${sc.expect}</span>`;
    $('roiLabel').textContent = `synthetic ${movie.N}×${movie.N}, ${movie.dxUm} µm/px`;
    videoEmpty.style.display = 'none';
    $('chipClock').textContent = 'synthetic';
    // Synthetic frames carry their own clock, so unlike the camera the demo
    // waits for GPU capacity instead of dropping frames (slow GPUs still get
    // every frame, just later than real time).
    let t = 0;
    const step = async () => {
      if (state.mode !== 'demo') return;
      if (t >= movie.T) {
        state.demoTimer = 0;
        state.demoDone = true;
        return;
      }
      const tStart = performance.now();
      if (movie.valid[t]) {
        const placed = state.clock!.place((t * 1000) / movie.fps, 'captureTime');
        if (placed) {
          await engine.waitForCapacity();
          if (state.mode !== 'demo') return;
          const f = movie.frames[t];
          const ok = engine.ingestLuma(Float32Array.from(f, (v) => v / 255), placed.slot);
          if (!ok) state.clock!.drop(placed.n);
          state.demoFrame = f;
          drawOverlay();
        }
      }
      t++;
      state.demoTimer = setTimeout(() => void step(), Math.max(0, 1000 / movie.fps - (performance.now() - tStart)));
    };
    state.demoDone = false;
    void step();
  } catch (e) {
    showBanner(`Demo failed: ${(e as Error).message}`);
    state.mode = 'idle';
  } finally {
    btnDemo.disabled = false;
  }
}

// ---------------------------------------------------------------- analysis loop

type FitReply = { id: number; result?: MotilityResult; error?: string };
let worker: Worker | null = null;
let fitId = 0;
const fitWaiters = new Map<number, (r: FitReply) => void>();
try {
  worker = new Worker(new URL('./motility_fit.js', import.meta.url), { type: 'module' });
  worker.onmessage = (ev: MessageEvent<FitReply>) => {
    const w = fitWaiters.get(ev.data.id);
    fitWaiters.delete(ev.data.id);
    w?.(ev.data);
  };
  worker.onerror = (e) => {
    console.warn('fit worker failed, fitting on the main thread', e.message);
    worker = null;
    for (const [id, w] of fitWaiters) w({ id, error: 'worker failed' });
    fitWaiters.clear();
  };
} catch {
  worker = null;
}

function runFit(radial: Radial, opt: AnalysisOptions): Promise<MotilityResult> {
  if (!worker) return Promise.resolve(analyzeMotility(radial, opt));
  const id = ++fitId;
  return new Promise((resolve, reject) => {
    fitWaiters.set(id, (r) => (r.result ? resolve(r.result) : r.error === 'worker failed' ? resolve(analyzeMotility(radial, opt)) : reject(new Error(r.error))));
    worker!.postMessage({ id, radial, opt });
  });
}

function currentOptions(): AnalysisOptions {
  const dx = parseFloat(pitchIn.value);
  return {
    dxUm: Number.isFinite(dx) && dx > 0 ? dx : 5,
    fps: state.fps,
    geometry: geometryIn.value === '2d' ? '2d' : '3d',
    exposureLocked: state.mode === 'demo' ? true : state.lock?.badge === 'LOCKED',
  };
}

async function analysisTick(): Promise<void> {
  updateChips();
  const engine = state.engine, clock = state.clock;
  if (state.busy || !engine || !clock || state.mode === 'idle') return;
  const valid = clock.validCount;
  if (valid < MIN_FRAMES) {
    $('verdictSub').textContent = `collecting history: ${valid}/${MIN_FRAMES} frames`;
    return;
  }
  state.busy = true;
  try {
    const guard = state.mode === 'demo' ? 0 : state.guard;
    const snap = clock.snapshot({ guard, shuffle: shuffleIn.checked });
    const t0 = performance.now();
    await engine.computeTemporal(snap.order, snap.mask);
    const radial = await engine.readRadial();
    const elapsedFrames = clock.newestN - snap.newestN;
    if (state.mode === 'camera' && elapsedFrames >= guard) {
      // Frames arriving during the analysis may have overwritten slots inside the
      // snapshot: discard this result and widen the guard.
      state.stale++;
      state.guard = Math.min(128, Math.ceil(elapsedFrames * 1.5) + 4);
      return;
    }
    const opt = currentOptions();
    const result = await runFit(radial, opt);
    state.lastMs = performance.now() - t0;
    state.lastResult = result;
    state.lastRadial = radial;
    state.lastOpt = opt;
    renderResult(result, snap.used);
    drawPlots();
  } catch (e) {
    console.error(e);
    showBanner(`Analysis failed: ${(e as Error).message}`);
  } finally {
    state.busy = false;
  }
}

// ---------------------------------------------------------------- rendering

const fmt = (x: number | undefined, d = 2) => (x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(d));

function renderResult(r: MotilityResult, used: number): void {
  const box = $('verdictBox');
  box.className = `verdict v-${r.verdict}`;
  $('verdict').textContent = r.verdict;
  const sub = [];
  if (r.verdict !== r.physicsVerdict) sub.push(`physics verdict: ${r.physicsVerdict} (not reportable)`);
  if (shuffleIn.checked) sub.push('Control 1: frame order shuffled');
  sub.push(`${used} frames`);
  if (state.mode === 'demo' && state.demoDone) sub.push('demo complete');
  $('verdictSub').textContent = sub.join(' · ');

  const unit = (v: string, u: string) => `${v} <span class="unit">${u}</span>`;
  $('cAlpha').innerHTML = fmt(r.alpha, 3);
  $('cAlphaNote').textContent = r.alphaResolved === undefined ? '' : r.alphaResolved ? `resolved in ${r.resolvedBins} q-bins` : 'swimmer decay not resolved';
  $('cVbar').innerHTML = unit(fmt(r.vbar, 1), 'µm/s');
  $('cSigma').innerHTML = unit(fmt(r.sigmaV, 1), 'µm/s');
  $('cZ').textContent = r.Z === undefined ? '' : `Schulz Z = ${fmt(r.Z, 1)}`;
  // Swimmer parameters are only a claim when the physics verdict is MOTILE.
  const claimed = r.physicsVerdict === 'MOTILE';
  document.querySelectorAll('.card.swim').forEach((c) => c.classList.toggle('dim', !claimed));
  $('cVbarNote').textContent = claimed ? 'Schulz mean' : 'fit value, not claimed';
  if (!claimed && r.alpha !== undefined) $('cAlphaNote').textContent = 'fit value, not claimed';
  const d = r.drift;
  $('cDrift').innerHTML = d ? unit(`${fmt(d.mag, 2)} ± ${fmt(d.sigma, 2)}`, 'µm/s') : '—';
  $('cDriftNote').textContent = d ? `${fmt(d.angleDeg, 0)}° (image frame, +y down)${d.significant ? '' : ' · n.s.'}` : '';
  $('cD').innerHTML = unit(fmt(r.Ddiff, 3), 'µm²/s');
  $('cBic').textContent = fmt(r.dBIC, 1);
  $('cBicNote').textContent = r.nEff ? `n_eff = ${fmt(r.nEff, 0)} of ${r.nData}` : '';

  const gl = $('gates');
  gl.innerHTML = '';
  for (const g of r.gates) {
    const v = typeof g.value === 'number' ? (Math.abs(g.value) >= 100 ? g.value.toFixed(0) : g.value.toFixed(3)) : String(g.value);
    const li = document.createElement('li');
    li.className = g.pass ? 'pass' : 'fail';
    li.innerHTML = `<span class="mark">${g.pass ? '✓' : '✗'}</span><span>${g.id} ${escapeHtml(g.name)}</span><span class="num">${v}</span>`;
    li.title = `threshold: ${g.threshold}`;
    gl.appendChild(li);
  }
  renderMeta(r);
}

function renderMeta(r: MotilityResult | null): void {
  const c = state.clock, e = state.engine;
  const rows: Array<[string, string]> = [
    ['μ (scaling)', fmt(r?.mu, 2)],
    ['TSI', fmt(r?.TSI, 3)],
    ['missed frames', String(c?.missed ?? 0)],
    ['GPU throttled', String(e?.throttled ?? 0)],
    ['stale analyses', String(state.stale)],
    ['analysis', `${fmt(state.lastMs, 0)} ms`],
    ['ingest path', state.mode === 'demo' ? 'ingestLuma (synthetic)' : state.ingest === 'external' ? 'importExternalTexture' : 'canvas readback'],
  ];
  $('meta').innerHTML = rows.map(([k, v]) => `<span>${k}</span><span class="v">${v}</span>`).join('');
}

function updateChips(): void {
  const c = state.clock;
  if (state.mode === 'camera' && c) {
    $('chipClock').textContent = c.timeSource ?? '—';
    const dt = c.medianDtMs;
    $('chipFps').textContent = Number.isFinite(dt) ? (1000 / dt).toFixed(1) : '—';
  }
  $('chipHist').textContent = `${c ? c.validCount : 0}/256`;
  if (!state.lastResult) renderMeta(null);
}

/** Size a canvas backing store to its CSS box at device pixel ratio. */
function fitCanvas(cv: HTMLCanvasElement): CanvasRenderingContext2D {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) {
    cv.width = w;
    cv.height = h;
  }
  const ctx = cv.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return ctx;
}

function drawOverlay(): void {
  const ctx = fitCanvas(overlay);
  const W = overlay.clientWidth, H = overlay.clientHeight;
  let vw = video.videoWidth, vh = video.videoHeight;
  if (state.mode === 'demo' && state.demoFrame) {
    const N = DDM.N;
    const img = new ImageData(N, N);
    const f = state.demoFrame;
    for (let i = 0; i < N * N; i++) {
      const v = f[i];
      img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v;
      img.data[4 * i + 3] = 255;
    }
    const tmp = state.roiCanvas ?? (state.roiCanvas = new OffscreenCanvas(N, N));
    tmp.getContext('2d')!.putImageData(img, 0, 0);
    const s = Math.min(W, H);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, (W - s) / 2, (H - s) / 2, s, s);
    vw = vh = N;
  }
  if (!vw || !vh) return;
  const scale = Math.min(W / vw, H / vh);
  const ox = (W - vw * scale) / 2, oy = (H - vh * scale) / 2;
  const N = DDM.N;
  const x = ox + Math.floor((vw - N) / 2) * scale, y = oy + Math.floor((vh - N) / 2) * scale;
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x, y, N * scale, N * scale);
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(11,15,20,0.75)';
  ctx.fillRect(x, y - 18, 128, 16);
  ctx.fillStyle = '#a5f3fc';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText('ROI 256×256 native', x + 4, y - 6);
}

interface Axes {
  x0: number; y0: number; w: number; h: number;
  xmin: number; xmax: number; ymin: number; ymax: number;
  logx: boolean; logy: boolean;
}

function axes(ctx: CanvasRenderingContext2D, cv: HTMLCanvasElement, a: Omit<Axes, 'x0' | 'y0' | 'w' | 'h'>, xlabel: string, ylabel: string): Axes & { X: (v: number) => number; Y: (v: number) => number } {
  const W = cv.clientWidth, H = cv.clientHeight;
  const ax = { x0: 58, y0: 10, w: W - 70, h: H - 44, ...a };
  const tx = (v: number) => (a.logx ? Math.log10(v) : v), ty = (v: number) => (a.logy ? Math.log10(v) : v);
  const X = (v: number) => ax.x0 + ((tx(v) - tx(a.xmin)) / (tx(a.xmax) - tx(a.xmin))) * ax.w;
  const Y = (v: number) => ax.y0 + ax.h - ((ty(v) - ty(a.ymin)) / (ty(a.ymax) - ty(a.ymin))) * ax.h;
  ctx.strokeStyle = '#243246';
  ctx.fillStyle = '#8193a8';
  ctx.font = '10px ui-monospace, monospace';
  ctx.lineWidth = 1;
  const ticks = (lo: number, hi: number, log: boolean) => {
    const out: number[] = [];
    if (log) {
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) {
        const v = m * 10 ** e;
        if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
      }
    } else {
      const step = niceStep((hi - lo) / 5);
      for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    }
    return out;
  };
  for (const v of ticks(a.xmin, a.xmax, a.logx)) {
    const x = X(v);
    ctx.beginPath();
    ctx.moveTo(x, ax.y0);
    ctx.lineTo(x, ax.y0 + ax.h);
    ctx.stroke();
    ctx.fillText(String(+v.toPrecision(3)), x - 8, ax.y0 + ax.h + 13);
  }
  ctx.textAlign = 'right';
  for (const v of ticks(a.ymin, a.ymax, a.logy)) {
    const y = Y(v);
    ctx.beginPath();
    ctx.moveTo(ax.x0, y);
    ctx.lineTo(ax.x0 + ax.w, y);
    ctx.stroke();
    ctx.fillText(String(+v.toPrecision(3)), ax.x0 - 4, y + 3);
  }
  ctx.textAlign = 'left';
  ctx.fillStyle = '#a7b6c8';
  ctx.fillText(xlabel, ax.x0 + ax.w / 2 - 30, H - 6);
  ctx.save();
  ctx.translate(10, ax.y0 + ax.h / 2 + 30);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0);
  ctx.restore();
  return { ...ax, X, Y };
}

function niceStep(x: number): number {
  const e = Math.floor(Math.log10(x));
  const f = x / 10 ** e;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * 10 ** e;
}

/** q-bin colour: blue (low q) to amber (high q). */
const binColor = (i: number, n: number, alpha = 1) => `hsla(${200 - (170 * i) / Math.max(1, n - 1)}, 85%, 60%, ${alpha})`;

function drawPlots(): void {
  const r = state.lastResult, radial = state.lastRadial, opt = state.lastOpt;
  const ctxD = fitCanvas(plotD);
  const ctxS = fitCanvas(plotS);
  if (!r || !radial || !opt || !r.fit || !r.bins) return;
  const L = radial.L, dt = 1 / opt.fps;
  const best = (r.dBIC ?? 0) > 0 ? r.fit.swimmer : r.fit.brownian;
  const swim = (r.dBIC ?? 0) > 0;
  // normalized decorrelation (D - B) / A per bin, data and model
  const A = axes(ctxD, plotD, { xmin: dt, xmax: (L - 1) * dt, ymin: -0.05, ymax: 1.15, logx: true, logy: false }, 'lag τ (s)', '(D − B) / A');
  const pr = prepare(radial, opt);
  const kernel = swimmerKernel(opt.geometry ?? '3d');
  const nb = pr.bins.length;
  let shown = 0;
  pr.bins.forEach((bin, i) => {
    const amp = best.amps[i];
    // only bins whose dynamic signal clearly exceeds the camera-noise floor
    if (!amp || !(amp.A > 0) || amp.A / (amp.A + amp.B) < 0.25) return;
    shown++;
    ctxD.strokeStyle = binColor(i, nb, 0.85);
    ctxD.lineWidth = 1.2;
    ctxD.beginPath();
    for (let t = 1; t < L; t++) {
      const v = (radial.D[bin.b * L + t] - amp.B) / amp.A;
      const x = A.X(t * dt), y = A.Y(Math.max(-0.05, Math.min(1.15, v)));
      if (t === 1) ctxD.moveTo(x, y);
      else ctxD.lineTo(x, y);
    }
    ctxD.stroke();
    if (i % 4 === 0) {
      const col = new Float64Array(pr.tau.length);
      modelColumn(pr, bin, best, kernel, swim, col);
      ctxD.setLineDash([3, 3]);
      ctxD.strokeStyle = '#e2e8f0';
      ctxD.lineWidth = 1;
      ctxD.beginPath();
      pr.tau.forEach((tau, j) => (j ? ctxD.lineTo(A.X(tau), A.Y(col[j])) : ctxD.moveTo(A.X(tau), A.Y(col[j]))));
      ctxD.stroke();
      ctxD.setLineDash([]);
    }
  });
  const legend = `${shown}/${nb} q-bins above noise, q ${fmt(pr.bins[0]?.q, 3)}→${fmt(pr.bins[nb - 1]?.q, 3)} µm⁻¹ blue→amber · dashed: ${swim ? 'swimmer' : 'Brownian'} model`;
  ctxD.font = '10px ui-monospace, monospace';
  const lw = ctxD.measureText(legend).width + 10;
  ctxD.fillStyle = 'rgba(11,17,24,0.85)';
  ctxD.fillRect(A.x0 + 4, A.y0 + 2, lw, 16);
  ctxD.fillStyle = '#a7b6c8';
  ctxD.fillText(legend, A.x0 + 9, A.y0 + 13);

  // scaling exponent
  const sc = r.scaling;
  if (!sc || sc.q.length < 2) {
    ctxS.fillStyle = '#8193a8';
    ctxS.font = '12px system-ui';
    ctxS.fillText('τ½ not resolved in enough q-bins', 16, 24);
    return;
  }
  const qmin = Math.min(...sc.q) / 1.3, qmax = Math.max(...sc.q) * 1.3;
  const tmin = Math.min(...sc.tauHalf) / 2, tmax = Math.max(...sc.tauHalf) * 2;
  const S = axes(ctxS, plotS, { xmin: qmin, xmax: qmax, ymin: tmin, ymax: tmax, logx: true, logy: true }, 'q (µm⁻¹)', 'τ½ (s)');
  const lx = sc.q.map(Math.log), ly = sc.tauHalf.map(Math.log);
  const mx = lx.reduce((a, b) => a + b) / lx.length, my = ly.reduce((a, b) => a + b) / ly.length;
  ctxS.save();
  ctxS.beginPath();
  ctxS.rect(S.x0, S.y0, S.w, S.h);
  ctxS.clip();
  const line = (mu: number, color: string, dash: number[]) => {
    ctxS.strokeStyle = color;
    ctxS.lineWidth = 1.4;
    ctxS.setLineDash(dash);
    ctxS.beginPath();
    const y = (q: number) => Math.exp(my - mu * (Math.log(q) - mx));
    ctxS.moveTo(S.X(qmin), S.Y(y(qmin)));
    ctxS.lineTo(S.X(qmax), S.Y(y(qmax)));
    ctxS.stroke();
    ctxS.setLineDash([]);
  };
  line(1, 'rgba(52,211,153,0.7)', [5, 4]);
  line(2, 'rgba(96,165,250,0.7)', [2, 4]);
  if (Number.isFinite(sc.mu)) line(sc.mu, '#fbbf24', []);
  ctxS.restore();
  // legend (top right)
  const items: Array<[string, string]> = [
    ['#fbbf24', `fit μ = ${fmt(sc.mu, 2)}`],
    ['rgba(52,211,153,0.9)', 'μ = 1 ballistic swimming'],
    ['rgba(96,165,250,0.9)', 'μ = 2 diffusion'],
  ];
  ctxS.font = '10px ui-monospace, monospace';
  const lx0 = S.x0 + S.w - 170;
  ctxS.fillStyle = 'rgba(11,17,24,0.85)';
  ctxS.fillRect(lx0 - 6, S.y0 + 2, 172, 46);
  items.forEach(([c, t], k) => {
    ctxS.fillStyle = c;
    ctxS.fillRect(lx0, S.y0 + 9 + 14 * k, 14, 3);
    ctxS.fillStyle = '#cbd5e1';
    ctxS.fillText(t, lx0 + 20, S.y0 + 13 + 14 * k);
  });
  sc.q.forEach((q, i) => {
    ctxS.fillStyle = '#e2e8f0';
    ctxS.beginPath();
    ctxS.arc(S.X(q), S.Y(sc.tauHalf[i]), 3, 0, 2 * Math.PI);
    ctxS.fill();
  });
}

// ---------------------------------------------------------------- wiring

setupDemoSelect();
btnStart.addEventListener('click', () => void startCamera());
btnLock.addEventListener('click', () => void lockCamera());
btnDemo.addEventListener('click', () => void runDemo());
for (const el of [pitchIn, geometryIn]) {
  el.addEventListener('change', () => {
    // Refit the last radial data with the new optics/geometry immediately.
    const radial = state.lastRadial;
    if (!radial || state.busy) return;
    const opt = currentOptions();
    void runFit(radial, opt).then((r) => {
      state.lastResult = r;
      state.lastOpt = opt;
      renderResult(r, state.clock?.validCount ?? 0);
      drawPlots();
    });
  });
}
video.addEventListener('loadedmetadata', drawOverlay);
window.addEventListener('resize', () => {
  drawOverlay();
  drawPlots();
});
setInterval(() => void analysisTick(), ANALYSIS_PERIOD_MS);
renderMeta(null);
$('hint').textContent += ` Gates: TSI ≥ ${GATES.TSI_MIN}, flow |V| ≤ ${GATES.FLOW_MIN_UM_S} µm/s or ≤ ${GATES.FLOW_SIGMA}σ, ΔBIC > ${GATES.DBIC_MIN}, α > ${GATES.ALPHA_MIN}, exposure locked.`;
if (!('gpu' in navigator)) showBanner('WebGPU is not available in this browser. Use Chrome 121+ on Android 12+, Windows or macOS.');

// Expose for automation (screen-recording scripts, smoke tests).
(window as unknown as { motilityDDM: unknown }).motilityDDM = { state, runDemo, startCamera, lockCamera };
