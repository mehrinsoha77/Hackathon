// Real camera + WebGPU smoke test (Command 3, run on the target phone).
//
// Expected console sequence (all within ~25 s):
//   step: getUserMedia
//   step: started {...settings}
//   step: locked {...}
//   step: engine ready
//   step: frame 1 t=...ms src=captureTime
//   step: frame 30 ...   step: frame 60 ...
//   step: temporal started (stress: 8 chunks while frames keep arriving)
//   step: loop done frames=>=71 slots=40
//   step: queue idle
//   RESULT {"lock":{...},"timeSource":"captureTime","frames":72,"slots":40,"missed":<=4,"dtMs":~33.3}
//
// Phase 1 ingests the first 40 frames into the ring through
// VideoFrame -> importExternalTexture. Phase 2 runs the full chunked temporal
// pass while frames keep being counted, which is the live-app worst case: if
// GPU work starves the compositor, frames stop arriving here.
//
// Bisection switch ?ingest=external (default) | canvas | none:
//   canvas  2D drawImage of the ROI + ingestLuma (no video import into WebGPU)
//   none    camera loop only, no GPU ingest (isolates rVFC/clock behaviour)

import { DdmEngine } from '../../src/ddm-engine';
import { CameraManager, FrameLoop, FrameClock } from '../../src/camera';
import type { FrameEvent, TimeSource } from '../../src/camera';

const TARGET_FRAMES = 72;
const TARGET_SLOTS = 40;

const logEl = document.getElementById('log');
function log(line: string): void {
  console.log(line);
  if (logEl) logEl.textContent += line + '\n';
}

const withTimeout = <T>(p: Promise<T>, ms: number) =>
  Promise.race([p.then(() => true), new Promise<boolean>((r) => setTimeout(() => r(false), ms))]);

type IngestMode = 'external' | 'canvas' | 'none';
const ingestParam = new URLSearchParams(location.search).get('ingest');
const INGEST: IngestMode = ingestParam === 'canvas' || ingestParam === 'none' ? ingestParam : 'external';

function canvasLuma(v: HTMLVideoElement, ctx: OffscreenCanvasRenderingContext2D): Float32Array {
  const N = 256;
  ctx.drawImage(v, Math.floor((v.videoWidth - N) / 2), Math.floor((v.videoHeight - N) / 2), N, N, 0, 0, N, N);
  const px = ctx.getImageData(0, 0, N, N).data;
  const out = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) out[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
  return out;
}

async function run(): Promise<Record<string, unknown>> {
  log(`step: ingest mode ${INGEST}`);
  const roiCtx = new OffscreenCanvas(256, 256).getContext('2d', { willReadFrequently: true })!;
  const video = document.getElementById('video') as HTMLVideoElement;
  const cam = new CameraManager();
  log('step: getUserMedia');
  const settings = await cam.start(video);
  log('step: started ' + JSON.stringify(settings));
  const lock = await cam.lock();
  log('step: locked ' + JSON.stringify({ exposure: lock.exposure, iso: lock.iso, focus: lock.focus, whiteBalance: lock.whiteBalance, badge: lock.badge, errors: lock.errors }));

  const shaderCode = await (await fetch(new URL('../../src/ddm.wgsl', location.href))).text();
  let gpuError = '';
  const engine = await DdmEngine.create({ shaderCode, onError: (m) => (gpuError ||= m), onLost: (m) => (gpuError ||= m) });
  log('step: engine ready');
  const tReady = performance.now();

  const clock = new FrameClock(settings.frameRate || 30);
  let frames = 0, slots = 0;
  let timeSource: TimeSource | '' = '';
  let temporal: Promise<void> | null = null;
  let finish: () => void = () => {};
  const done = new Promise<void>((r) => (finish = r));

  const onFrame = (e: FrameEvent) => {
    frames++;
    timeSource = e.timeSource;
    if (frames === 1 || frames % 30 === 0) log(`step: frame ${frames} t=${e.timestampMs.toFixed(1)}ms src=${e.timeSource}`);
    const placed = clock.place(e.timestampMs, e.timeSource);
    if (placed && slots < TARGET_SLOTS) {
      let ok = false;
      if (INGEST === 'external') {
        let src: VideoFrame | HTMLVideoElement = e.video;
        try {
          src = new VideoFrame(e.video, { timestamp: Math.round(e.timestampMs * 1000) });
        } catch {
          src = e.video;
        }
        ok = engine.ingestFrame(src, placed.slot);
      } else if (INGEST === 'canvas') {
        ok = engine.ingestLuma(canvasLuma(e.video, roiCtx), placed.slot);
      } else {
        ok = true;
      }
      if (ok) slots++;
      else clock.drop(placed.n);
    }
    if (slots >= TARGET_SLOTS && !temporal && INGEST !== 'none') {
      log('step: temporal started (stress: 8 chunks while frames keep arriving)');
      const snap = clock.snapshot();
      temporal = engine.computeTemporal(snap.order, snap.mask);
    }
    if (frames >= TARGET_FRAMES && slots >= TARGET_SLOTS) finish();
  };
  const loop = new FrameLoop(video, onFrame);
  loop.start();

  const silentCheck = setTimeout(() => {
    if (frames === 0) log('step: NO frame within 5 s of engine ready (requestVideoFrameCallback and watchdog both silent)');
  }, 5000);
  await withTimeout(done, 25000 - (performance.now() - tReady));
  clearTimeout(silentCheck);
  loop.stop();
  log(`step: loop done frames=${frames} slots=${slots}`);

  const idle = await withTimeout(Promise.all([temporal ?? Promise.resolve(), engine.idle()]), 10000);
  log(idle ? 'step: queue idle' : 'step: queue NOT idle after 10 s (GPU queue stalled)');
  if (gpuError) log(`step: GPU error: ${gpuError}`);

  const result = {
    lock: {
      exposure: lock.exposure,
      iso: lock.iso,
      focus: lock.focus,
      whiteBalance: lock.whiteBalance,
      badge: lock.badge,
      measurementAllowed: lock.measurementAllowed,
    },
    timeSource,
    frames,
    slots,
    missed: clock.missed,
    dtMs: +clock.medianDtMs.toFixed(2),
    watchdogFrames: loop.watchdogFrames,
    throttled: engine.throttled,
    queueIdle: idle,
    ingest: INGEST,
    gpuError: gpuError || undefined,
    settings,
  };
  engine.destroy();
  cam.stop();
  return result;
}

run()
  .then((r) => {
    log('RESULT ' + JSON.stringify(r));
    (window as unknown as { __RESULT: unknown }).__RESULT = r;
  })
  .catch((e: unknown) => {
    const err = e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ''}` : String(e);
    log('RESULT ' + JSON.stringify({ error: err }));
    (window as unknown as { __RESULT: unknown }).__RESULT = { error: err };
  });
