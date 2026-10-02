// Camera: hardware register locks, frame clock and frame-loop watchdog.
//
// Lock procedure (lock()):
//   1. single-shot autofocus on the sample, wait for it to settle
//   2. ONE applyConstraints() call that freezes focus, exposure time, ISO and
//      white balance at the values the auto loops converged to (all at once,
//      so no auto loop can react to another one being frozen)
//   3. read back getSettings() and classify each register
// Android Chrome reports `iso` (and exposureTime, focusDistance,
// colorTemperature) as a range dictionary {min, max, step}, not a list; every
// capability goes through clampToCapability(), which accepts both forms.
// Measurement is always allowed, but when the exposure readback is not
// 'manual' every verdict is reported as UNLOCKED_EXPOSURE (auto-exposure steps
// decorrelate the image and look like motion).
//
// Frame clock: frames are slotted into the 256-slot ring by capture time,
// slot n = round((t - t0) / dt). Gaps are masked, never interpolated.
// Time sources, best first: rVFC metadata.captureTime ('captureTime'),
// metadata.mediaTime ('mediaTime'), and a watchdog that polls
// video.currentTime every 50 ms with setTimeout and uses performance.now()
// ('watchdog') when requestVideoFrameCallback has gone silent.

import { DDM, shuffleOrder } from './ddm_ref.js';

export type LockState = 'locked' | 'mismatch' | 'unsupported' | 'unlocked';
export type LockBadge = 'LOCKED' | 'UNLOCKED_EXPOSURE' | 'MISMATCH';
export type TimeSource = 'captureTime' | 'mediaTime' | 'watchdog';

export interface LockReport {
  exposure: LockState;
  iso: LockState;
  focus: LockState;
  whiteBalance: LockState;
  /** Measurement may run; verdicts are flagged when exposure is not locked. */
  measurementAllowed: boolean;
  badge: LockBadge;
  requested: Record<string, unknown>;
  settings: Record<string, unknown>;
  errors: string[];
}

type Range = { min?: number; max?: number; step?: number };
type Capability = number[] | Range | undefined;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * Clamp a value to a capability that is either a discrete list or a
 * {min, max, step} range (snapped to the step grid from min).
 */
export function clampToCapability(value: number, cap: Capability): number {
  if (cap === undefined || cap === null) return value;
  if (Array.isArray(cap)) {
    if (!cap.length) return value;
    return cap.reduce((best, v) => (Math.abs(v - value) < Math.abs(best - value) ? v : best), cap[0]);
  }
  const lo = cap.min ?? -Infinity, hi = cap.max ?? Infinity;
  let v = Math.min(hi, Math.max(lo, value));
  if (cap.step && cap.step > 0 && Number.isFinite(lo)) {
    v = lo + Math.round((v - lo) / cap.step) * cap.step;
    v = Math.min(hi, Math.max(lo, v));
  }
  return v;
}

function modes(cap: unknown): string[] {
  return Array.isArray(cap) ? (cap as string[]) : [];
}

const near = (a: unknown, b: unknown, rel = 0.1) =>
  typeof a === 'number' && typeof b === 'number' && Math.abs(a - b) <= rel * Math.max(Math.abs(b), 1e-9);

export class CameraManager {
  stream: MediaStream | null = null;
  track: MediaStreamTrack | null = null;
  lockReport: LockReport | null = null;

  /** Open the rear camera at 1280x720, 30 fps, and attach it to `video`. */
  async start(video: HTMLVideoElement): Promise<MediaTrackSettings> {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia is not available (needs HTTPS or localhost)');
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
    });
    this.track = this.stream.getVideoTracks()[0];
    video.srcObject = this.stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    return this.track.getSettings();
  }

  get frameRate(): number {
    return this.track?.getSettings().frameRate || 30;
  }

  /** Freeze focus, exposure, ISO and white balance; read back and classify. */
  async lock(): Promise<LockReport> {
    const track = this.track;
    if (!track) throw new Error('camera not started');
    const errors: string[] = [];
    const caps = (track.getCapabilities?.() ?? {}) as Record<string, Capability | string[]>;
    const apply = async (c: Record<string, unknown>) => {
      await track.applyConstraints({ advanced: [c as MediaTrackConstraintSet] });
    };

    // 1. single-shot AF first
    if (modes(caps.focusMode).includes('single-shot')) {
      try {
        await apply({ focusMode: 'single-shot' });
        await sleep(1500);
      } catch (e) {
        errors.push(`single-shot focus: ${(e as Error).message}`);
      }
    }

    // 2. freeze everything at the converged values in one call
    const s0 = track.getSettings() as Record<string, unknown>;
    const fps = (s0.frameRate as number) || 30;
    const req: Record<string, unknown> = {};
    if (modes(caps.focusMode).includes('manual')) {
      req.focusMode = 'manual';
      if (caps.focusDistance && typeof s0.focusDistance === 'number') req.focusDistance = clampToCapability(s0.focusDistance, caps.focusDistance as Capability);
    }
    if (modes(caps.exposureMode).includes('manual')) {
      req.exposureMode = 'manual';
      if (caps.exposureTime) {
        // exposureTime is in 100 us units; never longer than one frame interval
        const frameLimit = 10000 / fps;
        const current = typeof s0.exposureTime === 'number' ? s0.exposureTime : frameLimit / 2;
        req.exposureTime = clampToCapability(Math.min(current, frameLimit), caps.exposureTime as Capability);
      }
      if (caps.iso) {
        const r = caps.iso as Capability;
        const current = typeof s0.iso === 'number' ? s0.iso : Array.isArray(r) ? r[0] : (r?.min ?? 100);
        req.iso = clampToCapability(current, r);
      }
    }
    if (modes(caps.whiteBalanceMode).includes('manual')) {
      req.whiteBalanceMode = 'manual';
      if (caps.colorTemperature && typeof s0.colorTemperature === 'number') req.colorTemperature = clampToCapability(s0.colorTemperature, caps.colorTemperature as Capability);
    }
    if (Object.keys(req).length) {
      try {
        await apply(req);
      } catch (e) {
        errors.push(`combined lock rejected (${(e as Error).name}: ${(e as Error).message}); retrying per register`);
        const groups: Array<Record<string, unknown>> = [
          pick(req, ['focusMode', 'focusDistance']),
          pick(req, ['exposureMode', 'exposureTime', 'iso']),
          pick(req, ['whiteBalanceMode', 'colorTemperature']),
        ];
        for (const g of groups) {
          if (!Object.keys(g).length) continue;
          try {
            await apply(g);
          } catch (e2) {
            errors.push(`${Object.keys(g).join('+')}: ${(e2 as Error).message}`);
          }
        }
      }
      await sleep(400);
    }

    // 3. readback
    const s1 = track.getSettings() as Record<string, unknown>;
    const classify = (modeKey: string, valueKey?: string): LockState => {
      if (!modes(caps[modeKey]).length) return 'unsupported';
      if (!modes(caps[modeKey]).includes('manual')) return 'unsupported';
      if (s1[modeKey] !== 'manual') return 'mismatch';
      if (valueKey && req[valueKey] !== undefined && s1[valueKey] !== undefined && !near(s1[valueKey], req[valueKey])) return 'mismatch';
      return 'locked';
    };
    const exposure = classify('exposureMode', 'exposureTime');
    let iso: LockState = 'unsupported';
    if (caps.iso) iso = exposure === 'locked' ? (req.iso === undefined || s1.iso === undefined || near(s1.iso, req.iso) ? 'locked' : 'mismatch') : 'unlocked';
    const report: LockReport = {
      exposure,
      iso,
      focus: classify('focusMode', 'focusDistance'),
      whiteBalance: classify('whiteBalanceMode', 'colorTemperature'),
      measurementAllowed: track.readyState === 'live',
      badge: exposure === 'locked' ? 'LOCKED' : exposure === 'mismatch' ? 'MISMATCH' : 'UNLOCKED_EXPOSURE',
      requested: req,
      settings: s1,
      errors,
    };
    this.lockReport = report;
    return report;
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.track = null;
  }
}

function pick(o: Record<string, unknown>, keys: string[]): Record<string, unknown> {
  const r: Record<string, unknown> = {};
  for (const k of keys) if (o[k] !== undefined) r[k] = o[k];
  return r;
}

export interface FrameEvent {
  video: HTMLVideoElement;
  /** Capture time in ms (performance.now() timebase for captureTime/watchdog). */
  timestampMs: number;
  timeSource: TimeSource;
}

/**
 * requestVideoFrameCallback loop with a setTimeout watchdog. If rVFC stays
 * silent for more than `stallMs` while video.currentTime keeps advancing (seen
 * on some devices while WebGPU work is in flight), the watchdog delivers the
 * frames instead, stamped with performance.now() minus the capture latency
 * measured from earlier rVFC frames. captureTime is in the performance.now()
 * timebase, so watchdog and captureTime frames share one clock and an
 * intermittent stall does not restart the history.
 */
export class FrameLoop {
  private running = false;
  private lastRvfcAt = 0;
  private lastCurrentTime = -1;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private rvfcHandle = 0;
  watchdogFrames = 0;
  rvfcFrames = 0;
  /** Running estimate of (callback time - captureTime), ms. */
  latencyMs = 0;
  private latencyN = 0;

  constructor(
    private video: HTMLVideoElement,
    private onFrame: (e: FrameEvent) => void,
    private opts: { pollMs?: number; stallMs?: number } = {},
  ) {}

  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastRvfcAt = performance.now();
    const v = this.video;
    const hasRvfc = typeof v.requestVideoFrameCallback === 'function';
    if (hasRvfc) {
      const cb = (_now: number, meta: VideoFrameCallbackMetadata) => {
        if (!this.running) return;
        this.lastRvfcAt = performance.now();
        this.lastCurrentTime = v.currentTime;
        let ts: number, src: TimeSource;
        if (typeof meta.captureTime === 'number' && meta.captureTime > 0) {
          ts = meta.captureTime;
          src = 'captureTime';
          const lat = performance.now() - ts;
          if (lat >= 0 && lat < 1000) {
            this.latencyN = Math.min(this.latencyN + 1, 30);
            this.latencyMs += (lat - this.latencyMs) / this.latencyN;
          }
        } else {
          ts = meta.mediaTime * 1000;
          src = 'mediaTime';
        }
        this.rvfcFrames++;
        this.onFrame({ video: v, timestampMs: ts, timeSource: src });
        this.rvfcHandle = v.requestVideoFrameCallback(cb);
      };
      this.rvfcHandle = v.requestVideoFrameCallback(cb);
    }
    const poll = this.opts.pollMs ?? 50;
    const stall = this.opts.stallMs ?? 200;
    const tick = () => {
      if (!this.running) return;
      const now = performance.now();
      const ct = v.currentTime;
      if (ct !== this.lastCurrentTime && (!hasRvfc || now - this.lastRvfcAt > stall)) {
        this.lastCurrentTime = ct;
        this.watchdogFrames++;
        this.onFrame({ video: v, timestampMs: now - this.latencyMs, timeSource: 'watchdog' });
      }
      this.timer = setTimeout(tick, poll);
    };
    this.timer = setTimeout(tick, poll);
  }

  stop(): void {
    this.running = false;
    if (this.timer) clearTimeout(this.timer);
    if (this.rvfcHandle && typeof this.video.cancelVideoFrameCallback === 'function') this.video.cancelVideoFrameCallback(this.rvfcHandle);
  }
}

export interface Snapshot {
  order: Uint32Array;
  mask: Float32Array;
  /** valid frames in the snapshot */
  used: number;
  /** newest frame index n at snapshot time */
  newestN: number;
}

/**
 * Uniform 256-slot frame clock. Frame n lives in slot n mod 256; the mask is
 * built from the frame index actually stored in each slot, so dropped, late,
 * throttled and overwritten frames are all excluded automatically.
 */
export class FrameClock {
  readonly T = DDM.T;
  readonly dtMs: number;
  private t0: number | null = null;
  private lastN = -1;
  private slotN = new Int32Array(DDM.T).fill(-1);
  private src: TimeSource | null = null;
  frames = 0;
  missed = 0;
  duplicates = 0;
  resets = 0;
  private lastTs = NaN;
  private dts: number[] = [];

  constructor(fps: number) {
    this.dtMs = 1000 / fps;
  }

  get timeSource(): TimeSource | null {
    return this.src;
  }

  get newestN(): number {
    return this.lastN;
  }

  reset(): void {
    this.t0 = null;
    this.lastN = -1;
    this.slotN.fill(-1);
    this.src = null;
    this.lastTs = NaN;
    this.resets++;
  }

  /**
   * Assign a capture time to a slot; null for duplicates. captureTime and
   * watchdog stamps share the performance.now() timebase and mix freely;
   * switching to or from mediaTime (media timeline) restarts the clock.
   */
  place(tMs: number, source: TimeSource): { slot: number; n: number } | null {
    if (this.src !== null && (source === 'mediaTime') !== (this.src === 'mediaTime')) this.reset();
    this.src = source;
    if (this.t0 === null) this.t0 = tMs;
    let n = Math.round((tMs - this.t0) / this.dtMs);
    if (n <= this.lastN) {
      this.duplicates++;
      return null;
    }
    if (this.lastN >= 0 && n - this.lastN > this.T) {
      this.reset();
      this.src = source;
      this.t0 = tMs;
      n = 0;
    }
    if (this.lastN >= 0) this.missed += n - this.lastN - 1;
    if (Number.isFinite(this.lastTs)) {
      this.dts.push(tMs - this.lastTs);
      if (this.dts.length > 120) this.dts.shift();
    }
    this.lastTs = tMs;
    const slot = n % this.T;
    this.slotN[slot] = n;
    this.lastN = n;
    this.frames++;
    return { slot, n };
  }

  /** Mark frame n as not ingested (e.g. dropped by GPU back-pressure). */
  drop(n: number): void {
    const slot = n % this.T;
    if (this.slotN[slot] === n) this.slotN[slot] = -1;
  }

  /** Median interval between delivered frames (ms). */
  get medianDtMs(): number {
    if (!this.dts.length) return NaN;
    const s = [...this.dts].sort((a, b) => a - b);
    return s[s.length >> 1];
  }

  /** Valid frames currently in the window. */
  get validCount(): number {
    let c = 0;
    for (let j = 0; j < this.T; j++) {
      const n = this.lastN - this.T + 1 + j;
      if (n >= 0 && this.slotN[n % this.T] === n) c++;
    }
    return c;
  }

  /**
   * Time-ordered (slot, mask) arrays for the temporal pass. The oldest `guard`
   * time indices are masked because new frames may overwrite those slots while
   * the analysis runs. shuffle = Control 1 (frame-order null test).
   */
  snapshot(opts: { guard?: number; shuffle?: boolean; rand?: () => number } = {}): Snapshot {
    const guard = opts.guard ?? 0;
    const order = new Uint32Array(this.T);
    const mask = new Float32Array(this.T);
    let used = 0;
    for (let j = 0; j < this.T; j++) {
      const n = this.lastN - this.T + 1 + j;
      const slot = ((n % this.T) + this.T) % this.T;
      order[j] = slot;
      const ok = n >= 0 && j >= guard && this.slotN[slot] === n;
      mask[j] = ok ? 1 : 0;
      if (ok) used++;
    }
    if (opts.shuffle) {
      const s = shuffleOrder({ order, mask }, opts.rand ?? Math.random);
      return { order: s.order, mask: s.mask, used, newestN: this.lastN };
    }
    return { order, mask, used, newestN: this.lastN };
  }
}
