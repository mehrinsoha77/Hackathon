// Double-precision CPU reference for the MotilityDDM pipeline.
//
// Every quantity the GPU computes (src/ddm.wgsl) has a float64 twin here with
// the same memory layout, so test/gpu/harness.ts can compare them element by
// element and the CPU validation matrix (test/t_validation_matrix.mjs) runs the
// identical analysis path without a GPU.
//
// Pipeline per frame:   luma ROI (N x N) -> mean -> window -> 2D FFT (half plane)
//                       -> history slot `slot` of a T-slot ring.
// Pipeline per analysis: for every Fourier mode, masked Wiener-Khinchin
//                       structure function D(q,tau), phase correlation C(q,tau),
//                       static power P(q) -> radial reduction into NB q-bins plus
//                       the phase-drift least-squares sums.

import { fft, blackmanHarris, spectrum2dHalf } from './fft.js';

/** Fixed geometry shared with the GPU engine. */
export const DDM = Object.freeze({
  N: 256, // ROI side, pixels
  NH: 129, // N/2 + 1 (Hermitian half plane along kx)
  MODES: 256 * 129, // 33024 Fourier modes per frame
  T: 256, // history slots (frames)
  TPAD: 512, // zero-padded temporal FFT length (linear, not circular, correlation)
  L: 128, // output lags tau = 0..L-1 (frames)
  NB: 32, // radial q-bins
  KMIN: 4, // inner edge of the first bin, in FFT index units |k|
  KMAX: 100, // outer edge of the last bin
  DRIFT_STRIDE: 8, // floats per (bin, lag) drift record
  TEMPORAL_CHUNKS: 8, // temporal pass split into 8 dispatches of MODES/8 workgroups
  KNODES: 4, // effective-|k| quadrature nodes per bin (bin width + window leakage)
});

/**
 * Power leakage of the Blackman-Harris window, |W(d)|^2 for fractional bin
 * offsets d on a grid, normalized to sum 1. The main lobe spans |d| < 4; side
 * lobes are below -92 dB and ignored.
 * @param {number} N
 * @param {number} step
 */
export function windowLeakage(N, step = 0.5) {
  const w = blackmanHarris(N);
  const d = [], p = [];
  for (let x = -4; x <= 4 + 1e-9; x += step) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const a = (-2 * Math.PI * x * n) / N;
      re += w[n] * Math.cos(a);
      im += w[n] * Math.sin(a);
    }
    d.push(x);
    p.push(re * re + im * im);
  }
  const s = p.reduce((a, b) => a + b, 0);
  return { d, p: p.map((v) => v / s) };
}

/**
 * Signed integer wavevector components of half-plane mode m.
 * @param {number} m
 * @param {number} [N]
 * @returns {[number, number]} [kx, ky]
 */
export function modeK(m, N = DDM.N) {
  const NH = (N >> 1) + 1;
  const kx = m % NH;
  const kyi = (m - kx) / NH;
  const ky = kyi < N / 2 ? kyi : kyi - N;
  return [kx, ky];
}

/**
 * Radial bin table. Bins are linear in |k| between KMIN and KMAX. Modes on the
 * kx = 0 axis with ky < 0 are the complex conjugates of ky > 0 and are skipped
 * so that no mode is counted twice.
 *
 * binModes holds 4 int32 per entry: [mode, kx, ky, 0] (the GPU reads it as
 * array<vec4<i32>>).
 *
 * @param {number} [N]
 * @param {number} [NB]
 * @param {number} [kmin]
 * @param {number} [kmax]
 */
export function makeBins(N = DDM.N, NB = DDM.NB, kmin = DDM.KMIN, kmax = DDM.KMAX) {
  const NH = (N >> 1) + 1;
  const width = (kmax - kmin) / NB;
  /** @type {number[][]} */
  const lists = Array.from({ length: NB }, () => []);
  for (let m = 0; m < N * NH; m++) {
    const [kx, ky] = modeK(m, N);
    if (kx === 0 && ky <= 0) continue;
    if (kx === N / 2 || ky === -N / 2) continue; // Nyquist lines
    const kr = Math.hypot(kx, ky);
    const b = Math.floor((kr - kmin) / width);
    if (b >= 0 && b < NB) lists[b].push(m);
  }
  const binOffsets = new Uint32Array(NB + 1);
  let total = 0;
  for (let b = 0; b < NB; b++) {
    binOffsets[b] = total;
    total += lists[b].length;
  }
  binOffsets[NB] = total;
  const binModes = new Int32Array(4 * total);
  const kCenter = new Float64Array(NB);
  const nModes = new Uint32Array(NB);
  // Effective |k| distribution of each bin: the bin's own modes convolved with
  // the window's power leakage, represented by KNODES equal-weight quantile
  // nodes. The fit averages its model over these nodes, so a single-rate decay
  // smeared over a finite ring of |k| is not mistaken for a velocity kernel.
  const K = DDM.KNODES;
  const kNodes = new Float64Array(NB * K);
  const leak = windowLeakage(N);
  for (let b = 0; b < NB; b++) {
    let sk = 0;
    lists[b].forEach((m, i) => {
      const [kx, ky] = modeK(m, N);
      const o = 4 * (binOffsets[b] + i);
      binModes[o] = m;
      binModes[o + 1] = kx;
      binModes[o + 2] = ky;
      sk += Math.hypot(kx, ky);
    });
    nModes[b] = lists[b].length;
    kCenter[b] = lists[b].length ? sk / lists[b].length : kmin + (b + 0.5) * width;
    // weighted histogram of |k + d| (resolution 0.01) instead of sorting samples
    const h0 = Math.max(0, kmin - 6), hres = 0.01;
    const hist = new Float64Array(Math.ceil((kmax + 6 - h0) / hres) + 1);
    const hk = new Float64Array(hist.length);
    for (const m of lists[b]) {
      const [kx, ky] = modeK(m, N);
      for (let i = 0; i < leak.d.length; i++) {
        const ax = kx + leak.d[i];
        for (let j = 0; j < leak.d.length; j++) {
          const k = Math.hypot(ax, ky + leak.d[j]);
          const wt = leak.p[i] * leak.p[j];
          const c = Math.min(hist.length - 1, Math.max(0, Math.round((k - h0) / hres)));
          hist[c] += wt;
          hk[c] += wt * k;
        }
      }
    }
    const wsum = hist.reduce((a, v) => a + v, 0);
    if (!(wsum > 0)) {
      for (let n = 0; n < K; n++) kNodes[b * K + n] = kCenter[b];
      continue;
    }
    // node n = weighted mean |k| of the n-th equal-weight quantile slice
    let acc = 0, n = 0, sw = 0, swk = 0;
    for (let c = 0; c < hist.length && n < K; c++) {
      let rem = hist[c];
      const kc = hist[c] > 0 ? hk[c] / hist[c] : 0;
      while (rem > 0 && n < K) {
        const take = Math.min(rem, ((n + 1) / K) * wsum - acc);
        sw += take;
        swk += take * kc;
        acc += take;
        rem -= take;
        if (acc >= ((n + 1) / K) * wsum * (1 - 1e-12)) {
          kNodes[b * K + n] = sw > 0 ? swk / sw : kc;
          n++;
          sw = 0;
          swk = 0;
        }
      }
    }
    for (; n < K; n++) kNodes[b * K + n] = kmax;
  }
  return { N, NB, kmin, kmax, binOffsets, binModes, kCenter, nModes, kNodes };
}

/** @typedef {ReturnType<typeof makeBins>} Bins */

/**
 * Ring history of half-plane spectra: hist[(mode * T + slot) * 2 + {0,1}].
 * Same layout as the GPU `hist` buffer.
 */
export class History {
  /** @param {number} [N] @param {number} [T] */
  constructor(N = DDM.N, T = DDM.T) {
    this.N = N;
    this.T = T;
    this.NH = (N >> 1) + 1;
    this.modes = N * this.NH;
    this.data = new Float64Array(this.modes * T * 2);
    this.win = blackmanHarris(N);
    this.frame = new Float64Array(2 * this.modes);
  }

  /**
   * Transform one N x N luma frame and store it in `slot`.
   * @param {ArrayLike<number>} luma
   * @param {number} slot
   * @returns {Float64Array} the spectrum just written (shared scratch buffer)
   */
  ingest(luma, slot) {
    const spec = spectrum2dHalf(luma, this.N, this.win, this.frame);
    const T = this.T;
    for (let m = 0; m < this.modes; m++) {
      const o = (m * T + slot) * 2;
      this.data[o] = spec[2 * m];
      this.data[o + 1] = spec[2 * m + 1];
    }
    return spec;
  }
}

/**
 * Masked structure function of one mode via the Wiener-Khinchin identity.
 *
 * With y(t) = m(t) x(t), p(t) = m(t)|x(t)|^2 and zero padding to TPAD:
 *   A(tau) = sum_t m(t) p(t+tau)        B(tau) = sum_t p(t) m(t+tau) = A(-tau)
 *   N(tau) = sum_t m(t) m(t+tau)        G(tau) = sum_t conj(y(t)) y(t+tau)
 *   D(tau) = [A(tau) + B(tau) - 2 Re G(tau)] / N(tau)
 *          = <|x(t+tau) - x(t)|^2> over valid pairs
 *   C(tau) = G(tau) / N(tau),  P = A(0) / N(0) = <|x|^2>
 * A and N come from one complex FFT of z = m + i p (two real signals packed).
 *
 * @param {Float64Array} hist history data (History#data layout)
 * @param {number} mode
 * @param {number} T slots
 * @param {Uint32Array|number[]} order order[j] = slot holding time index j
 * @param {Float32Array|Float64Array|number[]} mask mask[j] in {0,1}
 * @param {number} L lags to output
 * @param {Float64Array} outD length L
 * @param {Float64Array} outC length 2L
 * @returns {number} P (static power)
 */
export function temporalMode(hist, mode, T, order, mask, L, outD, outC) {
  const n = 2 * T;
  const z = tmp(n, 'z');
  const y = tmp(n, 'y');
  z.fill(0);
  y.fill(0);
  const base = mode * T * 2;
  for (let j = 0; j < T; j++) {
    const m = mask[j] ? 1 : 0;
    if (!m) continue;
    const s = order[j];
    const xr = hist[base + 2 * s];
    const xi = hist[base + 2 * s + 1];
    z[2 * j] = 1;
    z[2 * j + 1] = xr * xr + xi * xi;
    y[2 * j] = xr;
    y[2 * j + 1] = xi;
  }
  fft(z);
  const S = tmp(n, 'S');
  for (let k = 0; k < n; k++) {
    const kk = (n - k) % n;
    const zr = z[2 * k], zi = z[2 * k + 1];
    const cr = z[2 * kk], ci = -z[2 * kk + 1]; // conj(Z[-k])
    const Mr = 0.5 * (zr + cr), Mi = 0.5 * (zi + ci);
    // P_k = (Z_k - conj Z_-k) / (2i)  ->  (a + ib)/(2i) = (b - ia)/2
    const dr = zr - cr, di = zi - ci;
    const Pr = 0.5 * di, Pi = -0.5 * dr;
    // conj(M) * P + i |M|^2
    S[2 * k] = Mr * Pr + Mi * Pi;
    S[2 * k + 1] = Mr * Pi - Mi * Pr + (Mr * Mr + Mi * Mi);
  }
  fft(S, true);
  fft(y);
  for (let k = 0; k < n; k++) {
    const yr = y[2 * k], yi = y[2 * k + 1];
    y[2 * k] = yr * yr + yi * yi;
    y[2 * k + 1] = 0;
  }
  fft(y, true);
  const inv = 1 / n;
  const P = S[0] * inv / Math.max(S[1] * inv, 1e-300);
  for (let tau = 0; tau < L; tau++) {
    const A = S[2 * tau] * inv;
    const B = S[2 * ((n - tau) % n)] * inv;
    const Np = Math.round(S[2 * tau + 1] * inv);
    if (Np > 0) {
      const Gr = y[2 * tau] * inv;
      const Gi = y[2 * tau + 1] * inv;
      outD[tau] = (A + B - 2 * Gr) / Np;
      outC[2 * tau] = Gr / Np;
      outC[2 * tau + 1] = Gi / Np;
    } else {
      outD[tau] = 0;
      outC[2 * tau] = 0;
      outC[2 * tau + 1] = 0;
    }
  }
  return S[1] * inv > 0.5 ? P : 0;
}

/** @type {Map<string, Float64Array>} */
const tmpCache = new Map();
function tmp(n, tag) {
  const key = tag + n;
  let a = tmpCache.get(key);
  if (!a) {
    a = new Float64Array(2 * n);
    tmpCache.set(key, a);
  }
  return a;
}

/**
 * Direct O(T*L) evaluation of the same masked estimators. Independent of the
 * FFT path; used by tests to validate the Wiener-Khinchin implementation.
 */
export function temporalModeDirect(hist, mode, T, order, mask, L, outD, outC) {
  const base = mode * T * 2;
  const xr = new Float64Array(T), xi = new Float64Array(T);
  let pSum = 0, nSum = 0;
  for (let j = 0; j < T; j++) {
    if (!mask[j]) continue;
    xr[j] = hist[base + 2 * order[j]];
    xi[j] = hist[base + 2 * order[j] + 1];
    pSum += xr[j] * xr[j] + xi[j] * xi[j];
    nSum++;
  }
  for (let tau = 0; tau < L; tau++) {
    let d = 0, gr = 0, gi = 0, cnt = 0;
    for (let t = 0; t + tau < T; t++) {
      if (!mask[t] || !mask[t + tau]) continue;
      const dr = xr[t + tau] - xr[t], di = xi[t + tau] - xi[t];
      d += dr * dr + di * di;
      // conj(x(t)) * x(t+tau)
      gr += xr[t] * xr[t + tau] + xi[t] * xi[t + tau];
      gi += xr[t] * xi[t + tau] - xi[t] * xr[t + tau];
      cnt++;
    }
    outD[tau] = cnt ? d / cnt : 0;
    outC[2 * tau] = cnt ? gr / cnt : 0;
    outC[2 * tau + 1] = cnt ? gi / cnt : 0;
  }
  return nSum ? pSum / nSum : 0;
}

/**
 * Radial reduction record (identical fields to the GPU `radial` pass output).
 * @typedef {object} Radial
 * @property {number} N
 * @property {number} NB
 * @property {number} L
 * @property {Float64Array} kCenter mean |k| per bin (FFT index units)
 * @property {Float64Array} kNodes effective-|k| quadrature nodes, kNodes[b*KNODES + n]
 * @property {Uint32Array} nModes modes per bin
 * @property {Float64Array} D D[b*L + tau], bin-averaged structure function
 * @property {Float64Array} P P[b], bin-averaged static power
 * @property {Float64Array} drift drift[(b*L + tau)*8 + c] phase least-squares sums
 *   for the lag pair (tau, tau+1): c = 0 Sw, 1 Sxx, 2 Sxy, 3 Syy, 4 Rx, 5 Ry, 6 Rpp, 7 count
 */

/**
 * Accumulate one mode into the radial sums (shared by the full and streaming paths).
 * @param {Radial} r
 * @param {number} b bin
 * @param {number} kx
 * @param {number} ky
 * @param {Float64Array} D length L
 * @param {Float64Array} C length 2L
 * @param {number} P
 */
function accumulateMode(r, b, kx, ky, D, C, P) {
  const L = r.L;
  r.P[b] += P;
  for (let tau = 0; tau < L; tau++) r.D[b * L + tau] += D[tau];
  for (let tau = 0; tau < L - 1; tau++) {
    const ar = C[2 * tau], ai = C[2 * tau + 1];
    const br = C[2 * tau + 2], bi = C[2 * tau + 3];
    // z = C(tau+1) * conj(C(tau))
    const zr = br * ar + bi * ai;
    const zi = bi * ar - br * ai;
    const w = Math.hypot(zr, zi);
    const phi = Math.atan2(zi, zr);
    const o = (b * L + tau) * DDM.DRIFT_STRIDE;
    r.drift[o] += w;
    r.drift[o + 1] += w * kx * kx;
    r.drift[o + 2] += w * kx * ky;
    r.drift[o + 3] += w * ky * ky;
    r.drift[o + 4] += w * kx * phi;
    r.drift[o + 5] += w * ky * phi;
    r.drift[o + 6] += w * phi * phi;
    r.drift[o + 7] += w > 0 ? 1 : 0;
  }
}

/**
 * @param {Bins} bins
 * @param {number} L
 * @returns {Radial}
 */
export function emptyRadial(bins, L = DDM.L) {
  return {
    N: bins.N,
    NB: bins.NB,
    L,
    kCenter: Float64Array.from(bins.kCenter),
    kNodes: Float64Array.from(bins.kNodes),
    nModes: Uint32Array.from(bins.nModes),
    D: new Float64Array(bins.NB * L),
    P: new Float64Array(bins.NB),
    drift: new Float64Array(bins.NB * L * DDM.DRIFT_STRIDE),
  };
}

/** @param {Radial} r */
function finalizeRadial(r) {
  for (let b = 0; b < r.NB; b++) {
    const n = r.nModes[b];
    if (!n) continue;
    r.P[b] /= n;
    for (let tau = 0; tau < r.L; tau++) r.D[b * r.L + tau] /= n;
  }
  return r;
}

/**
 * Temporal analysis of the binned modes only, reduced straight to radial sums.
 * This is what the CPU validation matrix runs.
 *
 * @param {History} history
 * @param {Uint32Array|number[]} order
 * @param {Float32Array|Float64Array|number[]} mask
 * @param {Bins} bins
 * @param {number} [L]
 * @returns {Radial}
 */
export function analyzeRadial(history, order, mask, bins, L = DDM.L) {
  const r = emptyRadial(bins, L);
  const D = new Float64Array(L);
  const C = new Float64Array(2 * L);
  for (let b = 0; b < bins.NB; b++) {
    for (let i = bins.binOffsets[b]; i < bins.binOffsets[b + 1]; i++) {
      const m = bins.binModes[4 * i];
      const P = temporalMode(history.data, m, history.T, order, mask, L, D, C);
      accumulateMode(r, b, bins.binModes[4 * i + 1], bins.binModes[4 * i + 2], D, C, P);
    }
  }
  return finalizeRadial(r);
}

/**
 * Full per-mode temporal analysis (all MODES), as the GPU `temporal` pass
 * produces it. Used by the GPU parity harness.
 *
 * @param {History} history
 * @param {Uint32Array|number[]} order
 * @param {Float32Array|Float64Array|number[]} mask
 * @param {number} [L]
 * @returns {{D: Float64Array, C: Float64Array, P: Float64Array}}
 *   D[mode*L+tau], C[(mode*L+tau)*2+{0,1}], P[mode]
 */
export function analyzeFull(history, order, mask, L = DDM.L) {
  const M = history.modes;
  const D = new Float64Array(M * L);
  const C = new Float64Array(M * L * 2);
  const P = new Float64Array(M);
  const d = new Float64Array(L);
  const c = new Float64Array(2 * L);
  for (let m = 0; m < M; m++) {
    P[m] = temporalMode(history.data, m, history.T, order, mask, L, d, c);
    D.set(d, m * L);
    C.set(c, m * L * 2);
  }
  return { D, C, P };
}

/**
 * Radial reduction of full per-mode arrays (CPU twin of the GPU `radial` pass).
 * Accepts float32 (GPU readback) or float64 arrays.
 *
 * @param {{D: ArrayLike<number>, C: ArrayLike<number>, P: ArrayLike<number>}} full
 * @param {Bins} bins
 * @param {number} [L]
 * @returns {Radial}
 */
export function radialFromFull(full, bins, L = DDM.L) {
  const r = emptyRadial(bins, L);
  const D = new Float64Array(L);
  const C = new Float64Array(2 * L);
  for (let b = 0; b < bins.NB; b++) {
    for (let i = bins.binOffsets[b]; i < bins.binOffsets[b + 1]; i++) {
      const m = bins.binModes[4 * i];
      for (let t = 0; t < L; t++) D[t] = full.D[m * L + t];
      for (let t = 0; t < 2 * L; t++) C[t] = full.C[m * L * 2 + t];
      accumulateMode(r, b, bins.binModes[4 * i + 1], bins.binModes[4 * i + 2], D, C, full.P[m]);
    }
  }
  return finalizeRadial(r);
}

/**
 * Identity time order with a given validity mask (no ring wrap, no shuffle).
 * @param {boolean[]|Uint8Array} valid length T
 */
export function linearOrder(valid) {
  const T = valid.length;
  const order = new Uint32Array(T);
  const mask = new Float32Array(T);
  for (let j = 0; j < T; j++) {
    order[j] = j;
    mask[j] = valid[j] ? 1 : 0;
  }
  return { order, mask };
}

/**
 * Shuffle the (slot, mask) pairs of an order with a seeded RNG: the Control 1
 * null test. Destroys temporal ordering while keeping every frame.
 * @param {{order: Uint32Array, mask: Float32Array}} o
 * @param {() => number} rand uniform [0,1)
 */
export function shuffleOrder(o, rand) {
  const order = Uint32Array.from(o.order);
  const mask = Float32Array.from(o.mask);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
    [mask[i], mask[j]] = [mask[j], mask[i]];
  }
  return { order, mask };
}

/** Byte layout of the GPU TemporalMeta struct (src/ddm.wgsl). */
export const TEMPORAL_META = Object.freeze({
  ORDER: 0, // u32[256]
  MASK: 1024, // f32[256]
  NPAIRS: 2048, // f32[128]
  MSPEC: 2560, // vec2<f32>[512]
  BYTES: 6656,
});

/**
 * Pack the per-analysis, mode-independent metadata for the GPU temporal pass:
 * slot order, mask, exact valid-pair counts N(tau) and the float64 spectrum of
 * the zero-padded mask M_k (rounded once to f32).
 * @param {Uint32Array|number[]} order length T
 * @param {Float32Array|number[]} mask length T
 * @returns {ArrayBuffer}
 */
export function temporalMeta(order, mask) {
  const { T, TPAD, L } = DDM;
  if (order.length !== T || mask.length !== T) throw new Error(`temporalMeta: order/mask must have length ${T}`);
  const buf = new ArrayBuffer(TEMPORAL_META.BYTES);
  const o = new Uint32Array(buf, TEMPORAL_META.ORDER, T);
  const m = new Float32Array(buf, TEMPORAL_META.MASK, T);
  const np = new Float32Array(buf, TEMPORAL_META.NPAIRS, L);
  const ms = new Float32Array(buf, TEMPORAL_META.MSPEC, 2 * TPAD);
  const z = new Float64Array(2 * TPAD);
  for (let j = 0; j < T; j++) {
    o[j] = order[j];
    m[j] = mask[j] ? 1 : 0;
    z[2 * j] = m[j];
  }
  for (let tau = 0; tau < L; tau++) {
    let c = 0;
    for (let t = 0; t + tau < T; t++) if (m[t] && m[t + tau]) c++;
    np[tau] = c;
  }
  fft(z);
  for (let i = 0; i < 2 * TPAD; i++) ms[i] = z[i];
  return buf;
}

/**
 * Float64 twiddle table exp(-2 pi i m / 512), m < 256, rounded once to f32 for
 * the GPU (one table serves the 256- and 512-point transforms).
 */
export function gpuTwiddles() {
  const t = new Float32Array(2 * 256);
  for (let mm = 0; mm < 256; mm++) {
    const a = (-2 * Math.PI * mm) / 512;
    t[2 * mm] = Math.cos(a);
    t[2 * mm + 1] = Math.sin(a);
  }
  return t;
}

/**
 * Convert the GPU radial readback into the Radial record motility_fit expects.
 * @param {Bins} bins
 * @param {Float32Array} radOut D[b*L+tau] followed by P[b]
 * @param {Float32Array} radDrift
 * @param {number} [L]
 * @returns {Radial}
 */
export function radialFromGpu(bins, radOut, radDrift, L = DDM.L) {
  const r = emptyRadial(bins, L);
  r.D.set(radOut.subarray(0, bins.NB * L));
  r.P.set(radOut.subarray(bins.NB * L, bins.NB * L + bins.NB));
  r.drift.set(radDrift.subarray(0, bins.NB * L * DDM.DRIFT_STRIDE));
  return r;
}
