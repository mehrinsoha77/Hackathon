// Double-precision radix-2 Stockham FFTs (1D and 2D) for the CPU reference.
//
// The algorithm is the same autosort Stockham formulation the WGSL kernels use
// (src/ddm.wgsl): for pass Ns = 1, 2, 4, ..., n/2 every butterfly j reads
// src[j] and src[j + n/2], multiplies the second by exp(-2*pi*i*(j % Ns)/(2*Ns))
// and writes dst[(j/Ns)*2*Ns + j%Ns] and dst[... + Ns]. Output is in natural
// order, no bit reversal. Here everything is float64, so this file is the
// precision reference the f32 GPU path is compared against.

/** @type {Map<number, Float64Array>} */
const twiddleCache = new Map();

/**
 * Twiddle table tw[m] = exp(-2*pi*i*m/n), m = 0..n/2-1, interleaved re/im.
 * @param {number} n power of two
 * @returns {Float64Array}
 */
export function twiddles(n) {
  let t = twiddleCache.get(n);
  if (t) return t;
  t = new Float64Array(n);
  for (let m = 0; m < n / 2; m++) {
    const a = (-2 * Math.PI * m) / n;
    t[2 * m] = Math.cos(a);
    t[2 * m + 1] = Math.sin(a);
  }
  twiddleCache.set(n, t);
  return t;
}

/**
 * @param {number} n
 * @returns {number} log2(n); throws if n is not a power of two
 */
export function log2Exact(n) {
  const l = Math.round(Math.log2(n));
  if (n < 1 || 1 << l !== n) throw new Error(`FFT length ${n} is not a power of two`);
  return l;
}

/** @type {Map<number, {a: Float64Array, b: Float64Array}>} */
const scratchCache = new Map();
function scratch(n) {
  let s = scratchCache.get(n);
  if (!s) {
    s = { a: new Float64Array(2 * n), b: new Float64Array(2 * n) };
    scratchCache.set(n, s);
  }
  return s;
}

/**
 * In-place complex FFT of interleaved data (re, im, re, im, ...).
 * Unnormalized in both directions: inverse(forward(x)) = n * x.
 * @param {Float64Array} data length 2n
 * @param {boolean} [inverse]
 * @returns {Float64Array} data
 */
export function fft(data, inverse = false) {
  const n = data.length >> 1;
  log2Exact(n);
  if (n === 1) return data;
  const tw = twiddles(n);
  const half = n >> 1;
  const sgn = inverse ? -1 : 1;
  const { a, b } = scratch(n);
  a.set(data);
  let src = a;
  let dst = b;
  for (let Ns = 1; Ns < n; Ns <<= 1) {
    const step = half / Ns; // twiddle index stride: exp(-2pi i k/(2Ns)) = tw[k*n/(2Ns)]
    for (let j = 0; j < half; j++) {
      const k = j % Ns;
      const wr = tw[2 * k * step];
      const wi = sgn * tw[2 * k * step + 1];
      const i0 = 2 * j;
      const i1 = 2 * (j + half);
      const xr = src[i1] * wr - src[i1 + 1] * wi;
      const xi = src[i1] * wi + src[i1 + 1] * wr;
      const o = 2 * ((j - k) * 2 + k);
      const o2 = o + 2 * Ns;
      dst[o] = src[i0] + xr;
      dst[o + 1] = src[i0 + 1] + xi;
      dst[o2] = src[i0] - xr;
      dst[o2 + 1] = src[i0 + 1] - xi;
    }
    const t = src;
    src = dst;
    dst = t;
  }
  data.set(src);
  return data;
}

/**
 * Direct O(n^2) DFT, used only by tests as an independent check.
 * @param {Float64Array} data interleaved complex, length 2n
 * @param {boolean} [inverse]
 * @returns {Float64Array} new array
 */
export function dft(data, inverse = false) {
  const n = data.length >> 1;
  const out = new Float64Array(2 * n);
  const sgn = inverse ? 1 : -1;
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let t = 0; t < n; t++) {
      const a = (sgn * 2 * Math.PI * ((k * t) % n)) / n;
      const c = Math.cos(a);
      const s = Math.sin(a);
      sr += data[2 * t] * c - data[2 * t + 1] * s;
      si += data[2 * t] * s + data[2 * t + 1] * c;
    }
    out[2 * k] = sr;
    out[2 * k + 1] = si;
  }
  return out;
}

/**
 * 4-term Blackman-Harris window, periodic form w(n) = a0 - a1 cos(2 pi n/N) + ...
 * @param {number} n
 * @returns {Float64Array}
 */
export function blackmanHarris(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = (2 * Math.PI * i) / n;
    w[i] = 0.35875 - 0.48829 * Math.cos(x) + 0.14128 * Math.cos(2 * x) - 0.01168 * Math.cos(3 * x);
  }
  return w;
}

/**
 * Half-plane 2D FFT of a real N x N image after mean subtraction and separable
 * windowing, scaled by 1/N. Mirrors the GPU passes reduce_mean -> fft_rows ->
 * fft_cols exactly (same order of operations, same scaling, same layout).
 *
 * Output layout: mode m = ky * (N/2+1) + kx, kx in [0, N/2], ky in [0, N-1];
 * out[2m], out[2m+1] = re, im.
 *
 * @param {ArrayLike<number>} img length N*N, row-major (y * N + x)
 * @param {number} N
 * @param {Float64Array} win length N
 * @param {Float64Array} [out] optional output, length 2 * N * (N/2+1)
 * @returns {Float64Array}
 */
export function spectrum2dHalf(img, N, win, out) {
  const NH = (N >> 1) + 1;
  if (!out) out = new Float64Array(2 * N * NH);
  let mean = 0;
  for (let i = 0; i < N * N; i++) mean += img[i];
  mean /= N * N;
  const row = new Float64Array(2 * N);
  const rows = new Float64Array(2 * N * NH); // rows[y][kx]
  for (let y = 0; y < N; y++) {
    const wy = win[y];
    for (let x = 0; x < N; x++) {
      row[2 * x] = (img[y * N + x] - mean) * win[x] * wy;
      row[2 * x + 1] = 0;
    }
    fft(row);
    for (let kx = 0; kx < NH; kx++) {
      rows[2 * (y * NH + kx)] = row[2 * kx];
      rows[2 * (y * NH + kx) + 1] = row[2 * kx + 1];
    }
  }
  const col = new Float64Array(2 * N);
  const s = 1 / N;
  for (let kx = 0; kx < NH; kx++) {
    for (let y = 0; y < N; y++) {
      col[2 * y] = rows[2 * (y * NH + kx)];
      col[2 * y + 1] = rows[2 * (y * NH + kx) + 1];
    }
    fft(col);
    for (let ky = 0; ky < N; ky++) {
      out[2 * (ky * NH + kx)] = col[2 * ky] * s;
      out[2 * (ky * NH + kx) + 1] = col[2 * ky + 1] * s;
    }
  }
  return out;
}
