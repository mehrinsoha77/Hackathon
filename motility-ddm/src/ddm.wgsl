// MotilityDDM compute kernels.
//
// Per frame:     ingest -> reduce_mean -> fft_rows -> fft_cols   (one submit)
// Per analysis:  temporal (8 dispatches of 4128 workgroups) -> radial
//
// Default WebGPU limits only: largest binding is `hist` at 67.6 MB (< 128 MiB
// maxStorageBufferBindingSize), largest workgroup footprint 10240 bytes
// (< 16384 maxComputeWorkgroupStorageSize), <= 7 storage buffers per pipeline.
//
// FFTs are radix-2 Stockham in workgroup memory, twiddles come from a table
// computed on the host in float64 (no sin/cos in any kernel: identical results on
// drivers with imprecise f32 transcendentals, and no long trig chains for the
// compiler to choke on). The float64 twin of every kernel is in src/ddm_ref.js.

const N: u32 = 256u;        // ROI side
const NH: u32 = 129u;       // N/2 + 1, Hermitian half plane
const T: u32 = 256u;        // history slots
const TPAD: u32 = 512u;     // zero-padded temporal FFT length
const L: u32 = 128u;        // output lags
const NB: u32 = 32u;        // radial bins

struct FrameParams {
  roi_x0: u32,
  roi_y0: u32,
  slot: u32,
  _pad: u32,
};

struct TemporalParams {
  mode_offset: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

// Host-computed (float64) per-analysis metadata shared by every mode:
// time index j -> ring slot, validity mask, valid pair counts N(tau), and the
// spectrum of the zero-padded mask M_k.
struct TemporalMeta {
  order: array<u32, 256>,
  mask: array<f32, 256>,
  npairs: array<f32, 128>,
  mspec: array<vec2<f32>, 512>,
};

@group(0) @binding(0) var ext: texture_external;
@group(0) @binding(1) var<storage, read_write> frame: array<f32>;          // N*N luma
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;          // [0] mean hi, [1] mean lo
@group(0) @binding(3) var<storage, read> win: array<f32>;                  // Blackman-Harris, N
@group(0) @binding(4) var<storage, read> tw: array<vec2<f32>>;             // exp(-2 pi i m / 512), m < 256
@group(0) @binding(5) var<storage, read_write> rows: array<vec2<f32>>;     // N * NH
@group(0) @binding(6) var<storage, read_write> hist: array<vec2<f32>>;     // (mode * T + slot)
@group(0) @binding(7) var<uniform> pf: FrameParams;
@group(0) @binding(8) var<uniform> pt: TemporalParams;
@group(0) @binding(9) var<storage, read> tmeta: TemporalMeta;
@group(0) @binding(10) var<storage, read_write> outD: array<f32>;          // mode * L + tau
@group(0) @binding(11) var<storage, read_write> outC: array<vec2<f32>>;    // mode * L + tau
@group(0) @binding(12) var<storage, read_write> outP: array<f32>;          // mode
@group(0) @binding(13) var<storage, read> binOff: array<u32>;              // NB + 1
@group(0) @binding(14) var<storage, read> binModes: array<vec4<i32>>;      // (mode, kx, ky, 0)
@group(0) @binding(15) var<storage, read_write> radOut: array<f32>;        // D[b*L+tau], then P[b]
@group(0) @binding(16) var<storage, read_write> radDrift: array<f32>;      // ((b*L+tau)*8 + c)

// 4096 + 4096 + 2048 = 10240 bytes for the FFT kernels.
var<workgroup> wa: array<vec2<f32>, 512>;
var<workgroup> wb: array<vec2<f32>, 512>;
var<workgroup> wtw: array<vec2<f32>, 256>;
var<workgroup> wsum: array<f32, 256>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Radix-2 Stockham FFT of length 2^logn on wa/wb, one butterfly per invocation
// (tid < 2^(logn-1)). Pass s reads one buffer and writes the other, so the result
// is in wa when logn is even (256-point) and in wb when logn is odd (512-point).
// The caller must place a barrier between filling the input and calling this.
fn fft_wg(logn: u32, tid: u32, inverse: bool) {
  let half = 1u << (logn - 1u);
  for (var s = 0u; s < logn; s = s + 1u) {
    let ns = 1u << s;
    let k = tid & (ns - 1u);
    var w = wtw[k * (256u >> s)];
    if (inverse) {
      w.y = -w.y;
    }
    let o = (tid - k) * 2u + k;
    if ((s & 1u) == 0u) {
      let a = wa[tid];
      let b = cmul(wa[tid + half], w);
      wb[o] = a + b;
      wb[o + ns] = a - b;
    } else {
      let a = wb[tid];
      let b = cmul(wb[tid + half], w);
      wa[o] = a + b;
      wa[o + ns] = a - b;
    }
    workgroupBarrier();
  }
}

// ---------------------------------------------------------------- per frame

// Native-resolution luma of the centred N x N ROI, no resampling.
@compute @workgroup_size(16, 16)
fn ingest(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= N || g.y >= N) {
    return;
  }
  let c = textureLoad(ext, vec2<u32>(pf.roi_x0 + g.x, pf.roi_y0 + g.y));
  frame[g.y * N + g.x] = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
}

fn tree_sum(tid: u32, v: f32) -> f32 {
  wsum[tid] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (tid < st) {
      wsum[tid] = wsum[tid] + wsum[tid + st];
    }
    workgroupBarrier();
  }
  let total = wsum[0];
  workgroupBarrier(); // everyone has read wsum[0] before it is reused
  return total;
}

// Mean as an unevaluated f32 pair (hi, lo): hi is a first-pass estimate, lo the
// mean of the residuals about it. fft_rows subtracts hi and then lo, so the
// mean is never rounded to a single f32. Any systematic mean error leaks into
// the low-k modes multiplied by (sum w)^2 / N ~ 33, which made the DC mode the
// worst spectral error with a single-pass f32 mean.
@compute @workgroup_size(256)
fn reduce_mean(@builtin(local_invocation_index) tid: u32) {
  var s = 0.0;
  for (var i = tid; i < N * N; i = i + 256u) {
    s = s + frame[i];
  }
  let hi = tree_sum(tid, s) / f32(N * N);
  var r = 0.0;
  for (var i = tid; i < N * N; i = i + 256u) {
    r = r + (frame[i] - hi);
  }
  let lo = tree_sum(tid, r) / f32(N * N);
  if (tid == 0u) {
    stats[0] = hi;
    stats[1] = lo;
  }
}

// One workgroup per image row: mean removal, separable Blackman-Harris window,
// 256-point FFT along x, keep kx = 0..128.
@compute @workgroup_size(128)
fn fft_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let y = wid.x;
  wtw[tid] = tw[tid];
  wtw[tid + 128u] = tw[tid + 128u];
  let mean_hi = stats[0];
  let mean_lo = stats[1];
  let wy = win[y];
  for (var r = 0u; r < 2u; r = r + 1u) {
    let x = tid + r * 128u;
    wa[x] = vec2<f32>(((frame[y * N + x] - mean_hi) - mean_lo) * win[x] * wy, 0.0);
  }
  workgroupBarrier();
  fft_wg(8u, tid, false); // result in wa
  rows[y * NH + tid] = wa[tid];
  if (tid == 0u) {
    rows[y * NH + 128u] = wa[128u];
  }
}

// One workgroup per kx column: 256-point FFT along y, scale 1/N, write the
// spectrum into ring slot pf.slot of the history.
@compute @workgroup_size(128)
fn fft_cols(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let kx = wid.x;
  wtw[tid] = tw[tid];
  wtw[tid + 128u] = tw[tid + 128u];
  for (var r = 0u; r < 2u; r = r + 1u) {
    let ky = tid + r * 128u;
    wa[ky] = rows[ky * NH + kx];
  }
  workgroupBarrier();
  fft_wg(8u, tid, false); // result in wa
  let s = 1.0 / f32(N);
  for (var r = 0u; r < 2u; r = r + 1u) {
    let ky = tid + r * 128u;
    hist[(ky * NH + kx) * T + pf.slot] = wa[ky] * s;
  }
}

// ---------------------------------------------------------------- per analysis

// One workgroup per Fourier mode. Masked Wiener-Khinchin estimators with
// y(t) = m(t) x(t), p(t) = |y(t)|^2, zero padded to 512:
//   A(tau) = sum_t m(t) p(t+tau) = IFFT(conj(M) P),  B(tau) = A(-tau)
//   G(tau) = sum_t conj(y(t)) y(t+tau) = IFFT(|Y|^2)
//   D = (A + B - 2 Re G) / N(tau),  C = G / N(tau),  P = sum p / sum m
// M and N(tau) are mode independent and come from the host in float64.
@compute @workgroup_size(256)
fn temporal(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let mode = wid.x + pt.mode_offset;
  wtw[tid] = tw[tid];
  let x = hist[mode * T + tmeta.order[tid]] * tmeta.mask[tid];

  // 1. P_k = FFT(p)
  wa[tid] = vec2<f32>(dot(x, x), 0.0);
  wa[tid + 256u] = vec2<f32>(0.0, 0.0);
  workgroupBarrier();
  fft_wg(9u, tid, false); // result in wb
  let psum = wb[0].x;
  for (var r = 0u; r < 2u; r = r + 1u) {
    let k = tid + r * 256u;
    let m = tmeta.mspec[k];
    wa[k] = cmul(vec2<f32>(m.x, -m.y), wb[k]);
  }
  workgroupBarrier();
  fft_wg(9u, tid, true); // A(tau) * 512 in wb
  let inv = 1.0 / f32(TPAD);
  var a_tau = 0.0;
  var b_tau = 0.0;
  if (tid < L) {
    a_tau = wb[tid].x * inv;
    b_tau = wb[(TPAD - tid) & (TPAD - 1u)].x * inv;
  }
  workgroupBarrier();

  // 2. G = IFFT(|FFT(y)|^2)
  wa[tid] = x;
  wa[tid + 256u] = vec2<f32>(0.0, 0.0);
  workgroupBarrier();
  fft_wg(9u, tid, false); // Y in wb
  for (var r = 0u; r < 2u; r = r + 1u) {
    let k = tid + r * 256u;
    let yk = wb[k];
    wa[k] = vec2<f32>(dot(yk, yk), 0.0);
  }
  workgroupBarrier();
  fft_wg(9u, tid, true); // G(tau) * 512 in wb

  if (tid < L) {
    let np = tmeta.npairs[tid];
    let o = mode * L + tid;
    if (np > 0.5) {
      let g = wb[tid] * inv;
      outD[o] = (a_tau + b_tau - 2.0 * g.x) / np;
      outC[o] = g / np;
    } else {
      outD[o] = 0.0;
      outC[o] = vec2<f32>(0.0, 0.0);
    }
  }
  if (tid == 0u) {
    let n0 = tmeta.npairs[0];
    outP[mode] = select(0.0, psum / n0, n0 > 0.5);
  }
}

// One workgroup per radial bin, one invocation per lag: bin-averaged D, static
// power P, and the phase-drift least-squares sums for the lag pair (tau, tau+1):
//   z = C(tau+1) conj C(tau), w = |z|, phi = arg z
//   [w, w kx^2, w kx ky, w ky^2, w kx phi, w ky phi, w phi^2, count(w > 0)]
@compute @workgroup_size(128)
fn radial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let b = wid.x;
  let i0 = binOff[b];
  let i1 = binOff[b + 1u];
  var dsum = 0.0;
  var psum = 0.0;
  var sw = 0.0;
  var sxx = 0.0;
  var sxy = 0.0;
  var syy = 0.0;
  var rx = 0.0;
  var ry = 0.0;
  var rpp = 0.0;
  var cnt = 0.0;
  for (var i = i0; i < i1; i = i + 1u) {
    let e = binModes[i];
    let m = u32(e.x);
    let kx = f32(e.y);
    let ky = f32(e.z);
    dsum = dsum + outD[m * L + t];
    if (t == 0u) {
      psum = psum + outP[m];
    }
    if (t + 1u < L) {
      let a = outC[m * L + t];
      let c = outC[m * L + t + 1u];
      let z = vec2<f32>(c.x * a.x + c.y * a.y, c.y * a.x - c.x * a.y);
      let w = length(z);
      if (w > 0.0) {
        let ph = atan2(z.y, z.x);
        sw = sw + w;
        sxx = sxx + w * kx * kx;
        sxy = sxy + w * kx * ky;
        syy = syy + w * ky * ky;
        rx = rx + w * kx * ph;
        ry = ry + w * ky * ph;
        rpp = rpp + w * ph * ph;
        cnt = cnt + 1.0;
      }
    }
  }
  let n = f32(i1 - i0);
  let inv = select(0.0, 1.0 / n, i1 > i0);
  radOut[b * L + t] = dsum * inv;
  if (t == 0u) {
    radOut[NB * L + b] = psum * inv;
  }
  let o = (b * L + t) * 8u;
  radDrift[o] = sw;
  radDrift[o + 1u] = sxx;
  radDrift[o + 2u] = sxy;
  radDrift[o + 3u] = syy;
  radDrift[o + 4u] = rx;
  radDrift[o + 5u] = ry;
  radDrift[o + 6u] = rpp;
  radDrift[o + 7u] = cnt;
}
