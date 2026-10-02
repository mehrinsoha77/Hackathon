// Motility analysis of radially averaged DDM data.
//
// Model (Wilson et al., PRL 106, 018101, 2011), per q-bin b and lag tau:
//
//   D_b(tau) = A_b [1 - E_b(tau) (1 - alpha + alpha g(q_b vbar tau))] + B_b
//   E_b(tau) = exp(-Dd q_b^2 tau) * J0(q_b |V| tau)
//
// i.e. swimmer amplitude W_b = alpha A_b, passive amplitude P_b = (1 - alpha) A_b
// and camera noise floor B_b. alpha is q-independent (swimmers and passive cells
// are the same scatterers). The linear amplitudes A_b, B_b >= 0 are solved
// exactly by NNLS inside every objective evaluation (variable projection,
// VarPro), so the nonlinear search only sees (vbar, Z, Dd, alpha); the model is
// averaged over each bin's effective |q| nodes. g is the swimmer self-ISF
// averaged over a Schulz speed distribution:
//   3D isotropic swimmers seen in projection:  <sin(q v tau)/(q v tau)>
//   2D in-plane swimmers:                      <J0(q v tau)>
// J0(q |V| tau) is the radial average of exp(-i q.V tau) and compensates a
// uniform drift V estimated from Fourier phases (see estimateDrift).
//
// Five falsification gates turn the fit into a verdict:
//   G1 temporal structure index TSI >= 0.15             else NO_DYNAMICS
//   G2 no uniform flow: not (|V| > 3 sigma_V and |V| > 2 um/s)
//                                                      else DIRECTED_TRANSPORT
//   G3 swimmer model preferred, Delta BIC > 10          else NOT_MOTILE
//   G4 motile contrast share alpha > 0.05               else NOT_MOTILE
//   G5 exposure locked                                  else UNLOCKED_EXPOSURE

export const GATES = Object.freeze({
  TSI_MIN: 0.15,
  FLOW_SIGMA: 3,
  FLOW_MIN_UM_S: 2,
  DBIC_MIN: 10,
  ALPHA_MIN: 0.05,
  DRIFT_ISF_MAX: 0.2,
  // Swimmer speeds below the flow floor are not claimable: a sub-2 um/s
  // "swimmer" is indistinguishable from residual convection or drift.
  SWIM_MIN_UM_S: 2,
});

export const VERDICTS = Object.freeze([
  'MOTILE',
  'NOT_MOTILE',
  'DIRECTED_TRANSPORT',
  'NO_DYNAMICS',
  'UNLOCKED_EXPOSURE',
]);

// ---------------------------------------------------------------- special functions

/**
 * Bessel J0 (rational approximations, |error| < 1e-8).
 * @param {number} x
 */
export function besselJ0(x) {
  const ax = Math.abs(x);
  if (ax < 8) {
    const y = x * x;
    const a1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))));
    const a2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
    return a1 / a2;
  }
  const z = 8 / ax;
  const y = z * z;
  const xx = ax - 0.785398164;
  const a1 = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  const a2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
  return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * a1 - z * Math.sin(xx) * a2);
}

/**
 * ln Gamma(x), Lanczos approximation (g = 7, n = 9), x > 0.
 * @param {number} x
 */
export function lnGamma(x) {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lnGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Schulz (Gamma) speed distribution with mean vbar and width parameter
 * Z = (vbar/sigma)^2 - 1.
 * @param {number} v
 * @param {number} vbar
 * @param {number} Z
 */
export function schulzPdf(v, vbar, Z) {
  if (v <= 0) return 0;
  const b = (Z + 1) / vbar;
  return Math.exp(Z * Math.log(v) - lnGamma(Z + 1) + (Z + 1) * Math.log(b) - b * v);
}

/** sigma_v of the Schulz distribution. */
export function schulzSigma(vbar, Z) {
  return vbar / Math.sqrt(Z + 1);
}

/** Z from mean and standard deviation. */
export function schulzZ(vbar, sigma) {
  return (vbar / sigma) ** 2 - 1;
}

/**
 * Legendre function P_nu(c), real nu >= 0, 0 < c <= 1, from Laplace's first
 * integral P_nu(cos t) = (1/pi) int_0^pi (cos t + i sin t cos phi)^nu dphi.
 * The integrand is smooth, even and 2 pi periodic in phi, so the trapezoid rule
 * converges exponentially; the phase nu * arg(...) spans at most nu * t, which
 * sets the node count (worst-case error ~4e-6 over nu in [0.5, 100]).
 * @param {number} nu
 * @param {number} c
 */
export function legendreP(nu, c) {
  const s = Math.sqrt(Math.max(0, 1 - c * c));
  const n = 16 + Math.ceil(nu * Math.atan2(s, c));
  let sum = 0;
  for (let i = 0; i <= n; i++) {
    const phi = (Math.PI * i) / n;
    const im = s * Math.cos(phi);
    const r2 = c * c + im * im;
    const val = Math.exp(0.5 * nu * Math.log(r2)) * Math.cos(nu * Math.atan2(im, c));
    sum += i === 0 || i === n ? 0.5 * val : val;
  }
  return sum / n;
}

/**
 * 3D isotropic swimmer self-ISF, Schulz-averaged <sin(q v tau)/(q v tau)>.
 * Closed form: sin(Z atan L) / (Z L (1 + L^2)^(Z/2)), L = x/(Z+1), x = q vbar tau.
 * @param {number} x q * vbar * tau (dimensionless)
 * @param {number} Z
 */
export function isf3d(x, Z) {
  const L = x / (Z + 1);
  if (L < 1e-4) return 1 - (x * x * (Z + 2)) / (6 * (Z + 1));
  return Math.sin(Z * Math.atan(L)) / (Z * L * Math.exp(0.5 * Z * Math.log1p(L * L)));
}

/**
 * 2D in-plane swimmer self-ISF, Schulz-averaged <J0(q v tau)>.
 * Closed form (Laplace transform of v^Z J0): P_Z(1/sqrt(1+L^2)) / (1+L^2)^((Z+1)/2).
 * @param {number} x q * vbar * tau
 * @param {number} Z
 */
export function isf2d(x, Z) {
  const L = x / (Z + 1);
  if (L < 1e-4) return 1 - (x * x * (Z + 2)) / (4 * (Z + 1));
  const u = 1 + L * L;
  const lnPre = -0.5 * (Z + 1) * Math.log(u);
  if (lnPre < -25) return 0; // |P_Z| <= 1, so |g| < 1.4e-11
  return legendreP(Z, 1 / Math.sqrt(u)) * Math.exp(lnPre);
}

// Fast 2D kernel for fitting. isf2d needs a Legendre quadrature per call, too slow
// inside an optimizer (~10^6 calls per fit). We tabulate it once on a uniform
// grid in ln Z (0.5..100) x ln Lambda. Along ln Lambda the oscillation phase of
// P_Z(cos atan L) is ~(Z+1/2) atan L, whose rate d/d(ln L) is at most (Z+1/2)/2;
// the step is chosen so that one step advances the phase by <= 0.1 rad, and
// rows end where (1+L^2)^(-(Z+1)/2) < e^-25. Catmull-Rom along ln Lambda,
// quadratic Lagrange across ln Z. Max deviation from isf2d: see t_kernels.mjs.
const T2D = { built: false, nz: 72, lz0: Math.log(0.5), lz1: Math.log(100), dlz: 0, rows: [] };
const LSER = 1e-4;

function buildTable2d() {
  T2D.dlz = (T2D.lz1 - T2D.lz0) / (T2D.nz - 1);
  for (let i = 0; i < T2D.nz; i++) {
    const Z = Math.exp(T2D.lz0 + i * T2D.dlz);
    const Lc = Math.min(1e4, Math.sqrt(Math.expm1(50 / (Z + 1))));
    const u0 = Math.log(LSER), u1 = Math.log(Lc);
    const du = Math.min(0.04, 0.1 / ((Z + 0.5) / 2));
    const n = Math.ceil((u1 - u0) / du) + 1;
    const v = new Float64Array(n + 2);
    for (let k = 0; k < n + 2; k++) v[k] = isf2dExactLambda(Math.exp(u0 + k * du), Z);
    T2D.rows.push({ Z, u0, du, n, v });
  }
  T2D.built = true;
}

function isf2dExactLambda(L, Z) {
  const u = 1 + L * L;
  const lnPre = -0.5 * (Z + 1) * Math.log(u);
  if (lnPre < -25) return 0;
  return legendreP(Z, 1 / Math.sqrt(u)) * Math.exp(lnPre);
}

function rowEval(r, x) {
  const L = x / (r.Z + 1);
  if (L < LSER) return 1 - (x * x * (r.Z + 2)) / (4 * (r.Z + 1));
  const f = (Math.log(L) - r.u0) / r.du;
  if (f >= r.n - 1) return 0;
  const i = Math.floor(f);
  const t = f - i;
  const v = r.v;
  const p1 = v[i], p2 = v[i + 1], p3 = v[i + 2];
  const p0 = i > 0 ? v[i - 1] : 2 * p1 - p2;
  return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
}

/**
 * Tabulated 2D in-plane swimmer ISF used by the fitter (same value as isf2d to
 * within the tolerance asserted in test/t_kernels.mjs).
 * @param {number} x
 * @param {number} Z
 */
export function isf2dFast(x, Z) {
  if (!T2D.built) buildTable2d();
  const zf = (Math.log(clampZ(Z)) - T2D.lz0) / T2D.dlz;
  const i0 = Math.min(Math.max(Math.round(zf) - 1, 0), T2D.nz - 3);
  const t = zf - i0;
  const g0 = rowEval(T2D.rows[i0], x), g1 = rowEval(T2D.rows[i0 + 1], x), g2 = rowEval(T2D.rows[i0 + 2], x);
  return 0.5 * (t - 1) * (t - 2) * g0 - t * (t - 2) * g1 + 0.5 * t * (t - 1) * g2;
}
const clampZ = (Z) => Math.min(100, Math.max(0.5, Z));

/**
 * Kernel used by the fitter for a geometry.
 * @param {'3d'|'2d'} geometry
 * @returns {(x: number, Z: number) => number}
 */
export function swimmerKernel(geometry) {
  return geometry === '2d' ? isf2dFast : isf3d;
}

// ---------------------------------------------------------------- NNLS (3 columns)

/**
 * Exact non-negative least squares for <= 3 columns from the Gram matrix:
 * the NNLS optimum is the unconstrained optimum on its own support, so we
 * enumerate all supports, keep the feasible ones and take the smallest RSS.
 *
 * @param {number[]} G 3x3 Gram matrix, row-major
 * @param {number[]} h C^T d
 * @param {number} dd d^T d
 * @returns {{a: number[], rss: number}}
 */
export function nnls3(G, h, dd) {
  let best = { a: [0, 0, 0], rss: dd };
  for (let mask = 1; mask < 8; mask++) {
    const idx = [];
    for (let i = 0; i < 3; i++) if (mask & (1 << i)) idx.push(i);
    const k = idx.length;
    const A = idx.map((i) => idx.map((j) => G[3 * i + j]));
    const b = idx.map((i) => h[i]);
    const x = solveSmall(A, b);
    if (!x) continue;
    if (x.some((v) => v < 0)) continue;
    let rss = dd;
    for (let i = 0; i < k; i++) rss -= x[i] * b[i];
    if (rss < best.rss) {
      const a = [0, 0, 0];
      idx.forEach((i, n) => (a[i] = x[n]));
      best = { a, rss: Math.max(rss, 0) };
    }
  }
  return best;
}

/**
 * Gaussian elimination with partial pivoting for k <= 3; null if singular.
 * @param {number[][]} A
 * @param {number[]} b
 */
function solveSmall(A, b) {
  const k = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  const scale = Math.max(...A.map((r, i) => Math.abs(r[i])), 1e-300);
  for (let c = 0; c < k; c++) {
    let p = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-11 * scale) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < k; r++) {
      const f = M[r][c] / M[c][c];
      for (let j = c; j <= k; j++) M[r][j] -= f * M[c][j];
    }
  }
  const x = new Array(k).fill(0);
  for (let r = k - 1; r >= 0; r--) {
    let s = M[r][k];
    for (let j = r + 1; j < k; j++) s -= M[r][j] * x[j];
    x[r] = s / M[r][r];
  }
  return x;
}

// ---------------------------------------------------------------- Nelder-Mead

/**
 * Bounded Nelder-Mead (bounds enforced by clamping inside the objective).
 * @param {(u: number[]) => number} f
 * @param {number[]} x0
 * @param {number[]} step
 * @param {{maxIter?: number, tol?: number}} [opt]
 */
export function nelderMead(f, x0, step, opt = {}) {
  const maxIter = opt.maxIter ?? 400;
  const tol = opt.tol ?? 1e-9;
  const n = x0.length;
  let simplex = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const x = x0.slice();
    x[i] += step[i];
    simplex.push(x);
  }
  let fs = simplex.map(f);
  let evals = n + 1;
  for (let it = 0; it < maxIter; it++) {
    const ord = fs.map((v, i) => i).sort((a, b) => fs[a] - fs[b]);
    simplex = ord.map((i) => simplex[i]);
    fs = ord.map((i) => fs[i]);
    if (Math.abs(fs[n] - fs[0]) <= tol * (Math.abs(fs[0]) + 1e-30)) break;
    const c = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += simplex[i][j] / n;
    const pt = (t) => c.map((cj, j) => cj + t * (simplex[n][j] - cj));
    const xr = pt(-1);
    const fr = f(xr);
    evals++;
    if (fr < fs[0]) {
      const xe = pt(-2);
      const fe = f(xe);
      evals++;
      if (fe < fr) [simplex[n], fs[n]] = [xe, fe];
      else [simplex[n], fs[n]] = [xr, fr];
    } else if (fr < fs[n - 1]) {
      [simplex[n], fs[n]] = [xr, fr];
    } else {
      const outside = fr < fs[n];
      const xc = pt(outside ? -0.5 : 0.5);
      const fc = f(xc);
      evals++;
      if (fc < (outside ? fr : fs[n])) [simplex[n], fs[n]] = [xc, fc];
      else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]));
          fs[i] = f(simplex[i]);
          evals++;
        }
      }
    }
  }
  const ib = fs.indexOf(Math.min(...fs));
  return { x: simplex[ib], f: fs[ib], evals };
}

// ---------------------------------------------------------------- data preparation

/**
 * Quasi-logarithmic lag grid: every lag 1..10, then ~8 % steps up to L-1.
 * @param {number} L
 */
export function lagGrid(L) {
  const lags = [];
  for (let t = 1; t <= Math.min(10, L - 1); t++) lags.push(t);
  let x = 10;
  for (;;) {
    x *= 1.08;
    const t = Math.round(x);
    if (t > L - 1) break;
    if (t > lags[lags.length - 1]) lags.push(t);
  }
  return lags;
}

/**
 * @typedef {object} AnalysisOptions
 * @property {number} dxUm pixel pitch at the sample plane, um/px
 * @property {number} fps nominal frame rate (frame interval of one lag)
 * @property {'3d'|'2d'} [geometry]
 * @property {boolean} [exposureLocked]
 */

/**
 * @param {import('./ddm_ref.js').Radial} radial
 * @param {AnalysisOptions} opt
 */
export function prepare(radial, opt) {
  const { N, NB, L } = radial;
  const dt = 1 / opt.fps;
  const lags = lagGrid(L);
  const tau = lags.map((l) => l * dt);
  const bins = [];
  for (let b = 0; b < NB; b++) {
    if (radial.nModes[b] < 3) continue;
    const d = lags.map((l) => radial.D[b * L + l]);
    const scale = Math.max(...d);
    if (!(scale > 0) || !Number.isFinite(scale)) continue;
    const qs = (2 * Math.PI) / (N * opt.dxUm);
    const q = qs * radial.kCenter[b];
    const K = radial.kNodes ? radial.kNodes.length / NB : 0;
    const qn = K ? Array.from(radial.kNodes.subarray(b * K, b * K + K), (k) => k * qs) : [q];
    bins.push({ b, q, qn, d: Float64Array.from(d), w: 1 / scale });
  }
  return { bins, lags, tau, dt, L, tauMax: (L - 1) * dt };
}

/** @typedef {ReturnType<typeof prepare>} Prepared */

// ---------------------------------------------------------------- models

/**
 * Model column c_b(tau) = 1 - <E (1 - alpha + alpha g)>_nodes for one bin, the
 * normalized decorrelation; D_b = A_b c_b + B_b. With swim = false, alpha = 0.
 * @param {Prepared} pr
 * @param {Prepared['bins'][number]} bin
 * @param {ModelParams} p
 * @param {(x: number, Z: number) => number} kernel
 * @param {boolean} swim
 * @param {Float64Array} out length = pr.tau.length
 */
export function modelColumn(pr, bin, p, kernel, swim, out) {
  const qn = bin.qn;
  const inv = 1 / qn.length;
  const alpha = swim ? p.alpha : 0;
  for (let j = 0; j < pr.tau.length; j++) {
    const t = pr.tau[j];
    let f = 0;
    for (const q of qn) {
      let E = Math.exp(-p.Dd * q * q * t);
      if (p.Vmag > 0) E *= besselJ0(q * p.Vmag * t);
      f += alpha > 0 ? E * (1 - alpha + alpha * kernel(q * p.vbar * t, p.Z)) : E;
    }
    out[j] = 1 - f * inv;
  }
  return out;
}

/**
 * @typedef {object} ModelParams
 * @property {number} vbar mean swimming speed (um/s)
 * @property {number} Z Schulz width parameter
 * @property {number} Dd diffusion coefficient (um^2/s)
 * @property {number} Vmag uniform drift speed compensated in the model (um/s)
 * @property {number} alpha swimmer fraction of the dynamic contrast, q-independent
 */

/**
 * Evaluate a model with exact NNLS amplitudes A_b, B_b >= 0 per bin (VarPro).
 * @param {Prepared} pr
 * @param {ModelParams} p
 * @param {(x: number, Z: number) => number} kernel
 * @param {boolean} swim include the swimmer term
 */
export function evalModel(pr, p, kernel, swim) {
  const nt = pr.tau.length;
  let rss = 0;
  const amps = [];
  const c = new Float64Array(nt);
  const alpha = swim ? p.alpha : 0;
  for (const bin of pr.bins) {
    modelColumn(pr, bin, p, kernel, swim, c);
    let g11 = 0, g12 = 0, h1 = 0, h2 = 0, dd = 0;
    const d = bin.d;
    for (let j = 0; j < nt; j++) {
      g11 += c[j] * c[j];
      g12 += c[j];
      h1 += c[j] * d[j];
      h2 += d[j];
      dd += d[j] * d[j];
    }
    // 2-column NNLS (A, B) through the 3-column solver with an empty third column
    const res = nnls3([g11, g12, 0, g12, nt, 0, 0, 0, 0], [h1, h2, 0], dd);
    rss += res.rss * bin.w * bin.w;
    const A = res.a[0];
    amps.push({ A, B: res.a[1], W: alpha * A, P: (1 - alpha) * A });
  }
  return { rss, amps };
}

/**
 * Nonlinear parameter bounds implied by the accessible (q, tau) window.
 * @param {Prepared} pr
 */
function bounds(pr) {
  const qs = pr.bins.map((b) => b.q);
  const qmin = Math.min(...qs), qmax = Math.max(...qs);
  return {
    vmin: Math.max(GATES.SWIM_MIN_UM_S, 0.3 / (qmax * pr.tauMax)),
    vmax: 3 / (qmin * pr.dt),
    Dmin: 0.01 / (qmax * qmax * pr.tauMax),
    Dmax: 10 / (qmin * qmin * pr.dt),
    Zmin: 0.5,
    Zmax: 100,
  };
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/**
 * Brownian-only fit (1 nonlinear parameter): log-grid scan + golden section.
 * @param {Prepared} pr
 * @param {number} Vmag
 */
export function fitBrownian(pr, Vmag = 0) {
  const B = bounds(pr);
  const f = (lnD) => evalModel(pr, { vbar: 0, Z: 1, Dd: Math.exp(lnD), Vmag, alpha: 0 }, isf3d, false).rss;
  const lo = Math.log(B.Dmin), hi = Math.log(B.Dmax);
  const n = 48;
  let bi = 0, bf = Infinity;
  const xs = [];
  for (let i = 0; i <= n; i++) {
    const x = lo + ((hi - lo) * i) / n;
    xs.push(x);
    const v = f(x);
    if (v < bf) [bf, bi] = [v, i];
  }
  let a = xs[Math.max(0, bi - 1)], b = xs[Math.min(n, bi + 1)];
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a), d = a + gr * (b - a);
  let fc = f(c), fd = f(d);
  for (let it = 0; it < 60 && b - a > 1e-6; it++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - gr * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + gr * (b - a);
      fd = f(d);
    }
  }
  const lnD = fc < fd ? c : d;
  const Dd = Math.exp(lnD);
  const ev = evalModel(pr, { vbar: 0, Z: 1, Dd, Vmag, alpha: 0 }, isf3d, false);
  return { Dd, vbar: 0, Z: 1, Vmag, alpha: 0, rss: ev.rss, amps: ev.amps, k: 1 + 2 * pr.bins.length };
}

/**
 * Passive null model with a uniform drift whose speed is refined inside the
 * band allowed by the phase measurement, [vlo, vhi]. Nonlinear (ln Dd, |V|).
 * @param {Prepared} pr
 * @param {number} vlo
 * @param {number} vhi
 */
export function fitBrownianDrift(pr, vlo, vhi) {
  const B = bounds(pr);
  const unpack = (u) => ({
    vbar: 0,
    Z: 1,
    Dd: Math.exp(clamp(u[0], Math.log(B.Dmin), Math.log(B.Dmax))),
    Vmag: clamp(u[1], vlo, vhi),
    alpha: 0,
  });
  const f = (u) => evalModel(pr, unpack(u), isf3d, false).rss;
  let best = null;
  for (let i = 0; i <= 24; i++) {
    const lD = Math.log(B.Dmin) + ((Math.log(B.Dmax) - Math.log(B.Dmin)) * i) / 24;
    for (let j = 0; j <= 6; j++) {
      const u = [lD, vlo + ((vhi - vlo) * j) / 6];
      const v = f(u);
      if (!best || v < best.f) best = { u, f: v };
    }
  }
  const r = nelderMead(f, best.u, [0.2, Math.max((vhi - vlo) / 6, 1e-3)], { maxIter: 200, tol: 1e-10 });
  const p = unpack(r.x);
  const ev = evalModel(pr, p, isf3d, false);
  return { ...p, rss: ev.rss, amps: ev.amps, k: 2 + 2 * pr.bins.length };
}

/**
 * Swimmer fit: VarPro over (ln vbar, ln Z, ln Dd, alpha), multi-start Nelder-Mead.
 * @param {Prepared} pr
 * @param {'3d'|'2d'} geometry
 * @param {number} Vmag
 * @param {number} Dstart starting diffusion coefficient (from the Brownian fit)
 */
export function fitSwimmer(pr, geometry, Vmag = 0, Dstart = 0) {
  const B = bounds(pr);
  const kernel = swimmerKernel(geometry);
  /** @returns {ModelParams} */
  const unpack = (u) => ({
    vbar: Math.exp(clamp(u[0], Math.log(B.vmin), Math.log(B.vmax))),
    Z: Math.exp(clamp(u[1], Math.log(B.Zmin), Math.log(B.Zmax))),
    Dd: Math.exp(clamp(u[2], Math.log(B.Dmin), Math.log(B.Dmax))),
    Vmag,
    alpha: clamp(u[3], 0, 1),
  });
  const f = (u) => evalModel(pr, unpack(u), kernel, true).rss;
  const D0 = Dstart > 0 ? clamp(Dstart, B.Dmin, B.Dmax) : Math.sqrt(B.Dmin * B.Dmax);
  // Coarse scan of vbar (the dominant nonlinearity), then refine the best starts.
  const starts = [];
  const nv = 16;
  for (let i = 0; i < nv; i++) {
    const lv = Math.log(B.vmin) + ((Math.log(B.vmax) - Math.log(B.vmin)) * (i + 0.5)) / nv;
    for (const lZ of [Math.log(1.5), Math.log(8)]) {
      for (const a0 of [0.3, 0.8]) {
        const u = [lv, lZ, Math.log(D0), a0];
        starts.push({ u, f: f(u) });
      }
    }
  }
  starts.sort((a, b) => a.f - b.f);
  let best = null;
  for (const s of starts.slice(0, 3)) {
    const r = nelderMead(f, s.u, [0.3, 0.7, 0.7, 0.2], { maxIter: 400, tol: 1e-10 });
    if (!best || r.f < best.f) best = r;
  }
  const p = unpack(best.x);
  const ev = evalModel(pr, p, kernel, true);
  return { ...p, rss: ev.rss, amps: ev.amps, k: 4 + 2 * pr.bins.length };
}

/**
 * BIC = n ln(RSS/n) + k ln n for normalized residuals.
 * @param {number} rss
 * @param {number} n effective number of independent data points
 * @param {number} k
 * @param {number} [nRaw] number of residuals RSS is summed over (defaults to n)
 */
export function bic(rss, n, k, nRaw = n) {
  return n * Math.log(Math.max(rss, 1e-300) / nRaw) + k * Math.log(n);
}

/**
 * Pooled lag-1 autocorrelation of the normalized residuals along tau. All lags
 * of a structure function are built from the same frames, so slow modes carry
 * noise that is strongly correlated across tau; treating 32 x 43 residuals as
 * independent would let any flexible model "win" BIC on correlated noise.
 * @param {Prepared} pr
 * @param {{vbar: number, Z: number, Dd: number, Vmag: number}} p
 * @param {(x: number, Z: number) => number} kernel
 * @param {boolean} swim
 */
export function residualAutocorr(pr, p, kernel, swim) {
  const ev = evalModel(pr, p, kernel, swim);
  const nt = pr.tau.length;
  const c = new Float64Array(nt);
  let num = 0, den = 0;
  pr.bins.forEach((bin, i) => {
    const { A, B } = ev.amps[i];
    modelColumn(pr, bin, p, kernel, swim, c);
    let prev = 0;
    for (let j = 0; j < nt; j++) {
      const r = (bin.d[j] - (A * c[j] + B)) * bin.w;
      den += r * r;
      if (j > 0) num += prev * r;
      prev = r;
    }
  });
  return den > 0 ? Math.min(0.99, Math.max(0, num / den)) : 0;
}

/** Effective sample size of an AR(1) residual sequence. */
export function effectiveN(n, rho) {
  return Math.max(2, (n * (1 - rho)) / (1 + rho));
}

// ---------------------------------------------------------------- drift from phases

/**
 * Weighted least squares for a uniform drift V from Fourier phase increments
 *   phi = arg[C(q, tau+1) conj C(q, tau)] = -q.V dt   (forward FFT e^{-i k r})
 * summed over modes on the GPU/CPU (Radial#drift), restricted to lag pairs where
 * the fitted swimmer self-ISF g(q vbar tau) < 0.2: swimmer phase memory has
 * decayed and the remaining coherent phase rotation is advection.
 *
 * @param {import('./ddm_ref.js').Radial} radial
 * @param {Prepared} pr
 * @param {AnalysisOptions} opt
 * @param {{vbar: number, Z: number} | null} swimmer null = no swimmer component
 */
export function estimateDrift(radial, pr, opt, swimmer) {
  const { L, N } = radial;
  const kernel = swimmerKernel(opt.geometry ?? '3d');
  const S = new Float64Array(8);
  let modesUsed = 0, pairs = 0;
  for (const bin of pr.bins) {
    const b = bin.b;
    let usedHere = false;
    for (let tau = 1; tau < L - 1; tau++) {
      if (swimmer) {
        const g = kernel(bin.q * swimmer.vbar * tau * pr.dt, swimmer.Z);
        if (!(Math.abs(g) < GATES.DRIFT_ISF_MAX)) continue;
      }
      const o = (b * L + tau) * 8;
      if (!(radial.drift[o] > 0)) continue;
      for (let c = 0; c < 8; c++) S[c] += radial.drift[o + c];
      usedHere = true;
      pairs++;
    }
    if (usedHere) modesUsed += radial.nModes[b];
  }
  const [Sw, Sxx, Sxy, Syy, Rx, Ry, Rpp, n] = S;
  const det = Sxx * Syy - Sxy * Sxy;
  const empty = { Vx: 0, Vy: 0, mag: 0, sigma: Infinity, angleDeg: 0, significant: false, pairs, samples: n };
  if (n < 10 || !(det > 1e-12 * Sxx * Syy) || !(Sw > 0)) return empty;
  // beta solves [Sxx Sxy; Sxy Syy] beta = [Rx; Ry], phi ~ kx bx + ky by
  const bx = (Syy * Rx - Sxy * Ry) / det;
  const by = (Sxx * Ry - Sxy * Rx) / det;
  const rss = Math.max(Rpp - bx * Rx - by * Ry, 0);
  // s^2 = RSS/(n-2); samples of one mode at different lags are correlated, so the
  // covariance is inflated by the mean number of lag pairs per mode.
  const inflate = Math.max(1, n / Math.max(modesUsed, 1));
  const s2 = (rss / (n - 2)) * inflate;
  const cxx = (s2 * Syy) / det, cyy = (s2 * Sxx) / det, cxy = (-s2 * Sxy) / det;
  const conv = -(N * opt.dxUm) / (2 * Math.PI * pr.dt); // beta -> V [um/s]
  const Vx = conv * bx, Vy = conv * by;
  const mag = Math.hypot(Vx, Vy);
  let varAlong;
  if (mag > 0) {
    const ux = Vx / mag, uy = Vy / mag;
    varAlong = conv * conv * (ux * ux * cxx + 2 * ux * uy * cxy + uy * uy * cyy);
  } else varAlong = (conv * conv * (cxx + cyy)) / 2;
  const sigma = Math.sqrt(Math.max(varAlong, 0));
  return {
    Vx,
    Vy,
    mag,
    sigma,
    angleDeg: (Math.atan2(Vy, Vx) * 180) / Math.PI,
    significant: mag > GATES.FLOW_SIGMA * sigma,
    pairs,
    samples: n,
  };
}

// ---------------------------------------------------------------- summary statistics

/**
 * Temporal Structure Index: per bin (D_late - D(1)) / D_late with D_late the
 * mean over lags L/2..L-1; TSI is the mean of the 4 largest bins. Frames with no
 * temporal order (shuffled, static, or pure noise) give D(1) = D_late, TSI ~ 0.
 * @param {import('./ddm_ref.js').Radial} radial
 */
export function temporalStructureIndex(radial) {
  const { NB, L } = radial;
  const r = [];
  for (let b = 0; b < NB; b++) {
    if (radial.nModes[b] < 3) continue;
    let late = 0, n = 0;
    for (let t = L >> 1; t < L; t++) (late += radial.D[b * L + t]), n++;
    late /= n;
    if (!(late > 0)) continue;
    r.push((late - radial.D[b * L + 1]) / late);
  }
  r.sort((a, b) => b - a);
  const top = r.slice(0, 4);
  return top.length ? Math.max(0, top.reduce((a, b) => a + b, 0) / top.length) : 0;
}

/**
 * The motile contrast share is the fitted, q-independent alpha. It is only
 * identifiable if at least one bin resolves the swimmer decorrelation inside
 * the window: q vbar dt <= 0.5 (not decorrelated within one frame) and
 * q vbar tau_max >= 3 (decorrelated before the last lag).
 * @param {Prepared} pr
 * @param {{vbar: number, alpha: number}} fit
 */
export function motileShare(pr, fit) {
  const resolvedBins = pr.bins.filter((bin) => bin.q * fit.vbar * pr.dt <= 0.5 && bin.q * fit.vbar * pr.tauMax >= 3).length;
  return { alpha: fit.alpha, resolved: resolvedBins > 0, resolvedBins };
}

/**
 * tau_1/2(q) from the data normalized by the fitted amplitudes, and the
 * scaling exponent mu in tau_1/2 ~ q^-mu (mu = 1 ballistic, 2 diffusive).
 * @param {import('./ddm_ref.js').Radial} radial
 * @param {Prepared} pr
 * @param {{amps: {W: number, P: number, B: number}[]}} fit
 */
export function scalingExponent(radial, pr, fit) {
  const { L } = radial;
  const q = [], th = [];
  pr.bins.forEach((bin, i) => {
    const { A, B } = fit.amps[i];
    if (!(A > 0.2 * (A + B))) return;
    let prev = 1, prevT = 0;
    for (let t = 1; t < L; t++) {
      const f = 1 - (radial.D[bin.b * L + t] - B) / A;
      if (f <= 0.5) {
        let tt;
        if (t === 1) tt = 1;
        else {
          const lt0 = Math.log(prevT), lt1 = Math.log(t);
          tt = Math.exp(lt0 + ((prev - 0.5) / (prev - f)) * (lt1 - lt0));
        }
        q.push(bin.q);
        th.push(tt * pr.dt);
        return;
      }
      prev = f;
      prevT = t;
    }
  });
  let mu = NaN;
  if (q.length >= 3) {
    const xs = q.map(Math.log), ys = th.map(Math.log);
    const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
    let sxy = 0, sxx = 0;
    xs.forEach((x, i) => ((sxy += (x - mx) * (ys[i] - my)), (sxx += (x - mx) ** 2)));
    if (sxx > 0) mu = -sxy / sxx;
  }
  return { q, tauHalf: th, mu };
}

// ---------------------------------------------------------------- top level

/**
 * @typedef {object} Gate
 * @property {string} id
 * @property {string} name
 * @property {number} value
 * @property {string} threshold
 * @property {boolean} pass
 * @property {number} [sigma]
 * @property {number} [resolvedBins]
 */

/**
 * @typedef {object} DriftEstimate
 * @property {number} Vx um/s, +x = image right
 * @property {number} Vy um/s, +y = image down
 * @property {number} mag
 * @property {number} sigma standard error of |V| along V
 * @property {number} angleDeg
 * @property {boolean} significant |V| > 3 sigma
 * @property {number} pairs (bin, lag) pairs used
 * @property {number} samples mode-lag samples used
 */

/**
 * @typedef {'MOTILE'|'NOT_MOTILE'|'DIRECTED_TRANSPORT'|'NO_DYNAMICS'|'UNLOCKED_EXPOSURE'} Verdict
 */

/**
 * @typedef {object} MotilityResult
 * @property {Verdict} verdict reported verdict (UNLOCKED_EXPOSURE overrides)
 * @property {Verdict} physicsVerdict verdict of gates G1-G4 alone
 * @property {Gate[]} gates
 * @property {number} TSI
 * @property {number} [alpha]
 * @property {boolean} [alphaResolved]
 * @property {number} [resolvedBins]
 * @property {number} [vbar]
 * @property {number} [sigmaV]
 * @property {number} [Z]
 * @property {number} [Ddiff]
 * @property {number} [DdiffBrownian]
 * @property {number} [dBIC]
 * @property {DriftEstimate} [drift]
 * @property {number} [mu]
 * @property {{q: number[], tauHalf: number[], mu: number}} [scaling]
 * @property {'3d'|'2d'} [geometry]
 * @property {{swimmer: ModelParams & {rss: number, amps: {A: number, B: number, W: number, P: number}[]}, brownian: ModelParams & {rss: number, amps: {A: number, B: number, W: number, P: number}[]}}} [fit]
 * @property {number[]} [q]
 * @property {number[]} [bins]
 * @property {number[]} [lags]
 * @property {number[]} [tau]
 * @property {number} [nData]
 * @property {number} [nEff]
 * @property {number} [residualRho]
 */

/**
 * Full analysis: fits, drift, gates, verdict.
 * @param {import('./ddm_ref.js').Radial} radial
 * @param {AnalysisOptions} opt
 * @returns {MotilityResult}
 */
export function analyzeMotility(radial, opt) {
  const geometry = opt.geometry ?? '3d';
  const exposureLocked = opt.exposureLocked ?? true;
  const pr = prepare(radial, opt);
  const TSI = temporalStructureIndex(radial);
  if (pr.bins.length < 4) {
    return finish({ TSI, pr, exposureLocked, insufficient: true });
  }
  const nData = pr.bins.length * pr.tau.length;

  // Pass 1: no drift compensation.
  let brown = fitBrownian(pr, 0);
  let swim = fitSwimmer(pr, geometry, 0, brown.Dd);
  let hasSwim = swim.alpha > 0.01;
  let drift = estimateDrift(radial, pr, opt, hasSwim ? swim : null);

  // Pass 2: a significant uniform drift is part of the passive null model. Its
  // speed is refined within the phase measurement's band |V| +/- (3 sigma + 25 %),
  // far too narrow to imitate swimmers, and the swimmer model reuses the same
  // |V|; both models count it as one parameter.
  let driftParams = 0;
  if (drift.significant && drift.mag > 0) {
    const half = GATES.FLOW_SIGMA * drift.sigma + 0.25 * drift.mag;
    brown = fitBrownianDrift(pr, Math.max(0, drift.mag - half), drift.mag + half);
    swim = fitSwimmer(pr, geometry, brown.Vmag, brown.Dd);
    driftParams = 1;
    hasSwim = swim.alpha > 0.01;
    const d2 = estimateDrift(radial, pr, opt, hasSwim ? swim : null);
    if (d2.samples >= 10) drift = d2;
  }

  // Effective sample size from the residual correlation of the more flexible
  // (swimmer) model, used for both models so the comparison stays fair.
  const rho = residualAutocorr(pr, swim, swimmerKernel(geometry), true);
  const nEff = effectiveN(nData, rho);
  const bicS = bic(swim.rss, nEff, swim.k + driftParams, nData);
  const bicB = bic(brown.rss, nEff, brown.k, nData);
  const dBIC = bicB - bicS;
  const share = motileShare(pr, swim);
  const best = dBIC > 0 ? swim : brown;
  const scaling = scalingExponent(radial, pr, best);
  return finish({ TSI, pr, exposureLocked, swim, brown, dBIC, drift, share, scaling, geometry, nData, nEff, rho });
}

/**
 * @param {any} s
 * @returns {MotilityResult}
 */
function finish(s) {
  const { TSI, pr, exposureLocked } = s;
  /** @type {Gate[]} */
  const gates = [];
  const g1 = TSI >= GATES.TSI_MIN;
  gates.push({ id: 'G1', name: 'Temporal structure index', value: TSI, threshold: `>= ${GATES.TSI_MIN}`, pass: g1 });
  /** @type {Verdict} */
  let verdict;
  /** @type {any} */
  let out = { TSI };
  if (s.insufficient) {
    verdict = 'NO_DYNAMICS';
    gates.push({ id: 'G0', name: 'Usable q-bins', value: pr.bins.length, threshold: '>= 4', pass: false });
  } else {
    const d = s.drift;
    const flow = d.mag > GATES.FLOW_SIGMA * d.sigma && d.mag > GATES.FLOW_MIN_UM_S;
    gates.push({
      id: 'G2',
      name: 'Uniform flow',
      value: d.mag,
      threshold: `not (|V| > ${GATES.FLOW_SIGMA} sigma and |V| > ${GATES.FLOW_MIN_UM_S} um/s)`,
      pass: !flow,
      sigma: d.sigma,
    });
    const g3 = s.dBIC > GATES.DBIC_MIN;
    gates.push({ id: 'G3', name: 'Swimmer vs Brownian Delta BIC', value: s.dBIC, threshold: `> ${GATES.DBIC_MIN}`, pass: g3 });
    const g4 = s.share.alpha > GATES.ALPHA_MIN && s.share.resolved;
    gates.push({
      id: 'G4',
      name: 'Motile contrast share alpha',
      value: s.share.alpha,
      threshold: `> ${GATES.ALPHA_MIN}, swimmer decay resolved in >= 1 bin`,
      pass: g4,
      resolvedBins: s.share.resolvedBins,
    });
    if (!g1) verdict = 'NO_DYNAMICS';
    else if (flow) verdict = 'DIRECTED_TRANSPORT';
    else if (!g3 || !g4) verdict = 'NOT_MOTILE';
    else verdict = 'MOTILE';
    out = {
      TSI,
      alpha: s.share.alpha,
      alphaResolved: s.share.resolved,
      resolvedBins: s.share.resolvedBins,
      vbar: s.swim.vbar,
      sigmaV: schulzSigma(s.swim.vbar, s.swim.Z),
      Z: s.swim.Z,
      Ddiff: s.swim.Dd,
      DdiffBrownian: s.brown.Dd,
      dBIC: s.dBIC,
      drift: d,
      mu: s.scaling.mu,
      scaling: s.scaling,
      geometry: s.geometry,
      fit: {
        swimmer: { vbar: s.swim.vbar, Z: s.swim.Z, Dd: s.swim.Dd, Vmag: s.swim.Vmag, alpha: s.swim.alpha, rss: s.swim.rss, amps: s.swim.amps },
        brownian: { vbar: 0, Z: 1, Dd: s.brown.Dd, Vmag: s.brown.Vmag, alpha: 0, rss: s.brown.rss, amps: s.brown.amps },
      },
      q: pr.bins.map((b) => b.q),
      bins: pr.bins.map((b) => b.b),
      lags: pr.lags,
      tau: pr.tau,
      nData: s.nData,
      nEff: s.nEff,
      residualRho: s.rho,
    };
  }
  gates.push({ id: 'G5', name: 'Exposure locked', value: exposureLocked ? 1 : 0, threshold: 'manual exposure readback', pass: exposureLocked });
  /** @type {Verdict} */
  const physicsVerdict = verdict;
  if (!exposureLocked) verdict = 'UNLOCKED_EXPOSURE';
  return { verdict, physicsVerdict, gates, ...out };
}

// ---------------------------------------------------------------- Web Worker entry
// src/main.ts runs analyzeMotility in a module Worker built from this file, so a
// fit (0.3-2 s on a phone) never blocks frame delivery. Inert in Node and on
// the main thread (WorkerGlobalScope only exists inside workers).
if (typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope) {
  self.onmessage = (ev) => {
    const { id, radial, opt } = ev.data;
    try {
      self.postMessage({ id, result: analyzeMotility(radial, opt) });
    } catch (e) {
      self.postMessage({ id, error: String((e && e.stack) || e) });
    }
  };
}
