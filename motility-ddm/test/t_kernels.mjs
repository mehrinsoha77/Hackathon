// Wilson ISF closed-form kernel validation.
//   - Bessel J0, ln Gamma, Legendre P_nu against known values
//   - 3D <sin(qv tau)/(qv tau)> and 2D <J0(qv tau)> closed forms vs direct
//     quadrature over the Schulz distribution
//   - exact 3-column NNLS satisfies the KKT conditions
//   - VarPro fit recovers (vbar, Z, D, alpha) from noise-free model data

import { pathToFileURL } from 'node:url';
import {
  besselJ0, lnGamma, legendreP, schulzPdf, isf3d, isf2d, isf2dFast, nnls3,
  fitSwimmer, fitBrownian, prepare, swimmerKernel, besselJ0 as J0,
} from '../src/motility_fit.js';
import { mulberry32 } from './sim.mjs';

/** Simpson quadrature of the Schulz average of f(v). */
function schulzAverage(f, vbar, Z) {
  const vmax = vbar * 30;
  const n = 60000;
  const h = vmax / n;
  let s = 0;
  for (let i = 0; i <= n; i++) {
    const v = i * h;
    const w = i === 0 || i === n ? 1 : i % 2 ? 4 : 2;
    s += w * schulzPdf(v, vbar, Z) * f(v);
  }
  return (s * h) / 3;
}

export function run() {
  let passed = 0, failed = 0;
  const check = (name, ok, detail = '') => {
    if (ok) passed++;
    else {
      failed++;
      console.log(`  FAIL ${name} ${detail}`);
    }
  };

  // Special functions
  for (const [x, ref] of [[0, 1], [1, 0.7651976865579666], [2.404825557695773, 0], [5, -0.1775967713143383], [10, -0.2459357644513483], [30, -0.0863679835810403]]) {
    check(`J0(${x})`, Math.abs(besselJ0(x) - ref) < 2e-8, `${besselJ0(x)} vs ${ref}`);
  }
  for (const [x, ref] of [[1, 0], [5, Math.log(24)], [0.5, Math.log(Math.sqrt(Math.PI))], [10.5, 13.940625219403763]]) {
    check(`lnGamma(${x})`, Math.abs(lnGamma(x) - ref) < 1e-12, `${lnGamma(x)} vs ${ref}`);
  }
  for (const c of [1, 0.9, 0.5, 0.1]) {
    const p2 = (3 * c * c - 1) / 2;
    const p5 = (63 * c ** 5 - 70 * c ** 3 + 15 * c) / 8;
    check(`P_2(${c})`, Math.abs(legendreP(2, c) - p2) < 1e-12, `${legendreP(2, c)} vs ${p2}`);
    check(`P_5(${c})`, Math.abs(legendreP(5, c) - p5) < 1e-12, `${legendreP(5, c)} vs ${p5}`);
  }
  // Schulz normalization and mean
  for (const Z of [0.8, 3, 40]) {
    const norm = schulzAverage(() => 1, 10, Z);
    const mean = schulzAverage((v) => v, 10, Z);
    check(`Schulz norm Z=${Z}`, Math.abs(norm - 1) < 1e-4, `${norm}`);
    check(`Schulz mean Z=${Z}`, Math.abs(mean - 10) < 1e-3, `${mean}`);
  }

  // Closed forms vs quadrature, x = q vbar tau
  const vbar = 1;
  for (const Z of [0.8, 2, 10, 60]) {
    let e3 = 0, e2 = 0;
    for (const x of [0.01, 0.3, 1, 2.5, 5, 12, 30]) {
      const q3 = schulzAverage((v) => (v > 0 ? Math.sin(x * v) / (x * v) : 1), vbar, Z);
      const q2 = schulzAverage((v) => J0(x * v), vbar, Z);
      e3 = Math.max(e3, Math.abs(isf3d(x, Z) - q3));
      e2 = Math.max(e2, Math.abs(isf2d(x, Z) - q2));
    }
    check(`3D ISF closed form vs quadrature Z=${Z}`, e3 < 2e-5, `max err ${e3}`);
    check(`2D ISF closed form vs quadrature Z=${Z}`, e2 < 2e-5, `max err ${e2}`);
  }
  // Small-argument series branch agrees with the closed forms at the switch point
  for (const Z of [1, 20]) {
    const L = 1.0001e-4, x = L * (Z + 1);
    const exact3 = Math.sin(Z * Math.atan(L)) / (Z * L * Math.pow(1 + L * L, Z / 2));
    const exact2 = legendreP(Z, 1 / Math.sqrt(1 + L * L)) * Math.pow(1 + L * L, -(Z + 1) / 2);
    const s3 = 1 - (x * x * (Z + 2)) / (6 * (Z + 1));
    const s2 = 1 - (x * x * (Z + 2)) / (4 * (Z + 1));
    check(`3D series vs closed form Z=${Z}`, Math.abs(s3 - exact3) < 1e-10, `${s3} ${exact3}`);
    check(`2D series vs closed form Z=${Z}`, Math.abs(s2 - exact2) < 1e-10, `${s2} ${exact2}`);
    check(`3D ISF continuous at series switch Z=${Z}`, Math.abs(isf3d(x * 0.9999, Z) - isf3d(x, Z)) < 1e-9);
    check(`2D ISF continuous at series switch Z=${Z}`, Math.abs(isf2d(x * 0.9999, Z) - isf2d(x, Z)) < 1e-9);
  }
  // Tabulated 2D kernel used by the fitter vs the exact closed form
  {
    let worst = 0, at = '';
    const t0 = performance.now();
    isf2dFast(1, 1);
    const tBuild = performance.now() - t0;
    for (let lz = Math.log(0.5); lz <= Math.log(100) + 1e-9; lz += 0.0371) {
      const Z = Math.exp(lz);
      for (let lx = -3; lx <= 3.5; lx += 0.0137) {
        const x = 10 ** lx;
        const e = Math.abs(isf2dFast(x, Z) - isf2d(x, Z));
        if (e > worst) [worst, at] = [e, `Z=${Z.toFixed(3)} x=${x.toFixed(4)}`];
      }
    }
    check('tabulated 2D ISF vs closed form', worst < 2e-5, `max err ${worst} at ${at}`);
    if (process.env.VERBOSE) console.log(`  2D table: max err ${worst.toExponential(2)} at ${at}, build ${tBuild.toFixed(0)} ms`);
    check('tabulated 2D ISF build time < 2 s', tBuild < 2000, `${tBuild.toFixed(0)} ms`);
  }
  // Narrow-distribution limit (sigma/vbar = 0.1): sin(x)/x and J0(x) up to the O(sigma^2) spread term
  check('3D ISF narrow limit', Math.abs(isf3d(1, 100) - Math.sin(1)) < 0.005, `${isf3d(1, 100)}`);
  check('2D ISF narrow limit', Math.abs(isf2d(1, 100) - J0(1)) < 0.005, `${isf2d(1, 100)}`);

  // NNLS KKT: a >= 0, grad_i = (G a - h)_i = 0 where a_i > 0, >= 0 where a_i = 0
  const rand = mulberry32(99);
  let kktOk = true;
  for (let trial = 0; trial < 300; trial++) {
    const n = 12;
    const C = Array.from({ length: 3 }, () => Array.from({ length: n }, () => rand() * 2 - 1));
    const d = Array.from({ length: n }, () => rand() * 2 - 1);
    const G = [], h = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) G.push(C[i].reduce((s, v, k) => s + v * C[j][k], 0));
      h.push(C[i].reduce((s, v, k) => s + v * d[k], 0));
    }
    const dd = d.reduce((s, v) => s + v * v, 0);
    const { a, rss } = nnls3(G, h, dd);
    let r2 = 0;
    for (let k = 0; k < n; k++) {
      const r = d[k] - a[0] * C[0][k] - a[1] * C[1][k] - a[2] * C[2][k];
      r2 += r * r;
    }
    for (let i = 0; i < 3; i++) {
      const g = G[3 * i] * a[0] + G[3 * i + 1] * a[1] + G[3 * i + 2] * a[2] - h[i];
      if (a[i] < 0) kktOk = false;
      if (a[i] > 1e-12 && Math.abs(g) > 1e-9) kktOk = false;
      if (a[i] <= 1e-12 && g < -1e-9) kktOk = false;
    }
    if (Math.abs(r2 - rss) > 1e-9) kktOk = false;
  }
  check('NNLS KKT conditions (300 random problems)', kktOk);

  // VarPro recovery on noise-free model data
  for (const geometry of ['3d', '2d']) {
    const truth = { vbar: 30, Z: 4, Dd: 0.3, alpha: 0.6 };
    const NB = 32, L = 128, N = 256, dxUm = 1, fps = 30;
    const kernel = swimmerKernel(geometry);
    const kCenter = Float64Array.from({ length: NB }, (_, b) => 4 + 3 * (b + 0.5));
    const D = new Float64Array(NB * L);
    for (let b = 0; b < NB; b++) {
      const q = (2 * Math.PI * kCenter[b]) / (N * dxUm);
      const A = 1000 * Math.exp(-q * q), B = 2;
      for (let t = 0; t < L; t++) {
        const tau = t / fps;
        const E = Math.exp(-truth.Dd * q * q * tau);
        const f = E * (1 - truth.alpha + truth.alpha * kernel(q * truth.vbar * tau, truth.Z));
        D[b * L + t] = A * (1 - f) + (t ? B : 0);
      }
    }
    const radial = { N, NB, L, kCenter, nModes: new Uint32Array(NB).fill(50), D, P: new Float64Array(NB), drift: new Float64Array(NB * L * 8) };
    const pr = prepare(radial, { dxUm, fps });
    const brown = fitBrownian(pr, 0);
    const fit = fitSwimmer(pr, geometry, 0, brown.Dd);

    check(`VarPro ${geometry} recovers vbar`, Math.abs(fit.vbar / truth.vbar - 1) < 1e-3, `${fit.vbar}`);
    check(`VarPro ${geometry} recovers Z`, Math.abs(fit.Z / truth.Z - 1) < 1e-2, `${fit.Z}`);
    check(`VarPro ${geometry} recovers D`, Math.abs(fit.Dd / truth.Dd - 1) < 1e-2, `${fit.Dd}`);
    check(`VarPro ${geometry} recovers alpha`, Math.abs(fit.alpha - truth.alpha) < 1e-3, `${fit.alpha}`);
    check(`VarPro ${geometry} swimmer beats Brownian`, fit.rss < 1e-3 * brown.rss, `${fit.rss} vs ${brown.rss}`);
  }
  return { passed, failed };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const r = run();
  console.log(`t_kernels: ${r.passed} passed, ${r.failed} failed`);
  process.exit(r.failed ? 1 : 0);
}
