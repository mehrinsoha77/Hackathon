// Phase-gradient drift velocity estimation tests.
//   1. recovery of V from per-mode phase correlations C(q, tau) with noise
//   2. no drift -> not significant
//   3. swimmer-ISF lag gating: exactly the lag pairs with g(q vbar tau) < 0.2 are
//      used, and gating suppresses bias from coherent early-lag swimmer phases
//   4. end-to-end sign convention: a translating image through the full
//      frame -> spectrum -> temporal -> radial -> drift path returns +V

import { pathToFileURL } from 'node:url';
import { makeBins, radialFromFull, History, analyzeRadial, linearOrder } from '../src/ddm_ref.js';
import { prepare, estimateDrift, isf3d, GATES } from '../src/motility_fit.js';
import { makeRng } from './sim.mjs';

const N = 64, NB = 16, L = 64, dxUm = 1, fps = 30, dt = 1 / fps;
const bins = makeBins(N, NB, 2, 30);

/**
 * Per-mode arrays from a generator c(kx, ky, tau) -> [re, im].
 * D is set to 2 (P - Re C), the structure function implied by C for a
 * stationary signal, so that prepare() sees realistic bins.
 */
function synthFull(gen) {
  const M = N * (N / 2 + 1);
  const D = new Float64Array(M * L), C = new Float64Array(M * L * 2), P = new Float64Array(M).fill(1);
  for (let b = 0; b < NB; b++) {
    for (let i = bins.binOffsets[b]; i < bins.binOffsets[b + 1]; i++) {
      const m = bins.binModes[4 * i], kx = bins.binModes[4 * i + 1], ky = bins.binModes[4 * i + 2];
      for (let t = 0; t < L; t++) {
        const [re, im] = gen(kx, ky, t);
        C[(m * L + t) * 2] = re;
        C[(m * L + t) * 2 + 1] = im;
        D[m * L + t] = 2 * (1 - re);
      }
    }
  }
  return radialFromFull({ D, C, P }, bins, L);
}

const qOf = (k) => (2 * Math.PI * k) / (N * dxUm);

export function run() {
  let passed = 0, failed = 0;
  const check = (name, ok, detail = '') => {
    if (ok) passed++;
    else {
      failed++;
      console.log(`  FAIL ${name} ${detail}`);
    }
  };
  const opt = { dxUm, fps, geometry: '3d' };

  // 1. recovery at several angles and speeds
  for (const [speed, deg] of [[6, -34], [2.5, 90], [12, 180], [4, 225]]) {
    const rng = makeRng(speed * 100 + deg);
    const Vx = speed * Math.cos((deg * Math.PI) / 180), Vy = speed * Math.sin((deg * Math.PI) / 180);
    const radial = synthFull((kx, ky, t) => {
      const ph = -(qOf(kx) * Vx + qOf(ky) * Vy) * t * dt;
      const amp = Math.exp(-0.2 * (qOf(kx) ** 2 + qOf(ky) ** 2) * t * dt);
      return [amp * Math.cos(ph) + 0.02 * rng.normal(), amp * Math.sin(ph) + 0.02 * rng.normal()];
    });
    const pr = prepare(radial, opt);
    const d = estimateDrift(radial, pr, opt, null);
    let dAng = Math.abs(d.angleDeg - deg) % 360;
    if (dAng > 180) dAng = 360 - dAng;
    check(`drift |V| ${speed} um/s @ ${deg} deg`, Math.abs(d.mag - speed) / speed < 0.01, `got ${d.mag}`);
    check(`drift angle ${deg} deg`, dAng < 0.5, `got ${d.angleDeg}`);
    check(`drift significant ${speed} um/s`, d.significant && d.sigma < 0.1 * speed, `sigma ${d.sigma}`);
  }

  // 2. no drift: incoherent phases
  {
    const rng = makeRng(77);
    const radial = synthFull(() => [0.3 * rng.normal(), 0.3 * rng.normal()]);
    const pr = prepare(radial, opt);
    const d = estimateDrift(radial, pr, opt, null);
    check('no drift: |V| below flow threshold', d.mag < GATES.FLOW_MIN_UM_S, `|V| ${d.mag}`);
  }

  // 3. swimmer-ISF lag gating
  {
    const swimmer = { vbar: 40, Z: 4 };
    const V = [5, 0], U = [-25, 0]; // passive drift V, coherent swimmer contaminant U
    const radial = synthFull((kx, ky, t) => {
      const qx = qOf(kx), qy = qOf(ky);
      const q = Math.hypot(qx, qy);
      const g = isf3d(q * swimmer.vbar * t * dt, swimmer.Z);
      const pv = -(qx * V[0] + qy * V[1]) * t * dt;
      const pu = -(qx * U[0] + qy * U[1]) * t * dt;
      return [Math.cos(pv) + 2 * g * Math.cos(pu), Math.sin(pv) + 2 * g * Math.sin(pu)];
    });
    const pr = prepare(radial, opt);
    let expectPairs = 0;
    for (const bin of pr.bins) {
      for (let t = 1; t < L - 1; t++) if (Math.abs(isf3d(bin.q * swimmer.vbar * t * dt, swimmer.Z)) < GATES.DRIFT_ISF_MAX) expectPairs++;
    }
    const gated = estimateDrift(radial, pr, opt, swimmer);
    const open = estimateDrift(radial, pr, opt, null);
    check('gating uses exactly the lag pairs with g < 0.2', gated.pairs === expectPairs, `${gated.pairs} vs ${expectPairs}`);
    const eG = Math.hypot(gated.Vx - V[0], gated.Vy - V[1]);
    const eO = Math.hypot(open.Vx - V[0], open.Vy - V[1]);
    check('gating suppresses early-lag swimmer phase bias', eG < 0.15 * Math.hypot(...V) && eG < 0.5 * eO, `gated err ${eG.toFixed(3)}, ungated err ${eO.toFixed(3)}`);
  }

  // 4. end-to-end sign convention through the full CPU pipeline
  for (const [sx, sy] of [[1, 0], [0, 1], [-1, 1]]) {
    const T = 64;
    const rng = makeRng(5);
    const base = new Float64Array(N * N).map(() => rng.uniform());
    // low-pass the random field so most power sits inside the bins
    const sm = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let s = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) s += base[((y + j + N) % N) * N + ((x + i + N) % N)];
      sm[y * N + x] = s / 9;
    }
    const hist = new History(N, T);
    const img = new Float64Array(N * N);
    for (let t = 0; t < T; t++) {
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const xs = (((x - sx * t) % N) + N) % N, ys = (((y - sy * t) % N) + N) % N;
        img[y * N + x] = sm[ys * N + xs];
      }
      hist.ingest(img, t);
    }
    const { order, mask } = linearOrder(new Array(T).fill(true));
    const radial = analyzeRadial(hist, order, mask, bins, 32);
    const pr = prepare(radial, opt);
    const d = estimateDrift(radial, pr, opt, null);
    const Vx = sx * dxUm * fps, Vy = sy * dxUm * fps;
    const err = Math.hypot(d.Vx - Vx, d.Vy - Vy) / Math.hypot(Vx, Vy);
    check(`translation (${sx},${sy}) px/frame -> V = (${Vx},${Vy}) um/s`, err < 0.1, `got (${d.Vx.toFixed(2)}, ${d.Vy.toFixed(2)})`);
  }
  return { passed, failed };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const r = run();
  console.log(`t_drift_phase: ${r.passed} passed, ${r.failed} failed`);
  process.exit(r.failed ? 1 : 0);
}
