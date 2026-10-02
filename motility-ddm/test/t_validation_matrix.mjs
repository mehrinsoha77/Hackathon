// `npm test`: the complete CPU test suite.
//   1. unit tests: FFT / Wiener-Khinchin, ISF kernels, phase-drift estimator
//   2. the 11-scenario synthetic validation matrix, end to end through the
//      float64 reference pipeline (frame -> spectrum -> temporal -> radial -> fit)
// Exit code 0 only if every check passes.

import { pathToFileURL } from 'node:url';
import { DDM, History, makeBins, analyzeRadial, linearOrder, shuffleOrder } from '../src/ddm_ref.js';
import { analyzeMotility } from '../src/motility_fit.js';
import { simulate, lumaOf, mulberry32, SCENARIOS } from './sim.mjs';
import { run as runFft } from './t_fft.mjs';
import { run as runKernels } from './t_kernels.mjs';
import { run as runDrift } from './t_drift_phase.mjs';

/**
 * Run one scenario through the CPU reference pipeline.
 * @param {(typeof SCENARIOS)[number]} sc
 * @param {{exposureLocked?: boolean}} [o]
 */
export function runScenario(sc, o = {}) {
  const t0 = performance.now();
  const movie = simulate(sc.sim);
  const hist = new History(movie.N, DDM.T);
  movie.frames.forEach((f, t) => {
    if (movie.valid[t]) hist.ingest(lumaOf(f), t);
  });
  let ord = linearOrder(movie.valid);
  if (sc.shuffle) ord = shuffleOrder(ord, mulberry32(4242));
  const bins = makeBins(movie.N);
  const radial = analyzeRadial(hist, ord.order, ord.mask, bins);
  const t1 = performance.now();
  const res = analyzeMotility(radial, {
    dxUm: movie.dxUm,
    fps: movie.fps,
    geometry: sc.geometry,
    exposureLocked: o.exposureLocked ?? true,
  });
  const t2 = performance.now();
  return { res, radial, movie, msPipeline: t1 - t0, msFit: t2 - t1 };
}

/**
 * @param {(typeof SCENARIOS)[number]} sc
 * @param {ReturnType<typeof analyzeMotility>} res
 * @returns {string[]} failures
 */
export function checkScenario(sc, res) {
  const fails = [];
  if (res.verdict !== sc.expect) fails.push(`verdict ${res.verdict}, expected ${sc.expect}`);
  const c = sc.checks || {};
  if (c.vbar) {
    const err = Math.abs(res.vbar - c.vbar.value) / c.vbar.value;
    if (!(err <= c.vbar.relTol)) fails.push(`vbar ${res.vbar?.toFixed(2)} um/s, expected ${c.vbar.value} +/- ${c.vbar.relTol * 100}%`);
  }
  if (c.alpha) {
    if (!(Math.abs(res.alpha - c.alpha.value) <= c.alpha.absTol)) fails.push(`alpha ${res.alpha?.toFixed(3)}, expected ${c.alpha.value.toFixed(3)} +/- ${c.alpha.absTol}`);
  }
  if (c.drift) {
    const d = res.drift;
    const errMag = Math.abs(d.mag - c.drift.speed) / c.drift.speed;
    let dAng = Math.abs(d.angleDeg - c.drift.angleDeg) % 360;
    if (dAng > 180) dAng = 360 - dAng;
    if (!(errMag <= c.drift.relTol)) fails.push(`|V| ${d.mag.toFixed(2)} um/s, expected ${c.drift.speed} +/- ${c.drift.relTol * 100}%`);
    if (!(dAng <= c.drift.degTol)) fails.push(`V angle ${d.angleDeg.toFixed(1)} deg, expected ${c.drift.angleDeg} +/- ${c.drift.degTol}`);
  }
  return fails;
}

const fmt = (x, d = 3) => (x === undefined || x === null || Number.isNaN(x) ? '-' : Number(x).toFixed(d));

function main() {
  let failures = 0;
  console.log('== Unit tests ==');
  for (const [name, fn] of [['t_fft', runFft], ['t_kernels', runKernels], ['t_drift_phase', runDrift]]) {
    const r = fn();
    console.log(`${r.failed ? 'FAIL' : 'PASS'}  ${name}: ${r.passed} passed, ${r.failed} failed`);
    failures += r.failed;
  }

  console.log('\n== Validation matrix (11 scenarios, float64 reference pipeline) ==');
  const rows = [];
  for (const sc of SCENARIOS) {
    const { res, msPipeline, msFit } = runScenario(sc);
    const fails = checkScenario(sc, res);
    failures += fails.length;
    rows.push({ sc, res, fails });
    console.log(
      `${fails.length ? 'FAIL' : 'PASS'}  ${sc.name.padEnd(22)} ${res.verdict.padEnd(19)}` +
        ` TSI=${fmt(res.TSI, 2)} dBIC=${fmt(res.dBIC, 0)} nEff=${fmt(res.nEff, 0)} alpha=${fmt(res.alpha, 2)}` +
        ` vbar=${fmt(res.vbar, 1)} sigma_v=${fmt(res.sigmaV, 1)} D=${fmt(res.Ddiff, 2)} mu=${fmt(res.mu, 2)}` +
        ` |V|=${fmt(res.drift?.mag, 2)}+/-${fmt(res.drift?.sigma, 2)} @${fmt(res.drift?.angleDeg, 0)}deg` +
        ` [${(msPipeline / 1000).toFixed(1)}s + fit ${(msFit / 1000).toFixed(2)}s]`,
    );
    for (const f of fails) console.log(`      - ${f}`);
  }

  // Gate 5: the same motile movie with exposure unlocked must never read MOTILE.
  const unlocked = runScenario(SCENARIOS[0], { exposureLocked: false }).res;
  const g5ok = unlocked.verdict === 'UNLOCKED_EXPOSURE' && unlocked.physicsVerdict === 'MOTILE';
  console.log(`${g5ok ? 'PASS' : 'FAIL'}  exposure gate (G5)        ${unlocked.verdict} (physics: ${unlocked.physicsVerdict})`);
  if (!g5ok) failures++;

  const passed = rows.filter((r) => !r.fails.length).length;
  console.log(`\nMatrix: ${passed}/${rows.length} scenarios correct. Total failures: ${failures}`);
  process.exit(failures ? 1 : 0);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) main();
