// GPU vs CPU parity harness (Command 1 on a desktop GPU, Command 2 on the phone).
//
// Renders the HARNESS_SIM movie (motile_3d, 256 slots, 37 dropped -> 219 frames
// used), pushes every valid frame through the real camera path
// (VideoFrame -> importExternalTexture -> ingest -> reduce_mean -> fft_rows ->
// fft_cols), runs the chunked temporal + radial passes, and compares each stage
// against the float64 reference in src/ddm_ref.js:
//
//   spectrum.maxAbsErrOverRms  max |F_gpu - F_cpu| / rms|F_cpu| over all modes and frames
//   temporal.D                 |D_gpu - D_cpu| / (2 P_cpu(mode))   (D's natural scale)
//   temporal.C                 |C_gpu - C_cpu| / P_cpu(mode)
//   temporal.P                 |P_gpu - P_cpu| / P_cpu(mode)
//   analysis.gpu / analysis.cpu  full motility analysis on each side's radial data
//
// Prints exactly one line `RESULT {...}` to the console (and the page), also on
// failure: `RESULT {"error": "..."}`. Query parameters:
//   ?src=buffer   bypass VideoFrame/importExternalTexture (ingestLuma path)

import { DdmEngine } from '../../src/ddm-engine';
import { DDM, History, analyzeRadial, linearOrder, temporalMode } from '../../src/ddm_ref.js';
import { analyzeMotility } from '../../src/motility_fit.js';
import { simulate, lumaOf, HARNESS_SIM } from '../sim.mjs';

/** Pass thresholds (identical to test/gpu/run_gpu_tests.mjs). */
export const THRESHOLDS = {
  spectrumMaxAbsErrOverRms: 1e-4,
  temporalDp99: 1e-3,
  temporalCp99: 1e-3,
  temporalPp99: 1e-4,
  alphaAbs: 0.01,
  vbarRel: 0.01,
};

const logEl = document.getElementById('log');
function log(line: string): void {
  console.log(line);
  if (logEl) logEl.textContent += line + '\n';
}

/** p99 and max of a Float32Array of non-negative errors (sorts in place). */
function stats(err: Float32Array, n: number): { p99: number; max: number; mean: number } {
  const a = err.subarray(0, n);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += a[i];
  a.sort();
  return { p99: a[Math.min(n - 1, Math.floor(0.99 * n))], max: a[n - 1], mean: sum / n };
}

function summarize(r: ReturnType<typeof analyzeMotility>) {
  return {
    verdict: r.verdict,
    alpha: r.alpha,
    vbar_um_s: r.vbar,
    sigma_v_um_s: r.sigmaV,
    D_um2_s: r.Ddiff,
    dBIC: r.dBIC,
    TSI: r.TSI,
    mu: r.mu,
    V_um_s: r.drift ? [r.drift.Vx, r.drift.Vy] : null,
  };
}

async function run(): Promise<Record<string, unknown>> {
  const params = new URLSearchParams(location.search);
  const useBuffer = params.get('src') === 'buffer';

  log('step: simulate');
  const movie = simulate(HARNESS_SIM);
  const framesUsed = movie.valid.filter(Boolean).length;

  log('step: engine');
  const shaderUrl = new URL('../../src/ddm.wgsl', location.href);
  const shaderCode = await (await fetch(shaderUrl)).text();
  let gpuError = '';
  const engine = await DdmEngine.create({ shaderCode, onError: (m) => (gpuError ||= m), onLost: (m) => (gpuError ||= m) });
  log(`step: adapter ${JSON.stringify(engine.adapterInfo)}`);

  log(`step: ingest ${framesUsed} frames via ${useBuffer ? 'ingestLuma' : 'VideoFrame'}`);
  const tIngest0 = performance.now();
  const N = DDM.N;
  for (let t = 0; t < movie.T; t++) {
    if (!movie.valid[t]) continue;
    await engine.waitForCapacity();
    const f = movie.frames[t];
    let ok: boolean;
    if (useBuffer) {
      ok = engine.ingestLuma(Float32Array.from(f, (v) => v / 255), t);
    } else {
      const rgba = new Uint8Array(N * N * 4);
      for (let i = 0; i < N * N; i++) {
        rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = f[i];
        rgba[4 * i + 3] = 255;
      }
      const vf = new VideoFrame(rgba, { format: 'RGBA', codedWidth: N, codedHeight: N, timestamp: Math.round((t * 1e6) / movie.fps) });
      ok = engine.ingestFrame(vf, t);
    }
    if (!ok) throw new Error(`frame ${t} was throttled after waitForCapacity`);
  }
  await engine.idle();
  const msIngest = performance.now() - tIngest0;
  if (gpuError) throw new Error(`GPU error during ingest: ${gpuError}`);

  log('step: temporal');
  const { order, mask } = linearOrder(movie.valid);
  const tT0 = performance.now();
  await engine.computeTemporal(order, mask);
  const msTemporal = performance.now() - tT0;
  if (gpuError) throw new Error(`GPU error during temporal: ${gpuError}`);

  log('step: readback');
  const gHist = await engine.readHistory();
  const gFull = await engine.readTemporalFull();
  const gRadial = await engine.readRadial();

  log('step: cpu spectra (float64)');
  const hist = new History(N, DDM.T);
  let maxAbs = 0, sumSq = 0, cnt = 0;
  let maxAt = { kx: 0, ky: 0, slot: 0, absF: 0 };
  for (let t = 0; t < movie.T; t++) {
    if (!movie.valid[t]) continue;
    const spec = hist.ingest(lumaOf(movie.frames[t]), t);
    for (let m = 0; m < DDM.MODES; m++) {
      const o = (m * DDM.T + t) * 2;
      const er = gHist[o] - spec[2 * m], ei = gHist[o + 1] - spec[2 * m + 1];
      const e = Math.hypot(er, ei);
      if (e > maxAbs) {
        maxAbs = e;
        maxAt = { kx: m % DDM.NH, ky: Math.floor(m / DDM.NH), slot: t, absF: Math.hypot(spec[2 * m], spec[2 * m + 1]) };
      }
      sumSq += spec[2 * m] ** 2 + spec[2 * m + 1] ** 2;
      cnt++;
    }
  }
  const rms = Math.sqrt(sumSq / cnt);

  log('step: cpu temporal (float64, all 33024 modes)');
  const L = DDM.L;
  const eD = new Float32Array(DDM.MODES * L), eC = new Float32Array(DDM.MODES * L), eP = new Float32Array(DDM.MODES);
  const d = new Float64Array(L), c = new Float64Array(2 * L);
  let nD = 0, nP = 0;
  for (let m = 0; m < DDM.MODES; m++) {
    const P = temporalMode(hist.data, m, DDM.T, order, mask, L, d, c);
    if (!(P > 0)) continue;
    eP[nP++] = Math.abs(gFull.P[m] - P) / P;
    for (let t = 0; t < L; t++) {
      const o = m * L + t;
      eD[nD] = Math.abs(gFull.D[o] - d[t]) / (2 * P);
      eC[nD] = Math.hypot(gFull.C[2 * o] - c[2 * t], gFull.C[2 * o + 1] - c[2 * t + 1]) / P;
      nD++;
    }
  }
  const temporal = { D: stats(eD, nD), C: stats(eC, nD), P: stats(eP, nP) };

  log('step: analysis (gpu radial vs cpu radial)');
  const cRadial = analyzeRadial(hist, order, mask, engine.bins);
  const opt = { dxUm: movie.dxUm, fps: movie.fps, geometry: '3d' as const, exposureLocked: true };
  const aG = analyzeMotility(gRadial, opt);
  const aC = analyzeMotility(cRadial, opt);
  // radial D error relative to each bin's largest value
  let radialMaxRel = 0;
  for (let b = 0; b < cRadial.NB; b++) {
    let scale = 0;
    for (let t = 0; t < L; t++) scale = Math.max(scale, cRadial.D[b * L + t]);
    if (!(scale > 0)) continue;
    for (let t = 0; t < L; t++) radialMaxRel = Math.max(radialMaxRel, Math.abs(gRadial.D[b * L + t] - cRadial.D[b * L + t]) / scale);
  }

  const spectrum = { maxAbsErrOverRms: maxAbs / rms, rms, maxAt };
  const gpu = summarize(aG), cpu = summarize(aC);
  const checks = {
    spectrum: spectrum.maxAbsErrOverRms < THRESHOLDS.spectrumMaxAbsErrOverRms,
    D: temporal.D.p99 < THRESHOLDS.temporalDp99,
    C: temporal.C.p99 < THRESHOLDS.temporalCp99,
    P: temporal.P.p99 < THRESHOLDS.temporalPp99,
    verdict: gpu.verdict === cpu.verdict,
    alpha: Math.abs((gpu.alpha ?? NaN) - (cpu.alpha ?? NaN)) < THRESHOLDS.alphaAbs,
    vbar: Math.abs((gpu.vbar_um_s ?? NaN) / (cpu.vbar_um_s ?? NaN) - 1) < THRESHOLDS.vbarRel,
  };
  engine.destroy();
  return {
    framesUsed,
    source: useBuffer ? 'ingestLuma' : 'VideoFrame',
    adapter: engine.adapterInfo,
    msIngest: Math.round(msIngest),
    msPerFrame: +(msIngest / framesUsed).toFixed(2),
    msTemporal: Math.round(msTemporal),
    throttled: engine.throttled,
    spectrum,
    temporal,
    radial: { DmaxRel: radialMaxRel },
    analysis: { gpu, cpu },
    checks,
    pass: Object.values(checks).every(Boolean),
  };
}

run()
  .then((r) => {
    log('RESULT ' + JSON.stringify(r));
    (window as unknown as { __RESULT: unknown }).__RESULT = r;
  })
  .catch((e: unknown) => {
    const err = e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e);
    const r = { error: err };
    log('RESULT ' + JSON.stringify(r));
    (window as unknown as { __RESULT: unknown }).__RESULT = r;
  });
