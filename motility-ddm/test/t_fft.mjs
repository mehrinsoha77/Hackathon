// FFT precision and Wiener-Khinchin validation.
//   - Stockham FFT vs direct DFT (forward and inverse), n = 2..512
//   - half-plane windowed 2D spectrum vs a direct 2D DFT
//   - masked, reordered Wiener-Khinchin estimators (FFT path) vs direct
//     O(T*L) time-domain sums, including dropped frames and shuffled order

import { pathToFileURL } from 'node:url';
import { fft, dft, blackmanHarris, spectrum2dHalf } from '../src/fft.js';
import { temporalMode, temporalModeDirect } from '../src/ddm_ref.js';
import { mulberry32 } from './sim.mjs';

export function run() {
  let passed = 0, failed = 0;
  const check = (name, ok, detail = '') => {
    if (ok) passed++;
    else {
      failed++;
      console.log(`  FAIL ${name} ${detail}`);
    }
  };
  const rand = mulberry32(1234);

  // 1. FFT vs DFT
  for (let n = 2; n <= 512; n *= 2) {
    const x = new Float64Array(2 * n).map(() => rand() * 2 - 1);
    for (const inv of [false, true]) {
      const ref = dft(x, inv);
      const got = fft(Float64Array.from(x), inv);
      let err = 0, mag = 0;
      for (let i = 0; i < 2 * n; i++) {
        err = Math.max(err, Math.abs(got[i] - ref[i]));
        mag = Math.max(mag, Math.abs(ref[i]));
      }
      check(`fft n=${n} inverse=${inv}`, err / mag < 1e-12, `rel err ${err / mag}`);
    }
    const y = fft(fft(Float64Array.from(x)), true);
    let rt = 0;
    for (let i = 0; i < 2 * n; i++) rt = Math.max(rt, Math.abs(y[i] / n - x[i]));
    check(`fft round trip n=${n}`, rt < 1e-13, `err ${rt}`);
  }

  // 2. Windowed half-plane 2D spectrum vs direct 2D DFT (N = 16)
  {
    const N = 16, NH = N / 2 + 1;
    const img = new Float64Array(N * N).map(() => rand());
    const win = blackmanHarris(N);
    const got = spectrum2dHalf(img, N, win);
    let mean = 0;
    for (const v of img) mean += v;
    mean /= N * N;
    let err = 0, mag = 0;
    for (let ky = 0; ky < N; ky++) {
      for (let kx = 0; kx < NH; kx++) {
        let sr = 0, si = 0;
        for (let y = 0; y < N; y++) {
          for (let x = 0; x < N; x++) {
            const v = (img[y * N + x] - mean) * win[x] * win[y];
            const a = (-2 * Math.PI * (kx * x + ky * y)) / N;
            sr += v * Math.cos(a);
            si += v * Math.sin(a);
          }
        }
        const m = ky * NH + kx;
        err = Math.max(err, Math.hypot(got[2 * m] - sr / N, got[2 * m + 1] - si / N));
        mag = Math.max(mag, Math.hypot(sr, si) / N);
      }
    }
    check('2D half-plane spectrum vs direct DFT', err / mag < 1e-12, `rel err ${err / mag}`);
  }

  // 3. Blackman-Harris window properties
  {
    const w = blackmanHarris(256);
    check('window w(0) = 6e-5', Math.abs(w[0] - 6e-5) < 1e-12, `${w[0]}`);
    check('window w(N/2) = 1', Math.abs(w[128] - 1) < 1e-12, `${w[128]}`);
    check('window symmetric', Math.abs(w[10] - w[246]) < 1e-14);
  }

  // 4. Wiener-Khinchin (FFT) vs direct sums with masks and permuted order
  for (const [label, dropP, shuffle] of [
    ['full', 0, false],
    ['20% dropped', 0.2, false],
    ['shuffled + 15% dropped', 0.15, true],
  ]) {
    const T = 256, L = 128;
    const hist = new Float64Array(T * 2 * 3); // 3 modes
    for (let i = 0; i < hist.length; i++) hist[i] = rand() * 2 - 1;
    // add a correlated drifting component to mode 1 so C has structure
    for (let t = 0; t < T; t++) {
      hist[(T + t) * 2] += 3 * Math.cos(0.07 * t);
      hist[(T + t) * 2 + 1] += 3 * Math.sin(0.07 * t);
    }
    const order = Uint32Array.from({ length: T }, (_, i) => i);
    if (shuffle) {
      for (let i = T - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    }
    const mask = new Float32Array(T).map(() => (rand() < dropP ? 0 : 1));
    mask[0] = 1;
    let errD = 0, errC = 0, errP = 0, d0 = 0;
    for (let m = 0; m < 3; m++) {
      const D1 = new Float64Array(L), C1 = new Float64Array(2 * L);
      const D2 = new Float64Array(L), C2 = new Float64Array(2 * L);
      const P1 = temporalMode(hist, m, T, order, mask, L, D1, C1);
      const P2 = temporalModeDirect(hist, m, T, order, mask, L, D2, C2);
      errP = Math.max(errP, Math.abs(P1 - P2) / P2);
      d0 = Math.max(d0, Math.abs(D1[0]) / P2);
      for (let t = 0; t < L; t++) errD = Math.max(errD, Math.abs(D1[t] - D2[t]) / P2);
      for (let t = 0; t < 2 * L; t++) errC = Math.max(errC, Math.abs(C1[t] - C2[t]) / P2);
    }
    check(`Wiener-Khinchin D (${label})`, errD < 1e-11, `err ${errD}`);
    check(`Wiener-Khinchin C (${label})`, errC < 1e-11, `err ${errC}`);
    check(`static power P (${label})`, errP < 1e-12, `err ${errP}`);
    check(`D(0) = 0 (${label})`, d0 < 1e-12, `D(0)/P ${d0}`);
  }

  // 5. A pure phase rotation x(t) = exp(-i w t) gives C(tau) = exp(-i w tau), D = 2 - 2 cos(w tau)
  {
    const T = 256, L = 128, w = 0.05;
    const hist = new Float64Array(2 * T);
    for (let t = 0; t < T; t++) {
      hist[2 * t] = Math.cos(-w * t);
      hist[2 * t + 1] = Math.sin(-w * t);
    }
    const order = Uint32Array.from({ length: T }, (_, i) => i);
    const mask = new Float32Array(T).fill(1);
    const D = new Float64Array(L), C = new Float64Array(2 * L);
    temporalMode(hist, 0, T, order, mask, L, D, C);
    let e = 0;
    for (let t = 0; t < L; t++) {
      e = Math.max(e, Math.abs(D[t] - (2 - 2 * Math.cos(w * t))));
      e = Math.max(e, Math.abs(C[2 * t] - Math.cos(w * t)), Math.abs(C[2 * t + 1] + Math.sin(w * t)));
    }
    check('phase rotation: C(tau) = exp(-i w tau), D = 2 - 2cos', e < 1e-12, `err ${e}`);
  }
  return { passed, failed };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const r = run();
  console.log(`t_fft: ${r.passed} passed, ${r.failed} failed`);
  process.exit(r.failed ? 1 : 0);
}
