// Synthetic DDM movie simulator and the 11-scenario validation matrix.
//
// Particles are rendered as Gaussian blobs (bright-field/dark-field agnostic,
// intensity only matters through its fluctuations) on a periodic N x N field,
// with additive Gaussian camera noise and 8-bit quantization, exactly what the
// camera path delivers. Three populations:
//   swimmers  straight runs at Schulz-distributed speeds, 3D-isotropic
//             directions seen in projection or 2D in-plane, plus diffusion
//   passive   Brownian particles (dead cells, tracers)
//   fixed     particles stuck to the glass (static contrast only)
// A uniform drift V advects swimmers and passive particles (not fixed ones).

/** Seeded uniform RNG (mulberry32). @param {number} seed */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** @param {number} seed */
export function makeRng(seed) {
  const u = mulberry32(seed);
  let spare = null;
  const normal = () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let x, y, r;
    do {
      x = 2 * u() - 1;
      y = 2 * u() - 1;
      r = x * x + y * y;
    } while (r >= 1 || r === 0);
    const f = Math.sqrt((-2 * Math.log(r)) / r);
    spare = y * f;
    return x * f;
  };
  /** Gamma(k, theta), Marsaglia-Tsang. */
  const gamma = (k, theta) => {
    if (k < 1) return gamma(k + 1, theta) * Math.pow(u(), 1 / k);
    const d = k - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do {
        x = normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const uu = u();
      if (uu < 1 - 0.0331 * x ** 4) return d * v * theta;
      if (Math.log(uu) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * theta;
    }
  };
  return { uniform: u, normal, gamma };
}

/**
 * @typedef {object} SimParams
 * @property {number} [N] ROI side (px)
 * @property {number} [T] frames
 * @property {number} [fps]
 * @property {number} [dxUm] um per pixel
 * @property {number} [seed]
 * @property {{n: number, vbar: number, sigma: number, geometry: '3d'|'2d'}} [swimmers]
 * @property {{n: number}} [passive]
 * @property {number} [Dum2s] diffusion coefficient of swimmers and passive particles (um^2/s)
 * @property {{n: number}} [fixed]
 * @property {{speed: number, angleDeg: number}} [drift] um/s, angle from +x towards +y (image down)
 * @property {number} [dropFrac] fraction of frames dropped at random
 * @property {number} [dropExact] exact number of frames dropped (overrides dropFrac)
 * @property {number} [blobSigmaPx]
 * @property {number} [amp] blob peak intensity (0..1 full scale)
 * @property {number} [background]
 * @property {number} [noise] camera noise sigma (0..1 full scale)
 */

/**
 * Render a movie.
 * @param {SimParams} p
 * @returns {{frames: Uint8Array[], valid: boolean[], N: number, T: number, fps: number, dxUm: number, truth: object}}
 */
export function simulate(p) {
  const N = p.N ?? 256;
  const T = p.T ?? 256;
  const fps = p.fps ?? 30;
  const dxUm = p.dxUm ?? 1;
  const dt = 1 / fps;
  const rng = makeRng(p.seed ?? 1);
  const sig = p.blobSigmaPx ?? 1.5;
  const amp = p.amp ?? 0.15;
  const bg = p.background ?? 0.25;
  const noise = p.noise ?? 0.01;
  const D = p.Dum2s ?? 0;
  const L = N * dxUm; // field size in um
  const Vx = p.drift ? p.drift.speed * Math.cos((p.drift.angleDeg * Math.PI) / 180) : 0;
  const Vy = p.drift ? p.drift.speed * Math.sin((p.drift.angleDeg * Math.PI) / 180) : 0;

  /** particles: x, y (um), vx, vy (um/s), mobile */
  const parts = [];
  const sw = p.swimmers;
  const speeds = [];
  if (sw && sw.n > 0) {
    const Z = (sw.vbar / sw.sigma) ** 2 - 1;
    for (let i = 0; i < sw.n; i++) {
      const v = rng.gamma(Z + 1, sw.vbar / (Z + 1));
      speeds.push(v);
      const phi = 2 * Math.PI * rng.uniform();
      let s = 1;
      if (sw.geometry !== '2d') {
        const cz = 2 * rng.uniform() - 1;
        s = Math.sqrt(1 - cz * cz);
      }
      parts.push({ x: rng.uniform() * L, y: rng.uniform() * L, vx: v * s * Math.cos(phi), vy: v * s * Math.sin(phi), mobile: true });
    }
  }
  for (let i = 0; i < (p.passive?.n ?? 0); i++) {
    parts.push({ x: rng.uniform() * L, y: rng.uniform() * L, vx: 0, vy: 0, mobile: true });
  }
  for (let i = 0; i < (p.fixed?.n ?? 0); i++) {
    parts.push({ x: rng.uniform() * L, y: rng.uniform() * L, vx: 0, vy: 0, mobile: false });
  }

  const R = Math.ceil(4 * sig);
  const gx = new Float64Array(2 * R + 2);
  const gy = new Float64Array(2 * R + 2);
  const img = new Float64Array(N * N);
  const frames = [];
  const step = Math.sqrt(2 * D * dt);
  for (let t = 0; t < T; t++) {
    img.fill(bg);
    for (const q of parts) {
      const px = q.x / dxUm, py = q.y / dxUm;
      const x0 = Math.floor(px) - R, y0 = Math.floor(py) - R;
      for (let i = 0; i < 2 * R + 2; i++) {
        const ddx = x0 + i - px, ddy = y0 + i - py;
        gx[i] = Math.exp((-ddx * ddx) / (2 * sig * sig));
        gy[i] = amp * Math.exp((-ddy * ddy) / (2 * sig * sig));
      }
      for (let j = 0; j < 2 * R + 2; j++) {
        const yy = (((y0 + j) % N) + N) % N;
        const row = yy * N;
        const gyj = gy[j];
        for (let i = 0; i < 2 * R + 2; i++) {
          const xx = (((x0 + i) % N) + N) % N;
          img[row + xx] += gx[i] * gyj;
        }
      }
    }
    const f = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) {
      const v = Math.round((img[i] + noise * rng.normal()) * 255);
      f[i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    frames.push(f);
    for (const q of parts) {
      if (!q.mobile) continue;
      q.x += (q.vx + Vx) * dt + step * rng.normal();
      q.y += (q.vy + Vy) * dt + step * rng.normal();
      q.x = ((q.x % L) + L) % L;
      q.y = ((q.y % L) + L) % L;
    }
  }

  const valid = new Array(T).fill(true);
  const drng = makeRng((p.seed ?? 1) * 7919 + 13);
  if (p.dropExact) {
    const idx = Array.from({ length: T - 1 }, (_, i) => i + 1);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(drng.uniform() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let i = 0; i < p.dropExact; i++) valid[idx[i]] = false;
  } else if (p.dropFrac) {
    for (let t = 1; t < T; t++) if (drng.uniform() < p.dropFrac) valid[t] = false;
  }
  const meanSpeed = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0;
  return { frames, valid, N, T, fps, dxUm, truth: { meanSpeed, Vx, Vy, D } };
}

/** Convert an 8-bit frame to float luma in [0,1] (what the GPU sees). */
export function lumaOf(frame) {
  const out = new Float64Array(frame.length);
  for (let i = 0; i < frame.length; i++) out[i] = frame[i] / 255;
  return out;
}

const BASE = { N: 256, T: 256, fps: 30, dxUm: 1, Dum2s: 0.4, blobSigmaPx: 1.5, amp: 0.15, background: 0.25, noise: 0.01 };

/**
 * @typedef {object} Scenario
 * @property {string} name
 * @property {string} description
 * @property {SimParams} sim
 * @property {'3d'|'2d'} geometry
 * @property {boolean} [shuffle]
 * @property {'MOTILE'|'NOT_MOTILE'|'DIRECTED_TRANSPORT'|'NO_DYNAMICS'} expect
 * @property {{vbar?: {value: number, relTol: number}, alpha?: {value: number, absTol: number}, drift?: {speed: number, angleDeg: number, relTol: number, degTol: number}}} checks
 */

/**
 * The 11-scenario validation matrix. `expect` is the required verdict; `checks`
 * are quantitative tolerances on the recovered parameters. Swimmers and passive
 * cells share one blob amplitude, so the true alpha is the swimmer number
 * fraction (150/300, 150/250, 25/325).
 */
/** @type {Scenario[]} */
export const SCENARIOS = [
  {
    name: 'motile_3d',
    description: '150 swimmers (3D isotropic, vbar 20 um/s, sigma 8) + 150 Brownian cells',
    sim: { ...BASE, seed: 11, swimmers: { n: 150, vbar: 20, sigma: 8, geometry: '3d' }, passive: { n: 150 } },
    geometry: '3d',
    expect: 'MOTILE',
    checks: { vbar: { value: 20, relTol: 0.2 }, alpha: { value: 0.5, absTol: 0.2 } },
  },
  {
    name: 'motile_2d',
    description: '150 in-plane swimmers (vbar 25 um/s, sigma 10) + 100 Brownian cells, 2D kernel',
    sim: { ...BASE, seed: 12, swimmers: { n: 150, vbar: 25, sigma: 10, geometry: '2d' }, passive: { n: 100 } },
    geometry: '2d',
    expect: 'MOTILE',
    checks: { vbar: { value: 25, relTol: 0.2 }, alpha: { value: 0.6, absTol: 0.2 } },
  },
  {
    name: 'sparse_swimmers',
    description: '25 swimmers among 300 Brownian cells (low motile fraction)',
    sim: { ...BASE, seed: 13, swimmers: { n: 25, vbar: 20, sigma: 8, geometry: '3d' }, passive: { n: 300 } },
    geometry: '3d',
    expect: 'MOTILE',
    checks: { alpha: { value: 25 / 325, absTol: 0.07 } },
  },
  {
    name: 'killed_brownian',
    description: 'Control 2: bleach-killed sample, 300 Brownian cells, no swimmers',
    sim: { ...BASE, seed: 14, passive: { n: 300 } },
    geometry: '3d',
    expect: 'NOT_MOTILE',
    checks: {},
  },
  {
    name: 'static_debris',
    description: '300 particles stuck to the glass, camera noise only',
    sim: { ...BASE, seed: 15, fixed: { n: 300 } },
    geometry: '3d',
    expect: 'NO_DYNAMICS',
    checks: {},
  },
  {
    name: 'shuffle_null',
    description: 'Control 1: the motile_3d movie with its frame order shuffled',
    sim: { ...BASE, seed: 11, swimmers: { n: 150, vbar: 20, sigma: 8, geometry: '3d' }, passive: { n: 150 } },
    geometry: '3d',
    shuffle: true,
    expect: 'NO_DYNAMICS',
    checks: {},
  },
  {
    name: 'flow_0deg_10ums',
    description: 'Control 3: Brownian cells in a uniform 10 um/s flow at 0 deg',
    sim: { ...BASE, seed: 17, passive: { n: 250 }, drift: { speed: 10, angleDeg: 0 } },
    geometry: '3d',
    expect: 'DIRECTED_TRANSPORT',
    checks: { drift: { speed: 10, angleDeg: 0, relTol: 0.15, degTol: 10 } },
  },
  {
    name: 'flow_135deg_5ums',
    description: 'Brownian cells in a uniform 5 um/s flow at 135 deg',
    sim: { ...BASE, seed: 18, passive: { n: 250 }, drift: { speed: 5, angleDeg: 135 } },
    geometry: '3d',
    expect: 'DIRECTED_TRANSPORT',
    checks: { drift: { speed: 5, angleDeg: 135, relTol: 0.15, degTol: 10 } },
  },
  {
    name: 'flow_60deg_1ums',
    description: 'Brownian cells in a slow 1 um/s drift (below the 2 um/s flow threshold)',
    sim: { ...BASE, seed: 19, passive: { n: 250 }, drift: { speed: 1, angleDeg: 60 } },
    geometry: '3d',
    expect: 'NOT_MOTILE',
    checks: {},
  },
  {
    name: 'motile_in_flow',
    description: '150 swimmers + 150 Brownian cells advected by 8 um/s at 210 deg',
    sim: { ...BASE, seed: 20, swimmers: { n: 150, vbar: 20, sigma: 8, geometry: '3d' }, passive: { n: 150 }, drift: { speed: 8, angleDeg: 210 } },
    geometry: '3d',
    expect: 'DIRECTED_TRANSPORT',
    checks: { drift: { speed: 8, angleDeg: 210, relTol: 0.2, degTol: 15 } },
  },
  {
    name: 'dropped_frames_20pct',
    description: 'motile_3d sample with 20 % of frames dropped at random',
    sim: { ...BASE, seed: 21, swimmers: { n: 150, vbar: 20, sigma: 8, geometry: '3d' }, passive: { n: 150 }, dropFrac: 0.2 },
    geometry: '3d',
    expect: 'MOTILE',
    checks: { vbar: { value: 20, relTol: 0.2 }, alpha: { value: 0.5, absTol: 0.2 } },
  },
];

/** The GPU parity harness movie: motile_3d with exactly 37 of 256 frames dropped (219 used). */
/** @type {SimParams} */
export const HARNESS_SIM = { ...SCENARIOS[0].sim, dropExact: 37, seed: 101 };
