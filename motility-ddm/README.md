# MotilityDDM

**Is this sample alive and swimming? A phone camera and the phone's GPU answer that with differential dynamic microscopy (DDM), and five falsification gates decide when the answer is "no".**

MotilityDDM records the 256 × 256 px centre of a phone camera image of a thin sealed sample cell. On WebGPU it computes the image structure function D(q, τ) for all 33,024 Fourier modes over a 256-frame history. It then fits a physical model of swimmers plus Brownian particles plus camera noise. It reports `MOTILE` only if all of the following hold:

- the movie has temporal structure,
- the motion is not a uniform flow,
- a swimming model beats a diffusion model by ΔBIC > 10,
- swimmers carry more than 5 % of the dynamic contrast,
- the camera's exposure was verifiably locked.

| Verdict | Meaning |
|---|---|
| `MOTILE` | All five gates pass |
| `NOT_MOTILE` | Dynamics present but explained by diffusion (ΔBIC ≤ 10) or swimmer share α ≤ 0.05 |
| `DIRECTED_TRANSPORT` | A uniform drift \|V\| > 3σ and \|V\| > 2 µm/s was detected from Fourier phases |
| `NO_DYNAMICS` | Temporal structure index TSI < 0.15 (static, pure noise, or shuffled frames) |
| `UNLOCKED_EXPOSURE` | Exposure lock could not be verified; the physics verdict is shown but never reported as `MOTILE` |

**What it does not do.** It does not size particles. There is no Stokes–Einstein inversion. It does not count cells, identify species, or make any diagnostic claim.

---

## Contents

1. [Optics: the thin cell](#1-optics-the-thin-cell)
2. [Bill of materials](#2-bill-of-materials)
3. [Pixel pitch calibration](#3-pixel-pitch-calibration)
4. [Physics and mathematics](#4-physics-and-mathematics)
5. [GPU architecture](#5-gpu-architecture)
6. [Setup, tests and results](#6-setup-tests-and-results)
7. [Running a measurement](#7-running-a-measurement)
8. [Repository layout](#8-repository-layout)
9. [Limitations](#9-limitations)

---

## 1. Optics: the thin cell

```
            Phone A (measurement): rear camera, focus/exposure/ISO/WB locked
            ┌───────────────────────────────┐
            │  sensor ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │   ROI: centre 256 × 256 native px
            │            ╲     ╱             │
            │             ╲   ╱  phone lens  │
            └──────────────╲─╱──────────────┘
                            ╳        ← clip-on macro lens (10–20×), working distance d_w
                           ╱ ╲
                          ╱   ╲      light scattered/absorbed by organisms
        ──────────────────────────────────────────  coverslip #1.5 (170 µm)
        ░░░░░░░░░░░ sample, h ≈ 100 µm ░░░░░░░░░░░   ← double-sided tape spacers, sealed
        ──────────────────────────────────────────  microscope slide (1 mm)
             ↑     ↑     ↑     ↑     ↑     ↑
        ┌──────────────────────────────────────┐
        │ diffuser (1–2 sheets of paper)       │
        │ Phone B screen, full white, 100 %    │   (backlight; spaced ≥ 5 mm by a
        └──────────────────────────────────────┘    slide box or folded card)
```

**Use a sealed thin cell, never a cuvette.** Two layers of double-sided tape (about 90–100 µm) between a slide and a coverslip, with the edges sealed by nail polish or petroleum jelly, gives a liquid layer h ≈ 100 µm. A 1 cm cuvette breaks the measurement in three ways:

1. **Depth of field.** At about 5 µm/px the in-focus slab is a few hundred µm thick. A 100 µm cell lies entirely inside it, so every organism is imaged at the same magnification. In a 1 cm cuvette most of the volume is out of focus: blurred organisms at depth-dependent magnification add a q-dependent background, and the pixel pitch is no longer one number.
2. **Thermal convection.** Phone B heats the sample *from below*, which is the Rayleigh–Bénard geometry. The Rayleigh number is Ra = gβΔT h³/(νκ). For water (β = 2.1·10⁻⁴ K⁻¹, ν = 10⁻⁶ m²/s, κ = 1.4·10⁻⁷ m²/s) with ΔT = 1 K:
   - h = 1 cm gives Ra ≈ 1.5·10⁴, far above the onset at Ra ≈ 1708. The cuvette convects at µm/s to tens of µm/s, and that flow looks like motility.
   - h = 100 µm gives Ra ≈ 0.015. Ra scales as h³, a factor of 10⁶, so there is no convection.
3. **Evaporation and sedimentation flows.** An open cuvette drifts. A sealed thin cell doesn't, and any residual drift is caught by gate G2.

The backlight must not flicker. OLED screens dim with PWM below full brightness, so run phone B at 100 % brightness on a white image, or use a battery LED torch through the diffuser. Flicker modulates every mode at once and lowers the TSI.

## 2. Bill of materials

| Item | Spec | Purpose |
|---|---|---|
| Phone A | Android 12+, Chrome 121+ with WebGPU (Adreno or Mali GPU) | Camera, GPU analysis, UI |
| Phone B | Any phone with a screen; or a white LED torch | Diffuse backlight |
| Clip-on macro lens | 10–20× clip-on macro (or a reversed small lens) | Brings the pixel pitch to about 1–5 µm/px |
| Microscope slides | 75 × 25 mm, 1 mm | Cell bottom |
| Coverslips | 22 × 22 mm, #1.5 | Cell top |
| Double-sided tape | About 90–100 µm per layer (measure with calipers) | Spacers that set h |
| Nail polish or petroleum jelly | — | Seal (no evaporation flow) |
| Stage micrometer | 1 mm in 0.01 mm divisions | Pixel pitch calibration |
| Diffuser | 1–2 sheets of printer paper | Uniform illumination |
| Stand | Slide box, books, or a 3D-printed clamp | Fixed working distance, no shake |
| Sample | Pond water (ciliates, *Euglena*), or a dense motile culture | Measurement |
| Dilute bleach | Household bleach about 1:20 | Control 2 (kill switch) |
| USB cable and laptop | `adb reverse` for development | Device tests (`docs/DEVICE.md`) |

## 3. Pixel pitch calibration

Δx is the size of one camera pixel *at the sample plane*. Every speed scales with it (v ∝ Δx), so calibrate it on the exact optical setup you measure with.

1. Mount the stage micrometer where the sample cell will go, with the lens at the same working distance.
2. Start the camera and press **Lock Camera**. Focus must be locked: changing focus changes magnification ("focus breathing"), which changes Δx.
3. Take a screenshot (or read pixel coordinates from the video in DevTools) inside the dashed 256 × 256 ROI. ROI pixels are native sensor pixels; the engine never resamples.
4. Measure the pixel distance P between two micrometer lines at least 0.5 mm apart (for example 50 divisions = 500 µm). Use the far edges of the lines, and average over three line pairs.
5. Δx = (distance in µm) / P. Example: 500 µm over 104 px gives Δx = 4.81 µm/px.
6. Enter Δx in **Pixel pitch at sample (µm/px)**. Changing it refits the last data immediately, with no new acquisition needed.
7. Recalibrate whenever the lens, the working distance or the camera resolution changes.

The q window follows from Δx. Bins span |k| = 4…100 FFT indices, so q = 2π|k|/(256 Δx). At Δx = 5 µm/px that is q = 0.020…0.49 µm⁻¹, or length scales of 320…13 µm.

## 4. Physics and mathematics

### 4.1 Image structure function

For a frame I(r, t), the windowed, mean-subtracted half-plane spectrum is

  F(q, t) = (1/N) Σ_r w(x) w(y) [I(r, t) − Ī(t)] e^{−i q·r},

where N = 256 and w is the 4-term Blackman–Harris window (side lobes −92 dB; it stops the hard ROI edge leaking into all q). The DDM structure function is

  D(q, τ) = ⟨ |F(q, t+τ) − F(q, t)|² ⟩_t = 2 [S(q) − Re C(q, τ)],

with C(q, τ) = ⟨F*(q, t) F(q, t+τ)⟩_t and static power S(q) = P(q) = ⟨|F|²⟩. For a dilute suspension, D(q, τ) = A(q)[1 − f(q, τ)] + B(q). Here f is the intermediate scattering function (ISF), A(q) the dynamic contrast, and B(q) = 2 × camera noise power.

### 4.2 Masked Wiener–Khinchin estimator

Frames are slotted by capture time, so dropped frames leave gaps. With validity mask m(t), y = m·F and p = m·|F|², zero-padded to 512 samples (linear, not circular, correlation):

| Quantity | Definition | FFT form |
|---|---|---|
| A(τ) | Σ_t m(t) p(t+τ) | IFFT(conj(M)·P) |
| B(τ) | Σ_t p(t) m(t+τ) | A(−τ) |
| G(τ) | Σ_t y*(t) y(t+τ) | IFFT(\|Y\|²) |
| N(τ) | Σ_t m(t) m(t+τ) | exact pair count |

From these:

- D(τ) = [A(τ) + B(τ) − 2 Re G(τ)] / N(τ)
- C(τ) = G(τ) / N(τ)
- P = Σp / Σm

This is exactly the average over valid pairs. Missing frames are never interpolated. `test/t_fft.mjs` checks it against direct O(T·L) sums with 20 % dropped and shuffled frames (agreement to 10⁻¹¹). M and N(τ) are the same for every mode, so the host computes them in float64. The GPU then never has to pack an O(1) mask into the same f32 FFT as an O(10⁻⁶) power signal.

### 4.3 Swimmer ISF (Wilson et al., PRL 106, 018101, 2011)

Straight-swimming organisms with speeds distributed as Schulz (Gamma),

  P(v) = vᶻ/Γ(Z+1) · ((Z+1)/v̄)^{Z+1} · e^{−v(Z+1)/v̄},  σ_v = v̄/√(Z+1),

have closed-form self-ISFs. Write Λ = q v̄ τ / (Z+1).

- **3D isotropic swimming, seen in projection:**
  ⟨sin(qvτ)/(qvτ)⟩ = sin(Z·arctan Λ) / [Z Λ (1+Λ²)^{Z/2}]
- **2D in-plane swimming:**
  ⟨J₀(qvτ)⟩ = P_Z(1/√(1+Λ²)) / (1+Λ²)^{(Z+1)/2}

The 2D form follows from the Laplace transform of vᶻ J₀(cv). P_Z is the Legendre function of real degree, evaluated from Laplace's first integral (1/π)∫₀^π (cos θ + i sin θ cos φ)^Z dφ. Its integrand is smooth and periodic, so the trapezoid rule converges exponentially. The fitter uses a precomputed (ln Z, ln Λ) table that agrees with the closed form to 5·10⁻⁶. `test/t_kernels.mjs` checks both forms against direct quadrature over P(v) to 2·10⁻⁵.

### 4.4 Model, VarPro and NNLS

For each radial bin b:

  D_b(τ) = A_b [1 − E_b(τ) ((1−α) + α g(q_b v̄ τ; Z))] + B_b,  E_b(τ) = e^{−D q_b² τ} J₀(q_b |V| τ)

- **α is q-independent.** Swimmers and passive cells are the same scatterers. With a free swimmer amplitude per bin (34 extra parameters) the fit absorbed correlated noise in pure-Brownian data and gave ΔBIC = +523. With one global α, null data sits at ΔBIC ≈ −15.
- **Bin averaging.** Each bin's model is averaged over 4 quantile nodes of its *effective* |q| distribution: the bin's own modes convolved with the window's spectral leakage.
- **J₀(q|V|τ)** is the radial average of e^{−iq·Vτ}. It compensates a measured uniform drift (§4.6).
- **Variable projection.** The amplitudes A_b, B_b ≥ 0 are linear. They are solved exactly inside every objective evaluation by NNLS: enumerate the supports, keep the feasible ones, take the minimum residual. KKT conditions are verified in the tests. The nonlinear search therefore sees only (v̄, Z, D, α): a multi-start Nelder–Mead in log space.
- **Speed floor.** Swimmer speeds are bounded below by 2 µm/s, the flow floor. A slower "swimmer" can't be told apart from residual drift.
- **Brownian null model.** A_b[1 − E_b(τ)] + B_b, with D (and |V| when a drift is significant) as the nonlinear parameters.

### 4.5 Model comparison with correlated residuals

All lags of D are built from the same frames, so residuals are strongly autocorrelated along τ (lag-1 ρ ≈ 0.6–0.8). BIC uses the effective sample size n_eff = n(1−ρ)/(1+ρ) of the swimmer-model residuals, for both models:

  BIC = n_eff ln(RSS/n) + k ln n_eff,  ΔBIC = BIC_Brownian − BIC_swimmer.

### 4.6 Drift velocity from Fourier phases

A uniform drift V multiplies every mode by e^{−iq·Vτ}. The radially averaged D only sees J₀(q|V|τ), which looks like in-plane swimming at a single speed, but each mode's *phase* rotates coherently. With z = C(q, τ+1) C*(q, τ), w = |z| and φ = arg z, a weighted least-squares fit φ ≈ −(q·V) Δt over all modes and lags gives V and its covariance. The covariance is inflated by the number of lags per mode, because those samples are correlated.

Lags are used only where the fitted swimmer self-ISF has decayed, g(q v̄ τ) < 0.2. There, swimmer phase memory is gone, and what remains is the coherent rotation of passive scatterers carried by the flow. The GPU reduces the eight sums (Σw, Σwk_x², Σwk_xk_y, Σwk_y², Σwk_xφ, Σwk_yφ, Σwφ², count) per (bin, lag), so phones never read back per-mode data.

Signs: +x is image right, +y is image *down*. `test/t_drift_phase.mjs` verifies the sign convention end to end with a translating image.

### 4.7 Falsification gates

| Gate | Test | Fails to |
|---|---|---|
| G1 | TSI ≥ 0.15. TSI is the mean of the 4 largest per-bin (D_late − D(1))/D_late, with D_late the mean over lags 64–127 | `NO_DYNAMICS` |
| G2 | not (\|V\| > 3σ_V and \|V\| > 2 µm/s) | `DIRECTED_TRANSPORT` |
| G3 | ΔBIC > 10 (swimmer vs Brownian, n_eff) | `NOT_MOTILE` |
| G4 | α > 0.05, and the swimmer decay is resolved (q v̄ Δt ≤ 0.5 and q v̄ τ_max ≥ 3) in at least one bin | `NOT_MOTILE` |
| G5 | Exposure readback is `manual` at the requested value | `UNLOCKED_EXPOSURE` |

The scaling exponent μ in τ½(q) ∝ q^{−μ} is reported as a diagnostic: μ ≈ 1 for ballistic swimming, μ ≈ 2 for diffusion. It is not a gate.

## 5. GPU architecture

`src/ddm.wgsl`. Every kernel has a float64 twin in `src/ddm_ref.js`.

| Pass | Dispatch | Work |
|---|---|---|
| `ingest` | 16 × 16 WG of 16 × 16 | `textureLoad` of the centred 256² ROI from `texture_external` at native resolution, BT.601 luma |
| `reduce_mean` | 1 WG of 256 | Two-pass mean kept as an f32 (hi, lo) pair, so it is never rounded to a single f32 |
| `fft_rows` | 256 WG of 128 | Subtract mean, Blackman–Harris w(x)w(y), 256-point radix-2 Stockham FFT, keep k_x 0…128 |
| `fft_cols` | 129 WG of 128 | 256-point FFT along y, scale 1/N, write ring slot |
| `temporal` | 8 × 4128 WG of 256 | Per mode: 4 × 512-point FFTs, masked Wiener–Khinchin → D, C (128 lags), P |
| `radial` | 32 WG of 128 | Bin-averaged D, P and the 8 drift sums per (bin, lag) |

**FFTs.** Stockham autosort in workgroup memory. Twiddles exp(−2πi m/512) come from a host-computed float64 table, so no kernel calls `sin`/`cos`. Results are identical on drivers with imprecise f32 transcendentals.

**Limits.** Only WebGPU defaults are used, so any conforming device qualifies:

| Resource | Used | Default limit |
|---|---|---|
| Largest binding (`hist`) | 67,633,152 B (67.6 MB = 64.5 MiB) | 128 MiB `maxStorageBufferBindingSize` |
| Workgroup memory | 10,240 B (2 × 4 KB FFT ping-pong + 2 KB twiddles) | 16,384 B |
| Storage buffers per pipeline | ≤ 7 | 8 |
| Workgroups per dispatch | 4128 | 65,535 |
| Total GPU memory | ≈ 119 MB (hist 67.6, C 33.8, D 16.9, small buffers) | — |

**Resource lifetime** (`src/ddm-engine.ts`):

- Readback uses **one persistent staging buffer**, reused in 8 MB chunks and never destroyed per call.
- An imported `VideoFrame` is closed only after the `onSubmittedWorkDone()` of the submit that sampled it.
- Frames are dropped, not queued, while more than 2 submits are pending. Dropped frames are masked out of the history.
- The temporal pass runs as 8 separate submits. The engine waits for each one and yields with `setTimeout(0)` so the compositor and camera keep their frame budget.
- New frames keep arriving during analysis, so the oldest 32 history slots are masked as a guard. A result is discarded, and the guard widened, if more frames than the guard arrived meanwhile.
- If the device is lost right after video import (a broken external-texture driver path), the app falls back to canvas readback ingest (`?ingest=canvas`).

## 6. Setup, tests and results

```bash
cd motility-ddm
npm install
npm test                  # CPU unit tests + 11-scenario validation matrix (about 1 min)
npm run build             # tsc && vite build -> dist/
npm run dev               # http://localhost:5173 (dashboard)
npm run build:harness     # test/gpu/harness.bundle.js
npm run build:smoke       # test/gpu/camera_smoke.bundle.js
npm run test:gpu          # headless GPU parity via playwright-core (add -- --swiftshader without a GPU)
```

On a phone: run `adb reverse tcp:5173 tcp:5173`, then open `http://localhost:5173/` in Chrome. localhost counts as a secure context, which camera and WebGPU both need. The device test protocol (Commands 1–3) is in [`docs/DEVICE.md`](docs/DEVICE.md).

### CPU validation matrix (`npm test`)

Float64 reference pipeline end to end: rendered 8-bit movie → spectrum → masked temporal → radial → fit → gates. Movies are 256 frames of 256² px at 1 µm/px and 30 fps, with blob particles, camera noise and 8-bit quantization.

| Scenario | Truth | Verdict | α (truth) | v̄ µm/s (truth) | \|V\| µm/s |
|---|---|---|---|---|---|
| motile_3d | 150 swimmers + 150 Brownian | MOTILE | 0.50 (0.50) | 23.4 (20, sample mean) | 0.09 |
| motile_2d | 150 in-plane swimmers + 100 Brownian, 2D kernel | MOTILE | 0.63 (0.60) | 21.8 (25) | 0.32 |
| sparse_swimmers | 25 swimmers + 300 Brownian | MOTILE | 0.08 (0.077) | 13.0 (20) | 0.12 |
| killed_brownian | 300 Brownian (Control 2) | NOT_MOTILE | 0.00 | — | 0.08 |
| static_debris | 300 particles stuck to glass | NO_DYNAMICS | — | — | 0.00 |
| shuffle_null | motile_3d with frames shuffled (Control 1) | NO_DYNAMICS | — | — | 0.01 |
| flow_0deg_10ums | Brownian + 10 µm/s at 0° (Control 3) | DIRECTED_TRANSPORT | — | — | 9.93 @ −1° |
| flow_135deg_5ums | Brownian + 5 µm/s at 135° | DIRECTED_TRANSPORT | — | — | 4.85 @ 135° |
| flow_60deg_1ums | Brownian + 1 µm/s (below the 2 µm/s floor) | NOT_MOTILE | 0.00 | — | 0.86 @ 60° |
| motile_in_flow | motile_3d + 8 µm/s at 210° | DIRECTED_TRANSPORT | — | — | 7.92 @ −150° |
| dropped_frames_20pct | motile_3d with 20 % frames dropped | MOTILE | 0.51 (0.50) | 17.7 (20) | 0.25 |

Notes on the table:

- **Gate G5.** The same motile movie with exposure unlocked reports `UNLOCKED_EXPOSURE` (physics: `MOTILE`).
- **Unit tests.** `t_fft` 44, `t_kernels` 55 and `t_drift_phase` 18 checks pass.
- **Seed robustness.** Under the final model, 7 of the 11 scenarios were re-run on 3–7 extra seeds each: killed_brownian (7), flow_60deg_1ums (6), sparse_swimmers and motile_3d (4 each), motile_2d, motile_in_flow and dropped_frames_20pct (3 each). Every run kept its verdict. The 13 null runs sit at ΔBIC ≈ −15 (worst +22, which G4 still rejects with α = 0.04); motile runs have ΔBIC ≥ 277.

### GPU kernels vs CPU reference

`test/gpu/harness.html` renders the `HARNESS_SIM` movie: 256 slots with 37 dropped, so 219 frames. It runs every kernel and compares against float64. Thresholds:

| Field | Pass condition |
|---|---|
| `spectrum.maxAbsErrOverRms` | < 1e-4 |
| `temporal.D.p99` | < 1e-3 |
| `temporal.C.p99` | < 1e-3 |
| `temporal.P.p99` | < 1e-4 |
| `analysis.gpu.verdict` | equals `analysis.cpu.verdict` |
| α | \|gpu − cpu\| < 0.01 |
| `vbar_um_s` | within 1 % |

**Recorded so far.** Hardware rows are filled in from `docs/DEVICE.md`.

| Device | Path | spectrum | D p99 | C p99 | P p99 | verdict gpu/cpu | \|Δα\| | Δv̄ | Result |
|---|---|---|---|---|---|---|---|---|---|
| SwiftShader (Chromium 141, headless, container) | `ingestLuma` (`?src=buffer`) | 1.21e-5 | 9.9e-7 | 2.9e-7 | 9.1e-7 | MOTILE / MOTILE | 1.1e-6 | 3.0e-7 | **PASS** |
| SwiftShader (same) | `VideoFrame` | — | — | — | — | — | — | — | emulator cannot import video, see `docs/DEVICE.md` |
| Desktop GPU (Command 1) | `VideoFrame` | | | | | | | | *pending* |
| Target phone (Command 2) | `VideoFrame` | | | | | | | | *pending* |

SwiftShader loses the device on the first submit that touches *any* imported video frame. The minimal reproduction in `docs/DEVICE.md` uses a 16 × 16 frame, any pixel format, `importExternalTexture` or `copyExternalImageToTexture`, with the frame held open past `onSubmittedWorkDone`. That is an emulator limitation, not an engine lifetime bug. The `ingest` kernel must therefore be validated by Command 1 on real hardware.

## 7. Running a measurement

1. **Build the thin cell.** Put two tape layers on the slide with a 10 mm channel between them. Add 15 µl of sample, lower the coverslip, and seal all four edges.
2. **Set up the optics.** Phone B goes underneath at 100 % white with a diffuser; phone A goes above with the macro lens, fixed on the stand.
3. **Start Camera.** Bring the organisms into focus inside the dashed ROI.
4. **Lock Camera.** Expect a `LOCKED` badge. `UNLOCKED_EXPOSURE` or `MISMATCH` means you should look at the readback line, because verdicts will not read `MOTILE`.
5. **Pixel pitch.** Enter the calibrated Δx (§3) and choose the geometry: 3D for free swimmers in a 100 µm cell, 2D only for organisms confined to a surface.
6. **Wait.** After 192 frames (6.4 s) a verdict appears and refreshes every 1.5 s.
7. **Run the controls**, as in `docs/VIDEO_SCRIPT.md`:
   - shuffle frames: must read `NO_DYNAMICS`;
   - bleach-killed sample: must read `NOT_MOTILE` or `NO_DYNAMICS`;
   - induced flow (unsealed cell with a paper wick): must read `DIRECTED_TRANSPORT`.

The **Run synthetic demo** button plays any validation scenario through the real GPU pipeline. It works without optics, and the UI labels it `SYNTHETIC`.

### Deployment (after the go/no-go)

`vite.config.ts` uses `base: './'`, so `dist/` works from any GitHub Pages sub-path. No COOP/COEP headers are needed, and GitHub Pages can't set them anyway: timing comes from `requestVideoFrameCallback` `captureTime`, not a cross-origin-isolated `performance.now()`. To publish, run `npm run build` and serve `dist/`, for example with a Pages workflow that uploads `motility-ddm/dist`.

## 8. Repository layout

```
motility-ddm/
├── index.html            dashboard (inline styles)
├── src/
│   ├── main.ts           UI, analysis loop, worker fit, plots, synthetic demo
│   ├── camera.ts         register locks, rVFC loop + watchdog, 256-slot frame clock
│   ├── ddm-engine.ts     WebGPU device, pipelines, buffers, lifetimes, readback
│   ├── ddm.wgsl          ingest, reduce_mean, fft_rows, fft_cols, temporal, radial
│   ├── ddm_ref.js        float64 twin of every kernel, bins, GPU metadata
│   ├── fft.js            float64 Stockham FFT, DFT, Blackman–Harris, 2D half-plane
│   ├── motility_fit.js   ISF kernels, VarPro/NNLS, BIC, drift, gates (+ worker entry)
│   └── env.d.ts
├── test/
│   ├── sim.mjs           movie simulator + the 11 scenarios
│   ├── t_fft.mjs  t_kernels.mjs  t_drift_phase.mjs  t_validation_matrix.mjs
│   └── gpu/  harness.{ts,html}  camera_smoke.{ts,html}  run_gpu_tests.mjs
└── docs/  DEVICE.md  QA.md  VIDEO_SCRIPT.md
```

## 9. Limitations

- **Calibration.** Speeds are only as good as Δx: a 5 % calibration error gives a 5 % speed error. Gates G1, G3 and G4 are dimensionless; the 2 µm/s flow and swimmer floors scale with Δx.
- **Global α.** α is the swimmer share of the *dynamic contrast*. It equals the number fraction only when swimmers and passive scatterers have the same optical contrast. Mixed samples (debris plus organisms) bias it.
- **Swimming model.** The model assumes straight swimming over the decorrelation time and an isotropic direction distribution. Strong run-and-tumble on the window time scale, chemotaxis or phototaxis break the J₀/sinc kernels. Taxis also shows up as a coherent drift (G2), which is intended.
- **Window length.** 256 frames (8.5 s at 30 fps) with lags up to 127 frames (4.2 s). Very slow swimmers whose decay isn't resolved fail G4 instead of being guessed.
- **Video import.** The `importExternalTexture` path has been validated only on hardware listed in `docs/DEVICE.md`.
