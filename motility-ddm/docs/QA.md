# Judge Q&A: 10 hostile questions, short answers

Each answer is two sentences: the claim, then the evidence.

---

**1. "Your pixel pitch is a guess. Every number you report is wrong by an unknown factor."**

Δx is calibrated per setup against a 0.01 mm stage micrometer, with focus locked first because focus breathing changes magnification (README §3). Speeds scale exactly linearly with Δx while the gates TSI, ΔBIC and α are dimensionless and don't depend on it, so a 5 % calibration error is a 5 % speed error and can never flip `NOT_MOTILE` to `MOTILE` (only the 2 µm/s flow and swimmer floors move by the same 5 %).

**2. "What range of q do you actually measure, and why trust its ends?"**

Bins span |k| = 4…100 FFT indices: q = 0.020–0.49 µm⁻¹ at 5 µm/px, or length scales of 13–320 µm. Below k = 4 the Blackman–Harris main lobe (±4 bins) and the finite ROI dominate, and above k = 100 the optics and the camera-noise floor B(q) take over. The fit averages each bin's model over its true |q| distribution (the bin's modes convolved with the window leakage), and the plot only shows bins whose dynamic amplitude clearly exceeds noise.

**3. "A phone screen under a sample is a heater. How is that not convection?"**

Heating from below is exactly Rayleigh–Bénard. Ra = gβΔT h³/(νκ) is about 1.5·10⁴ for a 1 cm cuvette at ΔT = 1 K (onset 1708), but about 0.015 for our sealed 100 µm cell, because Ra ∝ h³ gives a factor of 10⁶. Any residual drift that remains is measured from Fourier phases and rejected by gate G2 as `DIRECTED_TRANSPORT`, which in simulation recovers 10, 5 and 8 µm/s flows to within 3 %.

**4. "33,024 modes × 256 frames on a phone GPU? You'll blow the storage limits."**

The history is 33,024 × 256 complex f32 = 67.6 MB, under the 128 MiB default `maxStorageBufferBindingSize`. The largest workgroup footprint is 10,240 B against a 16,384 B default, and no pipeline binds more than 7 storage buffers. The engine requests only default limits and checks every one at startup, so any conforming WebGPU device runs it, and the temporal pass is split into 8 submits of 4128 workgroups that keep the compositor fed.

**5. "Phones drop frames. Your correlation functions are garbage."**

Frames are slotted by `captureTime` into a uniform 256-slot clock, and gaps are masked, never interpolated. The masked Wiener–Khinchin estimator divides by the exact pair count N(τ) at every lag, which matches direct time-domain sums to 10⁻¹¹ with 20 % of frames missing (`t_fft.mjs`). The `dropped_frames_20pct` scenario still reads `MOTILE` with v̄ = 17.7 µm/s against 20 µm/s.

**6. "A uniform flow and in-plane swimmers give the same radially averaged D(q, τ). How do you tell them apart?"**

They are identical in the radial average: J₀(qVτ) is a delta-speed 2D swimmer. They differ in phase, because a flow rotates every mode's phase coherently, e^{−iq·Vτ}, while isotropic swimmers give incoherent phases. We fit V by weighted least squares on arg[C(τ+1)C̄(τ)] only at lags where the swimmer ISF is below 0.2, and the GPU reduces this to 8 sums per (bin, lag); a 1 µm/s drift below the floor is then compensated in both models and correctly reads `NOT_MOTILE`.

**7. "Auto-exposure changes look like motion. How do you know the camera didn't adjust?"**

We run single-shot autofocus, then freeze focus, exposure time, ISO and white balance in one `applyConstraints` call, handling Android's `{min, max, step}` ISO range, and then *read back* `getSettings()`. If exposure isn't verifiably `manual` at the requested value, measurement continues, but every verdict is labelled `UNLOCKED_EXPOSURE` and can never be reported as `MOTILE`.

**8. "α is not identifiable: swimmers that decorrelate inside one frame look like camera noise."**

Correct, and the code says so. G4 requires the swimmer decay to be resolved in at least one q-bin (q v̄ Δt ≤ 0.5 and q v̄ τ_max ≥ 3), otherwise it fails. α is a single q-independent fraction (swimmers and passive cells are the same scatterers), identified from the intermediate plateau of the low-q bins, and it recovered true swimmer fractions of 0.50, 0.60 and 0.077 as 0.50, 0.63 and 0.08.

**9. "With 1,376 data points, BIC will always prefer the bigger model."**

It did. Naive BIC with per-bin swimmer amplitudes gave ΔBIC = +523 on pure Brownian data, because the residuals of all lags share the same frames (lag-1 ρ ≈ 0.7). We now use n_eff = n(1−ρ)/(1+ρ) and one global α, and across 13 null runs ΔBIC sits near −15 (worst +22, which G4 still rejects), while motile runs give ΔBIC ≥ 277.

**10. "How do you know an f32 phone GPU computes what your float64 code computes?"**

Every kernel has a float64 twin with the same memory layout. The parity harness compares the 219-frame spectral history, all 4.2 million D/C values and the final verdict against fixed thresholds (spectrum < 1e-4, D/C p99 < 1e-3). Measured on SwiftShader with buffer ingest: spectrum 1.2e-5, D p99 9.9e-7, C p99 2.9e-7, identical verdict. Twiddles come from a float64 host table, so no kernel depends on the GPU's `sin`/`cos`; hardware results are logged in `docs/DEVICE.md`.
