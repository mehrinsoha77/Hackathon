# Video script: 4:00, screen recording plus bench camera

**Format.** 16:9, 1080p. Mirror phone A's screen to the laptop with `scrcpy` and record the laptop.

- **A-roll:** a second camera on the bench.
- **Overlay:** the timestamps below are hard cut points.
- **Narration:** about 150 words per minute.

Every number shown on screen must come from the live dashboard or from `npm test` / `docs/DEVICE.md` output recorded the same day.

**Before recording:**

- Commands 1–3 are recorded in `docs/DEVICE.md` with a GO decision.
- Pixel pitch is calibrated with the micrometer, and the value is written on a card.
- Three thin cells are built: live sample, bleach-killed sample, and an unsealed flow cell with a paper wick.
- Phone B is at 100 % brightness.

---

## 0:00–0:15: hook

| Time | Picture | Narration |
|---|---|---|
| 0:00–0:06 | Close-up: a drop of pond water, then phone A over phone B on the bench. | "Is there anything alive and swimming in this drop?" |
| 0:06–0:15 | Dashboard: `MOTILE` badge, the D(q, τ) fan, α and v̄ cards. | "Two phones, no microscope, and an answer that is built to say *no* when it should." |

## 0:15–0:40: the claim and the method

| Time | Picture | Narration |
|---|---|---|
| 0:15–0:27 | Animated: frame differences → FFT → D(q, τ). | "Differential dynamic microscopy asks how fast the image decorrelates at each length scale. Swimmers decorrelate as q to the minus one, diffusion as q to the minus two." |
| 0:27–0:40 | Pipeline diagram: camera → WebGPU (6 kernels) → fit → 5 gates. | "All of it runs on the phone's GPU: 33,000 Fourier modes, 256 frames of history, refreshed every one and a half seconds." |

## 0:40–1:10: dual-phone optics and the thin cell

| Time | Picture | Narration |
|---|---|---|
| 0:40–0:52 | Hands: two tape layers on a slide, 15 µl of sample, coverslip, nail-polish seal. Caption: "h ≈ 100 µm, sealed". | "The sample lives in a sealed cell one tenth of a millimetre deep, never a cuvette." |
| 0:52–1:02 | Phone B under the cell (white, diffuser), phone A above with the clip-on macro lens. Caption: "Ra ∝ h³: 10⁶× less convection". | "Phone B is the backlight, and it heats from below. In a centimetre cuvette that drives convection; at a hundred microns the Rayleigh number drops a million-fold." |
| 1:02–1:10 | Stage micrometer on the dashboard; cursor measures 500 µm. Caption: "Δx = 4.81 µm/px" (use the real value). | "We calibrate the pixel pitch against a stage micrometer. Every speed scales with it." |

## 1:10–1:35: camera register lock

| Time | Picture | Narration |
|---|---|---|
| 1:10–1:20 | Tap **Start Camera**: live video with the dashed ROI 256 × 256. | "We analyse the centre 256 by 256 pixels at native resolution, with no resampling." |
| 1:20–1:35 | Tap **Lock Camera**: the badge turns `LOCKED`; zoom into the readback line (exposure, ISO, focus, white balance). | "Auto-exposure steps look exactly like motion. We freeze exposure, ISO, focus and white balance in one call and read them back. If the phone won't confirm the lock, every verdict says UNLOCKED_EXPOSURE." |

## 1:35–2:15: live measurement

| Time | Picture | Narration |
|---|---|---|
| 1:35–1:45 | History chip counts to 192/256; the D(q, τ) curves appear. | "After six seconds of history the structure function resolves, one curve per length scale." |
| 1:45–2:00 | Zoom on the scaling plot: points on the μ = 1 line. | "The half-decay time scales as q to the minus one: ballistic swimming, not diffusion." |
| 2:00–2:15 | Zoom on the verdict panel: `MOTILE`, α, v̄ ± σ_v, all five gates ticked, ΔBIC. Read the numbers off the screen. | "The swimming model beats pure diffusion by a ΔBIC of [value], swimmers carry [α] of the dynamic contrast, at a mean speed of [v̄] microns per second." |

## 2:15–2:40: Control 1, frame-shuffle null test

| Time | Picture | Narration |
|---|---|---|
| 2:15–2:25 | Tick **Shuffle frame order**. | "Control one: same frames, order destroyed." |
| 2:25–2:40 | Within one or two refreshes: `NO_DYNAMICS`, G1 TSI ≈ 0.0x ✗. Untick; it returns to `MOTILE`. | "Without temporal order there is no dynamics, so the detector reports none, then recovers when we restore the order." |

## 2:40–3:05: Control 2, bleach kill switch

| Time | Picture | Narration |
|---|---|---|
| 2:40–2:50 | Swap in the bleach-killed cell (same culture, about 1:20 bleach, 2 min). Re-lock. | "Control two: the same culture, killed with bleach." |
| 2:50–3:05 | After 6 s: `NOT_MOTILE` or `NO_DYNAMICS` (say whichever appears); μ moves toward 2 if Brownian motion is visible. | "Dead cells still jiggle or sit still, and both read as not motile. Same optics, same lock, opposite verdict." |

## 3:05–3:30: Control 3, directed flow

| Time | Picture | Narration |
|---|---|---|
| 3:05–3:15 | Unsealed flow cell: touch a paper wick to one edge. | "Control three: dead cells carried by a flow, the classic false positive." |
| 3:15–3:30 | `DIRECTED_TRANSPORT`, drift card \|V\| ± σ @ angle, G2 ✗; the D curves ring with J₀ oscillations. | "A uniform flow rotates every Fourier phase coherently. We measure that rotation directly, and the detector reports transport, not life." |

## 3:30–3:50: why to trust the numbers

| Time | Picture | Narration |
|---|---|---|
| 3:30–3:40 | Terminal: `npm test` tail, 11/11 scenarios correct. | "Eleven synthetic scenarios, including dropped frames, sparse swimmers and flows at three speeds, all classified correctly." |
| 3:40–3:50 | Terminal: the GPU parity RESULT line from `docs/DEVICE.md` (Command 2, on this phone). | "Every GPU kernel is checked against a double-precision reference on this phone: errors below one part in ten thousand." |

## 3:50–4:00: close

| Time | Picture | Narration |
|---|---|---|
| 3:50–4:00 | Dashboard wide shot, repo URL. | "MotilityDDM: motility you can falsify, on hardware you already own." |

---

## What the video MUST NOT claim

1. **No particle sizing.** MotilityDDM has no Stokes–Einstein inversion and reports no particle size or hydrodynamic radius.
2. **No counts or concentrations.** α is the swimmer share of the *dynamic contrast*. It is not a cell count, a concentration, or a "percent alive", except under the equal-contrast assumption stated in the README.
3. **No species identification and no diagnosis.** Nothing about pathogens, water safety, fertility or semen analysis, or clinical use.
4. **No unmeasured speed accuracy.** v̄ is only as accurate as the Δx calibration shown on screen. Never quote speeds from an uncalibrated setup.
5. **No synthetic data shown as real.** If the synthetic demo appears, the `SYNTHETIC` badge must be visible and the narration must say "simulated".
6. **No cuvette.** Never show or suggest a cuvette or an open drop as a valid sample cell.
7. **No unrecorded hardware validation.** Only claim GPU parity on devices whose RESULT line is in `docs/DEVICE.md`.
8. **No UNLOCKED_EXPOSURE result presented as motile.** If the phone can't lock exposure, show the badge and say so.
9. **No sensitivity or specificity figures** beyond the 11 simulated scenarios and the three live controls actually performed.
10. **No "bacteria detection" in general.** At about 5 µm/px the demonstrated samples are protists and other large swimmers; anything smaller requires its own calibration and controls.
