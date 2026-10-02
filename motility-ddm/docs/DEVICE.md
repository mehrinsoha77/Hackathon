# Device test log: Commands 1–3

Nothing else starts until Commands 1, 2 and 3 each have a recorded result here. Paste the **full** `RESULT` line and say which branch was taken. Never edit a RESULT line by hand.

The Section B fixes the branches call for are **already in the code**, so these commands now *verify* them:

| Branch fix | Where | Status |
|---|---|---|
| Persistent staging buffer, no per-call `destroy()` | `DdmEngine.read` (`src/ddm-engine.ts`) | in place |
| `VideoFrame` kept open until that submit's `onSubmittedWorkDone()` | `DdmEngine.ingestFrame` → `track(close)` | in place |
| Precomputed twiddle table, no `sin`/`cos` in any kernel | `gpuTwiddles()` + `wtw` in `src/ddm.wgsl` | in place |
| rVFC watchdog (50 ms `setTimeout` poll of `video.currentTime`, `timeSource: 'watchdog'`) | `FrameLoop` (`src/camera.ts`) | in place |
| Temporal pass as 8 × 4128 workgroups with yields; skip ingest while > 2 submits are pending | `computeTemporal`, `MAX_PENDING` | in place |
| Fallback if video import is broken: canvas readback ingest | `?ingest=canvas`, automatic on device loss | in place |

## Setup (once, about 5 min)

```bash
cd motility-ddm
npm install
npm run build:harness            # -> test/gpu/harness.bundle.js      ("⚡ Done")
npm run build:smoke              # -> test/gpu/camera_smoke.bundle.js ("⚡ Done")
npx vite --port 5173 --strictPort    # leave running; prints Local: http://localhost:5173/
adb reverse tcp:5173 tcp:5173        # phone on USB with USB debugging; prints nothing, exit 0
```

If `adb devices` doesn't list the phone, fix USB debugging first. Every later step depends on it.

---

## Record 0: container pre-check (emulator, not hardware)

- **Date:** 2026-10-02
- **Environment:** Linux container, headless Chromium 141.0.7390.37
- **Flags:** `--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=swiftshader --use-webgpu-adapter=swiftshader`
- **Adapter:** `{"vendor":"google","architecture":"swiftshader"}`

This record is **not** a substitute for Commands 1–3. It establishes what the emulator can and cannot show.

### 0a. Harness, `VideoFrame` path (`test/gpu/harness.html`)

```
RESULT {"error":"A valid external Instance reference no longer exists.\nOperationError: A valid external Instance reference no longer exists."}
```

There's no JS stack frame. The rejection comes from `onSubmittedWorkDone` after the first submit. Minimal reproduction, a standalone page:

1. Create a 16 × 16 `VideoFrame` in RGBA, RGBX, BGRA, or from an `OffscreenCanvas`.
2. `importExternalTexture`, then a 1-workgroup compute dispatch that does `textureLoad`, then `submit`.
3. The frame stays open until after the `onSubmittedWorkDone` attempt.

Result: `onSubmittedWorkDone` and `mapAsync` both reject with the same message on the **first** submit, and the next `importExternalTexture` throws "Failed to import texture from video". The same happens with `copyExternalImageToTexture` into an ordinary `rgba8unorm` texture.

**Conclusion:** SwiftShader can't take *any* video frame into WebGPU. This is an emulator artifact, not an engine lifetime bug. The `ingest` kernel must be validated on hardware (Command 1).

### 0b. Harness, `?src=buffer` path (every kernel except `ingest`)

Run with `node test/gpu/run_gpu_tests.mjs --swiftshader`.

```
RESULT {"framesUsed":219,"source":"ingestLuma","adapter":{"vendor":"google","architecture":"swiftshader","description":""},"msIngest":27280,"msPerFrame":124.56,"msTemporal":27811,"throttled":0,"spectrum":{"maxAbsErrOverRms":0.000012055366377343895,"rms":0.007468479488531222,"maxAt":{"kx":0,"ky":0,"slot":245,"absF":0.014273420604523789}},"temporal":{"D":{"p99":9.914784868669813e-7,"max":0.0000024118219243973726,"mean":5.654686058239844e-7},"C":{"p99":2.9224915465420054e-7,"max":0.000001104332341128611,"mean":8.959487024260914e-8},"P":{"p99":9.141142527369084e-7,"max":0.0000010333480986446375,"mean":5.982031120128689e-7}},"radial":{"DmaxRel":0.000002073731781911814},"analysis":{"gpu":{"verdict":"MOTILE","alpha":0.5457022420707572,"vbar_um_s":19.561726819791165,"sigma_v_um_s":9.907492972149996,"D_um2_s":0.25580194738775724,"dBIC":337.388173212082,"TSI":0.9831578050520072,"mu":1.0045816279222564,"V_um_s":[0.006621989594907793,-0.03472709599331667]},"cpu":{"verdict":"MOTILE","alpha":0.5457033626061936,"vbar_um_s":19.56172099527774,"sigma_v_um_s":9.907492049118686,"D_um2_s":0.25580265125816387,"dBIC":337.3893405790099,"TSI":0.9831578635789362,"mu":1.0045819237483724,"V_um_s":[0.006622018711396554,-0.03472709258605694]}},"checks":{"spectrum":true,"D":true,"C":true,"P":true,"verdict":true,"alpha":true,"vbar":true},"pass":true}
```

| Field | Value | Pass condition | |
|---|---|---|---|
| `spectrum.maxAbsErrOverRms` | 1.206e-5 | < 1e-4 | PASS |
| `temporal.D.p99` | 9.915e-7 | < 1e-3 | PASS |
| `temporal.C.p99` | 2.922e-7 | < 1e-3 | PASS |
| `temporal.P.p99` | 9.141e-7 | < 1e-4 | PASS |
| verdict gpu / cpu | MOTILE / MOTILE | equal | PASS |
| \|α_gpu − α_cpu\| | 1.1e-6 | < 0.01 | PASS |
| v̄ relative difference | 3.0e-7 | < 1 % | PASS |

Two shader changes came out of this run. The RESULT above is verbatim from 2026-10-02 and was reproduced bit-identically on a second run:

- With a single-pass f32 mean, the spectral error was 2.56e-4, concentrated at the DC mode: a mean error is multiplied by (Σw)²/N ≈ 33.
- `reduce_mean` now returns the mean as an f32 (hi, lo) pair and `fft_rows` subtracts both parts. The error dropped to 1.2e-5.

### 0c. Camera smoke with Chromium's fake camera (`--use-fake-device-for-media-stream`)

| `?ingest=` | Outcome |
|---|---|
| `none` | Full sequence: `frames=72 slots=40`, `timeSource:"captureTime"`, `missed:0`, `dtMs:50.2` (the fake device runs at 20 fps), `queue idle`. The lock readback is `LOCKED`, because the fake device exposes manual exposure and focus. |
| `canvas` | Frames keep arriving during the temporal pass (`frames=97`), so the compositor is not starved. `queue NOT idle after 10 s` is expected on SwiftShader, whose temporal pass takes about 26 s on the CPU. |
| `external` | Stops after `step: frame 1`. This is the same SwiftShader video-import failure as 0a. |

---

## Command 1: GPU parity on a real desktop GPU

Use desktop Chrome stable on **Windows or macOS** (Linux Chrome needs WebGPU flags). Open `http://localhost:5173/test/gpu/harness.html` and watch DevTools → Console. Or run headless: `npm run test:gpu`, which applies the same thresholds. Time budget: 10 min.

```
Date:
Machine / OS:
Chrome version:
Adapter (from "step: adapter"):
RESULT (full line):

```

| Field | Value | Pass condition | PASS/FAIL |
|---|---|---|---|
| `spectrum.maxAbsErrOverRms` | | < 1e-4 | |
| `temporal.D.p99` | | < 1e-3 | |
| `temporal.C.p99` | | < 1e-3 | |
| `temporal.P.p99` | | < 1e-4 | |
| `analysis.gpu.verdict` = `analysis.cpu.verdict` | | equal | |
| \|α_gpu − α_cpu\| | | < 0.01 | |
| v̄ relative difference | | < 1 % | |

**Branch taken:**

| Outcome | Action |
|---|---|
| All pass | SwiftShader's error is confirmed as an emulator artifact. Copy the RESULT into the README parity table and go to Command 2. |
| `external Instance reference` on hardware | Read the error. The two Section B fixes are already in place, so this is new. Record the full message and adapter. Run `?src=buffer`: if that passes, the fault is in video import on this driver, and the app's automatic `?ingest=canvas` fallback covers it. Note it here. |
| `spectrum` ≥ 1e-4 | `maxAt` in the RESULT gives (k_x, k_y, slot). At (0,0) suspect the mean/`reduce_mean`; otherwise suspect `fft_rows`/`fft_cols`. Dump `frame`, `stats` and `rows` with `engine.read(...)`. |
| Spectrum passes, `D`/`C` p99 ≥ 1e-3 | `temporal` kernel. Dump `wb` after each `fft_wg` into a debug buffer. |
| `analysis.cpu.verdict` is not MOTILE | Judge parity on the error fields only. |
| "WebGPU is not available" | Wrong browser or OS: switch machines, don't debug. |

---

## Command 2: the same harness on the target phone's GPU

On the phone, open `http://localhost:5173/test/gpu/harness.html` in Chrome, with the console mirrored on the desktop via `chrome://inspect`. Allow up to 3 min, because the float64 reference runs on the phone CPU. Time budget: 20 min.

```
Date:
Phone model / Android version:
Chrome version:
GPU (Adreno xxx / Mali-Gxx) and adapter info:
RESULT (full line):

```

Same pass table as Command 1:

| Field | Value | PASS/FAIL |
|---|---|---|
| spectrum | | |
| D p99 | | |
| C p99 | | |
| P p99 | | |
| verdict | | |
| α | | |
| v̄ | | |

**Branch taken:**

| Outcome | Action |
|---|---|
| Pass | Go to Command 3. |
| "WebGPU is not available" | This phone can't be the target: it needs Android 12+, Chrome 121+ and an Adreno or Mali GPU. Try every team phone now. If none qualifies, the project is **NO-GO now**: take the Section F pivot immediately, without waiting for H18. |
| `GPU limit … needs …` | Record the exact message. Every limit the engine asks for is a WebGPU default, so treat this as "no qualifying phone". |
| `WGSL compilation failed: line L:C …` or `Pipeline creation failed for <entry>` | Driver bug. Record the line and entry point. Twiddles are already table-driven, so look at the named entry point. |
| Desktop passed, phone fails D/C p99 | f32 precision on the mobile GPU. Record `max` and `mean` as well as `p99`. |

---

## Command 3: real camera plus WebGPU on the target phone

On the phone, open `http://localhost:5173/test/gpu/camera_smoke.html` and tap **Allow**. Console via `chrome://inspect`. Expected within 25 s:

```
step: ingest mode external
step: getUserMedia
step: started {"width":…,"height":…,"frameRate":…,…}
step: locked {…}
step: engine ready
step: frame 1 t=…ms src=captureTime
step: frame 30 …   step: frame 60 …
step: temporal started (stress: 8 chunks while frames keep arriving)
step: loop done frames=≥71 slots=40
step: queue idle
RESULT {"lock":{"exposure":"locked",…,"measurementAllowed":true},"timeSource":"captureTime","frames":≥71,"slots":40,"missed":≤4,"dtMs":≈33.3,…}
```

`dtMs` is about 33.3 at 30 fps or about 16.7 at 60 fps. Time budget: 15 min.

```
Date:
Phone:
"step: started" settings line (full):

RESULT (full line):

```

**Branch taken:**

| Outcome | Action |
|---|---|
| Full sequence, `slots:40`, `missed` ≤ 4 | The live loop is viable: go to the Section C schedule. |
| `watchdogFrames` > 0 or `src=watchdog` | rVFC went silent with WebGPU active (unknown #3 confirmed). The watchdog is already delivering frames; record the count. |
| No `step: frame 1` within 5 s of `engine ready` (the page logs `NO frame within 5 s`) | Rerun with `?ingest=none`. If frames arrive then, video import is blocking the loop: run the app with `?ingest=canvas`. If they still don't arrive, it's a camera or rVFC problem: record it. |
| Frames stop before `slots=40` | GPU work is starving the compositor. Record `throttled`. Compare with `?ingest=canvas`. |
| `loop done` but `queue NOT idle after 10 s` | The GPU queue is stalled on the imported frame. Record it, and compare with `?ingest=canvas`. |
| `lock.exposure` is `"mismatch"` or `"unsupported"` | Paste the full `step: started` settings line above: it is the constraint shape to patch against. Verdicts on this phone will read `UNLOCKED_EXPOSURE`. |
| `timeSource:"mediaTime"` | Coarser timestamps but usable. Note it; no fix needed. |

---

## Go / no-go (by H14, over USB)

```
Command 1:  PASS / FAIL   (date, machine)
Command 2:  PASS / FAIL   (date, phone)
Command 3:  PASS / FAIL   (date, phone)
Thin-cell live sample reads MOTILE, and Controls 1–3 read as specified:  YES / NO
Decision:   GO / NO-GO (Section F pivot)
Signed:
```

GitHub Pages deployment happens only after this decision.
