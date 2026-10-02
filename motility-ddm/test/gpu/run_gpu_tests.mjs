// Headless runner for the GPU parity harness: starts the Vite dev server,
// opens test/gpu/harness.html in Chromium, waits for the RESULT line and applies
// the pass thresholds (identical to THRESHOLDS in harness.ts).
//
//   node test/gpu/run_gpu_tests.mjs                 # hardware GPU (Chrome/Chromium)
//   node test/gpu/run_gpu_tests.mjs --swiftshader   # CPU emulation (CI, containers)
//   node test/gpu/run_gpu_tests.mjs --src=buffer    # bypass VideoFrame import
//
// Browser: CHROME_PATH, else playwright-core's resolution (PLAYWRIGHT_BROWSERS_PATH).
// Builds harness.bundle.js first. Exit code 0 only on a full pass.

import { execSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright-core';

const args = process.argv.slice(2);
const swiftshader = args.includes('--swiftshader');
// SwiftShader cannot import video frames at all: importExternalTexture and
// copyExternalImageToTexture with any VideoFrame lose the device on the first
// submit ("A valid external Instance reference no longer exists"), even for a
// 16x16 frame kept open past onSubmittedWorkDone. So under --swiftshader the
// harness defaults to the ingestLuma path, which validates every kernel except
// `ingest`; `ingest` is validated on real hardware (docs/DEVICE.md, Command 1).
const src = args.find((a) => a.startsWith('--src='))?.slice(6) ?? (swiftshader ? 'buffer' : undefined);
const timeoutMs = Number(args.find((a) => a.startsWith('--timeout='))?.slice(10) ?? 600000);

export const THRESHOLDS = {
  spectrumMaxAbsErrOverRms: 1e-4,
  temporalDp99: 1e-3,
  temporalCp99: 1e-3,
  temporalPp99: 1e-4,
  alphaAbs: 0.01,
  vbarRel: 0.01,
};

/** CHROME_PATH, else playwright's own browser, else any chromium-* under PLAYWRIGHT_BROWSERS_PATH. */
function resolveChrome() {
  if (process.env.CHROME_PATH && existsSync(process.env.CHROME_PATH)) return process.env.CHROME_PATH;
  try {
    const p = chromium.executablePath();
    if (p && existsSync(p)) return p;
  } catch {
    /* fall through */
  }
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (root && existsSync(root)) {
    for (const dir of readdirSync(root).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse()) {
      for (const sub of ['chrome-linux64', 'chrome-linux', 'chrome-win', 'chrome-mac']) {
        const exe = join(root, dir, sub, process.platform === 'win32' ? 'chrome.exe' : 'chrome');
        if (existsSync(exe)) return exe;
      }
    }
  }
  return undefined;
}

function judge(r) {
  const rows = [
    ['spectrum.maxAbsErrOverRms', r.spectrum.maxAbsErrOverRms, `< ${THRESHOLDS.spectrumMaxAbsErrOverRms}`, r.spectrum.maxAbsErrOverRms < THRESHOLDS.spectrumMaxAbsErrOverRms],
    ['temporal.D.p99', r.temporal.D.p99, `< ${THRESHOLDS.temporalDp99}`, r.temporal.D.p99 < THRESHOLDS.temporalDp99],
    ['temporal.C.p99', r.temporal.C.p99, `< ${THRESHOLDS.temporalCp99}`, r.temporal.C.p99 < THRESHOLDS.temporalCp99],
    ['temporal.P.p99', r.temporal.P.p99, `< ${THRESHOLDS.temporalPp99}`, r.temporal.P.p99 < THRESHOLDS.temporalPp99],
    ['verdict gpu == cpu', `${r.analysis.gpu.verdict} / ${r.analysis.cpu.verdict}`, 'equal', r.analysis.gpu.verdict === r.analysis.cpu.verdict],
    ['|alpha_gpu - alpha_cpu|', Math.abs(r.analysis.gpu.alpha - r.analysis.cpu.alpha), `< ${THRESHOLDS.alphaAbs}`, Math.abs(r.analysis.gpu.alpha - r.analysis.cpu.alpha) < THRESHOLDS.alphaAbs],
    ['vbar rel. difference', Math.abs(r.analysis.gpu.vbar_um_s / r.analysis.cpu.vbar_um_s - 1), `< ${THRESHOLDS.vbarRel}`, Math.abs(r.analysis.gpu.vbar_um_s / r.analysis.cpu.vbar_um_s - 1) < THRESHOLDS.vbarRel],
  ];
  return rows;
}

async function main() {
  execSync('npm run -s build:harness', { stdio: 'inherit' });
  const server = await createServer({ server: { port: 5173, strictPort: false, host: '127.0.0.1' }, logLevel: 'warn' });
  await server.listen();
  const port = server.config.server.port ?? 5173;
  const addr = server.httpServer?.address();
  const realPort = typeof addr === 'object' && addr ? addr.port : port;

  const flags = ['--enable-unsafe-webgpu'];
  if (swiftshader) flags.push('--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface');
  else flags.push('--enable-features=Vulkan');
  const launch = { args: flags, headless: true, channel: 'chromium' };
  const exe = resolveChrome();
  if (exe) launch.executablePath = exe;
  const browser = await chromium.launch(launch);
  let code = 1;
  try {
    const page = await browser.newPage();
    const result = new Promise((resolve) => {
      page.on('console', (msg) => {
        const t = msg.text();
        if (t.startsWith('RESULT ')) resolve(JSON.parse(t.slice(7)));
        else if (t.startsWith('step:')) console.log('  ' + t);
        else if (msg.type() === 'error') console.log('  [console.error] ' + t);
      });
      page.on('pageerror', (e) => resolve({ error: `pageerror: ${e.message}` }));
    });
    const url = `http://127.0.0.1:${realPort}/test/gpu/harness.html${src ? `?src=${src}` : ''}`;
    console.log(`Opening ${url} (${swiftshader ? 'SwiftShader' : 'hardware GPU'})`);
    await page.goto(url);
    const r = await Promise.race([result, new Promise((res) => setTimeout(() => res({ error: `timeout after ${timeoutMs} ms` }), timeoutMs))]);
    console.log('RESULT ' + JSON.stringify(r));
    if (r.error) {
      console.log(`\nFAIL: ${r.error}`);
    } else {
      console.log('');
      let ok = true;
      for (const [name, value, cond, pass] of judge(r)) {
        ok &&= pass;
        const v = typeof value === 'number' ? value.toExponential(3) : value;
        console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(28)} ${String(v).padEnd(26)} ${cond}`);
      }
      console.log(`\nframes ${r.framesUsed}, ingest ${r.msIngest} ms (${r.msPerFrame} ms/frame), temporal ${r.msTemporal} ms, adapter ${JSON.stringify(r.adapter)}`);
      console.log(ok ? '\nGPU parity: PASS' : '\nGPU parity: FAIL');
      code = ok ? 0 : 1;
    }
  } finally {
    await browser.close();
    await server.close();
  }
  process.exit(code);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
