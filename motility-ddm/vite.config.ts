import { defineConfig } from 'vite';

// Relative base so the build works from any GitHub Pages sub-path.
// No COOP/COEP headers: GitHub Pages cannot set them and nothing needs them.
// Frame timing comes from VideoFrame/rVFC captureTime, not from a
// cross-origin-isolated high-resolution performance.now().
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5173,
    host: true,
  },
});
