import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';

// Builds straight into the server's resource bundle; the output is committed so the server (and a cloud bootstrap)
// never needs Node. `npm run dev` proxies the API to a local `aistudio-cli serve`.
export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: '../../Sources/AIStudioServer/Resources/dashboard',
    emptyOutDir: true,
    target: 'es2022',
    assetsInlineLimit: 0,
    reportCompressedSize: false,
  },
  server: {
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8188', ws: true },
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
});
