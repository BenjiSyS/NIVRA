import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Three HTML entries (Studio, Worker and App) share code in src/shared.
// The Go binaries embed internal/webui/dist, so the build writes there.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../internal/webui/dist',
    emptyOutDir: true,
    assetsDir: 'assets',
    sourcemap: false,
    modulePreload: { polyfill: false }, // no inline script injection (strict CSP)
    rollupOptions: {
      input: { studio: 'studio.html', worker: 'worker.html', app: 'app.html' },
    },
  },
  server: {
    // `npm run dev` proxies to a locally running controller and worker.
    proxy: {
      '/api': { target: 'http://127.0.0.1:8740' },
      '/local': { target: 'http://127.0.0.1:8741' },
    },
  },
})
