import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies all backend calls to whichever URL VITE_API_TARGET
// resolves to. Vite picks env files based on the command:
//   npm run dev    → loads .env.development (+ .env.development.local override)
//   npm run build  → loads .env.production  (+ .env.production.local override)
// Per-developer overrides go in .env.development.local / .env.production.local
// (gitignored). The hardcoded fallback below is only a safety net for new
// clones that forgot to copy the .env files.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // Fallback only matters for fresh clones without a .env file — the deployed
  // ALB is the safer default so the app boots and serves real data.
  const API_TARGET =
    env.VITE_API_TARGET ||
    'http://1audit-dev-ec2-alb-253123334.eu-west-1.elb.amazonaws.com:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        // For /admin, we use a bypass so that browser navigations (which
        // request text/html) reach Vite and React Router instead of being
        // proxied to FastAPI as API calls.
        '/admin': {
          target: API_TARGET,
          changeOrigin: true,
          bypass(req) {
            const accept = req.headers && req.headers.accept
            if (accept && accept.includes('text/html')) {
              return '/index.html'
            }
          },
        },
        '/ask': {
          target: API_TARGET,
          changeOrigin: true,
          // Streaming responses must not be buffered by the proxy.
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              proxyRes.headers['x-accel-buffering'] = 'no'
              proxyRes.headers['cache-control'] = 'no-cache'
            })
          },
        },
        '/session': { target: API_TARGET, changeOrigin: true },
        '/feedback': { target: API_TARGET, changeOrigin: true },
        '/health': { target: API_TARGET, changeOrigin: true },
      },
    },
  }
})
