import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward backend API calls in dev. For /admin, we use a bypass so
      // that browser navigations (which request text/html) reach Vite and
      // React Router instead of being proxied to FastAPI as API calls.
      '/admin': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass(req) {
          const accept = req.headers && req.headers.accept
          if (accept && accept.includes('text/html')) {
            // Let Vite serve index.html so React Router handles the route.
            return '/index.html'
          }
        },
      },
      '/ask': 'http://localhost:8000',
      '/session': 'http://localhost:8000',
      '/feedback': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
