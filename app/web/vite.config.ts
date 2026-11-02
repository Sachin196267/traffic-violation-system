import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev: npm run dev (proxies the API to the FastAPI server on :8600); prod: npm run build -> dist/ served by FastAPI
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8600' } },
})
