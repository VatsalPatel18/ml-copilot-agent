// frontend/vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Optional: Configure server port if needed (default is 5173)
  server: {
   port: 5173, // Explicitly set the default port
   // Optional: Proxy API requests to your backend during development
   // proxy: {
   //   '/api': {
   //     target: 'http://localhost:8000', // Your FastAPI backend
   //     changeOrigin: true,
   //     // rewrite: (path) => path.replace(/^\/api/, ''), // Remove /api prefix if backend doesn't expect it
   //   }
   // }
  },
  // Optional: Configure build output directory if needed (default is 'dist')
  // build: {
  //   outDir: 'build'
  // }
})
