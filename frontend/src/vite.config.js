import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Optional: Configure server port if needed (default is 5173)
  // server: {
  //   port: 3000, // Example: Run dev server on port 3000
  // },
  // Optional: Configure build output directory if needed (default is 'dist')
  // build: {
  //   outDir: 'build'
  // }
})
