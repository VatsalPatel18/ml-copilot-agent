/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html", // Include the main HTML file
    "./src/**/*.{js,ts,jsx,tsx}", // Scan all JS/TS/JSX/TSX files in src/
  ],
  theme: {
    extend: {
       fontFamily: {
         // Ensure 'Inter' is available (e.g., via index.html or index.css)
         sans: ['Inter', 'sans-serif'],
       },
       // Add custom theme extensions here if needed
    },
  },
  plugins: [],
}
