// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html", // Include the main HTML file in public/
      "./src/**/*.{js,ts,jsx,tsx}", // Scan all JS/TS/JSX/TSX files in src/
    ],
    theme: {
      extend: {
          fontFamily: {
            // Ensure 'Inter' is available (e.g., via index.html or index.css)
            sans: ['Inter', 'sans-serif'],
          },
          colors: { // Example custom colors matching Shadcn defaults (optional)
              border: 'hsl(214.3 31.8% 91.4%)',
              input: 'hsl(214.3 31.8% 91.4%)',
              ring: 'hsl(215 20.2% 65.1%)',
              background: 'hsl(0 0% 100%)',
              foreground: 'hsl(222.2 84% 4.9%)',
              primary: {
                DEFAULT: 'hsl(222.2 47.4% 11.2%)',
                foreground: 'hsl(210 40% 98%)',
              },
              secondary: {
                DEFAULT: 'hsl(210 40% 96.1%)',
                foreground: 'hsl(222.2 47.4% 11.2%)',
              },
              destructive: {
                DEFAULT: 'hsl(0 84.2% 60.2%)',
                foreground: 'hsl(210 40% 98%)',
              },
              muted: {
                DEFAULT: 'hsl(210 40% 96.1%)',
                foreground: 'hsl(215.4 16.3% 46.9%)',
              },
              accent: {
                DEFAULT: 'hsl(210 40% 96.1%)',
                foreground: 'hsl(222.2 47.4% 11.2%)',
              },
              popover: {
                DEFAULT: 'hsl(0 0% 100%)',
                foreground: 'hsl(222.2 84% 4.9%)',
              },
              card: {
                DEFAULT: 'hsl(0 0% 100%)',
                foreground: 'hsl(222.2 84% 4.9%)',
              },
          },
          borderRadius: { // Example matching Shadcn defaults (optional)
              lg: `0.5rem`,
              md: `calc(0.5rem - 2px)`,
              sm: `calc(0.5rem - 4px)`,
          },
          // Add other theme extensions if needed
      },
    },
    plugins: [
        // require("tailwindcss-animate"), // If using animations
    ],
  }
  