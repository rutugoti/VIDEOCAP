/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bgMain: "#070B14",
        cardMain: "#101827",
        accentPurple: {
          light: "#c084fc",
          DEFAULT: "#a855f7",
          dark: "#7e22ce"
        },
        accentBlue: {
          light: "#60a5fa",
          DEFAULT: "#3b82f6",
          dark: "#1d4ed8"
        },
        accentCyan: {
          light: "#22d3ee",
          DEFAULT: "#06b6d4",
          dark: "#0891b2"
        }
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },
      animation: {
        'glow-pulse': 'glow-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'border-spin': 'border-spin 8s linear infinite',
      },
      keyframes: {
        'glow-pulse': {
          '0%, 100%': { opacity: 0.8, filter: 'drop-shadow(0 0 8px rgba(168, 85, 247, 0.4))' },
          '50%': { opacity: 1, filter: 'drop-shadow(0 0 20px rgba(168, 85, 247, 0.8))' },
        },
        'border-spin': {
          '100%': { transform: 'rotate(360deg)' },
        }
      }
    },
  },
  plugins: [],
}
