/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  important: '#root',
  theme: {
    extend: {
      colors: {
        border: "rgba(34, 211, 238, 0.1)",
        input: "rgba(34, 211, 238, 0.05)",
        ring: "#22d3ee",
        background: "#02040a",
        foreground: "#f8fafc",
        primary: {
          DEFAULT: "#22d3ee",
          foreground: "#000000",
        },
        secondary: {
          DEFAULT: "#0f172a",
          foreground: "#22d3ee",
        },
        destructive: {
          DEFAULT: "#ef4444",
          foreground: "#f8fafc",
        },
        muted: {
          DEFAULT: "#1e293b",
          foreground: "#64748b",
        },
        accent: {
          DEFAULT: "rgba(34, 211, 238, 0.1)",
          foreground: "#22d3ee",
        },
        cyan: {
          400: "#22d3ee",
          500: "#06b6d4",
          glow: "rgba(34, 211, 238, 0.4)",
        }
      },
    },
  },
  plugins: [],
}
