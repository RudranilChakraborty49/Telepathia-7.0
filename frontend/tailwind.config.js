/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg:       "#05070d",
          surface:  "#0d1117",
          surface2: "#161b22",
          border:   "#1f2937",
          muted:    "#6b7280",
          text:     "#e5e7eb",
        },
        signal: {
          buy:      "#10b981",
          sell:     "#ef4444",
          hold:     "#6b7280",
        },
        neon: {
          cyan:    "#00f0ff",
          purple:  "#b400ff",
          pink:    "#ff006e",
          green:   "#39ff14",
          blue:    "#00d4ff",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "monospace"],
        display: ["Orbitron", "Inter", "sans-serif"],
      },
      boxShadow: {
        'neon-cyan':   "0 0 20px rgba(0, 240, 255, 0.5), 0 0 40px rgba(0, 240, 255, 0.2)",
        'neon-purple': "0 0 20px rgba(180, 0, 255, 0.5), 0 0 40px rgba(180, 0, 255, 0.2)",
        'neon-green':  "0 0 20px rgba(57, 255, 20, 0.5), 0 0 40px rgba(57, 255, 20, 0.2)",
        'neon-pink':   "0 0 20px rgba(255, 0, 110, 0.5), 0 0 40px rgba(255, 0, 110, 0.2)",
      },
      animation: {
        'gradient':  "gradient 8s linear infinite",
        'pulse-slow':"pulse 3s ease-in-out infinite",
        'float':     "float 6s ease-in-out infinite",
        'glow':      "glow 2s ease-in-out infinite alternate",
        'scan':      "scan 4s linear infinite",
        'fade-up':   "fadeUp 0.8s ease-out",
        'fade-in':   "fadeIn 1s ease-out",
      },
      keyframes: {
        gradient: {
          '0%, 100%': { 'background-position': '0% 50%' },
          '50%':       { 'background-position': '100% 50%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%':       { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%':   { 'text-shadow': '0 0 10px rgba(0,240,255,0.5), 0 0 20px rgba(0,240,255,0.3)' },
          '100%': { 'text-shadow': '0 0 20px rgba(180,0,255,0.7), 0 0 40px rgba(180,0,255,0.4)' },
        },
        scan: {
          '0%':   { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        fadeUp: {
          '0%':   { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      backgroundImage: {
        'grid-pattern': "linear-gradient(rgba(0,240,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0,240,255,0.05) 1px, transparent 1px)",
        'radial-glow':  "radial-gradient(circle at center, rgba(180,0,255,0.15) 0%, transparent 70%)",
      },
    },
  },
  plugins: [],
}