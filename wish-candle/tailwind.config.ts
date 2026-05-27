import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        stone: {
          950: "#0a0905",
          900: "#14110a",
          800: "#1e1a10",
          700: "#2a2515",
          600: "#3a341f",
        },
        gold: {
          50: "#fffbeb",
          100: "#fef3c7",
          200: "#fde68a",
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
          700: "#b45309",
          800: "#92400e",
          900: "#78350f",
        },
        bronze: {
          400: "#cd8b4a",
          500: "#b5722a",
          600: "#9a5e1a",
        },
        candle: {
          flame: "#ff9d00",
          glow: "#ff6a00",
          wax: "#f5e6c8",
        },
      },
      fontFamily: {
        cinzel: ["'Cinzel Decorative'", "serif"],
        uncial: ["'UnifrakturMaguntia'", "serif"],
        medieval: ["'MedievalSharp'", "serif"],
      },
      animation: {
        "flame-flicker": "flicker 1.5s ease-in-out infinite alternate",
        "glow-pulse": "glowPulse 2s ease-in-out infinite",
        "float-up": "floatUp 3s ease-in forwards",
        "sparkle": "sparkle 1.5s ease-in-out infinite",
        "drift": "drift 4s ease-in-out infinite",
        "ember-rise": "emberRise 2s ease-out forwards",
        "heart-rise": "heartRise 2.5s ease-out forwards",
        "smoke-rise": "smokeRise 3s ease-out forwards",
        "candle-appear": "candleAppear 0.8s ease-out forwards",
        "text-reveal": "textReveal 1s ease-out forwards",
      },
      keyframes: {
        flicker: {
          "0%, 100%": { transform: "scaleX(1) scaleY(1)", opacity: "1" },
          "25%": { transform: "scaleX(0.95) scaleY(1.05)", opacity: "0.9" },
          "50%": { transform: "scaleX(1.05) scaleY(0.95)", opacity: "0.95" },
          "75%": { transform: "scaleX(0.97) scaleY(1.03)", opacity: "0.88" },
        },
        glowPulse: {
          "0%, 100%": { boxShadow: "0 0 15px 5px rgba(251,191,36,0.3)" },
          "50%": { boxShadow: "0 0 30px 10px rgba(251,191,36,0.5)" },
        },
        floatUp: {
          "0%": { transform: "translateY(0) scale(1)", opacity: "1" },
          "100%": { transform: "translateY(-200px) scale(0)", opacity: "0" },
        },
        sparkle: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.4", transform: "scale(0.6)" },
        },
        drift: {
          "0%, 100%": { transform: "translateX(0)" },
          "50%": { transform: "translateX(8px)" },
        },
        emberRise: {
          "0%": { transform: "translateY(0) translateX(0)", opacity: "1" },
          "100%": { transform: "translateY(-120px) translateX(20px)", opacity: "0" },
        },
        heartRise: {
          "0%": { transform: "translateY(0) scale(0)", opacity: "0" },
          "20%": { transform: "translateY(-20px) scale(1)", opacity: "1" },
          "100%": { transform: "translateY(-180px) scale(0.3)", opacity: "0" },
        },
        smokeRise: {
          "0%": { transform: "translateY(0) scale(0.5)", opacity: "0.5" },
          "100%": { transform: "translateY(-150px) scale(2)", opacity: "0" },
        },
        candleAppear: {
          "0%": { transform: "scale(0.5)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
        textReveal: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
