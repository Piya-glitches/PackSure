/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#F4F7F5",
          100: "#E4EBE6",
          600: "#2F6B4F",
          700: "#245A41",
          800: "#1B4633",
        },
        ink: "#1C2321",
        muted: "#5B6560",
        surface: "#FFFFFF",
        canvas: "#F7F8F6",
        line: "#E4E7E3",
        amber: "#C97A2B",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: { md: "8px", lg: "12px" },
    },
  },
  plugins: [],
};
