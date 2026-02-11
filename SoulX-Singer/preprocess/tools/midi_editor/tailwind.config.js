/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Space Grotesk"', '"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        ink: {
          50: '#f4f7fb',
          100: '#dfe7f5',
          200: '#beceec',
          300: '#95addf',
          400: '#6a87ce',
          500: '#4b64bc',
          600: '#3b4ea7',
          700: '#32418a',
          800: '#2c376f',
          900: '#262f5c',
        },
        ember: '#ff7043',
        mint: '#48e4c2',
      },
      boxShadow: {
        panel: '0 14px 35px rgba(0, 0, 0, 0.25)',
      },
    },
  },
  plugins: [],
}

