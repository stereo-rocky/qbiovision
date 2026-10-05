/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'quantum-blue':   '#0ea5e9',
        'quantum-purple': '#8b5cf6',
        'quantum-green':  '#10b981',
        'dark-bg':        '#0f172a',
        'dark-card':      '#1e293b',
        'dark-border':    '#334155',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        'quantum':        '0 0 20px rgba(14, 165, 233, 0.3)',
        'quantum-lg':     '0 0 40px rgba(14, 165, 233, 0.4)',
        'quantum-purple': '0 0 20px rgba(139, 92, 246, 0.3)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow':  'spin 3s linear infinite',
      },
    },
  },
  plugins: [],
}
