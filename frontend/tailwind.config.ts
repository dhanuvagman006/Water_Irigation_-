import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0F6E56',
          light: '#1D9E75',
          dark: '#0A5240',
        },
        secondary: '#1D9E75',
        accent: '#378ADD',
        danger: '#E24B4A',
        warning: '#BA7517',
        success: '#3B6D11',
        surface: {
          DEFAULT: '#FFFFFF',
          dark: '#1A1F2E',
        },
        background: {
          DEFAULT: '#F8FAFB',
          dark: '#0F1117',
        },
        text: {
          primary: '#1A1A1A',
          muted: '#6B7280',
          dark: '#E5E7EB',
          'dark-muted': '#9CA3AF',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'count-up': 'countUp 0.8s ease-out',
      },
      keyframes: {
        countUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

export default config
