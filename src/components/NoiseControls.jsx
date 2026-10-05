// ================================================================
// src/components/NoiseControls.jsx
// Noise configuration panel — T1/T2 relaxation, depolarizing, mitigation
// ================================================================
import React from 'react'
import { Zap, Shield, Thermometer, Radio, HelpCircle } from 'lucide-react'

// ---------------------------------------------------------------------------
// Toggle switch
// ---------------------------------------------------------------------------
function Toggle({ checked, onChange, label, color = 'sky' }) {
  const colorMap = {
    sky:     'bg-sky-500',
    violet:  'bg-violet-500',
    emerald: 'bg-emerald-500',
  }
  return (
    <label className="flex items-center justify-between cursor-pointer group">
      <span className="text-sm text-slate-300 group-hover:text-slate-100 transition-colors">
        {label}
      </span>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`
          relative inline-flex h-5 w-9 shrink-0 rounded-full transition-colors duration-200
          focus:outline-none focus:ring-2 focus:ring-sky-500/50
          ${checked ? colorMap[color] : 'bg-slate-700'}
        `}
      >
        <span
          className={`
            inline-block h-4 w-4 rounded-full bg-white shadow transform transition-transform duration-200 mt-0.5
            ${checked ? 'translate-x-4.5' : 'translate-x-0.5'}
          `}
          style={{ transform: checked ? 'translateX(18px)' : 'translateX(2px)' }}
        />
      </button>
    </label>
  )
}

// ---------------------------------------------------------------------------
// Labeled slider
// ---------------------------------------------------------------------------
function Slider({
  label, value, min, max, step = 1,
  onChange, unit = '', disabled = false,
  color = 'sky', formatValue,
}) {
  const displayValue = formatValue ? formatValue(value) : `${value}${unit}`
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className={`${disabled ? 'opacity-40 pointer-events-none' : ''}`}>
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`text-xs font-mono font-semibold ${
          color === 'violet' ? 'text-violet-400' : 'text-sky-400'
        }`}>
          {displayValue}
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className={`w-full ${color === 'violet' ? 'purple-thumb' : ''}`}
          style={{
            background: `linear-gradient(to right, ${
              color === 'violet' ? '#8b5cf6' : '#0ea5e9'
            } ${pct}%, #334155 ${pct}%)`,
          }}
        />
      </div>
      <div className="flex justify-between mt-0.5">
        <span className="text-[10px] text-slate-600">{min}{unit}</span>
        <span className="text-[10px] text-slate-600">{max}{unit}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section card wrapper
// ---------------------------------------------------------------------------
function Section({ title, icon: Icon, accentColor, children }) {
  const colors = {
    sky:    { border: 'border-sky-500/30',    icon: 'text-sky-400',    bg: 'bg-sky-500/10'    },
    violet: { border: 'border-violet-500/30', icon: 'text-violet-400', bg: 'bg-violet-500/10' },
  }
  const c = colors[accentColor] || colors.sky

  return (
    <div className={`bg-slate-800 border ${c.border} rounded-xl p-5`}>
      <div className="flex items-center gap-2 mb-5">
        <div className={`w-7 h-7 rounded-lg ${c.bg} flex items-center justify-center`}>
          <Icon className={`w-4 h-4 ${c.icon}`} />
        </div>
        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
      </div>
      <div className="space-y-5">
        {children}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tooltip helper
// ---------------------------------------------------------------------------
function TooltipBadge({ text }) {
  const [show, setShow] = React.useState(false)
  return (
    <span className="relative">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="text-slate-600 hover:text-slate-400 transition-colors ml-1"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>
      {show && (
        <span className="absolute z-50 bottom-5 left-1/2 -translate-x-1/2 w-48 text-[11px]
                         text-slate-200 bg-slate-700 border border-slate-600 rounded-lg p-2 shadow-xl leading-relaxed">
          {text}
        </span>
      )}
    </span>
  )
}

// ---------------------------------------------------------------------------
// NoiseControls — Main export
// ---------------------------------------------------------------------------
export default function NoiseControls({ config, onChange }) {
  // Safe defaults if config is missing fields
  const cfg = {
    enableT1T2:        false,
    t1:                100,
    t2:                80,
    enableDepolarizing: false,
    depolarizingRate:  0.01,
    enableZNE:         false,
    enableTREX:        false,
    ...config,
  }

  // Helper — update a single field
  const update = (key) => (val) => onChange({ ...cfg, [key]: val })

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------------------------ */}
      {/* Section 1 — Noise Sources                                           */}
      {/* ------------------------------------------------------------------ */}
      <Section title="Noise Sources" icon={Zap} accentColor="sky">
        {/* T1/T2 Thermal Relaxation */}
        <div className="space-y-3">
          <Toggle
            checked={cfg.enableT1T2}
            onChange={update('enableT1T2')}
            label={
              <span className="flex items-center gap-1">
                <Thermometer className="w-3.5 h-3.5 text-orange-400" />
                T1/T2 Thermal Relaxation
                <TooltipBadge text="T1 = energy relaxation time, T2 = dephasing time. Shorter = noisier qubit." />
              </span>
            }
            color="sky"
          />
          <Slider
            label="T1 — Energy Relaxation"
            value={cfg.t1}
            min={1} max={200} step={1}
            unit=" µs"
            onChange={update('t1')}
            disabled={!cfg.enableT1T2}
            color="sky"
          />
          <Slider
            label="T2 — Dephasing Time"
            value={cfg.t2}
            min={1} max={200} step={1}
            unit=" µs"
            onChange={update('t2')}
            disabled={!cfg.enableT1T2}
            color="sky"
          />
        </div>

        <hr className="border-slate-700" />

        {/* Depolarizing Noise */}
        <div className="space-y-3">
          <Toggle
            checked={cfg.enableDepolarizing}
            onChange={update('enableDepolarizing')}
            label={
              <span className="flex items-center gap-1">
                <Radio className="w-3.5 h-3.5 text-red-400" />
                Depolarizing Noise
                <TooltipBadge text="Random Pauli errors (X, Y, Z) applied to each gate with probability p." />
              </span>
            }
            color="sky"
          />
          <Slider
            label="Depolarizing Rate (p)"
            value={cfg.depolarizingRate}
            min={0.001} max={0.1} step={0.001}
            onChange={update('depolarizingRate')}
            disabled={!cfg.enableDepolarizing}
            color="sky"
            formatValue={v => v.toFixed(3)}
          />
        </div>
      </Section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 2 — Error Mitigation                                        */}
      {/* ------------------------------------------------------------------ */}
      <Section title="Error Mitigation" icon={Shield} accentColor="violet">
        <Toggle
          checked={cfg.enableZNE}
          onChange={update('enableZNE')}
          label={
            <span className="flex items-center gap-1 text-violet-300">
              ZNE — Zero-Noise Extrapolation
              <TooltipBadge text="Runs circuit at multiple noise levels and extrapolates to the zero-noise limit." />
            </span>
          }
          color="violet"
        />

        <Toggle
          checked={cfg.enableTREX}
          onChange={update('enableTREX')}
          label={
            <span className="flex items-center gap-1 text-violet-300">
              TREX — Twirled Readout Error Extinction
              <TooltipBadge text="Uses Pauli twirling to convert coherent errors into stochastic noise, then applies measurement error mitigation." />
            </span>
          }
          color="violet"
        />

        {/* Visual hint when mitigation is enabled */}
        {(cfg.enableZNE || cfg.enableTREX) && (
          <div className="bg-violet-500/10 border border-violet-500/30 rounded-lg p-3 text-xs text-violet-300">
            <Shield className="w-3.5 h-3.5 inline mr-1.5 mb-0.5" />
            {cfg.enableZNE && cfg.enableTREX
              ? 'ZNE + TREX combined for maximum noise suppression'
              : cfg.enableZNE
              ? 'ZNE will extrapolate expectation values to zero noise'
              : 'TREX will mitigate readout measurement errors'}
          </div>
        )}
      </Section>
    </div>
  )
}
