// ================================================================
// src/App.jsx — Root application component for Q-BioVision
// Dark-mode SPA with 4 quantum medical AI tabs
// ================================================================
import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Microscope, Cpu, Zap, BarChart2, Atom } from 'lucide-react'

// Tab components — each represents a major feature area
import DiagnosticStudio  from './components/DiagnosticStudio.jsx'
import CircuitBuilder    from './components/CircuitBuilder.jsx'
import NoiseSandbox      from './components/NoiseSandbox.jsx'
import BenchmarkReport   from './components/BenchmarkReport.jsx'

// ---------------------------------------------------------------------------
// Tab configuration
// ---------------------------------------------------------------------------
const TABS = [
  {
    id:          'diagnostic',
    label:       'Diagnostic Studio',
    icon:        Microscope,
    description: 'Upload & encode medical images into quantum feature maps',
    color:       'text-sky-400',
    activeColor: 'border-sky-400 text-sky-400',
  },
  {
    id:          'circuit',
    label:       'Circuit Builder',
    icon:        Cpu,
    description: 'Design & train quantum neural network architectures',
    color:       'text-violet-400',
    activeColor: 'border-violet-400 text-violet-400',
  },
  {
    id:          'noise',
    label:       'Noise Sandbox',
    icon:        Zap,
    description: 'Simulate quantum noise & apply mitigation strategies',
    color:       'text-emerald-400',
    activeColor: 'border-emerald-400 text-emerald-400',
  },
  {
    id:          'benchmark',
    label:       'Benchmark',
    icon:        BarChart2,
    description: 'Compare classical vs quantum model performance',
    color:       'text-amber-400',
    activeColor: 'border-amber-400 text-amber-400',
  },
]

// ---------------------------------------------------------------------------
// Tab page animation variants (framer-motion)
// ---------------------------------------------------------------------------
const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
  exit:    { opacity: 0, y: -8, transition: { duration: 0.2, ease: 'easeIn' } },
}

// ---------------------------------------------------------------------------
// App Component
// ---------------------------------------------------------------------------
export default function App() {
  // Active tab identifier
  const [activeTab, setActiveTab] = useState('diagnostic')

  // Global shared state — data flows between tabs (e.g. preprocessed image
  // used in circuit training, trained model used in benchmarking, etc.)
  const [globalState, setGlobalState] = useState({
    // DiagnosticStudio outputs
    preprocessedData:   null,  // { original_b64, feature_map_b64, patch_grid_b64, quantum_features }
    selectedDataset:    null,  // 'breakhis' | 'ham10000' | 'chestxr'
    nQubits:            8,     // shared qubit count

    // CircuitBuilder outputs
    circuitData:        null,  // { svg_b64, qasm, n_params, description }
    trainingResults:    null,  // { accuracy, loss_curve, accuracy_curve }
    modelConfig:        null,  // { architecture, dataset, n_qubits, depth, entangler }

    // BenchmarkReport outputs
    benchmarkResults:   null,  // full benchmark data
    reportContent:      null,  // { markdown_content, pdf_b64 }
  })

  // Convenience updater — merges partial state
  const updateGlobalState = useCallback((updates) => {
    setGlobalState(prev => ({ ...prev, ...updates }))
  }, [])

  // Map tab id to its component
  const renderActiveTab = () => {
    const props = { globalState, updateGlobalState }
    switch (activeTab) {
      case 'diagnostic':  return <DiagnosticStudio  {...props} />
      case 'circuit':     return <CircuitBuilder    {...props} />
      case 'noise':       return <NoiseSandbox      {...props} />
      case 'benchmark':   return <BenchmarkReport   {...props} />
      default:            return <DiagnosticStudio  {...props} />
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col">
      {/* ------------------------------------------------------------------ */}
      {/* HEADER                                                               */}
      {/* ------------------------------------------------------------------ */}
      <header className="border-b border-slate-700 bg-slate-900/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          {/* Logo + Title */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-violet-600 flex items-center justify-center shadow-quantum">
                <Atom className="w-6 h-6 text-white" />
              </div>
              {/* Pulsing ring */}
              <div className="absolute inset-0 rounded-xl border border-sky-400/50 animate-ping opacity-30" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">
                Q-BioVision
              </h1>
              <p className="text-xs text-slate-400 leading-none mt-0.5">
                Quantum Medical AI Platform
              </p>
            </div>
          </div>

          {/* Status badge */}
          <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 rounded-full px-3 py-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Qiskit Fall Fest 2026 — Healthcare Track
          </div>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* TAB NAVIGATION BAR                                                */}
        {/* ---------------------------------------------------------------- */}
        <nav className="max-w-7xl mx-auto px-6 flex gap-1 pb-0">
          {TABS.map(tab => {
            const Icon    = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  group flex items-center gap-2 px-4 py-3 text-sm font-medium
                  border-b-2 transition-all duration-200 whitespace-nowrap
                  ${isActive
                    ? tab.activeColor + ' bg-slate-800/50'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-500'}
                `}
              >
                <Icon className={`w-4 h-4 ${isActive ? '' : 'group-hover:scale-110 transition-transform'}`} />
                {tab.label}
              </button>
            )
          })}
        </nav>
      </header>

      {/* ------------------------------------------------------------------ */}
      {/* MAIN CONTENT AREA                                                   */}
      {/* ------------------------------------------------------------------ */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            {renderActiveTab()}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* ------------------------------------------------------------------ */}
      {/* FOOTER                                                               */}
      {/* ------------------------------------------------------------------ */}
      <footer className="border-t border-slate-800 py-4 px-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-xs text-slate-500">
          <span>
            Q-BioVision v1.0 &nbsp;|&nbsp; Qiskit Fall Fest 2026 &nbsp;|&nbsp; Healthcare Track
          </span>
          <span className="flex items-center gap-1.5">
            <Atom className="w-3 h-3 text-sky-500" />
            Powered by Qiskit &amp; React
          </span>
        </div>
      </footer>
    </div>
  )
}
