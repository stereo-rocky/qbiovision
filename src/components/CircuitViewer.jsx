// ================================================================
// src/components/CircuitViewer.jsx
// Quantum circuit display: diagram + QASM code with syntax highlighting
// ================================================================
import React, { useState } from 'react'
import { Code2, Image as ImageIcon, Copy, CheckCheck, Cpu, Hash, ChevronRight } from 'lucide-react'

// ---------------------------------------------------------------------------
// QASM syntax highlighter — simple regex-based token coloring
// ---------------------------------------------------------------------------
function highlightQASM(code) {
  if (!code) return ''
  // Tokenize and wrap with spans using dangerouslySetInnerHTML approach
  return code
    // Keywords
    .replace(/\b(OPENQASM|include|qreg|creg|gate|measure|barrier|if|reset)\b/g,
             '<span class="text-violet-400 font-semibold">$1</span>')
    // Gate names
    .replace(/\b(h|x|y|z|cx|cz|rx|ry|rz|ccx|swap|u1|u2|u3|id|s|t|sdg|tdg)\b/g,
             '<span class="text-sky-400">$1</span>')
    // Numbers / angles
    .replace(/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g,
             '<span class="text-amber-400">$1</span>')
    // pi constant
    .replace(/\bpi\b/g,
             '<span class="text-emerald-400">pi</span>')
    // Comments
    .replace(/(\/\/.*$)/gm,
             '<span class="text-slate-500 italic">$1</span>')
    // Strings
    .replace(/"([^"]*)"/g,
             '<span class="text-orange-400">"$1"</span>')
}

// ---------------------------------------------------------------------------
// ASCII-art placeholder when no circuit SVG is available
// ---------------------------------------------------------------------------
function ASCIIPlaceholder({ architecture = 'QCNN', nQubits = 4 }) {
  const lines = []
  for (let i = 0; i < nQubits; i++) {
    lines.push(`q[${i}]: ──H──●──RZ(θ${i})──M──`)
  }
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-6 font-mono text-sm text-slate-400">
      <p className="text-sky-400 mb-3 text-xs font-semibold uppercase tracking-widest">
        {architecture} — {nQubits} qubits (placeholder)
      </p>
      {lines.map((line, i) => (
        <div key={i} className="leading-7">{line}</div>
      ))}
      <div className="mt-3 text-xs text-slate-600">
        ── = wire &nbsp; ● = CNOT control &nbsp; M = measurement
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Stats bar — shows circuit metadata
// ---------------------------------------------------------------------------
function StatsBar({ nParams, architecture, description }) {
  return (
    <div className="flex flex-wrap gap-3 mb-4">
      {architecture && (
        <div className="flex items-center gap-1.5 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5">
          <Cpu className="w-3.5 h-3.5 text-violet-400" />
          <span className="text-xs text-slate-300 font-medium">{architecture}</span>
        </div>
      )}
      {nParams !== undefined && nParams !== null && (
        <div className="flex items-center gap-1.5 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5">
          <Hash className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-xs text-slate-300 font-medium">{nParams} parameters</span>
        </div>
      )}
      {description && (
        <div className="flex items-center gap-1.5 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 flex-1 min-w-0">
          <ChevronRight className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span className="text-xs text-slate-400 truncate">{description}</span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// CircuitViewer — Main export
// ---------------------------------------------------------------------------
export default function CircuitViewer({
  svgB64,
  qasm,
  isLoading = false,
  nParams,
  description,
  architecture,
}) {
  const [activeView, setActiveView] = useState('diagram') // 'diagram' | 'qasm'
  const [copied,     setCopied]     = useState(false)

  // Build SVG data URI
  const svgSrc = svgB64 ? `data:image/svg+xml;base64,${svgB64}` : null

  // Copy QASM to clipboard
  const handleCopy = async () => {
    if (!qasm) return
    try {
      await navigator.clipboard.writeText(qasm)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      console.error('Clipboard write failed')
    }
  }

  if (isLoading) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-6">
        <div className="h-4 w-40 bg-slate-700 rounded animate-pulse mb-4" />
        <div className="aspect-video bg-slate-700 rounded-lg animate-pulse" />
      </div>
    )
  }

  return (
    <div className="bg-slate-800 border border-sky-500/30 rounded-xl overflow-hidden shadow-quantum">
      {/* ------------------------------------------------------------------ */}
      {/* Stats bar                                                            */}
      {/* ------------------------------------------------------------------ */}
      <div className="px-5 pt-5">
        <StatsBar nParams={nParams} architecture={architecture} description={description} />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Tab switcher                                                         */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex border-b border-slate-700 px-5">
        {[
          { id: 'diagram', label: 'Circuit Diagram', icon: ImageIcon },
          { id: 'qasm',    label: 'QASM Code',       icon: Code2     },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveView(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2
              transition-all duration-200
              ${activeView === tab.id
                ? 'border-sky-400 text-sky-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'}
            `}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}

        {/* Copy button — only shown in QASM view */}
        {activeView === 'qasm' && (
          <button
            onClick={handleCopy}
            disabled={!qasm}
            className="ml-auto flex items-center gap-1.5 text-xs px-3 py-2 mb-1
                       text-slate-400 hover:text-sky-400 disabled:opacity-40
                       transition-colors"
          >
            {copied
              ? <><CheckCheck className="w-3.5 h-3.5 text-emerald-400" /> Copied!</>
              : <><Copy className="w-3.5 h-3.5" /> Copy QASM</>}
          </button>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Content area                                                         */}
      {/* ------------------------------------------------------------------ */}
      <div className="p-5">
        {activeView === 'diagram' ? (
          // Circuit diagram view
          svgSrc ? (
            <div className="bg-white rounded-lg overflow-hidden">
              <img
                src={svgSrc}
                alt="Quantum Circuit Diagram"
                className="w-full object-contain max-h-80"
              />
            </div>
          ) : (
            <ASCIIPlaceholder architecture={architecture} />
          )
        ) : (
          // QASM code view
          qasm ? (
            <pre
              className="qasm-code bg-slate-900 border border-slate-700 rounded-lg p-4
                         text-slate-300 overflow-auto max-h-72 text-xs leading-relaxed"
              dangerouslySetInnerHTML={{ __html: highlightQASM(qasm) }}
            />
          ) : (
            <div className="bg-slate-900 border border-dashed border-slate-700 rounded-lg p-8
                            text-center text-slate-600">
              <Code2 className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No QASM code available yet.</p>
              <p className="text-xs mt-1">Build a circuit to generate QASM output.</p>
            </div>
          )
        )}
      </div>
    </div>
  )
}
