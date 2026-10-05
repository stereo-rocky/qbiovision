// ================================================================
// src/components/CircuitBuilder.jsx
// Tab 2 — Architecture selection, circuit preview, model training
// ================================================================
import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  Cpu, Play, Loader2, CheckCircle2, AlertCircle,
  Network, Sliders, GitBranch, Clock, Hash,
  TrendingUp, Award, ChevronDown,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { buildCircuit, trainModel } from '../api/client.js'
import CircuitViewer from './CircuitViewer.jsx'

// ---------------------------------------------------------------------------
// Architecture definitions
// ---------------------------------------------------------------------------
const ARCHITECTURES = [
  {
    id:          'qcnn',
    name:        'QCNN',
    fullName:    'Quantum Convolutional Neural Network',
    icon:        Network,
    description: 'Hierarchical qubit reduction via alternating conv/pool layers',
    formula:     'U(θ) = ∏ᵢ U_conv(θᵢ) · U_pool',
    params:      (q, d) => q * d * 2,
    color:       'sky',
  },
  {
    id:          'qsvc',
    name:        'QSVC',
    fullName:    'Quantum Support Vector Classifier',
    icon:        GitBranch,
    description: 'Quantum kernel estimation via feature map inner products',
    formula:     "K(x,x') = |⟨φ(x)|φ(x')⟩|²",
    params:      (q, d) => q * d,
    color:       'violet',
  },
  {
    id:          'vqc',
    name:        'VQC',
    fullName:    'Variational Quantum Classifier',
    icon:        Sliders,
    description: 'Parameterized circuit optimized via gradient descent',
    formula:     'C(θ) = ⟨ψ(θ)|H_obs|ψ(θ)⟩',
    params:      (q, d) => q * d * 3,
    color:       'emerald',
  },
]

const ENTANGLERS  = ['CX', 'CZ', 'RXX']
const DATASETS_OPTS = ['BreakHis', 'HAM10000', 'ChestXR']

const ARCH_COLORS = {
  sky:     { border: 'border-sky-500/60',    bg: 'bg-sky-500/10',     badge: 'text-sky-400',     active: 'shadow-quantum' },
  violet:  { border: 'border-violet-500/60', bg: 'bg-violet-500/10',  badge: 'text-violet-400',  active: 'shadow-quantum-purple' },
  emerald: { border: 'border-emerald-500/60',bg: 'bg-emerald-500/10', badge: 'text-emerald-400', active: '' },
}

// ---------------------------------------------------------------------------
// Architecture card
// ---------------------------------------------------------------------------
function ArchCard({ arch, selected, onSelect }) {
  const Icon   = arch.icon
  const colors = ARCH_COLORS[arch.color]
  return (
    <button
      onClick={() => onSelect(arch.id)}
      className={`
        flex-1 p-4 rounded-xl border text-left transition-all duration-200
        ${selected
          ? `${colors.border} ${colors.bg} ${colors.active}`
          : 'border-slate-700 hover:border-slate-500 bg-slate-800/50 hover:bg-slate-800'}
      `}
    >
      <div className={`flex items-center gap-2 mb-2 ${selected ? colors.badge : 'text-slate-400'}`}>
        <Icon className="w-4 h-4" />
        <span className="text-base font-bold text-slate-200">{arch.name}</span>
        {selected && <CheckCircle2 className={`w-3.5 h-3.5 ml-auto ${colors.badge}`} />}
      </div>
      <p className="text-xs text-slate-400 mb-2 leading-relaxed">{arch.description}</p>
      <div className="font-mono text-[11px] bg-slate-900/60 rounded px-2 py-1 text-slate-400 mt-2 break-all">
        {arch.formula}
      </div>
    </button>
  )
}

// ---------------------------------------------------------------------------
// Simple labeled select dropdown
// ---------------------------------------------------------------------------
function Select({ label, value, options, onChange }) {
  return (
    <div>
      <label className="block text-xs text-slate-400 mb-1.5">{label}</label>
      <div className="relative">
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2
                     text-sm text-slate-200 appearance-none focus:outline-none
                     focus:border-sky-500 cursor-pointer pr-8"
        >
          {options.map(o => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
        <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Training result badge
// ---------------------------------------------------------------------------
function ResultBadge({ label, value, color = 'sky', unit = '' }) {
  const colorMap = {
    sky:     'bg-sky-500/10 border-sky-500/30 text-sky-300',
    emerald: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300',
    violet:  'bg-violet-500/10 border-violet-500/30 text-violet-300',
    amber:   'bg-amber-500/10 border-amber-500/30 text-amber-300',
  }
  return (
    <div className={`border rounded-lg p-3 ${colorMap[color]}`}>
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className="text-xl font-bold">{value}{unit}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// CircuitBuilder — Main export
// ---------------------------------------------------------------------------
export default function CircuitBuilder({ globalState, updateGlobalState }) {
  // Architecture params
  const [archId,      setArchId]    = useState('qcnn')
  const [nQubits,     setNQubits]   = useState(globalState.nQubits || 8)
  const [depth,       setDepth]     = useState(2)
  const [entangler,   setEntangler] = useState('CX')
  const [dataset,     setDataset]   = useState('BreakHis')

  // Circuit state
  const [circuitData,    setCircuitData]   = useState(globalState.circuitData || null)
  const [circuitLoading, setCircuitLoading] = useState(false)
  const [circuitError,   setCircuitError]  = useState(null)

  // Training state
  const [isTraining,    setIsTraining]    = useState(false)
  const [trainError,    setTrainError]    = useState(null)
  const [trainingData,  setTrainingData]  = useState(null) // loss/accuracy curves
  const [trainResults,  setTrainResults]  = useState(globalState.trainingResults || null)

  // Debounce ref
  const debounceTimer = useRef(null)

  const arch     = ARCHITECTURES.find(a => a.id === archId)
  const nParams  = arch ? arch.params(nQubits, depth) : 0

  // ---------------------------------------------------------------------------
  // Auto-build circuit when params change (debounced 500ms)
  // ---------------------------------------------------------------------------
  const triggerBuildCircuit = useCallback(async (config) => {
    setCircuitLoading(true)
    setCircuitError(null)
    try {
      const result = await buildCircuit(config)
      setCircuitData(result)
      updateGlobalState({ circuitData: result, modelConfig: config })
    } catch (err) {
      setCircuitError(err.message)
      setCircuitData(null)
    } finally {
      setCircuitLoading(false)
    }
  }, [updateGlobalState])

  useEffect(() => {
    clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => {
      triggerBuildCircuit({ architecture: archId, n_qubits: nQubits, depth, entangler })
    }, 500)
    return () => clearTimeout(debounceTimer.current)
  }, [archId, nQubits, depth, entangler, triggerBuildCircuit])

  // ---------------------------------------------------------------------------
  // Train model
  // ---------------------------------------------------------------------------
  const handleTrain = async () => {
    setIsTraining(true)
    setTrainError(null)
    setTrainingData(null)
    setTrainResults(null)

    try {
      const config = { architecture: archId, dataset, n_qubits: nQubits, depth, entangler }
      const result = await trainModel(config)

      // Build chart data from returned curves
      if (result.loss_curve && result.accuracy_curve) {
        const chartPoints = result.loss_curve.map((loss, i) => ({
          epoch:    i + 1,
          loss:     +loss.toFixed(4),
          accuracy: +((result.accuracy_curve[i] || 0) * 100).toFixed(2),
        }))
        setTrainingData(chartPoints)
      }

      setTrainResults(result)
      updateGlobalState({ trainingResults: result, modelConfig: config })
    } catch (err) {
      setTrainError(err.message || 'Training failed. Check backend connection.')
    } finally {
      setIsTraining(false)
    }
  }

  // Simulated live chart data for demo when API returns partial data
  const chartData = trainingData || []

  return (
    <div className="space-y-6 fade-in">
      {/* ------------------------------------------------------------------ */}
      {/* Header                                                               */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center">
          <Cpu className="w-5 h-5 text-violet-400" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-white">Quantum Circuit Builder</h2>
          <p className="text-sm text-slate-400">Design and train quantum ML architectures on medical image data.</p>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Section A — Architecture selector                                    */}
      {/* ------------------------------------------------------------------ */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
          <Network className="w-4 h-4 text-violet-400" />
          Select Architecture
        </h3>
        <div className="flex gap-4">
          {ARCHITECTURES.map(a => (
            <ArchCard
              key={a.id}
              arch={a}
              selected={archId === a.id}
              onSelect={setArchId}
            />
          ))}
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Section B — Parameters                                               */}
      {/* ------------------------------------------------------------------ */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-5 flex items-center gap-2">
          <Sliders className="w-4 h-4 text-sky-400" />
          Circuit Parameters
          <span className="ml-auto text-xs text-slate-500">
            ~{nParams} trainable params
          </span>
        </h3>

        <div className="grid grid-cols-3 gap-6">
          {/* Qubit count */}
          <div>
            <div className="flex justify-between mb-1.5">
              <span className="text-xs text-slate-400">Qubit Count</span>
              <span className="text-xs font-mono font-bold text-sky-400">{nQubits}</span>
            </div>
            <input
              type="range" min={4} max={16} step={2}
              value={nQubits}
              onChange={e => setNQubits(parseInt(e.target.value))}
              className="w-full"
              style={{ background: `linear-gradient(to right, #0ea5e9 ${((nQubits-4)/12)*100}%, #334155 ${((nQubits-4)/12)*100}%)` }}
            />
            <div className="flex justify-between text-[10px] text-slate-600 mt-0.5">
              <span>4</span><span>16</span>
            </div>
          </div>

          {/* Ansatz depth */}
          <div>
            <div className="flex justify-between mb-1.5">
              <span className="text-xs text-slate-400">Ansatz Depth</span>
              <span className="text-xs font-mono font-bold text-violet-400">{depth}</span>
            </div>
            <input
              type="range" min={1} max={5} step={1}
              value={depth}
              onChange={e => setDepth(parseInt(e.target.value))}
              className="w-full purple-thumb"
              style={{ background: `linear-gradient(to right, #8b5cf6 ${((depth-1)/4)*100}%, #334155 ${((depth-1)/4)*100}%)` }}
            />
            <div className="flex justify-between text-[10px] text-slate-600 mt-0.5">
              <span>1</span><span>5</span>
            </div>
          </div>

          {/* Entangler */}
          <Select
            label="Entangler Gate"
            value={entangler}
            options={ENTANGLERS}
            onChange={setEntangler}
          />
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Section C — Circuit Viewer                                           */}
      {/* ------------------------------------------------------------------ */}
      <div>
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-sky-400" />
          Circuit Preview
          {circuitLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-sky-400 ml-1" />}
        </h3>
        {circuitError && (
          <div className="mb-3 flex items-center gap-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
            <AlertCircle className="w-3.5 h-3.5" />
            {circuitError} — showing placeholder
          </div>
        )}
        <CircuitViewer
          svgB64={circuitData?.svg_b64}
          qasm={circuitData?.qasm}
          isLoading={circuitLoading}
          nParams={circuitData?.n_params ?? nParams}
          description={circuitData?.description ?? arch?.fullName}
          architecture={arch?.name}
        />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Section D — Train Model                                              */}
      {/* ------------------------------------------------------------------ */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 space-y-5">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-emerald-400" />
          Train Model
        </h3>

        <div className="flex items-end gap-4">
          <div className="flex-1">
            <Select
              label="Training Dataset"
              value={dataset}
              options={DATASETS_OPTS}
              onChange={setDataset}
            />
          </div>
          <button
            onClick={handleTrain}
            disabled={isTraining}
            className={`
              flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-sm
              transition-all duration-200 shrink-0
              ${isTraining
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white'}
            `}
          >
            {isTraining
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Training…</>
              : <><Play className="w-4 h-4" /> Train Model</>}
          </button>
        </div>

        {/* Training error */}
        {trainError && (
          <div className="flex items-center gap-2 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
            <AlertCircle className="w-4 h-4" /> {trainError}
          </div>
        )}

        {/* Live training chart */}
        {(isTraining || chartData.length > 0) && (
          <div>
            <p className="text-xs text-slate-400 mb-3">Training Progress</p>

            {/* Animated progress bar while training */}
            {isTraining && (
              <div className="mb-3">
                <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full animate-pulse" style={{ width: '60%' }} />
                </div>
              </div>
            )}

            {chartData.length > 0 && (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 10, left: 0 }}>
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis yAxisId="left"  tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={v => `${v}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0', fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />
                  <Line yAxisId="left"  type="monotone" dataKey="loss"     name="Loss"     stroke="#f59e0b" strokeWidth={2} dot={false} />
                  <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy" stroke="#10b981" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        )}

        {/* Final results */}
        {trainResults && !isTraining && (
          <div className="fade-in">
            <p className="text-xs text-slate-400 mb-3 flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5 text-amber-400" />
              Training Complete
            </p>
            <div className="grid grid-cols-4 gap-3">
              <ResultBadge
                label="Test Accuracy"
                value={`${((trainResults.accuracy || 0) * 100).toFixed(1)}`}
                unit="%"
                color="emerald"
              />
              <ResultBadge
                label="Parameters"
                value={trainResults.n_params ?? nParams}
                color="sky"
              />
              <ResultBadge
                label="Training Time"
                value={(trainResults.training_time || 0).toFixed(1)}
                unit="s"
                color="violet"
              />
              <ResultBadge
                label="Final Loss"
                value={(trainResults.loss_curve?.[trainResults.loss_curve.length - 1] || 0).toFixed(4)}
                color="amber"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
