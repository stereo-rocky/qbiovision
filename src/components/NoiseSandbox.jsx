import React, { useState } from 'react';
import { Zap, BarChart3, AlertTriangle, CheckCircle, Play, RotateCcw } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import { simulateNoise } from '../api/client';
import NoiseControls from './NoiseControls';

// ── Custom Recharts Tooltip ─────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 shadow-xl">
      <p className="text-slate-300 text-sm font-medium mb-1">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.fill || entry.color }} className="text-sm">
          ⟨Z⟩ = {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
};

// ── Fidelity Gauge ───────────────────────────────────────────────────────────
const FidelityGauge = ({ fidelityLoss }) => {
  const fidelity = Math.max(0, Math.min(100, 100 - (fidelityLoss || 0)));
  const color = fidelity >= 80 ? '#10b981' : fidelity >= 60 ? '#f59e0b' : '#ef4444';

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-28 h-28">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          {/* Background track */}
          <circle cx="50" cy="50" r="40" fill="none" stroke="#334155" strokeWidth="10" />
          {/* Fidelity arc */}
          <circle
            cx="50" cy="50" r="40" fill="none"
            stroke={color} strokeWidth="10"
            strokeDasharray={`${fidelity * 2.51} 251`}
            strokeLinecap="round"
          />
        </svg>
        {/* Center label */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold" style={{ color }}>{fidelity.toFixed(0)}%</span>
          <span className="text-slate-400 text-xs">Fidelity</span>
        </div>
      </div>
      <p className="text-slate-400 text-xs mt-2 text-center">
        {(fidelityLoss || 0).toFixed(1)}% lost to noise
      </p>
    </div>
  );
};

// ── Method Explanation Card ──────────────────────────────────────────────────
const MethodCard = ({ title, color, icon, description, formula }) => (
  <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
    <div className="flex items-center gap-2 mb-2">
      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-sm font-semibold text-slate-200">{title}</span>
    </div>
    <p className="text-slate-400 text-xs leading-relaxed mb-2">{description}</p>
    {formula && (
      <code className="text-xs bg-slate-900 text-sky-400 px-2 py-1 rounded font-mono">
        {formula}
      </code>
    )}
  </div>
);

// ────────────────────────────────────────────────────────────────────────────
// Main Component: NoiseSandbox
// ────────────────────────────────────────────────────────────────────────────
const NoiseSandbox = ({ globalState, onStateChange }) => {
  const [architecture, setArchitecture] = useState('VQC');
  const [nQubits, setNQubits] = useState(4);
  const [noiseConfig, setNoiseConfig] = useState({
    enableThermal: true,
    t1_us: 50,
    t2_us: 70,
    enableDepolarizing: true,
    depolarizing_rate: 0.01,
    use_zne: true,
    use_trex: true,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleRunSimulation = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const payload = {
        architecture,
        n_qubits: nQubits,
        t1_us: noiseConfig.enableThermal ? noiseConfig.t1_us : 10000,
        t2_us: noiseConfig.enableThermal ? noiseConfig.t2_us : 10000,
        depolarizing_rate: noiseConfig.enableDepolarizing ? noiseConfig.depolarizing_rate : 0.0,
        use_zne: noiseConfig.use_zne,
        use_trex: noiseConfig.use_trex,
        shots: 1024,
      };
      const data = await simulateNoise(payload);
      setResults(data);
      onStateChange?.({ noiseResults: data });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Simulation failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Build chart data from results
  const chartData = results?.chart_data || [];

  // ZNE recovery calculation
  const zneRecovery = results && results.zne?.mitigated_expectation != null
    ? Math.max(0, Math.min(100,
        (1 - Math.abs(results.ideal?.expectation_value - results.zne.mitigated_expectation) /
          Math.max(Math.abs(results.ideal?.expectation_value - results.noisy?.expectation_value), 0.001)) * 100
      )).toFixed(1)
    : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
          <div className="w-8 h-8 bg-violet-500/20 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-violet-400" />
          </div>
          Noise &amp; Mitigation Sandbox
        </h2>
        <p className="text-slate-400 mt-1 text-sm">
          Simulate realistic NISQ hardware noise and apply quantum error mitigation (ZNE, TREX)
          to inspect circuit fidelity degradation and recovery.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Configuration */}
        <div className="space-y-4">
          {/* Circuit Selector */}
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
            <h3 className="text-slate-200 font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-sky-400" />
              Circuit Configuration
            </h3>

            {/* Architecture */}
            <div className="mb-4">
              <label className="block text-xs text-slate-400 mb-2 uppercase tracking-wider">
                Architecture
              </label>
              <div className="flex gap-2">
                {['VQC', 'QCNN', 'QSVC'].map(arch => (
                  <button
                    key={arch}
                    onClick={() => setArchitecture(arch)}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                      architecture === arch
                        ? 'bg-violet-600 text-white shadow-lg shadow-violet-900/30'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {arch}
                  </button>
                ))}
              </div>
            </div>

            {/* Qubit Count */}
            <div>
              <label className="flex justify-between text-xs text-slate-400 mb-2 uppercase tracking-wider">
                <span>Qubits</span>
                <span className="text-violet-400 font-semibold">{nQubits}</span>
              </label>
              <input
                type="range" min={4} max={8} step={2}
                value={nQubits}
                onChange={e => setNQubits(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>4</span><span>6</span><span>8</span>
              </div>
            </div>
          </div>

          {/* Noise Controls */}
          <NoiseControls config={noiseConfig} onChange={setNoiseConfig} />

          {/* Run Button */}
          <button
            onClick={handleRunSimulation}
            disabled={isLoading}
            className="w-full py-3 px-6 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all
              bg-gradient-to-r from-violet-600 to-sky-600 hover:from-violet-500 hover:to-sky-500
              text-white shadow-lg shadow-violet-900/30 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Running Simulation...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Simulation
              </>
            )}
          </button>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 bg-red-900/30 border border-red-700/50 rounded-lg p-3">
              <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="space-y-4">
          {/* Fidelity & Key Stats */}
          {results ? (
            <>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
                <h3 className="text-slate-200 font-semibold mb-4 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  Simulation Results
                </h3>

                <div className="flex items-center justify-around mb-4">
                  {/* Fidelity Gauge */}
                  <FidelityGauge fidelityLoss={results.fidelity_loss_pct} />

                  {/* Key Stats */}
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                      <span className="text-slate-400">Ideal:</span>
                      <span className="text-slate-200 font-mono">
                        {results.ideal?.expectation_value?.toFixed(4) ?? 'N/A'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                      <span className="text-slate-400">Noisy:</span>
                      <span className="text-slate-200 font-mono">
                        {results.noisy?.expectation_value?.toFixed(4) ?? 'N/A'}
                      </span>
                    </div>
                    {results.zne?.mitigated_expectation != null && (
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full bg-sky-500" />
                        <span className="text-slate-400">ZNE:</span>
                        <span className="text-slate-200 font-mono">
                          {results.zne.mitigated_expectation?.toFixed(4)}
                        </span>
                      </div>
                    )}
                    {results.trex?.mitigated_expectation != null && (
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full bg-violet-500" />
                        <span className="text-slate-400">TREX:</span>
                        <span className="text-slate-200 font-mono">
                          {results.trex.mitigated_expectation?.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* ZNE Recovery Callout */}
                {zneRecovery && (
                  <div className="bg-sky-900/30 border border-sky-700/40 rounded-lg p-3">
                    <p className="text-sky-300 text-xs">
                      <strong>ZNE Mitigation</strong> recovered{' '}
                      <strong>{zneRecovery}%</strong> of fidelity lost to NISQ noise using
                      Richardson linear extrapolation at scale factors [1, 2, 3].
                    </p>
                  </div>
                )}
              </div>

              {/* Bar Chart */}
              {chartData.length > 0 && (
                <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
                  <h3 className="text-slate-200 font-semibold mb-4 text-sm">
                    ⟨Z⟩ Expectation Comparison
                  </h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        dataKey="method"
                        tick={{ fill: '#94a3b8', fontSize: 10 }}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[-1, 1]}
                        tick={{ fill: '#94a3b8', fontSize: 10 }}
                        tickLine={false}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                      <Bar dataKey="expectation" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          ) : (
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-8 flex flex-col items-center justify-center text-center min-h-[300px]">
              <Zap className="w-12 h-12 text-slate-600 mb-3" />
              <p className="text-slate-400 text-sm">
                Configure noise parameters and run a simulation to compare<br />
                ideal, noisy, and error-mitigated quantum circuit outputs.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Method Explanation Cards */}
      <div>
        <h3 className="text-slate-300 font-semibold text-sm mb-3 uppercase tracking-wider">
          Mitigation Methods Explained
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          <MethodCard
            title="Ideal Simulation"
            color="#10b981"
            description="Noiseless AerSimulator with statevector method. Provides the theoretical upper bound on circuit performance."
            formula="⟨Z⟩ = ⟨ψ|Z|ψ⟩"
          />
          <MethodCard
            title="NISQ Noise"
            color="#ef4444"
            description="T1/T2 thermal relaxation and depolarizing errors calibrated to IBM Quantum hardware specifications."
            formula="ρ_noisy = ε(ρ_ideal)"
          />
          <MethodCard
            title="ZNE (Zero-Noise Extrapolation)"
            color="#0ea5e9"
            description="Circuit folding at scale factors [1,2,3] creates amplified noise instances. Richardson linear extrapolation recovers the zero-noise limit."
            formula="E[λ→0] = 2E[1] − E[2]"
          />
          <MethodCard
            title="TREX (Twirled Readout)"
            color="#8b5cf6"
            description="Randomized Pauli-X twirling on measurement qubits symmetrizes readout errors, enabling bias-free correction via averaging."
            formula="⟨Z⟩_trex = avg over N twirls"
          />
        </div>
      </div>
    </div>
  );
};

export default NoiseSandbox;
