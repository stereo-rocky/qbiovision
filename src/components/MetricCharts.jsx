// ================================================================
// src/components/MetricCharts.jsx
// Interactive Recharts dashboard — 4 charts comparing model metrics
// ================================================================
import React from 'react'
import {
  LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

// ---------------------------------------------------------------------------
// Color palette matching the app theme
// ---------------------------------------------------------------------------
const COLORS = {
  classical:  '#f59e0b',  // amber  — classical CNN
  quantum:    '#0ea5e9',  // sky    — quantum unmitigated
  mitigated:  '#10b981',  // emerald — quantum mitigated
}

// ---------------------------------------------------------------------------
// Dark chart theme shared props
// ---------------------------------------------------------------------------
const CHART_STYLE = {
  backgroundColor: 'transparent',
}

const TOOLTIP_STYLE = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: '8px',
  color: '#e2e8f0',
  fontSize: '12px',
}

const AXIS_TICK_STYLE = { fill: '#94a3b8', fontSize: 11 }
const GRID_STYLE      = { stroke: '#1e293b', strokeDasharray: '3 3' }

// ---------------------------------------------------------------------------
// Section heading
// ---------------------------------------------------------------------------
function ChartHeading({ title, subtitle }) {
  return (
    <div className="mb-4">
      <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
      {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Card wrapper
// ---------------------------------------------------------------------------
function ChartCard({ children, className = '' }) {
  return (
    <div className={`bg-slate-800 border border-slate-700 rounded-xl p-5 ${className}`}>
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Chart 1 — AUC-ROC Multi-Curve
// ---------------------------------------------------------------------------
function ROCChart({ rocData }) {
  // rocData: Array<{ fpr, classical_tpr, quantum_tpr, mitigated_tpr }>
  const data = rocData || generateDefaultROC()

  return (
    <ChartCard>
      <ChartHeading
        title="AUC-ROC Curves"
        subtitle="False Positive Rate vs True Positive Rate — higher left-bulge = better"
      />
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis
            dataKey="fpr"
            label={{ value: 'FPR', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 11 }}
            tick={AXIS_TICK_STYLE}
            domain={[0, 1]}
          />
          <YAxis
            label={{ value: 'TPR', angle: -90, position: 'insideLeft', offset: 10, fill: '#94a3b8', fontSize: 11 }}
            tick={AXIS_TICK_STYLE}
            domain={[0, 1]}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(v, name) => [v.toFixed(3), name]}
          />
          <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8', paddingTop: '8px' }} />
          <ReferenceLine x={0} y={0} stroke="#334155" />
          <Line
            type="monotone" dataKey="classical_tpr" name="Classical CNN"
            stroke={COLORS.classical} strokeWidth={2} dot={false}
          />
          <Line
            type="monotone" dataKey="quantum_tpr" name="Quantum (Unmitigated)"
            stroke={COLORS.quantum} strokeWidth={2} dot={false}
          />
          <Line
            type="monotone" dataKey="mitigated_tpr" name="Quantum (Mitigated)"
            stroke={COLORS.mitigated} strokeWidth={2} dot={false} strokeDasharray="4 2"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

// ---------------------------------------------------------------------------
// Chart 2 — Learning Curves (Accuracy vs Training Size)
// ---------------------------------------------------------------------------
function LearningCurveChart({ learningCurveData }) {
  const data = learningCurveData || generateDefaultLearningCurves()

  return (
    <ChartCard>
      <ChartHeading
        title="Learning Curves"
        subtitle="Test accuracy as a function of training set size"
      />
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis
            dataKey="train_size"
            label={{ value: 'Training Samples', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 11 }}
            tick={AXIS_TICK_STYLE}
          />
          <YAxis
            label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', offset: 10, fill: '#94a3b8', fontSize: 11 }}
            tick={AXIS_TICK_STYLE}
            domain={[0.4, 1.0]}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(v, name) => [`${(v * 100).toFixed(1)}%`, name]}
          />
          <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8', paddingTop: '8px' }} />
          <Line
            type="monotone" dataKey="classical" name="Classical CNN"
            stroke={COLORS.classical} strokeWidth={2} dot={{ r: 3, fill: COLORS.classical }}
          />
          <Line
            type="monotone" dataKey="quantum" name="Quantum (Unmitigated)"
            stroke={COLORS.quantum} strokeWidth={2} dot={{ r: 3, fill: COLORS.quantum }}
            strokeDasharray="5 2"
          />
          <Line
            type="monotone" dataKey="mitigated" name="Quantum (Mitigated)"
            stroke={COLORS.mitigated} strokeWidth={2} dot={{ r: 3, fill: COLORS.mitigated }}
            strokeDasharray="2 2"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

// ---------------------------------------------------------------------------
// Chart 3 — Parameter Efficiency (Accuracy per Parameter)
// ---------------------------------------------------------------------------
function ParamEfficiencyChart({ paramEfficiencyData }) {
  const data = paramEfficiencyData || [
    { name: 'Classical CNN',        accuracy_per_param: 0.00012, params: 6500, accuracy: 0.78 },
    { name: 'Quantum (Unmitigated)',accuracy_per_param: 0.00980, params: 82,   accuracy: 0.81 },
    { name: 'Quantum (Mitigated)',  accuracy_per_param: 0.01060, params: 82,   accuracy: 0.87 },
  ]

  return (
    <ChartCard>
      <ChartHeading
        title="Parameter Efficiency"
        subtitle="Test accuracy per trainable parameter — quantum advantage metric"
      />
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 40, left: 10 }}>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis
            dataKey="name" tick={{ ...AXIS_TICK_STYLE, fontSize: 10 }}
            interval={0} angle={-10} textAnchor="end"
          />
          <YAxis
            tick={AXIS_TICK_STYLE}
            tickFormatter={v => v.toExponential(1)}
            label={{ value: 'Acc/Param', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(v, name, props) => [
              v.toExponential(3),
              'Accuracy / Parameter',
            ]}
          />
          <Bar dataKey="accuracy_per_param" name="Acc/Param" radius={[4, 4, 0, 0]}
               fill="url(#effGrad)" />
          <defs>
            <linearGradient id="effGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.mitigated} stopOpacity={0.9} />
              <stop offset="100%" stopColor={COLORS.mitigated} stopOpacity={0.4} />
            </linearGradient>
          </defs>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

// ---------------------------------------------------------------------------
// Chart 4 — Multi-Metric Grouped BarChart (AUC, F1, Sensitivity, Precision)
// ---------------------------------------------------------------------------
function MetricComparisonChart({ classicalMetrics, quantumMetrics, quantumMitigatedMetrics }) {
  const metrics = ['auc', 'f1', 'sensitivity', 'precision']
  const labels  = { auc: 'AUC', f1: 'F1', sensitivity: 'Sensitivity', precision: 'Precision' }

  const cm = classicalMetrics        || { auc: 0.78, f1: 0.74, sensitivity: 0.71, precision: 0.77 }
  const qm = quantumMetrics          || { auc: 0.81, f1: 0.79, sensitivity: 0.76, precision: 0.82 }
  const mm = quantumMitigatedMetrics || { auc: 0.87, f1: 0.85, sensitivity: 0.83, precision: 0.88 }

  const data = metrics.map(m => ({
    metric:    labels[m],
    Classical: +(cm[m] * 100).toFixed(1),
    Quantum:   +(qm[m] * 100).toFixed(1),
    Mitigated: +(mm[m] * 100).toFixed(1),
  }))

  return (
    <ChartCard>
      <ChartHeading
        title="Model Metric Comparison"
        subtitle="AUC, F1-Score, Sensitivity, Precision — higher is better"
      />
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }} barGap={3}>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis dataKey="metric" tick={AXIS_TICK_STYLE} />
          <YAxis
            tick={AXIS_TICK_STYLE}
            tickFormatter={v => `${v}%`}
            domain={[50, 100]}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(v) => [`${v}%`]}
          />
          <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8', paddingTop: '8px' }} />
          <Bar dataKey="Classical" name="Classical CNN"       fill={COLORS.classical} radius={[3,3,0,0]} />
          <Bar dataKey="Quantum"   name="Quantum (Unmit.)"   fill={COLORS.quantum}   radius={[3,3,0,0]} />
          <Bar dataKey="Mitigated" name="Quantum (Mitigated)" fill={COLORS.mitigated} radius={[3,3,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

// ---------------------------------------------------------------------------
// Default data generators (used when props are null/undefined)
// ---------------------------------------------------------------------------
function generateDefaultROC() {
  const pts = []
  for (let i = 0; i <= 20; i++) {
    const fpr = i / 20
    pts.push({
      fpr,
      classical_tpr:  Math.min(1, Math.pow(fpr, 0.55) * 1.05),
      quantum_tpr:    Math.min(1, Math.pow(fpr, 0.45) * 1.08),
      mitigated_tpr:  Math.min(1, Math.pow(fpr, 0.35) * 1.1),
    })
  }
  return pts
}

function generateDefaultLearningCurves() {
  const sizes = [10, 25, 50, 75, 100, 150, 200]
  return sizes.map(s => ({
    train_size: s,
    classical:  Math.min(0.95, 0.5 + 0.35 * Math.log(s / 10) / Math.log(20)),
    quantum:    Math.min(0.96, 0.55 + 0.33 * Math.log(s / 10) / Math.log(20)),
    mitigated:  Math.min(0.98, 0.6  + 0.34 * Math.log(s / 10) / Math.log(20)),
  }))
}

// ---------------------------------------------------------------------------
// MetricCharts — Main export (renders all 4 charts in a 2x2 grid)
// ---------------------------------------------------------------------------
export default function MetricCharts({
  classicalMetrics,
  quantumMetrics,
  quantumMitigatedMetrics,
  rocData,
  learningCurveData,
  paramEfficiencyData,
}) {
  return (
    <div className="grid grid-cols-2 gap-5">
      <ROCChart rocData={rocData} />
      <LearningCurveChart learningCurveData={learningCurveData} />
      <ParamEfficiencyChart paramEfficiencyData={paramEfficiencyData} />
      <MetricComparisonChart
        classicalMetrics={classicalMetrics}
        quantumMetrics={quantumMetrics}
        quantumMitigatedMetrics={quantumMitigatedMetrics}
      />
    </div>
  )
}
