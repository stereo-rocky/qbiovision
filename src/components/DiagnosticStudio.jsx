// ================================================================
// src/components/DiagnosticStudio.jsx
// Tab 1 — Medical image upload, quantum encoding, feature visualization
// ================================================================
import React, { useState, useCallback } from 'react'
import { useDropzone }  from 'react-dropzone'
import {
  Microscope, Upload, ChevronRight, Loader2, CheckCircle2,
  AlertCircle, Sliders, Database, Image as ImageIcon, BarChart2,
  X, RefreshCw,
} from 'lucide-react'
import { preprocessImage, analyzeImage } from '../api/client.js'
import ImagePanel from './ImagePanel.jsx'

// ---------------------------------------------------------------------------
// Dataset preset cards
// ---------------------------------------------------------------------------
const DATASETS = [
  {
    id:          'breakhis',
    name:        'BreakHis',
    description: 'Breast cancer histopathological images — benign vs malignant classification',
    icon:        '🔬',
    samples:     7909,
    classes:     2,
    type:        'Histopathology',
  },
  {
    id:          'ham10000',
    name:        'HAM10000',
    description: 'Skin lesion dermoscopy — 7-class skin cancer diagnosis',
    icon:        '🩺',
    samples:     10015,
    classes:     7,
    type:        'Dermoscopy',
  },
  {
    id:          'chestxr',
    name:        'ChestXR',
    description: 'Chest X-ray radiographs — pneumonia vs normal binary classification',
    icon:        '🫁',
    samples:     5863,
    classes:     2,
    type:        'Radiology',
  },
]

// ---------------------------------------------------------------------------
// Dataset card component
// ---------------------------------------------------------------------------
function DatasetCard({ dataset, selected, onSelect }) {
  return (
    <button
      onClick={() => onSelect(dataset.id)}
      className={`
        flex-1 p-4 rounded-xl border text-left transition-all duration-200
        ${selected
          ? 'border-sky-500/60 bg-sky-500/10 shadow-quantum'
          : 'border-slate-700 bg-slate-800/50 hover:border-slate-500 hover:bg-slate-800'}
      `}
    >
      <div className="text-2xl mb-2">{dataset.icon}</div>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-sm font-semibold text-slate-200">{dataset.name}</span>
        {selected && <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" />}
      </div>
      <p className="text-xs text-slate-400 mb-2 leading-relaxed">{dataset.description}</p>
      <div className="flex gap-2 flex-wrap">
        <span className="text-[10px] bg-slate-700 text-slate-400 rounded-md px-1.5 py-0.5">
          {dataset.samples.toLocaleString()} samples
        </span>
        <span className="text-[10px] bg-slate-700 text-slate-400 rounded-md px-1.5 py-0.5">
          {dataset.classes} classes
        </span>
        <span className="text-[10px] bg-slate-700 text-slate-400 rounded-md px-1.5 py-0.5">
          {dataset.type}
        </span>
      </div>
    </button>
  )
}

// ---------------------------------------------------------------------------
// Quantum features bar visualization
// ---------------------------------------------------------------------------
function QuantumFeaturesPanel({ features }) {
  if (!features || features.length === 0) return null

  const maxVal = Math.max(...features.map(Math.abs))

  return (
    <div className="bg-slate-800 border border-violet-500/30 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <BarChart2 className="w-4 h-4 text-violet-400" />
        <h3 className="text-sm font-semibold text-slate-200">Quantum Feature Vector</h3>
        <span className="text-xs text-slate-500 ml-auto">{features.length} components</span>
      </div>

      <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
        {features.map((val, i) => {
          const pct = maxVal > 0 ? (Math.abs(val) / maxVal) * 100 : 0
          const isNeg = val < 0
          return (
            <div key={i} className="flex items-center gap-3">
              <span className="text-[11px] font-mono text-slate-500 w-10 shrink-0">
                f[{i.toString().padStart(2, '0')}]
              </span>
              <div className="flex-1 bg-slate-700 rounded-full h-2 relative overflow-hidden">
                <div
                  className={`absolute h-full rounded-full transition-all duration-500 ${
                    isNeg ? 'bg-violet-500' : 'bg-sky-500'
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className={`text-[11px] font-mono w-16 text-right shrink-0 ${
                isNeg ? 'text-violet-400' : 'text-sky-400'
              }`}>
                {val.toFixed(4)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// File dropzone
// ---------------------------------------------------------------------------
function FileDropzone({ onFileDrop, preview, onClear }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) onFileDrop(acceptedFiles[0])
  }, [onFileDrop])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] },
    maxFiles: 1,
    maxSize:  20 * 1024 * 1024, // 20 MB
  })

  return (
    <div
      {...getRootProps()}
      className={`
        relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
        transition-all duration-200
        ${isDragActive && !isDragReject ? 'border-sky-400 bg-sky-500/10 shadow-quantum' : ''}
        ${isDragReject ? 'border-red-500 bg-red-500/10' : ''}
        ${!isDragActive ? 'border-slate-600 hover:border-slate-400 hover:bg-slate-800/50' : ''}
      `}
    >
      <input {...getInputProps()} />

      {preview ? (
        // Show image preview
        <div className="relative inline-block">
          <img
            src={preview}
            alt="Uploaded preview"
            className="max-h-40 max-w-full object-contain rounded-lg mx-auto"
          />
          <button
            type="button"
            onClick={e => { e.stopPropagation(); onClear() }}
            className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-slate-600 hover:bg-red-500
                       flex items-center justify-center transition-colors"
          >
            <X className="w-3.5 h-3.5 text-white" />
          </button>
          <p className="text-xs text-slate-400 mt-2">Click to replace image</p>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="w-12 h-12 mx-auto rounded-xl bg-slate-700 flex items-center justify-center">
            {isDragActive
              ? <ChevronRight className="w-6 h-6 text-sky-400 animate-bounce" />
              : <Upload className="w-6 h-6 text-slate-400" />
            }
          </div>
          <div>
            <p className="text-sm font-medium text-slate-300">
              {isDragActive ? 'Drop image here' : 'Drag & drop medical image'}
            </p>
            <p className="text-xs text-slate-500 mt-1">PNG, JPEG up to 20MB</p>
          </div>
          {isDragReject && (
            <p className="text-xs text-red-400">Only PNG/JPEG images are accepted</p>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// DiagnosticStudio — Main export
// ---------------------------------------------------------------------------
export default function DiagnosticStudio({ globalState, updateGlobalState }) {
  // Input state
  const [uploadedFile,     setUploadedFile]     = useState(null)
  const [previewURL,       setPreviewURL]       = useState(null)
  const [selectedDataset,  setSelectedDataset]  = useState(globalState.selectedDataset || null)
  const [nQubits,          setNQubits]          = useState(globalState.nQubits || 8)

  // Async state
  const [isLoading,        setIsLoading]        = useState(false)
  const [error,            setError]            = useState(null)
  const [success,          setSuccess]          = useState(false)

  // Results
  const [preprocessed,    setPreprocessed]      = useState(globalState.preprocessedData || null)

  // Handle file upload via dropzone
  const handleFileDrop = (file) => {
    setUploadedFile(file)
    setPreviewURL(URL.createObjectURL(file))
    setSelectedDataset(null) // Uploading custom image clears dataset preset
    setError(null)
    setSuccess(false)
  }

  const handleClearFile = () => {
    setUploadedFile(null)
    if (previewURL) URL.revokeObjectURL(previewURL)
    setPreviewURL(null)
  }

  const handleDatasetSelect = (id) => {
    setSelectedDataset(id)
    setUploadedFile(null)
    setPreviewURL(null)
    setError(null)
    updateGlobalState({ selectedDataset: id })
  }

  // Main encode action
  const handleEncode = async () => {
    if (!uploadedFile && !selectedDataset) {
      setError('Please upload an image or select a dataset first.')
      return
    }

    setIsLoading(true)
    setError(null)
    setSuccess(false)

    try {
      const formData = new FormData()
      if (uploadedFile) {
        formData.append('file', uploadedFile)
      } else {
        // For dataset presets, send dataset name
        formData.append('dataset', selectedDataset)
        formData.append('dataset_name', selectedDataset)
        formData.append('use_sample', 'true')
      }
      formData.append('n_qubits', String(nQubits))

      const result = await analyzeImage(formData)
      setPreprocessed(result)
      updateGlobalState({ preprocessedData: result, nQubits })
      setSuccess(true)
      setTimeout(() => setSuccess(false), 4000)
    } catch (err) {
      console.error('Analysis error:', err)
      setError('Analysis service is temporarily unavailable. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setPreprocessed(null)
    setUploadedFile(null)
    setPreviewURL(null)
    setSelectedDataset(null)
    setError(null)
    setSuccess(false)
    updateGlobalState({ preprocessedData: null })
  }

  return (
    <div className="space-y-6 fade-in">
      {/* ------------------------------------------------------------------ */}
      {/* Header                                                               */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-8 h-8 rounded-lg bg-sky-500/20 flex items-center justify-center">
              <Microscope className="w-5 h-5 text-sky-400" />
            </div>
            <h2 className="text-xl font-bold text-white">Diagnostic Studio</h2>
          </div>
          <p className="text-sm text-slate-400 ml-10.5">
            Upload medical images or use curated datasets to generate quantum feature encodings.
          </p>
        </div>

        {preprocessed && (
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200
                       border border-slate-700 hover:border-slate-600 rounded-lg px-3 py-1.5 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Reset
          </button>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Section A — Data Input                                               */}
      {/* ------------------------------------------------------------------ */}
      <div className="grid grid-cols-5 gap-5">
        {/* Upload panel (3 cols) */}
        <div className="col-span-3 bg-slate-800 border border-slate-700 rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Upload className="w-4 h-4 text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-200">Upload Custom Image</h3>
          </div>
          <FileDropzone
            onFileDrop={handleFileDrop}
            preview={previewURL}
            onClear={handleClearFile}
          />
        </div>

        {/* Dataset presets (2 cols) */}
        <div className="col-span-2 bg-slate-800 border border-slate-700 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-violet-400" />
            <h3 className="text-sm font-semibold text-slate-200">Or Use Dataset Preset</h3>
          </div>
          <div className="space-y-2">
            {DATASETS.map(ds => (
              <DatasetCard
                key={ds.id}
                dataset={ds}
                selected={selectedDataset === ds.id && !uploadedFile}
                onSelect={handleDatasetSelect}
              />
            ))}
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Qubit slider + Encode button                                         */}
      {/* ------------------------------------------------------------------ */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 flex items-center gap-8">
        {/* Qubit count */}
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-3">
            <Sliders className="w-4 h-4 text-sky-400" />
            <span className="text-sm font-semibold text-slate-200">Qubit Count</span>
            <span className="ml-auto text-lg font-bold text-sky-400 font-mono">{nQubits}</span>
          </div>
          <input
            type="range" min={4} max={16} step={1}
            value={nQubits}
            onChange={e => setNQubits(parseInt(e.target.value))}
            className="w-full"
            style={{
              background: `linear-gradient(to right, #0ea5e9 ${((nQubits - 4) / 12) * 100}%, #334155 ${((nQubits - 4) / 12) * 100}%)`,
            }}
          />
          <div className="flex justify-between text-[10px] text-slate-600 mt-1">
            <span>4 qubits</span>
            <span>Hilbert space: 2^{nQubits} = {Math.pow(2, nQubits).toLocaleString()} dims</span>
            <span>16 qubits</span>
          </div>
        </div>

        {/* Encode button */}
        <button
          onClick={handleEncode}
          disabled={isLoading || (!uploadedFile && !selectedDataset)}
          className={`
            flex items-center gap-2.5 px-8 py-3.5 rounded-xl font-semibold text-sm
            transition-all duration-200 shrink-0
            ${isLoading || (!uploadedFile && !selectedDataset)
              ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
              : 'bg-sky-600 hover:bg-sky-500 text-white shadow-quantum hover:shadow-quantum-lg'}
          `}
        >
          {isLoading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Encoding…</>
            : <><ImageIcon className="w-4 h-4" /> Encode &amp; Analyze</>}
        </button>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Status messages                                                      */}
      {/* ------------------------------------------------------------------ */}
      {error && (
        <div className="flex items-start gap-2.5 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-sm text-red-400">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
      {success && (
        <div className="flex items-center gap-2.5 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-sm text-emerald-400 fade-in">
          <CheckCircle2 className="w-4 h-4" />
          Image successfully encoded into {nQubits}-qubit quantum feature map!
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Section C — Demo Analysis & Visual Results                         */}
      {/* ------------------------------------------------------------------ */}
      <div className="space-y-4">
        {preprocessed?.analysis && (
          <div className="bg-slate-800/90 border border-sky-500/40 rounded-xl p-5 shadow-quantum fade-in">
            <div className="flex items-center justify-between pb-3 border-b border-slate-700/80 mb-4">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                <h4 className="text-sm font-semibold text-slate-100 uppercase tracking-wider">
                  Demo Analysis Result
                </h4>
              </div>
              <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-medium">
                {preprocessed.analysis.status || 'Processed'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
              <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-700/60">
                <span className="text-xs text-slate-400 block mb-1">Status</span>
                <span className="text-sm font-semibold text-emerald-400 font-mono flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {preprocessed.analysis.status}
                </span>
              </div>
              <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-700/60">
                <span className="text-xs text-slate-400 block mb-1">Confidence</span>
                <span className="text-sm font-semibold text-sky-400 font-mono">
                  {typeof preprocessed.analysis.confidence === 'number'
                    ? `${(preprocessed.analysis.confidence * 100).toFixed(1)}% (${preprocessed.analysis.confidence})`
                    : preprocessed.analysis.confidence}
                </span>
              </div>
              <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-700/60">
                <span className="text-xs text-slate-400 block mb-1">Classification</span>
                <span className="text-sm font-semibold text-violet-300 font-mono">
                  {preprocessed.analysis.classification}
                </span>
              </div>
              <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-700/60">
                <span className="text-xs text-slate-400 block mb-1">Quantum Encoding</span>
                <span className="text-sm font-semibold text-cyan-300 font-mono">
                  {preprocessed.analysis.quantum_feature_encoding}
                  {preprocessed.analysis.processing_time_ms != null && (
                    <span className="text-xs text-slate-400 ml-1 font-normal">
                      ({preprocessed.analysis.processing_time_ms} ms)
                    </span>
                  )}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs text-amber-400/90 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
              <AlertCircle className="w-4 h-4 shrink-0 text-amber-400" />
              <span>
                <strong>Research Prototype:</strong> Demonstration only for hackathon evaluation. Not intended for real clinical diagnosis.
              </span>
            </div>
          </div>
        )}

        <div>
          <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-sky-400" />
            Image Analysis Results
          </h3>
          <ImagePanel
            originalB64={preprocessed?.original_b64}
            featureMapB64={preprocessed?.feature_map_b64}
            patchGridB64={preprocessed?.patch_grid_b64}
            isLoading={isLoading}
          />
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Section D — Quantum Features readout                                 */}
      {/* ------------------------------------------------------------------ */}
      {preprocessed?.quantum_features && (
        <QuantumFeaturesPanel features={preprocessed.quantum_features} />
      )}
    </div>
  )
}
