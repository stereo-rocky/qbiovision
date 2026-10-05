// ================================================================
// src/components/ImagePanel.jsx
// Three-panel image comparison: Original | Quantum Feature Map | Patch Grid
// ================================================================
import React, { useState } from 'react'
import { Info, Image as ImageIcon, ZoomIn } from 'lucide-react'

// ---------------------------------------------------------------------------
// Tooltip component — shows descriptive text on hover
// ---------------------------------------------------------------------------
function InfoTooltip({ text }) {
  const [visible, setVisible] = useState(false)
  return (
    <span className="relative inline-flex items-center ml-1.5">
      <button
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        className="text-slate-500 hover:text-sky-400 transition-colors"
        aria-label="Info"
      >
        <Info className="w-3.5 h-3.5" />
      </button>
      {visible && (
        <span className="
          absolute z-50 bottom-6 left-1/2 -translate-x-1/2
          w-56 text-xs text-slate-200 bg-slate-700 border border-slate-600
          rounded-lg p-2.5 shadow-xl leading-relaxed
        ">
          {text}
        </span>
      )}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Skeleton loading card
// ---------------------------------------------------------------------------
function SkeletonCard({ title }) {
  return (
    <div className="flex-1 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700">
        <div className="h-4 w-32 bg-slate-700 rounded animate-pulse" />
      </div>
      <div className="p-4">
        <div className="aspect-square bg-slate-700 rounded-lg animate-pulse flex items-center justify-center">
          <ImageIcon className="w-8 h-8 text-slate-600" />
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Individual image card with zoom-on-hover
// ---------------------------------------------------------------------------
function ImageCard({ title, imageSrc, tooltipText, accentColor = 'border-sky-500/40', isEmpty }) {
  const [zoomed, setZoomed] = useState(false)

  return (
    <div className={`
      flex-1 bg-slate-800 border rounded-xl overflow-hidden
      transition-all duration-300 group
      ${accentColor}
      ${!isEmpty ? 'hover:shadow-quantum hover:border-sky-500/60' : 'border-slate-700'}
    `}>
      {/* Card header */}
      <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-200 flex items-center">
          {title}
          <InfoTooltip text={tooltipText} />
        </span>
        {imageSrc && (
          <button
            onClick={() => setZoomed(z => !z)}
            className="text-slate-500 hover:text-sky-400 transition-colors"
            title={zoomed ? 'Zoom out' : 'Zoom in'}
          >
            <ZoomIn className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Image area */}
      <div className="p-4">
        {imageSrc ? (
          <div className={`relative overflow-hidden rounded-lg transition-all duration-500 ${zoomed ? 'scale-150 origin-center' : ''}`}>
            <img
              src={imageSrc}
              alt={title}
              className="w-full aspect-square object-contain rounded-lg
                         transition-transform duration-300 group-hover:scale-105"
            />
          </div>
        ) : (
          <div className="aspect-square bg-slate-900/50 rounded-lg flex flex-col items-center justify-center gap-2 border border-dashed border-slate-700">
            <ImageIcon className="w-8 h-8 text-slate-600" />
            <p className="text-xs text-slate-600 text-center px-4">
              Process an image to see results here
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Full-screen zoom modal
// ---------------------------------------------------------------------------
function ZoomModal({ imageSrc, title, onClose }) {
  if (!imageSrc) return null
  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8 cursor-zoom-out"
      onClick={onClose}
    >
      <div className="max-w-3xl max-h-full" onClick={e => e.stopPropagation()}>
        <p className="text-white text-sm font-medium mb-3 text-center">{title}</p>
        <img src={imageSrc} alt={title} className="max-h-[80vh] object-contain rounded-xl" />
        <p className="text-slate-400 text-xs text-center mt-3">Click anywhere to close</p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ImagePanel — Main export
// ---------------------------------------------------------------------------
export default function ImagePanel({
  originalB64,
  featureMapB64,
  patchGridB64,
  isLoading = false,
}) {
  const [zoomTarget, setZoomTarget] = useState(null) // { src, title }

  // Build data URIs from base64 strings
  const originalSrc    = originalB64    ? `data:image/png;base64,${originalB64}`    : null
  const featureMapSrc  = featureMapB64  ? `data:image/png;base64,${featureMapB64}`  : null
  const patchGridSrc   = patchGridB64   ? `data:image/png;base64,${patchGridB64}`   : null

  if (isLoading) {
    return (
      <div className="flex gap-4">
        <SkeletonCard title="Original Image" />
        <SkeletonCard title="Quantum Feature Map" />
        <SkeletonCard title="Patch Grid" />
      </div>
    )
  }

  return (
    <>
      <div className="flex gap-4">
        {/* Panel 1 — Original */}
        <ImageCard
          title="Original Image"
          imageSrc={originalSrc}
          tooltipText="The raw medical image as uploaded. This is the input to the quantum encoding pipeline."
          accentColor="border-sky-500/40"
          isEmpty={!originalSrc}
        />

        {/* Panel 2 — Quantum Feature Map */}
        <ImageCard
          title="Quantum Feature Map"
          imageSrc={featureMapSrc}
          tooltipText="Amplitude-encoded quantum state representation of the image. Each pixel value is encoded as a rotation angle in the quantum circuit, revealing features invisible to classical methods."
          accentColor="border-violet-500/40"
          isEmpty={!featureMapSrc}
        />

        {/* Panel 3 — Patch Grid */}
        <ImageCard
          title="Patch Grid"
          imageSrc={patchGridSrc}
          tooltipText="The image divided into NxN patches, each processed by an independent quantum circuit. This hierarchical approach reduces circuit depth while maintaining spatial locality."
          accentColor="border-emerald-500/40"
          isEmpty={!patchGridSrc}
        />
      </div>

      {/* Zoom modal overlay */}
      {zoomTarget && (
        <ZoomModal
          imageSrc={zoomTarget.src}
          title={zoomTarget.title}
          onClose={() => setZoomTarget(null)}
        />
      )}
    </>
  )
}
