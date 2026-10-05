// ================================================================
// src/components/ReportExport.jsx
// Report export: markdown preview + PDF/Markdown download buttons
// ================================================================
import React from 'react'
import {
  FileText, Download, Loader2, FileCode2,
  AlertCircle, CheckCircle2, RefreshCw,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Helper — download a blob URL
// ---------------------------------------------------------------------------
function downloadBlob(url, filename) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// ---------------------------------------------------------------------------
// Markdown preview with simple line-by-line rendering
// ---------------------------------------------------------------------------
function MarkdownPreview({ content }) {
  if (!content) return null

  // Minimal markdown → HTML for display purposes
  const lines = content.split('\n')
  return (
    <div className="qasm-code text-slate-300 space-y-0.5 leading-relaxed text-xs">
      {lines.map((line, i) => {
        if (line.startsWith('# '))
          return <p key={i} className="text-sky-400 font-bold text-sm mt-3">{line.slice(2)}</p>
        if (line.startsWith('## '))
          return <p key={i} className="text-sky-300 font-semibold mt-2">{line.slice(3)}</p>
        if (line.startsWith('### '))
          return <p key={i} className="text-violet-300 font-medium mt-1.5">{line.slice(4)}</p>
        if (line.startsWith('- '))
          return <p key={i} className="ml-3 text-slate-300">• {line.slice(2)}</p>
        if (line.startsWith('**') && line.endsWith('**'))
          return <p key={i} className="text-slate-200 font-semibold">{line.slice(2, -2)}</p>
        if (line.trim() === '')
          return <div key={i} className="h-2" />
        if (line.startsWith('|'))
          return <p key={i} className="text-slate-400 font-mono text-[11px]">{line}</p>
        return <p key={i} className="text-slate-300">{line}</p>
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ReportExport — Main export
// ---------------------------------------------------------------------------
export default function ReportExport({
  markdownContent,
  pdfB64,
  isLoading = false,
  onGenerate,
}) {
  const hasContent = !!markdownContent

  // Download handlers
  const handleDownloadMarkdown = () => {
    if (!markdownContent) return
    const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8' })
    const url  = URL.createObjectURL(blob)
    downloadBlob(url, 'q_biovision_clinical_report.md')
  }

  const handleDownloadPDF = () => {
    if (!pdfB64) return
    // Decode base64 PDF and create a blob URL
    const byteChars  = atob(pdfB64)
    const byteArrays = []
    for (let i = 0; i < byteChars.length; i += 512) {
      const slice = byteChars.slice(i, i + 512)
      const arr   = new Uint8Array(slice.length)
      for (let j = 0; j < slice.length; j++) arr[j] = slice.charCodeAt(j)
      byteArrays.push(arr)
    }
    const blob = new Blob(byteArrays, { type: 'application/pdf' })
    const url  = URL.createObjectURL(blob)
    downloadBlob(url, 'q_biovision_clinical_report.pdf')
  }

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-sky-400" />
          <span className="text-sm font-semibold text-slate-200">Clinical Report</span>
        </div>

        {/* Status indicator */}
        {hasContent && (
          <div className="flex items-center gap-1.5 text-xs text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Report Ready
          </div>
        )}
      </div>

      <div className="p-5">
        {/* No content yet — show generate button */}
        {!hasContent && !isLoading && (
          <div className="flex flex-col items-center justify-center py-12 gap-4">
            <div className="w-16 h-16 rounded-2xl bg-slate-700/50 flex items-center justify-center">
              <FileCode2 className="w-8 h-8 text-slate-500" />
            </div>
            <div className="text-center">
              <p className="text-slate-300 text-sm font-medium">No report generated yet</p>
              <p className="text-slate-500 text-xs mt-1">
                Run a benchmark first, then generate the clinical report.
              </p>
            </div>
            {onGenerate && (
              <button
                onClick={onGenerate}
                className="flex items-center gap-2 px-5 py-2.5 bg-sky-600 hover:bg-sky-500
                           text-white text-sm font-medium rounded-lg transition-colors"
              >
                <FileText className="w-4 h-4" />
                Generate Report
              </button>
            )}
          </div>
        )}

        {/* Loading state */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-12 gap-3">
            <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
            <p className="text-slate-400 text-sm">Generating clinical report…</p>
            <p className="text-slate-600 text-xs">Compiling metrics, charts, and clinical interpretation</p>
          </div>
        )}

        {/* Content available */}
        {hasContent && !isLoading && (
          <div className="space-y-4">
            {/* Markdown preview */}
            <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 max-h-96 overflow-y-auto">
              <MarkdownPreview content={markdownContent} />
            </div>

            {/* Action buttons */}
            <div className="flex gap-3">
              <button
                onClick={handleDownloadMarkdown}
                className="flex-1 flex items-center justify-center gap-2
                           px-4 py-2.5 bg-slate-700 hover:bg-slate-600 border border-slate-600
                           text-slate-200 text-sm font-medium rounded-lg transition-colors"
              >
                <FileCode2 className="w-4 h-4 text-violet-400" />
                Download Markdown
              </button>

              <button
                onClick={handleDownloadPDF}
                disabled={!pdfB64}
                className="flex-1 flex items-center justify-center gap-2
                           px-4 py-2.5 bg-sky-700 hover:bg-sky-600 disabled:opacity-40
                           disabled:cursor-not-allowed border border-sky-600
                           text-white text-sm font-medium rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                {pdfB64 ? 'Download PDF' : 'PDF Unavailable'}
              </button>

              {onGenerate && (
                <button
                  onClick={onGenerate}
                  className="px-4 py-2.5 text-slate-400 hover:text-slate-200
                             border border-slate-700 hover:border-slate-600
                             rounded-lg transition-colors flex items-center gap-1.5 text-sm"
                  title="Regenerate report"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* File size indicator */}
            <p className="text-xs text-slate-600 text-center">
              Markdown: {(markdownContent.length / 1024).toFixed(1)} KB
              {pdfB64 && ` · PDF: ${(pdfB64.length * 0.75 / 1024).toFixed(0)} KB`}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
