// ================================================================
// src/api/client.js — Axios API client for Q-BioVision backend
// All requests proxy to http://localhost:8000 via Vite proxy config
// ================================================================
import axios from 'axios'

// ---------------------------------------------------------------------------
// Base Axios instance
// ---------------------------------------------------------------------------
const apiClient = axios.create({
  baseURL: '/',          // Vite proxies /api/* -> http://localhost:8000
  timeout: 60000,        // 60 second timeout for long-running quantum jobs
  headers: {
    'Accept': 'application/json',
  },
})

// ---------------------------------------------------------------------------
// Request interceptor — log outgoing requests in development
// ---------------------------------------------------------------------------
apiClient.interceptors.request.use(
  (config) => {
    if (import.meta.env.DEV) {
      console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data)
    }
    return config
  },
  (error) => Promise.reject(error)
)

// ---------------------------------------------------------------------------
// Response interceptor — normalize error messages
// ---------------------------------------------------------------------------
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'Unknown API error'
    console.error(`[API Error] ${message}`, error.response?.data)
    return Promise.reject(new Error(message))
  }
)

// ---------------------------------------------------------------------------
// API Functions
// ---------------------------------------------------------------------------

/**
 * Analyze an uploaded clinical image with quantum feature encoding.
 * POST /api/analyze
 * @param {FormData} formData — multipart form data with file or dataset
 * @returns {Promise<{success, message, analysis, original_b64, feature_map_b64, patch_grid_b64, quantum_features, n_qubits}>}
 */
export async function analyzeImage(formData) {
  const response = await apiClient.post('/api/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

/**
 * Preprocess a medical image — encodes to quantum feature map and patch grid.
 * @param {FormData} formData — must contain 'file' key with image file
 * @returns {Promise<{original_b64, feature_map_b64, patch_grid_b64, quantum_features, n_qubits}>}
 */
export async function preprocessImage(formData) {
  const response = await apiClient.post('/api/preprocess', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

/**
 * Build and return a quantum circuit diagram for a given architecture.
 * @param {Object} config — { architecture, n_qubits, depth, entangler }
 * @returns {Promise<{svg_b64, qasm, n_params, description, architecture}>}
 */
export async function buildCircuit(config) {
  const response = await apiClient.post('/api/circuit/build', config)
  return response.data
}

/**
 * Train a quantum ML model on a medical image dataset.
 * @param {Object} config — { architecture, dataset, n_qubits, depth, entangler }
 * @returns {Promise<{accuracy, loss_curve, accuracy_curve, n_params, training_time}>}
 */
export async function trainModel(config) {
  const response = await apiClient.post('/api/model/train', config)
  return response.data
}

/**
 * Run inference on a single image using the trained quantum model.
 * @param {FormData} formData — must contain 'file' key with image file
 * @returns {Promise<{prediction, confidence, class_probabilities}>}
 */
export async function predictImage(formData) {
  const response = await apiClient.post('/api/model/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

/**
 * Simulate quantum noise effects on a circuit and apply mitigation techniques.
 * @param {Object} config — { architecture, n_qubits, t1, t2, depolarizing_rate, enable_zne, enable_trex }
 * @returns {Promise<{ideal, noisy, zne, trex, fidelity_loss, key_findings}>}
 */
export async function simulateNoise(config) {
  const response = await apiClient.post('/api/noise/simulate', config)
  return response.data
}

/**
 * Run a full benchmark comparing Classical CNN vs Quantum (unmitigated) vs Quantum (mitigated).
 * @param {Object} config — { dataset, architecture, n_qubits }
 * @returns {Promise<{classical_metrics, quantum_metrics, quantum_mitigated_metrics, roc_data, learning_curve_data, param_efficiency_data}>}
 */
export async function runBenchmark(config) {
  const response = await apiClient.post('/api/benchmark/run', config)
  return response.data
}

/**
 * Generate a clinical report in markdown and/or PDF format.
 * @param {Object} config — { format, benchmark_results, model_info }
 * @returns {Promise<{markdown_content, pdf_b64}>}
 */
export async function generateReport(config) {
  const response = await apiClient.post('/api/report/generate', config)
  return response.data
}

/**
 * Fetch sample images from available medical datasets for preview.
 * @returns {Promise<Array<{dataset, label, image_b64, description}>>}
 */
export async function getDatasetSamples() {
  const response = await apiClient.get('/api/datasets/samples')
  return response.data
}

export default apiClient
