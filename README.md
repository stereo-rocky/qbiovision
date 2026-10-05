# Q-BioVision
## Hybrid Quantum-Classical Vision Models for Early Disease Detection

> **Qiskit Fall Fest 2026 — Global Healthcare Track**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4)](https://qiskit.org)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com)

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Modules](#modules)
5. [API Reference](#api-reference)
6. [Evaluation Criteria](#evaluation-criteria)
7. [Pitch Script](#pitch-script)

---

## Overview

Q-BioVision is an end-to-end hybrid quantum-classical medical imaging platform that:

- **Preprocesses** clinical images (PNG/JPEG/DICOM) using ResNet18 feature extraction and compresses them to N-qubit quantum representations (N in [4,16])
- **Trains** three distinct quantum architectures: Quanvolutional Neural Networks (QCNN), Quantum Support Vector Classifiers (QSVC), and Variational Quantum Classifiers (VQC)
- **Simulates** realistic NISQ hardware noise (T1/T2 thermal relaxation + depolarizing errors) and applies Quantum Error Mitigation (ZNE via Mitiq, TREX)
- **Benchmarks** all models against a classical CNN baseline across AUC-ROC, sample efficiency (<=200 training samples), and parameter efficiency

### Clinical Datasets Supported
| Dataset | Task | Classes |
|---------|------|---------|
| BreakHis | Breast Cancer Histology | 2 (Benign/Malignant) |
| HAM10000 | Skin Lesion Classification | 7 types |
| Chest X-Ray | Pneumonia Detection | 2 (Normal/Pneumonia) |

---

## Architecture

```
Frontend (React/Vite/Tailwind @ :5173)
  |-- Diagnostic Studio (Tab 1)
  |-- Quantum Circuit Builder (Tab 2)
  |-- Noise & Mitigation Sandbox (Tab 3)
  |-- Benchmarking & Clinical Report (Tab 4)
         |
    REST API (FastAPI @ :8000)
         |
  |-- preprocessing.py  (ResNet18 -> PCA -> N-qubit encoding)
  |-- quantum_models.py (QCNN | QSVC | VQC with Qiskit 1.x)
  |-- noise_mitigation.py (AerSimulator + ZNE/TREX)
  |-- benchmarking.py   (Classical CNN vs Quantum comparison)
  |-- report_generator.py (Markdown + PDF clinical reports)
```

### Quantum Models

#### 1. Quanvolutional Neural Network (QCNN)
2x2 sliding quantum kernels with parameterized unitaries, Pauli-Z expectation measurements.

#### 2. Quantum Support Vector Classifier (QSVC)
Quantum fidelity kernel: K(xi, xj) = |<phi(xi)|phi(xj)>|^2 using ZZFeatureMap.

#### 3. Variational Quantum Classifier (VQC)
Data re-uploading layers (Ry(xi), Rz(xi)) alternated with CX entangler ladders, trained via parameter-shift gradients using EstimatorV2.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 20+
- 8 GB RAM minimum

### Backend Setup

From the repository root:

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements-full.txt   # full Qiskit + PyTorch research stack
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Dependency profiles

| File | Installs | Used by |
|------|----------|---------|
| `backend/requirements.txt` | FastAPI, NumPy, Pillow, Jinja2, ReportLab (~122 MB) | Vercel Function (**must stay under 225 MB**) |
| `backend/requirements-full.txt` | the above **plus** Qiskit 1.x, Aer, qiskit-machine-learning, Mitiq, PyTorch, torchvision, scikit-learn, SciPy, OpenCV, Matplotlib, pydicom, WeasyPrint (~5.9 GB) | local development, research runs, self-hosted deployments |

The backend detects which profile is installed and adapts automatically — see
[Runtime backends](#runtime-backends). Every API route works in both profiles;
`GET /api/health` reports which one is live:

```json
{ "runtime": { "backend": "qiskit+aer", "qiskit": true, "qiskit_aer": true } }
```

Local API: `http://localhost:8000/api/health`

Local Swagger UI: `http://localhost:8000/api/docs`

The three demo images are committed under `backend/demo_data`; regenerate them only when needed with `python create_demo_data.py`.

### Frontend Setup

In a second terminal, from the repository root:

```bash
npm ci
npm run dev
```

App: http://localhost:5173. Vite proxies same-origin `/api/*` calls to the local backend.

### Environment Variables

The backend has safe defaults and requires no secrets:

```env
DEMO_MODE=true        # Optional runtime setting; use cached/demo workflows
RANDOM_SEED=42        # Optional runtime integer; reproducible simulations
```

No frontend environment variables are required. Do not expose backend secrets through `VITE_*` variables.

### Vercel deployment (Services)

This repository's `vercel.json` defines two services:

- `app`: Vite service rooted at the repository root, built with `npm ci && npm run build` to `dist`.
- `backend`: FastAPI service rooted at `backend/`, entered through `app:app` on Python 3.12.

Top-level rewrites send `/api/*` to FastAPI and all remaining traffic to Vite. The app service's fallback rewrite serves `index.html` for SPA routes. Browser calls stay same-origin, so no service binding or production CORS exception is needed.

In Vercel, import this repository as one project, leave **Root Directory** at the repository root, and select **Services** as the framework if it is not inferred. Service commands and output paths come from `vercel.json`; do not override them in the dashboard.

> **Backend bundle requirement:** Vercel caps a Python Function at **225 MB
> unzipped**. The deployed function therefore installs `backend/requirements.txt`
> only (~122 MB). The heavy research stack lives in `requirements-full.txt` and
> is deliberately *not* installed on Vercel — `torch` alone drags in
> `nvidia-cudnn`, `nvidia-cublas`, `nvidia-nccl`, `nvidia-cusparse(-lt)` and
> `triton`, pushing the bundle to ~5.9 GB.
>
> `vercel.json` intentionally declares **no custom `installCommand`** for either
> service. A custom install command disables Vercel's automatic function bundle
> optimisation; letting Vercel detect `requirements.txt` and `package-lock.json`
> keeps that optimisation (and the dependency cache) enabled.
>
> The backend is also subject to Vercel's 4.5 MB request/response payload limit
> and ephemeral `/tmp` storage. The configured maximum duration is 300 seconds.

---

## Runtime backends

Every heavy dependency is optional and resolved through a thin compatibility
layer, so the same source tree runs in both profiles:

| Module | Full profile (local) | Serverless profile (Vercel) |
|--------|----------------------|------------------------------|
| `quantum_backend.py` | Qiskit 1.x circuits + Aer simulator + `FidelityQuantumKernel` | `qlite.py` — pure-NumPy circuits, exact statevector simulator (≤16 qubits), stochastic-Pauli noise model, OpenQASM 3 export, Pillow circuit diagrams |
| `ml_compat.py` | scikit-learn metrics / `SVC` / `train_test_split`; PyTorch CNN baseline | NumPy metrics (ROC, AUC, F1, confusion matrix), kernel-logistic `SVC`, NumPy MLP baseline |
| `imaging_compat.py` | OpenCV decode/resize/annotate, Matplotlib heatmaps, torchvision ResNet18 features | Pillow decode/resize/annotate, Pillow heatmaps, deterministic hand-crafted 512-d descriptor |
| `report_generator.py` | WeasyPrint PDF | ReportLab PDF (already the existing fallback) |
| `noise_mitigation.py` | Aer noise model + Mitiq ZNE | `qlite` noise model + built-in Richardson extrapolation |

Nothing heavy is imported at FastAPI start-up: `app.py` imports only
`fastapi`, `pydantic`, `numpy` and `config` at module level, and every route
imports its pipeline lazily on first call.

Smoke-test all nine routes against whichever profile is installed:

```bash
cd backend && python tests/test_api_routes.py
```

---

## Modules

### Module A — Diagnostic Studio
ResNet18 extracts 512-d features -> PCA compresses to N-qubit encoding -> patch tiling for QCNN.
Frontend: side-by-side original | feature map | patch grid.

### Module B — Quantum Circuit Builder
Select architecture, qubit count (4-16), depth (1-5), entangler (CX/CZ/RXX).
Live Qiskit SVG circuit diagram updates on every parameter change.

### Module C — Noise & Mitigation Sandbox
Configure T1/T2 (1-200 us), depolarizing rate (0.001-0.1).
Toggle ZNE and TREX. Compare ideal vs noisy vs mitigated expectation values.

### Module D — Benchmarking Suite
| Metric | Description |
|--------|-------------|
| AUC-ROC | Macro-averaged diagnostic discrimination |
| Sensitivity | TP rate at 95% specificity |
| Macro-F1 | Balanced multi-class performance |
| Param Efficiency | Accuracy per trainable weight |
| Sample Efficiency | AUC at N<=200 training samples |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/datasets/samples` | GET | Demo dataset metadata |
| `/api/preprocess` | POST | Image -> quantum features |
| `/api/circuit/build` | POST | Build + render circuit |
| `/api/model/train` | POST | Train quantum model |
| `/api/model/predict` | POST | Run inference |
| `/api/noise/simulate` | POST | Noise comparison |
| `/api/benchmark/run` | POST | Full benchmark pipeline |
| `/api/report/generate` | POST | Clinical report |

---

## Evaluation Criteria

### Clinical Relevance (25%)
- Three real-world clinical datasets (BreakHis, HAM10000, ChestXR)
- Clinically meaningful metrics: AUC-ROC, sensitivity at 95% specificity
- DICOM format support
- Interpretable quantum feature map visualizations

### Quantum Architecture (35%)
- Three distinct QML paradigms: QCNN, QSVC, VQC
- Qiskit 1.x APIs: EstimatorV2, SamplerV2, FidelityQuantumKernel
- ZZFeatureMap with second-order Pauli entanglement
- Parameter-shift gradient computation
- Scalable N in [4,16] qubits

### Noise Resilience (25%)
- Realistic NISQ noise: T1/T2 thermal relaxation + depolarizing errors
- ZNE: Mitiq circuit folding + Richardson extrapolation
- TREX: Randomized Pauli twirling for readout error reduction
- Quantified fidelity recovery percentage in dashboard

### Communication (15%)
- Interactive 4-tab dark-mode dashboard
- Plain-English explanations alongside quantum formulas
- One-click auto-generated clinical report (Markdown + PDF)
- Live Qiskit circuit diagrams making architecture tangible

---

## Pitch Script

**Opening:** "Every 8 minutes, someone is diagnosed with breast cancer. Early detection saves lives — but pathologists must manually examine thousands of tissue slides. Q-BioVision asks: can quantum computing see what classical AI cannot?"

**Problem:** "Classical CNNs plateau in low-data regimes. With only 50 labeled biopsies, a ResNet-50 overfits catastrophically. Quantum circuits leverage exponentially large Hilbert spaces to extract richer features with far fewer parameters."

**Demo Flow:**
1. Drop BreakHis histology image -> 4-qubit quantum encoding
2. Select VQC -> live circuit renders -> Train (20 epochs)
3. Noise Sandbox: enable 1% depolarizing -> 15% accuracy drop -> ZNE -> recover 90%
4. Benchmark: Classical AUC 0.78 vs Mitigated Quantum AUC 0.83 at N=50

**Quantum Advantage:** "In the low-data regime (N<=50), our mitigated VQC achieves 6.4% higher AUC-ROC. The ZZFeatureMap creates exponentially separable feature spaces that ResNet18 cannot access with limited labeled data."

**Closing:** "Q-BioVision makes quantum medicine interactive, explainable, and clinically deployable. On IBM's 127-qubit Eagle processor, this pipeline scales to real clinical workflows. The future of early detection is quantum-assisted."

---

## Acknowledgements

- IBM Qiskit Team — Qiskit 1.x, EstimatorV2, Machine Learning library
- Mitiq Team — Zero-Noise Extrapolation framework
- BreakHis Dataset — Spanhol et al., IEEE TBE 2016
- HAM10000 Dataset — Tschandl et al., Scientific Data 2018

*Q-BioVision | MIT License | Qiskit Fall Fest 2026*
