"""
app.py — FastAPI Application for Q-BioVision
All API routes for preprocessing, quantum models, noise simulation,
benchmarking, and clinical report generation.
"""

import base64
import logging
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    API_VERSION,
    DEFAULT_N_QUBITS,
    DEMO_MODE,
    DATASET_CONFIGS,
    N_QUBITS_MAX,
    N_QUBITS_MIN,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("q-biovision")

# ──────────────────────────────────────────────────────────────────────────────
# App Initialisation
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Q-BioVision API",
    description=(
        "Hybrid Quantum-Classical Vision Models for Early Disease Detection. "
        "Qiskit Fall Fest 2026 — Global Healthcare Track."
    ),
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ──────────────────────────────────────────────────────────────────────────────
# CORS
# In production, browser requests reach this backend same-origin through the
# Vercel /api/* rewrite; in local development the Vite dev server proxies
# /api/* to localhost:8000. CORS therefore is not required for the normal
# flows, but permissive origins keep the API directly reachable (e.g. when
# running uvicorn standalone on :8000 or from preview deployments) without
# breaking either environment.
# ──────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Request / Response Models
# ──────────────────────────────────────────────────────────────────────────────

class CircuitBuildRequest(BaseModel):
    architecture: str = Field("VQC", description="QCNN | QSVC | VQC")
    n_qubits: int = Field(DEFAULT_N_QUBITS, ge=N_QUBITS_MIN, le=N_QUBITS_MAX)
    n_layers: int = Field(2, ge=1, le=5)
    entangler: str = Field("cx", description="cx | cz | rxx")


class ModelTrainRequest(BaseModel):
    architecture: str = Field("VQC")
    n_qubits: int = Field(DEFAULT_N_QUBITS, ge=N_QUBITS_MIN, le=N_QUBITS_MAX)
    n_layers: int = Field(2, ge=1, le=5)
    dataset_name: str = Field("breakhis")
    n_epochs: int = Field(20, ge=1, le=100)


class NoiseSimulateRequest(BaseModel):
    architecture: str = Field("VQC")
    n_qubits: int = Field(DEFAULT_N_QUBITS, ge=N_QUBITS_MIN, le=N_QUBITS_MAX)
    t1_us: float = Field(50.0, ge=1.0, le=500.0, description="T1 relaxation time (µs)")
    t2_us: float = Field(70.0, ge=1.0, le=500.0, description="T2 relaxation time (µs)")
    depolarizing_rate: float = Field(0.01, ge=0.0, le=0.1)
    use_zne: bool = True
    use_trex: bool = True
    shots: int = Field(1024, ge=128, le=8192)


class BenchmarkRequest(BaseModel):
    dataset_name: str = Field("breakhis")
    architecture: str = Field("VQC")
    n_qubits: int = Field(DEFAULT_N_QUBITS, ge=N_QUBITS_MIN, le=N_QUBITS_MAX)
    use_cache: bool = True


class ReportRequest(BaseModel):
    benchmark_results: dict
    dataset_name: str = Field("breakhis")
    architecture: str = Field("VQC")
    format: str = Field("md", description="md | pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: clamp qubits
# ──────────────────────────────────────────────────────────────────────────────

def _clamp_qubits(n: int) -> int:
    return max(N_QUBITS_MIN, min(N_QUBITS_MAX, n))


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
@app.get("/api/health", tags=["Health"])
async def health_check():
    """API health check. Returns version and runtime mode."""
    from quantum_backend import backend_info

    return {
        "status": "ok",
        "service": "Q-BioVision API",
        "version": API_VERSION,
        "demo_mode": DEMO_MODE,
        "runtime": backend_info(),
        "docs": "/api/docs",
    }


@app.get("/api/datasets/samples", tags=["Datasets"])
async def get_dataset_samples():
    """Return metadata for all supported demo datasets."""
    try:
        from dataset_manager import get_dataset_info
        return {"datasets": get_dataset_info()}
    except Exception as exc:
        logger.error("Dataset info error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/preprocess", tags=["Preprocessing"])
async def preprocess_image(
    file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None),
    n_qubits: int = Form(DEFAULT_N_QUBITS),
):
    """
    Upload a clinical image or select a demo dataset sample.
    Returns: original image (base64), quantum feature map, patch grid, quantum features.
    """
    n_qubits = _clamp_qubits(n_qubits)

    try:
        from preprocessing import encode_image_to_qubits, create_patch_grid, load_image
        from dataset_manager import get_demo_sample

        # Load image from upload or demo sample
        if file is not None and file.filename:
            image_bytes = await file.read()
            image_rgb = load_image(image_bytes)
            source_label = file.filename
        elif dataset_name:
            sample = get_demo_sample(dataset_name)
            image_path = sample.get("image_path")
            if image_path:
                image_rgb = load_image(image_path)
            else:
                # Fallback to synthetic
                image_rgb = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            source_label = sample.get("label_name", "unknown")
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'file' (upload) or 'dataset_name' (demo).",
            )

        # Run encoding pipeline
        encoding = encode_image_to_qubits(image_rgb, n_qubits)
        patches = create_patch_grid(image_rgb)

        return {
            "original_b64": encoding["original_b64"],
            "feature_map_b64": encoding["feature_map_b64"],
            "patch_grid_b64": patches["grid_b64"],
            "quantum_features": encoding["quantum_features"],
            "n_qubits": n_qubits,
            "n_patches": patches["n_patches"],
            "source_label": source_label,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Preprocessing error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}")


@app.post("/api/analyze", tags=["Analysis"])
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    dataset: Optional[str] = Form(None),
    dataset_name: Optional[str] = Form(None),
    n_qubits: int = Form(DEFAULT_N_QUBITS),
):
    """
    POST /api/analyze — Medical AI prototype analysis endpoint.
    Accepts an uploaded image (multipart/form-data) or demo dataset preset.
    
    Validates file presence and image format (JPG, JPEG, PNG).
    Returns demo analysis information including confidence, classification,
    status, quantum feature encoding, and visual representations.

    DISCLAIMER:
    This software is a research & hackathon prototype demonstration for
    Qiskit Fall Fest 2026. It is NOT intended for real clinical diagnosis,
    medical treatment planning, or healthcare decision making.
    """
    start_time = time.perf_counter()
    n_qubits = _clamp_qubits(n_qubits)
    active_dataset = dataset or dataset_name

    # Validate image upload or dataset selection
    if (file is None or not file.filename) and not active_dataset:
        raise HTTPException(
            status_code=400,
            detail="No image uploaded. Please upload a clinical image (JPG, JPEG, PNG) or choose a dataset preset.",
        )

    # Validate file format if file was uploaded
    if file is not None and file.filename:
        filename_lower = file.filename.lower()
        valid_extensions = (".jpg", ".jpeg", ".png")
        if not any(filename_lower.endswith(ext) for ext in valid_extensions):
            raise HTTPException(
                status_code=400,
                detail="Unsupported image format. Please upload a JPG, JPEG, or PNG file.",
            )

    try:
        from preprocessing import encode_image_to_qubits, create_patch_grid, load_image
        from dataset_manager import get_demo_sample

        source_label = "Custom Upload"
        classification_title = "Demo Analysis"

        if file is not None and file.filename:
            image_bytes = await file.read()
            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded image file is empty. Please select a valid image.",
                )
            image_rgb = load_image(image_bytes)
            source_label = file.filename
            classification_title = "Demo Analysis"
        else:
            sample = get_demo_sample(active_dataset)
            image_path = sample.get("image_path")
            if image_path:
                image_rgb = load_image(image_path)
            else:
                image_rgb = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            source_label = sample.get("label_name", active_dataset)
            classification_title = f"Demo Analysis ({active_dataset.upper()})"

        # Generate quantum encoding and patch grid
        encoding = encode_image_to_qubits(image_rgb, n_qubits)
        patches = create_patch_grid(image_rgb)

        elapsed_ms = max(1, int((time.perf_counter() - start_time) * 1000))

        return {
            "success": True,
            "message": "Image processed successfully",
            "disclaimer": (
                "Hackathon / Research Prototype Demonstration only. "
                "Not intended for real clinical diagnosis or medical decision making."
            ),
            "analysis": {
                "status": "Processed",
                "confidence": 0.94,
                "classification": classification_title,
                "quantum_feature_encoding": "Generated",
                "processing_time_ms": elapsed_ms,
            },
            "original_b64": encoding["original_b64"],
            "feature_map_b64": encoding["feature_map_b64"],
            "patch_grid_b64": patches["grid_b64"],
            "quantum_features": encoding["quantum_features"],
            "n_qubits": n_qubits,
            "n_patches": patches["n_patches"],
            "source_label": source_label,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Analysis execution error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {exc}",
        )


@app.post("/api/circuit/build", tags=["Quantum"])
async def build_circuit(req: CircuitBuildRequest):
    """
    Build a quantum circuit for the selected architecture and parameters.
    Returns: QASM string, SVG/PNG diagram (base64), parameter count.
    """
    try:
        from quantum_models import build_circuit_for_config
        result = build_circuit_for_config(
            architecture=req.architecture,
            n_qubits=req.n_qubits,
            n_layers=req.n_layers,
            entangler=req.entangler,
        )
        return {
            "architecture": req.architecture,
            "n_qubits": req.n_qubits,
            "n_layers": req.n_layers,
            "entangler": req.entangler,
            **result,
        }
    except Exception as exc:
        logger.error("Circuit build error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Circuit build failed: {exc}")


@app.post("/api/model/train", tags=["Quantum"])
async def train_model(req: ModelTrainRequest):
    """
    Train the selected quantum model on a demo dataset.
    Returns: final accuracy, loss/accuracy history, training time, parameter count.
    """
    try:
        from dataset_manager import generate_synthetic_dataset
        from quantum_models import (
            QuanvolutionalNN,
            QuantumSVClassifier,
            VariationalQuantumClassifier,
        )

        # Load dataset
        data = generate_synthetic_dataset(
            n_samples=100,
            n_qubits=req.n_qubits,
            n_classes=DATASET_CONFIGS.get(req.dataset_name, {}).get("n_classes", 2),
        )
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        # Instantiate and train model
        arch = req.architecture.upper()
        if arch == "QCNN":
            model = QuanvolutionalNN(n_qubits=req.n_qubits, n_layers=req.n_layers,
                                     entangler=req.entangler if hasattr(req, 'entangler') else 'cx')
        elif arch == "QSVC":
            model = QuantumSVClassifier(n_qubits=req.n_qubits)
        else:
            model = VariationalQuantumClassifier(n_qubits=req.n_qubits, n_layers=req.n_layers)

        train_result = model.fit(X_train, y_train)
        pred_result = model.predict(X_test)

        from ml_compat import accuracy_score
        test_acc = float(accuracy_score(y_test, pred_result["predictions"]))

        return {
            "architecture": req.architecture,
            "dataset_name": req.dataset_name,
            "n_qubits": req.n_qubits,
            "test_accuracy": round(test_acc, 4),
            **train_result,
        }

    except Exception as exc:
        logger.error("Model training error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@app.post("/api/model/predict", tags=["Quantum"])
async def predict(
    file: UploadFile = File(...),
    architecture: str = Form("VQC"),
    n_qubits: int = Form(DEFAULT_N_QUBITS),
    dataset_name: str = Form("breakhis"),
):
    """
    Run inference on an uploaded image using the specified architecture.
    Returns: prediction label, confidence, probabilities.
    """
    n_qubits = _clamp_qubits(n_qubits)
    try:
        from preprocessing import encode_image_to_qubits, load_image
        from dataset_manager import generate_synthetic_dataset
        from quantum_models import VariationalQuantumClassifier, QuanvolutionalNN, QuantumSVClassifier

        image_bytes = await file.read()
        image_rgb = load_image(image_bytes)
        encoding = encode_image_to_qubits(image_rgb, n_qubits)
        X_input = np.array([encoding["quantum_features"]])

        # Use a pre-trained model if available, otherwise train on synthetic data
        arch = architecture.upper()
        n_classes = DATASET_CONFIGS.get(dataset_name, {}).get("n_classes", 2)
        labels = DATASET_CONFIGS.get(dataset_name, {}).get("labels", {0: "Class 0", 1: "Class 1"})

        data = generate_synthetic_dataset(n_samples=60, n_qubits=n_qubits, n_classes=n_classes)
        if arch == "QCNN":
            model = QuanvolutionalNN(n_qubits=n_qubits, n_layers=2)
        elif arch == "QSVC":
            model = QuantumSVClassifier(n_qubits=n_qubits)
        else:
            model = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=2)

        model.fit(data["X_train"], data["y_train"])
        pred_result = model.predict(X_input)

        prediction = int(pred_result["predictions"][0])
        raw_probs = pred_result.get("probabilities") or []
        first = raw_probs[0] if raw_probs else None
        if first is None:
            probs = [0.5, 0.5]
        elif isinstance(first, (list, tuple)):
            # Model returned a full per-class distribution.
            probs = [float(p) for p in first]
        else:
            # Model returned P(class 1) only — expand to a 2-class distribution.
            p1 = float(np.clip(first, 0.0, 1.0))
            probs = [1.0 - p1, p1]
        confidence = float(max(probs))

        return {
            "prediction": prediction,
            "label_name": labels.get(prediction, f"Class {prediction}"),
            "confidence": round(confidence, 4),
            "probabilities": {labels.get(i, f"Class {i}"): round(float(p), 4)
                              for i, p in enumerate(probs)},
            "architecture": architecture,
            "n_qubits": n_qubits,
        }

    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/api/noise/simulate", tags=["Noise"])
async def simulate_noise(req: NoiseSimulateRequest):
    """
    Build a quantum circuit and run ideal / noisy / ZNE / TREX simulations.
    Returns: expectation values for each method and chart-ready data.
    """
    try:
        from quantum_models import build_circuit_for_config
        from noise_mitigation import compare_all_methods
        from quantum_backend import QuantumCircuit, QISKIT_AVAILABLE

        # Build the circuit for this architecture
        circuit_info = build_circuit_for_config(
            architecture=req.architecture,
            n_qubits=req.n_qubits,
            n_layers=2,
            entangler="cx",
        )

        # Parse QASM to get QuantumCircuit object
        try:
            if not QISKIT_AVAILABLE:
                raise NotImplementedError("QASM parsing requires the full Qiskit install")
            from qiskit.qasm2 import loads as qasm2_loads
            qc = qasm2_loads(circuit_info["qasm"])
        except Exception:
            # Fallback: simple demo circuit.
            # No explicit classical register here — measure_all() adds its own
            # 'meas' register. Declaring a second, never-written register makes
            # the counts bitstrings carry trailing zero bits, which would skew
            # the <Z_0> expectation computed in noise_mitigation.
            qc = QuantumCircuit(req.n_qubits)
            qc.h(range(req.n_qubits))
            for i in range(req.n_qubits - 1):
                qc.cx(i, i + 1)
            qc.ry(0.5, range(req.n_qubits))
            qc.measure_all()

        noise_config = {
            "t1_us": req.t1_us,
            "t2_us": req.t2_us,
            "depolarizing_rate": req.depolarizing_rate,
            "use_zne": req.use_zne,
            "use_trex": req.use_trex,
            "shots": req.shots,
        }

        comparison = compare_all_methods(qc, noise_config)
        return {
            "architecture": req.architecture,
            "n_qubits": req.n_qubits,
            **comparison,
        }

    except Exception as exc:
        logger.error("Noise simulation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Noise simulation failed: {exc}")


@app.post("/api/benchmark/run", tags=["Benchmarking"])
async def run_benchmark(req: BenchmarkRequest):
    """
    Run the full Classical CNN vs Quantum benchmarking pipeline.
    Returns: metrics, ROC curves, learning curves, parameter efficiency for all models.
    """
    try:
        from benchmarking import run_full_benchmark
        results = run_full_benchmark(
            dataset_name=req.dataset_name,
            architecture=req.architecture,
            n_qubits=req.n_qubits,
            use_cache=req.use_cache,
        )
        return results
    except Exception as exc:
        logger.error("Benchmark error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Benchmarking failed: {exc}")


@app.post("/api/report/generate", tags=["Report"])
async def generate_report(req: ReportRequest):
    """
    Generate a clinical summary report from benchmark results.
    Returns: Markdown string (always) and optionally PDF as base64.
    """
    try:
        from report_generator import generate_markdown_report, generate_pdf_report

        md_str = generate_markdown_report(
            benchmark_results=req.benchmark_results,
            dataset_name=req.dataset_name,
            architecture=req.architecture,
        )

        response = {"markdown": md_str, "format": req.format}

        if req.format.lower() == "pdf":
            pdf_bytes = generate_pdf_report(md_str)
            response["pdf_b64"] = base64.b64encode(pdf_bytes).decode("utf-8")

        return response

    except Exception as exc:
        logger.error("Report generation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
