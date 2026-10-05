"""
config.py — Global Configuration for Q-BioVision
Sets random seeds, model registry, dataset metadata, and all shared constants.
"""

import os
import random
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility Seeds
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

try:
    import torch
    torch.manual_seed(RANDOM_SEED)
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Qubit Configuration
# ──────────────────────────────────────────────────────────────────────────────
N_QUBITS_MIN: int = 4
N_QUBITS_MAX: int = 16
DEFAULT_N_QUBITS: int = 4

# ──────────────────────────────────────────────────────────────────────────────
# Demo Mode  (set DEMO_MODE=false in environment to use real datasets)
# ──────────────────────────────────────────────────────────────────────────────
_raw_demo = os.environ.get("DEMO_MODE", "true")
DEMO_MODE: bool = _raw_demo.strip().lower() not in ("false", "0", "no")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Label Maps
# ──────────────────────────────────────────────────────────────────────────────
BREAKHIS_LABELS: dict = {
    0: "benign",
    1: "malignant",
}

HAM10000_LABELS: dict = {
    0: "Melanocytic nevi (nv)",
    1: "Melanoma (mel)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Basal cell carcinoma (bcc)",
    4: "Actinic keratoses (akiec)",
    5: "Vascular lesions (vasc)",
    6: "Dermatofibroma (df)",
}

CHESTXR_LABELS: dict = {
    0: "normal",
    1: "pneumonia",
}

# Unified lookup used throughout the application
LABEL_MAPS: dict = {
    "breakhis": BREAKHIS_LABELS,
    "ham10000": HAM10000_LABELS,
    "chestxr":  CHESTXR_LABELS,
}

# ──────────────────────────────────────────────────────────────────────────────
# Image & Patch Parameters
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE: int = 224          # Resize target for all input images (pixels)
PATCH_SIZE: int = 2            # Patch grid dimension (2x2 = 4 patches)

# ──────────────────────────────────────────────────────────────────────────────
# API / CORS
# ──────────────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list = [
    "http://localhost:5173",   # Vite dev server (React / Vue)
    "http://localhost:3000",   # CRA / Next.js dev server
]

# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_DIM: int = 512         # ResNet18 penultimate (avgpool) output dimension

MAX_TRAIN_SAMPLES: int = 200   # Upper cap on training samples per run
DEFAULT_EPOCHS: int = 20
DEFAULT_LR: float = 0.1
LEARNING_CURVE_SIZES: list = [10, 25, 50, 100, 200]
DEFAULT_N_LAYERS: int = 2
DEFAULT_ENTANGLER: str = "cx"


# ──────────────────────────────────────────────────────────────────────────────
# File-system Directories
# ──────────────────────────────────────────────────────────────────────────────
CACHE_DIR: Path = Path("./cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEMO_DATA_DIR: Path = Path("./demo_data")
DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model Registry
# Maps short architecture keys to their implementing class names in
# quantum_models.py — used by the factory function and the API layer.
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict = {
    "QCNN": "QuanvolutionalNN",
    "QSVC": "QuantumSVClassifier",
    "VQC":  "VariationalQuantumClassifier",
}

# ──────────────────────────────────────────────────────────────────────────────
# API Version
# ──────────────────────────────────────────────────────────────────────────────
API_VERSION: str = "1.0.0"

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Configurations (used by app.py and dataset_manager.py)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_CONFIGS: dict = {
    "breakhis": {
        "name": "BreakHis",
        "n_classes": 2,
        "labels": {0: "Benign", 1: "Malignant"},
        "description": "Breast cancer histology (H&E stained)",
        "sample_image": "breakhis_sample.png",
    },
    "ham10000": {
        "name": "HAM10000",
        "n_classes": 2,
        "labels": {0: "Non-Melanoma", 1: "Melanoma"},
        "description": "Dermoscopic skin lesion classification",
        "sample_image": "ham10000_sample.png",
    },
    "chestxr": {
        "name": "Chest X-Ray",
        "n_classes": 2,
        "labels": {0: "Normal", 1: "Pneumonia"},
        "description": "Chest radiograph pneumonia detection",
        "sample_image": "chestxr_sample.png",
    },
}

