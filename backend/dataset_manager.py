"""
dataset_manager.py — Demo Dataset Manager for Q-BioVision
Handles loading of bundled demo samples and synthetic data generation.
All random operations use RANDOM_SEED from config for reproducibility.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED,
    LABEL_MAPS,
    DEMO_DATA_DIR,
    MAX_TRAIN_SAMPLES,
    N_QUBITS_MIN,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from dataset name to the demo file stored in demo_data/
_DEMO_FILES: dict = {
    "breakhis":  ("breakhis_sample.png",  0, "benign"),
    "ham10000":  ("ham10000_sample.png",  1, "Melanoma (mel)"),
    "chestxr":   ("chestxr_sample.png",   1, "pneumonia"),
}

# Number of classes per dataset
_N_CLASSES: dict = {
    "breakhis": 2,
    "ham10000": 7,
    "chestxr":  2,
}

# Human-readable descriptions for the info endpoint
_DESCRIPTIONS: dict = {
    "breakhis": (
        "Breast histopathology images (BreakHis dataset). "
        "Binary classification: benign vs malignant tumour."
    ),
    "ham10000": (
        "Human Against Machine with 10000 training images (HAM10000). "
        "7-class skin lesion classification."
    ),
    "chestxr": (
        "Chest X-ray images for pneumonia detection. "
        "Binary classification: normal vs pneumonia."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_demo_sample(dataset_name: str) -> dict:
    """Return a representative demo sample for the given dataset.

    Looks for a pre-saved PNG in *demo_data/*.  If the file is absent
    (e.g. first run before create_demo_data.py has been executed), falls back
    to a synthetic 224×224×3 random uint8 NumPy array that still exercises
    the full preprocessing pipeline.

    Parameters
    ----------
    dataset_name : str
        One of ``'breakhis'``, ``'ham10000'``, ``'chestxr'``.

    Returns
    -------
    dict with keys:
        - ``image_path``  : absolute path string (or ``None`` for synthetic)
        - ``image_array`` : ``np.ndarray`` uint8 shape (224, 224, 3)
        - ``label``       : integer class index
        - ``label_name``  : human-readable class name
        - ``dataset``     : normalised dataset key
        - ``is_synthetic``: bool flag
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in _DEMO_FILES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(_DEMO_FILES.keys())}"
        )

    filename, label_idx, label_name = _DEMO_FILES[dataset_name]
    image_path = DEMO_DATA_DIR / filename

    # ── attempt to load the real demo image ──────────────────────────────────
    if image_path.exists():
        try:
            from PIL import Image  # type: ignore
            pil_img = Image.open(image_path).convert("RGB").resize((224, 224))
            image_array = np.array(pil_img, dtype=np.uint8)
            return {
                "image_path":   str(image_path.resolve()),
                "image_array":  image_array,
                "label":        label_idx,
                "label_name":   label_name,
                "dataset":      dataset_name,
                "is_synthetic": False,
            }
        except Exception as exc:
            logger.warning("Failed to open demo image %s: %s", image_path, exc)

    # ── synthetic fallback ────────────────────────────────────────────────────
    logger.info(
        "Demo image not found at %s; generating synthetic fallback.", image_path
    )
    rng = np.random.default_rng(RANDOM_SEED)
    image_array = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    return {
        "image_path":   None,
        "image_array":  image_array,
        "label":        label_idx,
        "label_name":   label_name,
        "dataset":      dataset_name,
        "is_synthetic": True,
    }


def generate_synthetic_dataset(
    n_samples: int = 50,
    n_qubits: int = 4,
    n_classes: int = 2,
    seed: int = RANDOM_SEED,
) -> dict:
    """Generate a synthetic quantum-feature dataset using class-conditional Gaussians.

    Each sample is drawn from a multivariate normal distribution whose mean is
    determined by its class label.  The resulting features can be treated as
    already-encoded quantum angles in [0, 2π].

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    n_qubits : int
        Feature dimensionality (one feature per qubit).
    n_classes : int
        Number of distinct output classes.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    dict with keys ``X_train``, ``y_train``, ``X_test``, ``y_test``
        Each value is a ``np.ndarray``.  X arrays have shape (n, n_qubits).
    """
    rng = np.random.default_rng(seed)

    X_list, y_list = [], []
    samples_per_class = max(1, n_samples // n_classes)

    for class_idx in range(n_classes):
        # Class mean uniformly spaced in [0, 2π]
        mean_val = (class_idx / max(n_classes - 1, 1)) * 2 * np.pi
        mean = np.full(n_qubits, mean_val)

        # Covariance: identity scaled so classes are separable
        cov = np.eye(n_qubits) * (np.pi / (n_classes * 2))

        samples = rng.multivariate_normal(mean, cov, size=samples_per_class)
        # Clip to valid angle range [0, 2π]
        samples = np.clip(samples, 0, 2 * np.pi)

        X_list.append(samples)
        y_list.extend([class_idx] * samples_per_class)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle together
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
    }


def load_dataset_split(
    dataset_name: str,
    max_samples: int = MAX_TRAIN_SAMPLES,
) -> dict:
    """Load (or generate) a train/test split for the requested dataset.

    In demo mode this always delegates to :func:`generate_synthetic_dataset`.
    When real dataset files are present a future implementation can load them
    directly; for now the synthetic generator covers all three datasets.

    Parameters
    ----------
    dataset_name : str
        One of ``'breakhis'``, ``'ham10000'``, ``'chestxr'``.
    max_samples : int
        Upper bound on the number of total samples generated.

    Returns
    -------
    dict
        Same structure as :func:`generate_synthetic_dataset`.
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in _N_CLASSES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(_N_CLASSES.keys())}"
        )

    n_classes = _N_CLASSES[dataset_name]
    # Use at least N_QUBITS_MIN qubits so the feature vector is wide enough
    n_qubits = max(N_QUBITS_MIN, n_classes)

    result = generate_synthetic_dataset(
        n_samples=min(max_samples, MAX_TRAIN_SAMPLES),
        n_qubits=n_qubits,
        n_classes=n_classes,
        seed=RANDOM_SEED,
    )
    result["dataset_name"] = dataset_name
    result["n_classes"]    = n_classes
    result["n_qubits"]     = n_qubits
    return result


def get_dataset_info() -> list:
    """Return metadata for all three bundled demo datasets.

    Returns
    -------
    list[dict]
        Each entry contains ``name``, ``classes``, ``description``,
        ``sample_count``, and ``label_map``.
    """
    info = []
    for ds_name, n_cls in _N_CLASSES.items():
        label_map = LABEL_MAPS.get(ds_name, {})
        info.append({
            "name":         ds_name,
            "display_name": ds_name.upper(),
            "classes":      n_cls,
            "description":  _DESCRIPTIONS[ds_name],
            "sample_count": MAX_TRAIN_SAMPLES,   # synthetic count always fixed
            "label_map":    label_map,
        })
    return info


def augment_for_low_data_regime(
    X: np.ndarray,
    y: np.ndarray,
    target_n: int = 200,
    noise_std: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Augment quantum feature vectors to reach *target_n* samples.

    Augmentation strategy: Gaussian noise injection with a small standard
    deviation relative to the feature range ([0, 2π]).  This preserves the
    quantum angle semantics while adding diversity.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Existing feature matrix.
    y : np.ndarray, shape (n_samples,)
        Corresponding labels.
    target_n : int
        Desired total sample count after augmentation.
    noise_std : float
        Standard deviation of the additive Gaussian noise (in radians).
    seed : int
        NumPy seed for reproducibility.

    Returns
    -------
    tuple (X_aug, y_aug)
        Both are ``np.ndarray`` with ``len == target_n`` (or more if
        ``target_n <= len(X)``, in which case the original arrays are
        returned unchanged).
    """
    if len(X) >= target_n:
        return X, y

    rng = np.random.default_rng(seed)
    n_needed = target_n - len(X)

    X_extra_list = []
    y_extra_list = []

    # Sample with replacement from the existing pool
    indices = rng.integers(0, len(X), size=n_needed)
    X_base   = X[indices]
    y_extra  = y[indices]

    noise = rng.normal(0, noise_std, size=X_base.shape).astype(np.float32)
    X_extra = np.clip(X_base + noise, 0, 2 * np.pi)

    X_aug = np.vstack([X, X_extra]).astype(np.float32)
    y_aug = np.concatenate([y, y_extra]).astype(y.dtype)

    # Final shuffle
    perm  = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]
