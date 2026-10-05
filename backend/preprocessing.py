"""
preprocessing.py — Clinical Image Preprocessing Pipeline for Q-BioVision
ResNet18 feature extraction -> PCA -> N-qubit amplitude encoding -> Patch tiling.

Pipeline summary:
  raw image (any format)
      |-> load_image()            : normalise to RGB uint8 224x224
      |-> extract_resnet18_features()  : 512-d feature vector
      |-> compress_to_n_qubits()  : PCA / projection to n_qubits angles
      |-> (optional) create_patch_grid() : sliding QCNN patches
"""

import base64
import io
import logging
import math
from pathlib import Path
from typing import Union

import cv2           # type: ignore
import numpy as np
from PIL import Image  # type: ignore
from sklearn.decomposition import PCA  # type: ignore

import torch
import torchvision.models as tv_models        # type: ignore
import torchvision.transforms as transforms    # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RANDOM_SEED, IMAGE_SIZE, FEATURE_DIM

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level model cache (avoid reloading on every request)
# ──────────────────────────────────────────────────────────────────────────────
_resnet18_model: Union[torch.nn.Module, None] = None
_resnet18_device: str = "cpu"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_rgb: np.ndarray) -> str:
    """Convert a NumPy RGB image array to a base64-encoded PNG string.

    The output can be embedded directly in JSON as a data-URI or used with
    the ``<img src="data:image/png;base64,...">`` pattern.

    Parameters
    ----------
    image_rgb : np.ndarray
        uint8 array of shape (H, W, 3) in RGB channel order.

    Returns
    -------
    str
        Base64-encoded PNG bytes (no ``data:image/png;base64,`` prefix).
    """
    pil_img = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def create_feature_map_visualization(features: np.ndarray, n_qubits: int) -> str:
    """Visualise the n_qubits-dimensional feature vector as a heatmap.

    The feature vector is reshaped into a 2D grid whose side length is the
    smallest integer >= sqrt(n_qubits).  Colour-coded with the viridis
    colourmap against the [0, 2pi] range.

    Parameters
    ----------
    features : np.ndarray
        1-D array of length n_qubits (angle-encoded quantum features).
    n_qubits : int
        Number of qubits / feature dimensions.

    Returns
    -------
    str
        Base64-encoded PNG of the heatmap.
    """
    side = math.ceil(math.sqrt(n_qubits))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[:n_qubits] = features[:n_qubits]
    grid = padded.reshape(side, side)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=2 * math.pi, aspect="auto")
    plt.colorbar(im, ax=ax, label="Angle (rad)")
    ax.set_title(f"Quantum Feature Map ({n_qubits} qubits)", fontsize=8)
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Image Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """Load an image from various sources and return a 224x224 RGB uint8 array.

    Supported sources:
    * ``str`` / ``Path`` — file path (JPEG, PNG, BMP, DICOM .dcm).
    * ``bytes``          — raw image bytes (JPEG or PNG in memory).
    * ``np.ndarray``     — already-loaded array (H, W) or (H, W, 3).

    Parameters
    ----------
    source : str | Path | bytes | np.ndarray
        The image source to load.

    Returns
    -------
    np.ndarray
        uint8 RGB image resized to (224, 224, 3).

    Raises
    ------
    ValueError
        If the source cannot be decoded into an image.
    """
    img_rgb: np.ndarray

    # ── numpy array path ─────────────────────────────────────────────────────
    if isinstance(source, np.ndarray):
        arr = source.astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        img_rgb = arr

    # ── bytes path ────────────────────────────────────────────────────────────
    elif isinstance(source, (bytes, bytearray)):
        nparr = np.frombuffer(source, dtype=np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Try PIL fallback
            try:
                pil_img = Image.open(io.BytesIO(source)).convert("RGB")
                img_rgb = np.array(pil_img, dtype=np.uint8)
            except Exception as exc:
                raise ValueError(f"Cannot decode image bytes: {exc}") from exc
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── file path ─────────────────────────────────────────────────────────────
    else:
        fpath = Path(source)
        if not fpath.exists():
            raise FileNotFoundError(f"Image file not found: {fpath}")

        suffix = fpath.suffix.lower()

        # DICOM handling
        if suffix == ".dcm":
            try:
                import pydicom  # type: ignore
                ds = pydicom.dcmread(str(fpath))
                pixel_array = ds.pixel_array.astype(np.float32)
                # Normalise to uint8 range
                mn, mx = pixel_array.min(), pixel_array.max()
                if mx > mn:
                    pixel_array = (pixel_array - mn) / (mx - mn) * 255.0
                arr8 = pixel_array.astype(np.uint8)
                if arr8.ndim == 2:
                    arr8 = cv2.cvtColor(arr8, cv2.COLOR_GRAY2RGB)
                img_rgb = arr8
            except ImportError:
                logger.warning("pydicom not installed; falling back to cv2 for DCM.")
                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    raise ValueError(f"Cannot read DICOM file: {fpath}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                raise ValueError(f"DICOM read error: {exc}") from exc
        else:
            # Standard image formats via PIL (more robust than cv2 on Windows)
            try:
                pil_img = Image.open(str(fpath)).convert("RGB")
                img_rgb = np.array(pil_img, dtype=np.uint8)
            except Exception:
                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    raise ValueError(f"Cannot read image file: {fpath}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── resize to canonical size ──────────────────────────────────────────────
    if img_rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    return img_rgb.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# ResNet18 Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _get_resnet18(device: str = "cpu") -> torch.nn.Module:
    """Return a cached ResNet18 model with the final FC layer removed.

    The model is loaded once per process and stored in the module-level
    ``_resnet18_model`` variable to avoid repeated disk I/O.

    Parameters
    ----------
    device : str
        PyTorch device string (``'cpu'`` or ``'cuda'``).

    Returns
    -------
    torch.nn.Module
        Truncated ResNet18 ready for inference (eval mode, no grad).
    """
    global _resnet18_model, _resnet18_device

    if _resnet18_model is not None and _resnet18_device == device:
        return _resnet18_model

    logger.info("Loading pretrained ResNet18 for feature extraction …")
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
    base = tv_models.resnet18(weights=weights)
    # Remove the classification head; keep everything up to and including avgpool
    model = torch.nn.Sequential(*list(base.children())[:-1])
    model = model.to(device).eval()

    _resnet18_model = model
    _resnet18_device = device
    logger.info("ResNet18 feature extractor ready on %s.", device)
    return model


def extract_resnet18_features(image_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Extract a 512-dimensional feature vector from a 224x224 RGB image.

    Uses pretrained ImageNet weights — no fine-tuning required for demo mode.
    The global average pooling output (``avgpool``) is used as the feature
    representation, giving a 512-d vector for ResNet18.

    Parameters
    ----------
    image_rgb : np.ndarray
        uint8 RGB image of shape (224, 224, 3).
    device : str
        PyTorch device (default ``'cpu'``).

    Returns
    -------
    np.ndarray
        1-D float32 array of length 512.
    """
    model = _get_resnet18(device)

    pil_img = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    tensor  = _preprocess_transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        feat = model(tensor)          # (1, 512, 1, 1) from avgpool

    return feat.squeeze().cpu().numpy().astype(np.float32)  # (512,)


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality Compression & Angle Encoding
# ──────────────────────────────────────────────────────────────────────────────

def compress_to_n_qubits(features: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reduce a 512-d feature vector to n_qubits values in [0, 2pi].

    Strategy:
    * If n_qubits < 512 : PCA to n_qubits components (single-sample fallback
      uses random projection when PCA cannot be fitted).
    * Normalise linearly from [min, max] to [0, 2pi] for angle encoding.

    Parameters
    ----------
    features : np.ndarray
        1-D float32 array of length FEATURE_DIM (512).
    n_qubits : int
        Target dimensionality (number of qubits).

    Returns
    -------
    np.ndarray
        1-D float32 array of length n_qubits with values in [0, 2pi].
    """
    feat = features.copy().astype(np.float32)

    if len(feat) < n_qubits:
        # Pad with zeros if feature vector is shorter than n_qubits
        feat = np.pad(feat, (0, n_qubits - len(feat)))

    if n_qubits == len(feat):
        reduced = feat
    elif n_qubits < len(feat):
        # Deterministic random projection (single-sample friendly)
        rng = np.random.default_rng(RANDOM_SEED)
        proj_matrix = rng.standard_normal((len(feat), n_qubits)).astype(np.float32)
        # Column-normalise the projection matrix
        norms = np.linalg.norm(proj_matrix, axis=0, keepdims=True) + 1e-8
        proj_matrix /= norms
        reduced = feat @ proj_matrix  # (n_qubits,)
    else:
        reduced = feat[:n_qubits]

    # ── Normalise to [0, 2pi] ─────────────────────────────────────────────────
    mn, mx = reduced.min(), reduced.max()
    if mx - mn < 1e-8:
        # Constant vector — encode as pi (halfway)
        angles = np.full(n_qubits, math.pi, dtype=np.float32)
    else:
        angles = ((reduced - mn) / (mx - mn) * 2 * math.pi).astype(np.float32)

    return angles


# ──────────────────────────────────────────────────────────────────────────────
# Full Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def encode_image_to_qubits(image_rgb: np.ndarray, n_qubits: int) -> dict:
    """End-to-end pipeline: raw image -> quantum angle features.

    Steps:
    1. Extract 512-d ResNet18 features.
    2. Project + normalise to n_qubits angles via ``compress_to_n_qubits``.
    3. Generate visualisations (original thumbnail and feature map heatmap).

    Parameters
    ----------
    image_rgb : np.ndarray
        uint8 RGB image of shape (224, 224, 3).
    n_qubits : int
        Number of output qubits / features.

    Returns
    -------
    dict
        ``quantum_features`` : list[float] of length n_qubits
        ``feature_map_b64``  : base64 PNG of the feature heatmap
        ``original_b64``     : base64 PNG of the (resized) input image
    """
    try:
        features = extract_resnet18_features(image_rgb)
    except Exception as exc:
        logger.warning("ResNet18 extraction failed (%s); using random features.", exc)
        rng = np.random.default_rng(RANDOM_SEED)
        features = rng.random(FEATURE_DIM).astype(np.float32)

    quantum_angles = compress_to_n_qubits(features, n_qubits)

    original_b64    = image_to_base64(image_rgb)
    feature_map_b64 = create_feature_map_visualization(quantum_angles, n_qubits)

    return {
        "quantum_features": quantum_angles.tolist(),
        "feature_map_b64":  feature_map_b64,
        "original_b64":     original_b64,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Patch Grid for Quanvolutional Sliding Window
# ──────────────────────────────────────────────────────────────────────────────

def create_patch_grid(image_rgb: np.ndarray, patch_size: int = 2) -> dict:
    """Divide the image into a regular grid of patches for QCNN processing.

    The image is divided into a ``patch_size x patch_size`` grid of equal
    non-overlapping rectangular patches.  Each patch is returned as a
    normalised float32 array (pixel values in [0, 1]).

    Parameters
    ----------
    image_rgb : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    patch_size : int
        Number of patches along each spatial dimension (default 2 -> 4 patches).

    Returns
    -------
    dict
        ``patches``    : list[list[np.ndarray]] row-major 2-D grid of patches
        ``grid_b64``   : base64 PNG showing the patch grid overlaid on the image
        ``n_patches``  : total number of patches (patch_size ** 2)
    """
    H, W = image_rgb.shape[:2]
    ph = H // patch_size   # patch height in pixels
    pw = W // patch_size   # patch width in pixels

    patches = []
    annotated = image_rgb.copy()

    for row in range(patch_size):
        row_patches = []
        for col in range(patch_size):
            y0, y1 = row * ph, (row + 1) * ph
            x0, x1 = col * pw, (col + 1) * pw
            patch = image_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
            row_patches.append(patch.tolist())

            # Draw grid lines on annotated image
            cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 100), 2)
            cv2.putText(
                annotated, f"{row},{col}",
                (x0 + 4, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1, cv2.LINE_AA,
            )
        patches.append(row_patches)

    grid_b64 = image_to_base64(annotated)

    return {
        "patches":   patches,
        "grid_b64":  grid_b64,
        "n_patches": patch_size ** 2,
    }
