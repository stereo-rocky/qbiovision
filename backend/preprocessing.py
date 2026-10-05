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
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image  # type: ignore

import imaging_compat
from imaging_compat import (
    annotate_patch_grid,
    classical_descriptor,
    decode_image_bytes,
    ensure_rgb,
    read_image_file,
    render_feature_heatmap,
    resize_rgb,
)

# PyTorch / torchvision are optional: they are required for the ResNet18
# extractor but cannot be bundled into a serverless function (see
# requirements-full.txt). When absent we fall back to a deterministic
# hand-crafted descriptor of the same dimensionality.
try:  # pragma: no cover - depends on the install profile
    import torch
    import torchvision.models as tv_models        # type: ignore
    import torchvision.transforms as transforms    # type: ignore

    TORCH_AVAILABLE = True
except ImportError:
    torch = None          # type: ignore
    tv_models = None      # type: ignore
    transforms = None     # type: ignore
    TORCH_AVAILABLE = False

from config import RANDOM_SEED, IMAGE_SIZE, FEATURE_DIM

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level model cache (avoid reloading on every request)
# ──────────────────────────────────────────────────────────────────────────────
_resnet18_model = None
_resnet18_device: str = "cpu"
_preprocess_transform = None

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _get_transform():
    """Build (and cache) the torchvision preprocessing transform."""
    global _preprocess_transform
    if _preprocess_transform is None:
        _preprocess_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    return _preprocess_transform


#: Name of the feature extractor active in this deployment.
FEATURE_EXTRACTOR = "resnet18" if TORCH_AVAILABLE else "classical-descriptor"


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
    return render_feature_heatmap(features, n_qubits)


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
        img_rgb = ensure_rgb(source)

    # ── bytes path ────────────────────────────────────────────────────────────
    elif isinstance(source, (bytes, bytearray)):
        img_rgb = decode_image_bytes(source)

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
                img_rgb = ensure_rgb(pixel_array.astype(np.uint8))
            except ImportError:
                logger.warning("pydicom not installed; falling back to a generic decoder.")
                img_rgb = read_image_file(fpath)
            except Exception as exc:
                raise ValueError(f"DICOM read error: {exc}") from exc
        else:
            # Standard image formats (Pillow first, OpenCV as a backstop)
            img_rgb = read_image_file(fpath)

    # ── resize to canonical size ──────────────────────────────────────────────
    if img_rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img_rgb = resize_rgb(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    return img_rgb.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# ResNet18 Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_writable_torch_hub_cache() -> None:
    """Ensure torch.hub's weight cache points at a writable directory.

    On read-only filesystems (e.g. Vercel serverless, where only /tmp is
    writable) the default ``~/.cache/torch/hub`` cannot be created and
    torchvision fails with a PermissionError while downloading the
    ResNet18 weights. Redirect the cache to the temp directory in that
    case; local development is unaffected.
    """
    hub_dir = Path(torch.hub.get_dir())
    try:
        hub_dir.mkdir(parents=True, exist_ok=True)
        probe = hub_dir / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "torch" / "hub"
        fallback.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(fallback))
        logger.info("torch.hub cache redirected to %s (default not writable)", fallback)


def _get_resnet18(device: str = "cpu"):
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

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch/torchvision are not installed in this deployment "
            "(see requirements-full.txt for the full ML profile)."
        )

    if _resnet18_model is not None and _resnet18_device == device:
        return _resnet18_model

    logger.info("Loading pretrained ResNet18 for feature extraction …")
    _ensure_writable_torch_hub_cache()
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
    tensor  = _get_transform()(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

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
    extractor = FEATURE_EXTRACTOR
    if TORCH_AVAILABLE:
        try:
            features = extract_resnet18_features(image_rgb)
        except Exception as exc:
            logger.warning(
                "ResNet18 extraction failed (%s); using the classical descriptor.", exc
            )
            features = classical_descriptor(image_rgb, dim=FEATURE_DIM)
            extractor = "classical-descriptor"
    else:
        # Serverless profile: torch/torchvision are not bundled. The
        # deterministic hand-crafted descriptor keeps the encoding pipeline
        # image-dependent and reproducible without the 5 GB CUDA stack.
        features = classical_descriptor(image_rgb, dim=FEATURE_DIM)

    quantum_angles = compress_to_n_qubits(features, n_qubits)

    original_b64    = image_to_base64(image_rgb)
    feature_map_b64 = create_feature_map_visualization(quantum_angles, n_qubits)

    return {
        "quantum_features": quantum_angles.tolist(),
        "feature_map_b64":  feature_map_b64,
        "original_b64":     original_b64,
        "feature_extractor": extractor,
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
    boxes, labels = [], []

    for row in range(patch_size):
        row_patches = []
        for col in range(patch_size):
            y0, y1 = row * ph, (row + 1) * ph
            x0, x1 = col * pw, (col + 1) * pw
            patch = image_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
            row_patches.append(patch.tolist())

            boxes.append((x0, y0, x1 - 1, y1 - 1))
            labels.append(f"{row},{col}")
        patches.append(row_patches)

    annotated = annotate_patch_grid(image_rgb, boxes, labels)
    grid_b64 = image_to_base64(annotated)

    return {
        "patches":   patches,
        "grid_b64":  grid_b64,
        "n_patches": patch_size ** 2,
    }
