"""
imaging_compat.py — OpenCV / Matplotlib / torchvision compatibility layer.

``opencv-python-headless`` (~225 MB), ``matplotlib`` (~170 MB) and
``torch`` + ``torchvision`` (several GB once the CUDA wheels are pulled in)
cannot be bundled into a Vercel Serverless Function (225 MB limit).

Q-BioVision only needs a handful of operations from them:

* colour conversion, decoding, resizing and annotation  (OpenCV)
* a colour-mapped heatmap of the quantum feature vector (Matplotlib)
* a fixed 512-d image descriptor                        (torchvision ResNet18)

This module provides those operations with the real libraries when they are
installed, and with Pillow + NumPy equivalents otherwise, so the API returns
the same payload shape in both environments.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

CV2_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:  # pragma: no cover - depends on the install profile
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore

try:  # pragma: no cover - depends on the install profile
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Decoding / conversion / resizing
# ──────────────────────────────────────────────────────────────────────────────

def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Coerce a (H, W), (H, W, 3) or (H, W, 4) uint8 array to RGB (H, W, 3)."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        if arr.shape[2] == 4:
            return arr[:, :, :3]
        return arr[:, :, :3]
    raise ValueError(f"Cannot interpret array of shape {arr.shape} as an image")


def decode_image_bytes(data: Union[bytes, bytearray]) -> np.ndarray:
    """Decode in-memory JPEG/PNG/BMP bytes into an RGB uint8 array."""
    if CV2_AVAILABLE:
        buf = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        pil = Image.open(io.BytesIO(bytes(data))).convert("RGB")
        return np.array(pil, dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - malformed uploads
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc


def read_image_file(path: Union[str, Path]) -> np.ndarray:
    """Read an image file from disk into an RGB uint8 array."""
    path = Path(path)
    try:
        pil = Image.open(str(path)).convert("RGB")
        return np.array(pil, dtype=np.uint8)
    except Exception:
        if CV2_AVAILABLE:
            bgr = cv2.imread(str(path))
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Cannot read image file: {path}")


def resize_rgb(image_rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize to ``(width, height)`` with high-quality downsampling."""
    width, height = size
    if image_rgb.shape[1] == width and image_rgb.shape[0] == height:
        return image_rgb
    if CV2_AVAILABLE:
        return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(ensure_rgb(image_rgb))
    return np.array(pil.resize((width, height), Image.LANCZOS), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Annotation (patch grid overlay)
# ──────────────────────────────────────────────────────────────────────────────

def annotate_patch_grid(image_rgb: np.ndarray, boxes, labels,
                        box_color=(0, 255, 100), text_color=(255, 255, 0)) -> np.ndarray:
    """Draw labelled rectangles over a copy of ``image_rgb``.

    ``boxes`` is a sequence of ``(x0, y0, x1, y1)`` tuples and ``labels`` the
    matching caption for each box.
    """
    annotated = ensure_rgb(image_rgb).copy()
    if CV2_AVAILABLE:
        for (x0, y0, x1, y1), text in zip(boxes, labels):
            cv2.rectangle(annotated, (x0, y0), (x1, y1), box_color, 2)
            cv2.putText(annotated, text, (x0 + 4, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        return annotated

    pil = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil)
    for (x0, y0, x1, y1), text in zip(boxes, labels):
        draw.rectangle([x0, y0, x1, y1], outline=tuple(box_color), width=2)
        draw.text((x0 + 4, y0 + 8), str(text), fill=tuple(text_color))
    return np.array(pil, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Feature-map heatmap
# ──────────────────────────────────────────────────────────────────────────────

# Sampled control points of the Matplotlib "viridis" colormap.
_VIRIDIS = np.array([
    [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141],
    [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 191, 112],
    [122, 209, 81], [189, 223, 38], [253, 231, 37],
], dtype=float)


def viridis(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB uint8 using a viridis approximation."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    pos = v * (len(_VIRIDIS) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS) - 1)
    frac = (pos - lo)[..., None]
    rgb = _VIRIDIS[lo] * (1 - frac) + _VIRIDIS[hi] * frac
    return rgb.astype(np.uint8)


def render_feature_heatmap(features: np.ndarray, n_qubits: int) -> str:
    """Render the quantum angle vector as a base64 PNG heatmap."""
    side = max(1, math.ceil(math.sqrt(n_qubits)))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[:n_qubits] = np.asarray(features, dtype=np.float32)[:n_qubits]
    grid = padded.reshape(side, side)

    if MATPLOTLIB_AVAILABLE:
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

    # ── Pillow fallback ──────────────────────────────────────────────────────
    cell = max(24, 160 // side)
    pad, bar_w, top = 14, 18, 26
    w = pad * 2 + side * cell + bar_w + 10
    h = top + side * cell + pad

    img = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(img)
    norm = np.clip(grid / (2 * math.pi), 0.0, 1.0)
    colors = viridis(norm)
    for r in range(side):
        for c in range(side):
            x0 = pad + c * cell
            y0 = top + r * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2],
                           fill=tuple(int(v) for v in colors[r, c]))

    # colour bar (0 at the bottom, 2π at the top)
    bar_x = pad + side * cell + 8
    bar_h = side * cell
    for i in range(bar_h):
        shade = viridis(np.array([1.0 - i / max(bar_h - 1, 1)]))[0]
        draw.line([(bar_x, top + i), (bar_x + bar_w, top + i)],
                  fill=tuple(int(v) for v in shade))
    draw.text((6, 8), f"Quantum Feature Map ({n_qubits} qubits)", fill=(226, 232, 240))
    draw.text((bar_x - 2, top - 12), "2π", fill=(148, 163, 184))
    draw.text((bar_x - 2, top + bar_h + 1), "0", fill=(148, 163, 184))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Torch-free image descriptor
# ──────────────────────────────────────────────────────────────────────────────

def classical_descriptor(image_rgb: np.ndarray, dim: int = 512) -> np.ndarray:
    """Deterministic hand-crafted image descriptor used when torch is absent.

    Concatenates per-channel intensity histograms, an 8x8 grid of block means
    and standard deviations, and Sobel-style gradient energy per block. The
    result is a fixed-length, image-dependent float32 vector that plays the
    same role as the ResNet18 penultimate activations in the encoding
    pipeline (it is *not* a learned representation, and the API reports which
    extractor produced it).
    """
    img = ensure_rgb(image_rgb).astype(np.float32) / 255.0
    h, w, _ = img.shape
    parts = []

    # 1. Per-channel histograms (3 x 32 bins)
    for ch in range(3):
        hist, _ = np.histogram(img[:, :, ch], bins=32, range=(0.0, 1.0))
        parts.append(hist.astype(np.float32) / max(hist.sum(), 1))

    # 2. Block means and standard deviations on an 8x8 grid (3 channels)
    gh, gw = 8, 8
    bh, bw = max(1, h // gh), max(1, w // gw)
    blocks = np.zeros((gh, gw, 3), dtype=np.float32)
    stds = np.zeros((gh, gw, 3), dtype=np.float32)
    for r in range(gh):
        for c in range(gw):
            tile = img[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            if tile.size:
                blocks[r, c] = tile.reshape(-1, 3).mean(axis=0)
                stds[r, c] = tile.reshape(-1, 3).std(axis=0)
    parts.append(blocks.reshape(-1))
    parts.append(stds.reshape(-1))

    # 3. Gradient energy per block (texture cue)
    gray = img.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    mag = gx + gy
    grad = np.zeros((gh, gw), dtype=np.float32)
    for r in range(gh):
        for c in range(gw):
            tile = mag[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            if tile.size:
                grad[r, c] = float(tile.mean())
    parts.append(grad.reshape(-1))

    vec = np.concatenate(parts).astype(np.float32)
    if len(vec) < dim:
        reps = int(np.ceil(dim / len(vec)))
        vec = np.tile(vec, reps)
    return vec[:dim].astype(np.float32)


__all__ = [
    "CV2_AVAILABLE",
    "MATPLOTLIB_AVAILABLE",
    "ensure_rgb",
    "decode_image_bytes",
    "read_image_file",
    "resize_rgb",
    "annotate_patch_grid",
    "render_feature_heatmap",
    "classical_descriptor",
    "viridis",
]
