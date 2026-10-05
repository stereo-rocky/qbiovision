"""
create_demo_data.py — Synthetic Demo Image Generator for Q-BioVision
Generates synthetic clinical images simulating histology, dermatoscopy, and chest X-ray patterns.
Run once before starting the backend: python create_demo_data.py
"""

import numpy as np
import os
from pathlib import Path

# Attempt PIL import (Pillow)
try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] Pillow not installed. Falling back to raw PNG writing.")

# Attempt OpenCV import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

DEMO_DATA_DIR = Path(__file__).parent / "demo_data"
DEMO_DATA_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
IMG_SIZE = 224


def save_as_png(arr_rgb: np.ndarray, filepath: Path):
    """Save a uint8 RGB numpy array as PNG."""
    arr_rgb = np.clip(arr_rgb, 0, 255).astype(np.uint8)
    if PIL_AVAILABLE:
        img = Image.fromarray(arr_rgb, mode="RGB")
        img.save(str(filepath))
    elif CV2_AVAILABLE:
        bgr = arr_rgb[:, :, ::-1]  # RGB -> BGR for OpenCV
        cv2.imwrite(str(filepath), bgr)
    else:
        # Minimal raw PNG writer using only stdlib
        import zlib, struct

        def write_png(filename, arr):
            height, width = arr.shape[:2]
            with open(filename, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")
                ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
                crc = zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF
                f.write(struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", crc))
                raw = b""
                for row in arr:
                    raw += b"\x00" + row.tobytes()
                compressed = zlib.compress(raw, 9)
                crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
                f.write(struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", crc))
                crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
                f.write(struct.pack(">I", 0) + b"IEND" + struct.pack(">I", crc))

        write_png(str(filepath), arr_rgb)

    print(f"  [OK] Saved: {filepath}")


def generate_breakhis_sample() -> np.ndarray:
    """
    Simulate a H&E-stained breast tissue histology image (BreakHis style).
    - Pinkish-purple hematoxylin/eosin background
    - Dark purple cell nuclei clusters
    - Irregular glandular structures
    """
    rng = np.random.RandomState(SEED)

    # Base tissue color: eosin pink
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img[:, :, 0] = rng.uniform(180, 220, (IMG_SIZE, IMG_SIZE))  # R - pink
    img[:, :, 1] = rng.uniform(120, 160, (IMG_SIZE, IMG_SIZE))  # G - less green
    img[:, :, 2] = rng.uniform(160, 200, (IMG_SIZE, IMG_SIZE))  # B - purple tint

    # Add cell nuclei (dark blue-purple ellipses)
    n_nuclei = rng.randint(30, 60)
    for _ in range(n_nuclei):
        cx = rng.randint(10, IMG_SIZE - 10)
        cy = rng.randint(10, IMG_SIZE - 10)
        rx = rng.randint(4, 10)
        ry = rng.randint(4, 10)
        y_grid, x_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask = ((x_grid - cx) ** 2 / rx**2 + (y_grid - cy) ** 2 / ry**2) <= 1
        img[mask, 0] = rng.uniform(60, 100)  # dark red
        img[mask, 1] = rng.uniform(40, 80)   # dark green
        img[mask, 2] = rng.uniform(120, 160) # blue-purple

    # Add fibrous stroma (linear structures)
    for _ in range(20):
        x1, y1 = rng.randint(0, IMG_SIZE, 2)
        x2, y2 = rng.randint(0, IMG_SIZE, 2)
        thickness = rng.randint(1, 3)
        color = np.array([rng.uniform(200, 240), rng.uniform(180, 220), rng.uniform(200, 240)])
        # Draw a simple line by interpolating
        n_pts = max(abs(x2 - x1), abs(y2 - y1)) * 2 + 1
        xs = np.linspace(x1, x2, int(n_pts)).astype(int)
        ys = np.linspace(y1, y2, int(n_pts)).astype(int)
        valid = (xs >= 0) & (xs < IMG_SIZE) & (ys >= 0) & (ys < IMG_SIZE)
        img[ys[valid], xs[valid]] = color

    # Add Gaussian noise (tissue texture)
    img += rng.normal(0, 8, img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_ham10000_sample() -> np.ndarray:
    """
    Simulate a dermatoscopic skin lesion image (HAM10000 style).
    - Skin-tone background with melanocytic pigmentation
    - Central dark lesion with irregular border
    - Hair artifacts and pigment network
    """
    rng = np.random.RandomState(SEED + 1)

    # Skin background: peach/beige
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img[:, :, 0] = rng.uniform(200, 230, (IMG_SIZE, IMG_SIZE))  # R - warm
    img[:, :, 1] = rng.uniform(160, 190, (IMG_SIZE, IMG_SIZE))  # G - mid
    img[:, :, 2] = rng.uniform(130, 160, (IMG_SIZE, IMG_SIZE))  # B - cool

    # Central melanocytic lesion
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    lesion_r = rng.randint(40, 70)
    y_grid, x_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]

    # Irregular border via radial noise
    angles = np.arctan2(y_grid - cy, x_grid - cx)
    radii = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    # Perturb the border
    border_noise = 1 + 0.3 * np.sin(4 * angles) + 0.15 * rng.uniform(-1, 1, (IMG_SIZE, IMG_SIZE))
    lesion_mask = radii < (lesion_r * border_noise)

    img[lesion_mask, 0] = rng.uniform(80, 120, lesion_mask.sum())
    img[lesion_mask, 1] = rng.uniform(50, 90, lesion_mask.sum())
    img[lesion_mask, 2] = rng.uniform(40, 70, lesion_mask.sum())

    # Pigment network (honeycomb pattern)
    for i in range(0, IMG_SIZE, 12):
        for j in range(0, IMG_SIZE, 12):
            if lesion_mask[min(i, IMG_SIZE - 1), min(j, IMG_SIZE - 1)]:
                yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                ring = (np.sqrt((xx - j) ** 2 + (yy - i) ** 2) > 4) & \
                       (np.sqrt((xx - j) ** 2 + (yy - i) ** 2) < 6)
                inner = lesion_mask & ring
                img[inner, 0] *= 0.6
                img[inner, 1] *= 0.6
                img[inner, 2] *= 0.6

    # Hair artifacts (thin dark lines)
    for _ in range(8):
        x1, y1 = rng.randint(0, IMG_SIZE, 2)
        angle = rng.uniform(0, np.pi)
        length = rng.randint(30, 80)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        n_pts = length * 3
        xs = np.linspace(x1, x2, n_pts).astype(int)
        ys = np.linspace(y1, y2, n_pts).astype(int)
        valid = (xs >= 0) & (xs < IMG_SIZE) & (ys >= 0) & (ys < IMG_SIZE)
        img[ys[valid], xs[valid]] = [20, 15, 10]  # near-black

    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_chestxr_sample() -> np.ndarray:
    """
    Simulate a chest X-ray image (grayscale converted to RGB).
    - Dark background (air)
    - Bright rib cage and vertebrae
    - Heart shadow
    - Lung fields with subtle texture
    """
    rng = np.random.RandomState(SEED + 2)

    # Start with a dark grayscale base
    gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Lung fields (slightly brighter than black, symmetric)
    y_grid, x_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    # Left lung
    left_lung = ((x_grid - (cx - 45)) ** 2 / 40**2 + (y_grid - cy) ** 2 / 65**2) < 1
    gray[left_lung] += rng.uniform(40, 60, left_lung.sum())

    # Right lung
    right_lung = ((x_grid - (cx + 45)) ** 2 / 38**2 + (y_grid - cy) ** 2 / 62**2) < 1
    gray[right_lung] += rng.uniform(40, 60, right_lung.sum())

    # Heart shadow (central, bright)
    heart = ((x_grid - (cx - 15)) ** 2 / 30**2 + (y_grid - (cy + 10)) ** 2 / 40**2) < 1
    gray[heart] = rng.uniform(140, 180, heart.sum())

    # Rib cage (horizontal bright bands)
    for i in range(6):
        rib_y = cy - 50 + i * 20
        rib_mask = (np.abs(y_grid - rib_y) < 2) & (x_grid > 20) & (x_grid < IMG_SIZE - 20)
        gray[rib_mask] = rng.uniform(180, 220, rib_mask.sum())

    # Vertebral column (central vertical stripe)
    spine = (np.abs(x_grid - cx) < 8) & (y_grid > 20) & (y_grid < IMG_SIZE - 20)
    gray[spine] = rng.uniform(150, 190, spine.sum())

    # Diaphragm (bottom bright curve)
    diaphragm = (np.abs(y_grid - (cy + 70)) < 5) & (x_grid > 10) & (x_grid < IMG_SIZE - 10)
    gray[diaphragm] = rng.uniform(160, 200, diaphragm.sum())

    # Add texture noise
    gray += rng.normal(0, 10, gray.shape)
    gray = np.clip(gray, 0, 255)

    # Convert grayscale to RGB
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    return rgb


def main():
    """Generate all three demo samples and save to demo_data/ directory."""
    print("=" * 60)
    print("Q-BioVision Demo Data Generator")
    print("=" * 60)
    print(f"Output directory: {DEMO_DATA_DIR.absolute()}")
    print()

    samples = [
        ("breakhis_sample.png",  generate_breakhis_sample,  "BreakHis histology (H&E stain)"),
        ("ham10000_sample.png",  generate_ham10000_sample,  "HAM10000 dermatoscopy"),
        ("chestxr_sample.png",   generate_chestxr_sample,   "Chest X-ray"),
    ]

    for filename, generator_fn, description in samples:
        print(f"Generating {description}...")
        try:
            img_arr = generator_fn()
            filepath = DEMO_DATA_DIR / filename
            save_as_png(img_arr, filepath)
            print(f"  Shape: {img_arr.shape}, dtype: {img_arr.dtype}")
        except Exception as e:
            print(f"  [ERROR] Failed to generate {filename}: {e}")

    print()
    print("Demo data generation complete!")
    print(f"Files saved to: {DEMO_DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
