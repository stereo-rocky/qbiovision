"""
Smoke tests for every production API route.

Run against either dependency profile:

    pip install -r requirements.txt       # serverless profile (NumPy fallbacks)
    pip install -r requirements-full.txt  # full Qiskit / torch profile

    python tests/test_api_routes.py
"""

import io
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

# Exercise the serverless filesystem path (ephemeral /tmp cache seeded from the
# bundled cache) and keep generated benchmark JSON out of the working tree.
os.environ.setdefault("VERCEL", "1")

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402

client = TestClient(app_module.app)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")


def png_bytes(color=(180, 90, 90)):
    from PIL import Image

    img = Image.new("RGB", (96, 96), color)
    for x in range(0, 96, 7):          # add structure so descriptors differ
        for y in range(0, 96, 11):
            img.putpixel((x, y), (20, 200, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    print("\n== /api/health ==")
    r = client.get("/api/health")
    check("health 200", r.status_code == 200, str(r.status_code))
    print("   runtime:", json.dumps(r.json().get("runtime", {})))

    print("\n== /api/datasets/samples ==")
    r = client.get("/api/datasets/samples")
    check("datasets 200", r.status_code == 200, r.text[:160])
    check("datasets payload", len(r.json().get("datasets", [])) == 3)

    print("\n== /api/preprocess (upload) ==")
    r = client.post(
        "/api/preprocess",
        files={"file": ("scan.png", png_bytes(), "image/png")},
        data={"n_qubits": "4"},
    )
    check("preprocess 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        d = r.json()
        check("preprocess features", len(d["quantum_features"]) == 4)
        check("preprocess images", bool(d["original_b64"]) and bool(d["feature_map_b64"])
              and bool(d["patch_grid_b64"]))

    print("\n== /api/analyze (dataset preset) ==")
    r = client.post("/api/analyze", data={"dataset": "breakhis", "n_qubits": "4"})
    check("analyze 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        check("analyze success flag", r.json().get("success") is True)

    print("\n== /api/analyze (upload) ==")
    r = client.post(
        "/api/analyze",
        files={"file": ("biopsy.jpg", png_bytes((40, 60, 160)), "image/jpeg")},
        data={"n_qubits": "6"},
    )
    check("analyze upload 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        check("analyze upload qubits", len(r.json()["quantum_features"]) == 6)

    print("\n== /api/analyze (validation) ==")
    r = client.post("/api/analyze", data={})
    check("analyze rejects empty", r.status_code == 400, str(r.status_code))
    r = client.post("/api/analyze", files={"file": ("x.gif", b"GIF89a", "image/gif")})
    check("analyze rejects bad format", r.status_code == 400, str(r.status_code))

    print("\n== /api/circuit/build ==")
    for arch in ("VQC", "QCNN", "QSVC"):
        r = client.post("/api/circuit/build",
                        json={"architecture": arch, "n_qubits": 4, "n_layers": 2,
                              "entangler": "cx"})
        ok = r.status_code == 200
        check(f"circuit {arch} 200", ok, r.text[:200])
        if ok:
            d = r.json()
            check(f"circuit {arch} qasm", len(d["qasm"]) > 40)
            check(f"circuit {arch} diagram", len(d["svg_b64"]) > 500)

    print("\n== /api/model/train ==")
    r = client.post("/api/model/train",
                    json={"architecture": "VQC", "n_qubits": 4, "n_layers": 1,
                          "dataset_name": "breakhis", "n_epochs": 2})
    check("train VQC 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        check("train accuracy present", "test_accuracy" in r.json())

    r = client.post("/api/model/train",
                    json={"architecture": "QSVC", "n_qubits": 4, "dataset_name": "breakhis"})
    check("train QSVC 200", r.status_code == 200, r.text[:200])

    print("\n== /api/model/predict ==")
    r = client.post(
        "/api/model/predict",
        files={"file": ("scan.png", png_bytes(), "image/png")},
        data={"architecture": "QSVC", "n_qubits": "4", "dataset_name": "breakhis"},
    )
    check("predict 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        check("predict confidence", 0.0 <= r.json()["confidence"] <= 1.0)

    print("\n== /api/noise/simulate ==")
    r = client.post("/api/noise/simulate",
                    json={"architecture": "VQC", "n_qubits": 4, "t1_us": 50, "t2_us": 70,
                          "depolarizing_rate": 0.02, "use_zne": True, "use_trex": True,
                          "shots": 256})
    check("noise 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        d = r.json()
        ideal = d["ideal"]["expectation_value"]
        noisy = d["noisy"]["expectation_value"]
        check("noise ideal != noisy", abs(ideal - noisy) > 1e-9,
              f"ideal={ideal} noisy={noisy}")
        check("noise chart data", len(d["chart_data"]) >= 2)
        print(f"   ideal={ideal}  noisy={noisy}  zne={d['zne'].get('mitigated_expectation')} "
              f"trex={d['trex'].get('mitigated_expectation')}")

    print("\n== /api/benchmark/run ==")
    r = client.post("/api/benchmark/run",
                    json={"dataset_name": "ham10000", "architecture": "VQC",
                          "n_qubits": 4, "use_cache": False})
    check("benchmark 200", r.status_code == 200, r.text[:300])
    bench = r.json() if r.status_code == 200 else {}
    if bench:
        check("benchmark roc curves", len(bench.get("roc_curves", [])) == 3)
        check("benchmark classical auc",
              0.0 <= bench["classical"]["metrics"]["auc_roc"] <= 1.0)

    print("\n== /api/report/generate ==")
    r = client.post("/api/report/generate",
                    json={"benchmark_results": bench, "dataset_name": "ham10000",
                          "architecture": "VQC", "format": "md"})
    check("report md 200", r.status_code == 200, r.text[:200])
    r = client.post("/api/report/generate",
                    json={"benchmark_results": bench, "dataset_name": "ham10000",
                          "architecture": "VQC", "format": "pdf"})
    check("report pdf 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        import base64
        check("report pdf bytes",
              base64.b64decode(r.json()["pdf_b64"])[:4] == b"%PDF")

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
