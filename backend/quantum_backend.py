"""
quantum_backend.py — Single import point for the quantum primitives.

Q-BioVision runs in two very different environments:

* **Local / full install** (``requirements-full.txt``) — the genuine Qiskit
  1.x + Qiskit Aer stack is available and is always preferred.
* **Serverless deployment** (``requirements.txt``) — Qiskit transitively
  requires SciPy, and ``qiskit + qiskit-aer + scipy`` unpack to >500 MB,
  which cannot fit inside a Vercel Function (225 MB limit). In that case we
  fall back to :mod:`qlite`, a pure-NumPy implementation of the subset of
  Qiskit that this project uses.

Importing from this module keeps ``quantum_models.py``, ``noise_mitigation.py``
and ``app.py`` identical in both environments.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: ``True`` when the real Qiskit package is installed.
QISKIT_AVAILABLE = False
#: ``True`` when the real ``qiskit-aer`` simulator is installed.
AER_AVAILABLE = False
#: ``True`` when ``qiskit-machine-learning`` is installed.
QML_AVAILABLE = False
#: Human-readable name of the active backend, surfaced through /api/health.
BACKEND_NAME = "qlite-numpy"

try:  # ── Preferred: real Qiskit ────────────────────────────────────────────
    from qiskit import QuantumCircuit, transpile  # type: ignore
    from qiskit.circuit import Parameter, ParameterVector  # type: ignore
    from qiskit.circuit.library import (  # type: ignore
        ZZFeatureMap,
        RealAmplitudes,
        TwoLocal,
    )

    QISKIT_AVAILABLE = True
    BACKEND_NAME = "qiskit"
except ImportError:  # pragma: no cover - exercised only on slim deployments
    from qlite import (  # type: ignore
        QuantumCircuit,
        transpile,
        Parameter,
        ParameterVector,
        ZZFeatureMap,
        RealAmplitudes,
        TwoLocal,
    )

    logger.info("Qiskit not installed — using the pure-NumPy qlite backend.")


# ── Simulator + noise model ───────────────────────────────────────────────────
if QISKIT_AVAILABLE:
    try:
        from qiskit_aer import AerSimulator  # type: ignore
        from qiskit_aer.noise import (  # type: ignore
            NoiseModel,
            ReadoutError,
            thermal_relaxation_error,
            depolarizing_error,
        )

        AER_AVAILABLE = True
        BACKEND_NAME = "qiskit+aer"
    except ImportError:  # pragma: no cover
        from qlite import (  # type: ignore
            AerSimulator,
            NoiseModel,
            ReadoutError,
            thermal_relaxation_error,
            depolarizing_error,
        )

        logger.warning("qiskit-aer not installed — simulating with the qlite backend.")
else:
    from qlite import (  # type: ignore
        AerSimulator,
        NoiseModel,
        ReadoutError,
        thermal_relaxation_error,
        depolarizing_error,
    )

try:
    from qiskit_machine_learning.kernels import FidelityQuantumKernel  # type: ignore

    QML_AVAILABLE = True
except ImportError:  # pragma: no cover
    FidelityQuantumKernel = None  # type: ignore


#: Every deployment can simulate circuits — either through Aer or through qlite.
SIMULATOR_AVAILABLE = True


def qasm_loads(text: str) -> "QuantumCircuit":
    """Parse an OpenQASM 2 program. Only available with the real Qiskit stack."""
    if QISKIT_AVAILABLE:
        from qiskit.qasm2 import loads  # type: ignore

        return loads(text)
    raise NotImplementedError("OpenQASM parsing requires the full Qiskit install.")


def circuit_png_b64(circuit) -> str:
    """Render ``circuit`` to a base64 PNG, using whichever drawer is available."""
    drawer = getattr(circuit, "png_b64", None)
    if drawer is not None:          # qlite circuits carry a Pillow renderer
        return drawer()
    try:                            # real Qiskit + matplotlib
        import io
        import base64

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = circuit.draw(
            output="mpl",
            style={
                "backgroundcolor": "#0f172a",
                "textcolor": "white",
                "gatefacecolor": "#1e40af",
                "gatetextcolor": "white",
                "subfontsize": 10,
            },
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor="#0f172a")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as exc:  # pragma: no cover - drawing is best-effort
        logger.warning("Circuit drawing failed: %s", exc)
        return ""


def backend_info() -> dict:
    """Summary of the active quantum backend (used by the health endpoint)."""
    return {
        "backend": BACKEND_NAME,
        "qiskit": QISKIT_AVAILABLE,
        "qiskit_aer": AER_AVAILABLE,
        "qiskit_machine_learning": QML_AVAILABLE,
    }


__all__ = [
    "QuantumCircuit",
    "transpile",
    "Parameter",
    "ParameterVector",
    "ZZFeatureMap",
    "RealAmplitudes",
    "TwoLocal",
    "AerSimulator",
    "NoiseModel",
    "ReadoutError",
    "thermal_relaxation_error",
    "depolarizing_error",
    "FidelityQuantumKernel",
    "QISKIT_AVAILABLE",
    "AER_AVAILABLE",
    "QML_AVAILABLE",
    "SIMULATOR_AVAILABLE",
    "BACKEND_NAME",
    "qasm_loads",
    "circuit_png_b64",
    "backend_info",
]
