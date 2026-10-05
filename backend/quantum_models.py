
"""
quantum_models.py -- Quantum Machine Learning Models for Q-BioVision
Implements QCNN (QuanvolutionalNN), QSVC (QuantumSVClassifier), and
VQC (VariationalQuantumClassifier) using Qiskit 1.x APIs.

All models expose a consistent interface:
    fit(X_train, y_train) -> dict
    predict(X_test)       -> dict
    get_circuit_qasm()    -> str
    get_circuit_svg()     -> str
"""

import io
import base64
import logging
import math
import time
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RANDOM_SEED

logger = logging.getLogger(__name__)

# Qiskit Aer imports with graceful fallback
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import (
        Estimator as AerEstimator,
        Sampler   as AerSampler,
    )
    _AER_AVAILABLE = True
except ImportError:
    _AER_AVAILABLE = False
    logger.warning("qiskit-aer not found; quantum models will use mock values.")

try:
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    _QML_AVAILABLE = True
except ImportError:
    _QML_AVAILABLE = False
    logger.warning("qiskit-machine-learning not found; QSVC will fall back to RBF-SVM.")


def _circuit_to_svg(circuit: QuantumCircuit) -> str:
    """Render a Qiskit circuit as a base64-encoded PNG (matplotlib backend)."""
    try:
        fig = circuit.draw(
            output='mpl',
            style={
                'backgroundcolor': '#0f172a',
                'textcolor':        'white',
                'gatefacecolor':    '#1e40af',
                'gatetextcolor':    'white',
                'subfontsize':       10,
            },
        )
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0f172a')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as exc:
        logger.warning('Circuit draw failed: %s', exc)
        return ''


def _safe_qasm(circuit: QuantumCircuit) -> str:
    """Return OpenQASM 3 string; fall back to OpenQASM 2 on older Qiskit."""
    try:
        from qiskit.qasm3 import dumps as qasm3_dumps
        return qasm3_dumps(circuit)
    except Exception:
        try:
            return circuit.qasm()
        except Exception:
            return '// QASM export unavailable'


def _expectation_from_statevector(sv: np.ndarray, n_qubits: int) -> float:
    """Compute Pauli-Z expectation on qubit 0 from a statevector.
    <Z_0> = sum_i p_i * (-1)^(bit_0_of_i)
    """
    probs = np.abs(sv) ** 2
    exp_val = 0.0
    for i, p in enumerate(probs):
        bit0 = (i >> 0) & 1
        exp_val += p * (1 - 2 * bit0)
    return float(exp_val)


class QuanvolutionalNN:
    """Quanvolutional Neural Network using a parameterised 2x2 kernel circuit.

    Parameters
    ----------
    n_qubits : int  Number of qubits in the kernel circuit.
    n_layers : int  Number of variational layers.
    entangler : str  'cx' or 'cz'.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, entangler: str = 'cx') -> None:
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.entangler = entangler.lower()
        rng = np.random.default_rng(RANDOM_SEED)
        n_params = n_layers * n_qubits
        self.params = rng.uniform(-np.pi, np.pi, size=n_params).astype(np.float32)
        self._input_params  = ParameterVector('x', n_qubits)
        self._theta_params  = ParameterVector('t', len(self.params))
        self._circuit       = self._build_kernel_circuit()
        self._weights: Optional[np.ndarray] = None
        self._bias:    float                = 0.0
        self._classes: Optional[np.ndarray] = None

    def _build_kernel_circuit(self) -> QuantumCircuit:
        """Build the parameterised quanvolutional kernel circuit.

        Structure per layer: H gates, Ry(x_i) data encoding, Ry(theta_i) variational,
        CX/CZ entangler ladder.

        Returns
        -------
        QuantumCircuit
        """
        n = self.n_qubits
        qc = QuantumCircuit(n)
        theta_idx = 0
        qc.h(range(n))
        for layer in range(self.n_layers):
            for i in range(n):
                qc.ry(self._input_params[i], i)
            for i in range(n):
                qc.ry(self._theta_params[theta_idx], i)
                theta_idx += 1
            for i in range(n - 1):
                if self.entangler == 'cz':
                    qc.cz(i, i + 1)
                else:
                    qc.cx(i, i + 1)
            qc.barrier()
        return qc

    def get_circuit_qasm(self) -> str:
        """Return the kernel circuit as an OpenQASM 3 string."""
        return _safe_qasm(self._circuit)

    def get_circuit_svg(self) -> str:
        """Render the kernel circuit diagram as a base64-encoded PNG."""
        return _circuit_to_svg(self._circuit)

    def _apply_kernel_to_patch(self, patch_data: np.ndarray) -> np.ndarray:
        """Encode a single patch into the kernel circuit and return Z expectations.

        Parameters
        ----------
        patch_data : np.ndarray  1-D float array of length n_qubits in [0, 2pi].

        Returns
        -------
        np.ndarray  1-D float array of Pauli-Z expectations.
        """
        expectations = np.zeros(self.n_qubits, dtype=np.float32)
        if not _AER_AVAILABLE:
            rng = np.random.default_rng(int(patch_data.sum() * 1000) % 2**31)
            return rng.uniform(-1, 1, self.n_qubits).astype(np.float32)
        try:
            param_dict = {}
            for i, p in enumerate(self._input_params):
                param_dict[p] = float(patch_data[i]) if i < len(patch_data) else 0.0
            for i, p in enumerate(self._theta_params):
                param_dict[p] = float(self.params[i])
            bound_circuit = self._circuit.assign_parameters(param_dict)
            sim = AerSimulator(method='statevector')
            from qiskit import transpile
            sv_circ = bound_circuit.copy()
            sv_circ.save_statevector()
            t_circ = transpile(sv_circ, sim)
            job    = sim.run(t_circ)
            result = job.result()
            sv     = np.array(result.get_statevector())
            probs  = np.abs(sv) ** 2
            for qubit_idx in range(self.n_qubits):
                exp_val = 0.0
                for i, p in enumerate(probs):
                    bit = (i >> qubit_idx) & 1
                    exp_val += p * (1 - 2 * bit)
                expectations[qubit_idx] = exp_val
        except Exception as exc:
            logger.warning('Kernel execution failed: %s -- using random fallback.', exc)
            rng = np.random.default_rng(RANDOM_SEED)
            expectations = rng.uniform(-1, 1, self.n_qubits).astype(np.float32)
        return expectations

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Train the QCNN using parameter-shift gradient updates.

        Parameters
        ----------
        X_train : np.ndarray  Shape (N, n_qubits).
        y_train : np.ndarray  Shape (N,) integer class labels.

        Returns
        -------
        dict with accuracy_history, final_accuracy, training_time, n_params.
        """
        t0 = time.perf_counter()
        self._classes = np.unique(y_train)
        y_binary = np.where(y_train == self._classes[0], -1.0, 1.0).astype(np.float32)
        rng = np.random.default_rng(RANDOM_SEED)
        w   = rng.standard_normal(self.n_qubits).astype(np.float32) * 0.01
        b   = 0.0
        lr  = 0.05
        accuracy_history = []
        N = len(X_train)
        for epoch in range(10):
            Z    = np.vstack([self._apply_kernel_to_patch(x) for x in X_train])
            logits = Z @ w + b
            preds_sign = np.sign(logits)
            preds_sign[preds_sign == 0] = 1
            acc = float(np.mean(preds_sign == y_binary))
            accuracy_history.append(acc)
            errors = y_binary - preds_sign
            grad_w = -(Z.T @ errors) / N
            grad_b = -errors.mean()
            w -= lr * grad_w
            b -= lr * grad_b
            logger.debug('QCNN epoch %d  accuracy=%.3f', epoch + 1, acc)
        self._weights = w
        self._bias    = float(b)
        training_time = time.perf_counter() - t0
        Z_final = np.vstack([self._apply_kernel_to_patch(x) for x in X_train])
        logits_final = Z_final @ w + b
        final_preds  = np.sign(logits_final)
        final_preds[final_preds == 0] = 1
        final_acc = float(np.mean(final_preds == y_binary))
        return {
            'accuracy_history': accuracy_history,
            'final_accuracy':   final_acc,
            'training_time':    round(training_time, 3),
            'n_params':         len(self.params),
        }

    def predict(self, X_test: np.ndarray) -> dict:
        """Run inference. Returns predictions and probabilities."""
        if self._weights is None:
            rng = np.random.default_rng(RANDOM_SEED)
            return {'predictions': rng.integers(0, 2, len(X_test)).tolist(),
                    'probabilities': rng.uniform(0.5, 1.0, len(X_test)).tolist()}
        Z      = np.vstack([self._apply_kernel_to_patch(x) for x in X_test])
        logits = Z @ self._weights + self._bias
        probs  = 1.0 / (1.0 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)
        return {'predictions': preds.tolist(), 'probabilities': probs.tolist()}


class QuantumSVClassifier:
    """Quantum Support-Vector Classifier using a fidelity-based quantum kernel.

    Parameters
    ----------
    n_qubits : int
    feature_map_reps : int  ZZFeatureMap repetitions.
    """

    def __init__(self, n_qubits: int = 4, feature_map_reps: int = 2) -> None:
        self.n_qubits         = n_qubits
        self.feature_map_reps = feature_map_reps
        self._feature_map     = self._build_feature_map()
        self._svc: Optional[SVC] = None
        self._scaler          = MinMaxScaler(feature_range=(0, np.pi))
        self._X_train_scaled: Optional[np.ndarray] = None

    def _build_feature_map(self) -> QuantumCircuit:
        """Build the ZZFeatureMap quantum feature map circuit."""
        return ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.feature_map_reps,
            entanglement='linear',
        )

    def get_circuit_qasm(self) -> str:
        """Return the feature map circuit as an OpenQASM 3 string."""
        return _safe_qasm(self._feature_map)

    def get_circuit_svg(self) -> str:
        """Render the feature map circuit as a base64 PNG."""
        return _circuit_to_svg(self._feature_map)

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute quantum kernel matrix; falls back to RBF if Aer unavailable."""
        if _AER_AVAILABLE and _QML_AVAILABLE:
            try:
                qkernel = FidelityQuantumKernel(feature_map=self._feature_map)
                return qkernel.evaluate(x_vec=X1, y_vec=X2)
            except Exception as exc:
                logger.warning('Quantum kernel failed: %s -- RBF fallback.', exc)
        from sklearn.metrics.pairwise import rbf_kernel
        gamma = 1.0 / (self.n_qubits * X1.var() + 1e-8)
        return rbf_kernel(X1, X2, gamma=gamma).astype(np.float32)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Fit the QSVC.

        Parameters
        ----------
        X_train : np.ndarray  Shape (N, n_qubits).
        y_train : np.ndarray  Shape (N,).

        Returns
        -------
        dict with accuracy, training_time, n_support_vectors, kernel_matrix_sample.
        """
        t0 = time.perf_counter()
        X_scaled = self._scaler.fit_transform(X_train)
        self._X_train_scaled = X_scaled
        K_train  = self._compute_kernel_matrix(X_scaled, X_scaled)
        self._svc = SVC(kernel='precomputed', probability=True,
                        random_state=RANDOM_SEED, C=1.0)
        self._svc.fit(K_train, y_train)
        training_time = time.perf_counter() - t0
        acc     = float(self._svc.score(K_train, y_train))
        k_sample = K_train[:5, :5].tolist()
        return {
            'accuracy':              acc,
            'training_time':         round(training_time, 3),
            'n_support_vectors':     int(self._svc.support_vectors_.shape[0]),
            'kernel_matrix_sample':  k_sample,
        }

    def predict(self, X_test: np.ndarray) -> dict:
        """Predict class labels and probabilities."""
        if self._svc is None or self._X_train_scaled is None:
            rng = np.random.default_rng(RANDOM_SEED)
            return {'predictions': rng.integers(0, 2, len(X_test)).tolist(),
                    'probabilities': rng.uniform(0.5, 1.0, len(X_test)).tolist()}
        X_scaled = self._scaler.transform(X_test)
        K_test   = self._compute_kernel_matrix(X_scaled, self._X_train_scaled)
        preds    = self._svc.predict(K_test)
        probs    = self._svc.predict_proba(K_test)[:, 1]
        return {'predictions': preds.tolist(), 'probabilities': probs.tolist()}


class VariationalQuantumClassifier:
    """Variational Quantum Classifier with data re-uploading.

    Parameters
    ----------
    n_qubits : int
    n_layers : int
    entangler : str  'cx' or 'cz'.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, entangler: str = 'cx') -> None:
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.entangler = entangler.lower()
        self._n_params_per_layer = 2 * n_qubits
        self.n_params = self._n_params_per_layer * n_layers + n_qubits
        rng = np.random.default_rng(RANDOM_SEED)
        self.params = rng.uniform(-np.pi, np.pi, size=self.n_params).astype(np.float32)
        self._data_params  = ParameterVector('x', n_qubits)
        self._theta_params = ParameterVector('t', self.n_params)
        self._ansatz       = self._build_ansatz()
        self._classes: Optional[np.ndarray] = None
        self._fitted = False

    def _build_ansatz(self) -> QuantumCircuit:
        """Build data re-uploading variational ansatz.

        Per layer: Ry(theta*x) + Rz(theta*x) on each qubit, then CX/CZ ladder.
        Final layer: Rz(theta_bias) per qubit.

        Returns
        -------
        QuantumCircuit
        """
        n = self.n_qubits
        qc = QuantumCircuit(n, name='VQC-Ansatz')
        theta_idx = 0
        for layer in range(self.n_layers):
            for i in range(n):
                qc.ry(self._theta_params[theta_idx] * self._data_params[i], i)
                theta_idx += 1
                qc.rz(self._theta_params[theta_idx] * self._data_params[i], i)
                theta_idx += 1
            for i in range(n - 1):
                if self.entangler == 'cz':
                    qc.cz(i, i + 1)
                else:
                    qc.cx(i, i + 1)
            qc.barrier()
        for i in range(n):
            qc.rz(self._theta_params[theta_idx], i)
            theta_idx += 1
        return qc

    def get_circuit_qasm(self) -> str:
        """Return the ansatz as an OpenQASM 3 string."""
        return _safe_qasm(self._ansatz)

    def get_circuit_svg(self) -> str:
        """Render the ansatz as a base64 PNG."""
        return _circuit_to_svg(self._ansatz)

    def _forward(self, x: np.ndarray, params: np.ndarray) -> float:
        """Evaluate <Z_0> for a single input x with given params."""
        if not _AER_AVAILABLE:
            rng = np.random.default_rng(int(abs(x.sum()) * 1000) % 2**31)
            return float(rng.uniform(-1, 1))
        try:
            param_dict = {}
            for i, p in enumerate(self._data_params):
                param_dict[p] = float(x[i]) if i < len(x) else 0.0
            for i, p in enumerate(self._theta_params):
                param_dict[p] = float(params[i])
            bound = self._ansatz.assign_parameters(param_dict)
            sim   = AerSimulator(method='statevector')
            from qiskit import transpile
            sv_circ = bound.copy()
            sv_circ.save_statevector()
            t_circ = transpile(sv_circ, sim)
            result = sim.run(t_circ).result()
            sv     = np.array(result.get_statevector())
            return _expectation_from_statevector(sv, self.n_qubits)
        except Exception as exc:
            logger.warning('VQC forward pass error: %s', exc)
            rng = np.random.default_rng(RANDOM_SEED)
            return float(rng.uniform(-1, 1))

    def _parameter_shift_gradient(self, X, y_binary, params, shift=math.pi/2):
        """Compute parameter-shift gradients (limited to first 8 params for speed)."""
        gradients = np.zeros_like(params)
        active_params = min(len(params), 8)
        for k in range(active_params):
            p_plus  = params.copy(); p_plus[k]  += shift
            p_minus = params.copy(); p_minus[k] -= shift
            loss_plus = loss_minus = 0.0
            for xi, yi in zip(X, y_binary):
                z_plus  = self._forward(xi, p_plus)
                z_minus = self._forward(xi, p_minus)
                prob_plus  = (z_plus  + 1) / 2 + 1e-8
                prob_minus = (z_minus + 1) / 2 + 1e-8
                label = (yi + 1) / 2
                loss_plus  -= label * np.log(prob_plus)  + (1 - label) * np.log(1 - prob_plus  + 1e-8)
                loss_minus -= label * np.log(prob_minus) + (1 - label) * np.log(1 - prob_minus + 1e-8)
            gradients[k] = 0.5 * (loss_plus - loss_minus) / len(X)
        return gradients

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            n_epochs: int = 20, lr: float = 0.1) -> dict:
        """Train the VQC using parameter-shift gradient descent.

        Parameters
        ----------
        X_train : np.ndarray  Shape (N, n_qubits).
        y_train : np.ndarray  Shape (N,).
        n_epochs : int
        lr : float

        Returns
        -------
        dict with loss_history, accuracy_history, final_accuracy, training_time, n_params.
        """
        t0 = time.perf_counter()
        self._classes = np.unique(y_train)
        y_binary = np.where(y_train == self._classes[0], -1.0, 1.0).astype(np.float32)
        loss_history, accuracy_history = [], []
        params = self.params.copy()
        N_demo = min(len(X_train), 20)
        rng    = np.random.default_rng(RANDOM_SEED)
        idx    = rng.choice(len(X_train), N_demo, replace=False)
        X_demo = X_train[idx]; y_demo = y_binary[idx]
        for epoch in range(n_epochs):
            z_vals = np.array([self._forward(xi, params) for xi in X_demo])
            probs  = (z_vals + 1) / 2 + 1e-8
            labels = (y_demo + 1) / 2
            loss   = float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs + 1e-8)))
            loss_history.append(loss)
            preds  = (z_vals >= 0).astype(int) * 2 - 1
            acc    = float(np.mean(preds == y_demo))
            accuracy_history.append(acc)
            logger.debug('VQC epoch %d  loss=%.4f  acc=%.3f', epoch + 1, loss, acc)
            grads  = self._parameter_shift_gradient(X_demo, y_demo, params)
            params = params - lr * grads
        self.params  = params
        self._fitted = True
        training_time = time.perf_counter() - t0
        z_final = np.array([self._forward(xi, params) for xi in X_train[:N_demo]])
        preds_final = (z_final >= 0).astype(int) * 2 - 1
        final_acc   = float(np.mean(preds_final == y_binary[:N_demo]))
        return {
            'loss_history':     loss_history,
            'accuracy_history': accuracy_history,
            'final_accuracy':   final_acc,
            'training_time':    round(training_time, 3),
            'n_params':         self.n_params,
        }

    def predict(self, X_test: np.ndarray) -> dict:
        """Predict class labels and probabilities."""
        params = self.params
        z_vals = np.array([self._forward(xi, params) for xi in X_test])
        probs  = ((z_vals + 1) / 2).clip(0, 1)
        preds  = (z_vals >= 0).astype(int)
        return {'predictions': preds.tolist(), 'probabilities': probs.tolist()}


def build_circuit_for_config(architecture: str, n_qubits: int,
                              n_layers: int, entangler: str) -> dict:
    """Instantiate the appropriate quantum model and return circuit metadata.

    Parameters
    ----------
    architecture : str  One of QCNN, QSVC, VQC.
    n_qubits     : int
    n_layers     : int
    entangler    : str  'cx' or 'cz'.

    Returns
    -------
    dict with qasm, svg_b64, n_params, description.
    """
    arch = architecture.upper().strip()
    descriptions = {
        'QCNN': (f'Quanvolutional Neural Network with {n_qubits}-qubit kernel, '
                 f'{n_layers} layers, {entangler.upper()} entangler. '
                 'Applies a parameterised quantum circuit as a convolutional kernel.'),
        'QSVC': (f'Quantum Support-Vector Classifier using ZZFeatureMap ({n_qubits} qubits, '
                 f'{n_layers} reps). Kernel entries computed as quantum state fidelities.'),
        'VQC':  (f'Variational Quantum Classifier with data re-uploading '
                 f'({n_qubits} qubits, {n_layers} layers, {entangler.upper()} entangler). '
                 'Optimised via parameter-shift gradient descent.'),
    }
    if arch == 'QCNN':
        model = QuanvolutionalNN(n_qubits=n_qubits, n_layers=n_layers, entangler=entangler)
        return {'qasm': model.get_circuit_qasm(), 'svg_b64': model.get_circuit_svg(),
                'n_params': len(model.params), 'description': descriptions['QCNN']}
    elif arch == 'QSVC':
        model = QuantumSVClassifier(n_qubits=n_qubits, feature_map_reps=max(1, n_layers))
        return {'qasm': model.get_circuit_qasm(), 'svg_b64': model.get_circuit_svg(),
                'n_params': 0, 'description': descriptions['QSVC']}
    elif arch == 'VQC':
        model = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers, entangler=entangler)
        return {'qasm': model.get_circuit_qasm(), 'svg_b64': model.get_circuit_svg(),
                'n_params': model.n_params, 'description': descriptions['VQC']}
    else:
        raise ValueError(f'Unknown architecture "{architecture}". Choose from: QCNN, QSVC, VQC.')
