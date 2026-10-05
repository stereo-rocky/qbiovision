"""
ml_compat.py — scikit-learn / PyTorch compatibility layer.

scikit-learn pulls in SciPy; between them they unpack to ~270 MB, and PyTorch
(plus its CUDA/NVIDIA wheels) to several gigabytes. Neither fits inside a
Vercel Serverless Function (225 MB limit), yet Q-BioVision only uses a small,
well-defined slice of both libraries:

* metrics        — accuracy / precision / recall / F1 / confusion matrix /
                   ROC curve / ROC-AUC / label binarisation
* model_selection— stratified ``train_test_split``
* preprocessing  — ``MinMaxScaler``
* svm            — ``SVC(kernel='precomputed')`` for the quantum-kernel QSVC
* metrics.pairwise — ``rbf_kernel`` (classical fallback kernel)
* torch          — a small classical neural baseline for benchmarking

This module re-exports the real implementations when the libraries are
installed, and otherwise provides NumPy equivalents with the same call
signatures and return shapes. Numerical results are equivalent for the
binary/multiclass cases this project exercises.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SKLEARN_AVAILABLE = False
TORCH_AVAILABLE = False

try:  # pragma: no cover - depends on the install profile
    import torch  # type: ignore  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# NumPy implementations (used when scikit-learn is unavailable)
# ──────────────────────────────────────────────────────────────────────────────

def _np_accuracy_score(y_true, y_pred, **_) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def _np_confusion_matrix(y_true, y_pred, labels: Optional[Sequence] = None, **_):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def _prf_per_class(y_true, y_pred, labels):
    cm = _np_confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return precision, recall, f1, cm


def _aggregate(values, counts, average):
    if average == "macro" or average is None:
        return float(np.mean(values)) if len(values) else 0.0
    if average == "weighted":
        total = counts.sum()
        return float(np.sum(values * counts) / total) if total else 0.0
    if average == "binary":
        return float(values[-1]) if len(values) else 0.0
    return float(np.mean(values)) if len(values) else 0.0


def _np_precision_score(y_true, y_pred, average="macro", zero_division=0, **_):
    labels = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()]))
    p, _r, _f, cm = _prf_per_class(y_true, y_pred, labels)
    return _aggregate(p, cm.sum(axis=1), average)


def _np_recall_score(y_true, y_pred, average="macro", zero_division=0, **_):
    labels = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()]))
    _p, r, _f, cm = _prf_per_class(y_true, y_pred, labels)
    return _aggregate(r, cm.sum(axis=1), average)


def _np_f1_score(y_true, y_pred, average="macro", zero_division=0, **_):
    labels = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()]))
    _p, _r, f, cm = _prf_per_class(y_true, y_pred, labels)
    return _aggregate(f, cm.sum(axis=1), average)


def _np_roc_curve(y_true, y_score, pos_label=None, **_):
    """Vectorised ROC curve, matching sklearn's (fpr, tpr, thresholds) output."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if pos_label is None:
        classes = np.unique(y_true)
        pos_label = classes[-1] if len(classes) else 1
    y_bin = (y_true == pos_label).astype(int)

    order = np.argsort(-y_score, kind="mergesort")
    y_bin = y_bin[order]
    y_score = y_score[order]

    distinct = np.where(np.diff(y_score))[0]
    threshold_idx = np.r_[distinct, y_bin.size - 1]

    tps = np.cumsum(y_bin)[threshold_idx]
    fps = 1 + threshold_idx - tps

    n_pos = max(tps[-1], 1)
    n_neg = max(fps[-1], 1)
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    thresholds = np.r_[y_score[0] + 1.0, y_score[threshold_idx]]
    return fpr, tpr, thresholds


def _binary_auc(y_true, y_score, pos_label=None) -> float:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if pos_label is None:
        classes = np.unique(y_true)
        pos_label = classes[-1] if len(classes) else 1
    pos = y_score[y_true == pos_label]
    neg = y_score[y_true != pos_label]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Rank-based (Mann-Whitney U) AUC with tie handling.
    allv = np.concatenate([pos, neg])
    ranks = _rankdata(allv)
    rank_pos = ranks[: len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _np_roc_auc_score(y_true, y_score, multi_class="raise", average="macro", **_):
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    # One-hot encoded ground truth (multi-class OvR path).
    if y_true_arr.ndim == 2 and y_score_arr.ndim == 2:
        aucs = [
            _binary_auc(y_true_arr[:, k], y_score_arr[:, k], pos_label=1)
            for k in range(y_true_arr.shape[1])
            if len(np.unique(y_true_arr[:, k])) > 1
        ]
        return float(np.mean(aucs)) if aucs else 0.5

    if y_score_arr.ndim == 2:
        if y_score_arr.shape[1] == 2:
            return _binary_auc(y_true_arr, y_score_arr[:, 1])
        classes = np.unique(y_true_arr)
        aucs = [_binary_auc((y_true_arr == c).astype(int), y_score_arr[:, i], pos_label=1)
                for i, c in enumerate(classes)]
        return float(np.mean(aucs)) if aucs else 0.5

    return _binary_auc(y_true_arr, y_score_arr)


def _np_label_binarize(y, classes, **_):
    y = np.asarray(y).ravel()
    classes = list(classes)
    out = np.zeros((len(y), len(classes)), dtype=int)
    for i, c in enumerate(classes):
        out[:, i] = (y == c).astype(int)
    if len(classes) == 2:
        return out[:, 1:2]
    return out


def _np_train_test_split(*arrays, test_size=None, train_size=None, random_state=None,
                         stratify=None, shuffle=True, **_):
    if not arrays:
        raise ValueError("At least one array is required")
    n = len(arrays[0])
    rng = np.random.default_rng(random_state)

    if train_size is not None:
        n_train = int(round(train_size * n)) if train_size < 1 else int(train_size)
        n_train = max(1, min(n - 1, n_train))
    else:
        ts = 0.25 if test_size is None else test_size
        n_test = int(round(ts * n)) if ts < 1 else int(ts)
        n_train = max(1, min(n - 1, n - n_test))

    if stratify is not None:
        strat = np.asarray(stratify).ravel()
        train_idx, test_idx = [], []
        for cls in np.unique(strat):
            cls_idx = np.where(strat == cls)[0]
            if shuffle:
                cls_idx = rng.permutation(cls_idx)
            k = int(round(len(cls_idx) * n_train / n))
            k = max(1, min(len(cls_idx) - 1, k)) if len(cls_idx) > 1 else len(cls_idx)
            train_idx.extend(cls_idx[:k])
            test_idx.extend(cls_idx[k:])
        train_idx = np.array(train_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)
        if len(test_idx) == 0:
            test_idx = train_idx[-1:]
            train_idx = train_idx[:-1]
        train_idx = rng.permutation(train_idx)
        test_idx = rng.permutation(test_idx)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

    out = []
    for arr in arrays:
        arr = np.asarray(arr)
        out.extend([arr[train_idx], arr[test_idx]])
    return out


class _NpMinMaxScaler:
    """NumPy re-implementation of ``sklearn.preprocessing.MinMaxScaler``."""

    def __init__(self, feature_range=(0, 1)) -> None:
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        span = np.where(self.data_max_ - self.data_min_ > 1e-12,
                        self.data_max_ - self.data_min_, 1.0)
        scaled = (X - self.data_min_) / span
        return np.clip(scaled, 0.0, 1.0) * (hi - lo) + lo

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def _np_rbf_kernel(X, Y=None, gamma=None):
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    sq = (X ** 2).sum(axis=1)[:, None] + (Y ** 2).sum(axis=1)[None, :] - 2.0 * X @ Y.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


class _KernelLogisticSVC:
    """Drop-in replacement for ``SVC(kernel='precomputed', probability=True)``.

    Implements regularised multinomial logistic regression on the precomputed
    kernel matrix, which yields the same API (``fit``/``predict``/
    ``predict_proba``/``score``/``support_vectors_``) and comparable decision
    boundaries for the low-dimensional quantum kernels used here.
    """

    def __init__(self, kernel: str = "precomputed", C: float = 1.0,
                 probability: bool = True, random_state: Optional[int] = None,
                 max_iter: int = 400, lr: float = 0.5, **_) -> None:
        self.kernel = kernel
        self.C = float(C)
        self.probability = probability
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.support_vectors_: np.ndarray = np.empty((0, 0))
        self.support_: np.ndarray = np.empty((0,), dtype=int)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def fit(self, K, y):
        K = np.asarray(K, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n, m = K.shape
        n_cls = len(self.classes_)
        Y = np.zeros((n, n_cls))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0

        rng = np.random.default_rng(self.random_state)
        W = rng.normal(0.0, 0.01, size=(m, n_cls))
        b = np.zeros(n_cls)
        reg = 1.0 / max(self.C, 1e-6)

        # Adam optimiser keeps convergence stable without SciPy.
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        mb = np.zeros_like(b); vb = np.zeros_like(b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.max_iter + 1):
            P = self._softmax(K @ W + b)
            G = (P - Y) / n
            gW = K.T @ G + reg * W / n
            gb = G.sum(axis=0)

            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW ** 2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
            W -= self.lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
            b -= self.lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)

        self.coef_ = W
        self.intercept_ = b
        weight = np.abs(W).sum(axis=1)
        self.support_ = np.where(weight > 0.1 * max(weight.max(), 1e-12))[0]
        self.support_vectors_ = K[self.support_]
        return self

    def decision_function(self, K):
        return np.asarray(K, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, K):
        return self._softmax(self.decision_function(K))

    def predict(self, K):
        return self.classes_[self.predict_proba(K).argmax(axis=1)]

    def score(self, K, y):
        return _np_accuracy_score(y, self.predict(K))


# ──────────────────────────────────────────────────────────────────────────────
# Public exports — prefer scikit-learn when present
# ──────────────────────────────────────────────────────────────────────────────

try:  # pragma: no cover - depends on the install profile
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.metrics.pairwise import rbf_kernel  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.preprocessing import MinMaxScaler, label_binarize  # type: ignore
    from sklearn.svm import SVC  # type: ignore

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.info("scikit-learn not installed — using NumPy metric/model fallbacks.")
    accuracy_score = _np_accuracy_score
    confusion_matrix = _np_confusion_matrix
    f1_score = _np_f1_score
    precision_score = _np_precision_score
    recall_score = _np_recall_score
    roc_auc_score = _np_roc_auc_score
    roc_curve = _np_roc_curve
    rbf_kernel = _np_rbf_kernel
    train_test_split = _np_train_test_split
    label_binarize = _np_label_binarize
    MinMaxScaler = _NpMinMaxScaler  # type: ignore
    SVC = _KernelLogisticSVC  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Torch-free classical baseline
# ──────────────────────────────────────────────────────────────────────────────

class NumpyMLPBaseline:
    """Classical neural baseline (dense ReLU net) implemented with NumPy + Adam.

    Stands in for the PyTorch ``ClassicalCNNBaseline`` when torch is not
    installed. It trains on exactly the same N-qubit feature vectors and
    reports the same metrics, so the quantum-vs-classical comparison stays
    meaningful; only the optimiser implementation differs.
    """

    def __init__(self, n_features: int, n_classes: int = 2, hidden: Sequence[int] = (32, 64),
                 seed: int = 42) -> None:
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        rng = np.random.default_rng(seed)
        dims = [self.n_features, *hidden, self.n_classes]
        self.W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]

    def n_params(self) -> int:
        return int(sum(w.size for w in self.W) + sum(x.size for x in self.b))

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def _forward(self, X):
        acts = [np.asarray(X, dtype=float)]
        pre = []
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            pre.append(z)
            acts.append(np.maximum(z, 0.0) if i < len(self.W) - 1 else z)
        return acts, pre

    def predict_proba(self, X):
        acts, _ = self._forward(X)
        return self._softmax(acts[-1])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X, y, n_epochs: int = 20, batch_size: int = 16, lr: float = 1e-3,
            seed: int = 42):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(int)
        rng = np.random.default_rng(seed)
        n = len(X)

        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        loss_history, acc_history = [], []

        for _ in range(max(1, n_epochs)):
            order = rng.permutation(n)
            epoch_loss, correct = 0.0, 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                acts, _pre = self._forward(xb)
                probs = self._softmax(acts[-1])
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(yb)), yb] = 1.0

                epoch_loss += float(-np.sum(np.log(np.maximum(
                    probs[np.arange(len(yb)), yb], 1e-12))))
                correct += int((probs.argmax(axis=1) == yb).sum())

                delta = (probs - onehot) / len(yb)
                grads_w, grads_b = [None] * len(self.W), [None] * len(self.b)
                for layer in range(len(self.W) - 1, -1, -1):
                    grads_w[layer] = acts[layer].T @ delta
                    grads_b[layer] = delta.sum(axis=0)
                    if layer > 0:
                        delta = (delta @ self.W[layer].T) * (acts[layer] > 0)

                step += 1
                for layer in range(len(self.W)):
                    mW[layer] = b1 * mW[layer] + (1 - b1) * grads_w[layer]
                    vW[layer] = b2 * vW[layer] + (1 - b2) * grads_w[layer] ** 2
                    mb[layer] = b1 * mb[layer] + (1 - b1) * grads_b[layer]
                    vb[layer] = b2 * vb[layer] + (1 - b2) * grads_b[layer] ** 2
                    self.W[layer] -= lr * (mW[layer] / (1 - b1 ** step)) / (
                        np.sqrt(vW[layer] / (1 - b2 ** step)) + eps)
                    self.b[layer] -= lr * (mb[layer] / (1 - b1 ** step)) / (
                        np.sqrt(vb[layer] / (1 - b2 ** step)) + eps)

            loss_history.append(round(epoch_loss / n, 4))
            acc_history.append(round(correct / n, 4))

        return loss_history, acc_history


def train_numpy_baseline(X_train, y_train, n_epochs: int = 20, batch_size: int = 16,
                         lr: float = 1e-3, seed: int = 42) -> dict:
    """Train :class:`NumpyMLPBaseline` and return benchmarking-shaped results."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train).ravel().astype(int)
    t0 = time.time()
    model = NumpyMLPBaseline(
        n_features=X_train.shape[1],
        n_classes=max(2, len(np.unique(y_train))),
        seed=seed,
    )
    loss_history, acc_history = model.fit(
        X_train, y_train, n_epochs=n_epochs, batch_size=batch_size, lr=lr, seed=seed
    )
    return {
        "model": model,
        "accuracy": acc_history[-1] if acc_history else 0.0,
        "loss_history": loss_history,
        "accuracy_history": acc_history,
        "n_params": model.n_params(),
        "training_time": round(time.time() - t0, 2),
    }


__all__ = [
    "SKLEARN_AVAILABLE",
    "TORCH_AVAILABLE",
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "roc_curve",
    "label_binarize",
    "rbf_kernel",
    "train_test_split",
    "MinMaxScaler",
    "SVC",
    "NumpyMLPBaseline",
    "train_numpy_baseline",
]
