"""
qlite.py — Pure-NumPy stand-in for the small subset of Qiskit / Qiskit Aer
that Q-BioVision actually uses.

Why this module exists
----------------------
The full quantum stack (``qiskit`` + ``qiskit-aer`` + their mandatory
``scipy`` dependency) unpacks to well over 500 MB, which cannot fit inside a
Vercel Serverless Function (225 MB limit). Rather than disabling the quantum
features in the deployed demo, this module re-implements the handful of
primitives the project needs on top of NumPy alone:

* gate-level circuit construction (``QuantumCircuit``)
* symbolic circuit parameters (``Parameter`` / ``ParameterVector``)
* the ``ZZFeatureMap`` / ``RealAmplitudes`` / ``TwoLocal`` library circuits
* an exact statevector simulator (``AerSimulator``)
* a shot-based sampler with a simplified NISQ noise model
  (``NoiseModel`` + thermal relaxation / depolarizing / readout errors)
* OpenQASM 3 export and a Pillow-rendered circuit diagram

The simulation is *real* — statevectors are propagated exactly for up to 16
qubits — but the noise model is an approximation (stochastic Pauli
unravelling) rather than Aer's full density-matrix treatment.

``quantum_backend.py`` prefers the genuine Qiskit stack whenever it is
installed (local development, ``requirements-full.txt``) and only falls back
to this module when it is not.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "Parameter",
    "ParameterExpression",
    "ParameterVector",
    "QuantumRegister",
    "ClassicalRegister",
    "QuantumCircuit",
    "transpile",
    "ZZFeatureMap",
    "RealAmplitudes",
    "TwoLocal",
    "AerSimulator",
    "NoiseModel",
    "ReadoutError",
    "thermal_relaxation_error",
    "depolarizing_error",
    "qasm_loads",
]


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic parameters
# ──────────────────────────────────────────────────────────────────────────────

class ParameterExpression:
    """A tiny symbolic expression tree over :class:`Parameter` and floats."""

    __slots__ = ("op", "args")

    def __init__(self, op: str, args: Sequence) -> None:
        self.op = op
        self.args = list(args)

    # -- arithmetic -----------------------------------------------------------
    def __add__(self, other):  return ParameterExpression("+", [self, other])
    def __radd__(self, other): return ParameterExpression("+", [other, self])
    def __sub__(self, other):  return ParameterExpression("-", [self, other])
    def __rsub__(self, other): return ParameterExpression("-", [other, self])
    def __mul__(self, other):  return ParameterExpression("*", [self, other])
    def __rmul__(self, other): return ParameterExpression("*", [other, self])
    def __truediv__(self, other): return ParameterExpression("/", [self, other])
    def __neg__(self):         return ParameterExpression("*", [-1.0, self])

    # -- evaluation -----------------------------------------------------------
    def parameters(self) -> set:
        out: set = set()
        for a in self.args:
            if isinstance(a, Parameter):
                out.add(a)
            elif isinstance(a, ParameterExpression):
                out |= a.parameters()
        return out

    def bind(self, mapping: Dict["Parameter", float]) -> float:
        vals = [_bind_value(a, mapping) for a in self.args]
        if self.op == "+":
            return vals[0] + vals[1]
        if self.op == "-":
            return vals[0] - vals[1]
        if self.op == "*":
            return vals[0] * vals[1]
        if self.op == "/":
            return vals[0] / vals[1]
        raise ValueError(f"Unknown operator {self.op!r}")

    def __repr__(self) -> str:
        a, b = (_expr_str(x) for x in self.args)
        return f"({a} {self.op} {b})"


class Parameter(ParameterExpression):
    """A named free parameter of a circuit."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:  # noqa: D107 - see class docstring
        self.name = name
        self.op = "param"
        self.args = []

    def parameters(self) -> set:
        return {self}

    def bind(self, mapping: Dict["Parameter", float]) -> float:
        if self in mapping:
            return float(mapping[self])
        raise KeyError(f"Parameter {self.name!r} was not assigned a value")

    def __hash__(self) -> int:
        return hash(("qlite.Parameter", self.name, id(self)))

    def __eq__(self, other) -> bool:
        return self is other

    def __repr__(self) -> str:
        return self.name


def _bind_value(x, mapping):
    if isinstance(x, (Parameter, ParameterExpression)):
        return x.bind(mapping)
    return float(x)


def _expr_str(x) -> str:
    if isinstance(x, Parameter):
        return x.name
    if isinstance(x, ParameterExpression):
        return repr(x)
    return f"{float(x):.6g}"


def _is_symbolic(x) -> bool:
    return isinstance(x, (Parameter, ParameterExpression))


class ParameterVector:
    """An indexable, iterable collection of named parameters (``x[0]``, ``x[1]`` …)."""

    def __init__(self, name: str, length: int) -> None:
        self.name = name
        self._params = [Parameter(f"{name}[{i}]") for i in range(length)]

    def __getitem__(self, idx):
        return self._params[idx]

    def __iter__(self):
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    @property
    def params(self) -> List[Parameter]:
        return list(self._params)


# ──────────────────────────────────────────────────────────────────────────────
# Registers and bits
# ──────────────────────────────────────────────────────────────────────────────

class _Bit:
    __slots__ = ("register", "index")

    def __init__(self, register, index: int) -> None:
        self.register = register
        self.index = index

    def __repr__(self) -> str:
        return f"{self.register.name}[{self.index}]"


class Qubit(_Bit):
    pass


class Clbit(_Bit):
    pass


class _Register:
    _bit_cls = _Bit

    def __init__(self, size: int, name: Optional[str] = None) -> None:
        self.size = int(size)
        self.name = name or "r"
        self._bits = [self._bit_cls(self, i) for i in range(self.size)]

    def __getitem__(self, idx):
        return self._bits[idx]

    def __iter__(self):
        return iter(self._bits)

    def __len__(self) -> int:
        return self.size


class QuantumRegister(_Register):
    _bit_cls = Qubit

    def __init__(self, size: int, name: Optional[str] = None) -> None:
        super().__init__(size, name or "q")


class ClassicalRegister(_Register):
    _bit_cls = Clbit

    def __init__(self, size: int, name: Optional[str] = None) -> None:
        super().__init__(size, name or "c")


# ──────────────────────────────────────────────────────────────────────────────
# Instructions
# ──────────────────────────────────────────────────────────────────────────────

# Gates that are their own inverse.
_SELF_INVERSE = {"h", "x", "y", "z", "cx", "cz", "swap", "id", "barrier"}
# Gates whose inverse negates the rotation angle.
_ANGLE_INVERSE = {"rx", "ry", "rz", "p", "rxx", "ryy", "rzz"}

_ONE_QUBIT = {"h", "x", "y", "z", "id", "rx", "ry", "rz", "p", "s", "sdg", "t", "tdg"}
_TWO_QUBIT = {"cx", "cz", "swap", "rxx", "ryy", "rzz"}


class Instruction:
    """A single gate (or directive) with optional symbolic parameters."""

    __slots__ = ("name", "params", "num_qubits", "num_clbits")

    def __init__(self, name: str, params: Sequence = (), num_qubits: int = 1,
                 num_clbits: int = 0) -> None:
        self.name = name
        self.params = list(params)
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits

    def inverse(self) -> "Instruction":
        if self.name in _SELF_INVERSE:
            return Instruction(self.name, self.params, self.num_qubits, self.num_clbits)
        if self.name in _ANGLE_INVERSE:
            return Instruction(self.name, [-p for p in self.params],
                               self.num_qubits, self.num_clbits)
        if self.name == "s":
            return Instruction("sdg", [], 1)
        if self.name == "sdg":
            return Instruction("s", [], 1)
        if self.name == "t":
            return Instruction("tdg", [], 1)
        if self.name == "tdg":
            return Instruction("t", [], 1)
        raise ValueError(f"Instruction {self.name!r} is not invertible")

    def __repr__(self) -> str:
        if self.params:
            return f"{self.name}({', '.join(_expr_str(p) for p in self.params)})"
        return self.name


class CircuitInstruction:
    """A gate bound to specific qubits/clbits (mirrors Qiskit's API shape)."""

    __slots__ = ("operation", "qubits", "clbits")

    def __init__(self, operation: Instruction, qubits: Sequence[Qubit],
                 clbits: Sequence[Clbit] = ()) -> None:
        self.operation = operation
        self.qubits = tuple(qubits)
        self.clbits = tuple(clbits)

    def __iter__(self):
        # Allows ``for op, qargs, cargs in circuit.data`` (legacy Qiskit style).
        return iter((self.operation, list(self.qubits), list(self.clbits)))

    def __repr__(self) -> str:
        return f"CircuitInstruction({self.operation!r}, {list(self.qubits)})"


# ──────────────────────────────────────────────────────────────────────────────
# QuantumCircuit
# ──────────────────────────────────────────────────────────────────────────────

class QuantumCircuit:
    """A minimal, NumPy-backed re-implementation of ``qiskit.QuantumCircuit``.

    Supports the construction/inspection surface used by Q-BioVision:
    ``h x y z rx ry rz cx cz swap barrier measure measure_all``,
    ``append``, ``copy``, ``assign_parameters``, ``qasm`` and ``png_b64``.
    """

    def __init__(self, *regs, name: Optional[str] = None) -> None:
        self.name = name or "circuit"
        self.qregs: List[QuantumRegister] = []
        self.cregs: List[ClassicalRegister] = []
        self.data: List[CircuitInstruction] = []
        self._save_statevector = False

        ints = [r for r in regs if isinstance(r, int)]
        if ints:
            if len(ints) != len(regs):
                raise TypeError("Mix of integers and registers is not supported")
            self.qregs.append(QuantumRegister(ints[0], "q"))
            if len(ints) > 1 and ints[1] > 0:
                self.cregs.append(ClassicalRegister(ints[1], "c"))
        else:
            for reg in regs:
                if isinstance(reg, QuantumRegister):
                    self.qregs.append(reg)
                elif isinstance(reg, ClassicalRegister):
                    self.cregs.append(reg)
                elif reg is None:
                    continue
                else:
                    raise TypeError(f"Unsupported register type: {type(reg)!r}")

    # -- sizes ----------------------------------------------------------------
    @property
    def qubits(self) -> List[Qubit]:
        return [b for reg in self.qregs for b in reg]

    @property
    def clbits(self) -> List[Clbit]:
        return [b for reg in self.cregs for b in reg]

    @property
    def num_qubits(self) -> int:
        return sum(len(r) for r in self.qregs)

    @property
    def num_clbits(self) -> int:
        return sum(len(r) for r in self.cregs)

    @property
    def parameters(self) -> List[Parameter]:
        seen: List[Parameter] = []
        for inst in self.data:
            for p in inst.operation.params:
                if _is_symbolic(p):
                    for prm in p.parameters():
                        if prm not in seen:
                            seen.append(prm)
        return seen

    # -- bit resolution -------------------------------------------------------
    def find_bit(self, bit: _Bit) -> int:
        collection = self.qregs if isinstance(bit, Qubit) else self.cregs
        offset = 0
        for reg in collection:
            if reg is bit.register:
                return offset + bit.index
            offset += len(reg)
        raise ValueError(f"Bit {bit!r} does not belong to this circuit")

    def _resolve_q(self, spec) -> List[Qubit]:
        qubits = self.qubits
        if spec is None:
            return list(qubits)
        if isinstance(spec, Qubit):
            return [spec]
        if isinstance(spec, (int, np.integer)):
            return [qubits[int(spec)]]
        if isinstance(spec, range) or isinstance(spec, (list, tuple, set, np.ndarray)):
            out: List[Qubit] = []
            for s in spec:
                out.extend(self._resolve_q(s))
            return out
        if isinstance(spec, QuantumRegister):
            return list(spec)
        raise TypeError(f"Cannot interpret {spec!r} as qubit(s)")

    def _resolve_c(self, spec) -> List[Clbit]:
        clbits = self.clbits
        if spec is None:
            return list(clbits)
        if isinstance(spec, Clbit):
            return [spec]
        if isinstance(spec, (int, np.integer)):
            return [clbits[int(spec)]]
        if isinstance(spec, range) or isinstance(spec, (list, tuple, set, np.ndarray)):
            out: List[Clbit] = []
            for s in spec:
                out.extend(self._resolve_c(s))
            return out
        if isinstance(spec, ClassicalRegister):
            return list(spec)
        raise TypeError(f"Cannot interpret {spec!r} as clbit(s)")

    # -- generic append -------------------------------------------------------
    def append(self, operation: Instruction, qargs=None, cargs=None) -> "QuantumCircuit":
        qubits = self._resolve_q(qargs) if qargs is not None else []
        clbits = self._resolve_c(cargs) if cargs else []
        self.data.append(CircuitInstruction(operation, qubits, clbits))
        return self

    def _add_1q(self, name: str, spec, params: Sequence = ()) -> "QuantumCircuit":
        for q in self._resolve_q(spec):
            self.data.append(CircuitInstruction(Instruction(name, params, 1), [q]))
        return self

    # -- single-qubit gates ---------------------------------------------------
    def h(self, q):  return self._add_1q("h", q)
    def x(self, q):  return self._add_1q("x", q)
    def y(self, q):  return self._add_1q("y", q)
    def z(self, q):  return self._add_1q("z", q)
    def s(self, q):  return self._add_1q("s", q)
    def sdg(self, q): return self._add_1q("sdg", q)
    def t(self, q):  return self._add_1q("t", q)
    def tdg(self, q): return self._add_1q("tdg", q)
    def id(self, q): return self._add_1q("id", q)
    def rx(self, theta, q): return self._add_1q("rx", q, [theta])
    def ry(self, theta, q): return self._add_1q("ry", q, [theta])
    def rz(self, phi, q):   return self._add_1q("rz", q, [phi])
    def p(self, phi, q):    return self._add_1q("p", q, [phi])

    # -- two-qubit gates ------------------------------------------------------
    def _add_2q(self, name: str, a, b, params: Sequence = ()) -> "QuantumCircuit":
        qa = self._resolve_q(a)
        qb = self._resolve_q(b)
        for x, y in zip(qa, qb):
            self.data.append(CircuitInstruction(Instruction(name, params, 2), [x, y]))
        return self

    def cx(self, c, t):   return self._add_2q("cx", c, t)
    def cnot(self, c, t): return self._add_2q("cx", c, t)
    def cz(self, c, t):   return self._add_2q("cz", c, t)
    def swap(self, a, b): return self._add_2q("swap", a, b)
    def rxx(self, theta, a, b): return self._add_2q("rxx", a, b, [theta])
    def ryy(self, theta, a, b): return self._add_2q("ryy", a, b, [theta])
    def rzz(self, theta, a, b): return self._add_2q("rzz", a, b, [theta])

    # -- directives -----------------------------------------------------------
    def barrier(self, *qargs) -> "QuantumCircuit":
        spec = qargs[0] if len(qargs) == 1 else (list(qargs) if qargs else None)
        qubits = self._resolve_q(spec)
        self.data.append(CircuitInstruction(Instruction("barrier", [], len(qubits)), qubits))
        return self

    def measure(self, qargs, cargs) -> "QuantumCircuit":
        qubits = self._resolve_q(qargs)
        clbits = self._resolve_c(cargs)
        for q, c in zip(qubits, clbits):
            self.data.append(CircuitInstruction(Instruction("measure", [], 1, 1), [q], [c]))
        return self

    def measure_all(self, inplace: bool = True, add_bits: bool = True) -> Optional["QuantumCircuit"]:
        target = self if inplace else self.copy()
        creg = ClassicalRegister(target.num_qubits, "meas")
        target.cregs.append(creg)
        target.barrier()
        for i, q in enumerate(target.qubits):
            target.data.append(
                CircuitInstruction(Instruction("measure", [], 1, 1), [q], [creg[i]])
            )
        return None if inplace else target

    def save_statevector(self, label: str = "statevector") -> "QuantumCircuit":
        self._save_statevector = True
        return self

    # -- copying / binding ----------------------------------------------------
    def copy(self, name: Optional[str] = None) -> "QuantumCircuit":
        new = QuantumCircuit(name=name or self.name)
        new.qregs = list(self.qregs)
        new.cregs = list(self.cregs)
        new.data = [CircuitInstruction(inst.operation, inst.qubits, inst.clbits)
                    for inst in self.data]
        new._save_statevector = self._save_statevector
        return new

    def assign_parameters(self, values, inplace: bool = False) -> "QuantumCircuit":
        """Bind free parameters. ``values`` may be a dict or an ordered sequence."""
        if isinstance(values, dict):
            mapping = dict(values)
        else:
            mapping = dict(zip(self.parameters, list(values)))

        target = self if inplace else self.copy()
        new_data: List[CircuitInstruction] = []
        for inst in target.data:
            op = inst.operation
            if op.params and any(_is_symbolic(p) for p in op.params):
                bound = [p.bind(mapping) if _is_symbolic(p) else float(p) for p in op.params]
                op = Instruction(op.name, bound, op.num_qubits, op.num_clbits)
            new_data.append(CircuitInstruction(op, inst.qubits, inst.clbits))
        target.data = new_data
        return target

    def compose(self, other: "QuantumCircuit", qubits=None, inplace: bool = False):
        target = self if inplace else self.copy()
        own = target.qubits
        mapping = self._resolve_q(qubits) if qubits is not None else own
        for inst in other.data:
            mapped = [mapping[other.find_bit(q)] for q in inst.qubits]
            target.data.append(CircuitInstruction(inst.operation, mapped, inst.clbits))
        return None if inplace else target

    def decompose(self, *args, **kwargs) -> "QuantumCircuit":
        return self.copy()

    # -- export ---------------------------------------------------------------
    def qasm(self) -> str:
        """Export the circuit as OpenQASM 3 text (free parameters become inputs)."""
        lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]
        free = self.parameters
        for prm in free:
            lines.append(f"input float[64] {_qasm_ident(prm.name)};")
        if free:
            lines.append("")
        lines.append(f"qubit[{self.num_qubits}] q;")
        for creg in self.cregs:
            lines.append(f"bit[{len(creg)}] {_qasm_ident(creg.name)};")
        lines.append("")

        for inst in self.data:
            op = inst.operation
            idx = [self.find_bit(q) for q in inst.qubits]
            if op.name == "barrier":
                lines.append("barrier " + ", ".join(f"q[{i}]" for i in idx) + ";")
            elif op.name == "measure":
                cbit = inst.clbits[0]
                creg = _qasm_ident(cbit.register.name)
                lines.append(f"{creg}[{cbit.index}] = measure q[{idx[0]}];")
            else:
                args = ""
                if op.params:
                    args = "(" + ", ".join(_qasm_expr(p) for p in op.params) + ")"
                lines.append(f"{op.name}{args} " + ", ".join(f"q[{i}]" for i in idx) + ";")
        return "\n".join(lines) + "\n"

    def draw(self, output: str = "text", **kwargs) -> str:
        if output in ("text", None):
            return self._draw_text()
        raise ValueError(
            f"qlite.QuantumCircuit.draw does not support output={output!r}; "
            "use png_b64() instead."
        )

    def _draw_text(self) -> str:
        n = self.num_qubits
        rows = [f"q{i}: " + "─" * 2 for i in range(n)]
        for inst in self.data:
            op = inst.operation
            if op.name == "barrier":
                for i in range(n):
                    rows[i] += "│"
                continue
            idx = [self.find_bit(q) for q in inst.qubits]
            label = op.name.upper()[:4]
            width = len(label) + 2
            for i in range(n):
                rows[i] += (f"[{label}]" if i in idx else "─" * width)
        return "\n".join(rows)

    def png_b64(self, scale: int = 1) -> str:
        """Render a circuit diagram as a base64-encoded PNG using Pillow."""
        return _render_circuit_png(self, scale=scale)

    def __repr__(self) -> str:
        return f"<qlite.QuantumCircuit {self.name!r} qubits={self.num_qubits} ops={len(self.data)}>"


def _qasm_ident(name: str) -> str:
    return name.replace("[", "_").replace("]", "").replace(" ", "_")


def _qasm_expr(p) -> str:
    if isinstance(p, Parameter):
        return _qasm_ident(p.name)
    if isinstance(p, ParameterExpression):
        a, b = p.args
        return f"({_qasm_expr(a)} {p.op} {_qasm_expr(b)})"
    return f"{float(p):.8g}"


def transpile(circuit, backend=None, **kwargs):
    """No-op transpiler — the NumPy simulator runs the circuit as written."""
    if isinstance(circuit, (list, tuple)):
        return [c.copy() for c in circuit]
    return circuit.copy()


def qasm_loads(text: str):
    """Not supported by the lightweight backend (callers must have a fallback)."""
    raise NotImplementedError("qlite cannot parse OpenQASM; build the circuit directly.")


# ──────────────────────────────────────────────────────────────────────────────
# Library circuits
# ──────────────────────────────────────────────────────────────────────────────

def ZZFeatureMap(feature_dimension: int, reps: int = 2,
                 entanglement: str = "linear", **kwargs) -> QuantumCircuit:
    """Second-order Pauli-Z evolution feature map (Havlíček et al., 2019)."""
    n = int(feature_dimension)
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n, name="ZZFeatureMap")
    pairs = _entangler_pairs(n, entanglement)
    for _ in range(max(1, int(reps))):
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.p(2.0 * x[i], i)
        for i, j in pairs:
            qc.cx(i, j)
            qc.p(2.0 * ((math.pi - x[i]) * (math.pi - x[j])), j)
            qc.cx(i, j)
    qc._feature_params = x  # type: ignore[attr-defined]
    return qc


def RealAmplitudes(num_qubits: int, reps: int = 3,
                   entanglement: str = "linear", **kwargs) -> QuantumCircuit:
    """Hardware-efficient Ry + CX ansatz."""
    n = int(num_qubits)
    reps = max(1, int(reps))
    theta = ParameterVector("θ", n * (reps + 1))
    qc = QuantumCircuit(n, name="RealAmplitudes")
    k = 0
    pairs = _entangler_pairs(n, entanglement)
    for _ in range(reps):
        for i in range(n):
            qc.ry(theta[k], i)
            k += 1
        for i, j in pairs:
            qc.cx(i, j)
    for i in range(n):
        qc.ry(theta[k], i)
        k += 1
    return qc


def TwoLocal(num_qubits: int, rotation_blocks="ry", entanglement_blocks="cx",
             reps: int = 3, entanglement: str = "linear", **kwargs) -> QuantumCircuit:
    """Generic alternating rotation/entanglement ansatz."""
    n = int(num_qubits)
    reps = max(1, int(reps))
    rots = [rotation_blocks] if isinstance(rotation_blocks, str) else list(rotation_blocks)
    ent = entanglement_blocks if isinstance(entanglement_blocks, str) else entanglement_blocks[0]
    theta = ParameterVector("θ", n * len(rots) * (reps + 1))
    qc = QuantumCircuit(n, name="TwoLocal")
    pairs = _entangler_pairs(n, entanglement)
    k = 0
    for rep in range(reps + 1):
        for r in rots:
            for i in range(n):
                getattr(qc, r)(theta[k], i)
                k += 1
        if rep < reps:
            for i, j in pairs:
                getattr(qc, ent)(i, j)
    return qc


def _entangler_pairs(n: int, entanglement: str) -> List[tuple]:
    entanglement = (entanglement or "linear").lower()
    if entanglement in ("full", "all"):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if entanglement in ("circular", "circle") and n > 1:
        return [(i, (i + 1) % n) for i in range(n)]
    return [(i, i + 1) for i in range(n - 1)]


# ──────────────────────────────────────────────────────────────────────────────
# Gate matrices
# ──────────────────────────────────────────────────────────────────────────────

_SQ2 = 1.0 / math.sqrt(2.0)
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
_TDG = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex)

_PAULIS = (_I2, _X, _Y, _Z)


def _gate_matrix_1q(name: str, params: Sequence[float]) -> np.ndarray:
    if name == "h":   return _H
    if name == "x":   return _X
    if name == "y":   return _Y
    if name == "z":   return _Z
    if name == "s":   return _S
    if name == "sdg": return _SDG
    if name == "t":   return _T
    if name == "tdg": return _TDG
    if name == "id":  return _I2
    th = float(params[0]) if params else 0.0
    c, s = math.cos(th / 2.0), math.sin(th / 2.0)
    if name == "rx": return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry": return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz": return np.array([[np.exp(-1j * th / 2), 0], [0, np.exp(1j * th / 2)]], dtype=complex)
    if name == "p":  return np.array([[1, 0], [0, np.exp(1j * th)]], dtype=complex)
    raise ValueError(f"Unsupported 1-qubit gate {name!r}")


def _gate_matrix_2q(name: str, params: Sequence[float]) -> np.ndarray:
    if name == "cx":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    if name == "cz":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    th = float(params[0]) if params else 0.0
    c, s = math.cos(th / 2.0), math.sin(th / 2.0)
    if name == "rxx":
        return np.array([[c, 0, 0, -1j * s], [0, c, -1j * s, 0],
                         [0, -1j * s, c, 0], [-1j * s, 0, 0, c]], dtype=complex)
    if name == "ryy":
        return np.array([[c, 0, 0, 1j * s], [0, c, -1j * s, 0],
                         [0, -1j * s, c, 0], [1j * s, 0, 0, c]], dtype=complex)
    if name == "rzz":
        ph = np.exp(1j * th / 2)
        return np.diag([np.conj(ph), ph, ph, np.conj(ph)]).astype(complex)
    raise ValueError(f"Unsupported 2-qubit gate {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Statevector engine
# ──────────────────────────────────────────────────────────────────────────────

def _apply_1q(psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    axis = n - 1 - q
    t = np.moveaxis(psi.reshape([2] * n), axis, 0)
    shape = t.shape
    t = (U @ t.reshape(2, -1)).reshape(shape)
    return np.moveaxis(t, 0, axis).reshape(-1)


def _apply_2q(psi: np.ndarray, U: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    a0, a1 = n - 1 - q0, n - 1 - q1
    t = np.moveaxis(psi.reshape([2] * n), (a0, a1), (0, 1))
    shape = t.shape
    t = (U @ t.reshape(4, -1)).reshape(shape)
    return np.moveaxis(t, (0, 1), (a0, a1)).reshape(-1)


def _simulate_statevector(circuit: QuantumCircuit,
                          noise: Optional["_NoiseSpec"] = None,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Propagate |0…0> through the circuit, optionally with stochastic Pauli noise."""
    n = circuit.num_qubits
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0

    for inst in circuit.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        idx = [circuit.find_bit(q) for q in inst.qubits]
        params = [float(p) if not _is_symbolic(p) else 0.0 for p in op.params]
        if any(_is_symbolic(p) for p in op.params):
            raise ValueError(
                "Circuit still contains free parameters; call assign_parameters() first."
            )

        if op.name in _ONE_QUBIT:
            psi = _apply_1q(psi, _gate_matrix_1q(op.name, params), idx[0], n)
            if noise is not None:
                psi = noise.apply_1q(psi, idx[0], n, rng)
        elif op.name in _TWO_QUBIT:
            psi = _apply_2q(psi, _gate_matrix_2q(op.name, params), idx[0], idx[1], n)
            if noise is not None:
                psi = noise.apply_2q(psi, idx[0], idx[1], n, rng)
        else:
            raise ValueError(f"Unsupported operation {op.name!r}")

    return psi


def _measured_qubits(circuit: QuantumCircuit) -> List[int]:
    """Return qubit indices ordered by the classical bit they are measured into."""
    pairs = []
    for inst in circuit.data:
        if inst.operation.name == "measure" and inst.clbits:
            pairs.append((circuit.find_bit(inst.clbits[0]), circuit.find_bit(inst.qubits[0])))
    pairs.sort()
    return [q for _, q in pairs]


# ──────────────────────────────────────────────────────────────────────────────
# Noise model (simplified stochastic Pauli unravelling)
# ──────────────────────────────────────────────────────────────────────────────

class QuantumError:
    """Approximate error channel: per-qubit bit/phase flip plus depolarizing."""

    def __init__(self, num_qubits: int = 1, p_x: float = 0.0, p_z: float = 0.0,
                 p_depol: float = 0.0) -> None:
        self.num_qubits = num_qubits
        self.p_x = float(p_x)
        self.p_z = float(p_z)
        self.p_depol = float(p_depol)

    def expand(self, other: "QuantumError") -> "QuantumError":
        return QuantumError(
            num_qubits=self.num_qubits + other.num_qubits,
            p_x=max(self.p_x, other.p_x),
            p_z=max(self.p_z, other.p_z),
            p_depol=1.0 - (1.0 - self.p_depol) * (1.0 - other.p_depol),
        )

    def compose(self, other: "QuantumError") -> "QuantumError":
        return QuantumError(
            num_qubits=max(self.num_qubits, other.num_qubits),
            p_x=1.0 - (1.0 - self.p_x) * (1.0 - other.p_x),
            p_z=1.0 - (1.0 - self.p_z) * (1.0 - other.p_z),
            p_depol=1.0 - (1.0 - self.p_depol) * (1.0 - other.p_depol),
        )

    def tensor(self, other: "QuantumError") -> "QuantumError":
        return self.expand(other)


def thermal_relaxation_error(t1: float, t2: float, gate_time: float,
                             excited_state_population: float = 0.0) -> QuantumError:
    """T1/T2 relaxation approximated by bit-flip (T1) and phase-flip (T2) rates."""
    t1 = max(float(t1), 1e-9)
    t2 = max(min(float(t2), 2.0 * t1), 1e-9)
    gate_time = max(float(gate_time), 0.0)
    p_amp = 1.0 - math.exp(-gate_time / t1)
    p_phase = 1.0 - math.exp(-gate_time / t2)
    return QuantumError(num_qubits=1, p_x=p_amp / 2.0, p_z=p_phase / 2.0)


def depolarizing_error(param: float, num_qubits: int = 1) -> QuantumError:
    return QuantumError(num_qubits=num_qubits, p_depol=float(param))


class ReadoutError:
    """Symmetric readout confusion matrix ``[[1-p, p], [p, 1-p]]``."""

    def __init__(self, probabilities) -> None:
        mat = np.asarray(probabilities, dtype=float)
        self.probabilities = mat
        self.p_flip = float(mat[0][1]) if mat.shape == (2, 2) else 0.0


class NoiseModel:
    """Collects the per-gate error channels used by :class:`AerSimulator`."""

    def __init__(self) -> None:
        self._errors_1q: Optional[QuantumError] = None
        self._errors_2q: Optional[QuantumError] = None
        self._readout_p: float = 0.0

    def add_all_qubit_quantum_error(self, error: QuantumError, instructions) -> None:
        if isinstance(instructions, str):
            instructions = [instructions]
        two_qubit = error.num_qubits >= 2 or any(
            name in _TWO_QUBIT for name in instructions
        )
        if two_qubit:
            self._errors_2q = error if self._errors_2q is None else self._errors_2q.compose(error)
        else:
            self._errors_1q = error if self._errors_1q is None else self._errors_1q.compose(error)

    def add_all_qubit_readout_error(self, error: ReadoutError) -> None:
        self._readout_p = max(self._readout_p, getattr(error, "p_flip", 0.0))

    # Convenience for the simulator ------------------------------------------
    def _spec(self) -> "_NoiseSpec":
        return _NoiseSpec(self._errors_1q, self._errors_2q, self._readout_p)

    def is_trivial(self) -> bool:
        return self._errors_1q is None and self._errors_2q is None and self._readout_p == 0.0

    def __repr__(self) -> str:
        return (f"<qlite.NoiseModel 1q={self._errors_1q is not None} "
                f"2q={self._errors_2q is not None} readout={self._readout_p:.4f}>")


class _NoiseSpec:
    """Runtime view of a :class:`NoiseModel` used during trajectory simulation."""

    def __init__(self, e1: Optional[QuantumError], e2: Optional[QuantumError],
                 readout_p: float) -> None:
        self.e1 = e1
        self.e2 = e2
        self.readout_p = readout_p

    @staticmethod
    def _random_pauli(psi, q, n, rng, p_x, p_z, p_depol):
        if p_depol > 0.0 and rng.random() < p_depol:
            k = int(rng.integers(1, 4))
            psi = _apply_1q(psi, _PAULIS[k], q, n)
        if p_x > 0.0 and rng.random() < p_x:
            psi = _apply_1q(psi, _X, q, n)
        if p_z > 0.0 and rng.random() < p_z:
            psi = _apply_1q(psi, _Z, q, n)
        return psi

    def apply_1q(self, psi, q, n, rng):
        if self.e1 is None:
            return psi
        return self._random_pauli(psi, q, n, rng, self.e1.p_x, self.e1.p_z, self.e1.p_depol)

    def apply_2q(self, psi, q0, q1, n, rng):
        if self.e2 is None:
            return psi
        for q in (q0, q1):
            psi = self._random_pauli(psi, q, n, rng,
                                     self.e2.p_x, self.e2.p_z, self.e2.p_depol / 2.0)
        return psi


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, statevector: Optional[np.ndarray], counts: Optional[dict]) -> None:
        self._statevector = statevector
        self._counts = counts

    def get_statevector(self, *args, **kwargs) -> np.ndarray:
        if self._statevector is None:
            raise RuntimeError("No statevector was saved for this run.")
        return self._statevector

    def get_counts(self, *args, **kwargs) -> dict:
        if self._counts is None:
            raise RuntimeError("Circuit has no measurements; counts are unavailable.")
        return self._counts


class _Job:
    def __init__(self, result: _Result) -> None:
        self._result = result

    def result(self) -> _Result:
        return self._result


class AerSimulator:
    """NumPy statevector simulator with an Aer-compatible ``run`` API."""

    #: Noise trajectories sampled per run, keyed by circuit width.
    TRAJECTORY_BUDGET = ((8, 24), (12, 8), (16, 3))

    def __init__(self, method: str = "statevector", noise_model: Optional[NoiseModel] = None,
                 seed_simulator: Optional[int] = None, **kwargs) -> None:
        self.method = method
        self.noise_model = noise_model
        self.seed_simulator = seed_simulator

    @classmethod
    def _n_trajectories(cls, n_qubits: int) -> int:
        for limit, traj in cls.TRAJECTORY_BUDGET:
            if n_qubits <= limit:
                return traj
        return 1

    def run(self, circuit, shots: int = 1024, seed_simulator: Optional[int] = None, **kwargs):
        if isinstance(circuit, (list, tuple)):
            circuit = circuit[0]
        n = circuit.num_qubits
        seed = seed_simulator if seed_simulator is not None else self.seed_simulator
        rng = np.random.default_rng(seed)

        noise = self.noise_model._spec() if (
            self.noise_model is not None and not self.noise_model.is_trivial()
        ) else None

        if noise is None:
            psi = _simulate_statevector(circuit)
            probs = np.abs(psi) ** 2
        else:
            psi = None
            n_traj = self._n_trajectories(n)
            probs = np.zeros(2 ** n, dtype=float)
            for _ in range(n_traj):
                traj = _simulate_statevector(circuit, noise=noise, rng=rng)
                probs += np.abs(traj) ** 2
            probs /= n_traj

        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(2 ** n, 1.0 / 2 ** n)

        counts = None
        meas = _measured_qubits(circuit)
        if meas:
            counts = self._sample_counts(probs, meas, n, int(shots), rng,
                                         noise.readout_p if noise else 0.0)

        statevector = psi if (circuit._save_statevector and psi is not None) else psi
        return _Job(_Result(statevector, counts))

    @staticmethod
    def _sample_counts(probs: np.ndarray, meas: List[int], n: int, shots: int,
                       rng: np.random.Generator, readout_p: float) -> dict:
        draws = rng.choice(len(probs), size=max(1, shots), p=probs)
        bits = ((draws[:, None] >> np.array(meas)[None, :]) & 1).astype(np.uint8)
        if readout_p > 0.0:
            flips = rng.random(bits.shape) < readout_p
            bits = bits ^ flips.astype(np.uint8)
        # Qiskit bitstring convention: leftmost char = highest classical bit.
        keys = ["".join(str(b) for b in row[::-1]) for row in bits]
        counts: Dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        return counts


# ──────────────────────────────────────────────────────────────────────────────
# Pillow circuit diagram renderer
# ──────────────────────────────────────────────────────────────────────────────

_BG = (15, 23, 42)
_WIRE = (148, 163, 184)
_GATE = (30, 64, 175)
_GATE2 = (139, 92, 246)
_TEXT = (255, 255, 255)
_LABEL = (203, 213, 225)


def _render_circuit_png(circuit: QuantumCircuit, scale: int = 1) -> str:
    """Draw ``circuit`` with Pillow and return base64-encoded PNG bytes."""
    from PIL import Image, ImageDraw  # imported lazily: Pillow is always present

    n = circuit.num_qubits
    ops = [i for i in circuit.data if i.operation.name != "measure"]
    # Pack operations into columns so non-overlapping gates share a column.
    columns: List[List[CircuitInstruction]] = []
    occupancy: List[set] = []
    for inst in ops:
        idx = {circuit.find_bit(q) for q in inst.qubits}
        if inst.operation.name == "barrier":
            columns.append([inst])
            occupancy.append(set(range(n)))
            continue
        span = set(range(min(idx), max(idx) + 1))
        placed = False
        for col in range(len(columns) - 1, -1, -1):
            if occupancy[col] & span:
                target = col + 1
                break
        else:
            target = 0
        while target >= len(columns):
            columns.append([])
            occupancy.append(set())
        columns[target].append(inst)
        occupancy[target] |= span
        placed = True

    col_w, row_h = 46, 44
    left, top = 54, 28
    width = left + max(1, len(columns)) * col_w + 30
    height = top + n * row_h + 30
    width, height = min(width, 2400), min(height, 1400)

    img = Image.new("RGB", (width * scale, height * scale), _BG)
    d = ImageDraw.Draw(img)

    def X(c):  # column centre
        return (left + c * col_w + col_w // 2) * scale

    def Y(q):  # wire centre
        return (top + q * row_h + row_h // 2) * scale

    # wires + labels
    for q in range(n):
        d.line([(left * scale, Y(q)), ((width - 20) * scale, Y(q))], fill=_WIRE, width=max(1, scale))
        d.text((14 * scale, (Y(q) // scale - 6) * scale), f"q{q}", fill=_LABEL)

    box = 13 * scale
    for c, col in enumerate(columns):
        for inst in col:
            op = inst.operation
            idx = [circuit.find_bit(q) for q in inst.qubits]
            if op.name == "barrier":
                for y in range(top * scale, (top + n * row_h) * scale, 7 * scale):
                    d.line([(X(c), y), (X(c), y + 3 * scale)], fill=(100, 116, 139), width=scale)
                continue
            if op.name in _TWO_QUBIT and len(idx) == 2:
                y0, y1 = Y(idx[0]), Y(idx[1])
                d.line([(X(c), y0), (X(c), y1)], fill=_GATE2, width=max(1, 2 * scale))
                d.ellipse([X(c) - 4 * scale, y0 - 4 * scale, X(c) + 4 * scale, y0 + 4 * scale],
                          fill=_GATE2)
                if op.name == "cx":
                    d.ellipse([X(c) - 8 * scale, y1 - 8 * scale, X(c) + 8 * scale, y1 + 8 * scale],
                              outline=_GATE2, width=max(1, 2 * scale))
                    d.line([(X(c) - 8 * scale, y1), (X(c) + 8 * scale, y1)], fill=_GATE2, width=scale)
                else:
                    d.ellipse([X(c) - 4 * scale, y1 - 4 * scale, X(c) + 4 * scale, y1 + 4 * scale],
                              fill=_GATE2)
                continue
            y = Y(idx[0])
            d.rectangle([X(c) - box, y - box, X(c) + box, y + box], fill=_GATE, outline=_GATE2)
            d.text((X(c) - 5 * scale, y - 5 * scale), op.name.upper()[:2], fill=_TEXT)

    d.text((14 * scale, 8 * scale),
           f"{circuit.name}  ({n} qubits, {len(ops)} ops)  -  NumPy backend", fill=_LABEL)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
