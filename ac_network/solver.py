"""Modified nodal analysis for sinusoidal steady state (complex, RMS phasors)."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

from .model import Circuit


class SingularError(RuntimeError):
    """Raised when the MNA system has no unique solution at a frequency."""


@dataclass
class BranchResult:
    id: str
    type: str
    from_node: str
    to_node: str
    voltage: complex
    current: complex
    power: complex  # absorbed complex power S = V * conj(I)


@dataclass
class FrequencyResult:
    frequency: float
    node_voltages: dict
    branches: list
    kcl_residual: complex      # max abs KCL residual over all nodes
    total_power: complex       # sum of absorbed complex power over branches


def _impedance(branch, omega: float) -> complex:
    if branch.type == "resistor":
        return complex(branch.value, 0.0)
    if branch.type == "inductor":
        return complex(0.0, omega * branch.value)
    return 1.0 / (1j * omega * branch.value)  # capacitor


def _solve(a: list, b: list) -> list:
    """Gaussian elimination with partial pivoting over complex numbers."""
    n = len(b)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) == 0.0:
            raise SingularError("MNA matrix is singular (no unique solution)")
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        pivot = a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] / pivot
            if factor == 0.0:
                continue
            for k in range(col, n):
                a[row][k] -= factor * a[col][k]
            b[row] -= factor * b[col]
    x = [0j] * n
    for row in range(n - 1, -1, -1):
        s = b[row] - sum(a[row][k] * x[k] for k in range(row + 1, n))
        if abs(a[row][row]) == 0.0:
            raise SingularError("MNA matrix is singular (no unique solution)")
        x[row] = s / a[row][row]
    return x


def solve_frequency(circuit: Circuit, frequency: float) -> FrequencyResult:
    omega = 2.0 * math.pi * frequency
    nodes = [n for n in circuit.nodes if n != circuit.ground]
    nidx = {n: i for i, n in enumerate(nodes)}
    vsrcs = [b for b in circuit.branches if b.type == "voltage_source"]
    vidx = {b.id: len(nodes) + i for i, b in enumerate(vsrcs)}
    size = len(nodes) + len(vsrcs)

    a = [[0j] * size for _ in range(size)]
    rhs = [0j] * size

    def stamp_admittance(b, y):
        i, j = nidx.get(b.from_node), nidx.get(b.to_node)
        if i is not None:
            a[i][i] += y
        if j is not None:
            a[j][j] += y
        if i is not None and j is not None:
            a[i][j] -= y
            a[j][i] -= y

    for b in circuit.branches:
        if b.type in ("resistor", "inductor", "capacitor"):
            stamp_admittance(b, 1.0 / _impedance(b, omega))
        elif b.type == "current_source":
            # Current from -> to; inject into KCL.
            i, j = nidx.get(b.from_node), nidx.get(b.to_node)
            if i is not None:
                rhs[i] -= b.phasor
            if j is not None:
                rhs[j] += b.phasor
        else:  # voltage_source
            k = vidx[b.id]
            i, j = nidx.get(b.from_node), nidx.get(b.to_node)
            if i is not None:
                a[i][k] += 1.0
                a[k][i] += 1.0
            if j is not None:
                a[j][k] -= 1.0
                a[k][j] -= 1.0
            rhs[k] += b.phasor

    x = _solve(a, rhs)

    voltages = {circuit.ground: 0j}
    for n, i in nidx.items():
        voltages[n] = x[i]
    vsrc_current = {b.id: x[vidx[b.id]] for b in vsrcs}

    branches = []
    for b in circuit.branches:
        v = voltages[b.from_node] - voltages[b.to_node]
        if b.type in ("resistor", "inductor", "capacitor"):
            cur = v / _impedance(b, omega)
        elif b.type == "current_source":
            cur = b.phasor
        else:
            cur = vsrc_current[b.id]
        branches.append(BranchResult(b.id, b.type, b.from_node, b.to_node,
                                     v, cur, v * cur.conjugate()))

    residual = 0j
    for n in circuit.nodes:
        net = 0j
        for br in branches:
            if br.from_node == n:
                net += br.current
            elif br.to_node == n:
                net -= br.current
        if abs(net) > abs(residual):
            residual = net

    total = sum((br.power for br in branches), 0j)
    return FrequencyResult(frequency, voltages, branches, residual, total)
