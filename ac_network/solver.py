"""Modified-nodal-analysis phasor solver (standard library only)."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from .model import Branch, Circuit


class SolveError(Exception):
    """No unique solution at a frequency (floating network / inconsistent sources)."""


@dataclass(frozen=True)
class FrequencyResult:
    frequency: float
    omega: float
    node_voltage: dict[str, complex]
    branch_voltage: dict[str, complex]
    branch_current: dict[str, complex]
    branch_impedance: dict[str, complex]
    branch_power: dict[str, complex]
    node_residual: dict[str, complex]
    total_power: complex
    max_node_residual: float
    power_balance_residual: complex
    ok: bool
    error: str = ""


def branch_impedance(branch: Branch, omega: float) -> complex:
    if branch.kind == "resistor":
        return complex(branch.value, 0.0)
    if branch.kind == "inductor":
        return complex(0.0, omega * branch.value)
    if branch.kind == "capacitor":
        return 1.0 / complex(0.0, omega * branch.value)
    return complex(math.nan, math.nan)  # sources have no impedance


def _solve_linear(matrix: list[list[complex]], rhs: list[complex]) -> list[complex]:
    """Gaussian elimination with partial pivoting. Raises SolveError if singular."""
    n = len(matrix)
    # Build augmented system; scale-aware pivot test.
    col_scale = [
        max(abs(matrix[r][c]) for r in range(n)) if n else 1.0
        for c in range(n)
    ]
    col_scale = [s if s > 0.0 else 1.0 for s in col_scale]

    for col in range(n):
        pivot_row = max(
            range(col, n),
            key=lambda r: abs(matrix[r][col]) / col_scale[col],
        )
        pivot = matrix[pivot_row][col]
        if abs(pivot) / col_scale[col] < 1e-14:
            raise SolveError("singular system: floating network or conflicting sources")
        if pivot_row != col:
            matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
            rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]
        for r in range(col + 1, n):
            factor = matrix[r][col] / pivot
            if factor == 0.0:
                continue
            matrix[r][col] = 0.0
            for c in range(col + 1, n):
                matrix[r][c] -= factor * matrix[col][c]
            rhs[r] -= factor * rhs[col]

    x = [0.0j] * n
    for r in range(n - 1, -1, -1):
        acc = rhs[r]
        row = matrix[r]
        for c in range(r + 1, n):
            acc -= row[c] * x[c]
        diag = row[r]
        if abs(diag) / col_scale[r] < 1e-14:
            raise SolveError("singular system: floating network or conflicting sources")
        x[r] = acc / diag
    return x


def solve_frequency(circuit: Circuit, frequency: float) -> FrequencyResult:
    omega = 2.0 * math.pi * frequency
    active_nodes = [n for n in circuit.nodes if n != circuit.ground]
    node_index = {n: i for i, n in enumerate(active_nodes)}
    voltage_sources = [b for b in circuit.branches if b.kind == "voltage_source"]
    vs_index = {b.id: k for k, b in enumerate(voltage_sources)}

    n_v = len(active_nodes)
    n_s = len(voltage_sources)
    size = n_v + n_s
    matrix = [[0.0j for _ in range(size)] for _ in range(size)]
    rhs = [0.0j for _ in range(size)]

    for branch in circuit.branches:
        if branch.kind in ("resistor", "inductor", "capacitor"):
            z = branch_impedance(branch, omega)
            y = 1.0 / z
            i_f = node_index.get(branch.n_from)
            i_t = node_index.get(branch.n_to)
            # Current leaving node through this branch: Y*(Vf - Vt).
            if i_f is not None:
                matrix[i_f][i_f] += y
            if i_t is not None:
                matrix[i_t][i_t] += y
            if i_f is not None and i_t is not None:
                matrix[i_f][i_t] -= y
                matrix[i_t][i_f] -= y
        elif branch.kind == "current_source":
            # Fixed current from->to; KCL: passive leaving + I_s = 0.
            i_f = node_index.get(branch.n_from)
            i_t = node_index.get(branch.n_to)
            if i_f is not None:
                rhs[i_f] -= branch.vs
            if i_t is not None:
                rhs[i_t] += branch.vs
        else:  # voltage source: unknown current from->to added to KCL.
            row_k = n_v + vs_index[branch.id]
            i_f = node_index.get(branch.n_from)
            i_t = node_index.get(branch.n_to)
            if i_f is not None:
                matrix[i_f][row_k] += 1.0
            if i_t is not None:
                matrix[i_t][row_k] -= 1.0
            # Constraint: V_from - V_to = Vs.
            if i_f is not None:
                matrix[row_k][i_f] += 1.0
            if i_t is not None:
                matrix[row_k][i_t] -= 1.0
            rhs[row_k] = branch.vs

    try:
        unknowns = _solve_linear(matrix, rhs)
    except SolveError as exc:
        zero = {n: 0.0j for n in circuit.nodes}
        return FrequencyResult(
            frequency=frequency,
            omega=omega,
            node_voltage=zero,
            branch_voltage={},
            branch_current={},
            branch_impedance={},
            branch_power={},
            node_residual=zero,
            total_power=0.0j,
            max_node_residual=math.inf,
            power_balance_residual=0.0j,
            ok=False,
            error=str(exc),
        )

    node_voltage = {circuit.ground: 0.0j}
    for i, n in enumerate(active_nodes):
        node_voltage[n] = unknowns[i]

    branch_voltage: dict[str, complex] = {}
    branch_current: dict[str, complex] = {}
    branch_z: dict[str, complex] = {}
    branch_power: dict[str, complex] = {}
    leaving: dict[str, complex] = {n: 0.0j for n in circuit.nodes}

    for branch in circuit.branches:
        vf = node_voltage[branch.n_from]
        vt = node_voltage[branch.n_to]
        v_branch = vf - vt
        branch_voltage[branch.id] = v_branch
        if branch.kind in ("resistor", "inductor", "capacitor"):
            z = branch_impedance(branch, omega)
            i_branch = v_branch / z
            branch_z[branch.id] = z
        elif branch.kind == "current_source":
            i_branch = branch.vs
        else:
            i_branch = unknowns[n_v + vs_index[branch.id]]
        branch_current[branch.id] = i_branch
        branch_power[branch.id] = v_branch * i_branch.conjugate()
        leaving[branch.n_from] += i_branch
        leaving[branch.n_to] -= i_branch

    total_power = sum(branch_power.values(), 0.0j)
    max_residual = max(abs(v) for v in leaving.values()) if leaving else 0.0

    # Power balance: sum of absorbed complex power over all branches must be ~0.
    return FrequencyResult(
        frequency=frequency,
        omega=omega,
        node_voltage=node_voltage,
        branch_voltage=branch_voltage,
        branch_current=branch_current,
        branch_impedance=branch_z,
        branch_power=branch_power,
        node_residual=leaving,
        total_power=total_power,
        max_node_residual=max_residual,
        power_balance_residual=total_power,
        ok=True,
    )
