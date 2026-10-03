"""CSV / JSON reporting and accuracy verification."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .model import Circuit
from .solver import FrequencyResult

TOLERANCE = 1e-8


def _phasor_parts(value: complex) -> dict[str, float]:
    mag, phase = abs(value), math.degrees(math.atan2(value.imag, value.real))
    return {"real": value.real, "imag": value.imag, "magnitude": mag, "phase_deg": phase}


def verify(result: FrequencyResult, circuit: Circuit) -> tuple[bool, str]:
    """Check KCL residuals and complex-power balance against absolute/relative 1e-8."""
    if not result.ok:
        return False, result.error
    current_scale = 1.0
    for current in result.branch_current.values():
        current_scale = max(current_scale, abs(current))
    if result.max_node_residual > TOLERANCE * current_scale:
        return False, (
            f"node current residual {result.max_node_residual:.3e} exceeds tolerance "
            f"(scale {current_scale:.3e})"
        )
    power_scale = 1.0 + sum(abs(p) for p in result.branch_power.values())
    if abs(result.power_balance_residual) > TOLERANCE * power_scale:
        return False, (
            f"power balance residual {abs(result.power_balance_residual):.3e} "
            f"exceeds tolerance"
        )
    return True, ""


def write_frequency_csv(
    out_dir: Path, index: int, circuit: Circuit, result: FrequencyResult
) -> None:
    tag = f"f{index:03d}_{result.frequency:g}hz"
    nodes_path = out_dir / f"nodes_{tag}.csv"
    with nodes_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frequency_hz",
                "node",
                "is_ground",
                "voltage_re_v",
                "voltage_im_v",
                "voltage_mag_v",
                "voltage_phase_deg",
                "kcl_residual_re_a",
                "kcl_residual_im_a",
                "kcl_residual_abs_a",
            ]
        )
        for node in circuit.nodes:
            voltage = result.node_voltage[node]
            residual = result.node_residual[node]
            vp = _phasor_parts(voltage)
            writer.writerow(
                [
                    result.frequency,
                    node,
                    int(node == circuit.ground),
                    repr(vp["real"]),
                    repr(vp["imag"]),
                    repr(vp["magnitude"]),
                    repr(vp["phase_deg"]),
                    repr(residual.real),
                    repr(residual.imag),
                    repr(abs(residual)),
                ]
            )

    branches_path = out_dir / f"branches_{tag}.csv"
    with branches_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frequency_hz",
                "branch",
                "type",
                "from",
                "to",
                "impedance_re_ohm",
                "impedance_im_ohm",
                "voltage_re_v",
                "voltage_im_v",
                "current_re_a",
                "current_im_a",
                "active_power_w",
                "reactive_power_var",
            ]
        )
        for branch in circuit.branches:
            voltage = result.branch_voltage.get(branch.id, 0.0j)
            current = result.branch_current.get(branch.id, 0.0j)
            power = result.branch_power.get(branch.id, 0.0j)
            z = result.branch_impedance.get(branch.id)
            writer.writerow(
                [
                    result.frequency,
                    branch.id,
                    branch.kind,
                    branch.n_from,
                    branch.n_to,
                    "" if z is None else repr(z.real),
                    "" if z is None else repr(z.imag),
                    repr(voltage.real),
                    repr(voltage.imag),
                    repr(current.real),
                    repr(current.imag),
                    repr(power.real),
                    repr(power.imag),
                ]
            )


def _result_summary(result: FrequencyResult, verified: bool, message: str) -> dict:
    entry: dict = {
        "frequency_hz": result.frequency,
        "omega_rad_s": result.omega,
        "status": "ok" if verified else "failed",
    }
    if not verified:
        entry["error"] = message
        return entry
    entry["nodes"] = {
        node: _phasor_parts(voltage) for node, voltage in result.node_voltage.items()
    }
    entry["branches"] = {
        bid: {
            "voltage": _phasor_parts(result.branch_voltage[bid]),
            "current": _phasor_parts(result.branch_current[bid]),
            "complex_power": {
                "active_w": result.branch_power[bid].real,
                "reactive_var": result.branch_power[bid].imag,
            },
        }
        for bid in result.branch_power
    }
    entry["checks"] = {
        "max_node_residual_abs_a": result.max_node_residual,
        "total_complex_power": {
            "active_w": result.total_power.real,
            "reactive_var": result.total_power.imag,
        },
        "power_balance_residual_abs_va": abs(result.power_balance_residual),
        "tolerance": TOLERANCE,
    }
    return entry


def write_summary(
    out_dir: Path,
    circuit: Circuit,
    outcomes: list[tuple[FrequencyResult, bool, str]],
) -> Path:
    summary = {
        "name": circuit.name,
        "ground": circuit.ground,
        "nodes": list(circuit.nodes),
        "branches": [
            {
                "id": b.id,
                "type": b.kind,
                "from": b.n_from,
                "to": b.n_to,
                "r": b.value if b.kind == "resistor" else None,
                "l": b.value if b.kind == "inductor" else None,
                "c": b.value if b.kind == "capacitor" else None,
                "phasor": {"real": b.vs.real, "imag": b.vs.imag}
                if b.kind in ("voltage_source", "current_source")
                else None,
            }
            for b in circuit.branches
        ],
        "results": [
            _result_summary(result, verified, message)
            for result, verified, message in outcomes
        ],
        "all_ok": all(verified for _, verified, _ in outcomes),
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return path
