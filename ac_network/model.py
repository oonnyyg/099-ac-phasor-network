"""Circuit description parsing and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass


class CircuitError(ValueError):
    """Raised when the circuit description is invalid."""


_PASSIVE_TYPES = {"resistor", "inductor", "capacitor"}
_SOURCE_TYPES = {"voltage_source", "current_source"}
BRANCH_TYPES = _PASSIVE_TYPES | _SOURCE_TYPES


def _finite(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CircuitError(f"{what}: expected a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise CircuitError(f"{what}: value must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class Branch:
    id: str
    type: str
    from_node: str
    to_node: str
    value: float = 0.0          # R / L / C for passive branches
    phasor: complex = 0j        # RMS phasor for sources


@dataclass(frozen=True)
class Circuit:
    ground: str
    nodes: tuple
    branches: tuple
    frequencies: tuple


def _require_str(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CircuitError(f"{what}: expected a non-empty string, got {value!r}")
    return value


def _parse_branch(raw, index: int) -> Branch:
    what = f"branches[{index}]"
    if not isinstance(raw, dict):
        raise CircuitError(f"{what}: expected an object")
    bid = _require_str(raw.get("id"), f"{what}.id")
    btype = raw.get("type")
    if btype not in BRANCH_TYPES:
        raise CircuitError(f"{what}.type: must be one of {sorted(BRANCH_TYPES)}, got {btype!r}")
    from_node = _require_str(raw.get("from"), f"{what}.from")
    to_node = _require_str(raw.get("to"), f"{what}.to")
    if from_node == to_node:
        raise CircuitError(f"{what}: branch {bid!r} connects node {from_node!r} to itself")

    value = 0.0
    phasor = 0j
    if btype in _PASSIVE_TYPES:
        key = {"resistor": "resistance", "inductor": "inductance",
               "capacitor": "capacitance"}[btype]
        value = _finite(raw.get(key), f"{what}.{key}")
        if value <= 0.0:
            raise CircuitError(f"{what}.{key}: must be positive, got {value!r}")
    else:
        re = _finite(raw.get("re", 0.0), f"{what}.re")
        im = _finite(raw.get("im", 0.0), f"{what}.im")
        phasor = complex(re, im)
    return Branch(bid, btype, from_node, to_node, value, phasor)


def parse_circuit(data) -> Circuit:
    if not isinstance(data, dict):
        raise CircuitError("top level: expected a JSON object")

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise CircuitError("nodes: expected a non-empty array")
    nodes = tuple(_require_str(n, f"nodes[{i}]") for i, n in enumerate(raw_nodes))
    if len(set(nodes)) != len(nodes):
        raise CircuitError("nodes: ids must be unique")

    ground = _require_str(data.get("ground"), "ground")
    if ground not in nodes:
        raise CircuitError(f"ground: node {ground!r} is not in nodes")

    raw_branches = data.get("branches")
    if not isinstance(raw_branches, list) or not raw_branches:
        raise CircuitError("branches: expected a non-empty array")
    branches = tuple(_parse_branch(b, i) for i, b in enumerate(raw_branches))
    ids = [b.id for b in branches]
    if len(set(ids)) != len(ids):
        raise CircuitError("branches: ids must be unique")
    node_set = set(nodes)
    for b in branches:
        for end in (b.from_node, b.to_node):
            if end not in node_set:
                raise CircuitError(f"branch {b.id!r}: unknown node {end!r}")

    raw_freqs = data.get("frequencies")
    if not isinstance(raw_freqs, list) or not raw_freqs:
        raise CircuitError("frequencies: expected a non-empty array")
    freqs = tuple(_finite(f, f"frequencies[{i}]") for i, f in enumerate(raw_freqs))
    for f in freqs:
        if f <= 0.0:
            raise CircuitError(f"frequencies: values must be positive, got {f!r}")

    return Circuit(ground, nodes, branches, freqs)
