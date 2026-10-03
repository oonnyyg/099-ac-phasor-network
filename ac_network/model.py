"""Circuit data model and JSON input validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CircuitError(ValueError):
    """Raised for an illegal circuit description (whole batch is rejected)."""


PASSIVE_KINDS = ("resistor", "inductor", "capacitor")
SOURCE_KINDS = ("voltage_source", "current_source")


@dataclass(frozen=True)
class Branch:
    id: str
    kind: str  # resistor | inductor | capacitor | voltage_source | current_source
    n_from: str
    n_to: str
    value: float  # R (ohm), L (henry), C (farad)
    vs: complex  # source phasor (V for voltage source, A for current source)


@dataclass(frozen=True)
class Circuit:
    ground: str
    nodes: tuple[str, ...]
    branches: tuple[Branch, ...]
    frequencies: tuple[float, ...]
    name: str = ""


def _is_real_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _parse_phasor(raw: Any, label: str) -> complex:
    if not isinstance(raw, dict):
        raise CircuitError(f"{label}: source phasor must be an object with 'real' and 'imag'")
    re = raw.get("real", 0.0)
    im = raw.get("imag", 0.0)
    if not _is_real_number(re) or not _is_real_number(im):
        raise CircuitError(f"{label}: phasor real/imag parts must be finite real numbers")
    return complex(float(re), float(im))


def load_circuit(path: str | Path) -> Circuit:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise CircuitError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CircuitError(f"invalid JSON in {path}: {exc}") from exc
    return parse_circuit(data)


def parse_circuit(data: Any) -> Circuit:
    if not isinstance(data, dict):
        raise CircuitError("top-level JSON must be an object")

    name = data.get("name", "")
    if not isinstance(name, str):
        raise CircuitError("'name' must be a string")

    ground = data.get("ground")
    if not isinstance(ground, str) or not ground:
        raise CircuitError("'ground' must be a non-empty node id string")

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise CircuitError("'nodes' must be a non-empty array")
    nodes: list[str] = []
    seen_nodes: set[str] = set()
    for item in raw_nodes:
        if isinstance(item, str):
            nid = item
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            nid = item["id"]
        else:
            raise CircuitError("each node must be a string id or an object with string 'id'")
        if not nid:
            raise CircuitError("node id must be non-empty")
        if nid in seen_nodes:
            raise CircuitError(f"duplicate node id: {nid!r}")
        seen_nodes.add(nid)
        nodes.append(nid)
    if ground not in seen_nodes:
        raise CircuitError(f"ground node {ground!r} is not among declared nodes")

    raw_branches = data.get("branches")
    if not isinstance(raw_branches, list):
        raise CircuitError("'branches' must be an array")
    branches: list[Branch] = []
    seen_branches: set[str] = set()
    for i, item in enumerate(raw_branches):
        label = f"branch[{i}]"
        if not isinstance(item, dict):
            raise CircuitError(f"{label} must be an object")
        bid = item.get("id")
        if not isinstance(bid, str) or not bid:
            raise CircuitError(f"{label}: 'id' must be a non-empty string")
        if bid in seen_branches:
            raise CircuitError(f"duplicate branch id: {bid!r}")
        seen_branches.add(bid)

        kind = item.get("type") or item.get("kind")
        if kind not in PASSIVE_KINDS + SOURCE_KINDS:
            raise CircuitError(
                f"{label} ({bid!r}): type must be one of "
                f"{PASSIVE_KINDS + SOURCE_KINDS}"
            )
        n_from = item.get("from")
        n_to = item.get("to")
        if not isinstance(n_from, str) or not isinstance(n_to, str):
            raise CircuitError(f"{label} ({bid!r}): 'from' and 'to' must be node id strings")
        if n_from not in seen_nodes:
            raise CircuitError(f"{label} ({bid!r}): 'from' node {n_from!r} does not exist")
        if n_to not in seen_nodes:
            raise CircuitError(f"{label} ({bid!r}): 'to' node {n_to!r} does not exist")
        if n_from == n_to:
            raise CircuitError(f"{label} ({bid!r}): branch must connect two different nodes")

        value = 0.0
        vs = 0.0j
        if kind in PASSIVE_KINDS:
            param_name = {"resistor": "r", "inductor": "l", "capacitor": "c"}[kind]
            raw_val = item.get(param_name)
            if raw_val is None:
                raw_val = item.get("value")
            if not _is_real_number(raw_val) or float(raw_val) <= 0.0:
                raise CircuitError(
                    f"{label} ({bid!r}): {param_name!r} must be a finite positive number"
                )
            value = float(raw_val)
        else:
            vs = _parse_phasor(item.get("phasor", item), f"{label} ({bid!r})")

        branches.append(Branch(bid, kind, n_from, n_to, value, vs))

    raw_freqs = data.get("frequencies")
    if not isinstance(raw_freqs, list) or not raw_freqs:
        raise CircuitError("'frequencies' must be a non-empty array of positive frequencies (Hz)")
    frequencies: list[float] = []
    seen_freqs: set[float] = set()
    for item in raw_freqs:
        if not _is_real_number(item) or float(item) <= 0.0:
            raise CircuitError("each frequency must be a finite positive number (Hz)")
        f = float(item)
        if f in seen_freqs:
            raise CircuitError(f"duplicate frequency: {f}")
        seen_freqs.add(f)
        frequencies.append(f)

    return Circuit(
        ground=ground,
        nodes=tuple(nodes),
        branches=tuple(branches),
        frequencies=tuple(frequencies),
        name=name,
    )
