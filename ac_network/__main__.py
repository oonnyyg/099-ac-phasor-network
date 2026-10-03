"""Command line entry: python -m ac_network input.json --out OUTPUT_DIR"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import CircuitError, parse_circuit
from .output import build_summary, write_frequency, write_summary
from .solver import SingularError, solve_frequency


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ac_network",
        description="Sinusoidal steady-state (phasor, RMS) analysis of small AC networks.")
    parser.add_argument("input", help="path to the circuit JSON file")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)

    try:
        with open(args.input, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 2

    try:
        circuit = parse_circuit(data)
    except CircuitError as exc:
        print(f"error: invalid circuit: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for f in circuit.frequencies:
        try:
            result = solve_frequency(circuit, f)
        except SingularError as exc:
            failures.append((f, str(exc)))
            print(f"warning: f={f:g} Hz failed: {exc}", file=sys.stderr)
            continue
        write_frequency(out_dir, result)
        results.append(result)

    write_summary(out_dir, build_summary(results, failures))

    if failures:
        print(f"error: {len(failures)} of {len(circuit.frequencies)} "
              f"frequencies failed to solve", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
