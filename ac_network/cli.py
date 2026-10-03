"""Command line entry point: python -m ac_network input.json --out outdir"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import CircuitError, load_circuit
from .report import TOLERANCE, verify, write_frequency_csv, write_summary
from .solver import solve_frequency


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ac_network",
        description="Sinusoidal steady-state (phasor) AC network analysis.",
    )
    parser.add_argument("input", help="path to the circuit JSON file")
    parser.add_argument(
        "--out",
        required=True,
        help="output directory (created if missing); must be empty/new to avoid mixing runs",
    )
    return parser


def run(input_path: str, out_path: str) -> int:
    try:
        circuit = load_circuit(input_path)
    except CircuitError as exc:
        print(f"error: illegal circuit: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(out_path)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create output directory {out_dir}: {exc}", file=sys.stderr)
        return 2

    outcomes = []
    any_failure = False
    for index, frequency in enumerate(circuit.frequencies):
        result = solve_frequency(circuit, frequency)
        verified, message = verify(result, circuit)
        if result.ok and verified:
            write_frequency_csv(out_dir, index, circuit, result)
            print(
                f"frequency {frequency:g} Hz: solved, "
                f"max KCL residual {result.max_node_residual:.3e} A, "
                f"power imbalance {abs(result.power_balance_residual):.3e} VA"
            )
        else:
            any_failure = True
            print(
                f"frequency {frequency:g} Hz: FAILED ({message})",
                file=sys.stderr,
            )
        outcomes.append((result, verified, message))

    summary_path = write_summary(out_dir, circuit, outcomes)
    print(f"summary written to {summary_path} (tolerance {TOLERANCE:g})")
    return 1 if any_failure else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args.input, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
