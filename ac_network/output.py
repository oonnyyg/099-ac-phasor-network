"""CSV and JSON output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .solver import FrequencyResult


def _fmt(z: complex) -> tuple:
    return (f"{z.real:.12e}", f"{z.imag:.12e}")


def freq_tag(frequency: float) -> str:
    return f"{frequency:g}".replace("-", "m").replace(".", "p")


def write_frequency(out_dir: Path, result: FrequencyResult) -> None:
    tag = freq_tag(result.frequency)

    nodes_path = out_dir / f"nodes_{tag}.csv"
    with nodes_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "node", "voltage_re_v", "voltage_im_v"])
        for node, v in result.node_voltages.items():
            re, im = _fmt(v)
            w.writerow([f"{result.frequency:g}", node, re, im])

    branches_path = out_dir / f"branches_{tag}.csv"
    with branches_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "branch", "type", "from", "to",
                    "voltage_re_v", "voltage_im_v",
                    "current_re_a", "current_im_a",
                    "power_re_w", "power_im_var"])
        for br in result.branches:
            vre, vim = _fmt(br.voltage)
            ire, iim = _fmt(br.current)
            pre, pim = _fmt(br.power)
            w.writerow([f"{result.frequency:g}", br.id, br.type,
                        br.from_node, br.to_node, vre, vim, ire, iim, pre, pim])


def build_summary(results: list, failures: list) -> dict:
    freqs = []
    for r in results:
        freqs.append({
            "frequency_hz": r.frequency,
            "status": "ok",
            "kcl_residual_a": {"re": r.kcl_residual.real, "im": r.kcl_residual.imag},
            "total_power": {"re_w": r.total_power.real, "im_var": r.total_power.imag},
        })
    for f, reason in failures:
        freqs.append({"frequency_hz": f, "status": "failed", "reason": reason})
    freqs.sort(key=lambda e: e["frequency_hz"])
    return {
        "units": "SI",
        "phasor_convention": "RMS",
        "frequencies": freqs,
    }


def write_summary(out_dir: Path, summary: dict) -> None:
    path = out_dir / "summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
