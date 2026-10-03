import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def run_cli(input_path, out_dir):
    return subprocess.run(
        [sys.executable, "-m", "ac_network", str(input_path), "--out", str(out_dir)],
        capture_output=True, text=True)


def write(tmp, data):
    path = Path(tmp) / "circuit.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


GOOD = {
    "ground": "0",
    "nodes": ["0", "1", "2"],
    "branches": [
        {"id": "V1", "type": "voltage_source", "from": "1", "to": "0",
         "re": 10.0, "im": 0.0},
        {"id": "R1", "type": "resistor", "from": "1", "to": "2", "resistance": 50.0},
        {"id": "L1", "type": "inductor", "from": "2", "to": "0", "inductance": 0.1},
    ],
    "frequencies": [50.0, 100.0],
}


class CliTest(unittest.TestCase):
    def test_success_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = run_cli(write(tmp, GOOD), out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for name in ("nodes_50.csv", "branches_50.csv",
                         "nodes_100.csv", "branches_100.csv", "summary.json"):
                self.assertTrue((out / name).exists(), name)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(all(e["status"] == "ok" for e in summary["frequencies"]))
            for e in summary["frequencies"]:
                self.assertLess(abs(complex(e["kcl_residual_a"]["re"],
                                            e["kcl_residual_a"]["im"])), 1e-8)
                self.assertLess(abs(complex(e["total_power"]["re_w"],
                                            e["total_power"]["im_var"])), 1e-8)
            with (out / "branches_50.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["branch"], "V1")

    def test_invalid_circuit_rejected_without_output(self):
        bad = dict(GOOD, ground="nope")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = run_cli(write(tmp, bad), out)
            self.assertEqual(proc.returncode, 2)
            self.assertFalse(out.exists())

    def test_partial_failure_exit_nonzero(self):
        data = dict(GOOD)
        data["branches"] = GOOD["branches"] + [
            {"id": "V2", "type": "voltage_source", "from": "1", "to": "0",
             "re": 3.0, "im": 0.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = run_cli(write(tmp, data), out)
            self.assertEqual(proc.returncode, 1)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            statuses = {e["status"] for e in summary["frequencies"]}
            self.assertEqual(statuses, {"failed"})


if __name__ == "__main__":
    unittest.main()
