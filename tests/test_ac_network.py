from __future__ import annotations

import cmath
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ac_network.model import CircuitError, parse_circuit
from ac_network.report import verify
from ac_network.solver import branch_impedance, solve_frequency


def make_circuit_dict(branches, nodes=("g", "n1", "n2"), frequencies=(50.0,), ground="g"):
    return {
        "ground": ground,
        "nodes": list(nodes),
        "branches": branches,
        "frequencies": list(frequencies),
    }


class TestValidation(unittest.TestCase):
    def test_missing_ground_rejected(self):
        with self.assertRaises(CircuitError):
            parse_circuit({"nodes": ["a"], "branches": [], "frequencies": [50]})

    def test_ground_not_in_nodes_rejected(self):
        with self.assertRaises(CircuitError):
            parse_circuit(
                make_circuit_dict([], nodes=("a", "b"), ground="g")
            )

    def test_duplicate_node_id(self):
        data = make_circuit_dict([], nodes=("g", "g"))
        with self.assertRaises(CircuitError):
            parse_circuit(data)

    def test_duplicate_branch_id(self):
        branches = [
            {"id": "b1", "type": "resistor", "from": "n1", "to": "n2", "r": 1},
            {"id": "b1", "type": "resistor", "from": "n1", "to": "g", "r": 1},
        ]
        with self.assertRaises(CircuitError):
            parse_circuit(make_circuit_dict(branches))

    def test_self_loop_rejected(self):
        branches = [{"id": "b1", "type": "resistor", "from": "n1", "to": "n1", "r": 1}]
        with self.assertRaises(CircuitError):
            parse_circuit(make_circuit_dict(branches))

    def test_unknown_endpoint_rejected(self):
        branches = [{"id": "b1", "type": "resistor", "from": "n1", "to": "zzz", "r": 1}]
        with self.assertRaises(CircuitError):
            parse_circuit(make_circuit_dict(branches))

    def test_nonpositive_parameter_rejected(self):
        for value in (0, -1):
            branches = [
                {"id": "b1", "type": "resistor", "from": "n1", "to": "g", "r": value}
            ]
            with self.assertRaises(CircuitError):
                parse_circuit(make_circuit_dict(branches))

    def test_nonfinite_parameter_rejected(self):
        branches = [
            {"id": "b1", "type": "inductor", "from": "n1", "to": "g", "l": "nan"}
        ]
        data = make_circuit_dict(branches)
        with self.assertRaises(CircuitError):
            parse_circuit(data)

    def test_bad_frequency_rejected(self):
        with self.assertRaises(CircuitError):
            parse_circuit(make_circuit_dict([], frequencies=(0,)))
        with self.assertRaises(CircuitError):
            parse_circuit(make_circuit_dict([], frequencies=(-50,)))


class TestSolver(unittest.TestCase):
    def _verified(self, data, frequency=50.0):
        circuit = parse_circuit(data)
        result = solve_frequency(circuit, frequency)
        self.assertTrue(result.ok, msg=result.error)
        ok, message = verify(result, circuit)
        self.assertTrue(ok, msg=message)
        return circuit, result

    def test_pure_resistor(self):
        branches = [
            {"id": "vs", "type": "voltage_source", "from": "n1", "to": "g",
             "phasor": {"real": 10, "imag": 0}},
            {"id": "r", "type": "resistor", "from": "n1", "to": "g", "r": 4},
        ]
        _, result = self._verified(make_circuit_dict(branches, nodes=("g", "n1")))
        self.assertAlmostEqual(result.node_voltage["n1"], 10.0 + 0j)
        self.assertAlmostEqual(result.branch_current["r"], 2.5 + 0j)
        self.assertAlmostEqual(result.branch_power["r"].real, 25.0)
        self.assertAlmostEqual(result.branch_power["vs"].real, -25.0)

    def test_inductor_capacitor_reactance(self):
        branches = [
            {"id": "vs", "type": "voltage_source", "from": "n1", "to": "g",
             "phasor": {"real": 5, "imag": 0}},
            {"id": "l", "type": "inductor", "from": "n1", "to": "g", "l": 1e-3},
        ]
        circuit = parse_circuit(make_circuit_dict(branches, nodes=("g", "n1")))
        result = solve_frequency(circuit, 50.0)
        z = branch_impedance(circuit.branches[1], 2 * math.pi * 50)
        self.assertAlmostEqual(z, complex(0, 2 * math.pi * 50 * 1e-3))
        self.assertTrue(result.ok)
        self.assertAlmostEqual(abs(result.branch_current["l"]), 5.0 / abs(z))

        branches[1] = {"id": "c", "type": "capacitor", "from": "n1", "to": "g", "c": 1e-4}
        circuit = parse_circuit(make_circuit_dict(branches, nodes=("g", "n1")))
        result = solve_frequency(circuit, 50.0)
        z = branch_impedance(circuit.branches[1], 2 * math.pi * 50)
        self.assertAlmostEqual(z, 1 / complex(0, 2 * math.pi * 50 * 1e-4))

    def test_floating_voltage_source_between_ungrounded_nodes(self):
        # n1 and n2 are both floating; a voltage source n1-n2 fixes only their
        # difference, and no path to ground exists -> singular.
        branches = [
            {"id": "vs", "type": "voltage_source", "from": "n1", "to": "n2",
             "phasor": {"real": 10, "imag": 0}},
            {"id": "r", "type": "resistor", "from": "n1", "to": "n2", "r": 2},
        ]
        data = make_circuit_dict(branches, nodes=("g", "n1", "n2"))
        circuit = parse_circuit(data)
        result = solve_frequency(circuit, 50.0)
        self.assertFalse(result.ok)

    def test_floating_source_with_path_to_ground_solves(self):
        # Floating source n1-n2 plus a resistor n2-g gives a unique solution,
        # proving voltage sources are not collapsed to ground.
        branches = [
            {"id": "vs", "type": "voltage_source", "from": "n1", "to": "n2",
             "phasor": {"real": 10, "imag": 0}},
            {"id": "r1", "type": "resistor", "from": "n1", "to": "g", "r": 4},
            {"id": "r2", "type": "resistor", "from": "n2", "to": "g", "r": 6},
        ]
        circuit, result = self._verified(
            make_circuit_dict(branches, nodes=("g", "n1", "n2"))
        )
        v1 = result.node_voltage["n1"]
        v2 = result.node_voltage["n2"]
        self.assertAlmostEqual(v1 - v2, 10.0 + 0j)
        # Source current: KCL at n1 with only r1: I = v1/4 (n1->n2 direction),
        # KCL at n2: -I + v2/6 = 0; v1 - v2 = 10 => solve and compare.
        i_vs = result.branch_current["vs"]
        self.assertAlmostEqual(i_vs, -v1 / 4)
        self.assertAlmostEqual(i_vs, v2 / 6)

    def test_current_source_direction(self):
        branches = [
            {"id": "is", "type": "current_source", "from": "g", "to": "n1",
             "phasor": {"real": 3, "imag": 0}},
            {"id": "r", "type": "resistor", "from": "n1", "to": "g", "r": 2},
        ]
        _, result = self._verified(make_circuit_dict(branches, nodes=("g", "n1")))
        # Current injected into n1 flows back through r: V_n1 = 6 V.
        self.assertAlmostEqual(result.node_voltage["n1"], 6.0 + 0j)
        self.assertAlmostEqual(result.branch_power["is"].real, -18.0)
        self.assertAlmostEqual(result.branch_power["r"].real, 18.0)

    def test_conflicting_voltage_sources(self):
        branches = [
            {"id": "v1", "type": "voltage_source", "from": "n1", "to": "g",
             "phasor": {"real": 5, "imag": 0}},
            {"id": "v2", "type": "voltage_source", "from": "n1", "to": "g",
             "phasor": {"real": 7, "imag": 0}},
        ]
        circuit = parse_circuit(make_circuit_dict(branches, nodes=("g", "n1")))
        result = solve_frequency(circuit, 50.0)
        self.assertFalse(result.ok)

    def test_two_voltage_sources_loop_consistent(self):
        # n1-n2 source and n2-g source imply unique n1 voltage; power balances.
        branches = [
            {"id": "v1", "type": "voltage_source", "from": "n1", "to": "n2",
             "phasor": {"real": 3, "imag": 4}},
            {"id": "v2", "type": "voltage_source", "from": "n2", "to": "g",
             "phasor": {"real": 1, "imag": 0}},
            {"id": "r", "type": "resistor", "from": "n1", "to": "g", "r": 10},
        ]
        _, result = self._verified(
            make_circuit_dict(branches, nodes=("g", "n1", "n2"))
        )
        self.assertAlmostEqual(result.node_voltage["n2"], 1.0 + 0j)
        self.assertAlmostEqual(result.node_voltage["n1"], 4.0 + 4j)

    def test_frequency_dependence_of_rlc(self):
        branches = [
            {"id": "vs", "type": "voltage_source", "from": "n1", "to": "g",
             "phasor": {"real": 10, "imag": 0}},
            {"id": "r", "type": "resistor", "from": "n1", "to": "n2", "r": 2},
            {"id": "l", "type": "inductor", "from": "n2", "to": "g", "l": 1e-3},
            {"id": "c", "type": "capacitor", "from": "n2", "to": "g", "c": 1e-4},
        ]
        data = make_circuit_dict(branches, frequencies=(50.0, 1000.0))
        circuit = parse_circuit(data)
        currents = []
        for f in circuit.frequencies:
            result = solve_frequency(circuit, f)
            self.assertTrue(result.ok)
            ok, message = verify(result, circuit)
            self.assertTrue(ok, msg=message)
            currents.append(result.branch_current["r"])
        self.assertNotAlmostEqual(abs(currents[0]), abs(currents[1]))


class TestCli(unittest.TestCase):
    def test_end_to_end_with_one_failed_frequency(self):
        root = Path(__file__).resolve().parents[1]
        payload = {
            "ground": "g",
            "nodes": ["g", "n1", "n2"],
            "branches": [
                {"id": "vs", "type": "voltage_source", "from": "n1",
                 "to": "n2", "phasor": {"real": 1, "imag": 0}},
                {"id": "r", "type": "resistor", "from": "n1", "to": "n2", "r": 1},
            ],
            "frequencies": [50],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "mixed.json"
            out_dir = tmp_path / "out"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "ac_network", str(input_path), "--out", str(out_dir)],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["all_ok"])
            self.assertEqual(summary["results"][0]["status"], "failed")

    def test_illegal_circuit_writes_nothing(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "bad.json"
            out_dir = tmp_path / "out"
            input_path.write_text(
                json.dumps(
                    {"ground": "g", "nodes": ["g"], "branches": [], "frequencies": []}
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, "-m", "ac_network", str(input_path), "--out", str(out_dir)],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
