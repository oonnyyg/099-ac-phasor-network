import cmath
import math
import unittest

from ac_network.model import parse_circuit
from ac_network.solver import SingularError, solve_frequency


def make_circuit(branches, nodes=("0", "1", "2"), ground="0", freqs=(50.0,)):
    return parse_circuit({
        "ground": ground,
        "nodes": list(nodes),
        "branches": branches,
        "frequencies": list(freqs),
    })


class SeriesRlcTest(unittest.TestCase):
    def test_series_rlc_matches_analytic(self):
        r, l, c, f = 10.0, 0.05, 100e-6, 60.0
        circuit = make_circuit([
            {"id": "V1", "type": "voltage_source", "from": "1", "to": "0",
             "re": 100.0, "im": 0.0},
            {"id": "R1", "type": "resistor", "from": "1", "to": "2", "resistance": r},
            {"id": "L1", "type": "inductor", "from": "2", "to": "0", "inductance": l},
            {"id": "C1", "type": "capacitor", "from": "1", "to": "0", "capacitance": c},
        ], freqs=(f,))
        res = solve_frequency(circuit, f)

        w = 2 * math.pi * f
        z_rl = r + 1j * w * l
        i_rl = 100.0 / z_rl
        i_c = 100.0 * 1j * w * c
        branches = {b.id: b for b in res.branches}
        self.assertAlmostEqual(branches["R1"].current.real, i_rl.real, places=9)
        self.assertAlmostEqual(branches["R1"].current.imag, i_rl.imag, places=9)
        self.assertAlmostEqual(branches["C1"].current.real, i_c.real, places=9)
        self.assertAlmostEqual(branches["C1"].current.imag, i_c.imag, places=9)
        # Resistor absorbs only real power; L and C only reactive.
        self.assertAlmostEqual(branches["L1"].power.real, 0.0, places=9)
        self.assertAlmostEqual(branches["C1"].power.real, 0.0, places=9)
        self.assertAlmostEqual(branches["R1"].power.imag, 0.0, places=9)
        self.assertGreater(branches["R1"].power.real, 0.0)
        # Source delivers (negative absorbed real power).
        self.assertLess(branches["V1"].power.real, 0.0)
        # Global balance.
        self.assertLess(abs(res.total_power), 1e-8)
        self.assertLess(abs(res.kcl_residual), 1e-8)


class FloatingSourceTest(unittest.TestCase):
    def test_voltage_source_between_non_ground_nodes(self):
        circuit = make_circuit([
            {"id": "V1", "type": "voltage_source", "from": "1", "to": "0",
             "re": 12.0, "im": 0.0},
            {"id": "V2", "type": "voltage_source", "from": "2", "to": "3",
             "re": 5.0, "im": 0.0},
            {"id": "R1", "type": "resistor", "from": "1", "to": "2", "resistance": 10.0},
            {"id": "R2", "type": "resistor", "from": "3", "to": "0", "resistance": 20.0},
        ], nodes=("0", "1", "2", "3"))
        res = solve_frequency(circuit, 50.0)
        v = res.node_voltages
        self.assertAlmostEqual((v["2"] - v["3"]).real, 5.0, places=9)
        self.assertAlmostEqual(v["1"].real, 12.0, places=9)
        self.assertLess(abs(res.total_power), 1e-8)
        self.assertLess(abs(res.kcl_residual), 1e-8)

    def test_current_source(self):
        circuit = make_circuit([
            {"id": "I1", "type": "current_source", "from": "1", "to": "0",
             "re": 0.0, "im": 2.0},
            {"id": "R1", "type": "resistor", "from": "1", "to": "0", "resistance": 4.0},
        ], nodes=("0", "1"))
        res = solve_frequency(circuit, 50.0)
        self.assertAlmostEqual(res.node_voltages["1"].imag, -8.0, places=9)
        self.assertLess(abs(res.total_power), 1e-8)


class SingularTest(unittest.TestCase):
    def test_conflicting_voltage_sources_fail(self):
        circuit = make_circuit([
            {"id": "V1", "type": "voltage_source", "from": "1", "to": "0",
             "re": 10.0, "im": 0.0},
            {"id": "V2", "type": "voltage_source", "from": "1", "to": "0",
             "re": 5.0, "im": 0.0},
        ], nodes=("0", "1"))
        with self.assertRaises(SingularError):
            solve_frequency(circuit, 50.0)

    def test_floating_subnetwork_fails(self):
        circuit = make_circuit([
            {"id": "V1", "type": "voltage_source", "from": "1", "to": "0",
             "re": 10.0, "im": 0.0},
            {"id": "R1", "type": "resistor", "from": "2", "to": "3", "resistance": 10.0},
        ], nodes=("0", "1", "2", "3"))
        with self.assertRaises(SingularError):
            solve_frequency(circuit, 50.0)


if __name__ == "__main__":
    unittest.main()
