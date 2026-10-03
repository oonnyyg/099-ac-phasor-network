import unittest

from ac_network.model import CircuitError, parse_circuit


def base(**overrides):
    data = {
        "ground": "0",
        "nodes": ["0", "1"],
        "branches": [
            {"id": "V1", "type": "voltage_source", "from": "1", "to": "0",
             "re": 10.0, "im": 0.0},
            {"id": "R1", "type": "resistor", "from": "1", "to": "0",
             "resistance": 5.0},
        ],
        "frequencies": [50.0],
    }
    data.update(overrides)
    return data


class ParseTest(unittest.TestCase):
    def test_valid(self):
        c = parse_circuit(base())
        self.assertEqual(c.ground, "0")
        self.assertEqual(len(c.branches), 2)

    def test_duplicate_node(self):
        with self.assertRaises(CircuitError):
            parse_circuit(base(nodes=["0", "0"]))

    def test_duplicate_branch_id(self):
        b = base()
        b["branches"][1]["id"] = "V1"
        with self.assertRaises(CircuitError):
            parse_circuit(b)

    def test_missing_ground(self):
        with self.assertRaises(CircuitError):
            parse_circuit(base(ground="9"))

    def test_unknown_endpoint(self):
        b = base()
        b["branches"][1]["to"] = "9"
        with self.assertRaises(CircuitError):
            parse_circuit(b)

    def test_self_loop(self):
        b = base()
        b["branches"][1]["to"] = "1"
        with self.assertRaises(CircuitError):
            parse_circuit(b)

    def test_nonpositive_passive(self):
        b = base()
        b["branches"][1]["resistance"] = 0.0
        with self.assertRaises(CircuitError):
            parse_circuit(b)

    def test_nonfinite_source(self):
        b = base()
        b["branches"][0]["re"] = float("nan")
        with self.assertRaises(CircuitError):
            parse_circuit(b)

    def test_nonpositive_frequency(self):
        with self.assertRaises(CircuitError):
            parse_circuit(base(frequencies=[0.0]))

    def test_bad_type(self):
        b = base()
        b["branches"][1]["type"] = "diode"
        with self.assertRaises(CircuitError):
            parse_circuit(b)


if __name__ == "__main__":
    unittest.main()
