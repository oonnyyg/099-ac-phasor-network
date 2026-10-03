# ac-phasor-network

Sinusoidal steady-state analysis of small AC networks using phasors
(RMS convention, SI units). Pure Python standard library.

## Usage

```
python -m ac_network input.json --out OUTPUT_DIR
```

The input JSON describes one circuit and a list of positive analysis
frequencies (Hz). For each frequency the tool writes per-frequency node
and branch CSV files plus a `summary.json` in the output directory.

## Input format

```json
{
  "ground": "0",
  "nodes": ["0", "1", "2"],
  "branches": [
    {"id": "V1", "type": "voltage_source", "from": "1", "to": "0", "re": 10.0, "im": 0.0},
    {"id": "R1", "type": "resistor", "from": "1", "to": "2", "resistance": 50.0},
    {"id": "L1", "type": "inductor", "from": "2", "to": "0", "inductance": 0.1},
    {"id": "C1", "type": "capacitor", "from": "2", "to": "0", "capacitance": 1e-5},
    {"id": "I1", "type": "current_source", "from": "0", "to": "1", "re": 0.0, "im": 0.5}
  ],
  "frequencies": [50.0, 100.0]
}
```

- `nodes` and branch `id`s must each be unique; exactly one `ground`
  node must be given and must exist in `nodes`.
- Each branch connects two distinct existing nodes (`from`, `to`).
- Branch types: `resistor` (`resistance`, ohm), `inductor`
  (`inductance`, H), `capacitor` (`capacitance`, F) — finite and
  positive; `voltage_source` / `current_source` with RMS phasor given
  as finite `re`/`im` parts (V or A).
- Sign convention: branch voltage = V(from) - V(to); positive current
  flows from `from` to `to`. A voltage source fixes that voltage, a
  current source fixes that current. Sources may span any two nodes,
  including two non-ground nodes.
- `frequencies` must be a non-empty list of finite positive values.

## Method

For each frequency f, omega = 2*pi*f. Impedances are R, j*omega*L and
1/(j*omega*C). The solver builds a modified nodal analysis (MNA)
system combining KCL at every non-ground node with voltage-source
constraint equations, and solves it with complex Gaussian elimination
(partial pivoting). Voltage sources between two non-ground nodes are
handled natively.

Each branch absorbs complex power S = V * conj(I) (real = active power
in W, imaginary = reactive power in var; negative means the branch
delivers power). The summary reports the maximum KCL residual and the
sum of all branch complex powers per frequency; well-posed circuits
converge to residuals below 1e-8 (absolute or relative).

## Output

- `nodes_<f>.csv` — node voltage phasors (re/im, V) per frequency.
- `branches_<f>.csv` — branch voltage, current and absorbed complex
  power per frequency.
- `summary.json` — per-frequency status, KCL residual and total
  complex power balance.

## Failure handling

- An invalid circuit description rejects the whole run (exit code 2)
  and writes no results.
- If a frequency has no unique solution (floating subnetwork,
  contradictory sources), that frequency is marked `failed` in the
  summary and the remaining frequencies are still solved. Any failed
  frequency makes the process exit with code 1. No artificial
  grounding or fictitious resistors are introduced.

## Tests

```
python -m unittest discover -s tests
```
