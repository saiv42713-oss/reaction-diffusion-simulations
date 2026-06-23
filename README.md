# Activator-Inhibitor Reaction-Diffusion Simulations
## Description
Simulations for an activator-inhibitor pair of partial differential equations representing a reaction-diffusion circuit. The activator can be of two forms: juxtacrine, or paracrine. In the former, it travels in space through neighbour activation, in the latter, through diffusion. The inhibitor is always diffusible. This repository contains code for running 1D and 2D simulations of a single circuit. Additionally, it enables running two independent or cross-inhibiting circuits in two dimensions.

## Structure
- 1D simulations
   - single runs
   - 1D_batch
- 2D simulations
   - single runs
   - 2D_batch
## Usage
Single, 1D simulations: python 1D simulations/main.py

With output 3D visualization: python 1D simulations/run_3D_surface_visualization.py --output test

Batch 1D simulations: python 1D simulations/1D_batch/batch_runner.py -c 1D simulations/1D_batch/config.yaml

Single 2D simulations for a single circuit: python simulations/main.py

Batch 2D simulations for a single circuit: python 2D_simulations/2D_batch/batch_runner_2D.py

Single 2D simulations for a dual circuit: python 2D simulations/main_coupled_2D.py

Batch 2D simulations for coupled circuits: python 2D_simulations/2D_batch/batch_coupled_2D.py

# JAPI Circuit Enumerator
A computational framework for enumerating, filtering, and evaluating
genetic circuit designs for biological patterning, based on Fig S15
from Kunnan et al. 2026 (Morsut Lab, USC).

---

## What This Does

### Phase 1 — Circuit Enumeration (Complete)
1. Generates all 21,609 possible 2-node circuit designs
2. Filters to 1,849 biologically valid circuits
3. Stores each circuit as a NetworkX graph and JSON object
4. Runs a Linear Stability Analysis (LSA) Turing check on each circuit
5. Draws top candidates as diagrams matching Fig S15 notation style
6. JAPI circuit confirmed at index 1285

### Phase 2 — Pattern Classification (Complete)
1. Replicates Fig. 1C from the paper (juxtacrine + paracrine rows, all 10 panels)
   — confirmed by PI (Leonardo Morsut)
2. Implements a dual-simulation regime classifier:
   - Sim 1: all_off + nucleation → tests spontaneous pattern formation
   - Sim 2: spike_steady_state → tests ON state stability
   - Classifies each parameter set as ON / IRREGULAR / OFF / TURING
3. Runs a 5×8 parameter sweep across ba and bi space (na=10, ni=4, Di=20)
4. Generates a color-coded parameter space map showing regime boundaries

---

## Output Files
- `valid_circuits.json` - all 1,849 valid circuits
- `turing_results.json` - Turing pass/fail for each circuit
- `circuit_top1.png` through `circuit_top5.png` — top Turing candidate diagrams
- `japi_circuit.png` - JAPI circuit diagram with legend
- `fig1c_replication.png`- validated Fig 1C replication (10 panels)
- `param_map.png` - regime map across ba/bi parameter space

---

## Setup

1. Clone this repo:
git clone https://github.com/saiv42713-oss/JAPI_Genecircuit_USC.git

2. Clone the lab simulation code:
git clone https://github.com/BenSwedlund/reaction_diffusion_simulations

3. Copy required files into this project folder:
cp reaction_diffusion_simulations/1D_simulations/finding_steady_states.py .
cp reaction_diffusion_simulations/1D_simulations/parameters.py .
cp reaction_diffusion_simulations/2D_simulations/simulation_2D.py .
cp reaction_diffusion_simulations/2D_simulations/parameters_2D.py .
cp reaction_diffusion_simulations/2D_simulations/visualize_2D.py .
cp reaction_diffusion_simulations/2D_simulations/res_analysis/radial_autocor_npz.py .

4. Install dependencies:
pip install networkx numpy matplotlib scipy scikit-image joblib tqdm pandas pyyaml

---

## How to Run

```bash
# Phase 1 — full circuit enumeration pipeline
python3 JAPI_Circuit.py

# Phase 2 — replicate Fig. 1C (na=ni=3, juxtacrine + paracrine)
python3 fig1c.py

# Phase 2 — run regime classifier calibration
python3 calibrate.py

# Phase 2 — run parameter sweep and generate regime map
python3 sweep.py
```

---

## File Structure

JAPI_Circuit.py              - Phase 1: enumeration, LSA, circuit diagrams

classify_regime.py           - Phase 2: dual-sim regime classifier

fig1c.py                     - Phase 2: Fig 1C replication (na=ni=3)

sweep.py                     - Phase 2: parameter sweep + regime map

calibrate.py                 - Phase 2: pattern metric calibration

pattern_metrics_reference.py - PI-provided pattern descriptor (describe_pattern)

valid_circuits.json          - 1,849 valid circuits

turing_results.json          - LSA Turing check results per circuit

param_map.png                - regime map (ba vs bi, na=10, ni=4, Di=20)

fig1c_replication.png        - validated Fig 1C replication

---

## Classifier Logic (classify_regime.py)

Each parameter set runs two simulations:

| Simulation | Init mode | Purpose |
|---|---|---|
| Sim 1 | `all_off` + nucleation | Can patterns emerge spontaneously? |
| Sim 2 | `spike_steady_state` | Can the ON state sustain itself? |

Classification order:
1. **ON** - nucleation sim converges to uniform high-A (CV < 0.05)
2. **ON** - spike sim shows >80% of cells above 0.5 × a_ss
3. **TURING** - autocorrelation oscillates + FFT ratio > 40
4. **IRREGULAR** - autocorrelation oscillates + FFT ratio ≤ 40
5. **OFF** - spike sim collapses to zero
6. **IRREGULAR** - fallback

---

## Parameter Space (Phase 2 Sweep)

Fixed parameters: `na=10, ni=4, Di=20, db=0.5, da=1, juxtacrine`

| Region | Condition |
|---|---|
| OFF | Low ba (activator too weak) |
| IRREGULAR | Balanced ba/bi — activator nucleates but inhibitor limits spread |
| ON | High ba, low bi (activator dominates) |

Turing patterns are rare at na=10, ni=4, the sharp switch forces cells to be
fully ON or OFF, breaking the gradual gradients Turing requires. This matches
the paper's finding that ~99% of non-homogeneous patterns are Irregular at
experimentally measured Hill coefficients.

---

## Current Status

### Complete ✓
- Circuit enumeration (21,609 → 1,849 valid)
- JSON circuit library and NetworkX graphs
- LSA Turing checker (validated on JAPI at index 1285 and PAPI)
- Fig S15-style circuit diagrams with legend
- Fig 1C replication - PI confirmed
- Dual-sim regime classifier (ON / IRREGULAR / OFF / TURING)
- 5×8 parameter sweep with color-coded regime map

### Next Steps
- Connect Phase 1 to Phase 2: run each of the 1,849 circuits through
  the regime classifier to identify which topologies produce IRREGULAR patterns
- Confirm circuit-specific parameter sets with PI (currently using JAPI params)
- LLM scorer integration (pending PI criteria)
- Extend enumeration to 3-node circuits if PI confirms scope

---

## Reference
Kunnan, Swedlund, Ben Tahar, Morsut.
"A novel reaction-diffusion architecture for engineering
self-organized patterns in mammalian cells." bioRxiv 2026.
https://doi.org/10.64898/2026.05.24.727552