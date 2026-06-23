# Activator-Inhibitor Reaction-Diffusion Simulations

## Description
Simulations for an activator-inhibitor pair of partial differential equations representing a reaction-diffusion circuit. The activator can be of two forms: juxtacrine, or paracrine. In the former, it travels in space through neighbour activation, in the latter, through diffusion. The inhibitor is always diffusible. This repository contains code for running 1D and 2D simulations of a single circuit. Additionally, it enables running two independent or cross-inhibiting circuits in two dimensions.

---

## Repository Structure

```
JAPI_GeneCircuit_USC/
├── core/                          # Python package — simulation engine + classification logic
│   ├── __init__.py
│   ├── JAPI_Circuit.py            # Circuit enumeration, LSA Turing check, diagram drawing
│   ├── classify_regime.py         # Dual-sim regime classifier (ON/TURING/IRREGULAR/OFF)
│   ├── pattern_metrics_reference.py  # PI-provided pattern descriptors (describe_pattern)
│   ├── finding_steady_states.py   # Steady-state solver and stability utilities
│   ├── simulation_2D.py           # 2D hex-grid simulator (juxtacrine + paracrine)
│   ├── radial_autocor_npz.py      # Radial autocorrelation analysis
│   ├── parameters.py              # Default 1D/shared simulation parameters
│   ├── parameters_2D.py           # Default 2D simulation parameters
│   └── visualize_2D.py            # Hex-grid visualization utilities
│
├── scripts/                       # Runnable entry points (run from project root)
│   ├── fig1c.py                   # Replicate Fig. 1C (juxtacrine + paracrine panels)
│   ├── sweep.py                   # Parameter sweep over ba/bi space + regime map
│   ├── calibrate.py               # Pattern metric calibration on known cases
│   ├── circuit_classifier_runner.py  # Batch classify all 1,849 circuits (parallel + checkpoint)
│   ├── pattern_metrics_runner.py  # Batch describe_pattern() over a directory of .npz files
│   ├── grid_sweep.py              # 3-axis grid sweep (bi × hill_coeff × diffusion)
│   └── make_animation.py          # Generate mp4 animations across bi values
│
├── analysis/                      # Post-processing tools for .npz simulation outputs
│   ├── radial_autocor_npz.py      # Radial autocorrelation batch analysis
│   ├── crosscorr_npz.py           # A-B cross-correlation analysis
│   ├── fft_analysis_2D.py         # FFT power spectrum analysis
│   ├── npz_count_activated_hex.py # Count activated hexagonal cells
│   ├── npz_feature_distribution.py  # Feature size/density distributions
│   ├── npz_overlap_analysis.py    # Overlap analysis between fields
│   └── plot_multiple_xcorr.py     # Multi-file cross-correlation plots
│
├── data/                          # Reference and result data files
│   ├── valid_circuits.json        # All 1,849 biologically valid circuits
│   ├── turing_results.json        # LSA Turing pass/fail for each circuit
│   ├── fig1c_results.json         # Fig. 1C simulation results
│   ├── classification_results.csv # Final regime classification for all circuits
│   └── nicole_grid_results.csv    # 3-axis grid sweep results
│
├── outputs/                       # Generated plots, diagrams, and animations
│   ├── fig1c.png                  # Validated Fig. 1C replication (10 panels)
│   ├── japi_circuit.png           # JAPI circuit diagram with legend
│   ├── circuits/                  # Top Turing candidate circuit diagrams
│   │   └── circuit_top1–5.png
│   └── field_*.png / param_map.png  # Sweep field snapshots and regime map
│
├── 1D_simuations/                 # Self-contained 1D simulation suite
├── 2D_simulations/                # Self-contained 2D simulation suite
├── 2D_batch/                      # Batch infrastructure for 2D sweeps
├── Samples/                       # Reference circuit topology images
└── reaction_diffusion_simulations/ # Git submodule (lab simulation base)
```

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
   - Sim 1: `all_off` + nucleation → tests spontaneous pattern formation
   - Sim 2: `spike_steady_state` → tests ON state stability
   - Classifies each parameter set as ON / IRREGULAR / OFF / TURING
3. Runs a 5×8 parameter sweep across ba and bi space (na=10, ni=4, Di=20)
4. Generates a color-coded parameter space map showing regime boundaries

---

## Setup

1. Clone this repo:
```bash
git clone https://github.com/saiv42713-oss/JAPI_Genecircuit_USC.git
cd JAPI_GeneCircuit_USC
```

2. Install dependencies:
```bash
pip install networkx numpy matplotlib scipy scikit-image joblib tqdm pandas pyyaml
```

---

## How to Run

All scripts are run from the **project root**:

```bash
# Phase 1 — full circuit enumeration pipeline
python3 -m core.JAPI_Circuit

# Phase 2 — replicate Fig. 1C
python3 scripts/fig1c.py

# Phase 2 — run regime classifier calibration
python3 scripts/calibrate.py

# Phase 2 — run parameter sweep and generate regime map
python3 scripts/sweep.py

# Phase 2 — classify all 1,849 circuits (parallel, with checkpointing)
python3 scripts/circuit_classifier_runner.py

# Phase 2 — batch pattern metrics over .npz outputs
python3 scripts/pattern_metrics_runner.py <input_dir>

# Phase 2 — 3-axis grid sweep (bi × hill_coeff × diffusion)
python3 scripts/grid_sweep.py

# Animations across bi values
python3 scripts/make_animation.py
```

---

## Classifier Logic (`core/classify_regime.py`)

Each parameter set runs two simulations:

| Simulation | Init mode | Purpose |
|---|---|---|
| Sim 1 | `all_off` + nucleation | Can patterns emerge spontaneously? |
| Sim 2 | `spike_steady_state` | Can the ON state sustain itself? |

Classification order:
1. **ON** — nucleation sim converges to uniform high-A (CV < 0.05)
2. **ON** — spike sim shows >80% of cells above 0.5 × a_ss
3. **TURING** — autocorrelation oscillates + FFT ratio > 35
4. **IRREGULAR** — autocorrelation oscillates + FFT ratio ≤ 35
5. **OFF** — spike sim mean < 0.07
6. **IRREGULAR** — fallback

---

## Parameter Space (Phase 2 Sweep)

Fixed parameters: `na=10, ni=4, Di=20, db=0.5, da=1, juxtacrine`

| Region | Condition |
|---|---|
| OFF | Low ba (activator too weak) |
| IRREGULAR | Balanced ba/bi — activator nucleates but inhibitor limits spread |
| ON | High ba, low bi (activator dominates) |

Turing patterns are rare at na=10, ni=4; the sharp switch forces cells fully ON or OFF, breaking the gradual gradients Turing requires. This matches the paper's finding that ~99% of non-homogeneous patterns are Irregular at experimentally measured Hill coefficients.

---

## Current Status

### Complete ✓
- Circuit enumeration (21,609 → 1,849 valid)
- JSON circuit library and NetworkX graphs
- LSA Turing checker (validated on JAPI at index 1285 and PAPI)
- Fig S15-style circuit diagrams with legend
- Fig 1C replication — PI confirmed
- Dual-sim regime classifier (ON / IRREGULAR / OFF / TURING)
- 5×8 parameter sweep with color-coded regime map
- Repo reorganized into `core/`, `scripts/`, `analysis/`, `data/`, `outputs/`

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
