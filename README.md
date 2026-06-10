# JAPI Circuit Enumerator
Acomputational framework for enumerating, filtering, and evaluating 
genetic circuit designs for biological patterning, based on Fig S15 
from Kunnan et al. 2026 (Morsut Lab, USC).

# What This Does
1. Generates all 21,609 possible 2-node circuit designs
2. Filters to 1,849 biologically valid circuits
3. Stores each circuit as a NetworkX graph and JSON object
4. Runs a Turing/LSA math check on each circuit
5. Draws the top candidates as diagrams matching Fig S15 style

# Output Files
- valid_circuits.json — all 1,849 valid circuits
- turing_results.json — Turing pass/fail for each circuit
- circuit_0.png through circuit_4.png — top 5 candidate diagrams

# Setup

1. Clone this repo:
   git clone https://github.com/saiv42713-oss/JAPI_Genecircuit_USC

2. Clone the lab simulation code:
   git clone https://github.com/BenSwedlund/reaction_diffusion_simulations

3. Copy required files into this project folder:
   cp reaction_diffusion_simulations/1D_simuations/finding_steady_states.py .
   cp reaction_diffusion_simulations/1D_simuations/parameters.py .

# Requirements
   pip install networkx numpy matplotlib

# How to Run
   python JAPI_Circuit.py

# Notes
- Turing check currently uses JAPI parameters for all circuits
- PI confirmation needed for circuit-specific parameter sets
- LLM scorer integration pending PI instructions

# Reference
Kunnan, Swedlund, Ben Tahar, Morsut. 
"A novel reaction-diffusion architecture for engineering 
self-organized patterns in mammalian cells." bioRxiv 2026.
https://doi.org/10.64898/2026.05.24.727552