import sys, time
from pathlib import Path
import numpy as np

REPO = Path("/home/claude/reaction-diffusion-simulations/2D_simulations")
sys.path.insert(0, str(REPO))

from simulation_2D import _vectorized_step, _build_hex_neighbor_arrays, initialize_fields_2d
from finding_steady_states import fast_stable_steady_state

OUT = Path("/home/claude/analysis/runs")
Ny = Nx = 100
dt = 0.01
dx = 1.0

BASE_PARAMS = dict(
    act_half_sat=1.0, inh_half_sat=1.0, act_decay_rate=1.0, basal_prod=0.0,
    act_diffusion=0.0, inh_diffusion=10.0, act_prod_rate=5.0, inh_decay_rate=0.5,
    act_hill_coeff=3, inh_hill_coeff=3, inh_prod_rate=5.0,  # bi=5, TURING
)

nbr_r, nbr_c, nbr_mask, nbr_count = _build_hex_neighbor_arrays(Ny, Nx)
p = BASE_PARAMS

a_ss, i_ss, _ = fast_stable_steady_state(p, "juxtacrine", tol=5e-4, max_newton=12)
np.random.seed(42)
activator, inhibitor = initialize_fields_2d(
    Ny, Nx, "random_uniform_over0", spike_value=1.0,
    spike_value_a=float(a_ss), spike_value_i=float(i_ss),
)

snapshot_steps = [0, 100, 400, 1000, 2500, 6000, 12000]
snapshots = {}
t0 = time.time()
for step in range(max(snapshot_steps) + 1):
    if step in snapshot_steps:
        snapshots[step] = activator.copy()
    activator, inhibitor = _vectorized_step(
        activator, inhibitor, dt, dx, p, "juxtacrine",
        nbr_r, nbr_c, nbr_mask, nbr_count,
    )
print(f"Captured {len(snapshots)} snapshots in {time.time()-t0:.1f}s")

np.savez_compressed(
    OUT / "early_window_snapshots.npz",
    steps=np.array(sorted(snapshots.keys())),
    **{f"A_{s}": snapshots[s] for s in snapshots},
)
print("Saved early_window_snapshots.npz")
