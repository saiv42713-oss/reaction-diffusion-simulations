import sys, time, json
from pathlib import Path
import numpy as np

REPO = Path("/home/claude/reaction-diffusion-simulations/2D_simulations")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "res_analysis"))

from simulation_2D import _vectorized_step, _build_hex_neighbor_arrays, initialize_fields_2d
from finding_steady_states import fast_stable_steady_state, find_unstable_fixed_point

OUT = Path("/home/claude/analysis/runs")
OUT.mkdir(parents=True, exist_ok=True)

Ny = Nx = 100
dt = 0.01
dx = 1.0
steps = 50000
save_every = 200
stopping_threshold = 1e-4
min_steps = 5000

BASE_PARAMS = dict(
    act_half_sat=1.0,
    inh_half_sat=1.0,
    act_decay_rate=1.0,
    basal_prod=0.0,
    act_diffusion=0.0,
    inh_diffusion=10.0,
    act_prod_rate=5.0,
    inh_decay_rate=0.5,
    act_hill_coeff=3,
    inh_hill_coeff=3,
)

# bi = inh_prod_rate sweep, Fig 1C juxtacrine row
BI_VALUES = [1, 3, 5, 12, 14]
LABELS = {1: "ON", 3: "STRIPES", 5: "TURING", 12: "IRREGULAR", 14: "OFF"}

nbr_r, nbr_c, nbr_mask, nbr_count = _build_hex_neighbor_arrays(Ny, Nx)


def run_one(bi, seed=42, record_every=save_every, run_steps=steps):
    p = dict(BASE_PARAMS)
    p["inh_prod_rate"] = float(bi)

    try:
        a_ss, i_ss, _ = fast_stable_steady_state(p, "juxtacrine", tol=5e-4, max_newton=12)
    except Exception:
        a_ss = i_ss = 0.0
    if not (a_ss > 0.0 and i_ss > 0.0 and np.isfinite(a_ss) and np.isfinite(i_ss)):
        a_ss = i_ss = 1.0

    np.random.seed(seed)
    activator, inhibitor = initialize_fields_2d(
        Ny, Nx, "random_uniform_over0", spike_value=1.0,
        spike_value_a=float(a_ss), spike_value_i=float(i_ss),
    )

    mean_hist, max_hist, t_hist = [], [], []
    prev_a, prev_i = activator.copy(), inhibitor.copy()
    mean_stop_step = None

    t0 = time.time()
    for step in range(run_steps):
        activator, inhibitor = _vectorized_step(
            activator, inhibitor, dt, dx, p, "juxtacrine",
            nbr_r, nbr_c, nbr_mask, nbr_count,
        )
        if step % record_every == 0:
            d_a = np.abs(activator - prev_a)
            d_i = np.abs(inhibitor - prev_i)
            mean_change = (d_a.sum() + d_i.sum()) / (2 * Ny * Nx)
            max_change = max(d_a.max(), d_i.max())
            mean_hist.append(mean_change)
            max_hist.append(max_change)
            t_hist.append(step * dt)
            if mean_stop_step is None and step > min_steps and mean_change < stopping_threshold:
                mean_stop_step = step
            prev_a, prev_i = activator.copy(), inhibitor.copy()

    elapsed = time.time() - t0
    print(f"bi={bi} ({LABELS.get(bi,'?')}): {run_steps} steps in {elapsed:.1f}s, "
          f"mean-stop at step={mean_stop_step}, final mean={mean_hist[-1]:.2e}, final max={max_hist[-1]:.2e}")

    np.savez_compressed(
        OUT / f"bi{bi}_final.npz",
        A_final=activator, R_final=inhibitor,
        mean_hist=np.array(mean_hist), max_hist=np.array(max_hist), t_hist=np.array(t_hist),
        mean_stop_step=np.array(mean_stop_step if mean_stop_step is not None else -1),
    )
    return dict(bi=bi, label=LABELS.get(bi, "?"), mean_stop_step=mean_stop_step,
                final_mean=float(mean_hist[-1]), final_max=float(max_hist[-1]))


if __name__ == "__main__":
    summary = [run_one(bi) for bi in BI_VALUES]
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
