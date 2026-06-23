import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.finding_steady_states import fast_stable_steady_state, find_unstable_fixed_point
from core.simulation_2D import run_coupled_hex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_OUTPUTS = Path(__file__).parent.parent / "outputs"
_OUTPUTS.mkdir(exist_ok=True)

BASE_PARAMS = {
    "act_half_sat": 1.0,
    "inh_half_sat": 1.0,
    "act_decay_rate": 1.0,
    "inh_decay_rate": 0.5,
    "act_diffusion": 0.0,
    "inh_diffusion": 10.0,
    "basal_prod": 0.0,
    "act_hill_coeff": 3,
    "inh_hill_coeff": 3,
    "act_prod_rate": 5.0,
}

bi_values = [1.0, 3.0, 5.0, 12.0, 14.0]

for bi in bi_values:
    params = {**BASE_PARAMS, "act_prod_rate": 5.0, "inh_prod_rate": bi}
    a_ss, i_ss, _ = fast_stable_steady_state(params, activator_type="juxtacrine")
    threshold = find_unstable_fixed_point(params, a_ss, i_ss, activator_type="juxtacrine")

    if a_ss == 0.0:
        init_mode = "random_uniform_over0"
        spike_value = 2.0
        init_amplitude = 0.4
    else:
        init_mode = "noise_around_state"
        spike_value = a_ss
        init_amplitude = threshold * 0.1 if threshold is not None else a_ss / 5

    A_hist, R_hist, final_step, _, _ = run_coupled_hex(
        100, 100,
        50000, 0.01, 1.0,
        params,
        1e-4,
        5000,
        init_mode=init_mode,
        activator_type="juxtacrine",
        spike_value=spike_value,
        save_every=40,
        nucleation_rate=0,
        init_amplitude=init_amplitude
    )

    indices = np.linspace(0, len(A_hist) - 1, 100).astype(int)
    selected_frames = [A_hist[i] for i in indices]
    print(f"bi={bi} -> frames captured: {len(A_hist)}, finished at step {final_step}")

    vmin = np.min([frame.min() for frame in selected_frames])
    vmax = np.max([frame.max() for frame in selected_frames])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(selected_frames[0], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"bi={bi}")

    def update(frame_index):
        im.set_array(selected_frames[frame_index])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(selected_frames), interval=50)
    out = _OUTPUTS / f"bi_{bi}_animation.mp4"
    ani.save(str(out), writer="ffmpeg", fps=20)
    print(f"Saved {out}")
    plt.close()
