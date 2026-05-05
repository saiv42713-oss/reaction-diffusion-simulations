"""
One-dimensional activator–inhibitor simulation with explicit Euler time stepping.

The model evolves two fields on a 1D grid:
- activator: self-promoting species. Either juxtacrine (activates neighbour points, not self; diffusion coeff = 0),
    or paracrine (activates self, has a diffusion coefficient > 0)
- inhibitor: downstream repressor that always diffuses (paracrine)

Production is regulated by a Hill-type function of the local activator/inhibitor
signals. Spatial coupling is handled with finite differences and zero-flux
(Neumann) boundary conditions.

The simulation can start from several initialization modes, including random
noise, single spikes, uniform activation, or steady-state-based perturbations.
"""

import numpy as np
from finding_steady_states import fast_stable_steady_state

def hill_function_vec(act_signal, inh_signal,
                      act_half_sat, inh_half_sat,
                      act_hill_coeff, inh_hill_coeff, basal_prod):
    """
    Vectorized Hill-type competitive activation-inhibition function.
    Works with scalars or NumPy arrays.
    (unused here) the basal_prod term provides a nonzero baseline production level.
    """
    act_term = np.power(np.maximum(act_signal, 0.0) / act_half_sat, act_hill_coeff)
    inh_term = np.power(np.maximum(inh_signal, 0.0) / inh_half_sat, inh_hill_coeff)
    return (act_term + basal_prod) / (act_term + inh_term + 1.0 + basal_prod)


def initialize_fields(N, init_mode, spike_value, spike_value_a=0, spike_value_i=0, seed=2):
    """
    Vectorized initialization.

    Different init_mode choices are used to probe pattern formation from
    uniform states, local perturbations, random noise, or steady-state spikes.
    """
    # Seed NumPy's random generator for reproducible initial conditions.
    rng = np.random.default_rng(seed)
    activator = np.zeros(N, dtype=float)
    inhibitor = np.zeros(N, dtype=float)

    if init_mode == "random_tight":
        # Small random perturbation around the steady-state value.
        activator = rng.uniform(0.95 * spike_value_a, 1.05 * spike_value_a, size=N)
        inhibitor = rng.uniform(0.95 * spike_value_i, 1.05 * spike_value_i, size=N)

    elif init_mode == "random_uniform_over0":
         # Random initial activator values in [0, spike_value].
        activator = rng.uniform(0.0, spike_value, size=N)
    elif init_mode == "random_both":
        # Random initial activator values in [0, spike_value] for a and i.
        activator = rng.uniform(0.0, spike_value, size=N)
        inhibitor = rng.uniform(0.0, spike_value, size=N)

    elif init_mode == "spike_steady_state":
        # Start from the steady-state value at a single grid point only.
        activator[N // 2] = spike_value_a
        inhibitor[N // 2] = spike_value_i

    elif init_mode == "activator_spike_steady_state":
        # Single activator spike placed at the center.
        activator[N // 2] = spike_value_a
    elif init_mode == "activator_spike":
        # Single activator spike placed at the center, user-defined height
        activator[N // 2] = spike_value
    elif init_mode == "two_activator_spikes":
        # Two activator spikes.
        activator[0] = spike_value
        if N > 100:
            activator[100] = spike_value
        else:
            activator[N // 2] = spike_value

    elif init_mode == "inhibitor_spike":
        # Single inhibitor spike placed at the center, user-defined height
        inhibitor[N // 2] = spike_value

    elif init_mode == "activator_on":
        #homogeneous user-defined activation value
        activator[:] = spike_value
    elif init_mode == "inhibitor_on":
        #homogeneous user-defined inhibition value
        inhibitor[:] = spike_value
    elif init_mode == "both_on":
        #homogeneous user-defined activation and inhibition value
        activator[:] = spike_value
        inhibitor[:] = spike_value
    elif init_mode == "all_off":
        #all set to zero
        pass

    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    return activator, inhibitor


def neumann_laplacian(u):
    """
    Discrete Laplacian with the same boundary treatment as your code:
    interior: u[i+1] - 2u[i] + u[i-1]
    left:     u[1] - u[0]
    right:    u[-2] - u[-1]
    """
    lap = np.empty_like(u)
    lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
    lap[0] = u[1] - u[0]
    lap[-1] = u[-2] - u[-1]
    return lap


def step_fields(activator, inhibitor, dt, dx, p, activator_type):
    """
    One fully vectorized time step.
    """
    # Activator signal used in transcriptional regulation
    if activator_type == "paracrine":
        act_signal = activator
        act_diffusion = p["act_diffusion"] * dt / dx**2 * neumann_laplacian(activator)
    else:
        act_signal = np.empty_like(activator)
        act_signal[1:-1] = 0.5 * (activator[:-2] + activator[2:])
        act_signal[0] = activator[1]
        act_signal[-1] = activator[-2]
        act_diffusion = 0.0

    # Inhibitor signal is always local
    inh_signal = inhibitor
    inh_diffusion = p["inh_diffusion"] * dt / dx**2 * neumann_laplacian(inhibitor)

    hill_value = hill_function_vec(
        act_signal, inh_signal,
        p["act_half_sat"], p["inh_half_sat"],
        p["act_hill_coeff"], p["inh_hill_coeff"],
        p["basal_prod"]
    )

    activator_new = (
        activator
        + dt * (p["act_prod_rate"] * hill_value - p["act_decay_rate"] * activator)
        + act_diffusion
    )

    inhibitor_new = (
        inhibitor
        + dt * (p["inh_prod_rate"] * hill_value - p["inh_decay_rate"] * inhibitor)
        + inh_diffusion
    )

    return activator_new, inhibitor_new


def run_coupled_neumann(
    N, steps, dt, dx, p, stopping_threshold, min_steps,
    init_mode="spikes",
    activator_type="juxtacrine",
    spike_value=5.0,
    save_every=10
):
    """
    Vectorized simulation loop.
    """
    try:
        a_ss, i_ss, H_ss = fast_stable_steady_state(p, activator_type, tol=5e-4, max_newton=12)
    except Exception:
        a_ss = i_ss = 0.0

    if not (a_ss > 0.0 and i_ss > 0.0 and np.isfinite(a_ss) and np.isfinite(i_ss)):
        a_ss = float(spike_value)
        i_ss = float(spike_value)

    activator, inhibitor = initialize_fields(
        N, init_mode, spike_value,
        spike_value_a=float(a_ss),
        spike_value_i=float(i_ss),
        seed=2
    )

    activator_history = [activator.copy()]
    inhibitor_history = [inhibitor.copy()]

    last_saved_activator = activator.copy()
    last_saved_inhibitor = inhibitor.copy()
    diff = np.inf
    step = 0

    for step in range(steps):
        activator, inhibitor = step_fields(activator, inhibitor, dt, dx, p, activator_type)

        if step % save_every == 0:
            diff = (
                np.sum(np.abs(activator - last_saved_activator)) +
                np.sum(np.abs(inhibitor - last_saved_inhibitor))
            )

            activator_history.append(activator.copy())
            inhibitor_history.append(inhibitor.copy())

            last_saved_activator = activator.copy()
            last_saved_inhibitor = inhibitor.copy()

            if step > min_steps and diff / (2 * N) < stopping_threshold:
                break

    print(f"Stopped at step {step}, total average difference per tile over {save_every} steps = {diff/(2*N)}")

    return activator_history, inhibitor_history, step, a_ss, i_ss

def run_simulation(params):
    """
    Thin wrapper to call run_coupled_neumann with a parameter dict.
    """
    result = run_coupled_neumann(
        params["N"],
        params["steps"],
        params["dt"],
        params["dx"],
        params,
        params.get("stopping_threshold", 1e-4),
        params.get("min_steps", 10000),
        init_mode=params.get("init_mode", "activator_spike"),
        activator_type=params.get("activator_type", "juxtacrine"),
        spike_value=params.get("spike_value", 5.0),
        save_every=params.get("save_every", 100),

    )

    activator_hist, inhibitor_hist, steps_used, a_ss, i_ss = result

    return {
        "status": "done",  # your loop prints convergence info already
        "steps_used": steps_used,
        "parameters": params,
        "activator_initial": activator_hist[0],
        "activator_final": activator_hist[-1],
        "inhibitor_initial": inhibitor_hist[0],
        "inhibitor_final": inhibitor_hist[-1],
        "activator_steady-state": a_ss,
        "inhibitor_steady-state": i_ss
    }
