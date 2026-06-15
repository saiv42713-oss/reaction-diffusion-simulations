"""
Core simulation routines for 2D hex-grid JAPI/PAPI models.

This module provides:
- initialization utilities for 2D fields,
- a vectorized single-pair hex-grid simulator,
- a vectorized two-pair coupled hex-grid simulator with cross-inhibition.

"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from finding_steady_states import fast_stable_steady_state


def initialize_fields_2d(
    Ny: int,
    Nx: int,
    init_mode: str,
    spike_value: float,
    spike_value_a: float = 0.0,
    spike_value_i: float = 0.0,
    n_points: int = 10,
    spike_indices: np.ndarray | None = None,
    set_peak_height: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize activator and inhibitor fields on a 2D hex grid.

    Parameters
    ----------
    Ny, Nx
        Grid dimensions.
    init_mode
        Initial condition mode.
    spike_value
        Default spike amplitude used by some modes.
    spike_value_a, spike_value_i
        Reference activator and inhibitor values, usually steady-state estimates.
    n_points
        Number of random spikes for ``activator_random_spikes``.
    spike_indices
        Optional preselected linear indices for spikes.
    set_peak_height
        Spike height used by ``activator_random_spikes``.

    Returns
    -------
    activator, inhibitor
        Arrays of shape ``(Ny, Nx)``.
    """
    activator = np.zeros((Ny, Nx), dtype=float)
    inhibitor = np.zeros((Ny, Nx), dtype=float)

    if init_mode == "random_tight":
        activator = np.random.uniform(
            spike_value_a - 0.05 * spike_value_a,
            spike_value_a + 0.05 * spike_value_a,
            size=(Ny, Nx),
        )
        inhibitor = np.random.uniform(
            spike_value_i - 0.05 * spike_value_i,
            spike_value_i + 0.05 * spike_value_i,
            size=(Ny, Nx),
        )

    elif init_mode == "random_uniform_over0":
        activator = np.random.uniform(0, spike_value, size=(Ny, Nx))

    elif init_mode == "spike_steady_state":
        activator[Ny // 2, Nx // 2] = spike_value_a
        inhibitor[Ny // 2, Nx // 2] = spike_value_i

    elif init_mode == "activator_random_spikes":
        n_points = int(n_points)
        if n_points > 0:
            if spike_indices is not None:
                indices = spike_indices
            else:
                indices = np.random.choice(Ny * Nx, size=n_points, replace=False)
            for idx in indices:
                activator[idx // Nx, idx % Nx] = set_peak_height

    elif init_mode == "inhibitor_spike":
        inhibitor[Ny // 2, Nx // 2] = spike_value

    elif init_mode == "both_on":
        activator[:] = spike_value
        inhibitor[:] = spike_value

    elif init_mode == "all_off":
        pass

    elif init_mode == "near_zero":
        activator = np.abs(np.random.randn(Ny, Nx)) * 0.01
        inhibitor = np.zeros((Ny, Nx), dtype=float)
    elif init_mode == "noise_around_state":
        activator = spike_value_a + 0.01 * np.random.randn(Ny, Nx)
        inhibitor = spike_value_i + 0.01 * np.random.randn(Ny, Nx)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode!r}")

    return activator, inhibitor


def _build_hex_neighbor_arrays(Ny: int, Nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute neighbor indices for a hex grid in even-r offset layout.

    Returns
    -------
    nbr_r, nbr_c
        Integer arrays of shape ``(Ny, Nx, 6)`` containing neighbor row/col indices.
    nbr_mask
        Boolean array of shape ``(Ny, Nx, 6)`` marking valid in-bounds neighbors.
    nbr_count
        Number of valid neighbors per cell, shape ``(Ny, Nx)``.
    """
    even_offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
    odd_offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]

    nbr_r = np.zeros((Ny, Nx, 6), dtype=np.int32)
    nbr_c = np.zeros((Ny, Nx, 6), dtype=np.int32)
    nbr_mask = np.zeros((Ny, Nx, 6), dtype=bool)

    for r in range(Ny):
        offsets = even_offsets if r % 2 == 0 else odd_offsets
        for c in range(Nx):
            for k, (dr, dc) in enumerate(offsets):
                rr, cc = r + dr, c + dc
                if 0 <= rr < Ny and 0 <= cc < Nx:
                    nbr_r[r, c, k] = rr
                    nbr_c[r, c, k] = cc
                    nbr_mask[r, c, k] = True
                else:
                    nbr_r[r, c, k] = r
                    nbr_c[r, c, k] = c

    nbr_count = nbr_mask.sum(axis=2).astype(np.float64)
    return nbr_r, nbr_c, nbr_mask, nbr_count


def _vectorized_step(
    activator: np.ndarray,
    inhibitor: np.ndarray,
    dt: float,
    dx: float,
    p: Mapping[str, float],
    activator_type: str,
    nbr_r: np.ndarray,
    nbr_c: np.ndarray,
    nbr_mask: np.ndarray,
    nbr_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one forward-Euler step for a single activator/inhibitor pair.
    """
    ka = p["act_half_sat"]
    ki = p["inh_half_sat"]
    n = p["act_hill_coeff"]
    m = p["inh_hill_coeff"]
    basal = p["basal_prod"]

    nbr_act = activator[nbr_r, nbr_c]
    nbr_inh = inhibitor[nbr_r, nbr_c]

    nbr_act_valid = nbr_act * nbr_mask
    nbr_inh_valid = nbr_inh * nbr_mask

    if activator_type == "juxtacrine":
        act_signal = nbr_act_valid.sum(axis=2) / nbr_count
    else:
        act_signal = activator

    inh_signal = inhibitor

    act_term = np.where(act_signal > 0, (act_signal / ka) ** n, 0.0)
    inh_term = np.where(inh_signal > 0, (inh_signal / ki) ** m, 0.0)
    hill_val = (act_term + basal) / (act_term + inh_term + 1.0 + basal)

    reaction_a = dt * (p["act_prod_rate"] * hill_val - p["act_decay_rate"] * activator)
    reaction_i = dt * (p["inh_prod_rate"] * hill_val - p["inh_decay_rate"] * inhibitor)

    lap_i = nbr_inh_valid.sum(axis=2) - nbr_count * inhibitor
    diffusion_i = p["inh_diffusion"] * dt / dx**2 * lap_i

    if activator_type == "paracrine":
        lap_a = nbr_act_valid.sum(axis=2) - nbr_count * activator
        diffusion_a = p["act_diffusion"] * dt / dx**2 * lap_a
    else:
        diffusion_a = 0.0

    return activator + reaction_a + diffusion_a, inhibitor + reaction_i + diffusion_i


def run_coupled_hex(
    Ny: int,
    Nx: int,
    steps: int,
    dt: float,
    dx: float,
    p: Mapping[str, float],
    stopping_threshold: float,
    min_steps: int,
    init_mode: str = "spike_steady_state",
    activator_type: str = "juxtacrine",
    n_points: int = 10,
    spike_value: float = 5.0,
    save_every: int = 10,
    nucleation_rate: float = 0.0,
    noise_amplitude: float = 0.0,
    set_peak_height: float = 25.0,
) -> tuple[list[np.ndarray], list[np.ndarray], int, float, float]:
    """
    Run the single-pair 2D hex-grid simulation.

    Returns
    -------
    activator_history, inhibitor_history
        Saved snapshots at the chosen interval.
    step
        Final step reached.
    a_ss, i_ss
        Stable ON-state reference values used for initialization and nucleation.
    """
    try:
        a_ss, i_ss, _ = fast_stable_steady_state(p, activator_type, tol=5e-4, max_newton=12)
    except Exception:
        a_ss = i_ss = 0.0

    if not (a_ss > 0.0 and i_ss > 0.0 and np.isfinite(a_ss) and np.isfinite(i_ss)):
        a_ss = i_ss = float(spike_value)

    init_seed = 2
    init_rng = np.random.default_rng(init_seed)
    spike_indices = None

    if init_mode == "activator_random_spikes":
        max_points = Ny * Nx
        if n_points > max_points:
            raise ValueError(f"n_points={n_points} is larger than the grid size Ny*Nx={max_points}.")
        spike_indices = init_rng.choice(max_points, size=n_points, replace=False)

    activator, inhibitor = initialize_fields_2d(
        Ny,
        Nx,
        init_mode,
        spike_value,
        spike_value_a=float(a_ss),
        spike_value_i=float(i_ss),
        n_points=n_points,
        spike_indices=spike_indices,
        set_peak_height=set_peak_height,
    )

    nbr_r, nbr_c, nbr_mask, nbr_count = _build_hex_neighbor_arrays(Ny, Nx)

    activator_history = [activator.copy()]
    inhibitor_history = [inhibitor.copy()]

    p_fire = nucleation_rate * dt
    diff = 0.0

    for step in range(steps):
        activator, inhibitor = _vectorized_step(
            activator,
            inhibitor,
            dt,
            dx,
            p,
            activator_type,
            nbr_r,
            nbr_c,
            nbr_mask,
            nbr_count,
        )

        if p_fire > 0.0:
            fired = np.random.random((Ny, Nx)) < p_fire
            mask = activator < a_ss
            activator[mask] += fired[mask] * 2 * a_ss

        if noise_amplitude > 0.0:
            activator = activator + noise_amplitude * np.sqrt(dt) * np.random.randn(Ny, Nx)
            np.clip(activator, 0.0, None, out=activator)

        if step % save_every == 0:
            diff = (
                np.sum(np.abs(activator - activator_history[-1]))
                + np.sum(np.abs(inhibitor - inhibitor_history[-1]))
            )
            activator_history.append(activator.copy())
            inhibitor_history.append(inhibitor.copy())

            if step > min_steps and diff / (2 * Ny * Nx) < stopping_threshold:
                break

    print(
        f"Stopped at step {step}, "
        f"total average difference per tile over {save_every} steps = "
        f"{diff / (2 * Ny * Nx):.2e}"
    )

    return activator_history, inhibitor_history, step, a_ss, i_ss


def _pair_hill(
    activator: np.ndarray,
    inhibitor: np.ndarray,
    p: Mapping[str, float],
    activator_type: str,
    nbr_r: np.ndarray,
    nbr_c: np.ndarray,
    nbr_mask: np.ndarray,
    nbr_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Hill response for one pair, vectorized over the whole grid.
    """
    ka = p["act_half_sat"]
    ki = p["inh_half_sat"]
    n = p["act_hill_coeff"]
    m = p["inh_hill_coeff"]
    basal = p["basal_prod"]

    nbr_act = activator[nbr_r, nbr_c]
    nbr_inh = inhibitor[nbr_r, nbr_c]

    nbr_act_valid = nbr_act * nbr_mask
    nbr_inh_valid = nbr_inh * nbr_mask

    if activator_type == "juxtacrine":
        act_signal = nbr_act_valid.sum(axis=2) / nbr_count
    else:
        act_signal = activator

    inh_signal = inhibitor

    act_term = np.where(act_signal > 0, (act_signal / ka) ** n, 0.0)
    inh_term = np.where(inh_signal > 0, (inh_signal / ki) ** m, 0.0)

    hill_val = (act_term + basal) / (act_term + inh_term + 1.0 + basal)
    return hill_val, nbr_act_valid, nbr_inh_valid


def _vectorized_step_two_pairs_cross(
    a1: np.ndarray,
    i1: np.ndarray,
    a2: np.ndarray,
    i2: np.ndarray,
    dt: float,
    dx: float,
    p1: Mapping[str, float],
    p2: Mapping[str, float],
    activator_type1: str,
    activator_type2: str,
    nbr_r: np.ndarray,
    nbr_c: np.ndarray,
    nbr_mask: np.ndarray,
    nbr_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one forward-Euler step for two coupled pairs with cross-inhibition.
    """
    hill1, nbr_act1_valid, nbr_inh1_valid = _pair_hill(
        a1, i1, p1, activator_type1, nbr_r, nbr_c, nbr_mask, nbr_count
    )
    hill2, nbr_act2_valid, nbr_inh2_valid = _pair_hill(
        a2, i2, p2, activator_type2, nbr_r, nbr_c, nbr_mask, nbr_count
    )

    reaction_a1 = dt * (p1["act_prod_rate"] * hill1 - p1["act_decay_rate"] * a1)
    reaction_i1 = dt * (
        p1["inh_prod_rate"] * hill1
        + p1["cross_inhibition_rate"] * hill2
        - p1["inh_decay_rate"] * i1
    )

    reaction_a2 = dt * (p2["act_prod_rate"] * hill2 - p2["act_decay_rate"] * a2)
    reaction_i2 = dt * (
        p2["inh_prod_rate"] * hill2
        + p2["cross_inhibition_rate"] * hill1
        - p2["inh_decay_rate"] * i2
    )

    lap_i1 = nbr_inh1_valid.sum(axis=2) - nbr_count * i1
    lap_i2 = nbr_inh2_valid.sum(axis=2) - nbr_count * i2

    diffusion_i1 = p1["inh_diffusion"] * dt / dx**2 * lap_i1
    diffusion_i2 = p2["inh_diffusion"] * dt / dx**2 * lap_i2

    if activator_type1 == "paracrine":
        lap_a1 = nbr_act1_valid.sum(axis=2) - nbr_count * a1
        diffusion_a1 = p1["act_diffusion"] * dt / dx**2 * lap_a1
    else:
        diffusion_a1 = 0.0

    if activator_type2 == "paracrine":
        lap_a2 = nbr_act2_valid.sum(axis=2) - nbr_count * a2
        diffusion_a2 = p2["act_diffusion"] * dt / dx**2 * lap_a2
    else:
        diffusion_a2 = 0.0

    a1_new = a1 + reaction_a1 + diffusion_a1
    i1_new = i1 + reaction_i1 + diffusion_i1
    a2_new = a2 + reaction_a2 + diffusion_a2
    i2_new = i2 + reaction_i2 + diffusion_i2

    return a1_new, i1_new, a2_new, i2_new


def run_two_pair_hex(
    Ny: int,
    Nx: int,
    steps: int,
    dt: float,
    dx: float,
    p1: Mapping[str, float],
    p2: Mapping[str, float],
    stopping_threshold: float,
    min_steps: int,
    init_mode: str = "spike_steady_state",
    activator_type1: str = "juxtacrine",
    activator_type2: str = "juxtacrine",
    n_points: int = 10,
    spike_value: float = 5.0,
    save_every: int = 10,
    nucleation_rate: float = 0.01,
    noise_amplitude: float = 0.0,
    set_peak_height: float = 25.0,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    int,
    dict[str, dict[str, float]],
]:
    """
    Simulate two coupled activator/inhibitor pairs on the same hex grid.

    Pair 1 uses parameters ``p1`` and fields ``a1, i1``.
    Pair 2 uses parameters ``p2`` and fields ``a2, i2``.

    Returns
    -------
    a1_history, i1_history, a2_history, i2_history
        Saved snapshots for each field.
    step
        Final step reached.
    steady_states
        Dictionary containing ON-state references for each pair.
    """
    try:
        a1_ss, i1_ss, _ = fast_stable_steady_state(p1, activator_type1, tol=5e-4, max_newton=12)
    except Exception:
        a1_ss = i1_ss = 0.0

    if not (a1_ss > 0.0 and i1_ss > 0.0 and np.isfinite(a1_ss) and np.isfinite(i1_ss)):
        a1_ss = i1_ss = float(spike_value)

    try:
        a2_ss, i2_ss, _ = fast_stable_steady_state(p2, activator_type2, tol=5e-4, max_newton=12)
    except Exception:
        a2_ss = i2_ss = 0.0

    if not (a2_ss > 0.0 and i2_ss > 0.0 and np.isfinite(a2_ss) and np.isfinite(i2_ss)):
        a2_ss = i2_ss = float(spike_value)

    init_seed = 2
    init_rng = np.random.default_rng(init_seed)

    spike_indices = None
    if init_mode == "activator_random_spikes":
        max_points = Ny * Nx
        if n_points > max_points:
            raise ValueError(f"n_points={n_points} is larger than the grid size Ny*Nx={max_points}.")
        spike_indices = init_rng.choice(max_points, size=n_points, replace=False)

    a1, i1 = initialize_fields_2d(
        Ny,
        Nx,
        init_mode,
        spike_value,
        spike_value_a=float(a1_ss),
        spike_value_i=float(i1_ss),
        n_points=n_points,
        spike_indices=spike_indices,
        set_peak_height=set_peak_height,
    )

    a2, i2 = initialize_fields_2d(
        Ny,
        Nx,
        init_mode,
        spike_value,
        spike_value_a=float(a2_ss),
        spike_value_i=float(i2_ss),
        n_points=n_points,
        spike_indices=spike_indices,
        set_peak_height=set_peak_height,
    )

    nbr_r, nbr_c, nbr_mask, nbr_count = _build_hex_neighbor_arrays(Ny, Nx)

    a1_history = [a1.copy()]
    i1_history = [i1.copy()]
    a2_history = [a2.copy()]
    i2_history = [i2.copy()]

    p_fire = nucleation_rate * dt
    diff = 0.0

    for step in range(steps):
        a1, i1, a2, i2 = _vectorized_step_two_pairs_cross(
            a1,
            i1,
            a2,
            i2,
            dt,
            dx,
            p1,
            p2,
            activator_type1,
            activator_type2,
            nbr_r,
            nbr_c,
            nbr_mask,
            nbr_count,
        )

        if p_fire > 0.0:
            fired1 = np.random.random((Ny, Nx)) < p_fire
            mask1 = a1 < a1_ss
            a1[mask1] += fired1[mask1] * 2.0 * a1_ss

        if p_fire > 0.0:
            fired2 = np.random.random((Ny, Nx)) < p_fire
            mask2 = a2 < a2_ss
            a2[mask2] += fired2[mask2] * 2.0 * a2_ss

        if noise_amplitude > 0.0:
            a1 += noise_amplitude * np.sqrt(dt) * np.random.randn(Ny, Nx)
            np.clip(a1, 0.0, None, out=a1)

        if noise_amplitude > 0.0:
            a2 += noise_amplitude * np.sqrt(dt) * np.random.randn(Ny, Nx)
            np.clip(a2, 0.0, None, out=a2)

        if step % save_every == 0:
            diff = (
                np.sum(np.abs(a1 - a1_history[-1]))
                + np.sum(np.abs(i1 - i1_history[-1]))
                + np.sum(np.abs(a2 - a2_history[-1]))
                + np.sum(np.abs(i2 - i2_history[-1]))
            )

            a1_history.append(a1.copy())
            i1_history.append(i1.copy())
            a2_history.append(a2.copy())
            i2_history.append(i2.copy())

            total_cells = 4 * Ny * Nx
            if step > min_steps and diff / total_cells < stopping_threshold:
                break

    print(
        f"Stopped at step {step}, "
        f"total average difference per tile over {save_every} steps = "
        f"{diff / (4 * Ny * Nx):.2e}"
    )

    steady_states = {
        "pair1": {"a_ss": a1_ss, "i_ss": i1_ss},
        "pair2": {"a_ss": a2_ss, "i_ss": i2_ss},
    }

    return a1_history, i1_history, a2_history, i2_history, step, steady_states
