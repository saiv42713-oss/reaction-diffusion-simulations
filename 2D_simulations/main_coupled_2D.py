"""
Run a coupled 2D two-pair hex-grid simulation from the command line.

This script:
- runs a coupled 2D simulation with two activator–inhibitor pairs,
- saves a parameter snapshot (.txt),
- saves the final state as a compressed NumPy file (.npz),
- saves static images of the initial and final states.

Outputs are written to:
    results/<prefix>_*.{png,txt,npz}
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from parameters_coupled_2D import (
    p1,
    p2,
    Ny,
    Nx,
    steps,
    dt,
    dx,
    save_every,
    spike_value,
    n_points,
    stopping_threshold,
    min_steps,
    init_mode,
    activator_type,
    nucleation_rate,
    set_peak_height,
)

from simulation_2D import run_two_pair_hex
from visualize_2D import plot_four_frames


def save_parameters_txt(
    filepath: Path,
    p1: dict,
    p2: dict,
    sim_params: dict,
) -> None:
    """
    Save simulation and model parameters to a human-readable text file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== Coupled 2D Simulation Parameters ===\n\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n\n")

        f.write("---- Simulation parameters ----\n")
        for k, v in sim_params.items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- Parameters p1 ----\n")
        for k, v in p1.items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- Parameters p2 ----\n")
        for k, v in p2.items():
            f.write(f"{k}: {v}\n")


def save_final_state_npz(
    filepath: Path,
    A1_hist,
    I1_hist,
    A2_hist,
    I2_hist,
    final_step: int,
) -> None:
    """
    Save the final state of the coupled simulation in compressed NumPy format.
    """
    np.savez_compressed(
        filepath,
        a1=np.asarray(A1_hist[-1]),
        i1=np.asarray(I1_hist[-1]),
        a2=np.asarray(A2_hist[-1]),
        i2=np.asarray(I2_hist[-1]),
        final_step=np.array(final_step),
    )


def main() -> None:
    """
    Run the coupled 2D simulation and export outputs.
    """
    parser = argparse.ArgumentParser(
        description="Run coupled 2-pair 2D hex simulation."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prefix stored in results/. Saves .png, .txt, and .npz.",
    )
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output if args.output else f"run_{timestamp}"

    png_final_path = results_dir / f"{prefix}_final.png"
    png_init_path = results_dir / f"{prefix}_init.png"
    param_path = results_dir / f"{prefix}_params.txt"
    npz_path = results_dir / f"{prefix}_final_state.npz"

    # Run the coupled simulation.
    A1_hist, I1_hist, A2_hist, I2_hist, final_step, steady_states = run_two_pair_hex(
        Ny=Ny,
        Nx=Nx,
        steps=steps,
        dt=dt,
        dx=dx,
        p1=p1,
        p2=p2,
        stopping_threshold=stopping_threshold,
        min_steps=min_steps,
        init_mode=init_mode,
        activator_type1=activator_type,
        activator_type2=activator_type,
        n_points=n_points,
        spike_value=spike_value,
        save_every=save_every,
        nucleation_rate=nucleation_rate,
        set_peak_height=set_peak_height,
    )

    print(f"Simulation finished at step {final_step}")
    print(f"Steady states: {steady_states}")

    sim_params = {
        "Ny": Ny,
        "Nx": Nx,
        "steps": steps,
        "dt": dt,
        "dx": dx,
        "save_every": save_every,
        "stopping_threshold": stopping_threshold,
        "min_steps": min_steps,
        "init_mode": init_mode,
        "activator_type": activator_type,
        "spike_value": spike_value,
        "n_points": n_points,
        "set_peak_height": set_peak_height,
        "nucleation_rate": nucleation_rate,
    }

    # Save metadata and final state.
    save_parameters_txt(param_path, p1, p2, sim_params)
    save_final_state_npz(npz_path, A1_hist, I1_hist, A2_hist, I2_hist, final_step)

    print(f"Saved parameters to {param_path}")
    print(f"Saved final state to {npz_path}")

    # Save final state visualization.
    plot_four_frames(
        A1_hist[-1],
        I1_hist[-1],
        A2_hist[-1],
        I2_hist[-1],
        final_step=final_step,
        outfile_png=str(png_final_path),
        title="Final State",
    )
    print(f"Saved final image to {png_final_path}")

    # Save initial condition visualization.
    plot_four_frames(
        A1_hist[0],
        I1_hist[0],
        A2_hist[0],
        I2_hist[0],
        final_step=0,
        outfile_png=str(png_init_path),
        title="Initial State",
    )
    print(f"Saved initial condition to {png_init_path}")


if __name__ == "__main__":
    main()
