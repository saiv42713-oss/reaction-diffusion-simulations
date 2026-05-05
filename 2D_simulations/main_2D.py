"""
Run a single 2D hex-grid simulation from the command line.

This script:
- runs one 2D coupled hex simulation,
- saves a parameter snapshot,
- saves either the final frame only or every stored snapshot,
- writes both .npz and .png outputs,
- generates an MP4 movie of the run.

By default, outputs are written under:
    results/<output_name>/
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from parameters_2D import (
    activator_type,
    dt,
    dx,
    init_mode,
    min_steps,
    n_points,
    noise_amplitude,
    nucleation_rate,
    Ny,
    Nx,
    params,
    save_every,
    spike_value,
    steps,
    stopping_threshold,
)
from simulation_2D import run_coupled_hex
from visualize_2D import animate_histories, plot_one_frame


def save_parameters_txt(
    filepath: Path,
    params_dict: dict,
    sim_params: dict,
) -> None:
    """
    Save a human-readable text file containing simulation and model parameters.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== Single 2D Simulation Parameters ===\n\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n\n")

        f.write("---- Simulation parameters ----\n")
        for k, v in sim_params.items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- Model parameters ----\n")
        for k, v in params_dict.items():
            f.write(f"{k}: {v}\n")


def save_snapshot_npz(
    filepath: Path,
    A,
    R,
    step: int,
    a_ss,
    i_ss,
) -> None:
    """
    Save a single simulation snapshot in compressed NumPy format.
    """
    np.savez_compressed(
        filepath,
        a=np.asarray(A),
        i=np.asarray(R),
        step=np.array(step),
        a_ss=np.array(a_ss),
        i_ss=np.array(i_ss),
    )


def main() -> None:
    """
    Run the 2D simulation and save plots, snapshots, and movie outputs.
    """
    parser = argparse.ArgumentParser(
        description="Run a single 2D hex simulation."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Name of the output folder inside results/.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Save one .npz and one .png every N simulation steps.",
    )
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Use the user-provided save interval if given; otherwise keep the default.
    save_interval = args.step_size if args.step_size is not None else save_every

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output if args.output else f"single_{timestamp}"

    output_dir = results_dir / output_name
    npz_dir = output_dir / "npz"
    png_dir = output_dir / "png"

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    movie_path = output_dir / "movie.mp4"
    param_path = output_dir / "params.txt"

    # Run the simulation.
    A_hist, R_hist, final_step, a_ss, i_ss = run_coupled_hex(
        Ny,
        Nx,
        steps,
        dt,
        dx,
        params,
        stopping_threshold,
        min_steps,
        init_mode=init_mode,
        activator_type=activator_type,
        spike_value=spike_value,
        save_every=save_interval,
        n_points=n_points,
        noise_amplitude=noise_amplitude,
        nucleation_rate=nucleation_rate,
    )

    print(f"Simulation finished at step {final_step}")
    print(f"Steady states: a_ss={a_ss}, i_ss={i_ss}")

    sim_params = {
        "Ny": Ny,
        "Nx": Nx,
        "steps": steps,
        "dt": dt,
        "dx": dx,
        "save_every": save_interval,
        "stopping_threshold": stopping_threshold,
        "min_steps": min_steps,
        "init_mode": init_mode,
        "activator_type": activator_type,
        "spike_value": spike_value,
        "n_points": n_points,
        "noise_amplitude": noise_amplitude,
        "nucleation_rate": nucleation_rate,
    }

    save_parameters_txt(param_path, params, sim_params)
    print(f"Saved parameters to {param_path}")

    # If no custom step size was requested, save only the final frame.
    if args.step_size is None:
        npz_file = npz_dir / f"final_step_{final_step:06d}.npz"
        png_file = png_dir / f"final_step_{final_step:06d}.png"

        save_snapshot_npz(npz_file, A_hist[-1], R_hist[-1], final_step, a_ss, i_ss)
        plot_one_frame(
            A_hist[-1],
            R_hist[-1],
            final_step,
            str(png_file),
            title=f"Final state (step {final_step})",
        )

        print("Saved only final frame")

    # Otherwise save every stored snapshot.
    else:
        for idx, (A, R) in enumerate(zip(A_hist, R_hist)):
            step = idx * args.step_size
            if idx == len(A_hist) - 1:
                step = final_step

            npz_file = npz_dir / f"step_{step:06d}.npz"
            png_file = png_dir / f"step_{step:06d}.png"

            save_snapshot_npz(npz_file, A, R, step, a_ss, i_ss)
            plot_one_frame(
                A,
                R,
                step,
                str(png_file),
                title=f"State at step {step}",
            )

        print(f"Saved snapshots every {args.step_size} steps")

    animate_histories(
        A_hist,
        R_hist,
        save_interval,
        title="Baseline Simulation (2D Hex Grid)",
        loop=False,
        savefile=str(movie_path),
    )
    print(f"Saved movie to {movie_path}")


if __name__ == "__main__":
    main()
