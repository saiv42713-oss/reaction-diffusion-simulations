"""
Main script to run a single 1D JAPI/PAPI simulation from the command line.

The script can:
- run the simulation,
- save a text summary and static plots,
- optionally save the initial frame,
- optionally generate a movie,
- optionally display visualization interactively.

Example:
python main.py --output test_run --movie
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from parameters import (
    activator_type,
    dt,
    dx,
    init_mode,
    min_steps,
    N,
    params,
    save_every,
    spike_value,
    steps,
    stopping_threshold,
)
from simulation import run_coupled_neumann
from visualize import animate_histories, plot_one_frame
from writing_simulation_results import str2bool, write_simulation_results


def main() -> None:
    """
    Run the coupled Neumann simulation from the command line.

    The script can:
    - run the simulation,
    - save a text summary and static plots,
    - optionally save the initial frame,
    - optionally generate a movie,
    - optionally display visualization interactively.
    """
    parser = argparse.ArgumentParser(
        description="Run coupled Neumann simulation."
    )
    parser.add_argument(
        "--output",
        type=str,
        help=(
            "Base name of output file (without extension). "
            "Will save simulation_results/NAME.txt and NAME.png."
        ),
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="If set, also save the initial frame as a PNG.",
    )
    parser.add_argument(
        "--vis",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Whether to show visualization (True/False). Default = True, "
            "unless --output is activated."
        ),
    )
    parser.add_argument(
        "--movie",
        action="store_true",
        help="If set, also save an MP4 movie using the same basename as --output.",
    )
    args = parser.parse_args()

    # If a movie is requested and the user did not explicitly set --vis,
    # disable interactive visualization so the script can run non-interactively.
    if args.movie and args.vis is True and "--vis" not in " ".join(os.sys.argv):
        args.vis = False

    # Run the baseline simulation:
    # - two activator spikes
    # - Neumann boundary conditions
    A_hist, R_hist, final_step, a_ss, i_ss = run_coupled_neumann(
        N,
        steps,
        dt,
        dx,
        params,
        stopping_threshold,
        min_steps,
        init_mode=init_mode,
        activator_type=activator_type,
        spike_value=spike_value,
        save_every=save_every,
    )

    # Save text output and a final-frame plot when --output is provided.
    if args.output:
        outfile_txt, outfile_png = write_simulation_results(
            args,
            activator_type,
            init_mode,
            spike_value,
            params,
            A_hist,
            R_hist,
            final_step,
        )

        # Save a static plot of the final state.
        plot_one_frame(A_hist[-1], R_hist[-1], final_step, outfile_png)
        print(f"Final state plot saved to {outfile_png}")

        # Optionally save the first recorded frame as well.
        if args.start:
            outdir = "simulation_results"
            outfile_start = os.path.join(outdir, args.output + "_start.png")
            plot_one_frame(A_hist[1], R_hist[1], 0, outfile_start)
            print(f"Initial state plot saved to {outfile_start}")

    # Show visualization and/or save a movie.
    if args.vis or args.movie:
        outdir = "simulation_results"
        movie_path: Optional[str] = None

        # Movie saving requires --output so the file has a basename.
        if args.movie:
            if not args.output:
                raise ValueError("--movie requires --output to be specified.")
            os.makedirs(outdir, exist_ok=True)
            movie_path = os.path.join(outdir, args.output + ".mp4")

        animate_histories(
            A_hist,
            R_hist,
            save_every,
            title="Baseline Simulation (Neumann)",
            loop=False,
            savefile=movie_path,
        )


if __name__ == "__main__":
    main()
