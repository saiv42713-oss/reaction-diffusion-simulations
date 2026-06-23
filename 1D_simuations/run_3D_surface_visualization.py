"""
Main script to run a single 1D JAPI/PAPI simulation from the command line,
and to visualize the output as a 3D plot of the evolution of the signal
of the activator accross all points accross time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from typing import Iterable, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from simulation import run_coupled_neumann
from parameters import (
    params,
    N,
    steps,
    dt,
    dx,
    save_every,
    spike_value,
    stopping_threshold,
    min_steps,
    init_mode,
    activator_type,
)
import parameters as parameters_module

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


def format_value(val) -> str:
    """
    Format a value for writing to a text snapshot file.

    Arrays are rendered compactly, while mappings and sequences use repr().
    """
    if isinstance(val, np.ndarray):
        try:
            return np.array2string(val, threshold=100, max_line_width=200)
        except Exception:
            return repr(val)

    if isinstance(val, (dict, list, tuple, set)):
        return repr(val)

    return repr(val)


def write_parameters_file(output_base: str, param_module, var_names: Sequence[str]) -> str:
    """
    Write a snapshot of selected runtime parameters to:

        simulation_results/3D_plots/{output_base}.txt

    Parameters
    ----------
    output_base
        Base filename without extension.
    param_module
        Module object from which parameters are read.
    var_names
        Names of variables to capture in the snapshot.

    Returns
    -------
    str
        Full path to the written text file.
    """
    out_dir = os.path.join("simulation_results", "3D_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_base + ".txt")

    now = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Snapshot of parameters from parameters.py\n")
        f.write(f"# Written at (UTC): {now}\n")
        f.write(f"# parameters module file: {getattr(param_module, '__file__', 'unknown')}\n\n")

        for name in var_names:
            # Prefer the live module value if present.
            try:
                val = getattr(param_module, name)
            except AttributeError:
                # Fall back to the current module namespace for imported symbols.
                val = globals().get(name, "<not found>")
            f.write(f"{name} = {format_value(val)}\n\n")

    print(f"Wrote parameter snapshot to {out_path}")
    return out_path


def prepare_arrays(
    A_hist: Sequence[np.ndarray],
    dt: float,
    save_every: int,
    physical_length: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a history of 1D activator arrays into arrays suitable for surface plotting.

    Parameters
    ----------
    A_hist
        Sequence of activator profiles over time. Each entry should be shape (N,).
    dt
        Simulation time step.
    save_every
        Number of simulation steps between recorded snapshots.
    physical_length
        If provided, use a physical x-axis from 0 to physical_length.
        Otherwise use grid indices.

    Returns
    -------
    positions, times, X, Y, Z
        positions : 1D spatial axis
        times     : 1D time axis
        X, Y      : 2D meshgrid arrays
        Z         : 2D activator surface with shape (T, N)
    """
    A = np.array(A_hist)
    T, N = A.shape

    if physical_length is None:
        positions = np.arange(N)
    else:
        positions = np.linspace(0.0, physical_length, N)

    times = np.arange(T) * (dt * save_every)
    X, Y = np.meshgrid(positions, times)
    Z = A

    return positions, times, X, Y, Z


def plot_surface_matplotlib(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    out_png: str | None = None,
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    """
    Render a 3D surface plot with Matplotlib.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # If needed, you can downsample here for speed on very large grids.
    Xs, Ys, Zs = X, Y, Z

    surf = ax.plot_surface(
        Xs,
        Ys,
        Zs,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
    )
    ax.set_xlabel("position")
    ax.set_ylabel("time (units)")
    ax.set_zlabel("activator concentration")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.6, aspect=12, label="activator")

    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"Saved matplotlib surface to {out_png}")
    else:
        plt.show()

    plt.close(fig)


def plot_surface_plotly(
    positions: np.ndarray,
    times: np.ndarray,
    Z: np.ndarray,
    out_html: str | None = None,
    title: str = "Activator surface",
) -> None:
    """
    Render an interactive 3D surface plot with Plotly.
    """
    if not PLOTLY_OK:
        raise RuntimeError("Plotly not available. Install plotly to use interactive output.")

    fig = go.Figure(
        data=[
            go.Surface(
                x=positions,
                y=times,
                z=Z,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="position",
            yaxis_title="time (units)",
            zaxis_title="activator concentration",
        ),
        title=title,
        width=1000,
        height=700,
    )

    if out_html:
        fig.write_html(out_html)
        print(f"Saved Plotly interactive surface to {out_html}")
    else:
        fig.show()


def save_simulation_data(
    output_base: str,
    positions: np.ndarray,
    times: np.ndarray,
    Z: np.ndarray,
) -> str:
    """
    Save the simulation output as a compressed NPZ file.

    Parameters
    ----------
    output_base
        Base filename without extension.
    positions
        1D spatial axis.
    times
        1D time axis.
    Z
        2D activator surface with shape (T, N).

    Returns
    -------
    str
        Full path to the written NPZ file.
    """
    out_dir = os.path.join("simulation_results", "3D_plots")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, output_base + "_data.npz")

    np.savez_compressed(
        out_path,
        activator=Z,
        positions=positions,
        times=times,
    )

    print(f"Saved simulation data to {out_path}")
    return out_path


def run_and_plot(args: argparse.Namespace) -> None:
    """
    Run the coupled Neumann simulation and generate the requested outputs.
    """
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

    positions, times, X, Y, Z = prepare_arrays(
        A_hist,
        dt,
        save_every,
        physical_length=None,
    )

    # Optional raw data export.
    if args.save_data and args.output:
        save_simulation_data(args.output, positions, times, Z)

    # Save a parameter snapshot whenever an output basename is given.
    if args.output:
        var_names = [
            "params",
            "N",
            "steps",
            "dt",
            "dx",
            "save_every",
            "spike_value",
            "stopping_threshold",
            "min_steps",
            "init_mode",
            "activator_type",
        ]
        write_parameters_file(args.output, parameters_module, var_names)

    # Matplotlib static output.
    if args.matplotlib:
        out_png = None
        if args.output:
            os.makedirs("simulation_results/3D_plots", exist_ok=True)
            out_png = os.path.join(
                "simulation_results/3D_plots",
                args.output + "_surface.png",
            )
        plot_surface_matplotlib(X, Y, Z, out_png=out_png, elev=args.elev, azim=args.azim)

    # Plotly interactive output.
    if args.plotly:
        out_html = None
        if args.output:
            os.makedirs("simulation_results/3D_plots", exist_ok=True)
            out_html = os.path.join(
                "simulation_results/3D_plots",
                args.output + "_surface.html",
            )
        plot_surface_plotly(positions, times, Z, out_html=out_html, title=args.title)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run simulation and produce 3D surface plots of activator vs position vs time."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Base name to save outputs in simulation_results/3D_plots/.",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Produce a static Matplotlib PNG.",
    )
    parser.add_argument(
        "--plotly",
        action="store_true",
        help="Produce an interactive Plotly HTML (requires plotly installed).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Elevation angle for Matplotlib view_init.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Azimuth angle for Matplotlib view_init.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Activator surface",
        help="Title for the Plotly figure.",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save full simulation data (activator, position, time) as compressed NPZ.",
    )

    args = parser.parse_args()

    # Default behavior: if the user selects neither backend, use both when available.
    if not args.matplotlib and not args.plotly:
        args.matplotlib = True
        args.plotly = PLOTLY_OK

    # If Plotly is unavailable, silently disable it rather than failing later.
    if args.plotly and not PLOTLY_OK:
        print("Plotly is not installed; skipping interactive HTML output.")
        args.plotly = False

    return args


if __name__ == "__main__":
    run_and_plot(parse_args())
