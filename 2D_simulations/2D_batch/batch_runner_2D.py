"""
Run a batch of 2D simulations from a YAML configuration file.

This script:
- loads a base parameter set and sweeps from a YAML config,
- builds a parameter grid,
- runs each simulation in parallel,
- saves a table of swept parameters,
- writes constants to a text file,
- saves one PNG and one NPZ file per simulation run,
- collects summary statistics into a CSV file.

Outputs are written under the configured output directory.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

# Allow importing simulation.py from the parent directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grid import make_param_grid
from io_utils import write_constants_txt
from simulation_2D import run_coupled_hex as run_coupled_hex_sim
from visualize_2D import animate_histories, plot_one_frame


# Set a fixed seed for reproducible random sampling.
random.seed(2)


def save_runs_table(
    param_list: list[dict[str, Any]],
    varied_keys: list[str],
    filepath: str,
) -> None:
    """
    Save a tab-separated table with one row per simulation.

    The table includes one column per swept parameter. Nested sweep parameters
    are flattened into columns of the form ``parent_child``.
    """
    rows: list[dict[str, Any]] = []

    for sim_number, p in enumerate(param_list):
        row: dict[str, Any] = {"sim_number": sim_number}

        for k in varied_keys:
            if isinstance(p[k], dict):
                for subk, subv in p[k].items():
                    row[f"{k}_{subk}"] = subv
            else:
                row[k] = p[k]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, sep="\t", index=False)


def run_one_simulation(params: dict[str, Any]) -> dict[str, Any]:
    """
    Run one 2D simulation and return the fields needed for downstream export.
    """
    result = run_coupled_hex_sim(
        params["Ny"],
        params["Nx"],
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
        noise_amplitude=params.get("noise_amplitude", 0.0),
        nucleation_rate=params.get("nucleation_rate", 0.02),
        n_points=int(params.get("n_points", 10)),
        dense_save_until_step=int(params.get("dense_save_until_step", 0)),
        dense_save_every=int(params.get("dense_save_every", params.get("save_every", 100))),
    )

    A_hist, R_hist, steps_used, a_ss, i_ss = result

    return {
        "status": "done",
        "steps_used": steps_used,
        "parameters": params,
        "A_hist": A_hist,
        "R_hist": R_hist,
        "activator_initial": A_hist[0],
        "activator_final": A_hist[-1],
        "inhibitor_initial": R_hist[0],
        "inhibitor_final": R_hist[-1],
        "activator_steady-state": a_ss,
        "inhibitor_steady-state": i_ss,
    }


# Plot max and mean per-cell change for analyzing termination conditions
def plot_pc_change(
    max_change: np.ndarray,
    mean_change: np.ndarray,
    save_every: int,
    filepath: str,
    max_change_R: np.ndarray | None = None,
    mean_change_R: np.ndarray | None = None,
) -> None:
    time_steps = np.arange(1, len(max_change) + 1) * save_every
    has_inhibitor = max_change_R is not None and mean_change_R is not None

    if has_inhibitor:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(time_steps, max_change, label="Max per-cell change")
    ax1.plot(time_steps, mean_change, label="Mean per-cell change")
    ax1.set_yscale("log")
    ax1.set_xlabel("Simulation step")
    ax1.set_ylabel("|ΔA| per saved frame")  # Change in activator field
    ax1.set_title("Activator termination metric")
    ax1.legend()

    if has_inhibitor:
        ax2.plot(time_steps, max_change_R, label="Max per-cell change")
        ax2.plot(time_steps, mean_change_R, label="Mean per-cell change")
        ax2.set_yscale("log")
        ax2.set_xlabel("Simulation step")
        ax2.set_ylabel("|ΔR| per saved frame")  # Change in inhibitor field
        ax2.set_title("Inhibitor termination metric")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def run_one(
    p: dict[str, Any],
    varied_keys: list[str],
    outdir: str,
    run_id: int,
) -> dict[str, Any]:
    """
    Run one parameter set, save outputs, and return a summary row.
    """
    r = run_one_simulation(p)

    row = {k: p[k] for k in varied_keys}
    row["steps_used"] = r.get("steps_used")
    row["a_ss"] = r.get("activator_steady-state")
    row["i_ss"] = r.get("inhibitor_steady-state")

    A_final = r.get("activator_final")
    R_final = r.get("inhibitor_final")

    outfile_png = os.path.join(outdir, f"run_{run_id:04d}.png")
    plot_one_frame(
        A_final,
        R_final,
        row["steps_used"],
        outfile_png,
        A_initial=r.get("activator_initial"),
        R_initial=r.get("inhibitor_initial"),
    )

    # Additional video logic similar to main_2D
    if p.get("video_enable", False):
        video_dir = os.path.join(outdir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        animate_histories(
            r["A_hist"],
            r["R_hist"],
            p.get("save_every", 100),
            title=f"Run {run_id:04d}",
            loop=False,
            savefile=os.path.join(video_dir, f"run_{run_id:04d}.mp4"),
            fps=60,  # shorter playback, no frame loss
        )

    a_max = A_final.max()
    a_min = A_final.min()
    a_diff = a_max - a_min
    r_max = R_final.max()
    r_min = R_final.min()
    r_diff = r_max - r_min

    # These are cheap numpy ops, computed unconditionally so the npz save below
    # never crashes regardless of term_plots_enable. Only the plot image and the
    # CSV summary columns are gated behind the toggle, matching the original intent
    # of skipping the *expensive* parts for large batches.
    A_hist = np.asarray(r["A_hist"])
    R_hist = np.asarray(r["R_hist"])
    dA = np.abs(np.diff(A_hist, axis=0))
    dR = np.abs(np.diff(R_hist, axis=0))
    max_change = dA.reshape(dA.shape[0], -1).max(axis=1)
    mean_change = dA.reshape(dA.shape[0], -1).mean(axis=1)
    max_change_R = dR.reshape(dR.shape[0], -1).max(axis=1)
    mean_change_R = dR.reshape(dR.shape[0], -1).mean(axis=1)

    # Termination conditions
    if p.get("term_plots_enable", False):
        term_dir = os.path.join(outdir, "termination_plots")
        os.makedirs(term_dir, exist_ok=True)

        plot_pc_change(
            max_change,
            mean_change,
            p.get("save_every", 100),
            os.path.join(term_dir, f"plot_{run_id:04d}.png"),
            max_change_R=max_change_R,
            mean_change_R=mean_change_R,
        )

        row.update(
            {
                "a_max": a_max,
                "a_min": a_min,
                "a_diff": a_diff,
                "r_max": r_max,
                "r_min": r_min,
                "r_diff": r_diff,
                "max_change_final": max_change[-1],
                "mean_change_final": mean_change[-1],
                "max_change_peak": max_change.max(),
                "mean_change_peak": mean_change.max(),
                "max_change_R_final": max_change_R[-1],
                "mean_change_R_final": mean_change_R[-1],
                "max_change_R_peak": max_change_R.max(),
                "mean_change_R_peak": mean_change_R.max(),
            }
        )

    np.savez(
        os.path.join(outdir, f"field_{run_id:04d}.npz"),
        A_initial=np.asarray(r["activator_initial"]),
        R_initial=np.asarray(r["inhibitor_initial"]),
        A_final=np.asarray(A_final),
        R_final=np.asarray(R_final),
        A_hist=A_hist,
        R_hist=R_hist,
        max_change=max_change,
        mean_change=mean_change,
        max_change_R=max_change_R,
        mean_change_R=mean_change_R,
        save_every=p.get("save_every", 100),
        steps_used=r["steps_used"],

    )

    return row


def main() -> None:
    """
    Parse the YAML config, run the sweep, and write batch outputs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to YAML config",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    outdir = cfg.get("outdir", "runs/default")
    mode = cfg.get("mode", "grid")
    base = cfg["base"]
    sweeps = cfg.get("sweeps", {})

    os.makedirs(outdir, exist_ok=True)
    varied_keys = list(sweeps.keys())

    param_list = make_param_grid(base, sweeps=sweeps, mode=mode)

    runs_txt_path = os.path.join(outdir, "runs.txt")
    save_runs_table(param_list, varied_keys, runs_txt_path)

    constants = {k: v for k, v in base.items() if k not in varied_keys}
    write_constants_txt(constants, os.path.join(outdir, "constants.txt"))

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(p, varied_keys, outdir, i)
        for i, p in enumerate(tqdm(param_list, desc="Running simulations"))
    )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "batch_results.csv"), index=False)

    print(f"Wrote {len(constants)} constants to {outdir}/constants.txt")
    print(f"Saved {len(df)} rows to {outdir}/batch_results.csv")


if __name__ == "__main__":
    main()
