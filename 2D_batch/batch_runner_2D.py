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

# Allow importing simulation.py from the parent directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grid import make_param_grid
from io_utils import write_constants_txt
from simulation_2D import run_coupled_hex as run_coupled_hex_sim
from visualize_2D import plot_one_frame


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
    )

    A_hist, R_hist, steps_used, a_ss, i_ss = result

    return {
        "status": "done",
        "steps_used": steps_used,
        "parameters": params,
        "activator_initial": A_hist[0],
        "activator_final": A_hist[-1],
        "inhibitor_initial": R_hist[0],
        "inhibitor_final": R_hist[-1],
        "activator_steady-state": a_ss,
        "inhibitor_steady-state": i_ss,
    }


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
    plot_one_frame(A_final, R_final, row["steps_used"], outfile_png)

    a_max = A_final.max()
    a_min = A_final.min()
    a_diff = a_max - a_min
    r_max = R_final.max()
    r_min = R_final.min()
    r_diff = r_max - r_min

    row.update(
        {
            "a_max": a_max,
            "a_min": a_min,
            "a_diff": a_diff,
            "r_max": r_max,
            "r_min": r_min,
            "r_diff": r_diff,
        }
    )

    np.savez(
        os.path.join(outdir, f"field_{run_id:04d}.npz"),
        A_final=np.asarray(A_final),
        R_final=np.asarray(R_final),
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
        default="config_2D.yaml",
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
