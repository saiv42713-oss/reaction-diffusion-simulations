"""
Run a batch of coupled 2D two-pair simulations from a YAML configuration file.

This script:
- loads a base parameter set and parameter sweeps from YAML,
- builds a parameter grid,
- runs each coupled simulation in parallel,
- saves a table of swept parameters,
- writes constants to a text file,
- saves one PNG and one NPZ file per simulation run,
- collects summary statistics into a CSV file.

Outputs are written under the configured output directory.
"""

from __future__ import annotations

import argparse
import json
import os
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
from simulation_2D import run_two_pair_hex as run_coupled_hex_sim
from visualize_2D import plot_four_frames


def _safe_name_value(v: Any) -> str:
    """
    Convert a parameter value into a filename-safe string.
    """
    if isinstance(v, float):
        return f"{v:g}"
    return str(v).replace("/", "_").replace(" ", "")


def _build_run_name(p: dict[str, Any], varied_keys: list[str]) -> str:
    """
    Build a compact run name from the varied parameters only.
    """
    parts = []
    for k in varied_keys:
        parts.append(f"{k}={_safe_name_value(p[k])}")
    return "_".join(parts) if parts else "run"


def run_one_simulation(p: dict[str, Any]) -> dict[str, Any]:
    """
    Run one coupled two-pair simulation and return the full histories.
    """
    result = run_coupled_hex_sim(
        p["Ny"],
        p["Nx"],
        p["steps"],
        p["dt"],
        p["dx"],
        p["p1"],
        p["p2"],
        p.get("stopping_threshold", 1e-4),
        p.get("min_steps", 10000),
        init_mode=p.get("init_mode", "activator_spike"),
        activator_type1=p.get("activator_type1", "juxtacrine"),
        activator_type2=p.get("activator_type2", "juxtacrine"),
        n_points=p.get("n_points", 10),
        spike_value=p.get("spike_value", 5.0),
        save_every=p.get("save_every", 100),
    )

    A1_hist, I1_hist, A2_hist, I2_hist, steps_used, steady_states = result

    return {
        "status": "done",
        "steps_used": steps_used,
        "parameters": p,
        "A1_initial": A1_hist[0],
        "A1_final": A1_hist[-1],
        "I1_initial": I1_hist[0],
        "I1_final": I1_hist[-1],
        "A2_initial": A2_hist[0],
        "A2_final": A2_hist[-1],
        "I2_initial": I2_hist[0],
        "I2_final": I2_hist[-1],
        "steady_states": steady_states,
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

    A1_final = np.asarray(r["A1_final"])
    I1_final = np.asarray(r["I1_final"])
    A2_final = np.asarray(r["A2_final"])
    I2_final = np.asarray(r["I2_final"])

    row = {k: p[k] for k in varied_keys}
    row["steps_used"] = r["steps_used"]

    ss = r["steady_states"]
    row["a1_ss"] = ss["pair1"]["a_ss"]
    row["i1_ss"] = ss["pair1"]["i_ss"]
    row["a2_ss"] = ss["pair2"]["a_ss"]
    row["i2_ss"] = ss["pair2"]["i_ss"]

    # Summary metrics.
    row.update(
        {
            "a1_max": float(A1_final.max()),
            "a1_min": float(A1_final.min()),
            "a1_diff": float(A1_final.max() - A1_final.min()),
            "i1_max": float(I1_final.max()),
            "i1_min": float(I1_final.min()),
            "i1_diff": float(I1_final.max() - I1_final.min()),
            "a2_max": float(A2_final.max()),
            "a2_min": float(A2_final.min()),
            "a2_diff": float(A2_final.max() - A2_final.min()),
            "i2_max": float(I2_final.max()),
            "i2_min": float(I2_final.min()),
            "i2_diff": float(I2_final.max() - I2_final.min()),
        }
    )

    # Keep filenames short to avoid path-length issues.
    base_name = f"run_{run_id:04d}"

    # Save PNG.
    outfile_png = os.path.join(outdir, f"{base_name}.png")
    plot_four_frames(
        A1_final,
        I1_final,
        A2_final,
        I2_final,
        final_step=r["steps_used"],
        outfile_png=outfile_png,
        title="Coupled Patterning Final State",
    )

    # Save final-state NPZ.
    np.savez_compressed(
        os.path.join(outdir, f"{base_name}_final_state.npz"),
        a1=A1_final,
        i1=I1_final,
        a2=A2_final,
        i2=I2_final,
        steps_used=np.array(r["steps_used"]),
        a1_ss=np.array(ss["pair1"]["a_ss"]),
        i1_ss=np.array(ss["pair1"]["i_ss"]),
        a2_ss=np.array(ss["pair2"]["a_ss"]),
        i2_ss=np.array(ss["pair2"]["i_ss"]),
        parameters=json.dumps(p, default=str),
    )

    return row


def _flatten_sweep_paths(
    sweeps: dict[str, Any],
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    """
    Recursively collect leaf paths from a nested sweeps dictionary.

    Example
    -------
    {"p1": {"act_prod_rate": [...]}, "p2": {"act_prod_rate": [...]}}
    becomes
    [("p1", "act_prod_rate"), ("p2", "act_prod_rate")]
    """
    paths: list[tuple[str, ...]] = []
    for k, v in sweeps.items():
        if isinstance(v, dict):
            paths.extend(_flatten_sweep_paths(v, prefix + (k,)))
        else:
            paths.append(prefix + (k,))
    return paths


def _get_nested_value(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    """
    Retrieve a nested value from a dictionary using a tuple path.
    """
    for key in path:
        d = d[key]
    return d


def save_runs_table(
    param_list: list[dict[str, Any]],
    sweeps: dict[str, Any],
    filepath: str,
) -> None:
    """
    Save a tab-separated table with one row per simulation and one column per sweep leaf.
    """
    sweep_paths = _flatten_sweep_paths(sweeps)

    rows: list[dict[str, Any]] = []
    for sim_number, p in enumerate(param_list):
        row: dict[str, Any] = {"sim_number": sim_number}
        for path in sweep_paths:
            col_name = "_".join(path)
            row[col_name] = _get_nested_value(p, path)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, sep="\t", index=False)


def main() -> None:
    """
    Parse the YAML config, run the coupled sweep, and write output files.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        "-c",
        default="config_coupled_2D.yaml",
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

    # Build parameter grid.
    param_list = make_param_grid(base, sweeps=sweeps, mode=mode)

    # Save table of run numbers and swept values.
    runs_txt_path = os.path.join(outdir, "runs.txt")
    save_runs_table(param_list, sweeps, runs_txt_path)

    # Write constants file for everything not varied.
    constants = {k: v for k, v in base.items() if k not in varied_keys}
    write_constants_txt(constants, os.path.join(outdir, "constants.txt"))

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(p, varied_keys, outdir, i)
        for i, p in enumerate(tqdm(param_list, desc="Running coupled simulations"))
    )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "batch_results.csv"), index=False)

    print(f"Wrote constants to {outdir}/constants.txt")
    print(f"Saved {len(df)} rows to {outdir}/batch_results.csv")


if __name__ == "__main__":
    main()
