"""
Run batch simulations from a YAML configuration file.

This script:
- Loads a base parameter set and parameter sweeps from a YAML config
- Generates a parameter grid (or other sweep mode)
- Runs simulations in parallel
- Saves:
    - constants.txt (non-varied parameters)
    - batch_results.csv (varied parameters + selected outputs)

Dependencies:
    pip install pandas joblib tqdm pyyaml

Example
-------
python batch_runner.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Any

import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

# Allow importing simulation.py from parent directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation import run_simulation
from grid import make_param_grid
from io_utils import _to_json_list, write_constants_txt


# Columns to include in the output CSV.
OUTPUT_COLS = [
    "steps_used",
    "activator_steady-state",
    "inhibitor_steady-state",
    "activator_final",
    "inhibitor_final",
]


def run_one(p: Mapping[str, Any], varied_keys: List[str]) -> Dict[str, Any]:
    """
    Run a single simulation and format its output for tabular storage.

    Parameters
    ----------
    p
        Parameter dictionary for one simulation run.
    varied_keys
        Keys corresponding to parameters that were varied in the sweep.

    Returns
    -------
    dict
        Row containing:
        - varied parameter values
        - selected simulation outputs
    """
    result = run_simulation(p)

    # Start with the varied parameters.
    row = {k: p[k] for k in varied_keys}

    # Add selected outputs (converted to JSON-friendly format if needed).
    row.update(
        {
            "steps_used": result.get("steps_used"),
            "activator_steady-state": _to_json_list(
                result.get("activator_steady-state")
            ),
            "inhibitor_steady-state": _to_json_list(
                result.get("inhibitor_steady-state")
            ),
            "activator_final": _to_json_list(result.get("activator_final")),
            "inhibitor_final": _to_json_list(result.get("inhibitor_final")),
        }
    )

    return row


def main() -> None:
    """
    Entry point for running parameter sweeps defined in a YAML configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Run batch simulations from a YAML configuration."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    # Load YAML configuration.
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract configuration fields.
    outdir = cfg.get("outdir", "runs/default")
    mode = cfg.get("mode", "grid")
    base_params = cfg["base"]
    sweeps = cfg.get("sweeps", {})

    os.makedirs(outdir, exist_ok=True)

    varied_keys = list(sweeps.keys())

    # Generate parameter combinations.
    param_list = make_param_grid(base_params, sweeps=sweeps, mode=mode)

    # Extract constants (non-varied parameters).
    constants = {k: v for k, v in base_params.items() if k not in varied_keys}
    write_constants_txt(constants, os.path.join(outdir, "constants.txt"))

    # Run simulations in parallel.
    results = Parallel(n_jobs=-1)(
        delayed(run_one)(p, varied_keys)
        for p in tqdm(param_list, desc="Running simulations")
    )

    # Build DataFrame with only varied parameters + selected outputs.
    df = pd.DataFrame(results)[varied_keys + OUTPUT_COLS]

    # Save results.
    csv_path = os.path.join(outdir, "batch_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"Wrote {len(constants)} constants to {outdir}/constants.txt")
    print(f"Saved {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
