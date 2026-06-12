#!/usr/bin/env python3

from pathlib import Path
import re
import argparse

import numpy as np
import pandas as pd


def extract_step(filename: str) -> int:
    """
    Extract step number from filenames like: step_004500.npz -> 4500
    """
    m = re.search(r"step_(\d+)\.npz$", filename)
    if not m:
        raise ValueError(f"Could not parse step number from filename: {filename}")
    return int(m.group(1))


def count_above_threshold(npz_path: Path, field: str, threshold: float) -> int:
    """
    Load the chosen array from the .npz and count entries strictly above threshold.
    """
    data = np.load(npz_path, allow_pickle=True)

    if field not in data:
        raise KeyError(f"Field '{field}' not found in {npz_path.name}. Available keys: {list(data.files)}")

    arr = np.asarray(data[field], dtype=float)
    return int(np.sum(np.isfinite(arr) & (arr > threshold)))


def process_folder(input_folder: str, threshold: float, field: str = "A_final", output_csv: str | None = None):
    input_folder = Path(input_folder)

    npz_files = sorted(
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".npz" and re.match(r"step_\d+\.npz$", p.name)
    )

    if not npz_files:
        raise FileNotFoundError(f"No files matching step_XXXXXX.npz found in {input_folder}")

    rows = []
    for npz_path in npz_files:
        step = extract_step(npz_path.name)
        count = count_above_threshold(npz_path, field=field, threshold=threshold)
        rows.append({"step": step, "n_hexagons_above_threshold": count})

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    if output_csv is None:
        output_csv = input_folder / "activated_hexagons_by_step.csv"
    else:
        output_csv = Path(output_csv)

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count number of hexagons above a threshold for all step_*.npz files in a folder."
    )
    parser.add_argument("input_folder", help="Folder containing step_*.npz files")
    parser.add_argument("--threshold", type=float, required=True, help="Input threshold value")
    parser.add_argument("--field", default="A_final", help="Array name inside each .npz file (default: A_final)")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path")

    args = parser.parse_args()

    process_folder(
        input_folder=args.input_folder,
        threshold=args.threshold,
        field=args.field,
        output_csv=args.output_csv,
    )
