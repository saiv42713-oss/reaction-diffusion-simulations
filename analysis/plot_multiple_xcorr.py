#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

FIGSIZE = (4, 3)
DPI = 300
OUTPUT_FORMAT = "pdf"


def load_profile(
    hdf5_file: Path,
    key: str,
    x_col: str,
    y_col: str,
    low_col: str | None = None,
    high_col: str | None = None,
):
    """
    Load x/y profile data from an HDF5 file.

    Supports two storage formats:
    1) pandas HDFStore table at `key`
    2) HDF5 group `key` containing datasets named by column
    """
    df = None

    # Try pandas HDFStore first
    try:
        df = pd.read_hdf(hdf5_file, key=key)
    except Exception:
        df = None

    # If that fails, try h5py group-style storage
    if df is None:
        try:
            with h5py.File(hdf5_file, "r") as h:
                if key not in h:
                    raise ValueError(f"Key/group '{key}' not found")
                grp = h[key]

                cols = [x_col, y_col]
                if low_col is not None and high_col is not None:
                    cols.extend([low_col, high_col])

                data = {}
                for col in cols:
                    if col in grp:
                        data[col] = np.asarray(grp[col])
                    else:
                        raise ValueError(
                            f"Column '{col}' not found in group '{key}' of {hdf5_file.name}"
                        )

                df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Could not read key '{key}' from {hdf5_file.name}: {e}")

    required_cols = {x_col, y_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{hdf5_file.name} is missing expected columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    cols = [x_col, y_col]
    if low_col is not None and high_col is not None:
        if low_col in df.columns and high_col in df.columns:
            cols += [low_col, high_col]
        else:
            low_col = None
            high_col = None

    df = df[cols].dropna().sort_values(x_col)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    if low_col is not None and high_col is not None:
        y_low = df[low_col].to_numpy()
        y_high = df[high_col].to_numpy()
    else:
        y_low = None
        y_high = None

    return x, y, y_low, y_high


def assign_group(filename: str, group_patterns: dict, default_group: str = "all"):
    """
    Assign a group label from filename using substring matching.
    If group_patterns is empty, everything goes to default_group.
    """
    if not group_patterns:
        return default_group

    for group_name, pattern in group_patterns.items():
        if pattern in filename:
            return group_name

    return default_group


def get_common_x(curves):
    """
    Build a common x-axis over the overlapping range of all curves.
    """
    mins = [x.min() for x, *_ in curves]
    maxs = [x.max() for x, *_ in curves]

    x_min = max(mins)
    x_max = min(maxs)

    if x_min >= x_max:
        raise ValueError("No overlapping x-range across curves in this group.")

    x_ref = curves[0][0]
    x_ref = x_ref[(x_ref >= x_min) & (x_ref <= x_max)]

    if len(x_ref) == 0:
        raise ValueError("No x values left after restricting to overlap range.")

    return x_ref


def plot_group_mean_sem(ax, curves, color, label):
    """
    Plot group mean ± SEM.
    curves is a list of (x, y, y_low, y_high).
    """
    x_ref = get_common_x(curves)

    y_interp = []
    for x, y, *_ in curves:
        y_i = np.interp(x_ref, x, y)
        y_interp.append(y_i)

    y_interp = np.vstack(y_interp)

    mean_y = np.mean(y_interp, axis=0)
    sem_y = (
        np.std(y_interp, axis=0, ddof=1) / np.sqrt(y_interp.shape[0])
        if y_interp.shape[0] > 1
        else np.zeros_like(mean_y)
    )

    ax.plot(x_ref, mean_y, color=color, linewidth=2.5, label=label)
    ax.fill_between(x_ref, mean_y - sem_y, mean_y + sem_y, color=color, alpha=0.25)


def plot_group_individual(
    ax,
    curves,
    color,
    label,
    shade_bands: bool = True,
    line_alpha: float = 0.35,
    band_alpha: float = 0.12,
):
    """
    Plot each curve individually.
    If y_low/y_high exist, shade them per curve.
    """
    for i, (x, y, y_low, y_high) in enumerate(curves):
        line_label = label if i == 0 else None
        ax.plot(x, y, color=color, linewidth=1.2, alpha=line_alpha, label=line_label)

        if shade_bands and y_low is not None and y_high is not None:
            ax.fill_between(x, y_low, y_high, color=color, alpha=band_alpha, linewidth=0)


def main(input_dir: Path, output_file: Path, recursive: bool = False):
    if recursive:
        files = sorted(
            f for f in input_dir.rglob("*.hdf5")
            if f.is_file() and not f.name.startswith("._")
        )
    else:
        files = sorted(
            f for f in input_dir.glob("*.hdf5")
            if f.is_file() and not f.name.startswith("._")
        )

    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {input_dir}")

    show_legend = not args.no_legend
    group_data = {}

    for f in files:
        try:
            x, y, y_low, y_high = load_profile(
                f,
                key=HDF5_KEY,
                x_col=X_COL,
                y_col=Y_COL,
                low_col=LOW_COL,
                high_col=HIGH_COL,
            )
            group = assign_group(f.name, GROUP_PATTERNS, default_group=DEFAULT_GROUP)
            group_data.setdefault(group, []).append((x, y, y_low, y_high))
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if not group_data:
        raise RuntimeError("No valid curves were loaded.")

    fig, ax = plt.subplots(figsize=(FIGSIZE))

    for group, curves in group_data.items():
        if len(curves) == 0:
            continue

        color = GROUP_COLORS.get(group, "gray")

        if PLOT_MODE == "mean_sem":
            plot_group_mean_sem(
                ax,
                curves,
                color=color,
                label=f"{group} (n={len(curves)})",
            )
        elif PLOT_MODE == "individual":
            plot_group_individual(
                ax,
                curves,
                color=color,
                label=f"{group} (n={len(curves)})",
                shade_bands=SHOW_INDIVIDUAL_BANDS,
            )
        else:
            raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE}")

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    #ax.set_title(PLOT_TITLE)

    if show_legend:
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close(fig)

    print(f"Saved plot to {output_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot radial autocorrelation profiles from .hdf5 files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .hdf5 files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("radial_autocorrelation_overlay.pdf"),
        help="Output image file",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .hdf5 files recursively in subdirectories",
    )
    parser.add_argument(
        "--mode",
        choices=["mean_sem", "individual"],
        default="mean_sem",
        help="Plot pooled group mean ± SEM or individual curves",
    )
    parser.add_argument(
        "--no-individual-bands",
        action="store_true",
        help="In individual mode, do not shade per-file uncertainty bands",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legend on the plot",
    )

    args = parser.parse_args()

    # =========================
    # USER-DEFINED VARIABLES
    # =========================

    HDF5_KEY = "radial_profile"

    # Column names in the HDF5 dataframe / group
    X_COL = "bin_center"
    Y_COL = "mean"

    # Optional per-file band columns, if present
    LOW_COL = "std"      # set to None if you do not want bands
    HIGH_COL = None      # set to None if you do not want bands

    # Axis labels
    X_LABEL = "Distance (cells)"
    Y_LABEL = "Cross-correlation"
#    PLOT_TITLE = "Radial autocorrelation profiles"

    # Grouping
    GROUP_PATTERNS = {
        "0": "xI_0",
        "25": "xI_25",
        "50": "xI_50"
#        "independent": "ind",
#        "co-induced": "coupled-coinitiation",
#        "cross-inhibited": "xi",
    }

    # Color per group
    GROUP_COLORS = {
        "0":  "#FEE0D2",  # light red
        "25": "#FC9272",  # medium red
        "50": "#CB181D",  # dark red
#        "independent": "gray",
#        "co-induced": "blue",
#        "cross-inhibited": "red",
    }

    DEFAULT_GROUP = "all"

    # Plot mode settings
    PLOT_MODE = args.mode
    SHOW_INDIVIDUAL_BANDS = not args.no_individual_bands

    # =========================

    main(args.input_dir, args.output, recursive=args.recursive)
