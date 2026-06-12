from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CROSS-CORRELATION CORE
# =========================

def normalized_cross_correlation_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a normalized 2D cross-correlation using FFT.

    The arrays are mean-centered first, so the result is closer to a Pearson-like
    correlation map. The output is shifted so zero lag is at the center.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Array shapes do not match: {a.shape} vs {b.shape}")

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")

    a0 = np.nan_to_num(a - np.nanmean(a), nan=0.0)
    b0 = np.nan_to_num(b - np.nanmean(b), nan=0.0)

    denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2))
    if denom == 0:
        return np.zeros_like(a0, dtype=float)

    S = np.fft.ifft2(np.fft.fft2(a0) * np.conj(np.fft.fft2(b0))).real
    S = np.fft.fftshift(S) / denom
    return S


def radial_average_2d(
    S: np.ndarray,
    nbins: int = 100,
    max_corr_dist: Optional[float] = None,
) -> pd.DataFrame:
    """
    Radially average a 2D correlation map around its center.

    Returns a dataframe with:
      - bin_left
      - bin_right
      - bin_center
      - mean
      - std
      - n
    """
    S = np.asarray(S, dtype=float)
    ny, nx = S.shape

    yy, xx = np.indices((ny, nx))
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if max_corr_dist is None:
        max_corr_dist = float(rr.max())

    bins = np.linspace(0, max_corr_dist, nbins + 1)
    flat_r = rr.ravel()
    flat_s = S.ravel()

    bin_ids = np.digitize(flat_r, bins) - 1

    rows = []
    for i in range(nbins):
        vals = flat_s[bin_ids == i]
        if vals.size == 0:
            rows.append(
                {
                    "bin_left": bins[i],
                    "bin_right": bins[i + 1],
                    "bin_center": 0.5 * (bins[i] + bins[i + 1]),
                    "mean": np.nan,
                    "std": np.nan,
                    "n": 0,
                }
            )
        else:
            rows.append(
                {
                    "bin_left": bins[i],
                    "bin_right": bins[i + 1],
                    "bin_center": 0.5 * (bins[i] + bins[i + 1]),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "n": int(vals.size),
                }
            )

    return pd.DataFrame(rows)


# =========================
# PLOTTING
# =========================

def plot_radial_cross_correlation(
    radial_df: pd.DataFrame,
    output_path: Optional[Path | str] = None,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Plot the radially averaged cross-correlation with an optional std band.
    """
    plt.figure(figsize=(7, 5))

    x = radial_df["bin_center"].to_numpy()
    y = radial_df["mean"].to_numpy()
    ystd = radial_df["std"].to_numpy()

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    ystd = ystd[ok]

    plt.plot(x, y, linewidth=2)

    if np.any(np.isfinite(ystd)):
        lower = y - ystd
        upper = y + ystd
        plt.fill_between(x, lower, upper, alpha=0.2)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Radial distance (units)")
    plt.ylabel("Cross-correlation")
    plt.title("Radially averaged cross-correlation")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        print(f"Saving radial cross-correlation plot to: {output_path.resolve().absolute()}")
        plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =========================
# FILE-LEVEL ANALYSIS
# =========================

@dataclass
class CrossCorrelationResult:
    filename: str
    var1_key: str
    var2_key: str
    shape_y: int
    shape_x: int
    peak_corr: float
    peak_radius_units: float


def analyze_npz_cross_correlation(
    npz_file: Path | str,
    var1_key: str = "A1",
    var2_key: str = "A2",
    nbins: int = 100,
    max_corr_dist: Optional[float] = None,
) -> tuple[CrossCorrelationResult, np.ndarray, pd.DataFrame]:
    """
    Load two arrays from a .npz file, compute the 2D normalized cross-correlation,
    and radially average it.
    """
    npz_file = Path(npz_file)

    with np.load(npz_file) as data:
        if var1_key not in data:
            raise KeyError(
                f"{var1_key!r} not found in {npz_file.name}. Available keys: {list(data.keys())}"
            )
        if var2_key not in data:
            raise KeyError(
                f"{var2_key!r} not found in {npz_file.name}. Available keys: {list(data.keys())}"
            )

        arr1 = np.asarray(data[var1_key])
        arr2 = np.asarray(data[var2_key])

    S = normalized_cross_correlation_2d(arr1, arr2)
    radial_df = radial_average_2d(S, nbins=nbins, max_corr_dist=max_corr_dist)

    peak_idx = int(np.nanargmax(radial_df["mean"].to_numpy()))
    peak_corr = float(radial_df.loc[peak_idx, "mean"])
    peak_radius = float(radial_df.loc[peak_idx, "bin_center"])

    result = CrossCorrelationResult(
        filename=npz_file.name,
        var1_key=var1_key,
        var2_key=var2_key,
        shape_y=int(arr1.shape[0]),
        shape_x=int(arr1.shape[1]),
        peak_corr=peak_corr,
        peak_radius_units=peak_radius,
    )

    return result, S, radial_df


# =========================
# BATCH RUN ON A FOLDER
# =========================

def main(
    input_dir: Path | str,
    analysis_output_dir: Path | str,
    var1_key: str = "A1",
    var2_key: str = "A2",
    nbins: int = 100,
    max_corr_dist: Optional[float] = None,
    plot_output_dir: Optional[Path | str] = None,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    input_dir = Path(input_dir)
    analysis_output_dir = Path(analysis_output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    if not analysis_output_dir.exists():
        raise ValueError(f"Output directory does not exist: {analysis_output_dir}")
    if not analysis_output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {analysis_output_dir}")

    plot_dir = None
    if plot_output_dir is not None:
        plot_dir = Path(plot_output_dir)
        if not plot_dir.exists():
            raise ValueError(f"Plot directory does not exist: {plot_dir}")
        if not plot_dir.is_dir():
            raise ValueError(f"Plot path is not a directory: {plot_dir}")

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found in: {input_dir}")

    summary_rows = []

    for npz_file in npz_files:
        result, S, radial_df = analyze_npz_cross_correlation(
            npz_file=npz_file,
            var1_key=var1_key,
            var2_key=var2_key,
            nbins=nbins,
            max_corr_dist=max_corr_dist,
        )

        summary_rows.append(result.__dict__)

        fname = f"{npz_file.stem}_cross_correlation"
        out_file = analysis_output_dir / f"{fname}.hdf5"

        with h5py.File(out_file, "w") as h:
            h.create_dataset("S", data=S)
            radial_group = h.create_group("radial_profile")
            for col in radial_df.columns:
                radial_group.create_dataset(col, data=radial_df[col].to_numpy())

            h.attrs["filename"] = result.filename
            h.attrs["var1_key"] = result.var1_key
            h.attrs["var2_key"] = result.var2_key
            h.attrs["shape_y"] = result.shape_y
            h.attrs["shape_x"] = result.shape_x
            h.attrs["peak_corr"] = result.peak_corr
            h.attrs["peak_radius_units"] = result.peak_radius_units

        print(f"Saved cross-correlation data to: {out_file.resolve().absolute()}")

        if plot_dir is not None:
            today = date.today().isoformat()
            plot_path = plot_dir / f"{today}_{fname}_radial_profile.{fmt}"
            plot_radial_cross_correlation(
                radial_df,
                output_path=plot_path,
                fmt=fmt,
                dpi=dpi,
            )

        print(
            f"{npz_file.name}: peak radial cross-correlation = "
            f"{result.peak_corr:.4f} at radius {result.peak_radius_units:.2f} units"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = analysis_output_dir / f"{date.today().isoformat()}_cross_correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path.resolve().absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute radially averaged cross-correlation from .npz files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing .npz files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where data files will be saved")
    parser.add_argument("--var1-key", type=str, default="a1", help="Name of first array inside each .npz")
    parser.add_argument("--var2-key", type=str, default="a2", help="Name of second array inside each .npz")
    parser.add_argument("--nbins", type=int, default=100, help="Number of radial bins")
    parser.add_argument(
        "--max-corr-dist",
        type=float,
        default=None,
        help="Maximum radial distance to include, in array units",
    )
    parser.add_argument("--plot-output-dir", type=Path, default=None, help="Optional folder for plots")
    parser.add_argument("--fmt", type=str, default="png")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        analysis_output_dir=args.output_dir,
        var1_key=args.var1_key,
        var2_key=args.var2_key,
        nbins=args.nbins,
        max_corr_dist=args.max_corr_dist,
        plot_output_dir=args.plot_output_dir,
        fmt=args.fmt,
        dpi=args.dpi,
    )
