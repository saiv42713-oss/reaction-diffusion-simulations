from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================
# CORE METRICS
# =========================

def dice_sorenson_coeff(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Dice-Sorenson coefficient for binary masks."""
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)

    n1 = m1.sum()
    n2 = m2.sum()
    if n1 == 0 and n2 == 0:
        return 1.0
    if n1 == 0 or n2 == 0:
        return 0.0

    overlap = np.logical_and(m1, m2).sum()
    return float(2.0 * overlap / (n1 + n2))


def manders_m1_m2_binary(mask1: np.ndarray, mask2: np.ndarray) -> tuple[float, float]:
    """
    Manders-like overlap fractions for binary masks.
    For binary masks, this reduces to:
      M1 = overlap / sum(mask1)
      M2 = overlap / sum(mask2)
    """
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)

    n1 = m1.sum()
    n2 = m2.sum()
    overlap = np.logical_and(m1, m2).sum()

    m1_coeff = float(overlap / n1) if n1 > 0 else np.nan
    m2_coeff = float(overlap / n2) if n2 > 0 else np.nan
    return m1_coeff, m2_coeff


# =========================
# OTSU THRESHOLDING
# =========================

def otsu_threshold(arr: np.ndarray, nbins: int = 256) -> float:
    """
    Compute Otsu threshold using only NumPy.
    Works on any numeric array after flattening and removing non-finite values.
    """
    values = np.asarray(arr).ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Cannot compute Otsu threshold on an empty/non-finite array.")

    vmin = float(values.min())
    vmax = float(values.max())

    if vmin == vmax:
        return vmin

    hist, bin_edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(float)

    prob = hist / hist.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros_like(denom)
    valid = denom > 0
    sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

    idx = int(np.nanargmax(sigma_b2))
    return float(bin_centers[idx])


def binarize_with_otsu(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Binarize an array using Otsu thresholding.
    Returns (binary_mask, threshold).
    """
    thr = otsu_threshold(arr)

    arr = np.asarray(arr)
    if np.all(np.isfinite(arr)) and arr.size > 0 and float(arr.min()) == float(arr.max()):
        if float(arr.flat[0]) > 0:
            return np.ones_like(arr, dtype=bool), thr
        else:
            return np.zeros_like(arr, dtype=bool), thr

    return (arr > thr), thr


# =========================
# OPTIONAL PLOTTING
# =========================

def plot_mask_overlap(
    mask1: np.ndarray,
    mask2: np.ndarray,
    figsize: tuple[int, int] = (10, 10),
    output_path: Optional[Path | str] = None,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Plot overlap between two binary masks.
    mask1 = red, mask2 = green, overlap = yellow.
    Axes are in array units.
    """
    import matplotlib.pyplot as plt

    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)

    overlap = np.zeros((*mask1.shape, 3), dtype=np.uint8)
    overlap[mask1] = [255, 0, 0]
    overlap[mask2] = [0, 255, 0]
    overlap[np.logical_and(mask1, mask2)] = [255, 255, 0]

    plt.figure(figsize=figsize)
    plt.imshow(overlap)
    plt.axis("equal")
    plt.xlabel("X (units)")
    plt.ylabel("Y (units)")

    if output_path is not None:
        output_path = Path(output_path)
        print(f"Saving overlap plot to: {output_path.resolve().absolute()}")
        plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =========================
# FILE-LEVEL ANALYSIS
# =========================

@dataclass
class NPZOverlapResult:
    filename: str
    var1_key: str
    var2_key: str
    var1_threshold: float
    var2_threshold: float
    mask1_mean: float
    mask2_mean: float
    M1_coeff: float
    M2_coeff: float
    Dice_coeff: float
    expected_M1: float
    expected_M2: float
    expected_Dice: float
    M1_excess: float
    M2_excess: float
    Dice_excess: float
    M1_excess_ratio: float
    M2_excess_ratio: float
    Dice_excess_ratio: float


def analyze_npz_overlap(
    npz_file: Path | str,
    var1_key: str = "A1",
    var2_key: str = "A2",
    plot_dir: Optional[Path | str] = None,
    fmt: str = "png",
    dpi: int = 300,
) -> tuple[NPZOverlapResult, np.ndarray, np.ndarray]:
    """
    Load two arrays from a .npz file, threshold them with Otsu, and compute overlap metrics.
    """
    npz_file = Path(npz_file)

    with np.load(npz_file) as data:
        if var1_key not in data:
            raise KeyError(f"{var1_key!r} not found in {npz_file.name}. Available keys: {list(data.keys())}")
        if var2_key not in data:
            raise KeyError(f"{var2_key!r} not found in {npz_file.name}. Available keys: {list(data.keys())}")

        arr1 = np.asarray(data[var1_key])
        arr2 = np.asarray(data[var2_key])

    mask1, thr1 = binarize_with_otsu(arr1)
    mask2, thr2 = binarize_with_otsu(arr2)

    m1, m2 = manders_m1_m2_binary(mask1, mask2)
    dice = dice_sorenson_coeff(mask1, mask2)

    mask1_mean = float(mask1.mean())
    mask2_mean = float(mask2.mean())

    expected_m1 = mask2_mean
    expected_m2 = mask1_mean
    expected_dice = (
        2 * mask1_mean * mask2_mean / (mask1_mean + mask2_mean)
        if (mask1_mean + mask2_mean) > 0
        else np.nan
    )

    m1_excess = m1 - expected_m1
    m2_excess = m2 - expected_m2
    dice_excess = dice - expected_dice

    m1_excess_ratio = m1 / expected_m1 if np.isfinite(expected_m1) and expected_m1 != 0 else np.nan
    m2_excess_ratio = m2 / expected_m2 if np.isfinite(expected_m2) and expected_m2 != 0 else np.nan
    dice_excess_ratio = dice / expected_dice if np.isfinite(expected_dice) and expected_dice != 0 else np.nan

    result = NPZOverlapResult(
        filename=npz_file.name,
        var1_key=var1_key,
        var2_key=var2_key,
        var1_threshold=float(thr1),
        var2_threshold=float(thr2),
        mask1_mean=mask1_mean,
        mask2_mean=mask2_mean,
        M1_coeff=float(m1),
        M2_coeff=float(m2),
        Dice_coeff=float(dice),
        expected_M1=float(expected_m1),
        expected_M2=float(expected_m2),
        expected_Dice=float(expected_dice),
        M1_excess=float(m1_excess),
        M2_excess=float(m2_excess),
        Dice_excess=float(dice_excess),
        M1_excess_ratio=float(m1_excess_ratio),
        M2_excess_ratio=float(m2_excess_ratio),
        Dice_excess_ratio=float(dice_excess_ratio),
    )

    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        plot_path = plot_dir / f"{today}_{npz_file.stem}_overlap.{fmt}"
        plot_mask_overlap(
            mask1,
            mask2,
            output_path=plot_path,
            fmt=fmt,
            dpi=dpi,
        )

    return result, mask1, mask2


# =========================
# BATCH RUN ON A FOLDER
# =========================

def main(
    input_dir: Path | str,
    analysis_output_dir: Path | str,
    var1_key: str = "A1",
    var2_key: str = "A2",
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

    results = []
    for npz_file in npz_files:
        result, _, _ = analyze_npz_overlap(
            npz_file=npz_file,
            var1_key=var1_key,
            var2_key=var2_key,
            plot_dir=plot_dir,
            fmt=fmt,
            dpi=dpi,
        )
        results.append(result)

        print(f"Overlap analysis for {npz_file.name}:")
        print(f" -- Otsu thresholds: {result.var1_threshold:.6g}, {result.var2_threshold:.6g}")
        print(f" -- M1: {result.M1_coeff:.3f} (expected: {result.expected_M1:.3f})")
        print(f" -- M2: {result.M2_coeff:.3f} (expected: {result.expected_M2:.3f})")
        print(f" -- D-S: {result.Dice_coeff:.3f} (expected: {result.expected_Dice:.3f})")
        print()

    df = pd.DataFrame([r.__dict__ for r in results])

    today = date.today().isoformat()
    output_path = analysis_output_dir / f"{today}_npz_overlap_analysis_results.csv"
    print(f"Saving overlap analysis results to: {output_path.resolve().absolute()}")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Dice overlap from .npz files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing .npz files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where the CSV will be saved")
    parser.add_argument("--var1-key", type=str, default="a1", help="Name of first array inside each .npz")
    parser.add_argument("--var2-key", type=str, default="a2", help="Name of second array inside each .npz")
    parser.add_argument("--plot-output-dir", type=Path, default=None, help="Optional folder for overlap plots")
    parser.add_argument("--fmt", type=str, default="png")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        analysis_output_dir=args.output_dir,
        var1_key=args.var1_key,
        var2_key=args.var2_key,
        plot_output_dir=args.plot_output_dir,
        fmt=args.fmt,
        dpi=args.dpi,
    )
