#!/usr/bin/env python3

from __future__ import annotations

from math import floor, ceil
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COLOR_CYCLE = ["gray"]

def load_npz_image(npz_file: Path, key: str | None = None) -> np.ndarray:
    """
    Load a 2D array from an .npz file.
    If key is not provided, the first 2D array found is used.
    """
    with np.load(npz_file) as data:
        if key is not None:
            if key not in data.files:
                raise KeyError(
                    f"{npz_file.name}: key '{key}' not found. Available keys: {data.files}"
                )
            im = np.asarray(data[key])
        else:
            im = None
            for k in data.files:
                candidate = np.squeeze(np.asarray(data[k]))
                if candidate.ndim == 2:
                    im = candidate
                    break
            if im is None:
                raise ValueError(
                    f"{npz_file.name}: no 2D array found. Available keys: {data.files}"
                )

    im = np.squeeze(np.asarray(im))
    if im.ndim != 2:
        raise ValueError(f"{npz_file.name}: expected a 2D array, got shape {im.shape}")

    return im.astype(float)


def radial_profile(
    X: np.ndarray,
    nbins=None,
    bin_edges=None,
    coords=None,
    center=None,
    max_dist=None,
):
    """Computes the radial profile of an image in pixel/cell units."""
    m, n = X.shape
    xx, yy = np.mgrid[:m, :n]
    xx = xx.flatten()
    yy = yy.flatten()

    if max_dist is None:
        max_dist = min(m, n) / 2

    if center is not None:
        x_c, y_c = center
    else:
        x_c = m / 2
        y_c = n / 2

    if coords is None:
        xcoords = xx - x_c
        ycoords = yy - y_c
    else:
        xcoords, ycoords = coords
        xcoords = xcoords - x_c
        ycoords = ycoords - y_c

    D = np.sqrt(xcoords**2 + ycoords**2)

    if bin_edges is not None:
        nbins = len(bin_edges) - 1
    else:
        if nbins is None:
            nbins = floor(np.sqrt(X.size))
        binsize = max_dist / nbins
        bin_edges = binsize * np.arange(nbins + 1)

    n_obs = np.zeros(nbins, dtype=int)
    profile = np.zeros(nbins, dtype=float)

    for i in range(nbins):
        bin_low = bin_edges[i]
        bin_high = bin_edges[i + 1]
        mask = (D > bin_low) & (D <= bin_high)
        n = mask.sum()
        n_obs[i] = n
        if n > 0:
            profile[i] = X[xx[mask], yy[mask]].mean()
        else:
            profile[i] = np.nan

    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    return bin_centers, n_obs, profile


def autocorrelation_2d(X: np.ndarray) -> np.ndarray:
    """Compute the 2D autocorrelation of an image X."""
    F = fft.fft2(X - X.mean())
    S = np.abs(F) ** 2
    acf = fft.ifft2(S).real
    acf = fft.fftshift(acf)
    acf /= acf.max()
    return acf


def analyze_image_autocorrelation(
    im: np.ndarray,
    nbins: int = 100,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Computes the autocorrelation of an image and its radial profile.

    Distances are in cell units.
    """
    A = autocorrelation_2d(im)

    # Automatic maximum radius in cell units.
    max_corr_dist = min(im.shape) / 2

    bin_centers, n_obs, Ar = radial_profile(
        A,
        nbins=nbins,
        max_dist=max_corr_dist,
    )

    df = pd.DataFrame(
        {
            "bin_centers": bin_centers,
            "n_observations": n_obs,
            "radial_autocorrelation": Ar,
        }
    )
    return A, df


def plot_image_autocorrelation_and_radial_profile(
    X: np.ndarray,
    A: np.ndarray,
    radial_distance: np.ndarray,
    A_radial: np.ndarray,
    title: str = "Preprocessed Image",
    fig=None,
    save: bool = False,
    fname: Path | str | None = None,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Makes a 3-column plot of the image, its autocorrelation, and the average radial autocorrelation."""
    if fig is None:
        fig = plt.figure(figsize=(15, 5))

    m, n = X.shape
    max_corr_dist = min(m, n) / 2
    w = ceil(max_corr_dist)

    m_center = A.shape[0] // 2
    n_center = A.shape[1] // 2
    A_trunc = A[m_center - w : m_center + w, n_center - w : n_center + w]

    # Image
    plt.subplot(1, 3, 1)
    plt.imshow(X, cmap="gray", origin="upper")
    plt.title(title)
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    plt.axis("equal")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Autocorrelation
    plt.subplot(1, 3, 2)
    ax1_pos = plt.imshow(
        A_trunc,
        cmap=plt.cm.RdBu_r,  # type: ignore
        vmin=-1,
        vmax=1,
        extent=(-max_corr_dist, max_corr_dist, -max_corr_dist, max_corr_dist),
        origin="upper",
    )
    plt.title("Autocorrelation")
    plt.xlabel("X displacement (cells)")
    plt.ylabel("Y displacement (cells)")
    plt.xticks(ticks=np.linspace(-max_corr_dist, max_corr_dist, 5))
    plt.yticks(ticks=np.linspace(-max_corr_dist, max_corr_dist, 5))
    fig.colorbar(ax1_pos, shrink=0.7)

    # Radial profile
    plt.subplot(1, 3, 3)
    plt.plot(radial_distance, A_radial, color="#ff7f0e", lw=1.5)
    plt.title("Average radial autocorrelation")
    plt.xlabel("Distance (cells)")
    plt.ylabel("Avg. autocorrelation")
    plt.ylim(min(np.nanmin(A_radial), 0), None)

    plt.tight_layout()

    if save:
        if fname is None:
            raise ValueError("fname must be provided when save=True")
        plt.savefig(fname, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {Path(fname).resolve().absolute()}")

def plot_overlay_radial_autocorrelation(results, out_pdf: Path, dpi: int = 300) -> None:
    """
    Plot all radial autocorrelation curves on one figure, colored by sample name.

    results: list of dicts with keys:
        - sample_name
        - bin_centers
        - radial_autocorrelation
    """
    if not results:
        return

    fig, ax = plt.subplots(figsize=(4, 3))

    for i, res in enumerate(results):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]

        x = res["bin_centers"]
        y = res["radial_autocorrelation"]

        ax.plot(x, y, color=color, lw=1.5)

#    ax.set_title("Overlay of radial autocorrelation curves")
    ax.set_xlabel("Distance (cells)")
    ax.set_ylabel("Avg. autocorrelation")

    ax.set_ylim(
        bottom=min(0, np.nanmin([np.nanmin(r["radial_autocorrelation"]) for r in results]))
    )

    ax.grid(True, alpha=0.25)

    # ❌ REMOVE LEGEND
    #ax.legend(...)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Overlay plot saved to: {out_pdf.resolve().absolute()}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute radially averaged autocorrelation for all .npz files in a directory."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .npz files")
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Array key inside each .npz file. If omitted, the first 2D array is used.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="Number of radial bins.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input_dir>/autocorr_analysis",
    )
    parser.add_argument(
        "--save_acf",
        action="store_true",
        help="Save the full 2D autocorrelation arrays as .npy files.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    out_dir = args.output_dir or (input_dir / "autocorr_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    acf_dir = out_dir / "acf"
    if args.save_acf:
        acf_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(
        p for p in input_dir.glob("*.npz")
        if p.is_file() and not p.name.startswith("._")
    )

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    combined = []
    overlay_results = []

    for npz_file in npz_files:
        print(f"Processing {npz_file.name} ...")
        im = load_npz_image(npz_file, key=args.key)
        A, df = analyze_image_autocorrelation(im=im, nbins=args.nbins)

        df.insert(0, "filename", npz_file.name)

        overlay_results.append(
            {
                "sample_name": npz_file.stem,
                "bin_centers": df["bin_centers"].to_numpy(),
                "radial_autocorrelation": df["radial_autocorrelation"].to_numpy(),
            }
        )

        out_csv = csv_dir / f"{npz_file.stem}_radial_autocorr.csv"
        df.to_csv(out_csv, index=False)

        out_pdf = plot_dir / f"{npz_file.stem}_radial_autocorrelation.pdf"
        plot_image_autocorrelation_and_radial_profile(
            X=im,
            A=A,
            radial_distance=df["bin_centers"].to_numpy(),
            A_radial=df["radial_autocorrelation"].to_numpy(),
            title=npz_file.stem,
            save=True,
            fname=out_pdf,
            fmt="pdf",
            dpi=300,
        )
        plt.close("all")

        if args.save_acf:
            out_acf = acf_dir / f"{npz_file.stem}_acf.npy"
            np.save(out_acf, A)

        combined.append(df)

    combined_df = pd.concat(combined, ignore_index=True)
    combined_csv = out_dir / "combined_radial_autocorr.csv"
    combined_df.to_csv(combined_csv, index=False)

    overlay_png = out_dir / "overlay_radial_autocorrelation.pdf"
    plot_overlay_radial_autocorrelation(overlay_results, overlay_png, dpi=300)

    print(f"Saved per-file CSVs to: {csv_dir.resolve()}")
    print(f"Saved plots to: {plot_dir.resolve()}")
    print(f"Saved combined CSV to: {combined_csv.resolve()}")
    if args.save_acf:
        print(f"Saved autocorrelation arrays to: {acf_dir.resolve()}")


if __name__ == "__main__":
    main()
