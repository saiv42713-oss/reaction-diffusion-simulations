# fft_analysis_2D.py

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable text)
plt.rcParams['ps.fonttype'] = 42

def mm_to_inches(mm):
    return mm / 25.4

def load_field(npz_path, field_name="A_final"):
    data = np.load(npz_path, allow_pickle=True)

    # Try a few common keys if the requested one is not present
    candidates = [
        field_name,
        "A_final",
        "activator_final",
        "R_final",
        "inhibitor_final",
    ]

    for key in candidates:
        if key in data.files:
            return np.asarray(data[key]), key

    raise KeyError(f"No matching field found in {npz_path}. Keys present: {data.files}")


def radial_power_spectrum(field, dx=1.0, n_bins=None):
    """
    Compute radial average of 2D FFT power spectrum.

    Returns:
        wavelength (cells), power
    """
    field = np.asarray(field, dtype=float)
    field = field - np.mean(field)  # remove DC component

    ny, nx = field.shape

    F = np.fft.fft2(field)
    power2d = np.abs(F) ** 2 / field.size

    fx = np.fft.fftfreq(nx, d=dx)  # cycles per cell
    fy = np.fft.fftfreq(ny, d=dx)

    kx, ky = np.meshgrid(fx, fy)
    kr = np.sqrt(kx**2 + ky**2)

    kr_flat = kr.ravel()
    p_flat = power2d.ravel()

    if n_bins is None:
        n_bins = min(ny, nx) // 2

    bins = np.linspace(0.0, kr_flat.max(), n_bins + 1)
    bin_idx = np.digitize(kr_flat, bins) - 1

    radial_power = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i, p in zip(bin_idx, p_flat):
        if 0 <= i < n_bins:
            radial_power[i] += p
            counts[i] += 1

    valid = counts > 0
    radial_power[valid] /= counts[valid]

    k_centers = 0.5 * (bins[:-1] + bins[1:])

    # Convert frequency to wavelength in cell units
    # wavelength = 1 / frequency
    valid = k_centers > 0
    wavelength = 1.0 / k_centers[valid]
    power = radial_power[valid]

    # Sort by wavelength increasing for nicer plotting
    order = np.argsort(wavelength)
    return wavelength[order], power[order]


def plot_spectrum(wavelength, power, title, outfile, normalize=False, color="blue"):
    y = np.asarray(power, dtype=float)

    if normalize:
        m = np.max(y)
        if m > 0:
            y = y / m

    fig, ax = plt.subplots(
        figsize=(mm_to_inches(40), mm_to_inches(30))
    )
    ax.plot(wavelength, y, lw=2, color=color)
    ax.set_xlabel("Wavelength (cells)")
    ax.set_ylabel("Normalized power" if normalize else "Power")
#    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile)  # no dpi needed for vector formats
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, required=True, help="Folder containing .npz files")
    ap.add_argument("--field", type=str, default="A_final",
                    help="Field to analyze: A_final, activator_final, R_final, inhibitor_final")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder for plots")
    ap.add_argument("--dx", type=float, default=1.0, help="Grid spacing in cell units")
    args = ap.parse_args()

    color_cycle = ["grey"]  # matplotlib default blue/orange

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    outdir = Path(args.outdir) if args.outdir else folder / "fft_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(folder.glob("*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    overlay = []

    for i, npz_file in enumerate(npz_files):
        field, used_key = load_field(npz_file, field_name=args.field)
        wavelength, power = radial_power_spectrum(field, dx=args.dx)

        stem = npz_file.stem

        # Use vector format (PDF here; SVG also fine)
        single_out = outdir / f"{stem}_fft.pdf"

        color = color_cycle[i % len(color_cycle)]

        plot_spectrum(
            wavelength,
            power,
            title=f"{stem} ({used_key})",
            outfile=single_out,
            normalize=False,
            color=color,
        )

        overlay.append((stem, wavelength, power, color))

    fig, ax = plt.subplots(figsize=(3, 2.5))
    for stem, wavelength, power, color in overlay:
        y = power / np.max(power) if np.max(power) > 0 else power
        ax.plot(wavelength, y, lw=1.5, color=color)

    ax.set_xlabel("Wavelength (cells)")
    ax.set_ylabel("Normalized power")
#    ax.set_title("Overlay FFT spectra")
    ax.grid(True, alpha=0.3)

    # ❌ Remove legend entirely
    # ax.legend(...)

    fig.tight_layout()
    fig.savefig(outdir / "overlay_normalized.pdf")
    plt.close(fig)

    ax.set_xlabel("Wavelength (cells)")
    ax.set_ylabel("Normalized power")
#    ax.set_title("Overlay FFT spectra")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "overlay_normalized.pdf", dpi=300)
    plt.close(fig)

    print(f"Saved individual spectra and overlay plot to: {outdir}")


if __name__ == "__main__":
    main()
