from pathlib import Path
import pandas as pd
import numpy as np

# Import existing analysis functions from Ben's repo
from fft_analysis_2D import load_field, radial_power_spectrum
from radial_autocor_npz import analyze_image_autocorrelation
from npz_feature_distribution import otsu_threshold, component_mask, approx_diameter_hexagons

script_path = Path(__file__).resolve()
run_dir = script_path.parents[1] / "2D_batch" / "runs" / "template-single-circuit"  # Avoid renaming this

# def temporal_metrics(npz, field="[history]")


# Just base summarize function for now
def base_summarize(npz, field="A_final", threshold="auto", layout="even-r", min_size=2):
    data, field_name = load_field(npz, field)
    threshold_val = otsu_threshold(data) if threshold == "auto" else float(threshold)
    active_mask = np.isfinite(data) & (data > threshold_val)

    components = [c for c in component_mask(active_mask, layout) if len(c) >= min_size]
    component_sizes = np.array([len(c) for c in components], dtype=float)
    diameters = np.array(
        [approx_diameter_hexagons(int(s)) for s in component_sizes], dtype=float
    )

    # FFT power spectrum
    wavelengths, spectrum = radial_power_spectrum(data)
    if len(spectrum) > 0 and np.any(np.isfinite(spectrum)):
        peak_wavelength = float(wavelengths[np.argmax(spectrum)])
        positive = spectrum[spectrum > 0]
        peak_sharpness = (
            float(np.nanmax(spectrum) / np.nanmedian(positive))
            if len(positive) > 0
            else np.nan
        )
    else:
        peak_wavelength = np.nan
        peak_sharpness = np.nan

    # Autocorrelation
    _, autocorr = analyze_image_autocorrelation(data)
    corr = autocorr["radial_autocorrelation"].to_numpy()
    distance = autocorr["bin_centers"].to_numpy()

    zero_idx = np.where(corr <= 0)[0]
    zero_crossing = (
        float(distance[zero_idx[0]])
        if len(zero_idx) > 0
        else np.nan
    )

    decay_idx = np.where(corr <= 1 / np.e)[0]
    corr_length = (
        float(distance[decay_idx[0]])
        if len(decay_idx) > 0
        else np.nan
    )

    if len(component_sizes) > 0:
        mean_component_size = float(component_sizes.mean())
        max_component_size = int(component_sizes.max())
        size_cv = (
            float(component_sizes.std() / component_sizes.mean())
            if component_sizes.mean() > 0
            else np.nan
        )
        mean_diameter = float(diameters.mean())
    else:
        mean_component_size = 0.0
        max_component_size = 0
        size_cv = np.nan
        mean_diameter = 0.0

    return {
        "file": npz.name,
        "field": field_name,
        "threshold": threshold_val,
        "activated_hexes": int(active_mask.sum()),
        "activated_fraction": float(active_mask.mean()),
        "n_components": int(len(components)),
        "mean_component_size": mean_component_size,
        "max_component_size": max_component_size,
        "component_size_cv": size_cv,
        "mean_diameter_hexes": mean_diameter,
        "peak_fft_wavelength": peak_wavelength,
        "fft_peak_sharpness": peak_sharpness,
        "autocorr_first_nonpositive_dist": zero_crossing,
        "autocorr_length_1e": corr_length,
    }


def main():
    results = [base_summarize(p) for p in sorted(run_dir.glob("*.npz"))]

    summary = pd.DataFrame(results)
    save_path = run_dir / "batch_metrics_summary.csv"
    summary.to_csv(save_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()