from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        "peak_fft_sharpness": peak_sharpness,
        "autocorr_zero_crossing": zero_crossing,
        "autocorr_length_1e": corr_length,
    }


# Extracts trailing sim numbers (e.g. 0004)
def _get_sim(values):
    return values.astype(str).str.extract(r"(\d+)(?=\D*$)")[0].astype(float).astype("Int64")


# Merges metrics with batch data that includes parameters and manipulated variables
def merge_run(summary, run_dir):
    summary = summary.copy()
    summary["sim_number"] = _get_sim(summary["file"])

    for path in [run_dir / "runs.txt", run_dir / "batch_results.csv"]:
        metadata = pd.read_csv(path, sep=None, engine="python")

        if "sim_number" not in metadata.columns:
            if "run" in metadata.columns:
                metadata["sim_number"] = metadata["run"]
            elif "file" in metadata.columns:
                metadata["sim_number"] = _get_sim(metadata["file"])
            else:
                metadata["sim_number"] = np.arange(len(metadata))

        metadata["sim_number"] = pd.to_numeric(metadata["sim_number"], errors="coerce").astype("Int64")
        return summary.merge(metadata, on="sim_number", how="left")

    return summary


# Plot selected metrics across parameters (matplotlib)
def plot_metrics(merged, run_dir, x_var="inh_prod_rate", col_var="inh_diffusion", hue_var="act_hill_coeff"):
    # Add more as needed to visrualize
    metrics = [
        "activated_fraction",
        "mean_diameter_hexes",
        "peak_fft_sharpness",
        "autocorr_length_1e",
    ]

    metrics = [m for m in metrics if m in merged.columns]
    required = [x_var, col_var, hue_var]
    missing = [c for c in required if c not in merged.columns]
    if not metrics or missing:
        print(f"Skipping plot. Missing columns: {missing}")
        return None

    # Work from copy for formatting
    merged = merged.copy()
    for col in required:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    col_values = sorted(merged[col_var].dropna().unique())
    hue_values = sorted(merged[hue_var].dropna().unique())
    fig, axes = plt.subplots(
        len(metrics),
        len(col_values),
        figsize=(3.0 * len(col_values), 2.4 * len(metrics)),
        sharex=True,
        squeeze=False,
    )

    for row_idx, metric in enumerate(metrics):
        values = merged[metric].replace([np.inf, -np.inf], np.nan)
        y_min = values.min()
        y_max = values.max()
        for col_idx, col_value in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            subset_col = merged[merged[col_var] == col_value]
            for hue_value in hue_values:
                subset = subset_col[subset_col[hue_var] == hue_value]
                if subset.empty:
                    continue
                grouped = subset.groupby(x_var, as_index=False)[metric].mean().sort_values(x_var)
                ax.plot(grouped[x_var], grouped[metric], marker="o", linewidth=1.4, markersize=3, label=f"{hue_var}={hue_value:g}")
            if row_idx == 0:
                ax.set_title(f"{col_var}={col_value:g}")
            if col_idx == 0:
                ax.set_ylabel(metric)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel(x_var)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
                margin = 0.05 * (y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)
            ax.grid(alpha=0.25)

    # Create shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False)
        fig.subplots_adjust(right=0.86)

    fig.tight_layout(rect=(0, 0, 0.86, 1))
    save_path = run_dir / "batch_metric_plots.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def main():
    results = [base_summarize(p) for p in sorted(run_dir.glob("*.npz"))]

    summary = pd.DataFrame(results)
    save_path = run_dir / "batch_metrics_summary.csv"
    summary.to_csv(save_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {save_path}")

    merged = merge_run(summary, run_dir)
    merged_path = run_dir / "batch_metrics_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved: {merged_path}")

    plot_path = plot_metrics(merged, run_dir)
    if plot_path is not None:
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
