"""
Reproduce Fig. 1C from the paper for the JAPI circuit.

This script:
- runs the same five inh_prod_rate (bi) values used in the published figure,
- for both juxtacrine and paracrine activator types,
- reusing run_coupled_hex directly, the same engine every other script here uses,
- and saves a single 10-panel comparison figure.

By default, outputs are written under:
    results/fig1c/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation_2D import run_coupled_hex

# Model parameters shared by every panel. Only inh_prod_rate (bi) and
# activator_type vary between panels; everything else matches the paper's
# own Fig. 1C setup.
BASE_PARAMS = {
    "act_prod_rate": 5.0,
    "act_half_sat": 1.0,
    "inh_half_sat": 1.0,
    "act_decay_rate": 1.0,
    "inh_decay_rate": 0.5,
    "inh_diffusion": 10.0,
    "basal_prod": 0.0,
    "act_hill_coeff": 3,
    "inh_hill_coeff": 3,
}

# The five reference points from the published figure, in order:
# ON, stripes (Turing), regular spots (Turing), irregular spots, OFF.
BI_VALUES = [1, 3, 5, 12, 14]
BI_LABELS = ["ON", "Stripes", "Reg Spots", "Irreg Spots", "OFF"]


def run_one_panel(
    bi: float,
    activator_type: str,
    Nx: int,
    Ny: int,
    seed: int,
    max_steps: int,
) -> tuple[np.ndarray, int]:
    """
    Run a single Fig. 1C panel and return its final activator field.
    """
    np.random.seed(seed)
    params = dict(BASE_PARAMS)
    params["inh_prod_rate"] = float(bi)
    params["act_diffusion"] = 0.0 if activator_type == "juxtacrine" else 1.0

    A_hist, _, final_step, _, _ = run_coupled_hex(
        Ny,
        Nx,
        max_steps,
        0.01,
        1.0,
        params,
        1e-6,
        5000,
        init_mode="random_uniform_over0",
        activator_type=activator_type,
        spike_value=1.0,
        save_every=10000,
    )
    return A_hist[-1], final_step


def make_figure(
    Nx: int,
    Ny: int,
    seed: int,
    max_steps: int,
    outfile: Path,
) -> None:
    """
    Run all ten panels and save the comparison figure.

    Uses one shared color scale across every panel, computed from the actual
    data range across all ten runs. A per-panel auto-scale would stretch a
    genuinely-OFF panel's floating-point noise floor (~1e-44, scientifically
    meaningless) across the full colormap, making a blank result look like
    a real gradient.
    """
    fields: dict[tuple[str, float], np.ndarray] = {}
    final_steps: dict[tuple[str, float], int] = {}

    for activator_type in ["juxtacrine", "paracrine"]:
        for bi in BI_VALUES:
            field, final_step = run_one_panel(bi, activator_type, Nx, Ny, seed, max_steps)
            fields[(activator_type, bi)] = field
            final_steps[(activator_type, bi)] = final_step
            print(f"{activator_type}, bi={bi}: stopped at step {final_step}")

    vmax = max(f.max() for f in fields.values())

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    for row, activator_type in enumerate(["juxtacrine", "paracrine"]):
        for col, (bi, label) in enumerate(zip(BI_VALUES, BI_LABELS)):
            ax = axes[row, col]
            ax.imshow(fields[(activator_type, bi)], cmap="RdBu_r", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"bi={bi}\n{label}", fontsize=10)
            if col == 0:
                ax.set_ylabel(activator_type.capitalize(), fontsize=11, fontweight="bold")

    fig.suptitle("Fig. 1C Replication \u00b7 JAPI Circuit", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main() -> None:
    """
    Run the Fig. 1C reproduction and save the comparison figure.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce Fig. 1C (JAPI circuit, juxtacrine + paracrine)."
    )
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Name of the output folder inside results/.",
    )
    args = parser.parse_args()

    results_dir = Path("results") / (args.output or "fig1c")
    results_dir.mkdir(parents=True, exist_ok=True)

    make_figure(args.nx, args.ny, args.seed, args.max_steps, results_dir / "fig1c.png")


if __name__ == "__main__":
    main()
