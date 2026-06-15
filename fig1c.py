#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from simulation_2D import run_coupled_hex  # your existing engine

# ── Simulation settings (from Table 1) ──────────────────────────────────────
SIM = dict(
    dt=0.01,
    dx=1.0,
    stopping_threshold=1e-6,
    min_steps=5000,
    save_every=10000,    # only need the final frame; saves memory
    spike_value=1.0,     # Table 1: 1.0 for random_uniform_over0 (NOT 2.0)
)

# Fixed biological params (Table 1, Fig 1C rows)
BASE_PARAMS = dict(
    act_prod_rate=5.0,      # ba  — fixed for the whole panel
    act_half_sat=1.0,
    inh_half_sat=1.0,
    act_decay_rate=1.0,
    inh_decay_rate=0.5,
    act_diffusion=0.0,      # overridden per condition
    inh_diffusion=10.0,
    basal_prod=0.0,
    act_hill_coeff=3,
    inh_hill_coeff=3,
    inh_prod_rate=None,     # overridden per condition
)

# Fig 1C conditions
JUX_CONDITIONS = [
    dict(bi=1,  Da=0.0, label="ON"),
    dict(bi=3,  Da=0.0, label="Stripes"),
    dict(bi=5,  Da=0.0, label="Reg Spots"),
    dict(bi=12, Da=0.0, label="Irreg Spots"),
    dict(bi=14, Da=0.0, label="OFF"),
]
PAR_CONDITIONS = [
    dict(bi=1, Da=1.0, label="ON"),
    dict(bi=3, Da=1.0, label="Stripes"),
    dict(bi=5, Da=1.0, label="Reg Spots"),
    dict(bi=7, Da=1.0, label="Irreg Spots"),
    dict(bi=8, Da=1.0, label="OFF"),
]


def run_one(bi, Da, Nx, Ny, seed):
    """Run a single condition; return the final A field."""
    np.random.seed(seed)
    params = {
        **BASE_PARAMS,
        "inh_prod_rate": float(bi),
        "act_diffusion":  float(Da),
    }
    act_type = "juxtacrine" if Da == 0.0 else "paracrine"

    A_hist, R_hist, final_step, a_ss, i_ss = run_coupled_hex(
        Ny, Nx,
        SIM["steps"] if hasattr(SIM, "steps") else 50000,  # corrected below
        SIM["dt"],
        SIM["dx"],
        params,
        SIM["stopping_threshold"],
        SIM["min_steps"],
        init_mode="random_uniform_over0",
        activator_type=act_type,
        spike_value=SIM["spike_value"],
        save_every=SIM["save_every"],
    )
    return A_hist[-1], final_step


def make_figure(Nx, Ny, seed, max_steps, out_path):
    SIM["steps"] = max_steps
    rows = [
        ("Juxtacrine", JUX_CONDITIONS),
        ("Paracrine",  PAR_CONDITIONS),
    ]

    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(2, 6, figure=fig,
                            hspace=0.10, wspace=0.05,
                            left=0.06, right=0.88,
                            top=0.88, bottom=0.08)
    fig.suptitle(
        "Fig. 1C Replication  ·  JAPI Circuit  (ba=5, na=ni=3, Di=10)",
        fontsize=11, fontweight="bold",
    )

    for row_idx, (row_name, conditions) in enumerate(rows):
        for col_idx, cond in enumerate(conditions):
            bi, Da, label = cond["bi"], cond["Da"], cond["label"]

            print(f"  [{row_name:10s}] bi={bi:2d} ({label:12s}) ...",
                  end="", flush=True)
            t0 = time.time()
            A, conv = run_one(bi, Da, Nx, Ny, seed)
            print(f"  step {conv:5d}  mean={A.mean():.3f}  "
                  f"std={A.std():.3f}  ({time.time()-t0:.1f}s)")

            ax = fig.add_subplot(gs[row_idx, col_idx])
            vmax = max(float(A.max()), 1e-3)
            im = ax.imshow(A, cmap="RdBu_r", vmin=0, vmax=vmax,
                           interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"bi={bi}\n{label}", fontsize=8, pad=3)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=9, fontweight="bold",
                              rotation=90, labelpad=6)

    # Shared colorbar (right side, for readability across panels)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.80])
    sm = cm.ScalarMappable(cmap="RdBu_r",
                           norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    cbar.set_label("Activator A\n(per-panel norm.)", fontsize=8, labelpad=6)

    # Parameter note 
    fig.text(0.5, 0.01,
             "Parameters: ba=5, da=1, na=ni=3, db=0.5, Di=10 | "
             "IC: random uniform | Grid: 100×100 | Seed: 42",
             ha="center", fontsize=7, color="gray")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")