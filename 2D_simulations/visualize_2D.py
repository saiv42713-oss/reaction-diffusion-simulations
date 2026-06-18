"""
Visualization utilities for 2D hex-grid simulations.

This module provides:
- helper functions for constructing and drawing pointy-top hexagonal lattices,
- animated visualizations of one-pair and two-pair simulations,
- static plots for final-state snapshots,
- an overlay plot for comparing two activator fields.

All functions assume field arrays are shaped as (Ny, Nx).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import RegularPolygon


def _global_max(hist: Sequence[np.ndarray]) -> float:
    """
    Return the maximum value across a history of arrays, with a small floor.

    The floor avoids zero-valued colormap limits.
    """
    return max(1e-12, max(float(np.max(np.asarray(frame))) for frame in hist))


def _hex_polygons(ny: int, nx: int, hex_radius: float = 1.0) -> list[np.ndarray]:
    """
    Build pointy-top hexagons in an even-r offset layout.

    Center coordinates:
        x = sqrt(3) * R * (c + 0.5 * (r % 2))
        y = 1.5 * R * r
    """
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    polys: list[np.ndarray] = []

    for r in range(ny):
        y = 1.5 * hex_radius * r
        x_offset = 0.5 * np.sqrt(3) * hex_radius if (r % 2 == 1) else 0.0
        for c in range(nx):
            x = np.sqrt(3) * hex_radius * c + x_offset
            verts = np.column_stack(
                [
                    x + hex_radius * np.cos(angles),
                    y + hex_radius * np.sin(angles),
                ]
            )
            polys.append(verts)

    return polys


def _hex_bounds(ny: int, nx: int, hex_radius: float = 1.0) -> tuple[float, float, float, float]:
    """
    Return axis limits that frame the hex grid neatly.
    """
    max_x = np.sqrt(3) * hex_radius * (nx - 1 + 0.5 * ((ny - 1) % 2))
    max_y = 1.5 * hex_radius * (ny - 1)
    pad = 1.1 * hex_radius
    return (-pad, max_x + pad, -pad, max_y + pad)


def _add_hex_field(
    ax,
    field: np.ndarray,
    cmap,
    norm,
    hex_radius: float = 1.0,
    edgecolor: str = "0.35",
    linewidth: float = 0.4,
):
    """
    Add a hexagonal scalar field to an axes and return the PolyCollection.
    """
    field = np.asarray(field)
    ny, nx = field.shape
    polys = _hex_polygons(ny, nx, hex_radius=hex_radius)

    collection = PolyCollection(
        polys,
        array=field.ravel(),
        cmap=cmap,
        norm=norm,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    ax.add_collection(collection)

    xmin, xmax, ymin, ymax = _hex_bounds(ny, nx, hex_radius=hex_radius)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return collection


def animate_histories(
    A_hist: Sequence[np.ndarray],
    R_hist: Sequence[np.ndarray],
    save_every: int,
    title: str = "Coupled Dynamics (2D Hex Grid)",
    loop: bool = False,
    savefile: Optional[str] = None,
    fps: int = 20,
    hex_radius: float = 1.0,
    cmap_A: str = "Greens",
    cmap_R: str = "Blues",
) -> None:
    """
    Animate activator and inhibitor histories on a hexagonal lattice.

    Parameters
    ----------
    A_hist, R_hist
        Sequences of 2D arrays with shape (Ny, Nx).
    save_every
        Number of simulation steps between stored frames.
    title
        Figure title prefix.
    loop
        Whether the animation should loop.
    savefile
        Optional output path for an MP4 movie.
    fps
        Frames per second for movie export.
    hex_radius
        Radius of each drawn hexagon.
    cmap_A, cmap_R
        Colormaps for activator and inhibitor.
    """
    num_frames = len(A_hist)
    A0 = np.asarray(A_hist[0])
    R0 = np.asarray(R_hist[0])

    vmax_A = _global_max(A_hist)
    vmax_R = _global_max(R_hist)

    norm_A = Normalize(vmin=0.0, vmax=vmax_A)
    norm_R = Normalize(vmin=0.0, vmax=vmax_R)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axA, axR = axes

    pcA = _add_hex_field(axA, A0, cmap=cmap_A, norm=norm_A, hex_radius=hex_radius)
    pcR = _add_hex_field(axR, R0, cmap=cmap_R, norm=norm_R, hex_radius=hex_radius)

    axA.set_title("Activator")
    axR.set_title("Inhibitor")
    axA.set_xlabel("x")
    axA.set_ylabel("y")
    axR.set_xlabel("x")
    axR.set_ylabel("y")

    cbarA = fig.colorbar(pcA, ax=axA, fraction=0.046, pad=0.04)
    cbarR = fig.colorbar(pcR, ax=axR, fraction=0.046, pad=0.04)
    cbarA.set_label("Concentration")
    cbarR.set_label("Concentration")

    fig.suptitle(f"{title}\nStep 0")

    def update(frame: int):
        A = np.asarray(A_hist[frame])
        R = np.asarray(R_hist[frame])

        pcA.set_array(A.ravel())
        pcR.set_array(R.ravel())

        fig.suptitle(f"{title}\nStep {frame * save_every}")
        return [pcA, pcR]

    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=50,
        blit=False,
        repeat=loop,
    )

    if savefile:
        writer = FFMpegWriter(fps=fps)
        ani.save(savefile, writer=writer)
        plt.close(fig)
        print(f"Movie saved to {savefile}")
    else:
        plt.show()


def plot_one_frame(
    A_final: np.ndarray,
    R_final: np.ndarray,
    final_step: int,
    outfile_png: str,
    title: str = "2D Hex Grid",
    hex_radius: float = 1.0,
    cmap_A: str = "Greens",
    cmap_R: str = "Blues",
    A_initial: np.ndarray | None = None,
    R_initial: np.ndarray | None = None,
) -> None:
    """Plot initial and final activator/inhibitor states and save as PNG."""

    A_final = np.asarray(A_final)
    R_final = np.asarray(R_final)

    if A_initial is None:
        A_initial = A_final
    if R_initial is None:
        R_initial = R_final

    A_initial = np.asarray(A_initial)
    R_initial = np.asarray(R_initial)

    vmax_A = max(1e-12, float(np.max([A_initial.max(), A_final.max()])))
    vmax_R = max(1e-12, float(np.max([R_initial.max(), R_final.max()])))

    norm_A = Normalize(vmin=0.0, vmax=vmax_A)
    norm_R = Normalize(vmin=0.0, vmax=vmax_R)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    axA0, axR0 = axes[0]
    axA1, axR1 = axes[1]

    pcA0 = _add_hex_field(axA0, A_initial, cmap=cmap_A, norm=norm_A, hex_radius=hex_radius)
    pcR0 = _add_hex_field(axR0, R_initial, cmap=cmap_R, norm=norm_R, hex_radius=hex_radius)
    pcA1 = _add_hex_field(axA1, A_final, cmap=cmap_A, norm=norm_A, hex_radius=hex_radius)
    pcR1 = _add_hex_field(axR1, R_final, cmap=cmap_R, norm=norm_R, hex_radius=hex_radius)

    axA0.set_title("Initial Activator")
    axR0.set_title("Initial Inhibitor")
    axA1.set_title("Final Activator")
    axR1.set_title("Final Inhibitor")

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(pcA1, ax=[axA0, axA1], fraction=0.046, pad=0.04, label="Activator")
    fig.colorbar(pcR1, ax=[axR0, axR1], fraction=0.046, pad=0.04, label="Inhibitor")

    fig.suptitle(f"{title}\nFinal step {final_step}")
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def animate_four_histories(
    A1_hist: Sequence[np.ndarray],
    I1_hist: Sequence[np.ndarray],
    A2_hist: Sequence[np.ndarray],
    I2_hist: Sequence[np.ndarray],
    save_every: int,
    title: str = "Coupled Dynamics (2D Hex Grid)",
    loop: bool = False,
    savefile: Optional[str] = None,
    fps: int = 20,
    hex_radius: float = 1.0,
    cmap_A: str = "Reds",
    cmap_I: str = "Blues",
) -> None:
    """
    Animate four fields (A1, I1, A2, I2) on a 2x2 hex-grid figure.
    """
    num_frames = len(A1_hist)

    A1_0 = np.asarray(A1_hist[0])
    I1_0 = np.asarray(I1_hist[0])
    A2_0 = np.asarray(A2_hist[0])
    I2_0 = np.asarray(I2_hist[0])

    vmax_A = max(_global_max(A1_hist), _global_max(A2_hist))
    vmax_I = max(_global_max(I1_hist), _global_max(I2_hist))

    norm_A = Normalize(vmin=0.0, vmax=vmax_A)
    norm_I = Normalize(vmin=0.0, vmax=vmax_I)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axA1, axI1, axA2, axI2 = axes.ravel()

    pcA1 = _add_hex_field(axA1, A1_0, cmap=cmap_A, norm=norm_A, hex_radius=hex_radius)
    pcI1 = _add_hex_field(axI1, I1_0, cmap=cmap_I, norm=norm_I, hex_radius=hex_radius)
    pcA2 = _add_hex_field(axA2, A2_0, cmap=cmap_A, norm=norm_A, hex_radius=hex_radius)
    pcI2 = _add_hex_field(axI2, I2_0, cmap=cmap_I, norm=norm_I, hex_radius=hex_radius)

    axA1.set_title("A1")
    axI1.set_title("I1")
    axA2.set_title("A2")
    axI2.set_title("I2")

    for ax in (axA1, axI1, axA2, axI2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbarA = fig.colorbar(pcA1, ax=[axA1, axA2], fraction=0.046, pad=0.04)
    cbarI = fig.colorbar(pcI1, ax=[axI1, axI2], fraction=0.046, pad=0.04)
    cbarA.set_label("Concentration")
    cbarI.set_label("Concentration")

    fig.suptitle(f"{title}\nStep 0")

    def update(frame: int):
        A1 = np.asarray(A1_hist[frame])
        I1 = np.asarray(I1_hist[frame])
        A2 = np.asarray(A2_hist[frame])
        I2 = np.asarray(I2_hist[frame])

        pcA1.set_array(A1.ravel())
        pcI1.set_array(I1.ravel())
        pcA2.set_array(A2.ravel())
        pcI2.set_array(I2.ravel())

        fig.suptitle(f"{title}\nStep {frame * save_every}")
        return [pcA1, pcI1, pcA2, pcI2]

    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=50,
        blit=False,
        repeat=loop,
    )

    if savefile:
        writer = FFMpegWriter(fps=fps)
        ani.save(savefile, writer=writer)
        plt.close(fig)
        print(f"Movie saved to {savefile}")
    else:
        plt.show()


def _add_overlay_hex_field_rgb(
    ax,
    A1: np.ndarray,
    A2: np.ndarray,
    norm1,
    norm2,
    hex_radius: float = 1.0,
):
    """
    Overlay A1 and A2 as additive RGB colors.

    A1 contributes to the red channel, A2 contributes to the green channel,
    and overlap appears yellow.
    """
    A1 = np.asarray(A1, dtype=float)
    A2 = np.asarray(A2, dtype=float)
    ny, nx = A1.shape

    patches = []
    colors = []

    dx = 3.0 * hex_radius / 2.0
    dy = np.sqrt(3.0) * hex_radius

    for r in range(ny):
        y = r * dy
        x_offset = 0.0 if (r % 2 == 0) else dx / 2.0
        for c in range(nx):
            x = c * dx + x_offset

            v1 = float(np.clip(norm1(A1[r, c]), 0.0, 1.0))
            v2 = float(np.clip(norm2(A2[r, c]), 0.0, 1.0))

            rgb = np.clip(np.array([v1, v2, 0.0]), 0.0, 1.0)

            hex_patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=np.radians(30),
                edgecolor="none",
            )
            patches.append(hex_patch)
            colors.append(rgb)

    pc = PatchCollection(patches, facecolor=colors, edgecolor="none", linewidth=0)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.autoscale_view()
    return pc


def plot_four_frames(
    A1: np.ndarray,
    I1: np.ndarray,
    A2: np.ndarray,
    I2: np.ndarray,
    final_step: int,
    outfile_png: str,
    title: str = "Final State (2D Hex Grid)",
    hex_radius: float = 1.0,
) -> None:
    """
    Plot four fields plus an RGB overlay and save the figure as a PNG.
    """
    A1 = np.asarray(A1)
    I1 = np.asarray(I1)
    A2 = np.asarray(A2)
    I2 = np.asarray(I2)

    vmax_A1 = max(1e-12, float(np.max(A1)))
    vmax_A2 = max(1e-12, float(np.max(A2)))
    vmax_I1 = max(1e-12, float(np.max(I1)))
    vmax_I2 = max(1e-12, float(np.max(I2)))

    norm_A1 = Normalize(vmin=0.0, vmax=vmax_A1)
    norm_A2 = Normalize(vmin=0.0, vmax=vmax_A2)
    norm_I1 = Normalize(vmin=0.0, vmax=vmax_I1)
    norm_I2 = Normalize(vmin=0.0, vmax=vmax_I2)

    cmap_A1 = LinearSegmentedColormap.from_list("A1_red", ["#fff5f0", "#fb6a4a", "#cb181d"])
    cmap_A2 = LinearSegmentedColormap.from_list("A2_green", ["#f7fcf5", "#74c476", "#238b45"])
    cmap_I1 = LinearSegmentedColormap.from_list("I1_light_blue", ["#f7fbff", "#9ecae1", "#6baed6"])
    cmap_I2 = LinearSegmentedColormap.from_list("I2_dark_blue", ["#eff3ff", "#6baed6", "#08306b"])

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axA1 = fig.add_subplot(gs[0, 0])
    axI1 = fig.add_subplot(gs[0, 1])
    axA2 = fig.add_subplot(gs[1, 0])
    axI2 = fig.add_subplot(gs[1, 1])
    axOverlay = fig.add_subplot(gs[:, 2])

    pcA1 = _add_hex_field(axA1, A1, cmap=cmap_A1, norm=norm_A1, hex_radius=hex_radius)
    pcI1 = _add_hex_field(axI1, I1, cmap=cmap_I1, norm=norm_I1, hex_radius=hex_radius)
    pcA2 = _add_hex_field(axA2, A2, cmap=cmap_A2, norm=norm_A2, hex_radius=hex_radius)
    pcI2 = _add_hex_field(axI2, I2, cmap=cmap_I2, norm=norm_I2, hex_radius=hex_radius)

    _add_overlay_hex_field_rgb(axOverlay, A1, A2, norm_A1, norm_A2, hex_radius=hex_radius)

    axA1.set_title("A1")
    axI1.set_title("I1")
    axA2.set_title("A2")
    axI2.set_title("I2")
    axOverlay.set_title("Overlay: A1 (red) + A2 (green)")

    for ax in (axA1, axI1, axA2, axI2, axOverlay):
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(pcA1, ax=axA1, fraction=0.046, pad=0.04, label="Concentration")
    fig.colorbar(pcI1, ax=axI1, fraction=0.046, pad=0.04, label="Concentration")
    fig.colorbar(pcA2, ax=axA2, fraction=0.046, pad=0.04, label="Concentration")
    fig.colorbar(pcI2, ax=axI2, fraction=0.046, pad=0.04, label="Concentration")

    fig.suptitle(f"{title}\nStep {final_step}")
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
