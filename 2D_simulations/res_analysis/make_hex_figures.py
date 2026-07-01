import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from pathlib import Path

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

RUNS = Path('/home/claude/analysis/runs')
FIGS = Path('/home/claude/analysis/figs')
FIGS.mkdir(exist_ok=True)

BI = [1, 3, 5, 12, 14]


def hex_polygons(ny, nx, hex_radius=1.0):
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    polys = []
    for r in range(ny):
        y = 1.5 * hex_radius * r
        x_offset = 0.5 * np.sqrt(3) * hex_radius if (r % 2 == 1) else 0.0
        for c in range(nx):
            x = np.sqrt(3) * hex_radius * c + x_offset
            verts = np.column_stack([
                x + hex_radius * np.cos(angles),
                y + hex_radius * np.sin(angles),
            ])
            polys.append(verts)
    return polys


def hex_bounds(ny, nx, hex_radius=1.0):
    max_x = np.sqrt(3) * hex_radius * (nx - 1 + 0.5 * ((ny - 1) % 2))
    max_y = 1.5 * hex_radius * (ny - 1)
    pad = 1.1 * hex_radius
    return (-pad, max_x + pad, -pad, max_y + pad)


def add_hex_field(ax, field, cmap="Greens", norm=None, hex_radius=1.0,
                   edgecolor="0.35", linewidth=0.15):
    field = np.asarray(field)
    ny, nx = field.shape
    polys = hex_polygons(ny, nx, hex_radius=hex_radius)
    if norm is None:
        norm = Normalize(vmin=0, vmax=max(1e-12, field.max()))
    coll = PolyCollection(polys, array=field.ravel(), cmap=cmap, norm=norm,
                           edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection(coll)
    xmin, xmax, ymin, ymax = hex_bounds(ny, nx, hex_radius=hex_radius)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return coll


# shared green scale across all five circuits so ON reads bright, OFF reads pale
runs = {bi: np.load(RUNS / f'bi{bi}_final.npz') for bi in BI}
vmax = max(runs[bi]['A_final'].max() for bi in BI)
norm = Normalize(vmin=0, vmax=vmax)

for bi in BI:
    A = runs[bi]['A_final']
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    add_hex_field(ax, A, cmap="Greens", norm=norm)
    fig.tight_layout(pad=0.05)
    fig.savefig(FIGS / f"field_bi{bi}.png", dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
print("Saved 5 hexagon field thumbnails (Greens)")

# early window strip, same hex/green treatment, own per-frame scale since
# amplitude grows hugely from the near-zero start
snap = np.load(RUNS / 'early_window_snapshots.npz')
steps = snap['steps']
fig, axes = plt.subplots(1, len(steps), figsize=(17, 2.7))
for ax, s in zip(axes, steps):
    A = snap[f'A_{s}']
    add_hex_field(ax, A, cmap="Greens", norm=Normalize(vmin=0, vmax=max(1e-12, A.max())),
                  edgecolor="none", linewidth=0.0)
    ax.set_xlabel(f"t = {s*0.01:.1f}", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "early_window_strip.png", dpi=200)
plt.close(fig)
print("Saved early_window_strip.png (hex, Greens)")
