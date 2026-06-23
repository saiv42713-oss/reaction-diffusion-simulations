import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.classify_regime import classify_regime, classify_regime_and_return_fields
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

_OUTPUTS = Path(__file__).parent.parent / "outputs"
_OUTPUTS.mkdir(exist_ok=True)

def save_field(A, ba, bi, label):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(A, cmap="RdBu_r", origin="lower")
    ax.set_title(f"ba={ba}  |  bi={bi}  |  regime: {label}",
                 fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("x (cells)", fontsize=10)
    ax.set_ylabel("y (cells)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Activator concentration (A)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(_OUTPUTS / f"field_ba{ba}_bi{bi}.png", dpi=150, bbox_inches="tight")
    plt.close()

ba_range = [2, 11, 2]
bi_range = [2, 31, 4]
results = []

for ba in range(ba_range[0], ba_range[1], ba_range[2]):
    for bi in range(bi_range[0], bi_range[1], bi_range[2]):
        regime = classify_regime(ba, bi)
        print(f"ba={ba}, bi={bi} → {regime}")
        results.append((ba, bi, regime))

cases_to_inspect = [(8,6), (8,10), (8,18), (10,10), (10,26)]
for ba, bi in cases_to_inspect:
    A, _ = classify_regime_and_return_fields(float(ba), float(bi))
    matched = next((r for b, bii, r in results if b == ba and bii == bi), "?")
    save_field(A, ba, bi, matched)

ba_vals = [2, 4, 6, 8, 10]
bi_vals = [2, 6, 10, 14, 18, 22, 26, 30]
color_map = {"ON": "#2166ac", "IRREGULAR": "#f5deb3", "OFF": "#b2182b"}
label_map  = {"ON": "ON", "IRREGULAR": "Irregular", "OFF": "OFF"}

rgb_grid = np.ones((len(bi_vals), len(ba_vals), 3))
for ba, bi, regime in results:
    i = ba_vals.index(ba)
    j = bi_vals.index(bi)
    r, g, b = tuple(int(color_map[regime][k:k+2], 16)/255 for k in (1, 3, 5))
    rgb_grid[j, i] = [r, g, b]

fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(rgb_grid, aspect="auto", origin="lower",
          extent=[1.5, 10.5, 1, 31])
ax.set_xticks(ba_vals); ax.set_xticklabels(ba_vals, fontsize=11)
ax.set_yticks(bi_vals); ax.set_yticklabels(bi_vals, fontsize=11)
ax.set_xlabel("ba  (activator production rate)", fontsize=12)
ax.set_ylabel("bi  (inhibitor production rate)", fontsize=12)
ax.set_title("Parameter space map\nna=10, ni=4, Di=20, juxtacrine",
             fontsize=12, fontweight="bold")
patches = [mpatches.Patch(color=color_map[r], label=label_map[r])
           for r in ["ON", "IRREGULAR", "OFF"]]
ax.legend(handles=patches, loc="upper left", fontsize=11,
          framealpha=0.9, edgecolor="gray")
plt.tight_layout()
plt.savefig(_OUTPUTS / "param_map.png", dpi=150, bbox_inches="tight")
print(f"Saved {_OUTPUTS / 'param_map.png'}")
