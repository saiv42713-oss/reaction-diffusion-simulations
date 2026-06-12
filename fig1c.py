import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classify_regime import classify_regime
from joblib import Parallel, delayed
from tqdm import tqdm
import json


# Full parameter grid matching paper Fig 1C
ba_values = list(range(1, 11))
bi_values = list(range(1, 11))

# Build all combinations
param_combinations = [
    (float(ba), float(bi))
    for ba in ba_values
    for bi in bi_values
]

def run_one(ba, bi):
    regime = classify_regime(ba, bi)
    return (ba, bi, regime)

# Run in parallel using all CPU cores
results_list = Parallel(n_jobs=-1)(
    delayed(run_one)(ba, bi)
    for ba, bi in tqdm(param_combinations, desc="Running simulations")
)

# Store results
results = {(ba, bi): regime for ba, bi, regime in results_list}

# Print summary
print("\nAll results:")
for (ba, bi), regime in sorted(results.items()):
    print(f"ba={ba}, bi={bi} → {regime}")
import json

# Save results to JSON for later use
results_serializable = {f"{k[0]},{k[1]}": v for k, v in results.items()}
with open("fig1c_results.json", "w") as f:
    json.dump(results_serializable, f, indent=2)

# Color map
regime_colors = {
    "ON":        "green",
    "OFF":       "red",
    "TURING":    "yellow",
    "IRREGULAR": "blue"
}

fig, ax = plt.subplots(figsize=(8, 8))

for ba in ba_values:
    for bi in bi_values:
        regime = results.get((float(ba), float(bi)), "OFF")
        color = regime_colors.get(regime, "gray")
        ax.add_patch(plt.Rectangle((bi - 1, ba - 1), 1, 1, color=color))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xticks(np.arange(0.5, 10.5))
ax.set_xticklabels(bi_values)
ax.set_yticks(np.arange(0.5, 10.5))
ax.set_yticklabels(ba_values)
ax.set_xlabel("Inhibitor production rate (bi)", fontsize=12)
ax.set_ylabel("Activator production rate (ba)", fontsize=12)
ax.set_title("Patterning regimes — JAPI circuit (na=3, ni=3, Di=10)", fontsize=13)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green",  label="ON"),
    Patch(facecolor="red",    label="OFF"),
    Patch(facecolor="yellow", label="TURING"),
    Patch(facecolor="blue",   label="IRREGULAR"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=11)
ax.grid(False)

plt.tight_layout()
plt.savefig("fig1c.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1c.png")