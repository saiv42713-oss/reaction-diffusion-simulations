import numpy as np
from pattern_metrics_reference import describe_pattern
from classify_regime import classify_regime_and_return_fields
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Known parameter sets to calibrate the pattern metrics
# Labels will be updated once classify_regime confirms each regime

test_cases = [
    (5.0, 2.0,  "IRREGULAR"),
    (4.0, 2.0,  "ON"),
    (6.0, 3.0,  "IRREGULAR"),
    (7.0, 4.0,  "IRREGULAR"),
    (5.0, 3.0,  "IRREGULAR"),
]

for ba, bi, expected in test_cases:
    final_A, final_B = classify_regime_and_return_fields(ba, bi)
    metrics = describe_pattern(final_A, final_B)

    print(f"\nba={ba}, bi={bi}  (expected: {expected})")
    print("  " + "-"*35)
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(final_A, cmap="viridis", origin="lower")
    ax.set_title(f"ba={ba}  bi={bi}  |  expected: {expected}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("x (cells)", fontsize=9)
    ax.set_ylabel("y (cells)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Activator concentration (A)", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"calibration_ba{int(ba)}_bi{int(bi)}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration_ba{int(ba)}_bi{int(bi)}.png")