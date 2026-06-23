import sys
import csv
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.classify_regime import classify_regime

_DATA = Path(__file__).parent.parent / "data"
_DATA.mkdir(exist_ok=True)

inh_prod_rate_values = [1, 3, 5, 12, 14]
act_hill_coeff_values = [1, 3, 5, 7, 9]
inh_diffusion_values = [2, 6, 10, 14, 18]

fieldnames = [
    "inh_prod_rate", "act_hill_coeff", "inh_diffusion",
    "regime", "fft_ratio", "uniformity", "fft_ratio_spike",
    "n_features", "spacing_regularity", "interfeature_dist",
    "elapsed_s", "error",
]

with open(_DATA / "nicole_grid_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    total = len(inh_prod_rate_values) * len(act_hill_coeff_values) * len(inh_diffusion_values)
    count = 0

    for bi in inh_prod_rate_values:
        for hill in act_hill_coeff_values:
            for diff in inh_diffusion_values:
                count += 1
                t0 = time.time()
                print(f"[{count}/{total}] bi={bi}, act_hill_coeff={hill}, inh_diffusion={diff}")

                try:
                    regime, diagnostics = classify_regime(
                        5.0, bi,
                        act_hill_coeff=hill,
                        inh_diffusion=diff,
                    )
                    row = {
                        "inh_prod_rate": bi,
                        "act_hill_coeff": hill,
                        "inh_diffusion": diff,
                        "regime": regime,
                        "fft_ratio": diagnostics.get("fft_ratio"),
                        "uniformity": diagnostics.get("uniformity"),
                        "fft_ratio_spike": diagnostics.get("fft_ratio_spike"),
                        "n_features": diagnostics.get("n_features"),
                        "spacing_regularity": diagnostics.get("spacing_regularity"),
                        "interfeature_dist": diagnostics.get("interfeature_dist"),
                        "elapsed_s": round(time.time() - t0, 2),
                        "error": "",
                    }
                except Exception as e:
                    row = {
                        "inh_prod_rate": bi,
                        "act_hill_coeff": hill,
                        "inh_diffusion": diff,
                        "regime": "ERROR",
                        "fft_ratio": None,
                        "uniformity": None,
                        "fft_ratio_spike": None,
                        "n_features": None,
                        "spacing_regularity": None,
                        "interfeature_dist": None,
                        "elapsed_s": round(time.time() - t0, 2),
                        "error": str(e)[:300],
                    }

                writer.writerow(row)
                f.flush()
