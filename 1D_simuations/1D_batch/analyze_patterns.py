import sys
from pathlib import Path
import numpy as np
from scipy.signal import detrend, find_peaks, windows
import pandas as pd
import re
import matplotlib.pyplot as plt

# allow importing simulation.py from parent directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from visualize import plot_one_frame

def analyze_pattern(a, dx=1.0, apply_window=True, exclude_bins=1,
                             score_threshold=8.0, amp_factor=5.0, initial_noise_sd=None,
                             plot=False):
    """
    Returns:
      dominant_freq, dominant_wavelength, score, amplitude_sd, is_pattern
    score = peak_power / median(background_power_excluding_peak_and_k0)
    is_pattern = score > score_threshold and amplitude condition satisfied and peak not at k=0
    """
    a = np.asarray(a, dtype=float)
    N = len(a)
    if N < 4:
        return np.nan, np.nan, np.nan, np.nan, False

    # 1) detrend (remove mean + linear trend)
    a0 = detrend(a, type='linear')

    # 2) window to reduce spectral leakage
    if apply_window:
        w = windows.hann(N)
        a0 = a0 * w

    # 3) positive-frequency power using rFFT
    # rfftfreq gives frequencies in cycles per unit length (not angular)
    freqs = np.fft.rfftfreq(N, d=dx)    # cycles per unit length
    fftvals = np.fft.rfft(a0)
    P = (np.abs(fftvals) ** 2) / N      # power (normalized by N; ratios are what matter)

    # 4) explicitly zero-out the k=0 (mean) bin for analysis
    P0 = P.copy()
    P0[0] = 0.0

    # 5) find dominant peak index (nonzero)
    # Option A: simple argmax (fast)
    peak_idx = int(np.argmax(P0))
    peak_power = float(P0[peak_idx])
    dominant_freq = float(freqs[peak_idx]) if peak_idx < len(freqs) else 0.0
    dominant_wavelength = (1.0 / dominant_freq) if dominant_freq != 0 else np.inf

    # 6) estimate background: median excluding k=0 and a small neighborhood around the peak
    idxs = np.arange(len(P0))
    exclude_mask = np.zeros_like(idxs, dtype=bool)
    exclude_mask[0] = True  # exclude mean
    lo = max(0, peak_idx - exclude_bins)
    hi = min(len(idxs) - 1, peak_idx + exclude_bins)
    exclude_mask[lo:hi + 1] = True
    background_vals = P0[~exclude_mask]
    if background_vals.size == 0:
        background = np.median(P0) + 1e-12
    else:
        # use median for robustness
        background = float(np.median(background_vals) + 1e-12)

    score = peak_power / background

    # 7) amplitude checks
    amp_sd = float(np.std(a))
    amp_range = float(a.max() - a.min())
    if initial_noise_sd is not None and initial_noise_sd > 0:
        amp_ok = (amp_sd > amp_factor * initial_noise_sd)
    else:
        # fallback: require amplitude > small relative epsilon
        eps = 1e-12 + 1e-3 * max(1.0, np.max(np.abs(a)))
        amp_ok = (amp_sd > eps)

    # 8) optional: more robust peak selection using scipy find_peaks (prominence)
    #    (uncomment to require a minimum prominence)
    # peaks, props = find_peaks(P0, prominence=(background*0.5))
    # if peaks.size > 0:
    #     # choose the most powerful of the qualified peaks
    #     chosen = peaks[np.argmax(P0[peaks])]
    #     peak_idx = int(chosen)
    #     peak_power = float(P0[peak_idx])
    #     dominant_freq = float(freqs[peak_idx])
    #     dominant_wavelength = 1.0 / dominant_freq if dominant_freq != 0 else np.inf
    #     # recompute score relative to local background if desired

    is_pattern = (score > score_threshold) and amp_ok and (peak_idx != 0)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(freqs, P, label='power')
        plt.axvline(dominant_freq, color='C1', linestyle='--',
                    label=f'peak f={dominant_freq:.3g}')
        plt.xlabel("spatial frequency (cycles / unit length)")
        plt.ylabel("power")
        plt.title(f"FFT: λ={dominant_wavelength:.3g}, score={score:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return dominant_freq, dominant_wavelength, score, amp_sd, bool(is_pattern)


def parse_list_string(s):
    """Safely convert a string like '[1,2,3]' or '[1 2 3]' to a list of floats."""
    if not isinstance(s, str):
        return None
    # remove brackets
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # replace multiple spaces with a single comma
    s = re.sub(r"\s+", ",", s)
    # ensure commas between numbers
    s = s.replace(",,", ",")
    try:
        return np.array([float(x) for x in s.split(",") if x.strip() != ""])
    except Exception:
        return None


# === Load file ===
if len(sys.argv) < 2:
    print("Usage: python analyze_batch.py <path/to/batch_results.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_folder = input_file.rsplit("/", 1)[0] + "/"

df = pd.read_csv(input_file)

dominant_freqs = []
dominant_wavelengths = []
a_max = []
a_diff = []
scores = []
pattern_flags = []

for i, row in df.iterrows():
    a = parse_list_string(row["activator_final"])
    if a is None or len(a) < 3:
        print(f"Skipping row {i}: could not parse activator_final")
        dominant_freqs.append(np.nan)
        dominant_wavelengths.append(np.nan)
        a_diff.append(np.nan)
        a_max.append(np.nan)
        continue

    amax_single = a.max()
    a_max.append(amax_single)
    a_diff.append(amax_single - a.min())

    if np.std(a) < 1e-6 or np.isnan(a).any():
        dominant_freqs.append(np.nan)
        dominant_wavelengths.append(np.nan)
        scores.append(np.nan)
        pattern_flags.append(np.nan)
        continue

    dom_freq, dom_lambda, score, amp_sd, is_pattern = analyze_pattern(a, dx=1.0, plot=False)
    dominant_freqs.append(dom_freq)
    dominant_wavelengths.append(dom_lambda)
    # you can also save score and is_pattern in the dataframe
    scores.append(score)
    pattern_flags.append(is_pattern)

df["dominant_freq"] = dominant_freqs
df["dominant_wavelength"] = dominant_wavelengths
df["max_a"] = a_max
df["diff_a"] = a_diff
df["score"] = scores
df["pattern_flag"] = pattern_flags

df.to_csv(output_folder + "/patterning_summary.csv", index=False)
