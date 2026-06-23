"""
Pattern metrics reference — 2D pattern descriptors for the enumeration classifier.

Three analyses + two extras, all built on standard libraries:
  - Structure factor (numpy FFT)            -> periodicity: spacing, regularity, pattern-vs-noise
  - Spatial autocorrelation (numpy FFT)     -> real-space length scale (cross-checks Fourier)
  - Feature segmentation (scipy.ndimage)    -> feature size, inter-feature distance, density, shape
  - Variance floor                          -> flat/noise gate
  - A-B cross-correlation                   -> activator/inhibitor co- vs anti-location

Nothing here is from-scratch algorithm work; it is assembly of numpy / scipy.
Validate every metric by eye on ~10 known examples per regime before trusting thresholds.
"""
import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------
# 0. VARIANCE FLOOR  — the cheap flat/noise gate (fixes "noise called irregular")
# ----------------------------------------------------------------------
def is_flat_or_noise(field, rel_std_threshold=0.05):
    """True if the field has essentially no spatial structure.
    rel_std = spatial std / mean. Below threshold => flat or pure noise, not a pattern.
    Tune the threshold on the runs you flagged (mean~0.028, low std)."""
    m = np.mean(field)
    if m <= 0:
        return True
    return (np.std(field) / m) < rel_std_threshold


# ----------------------------------------------------------------------
# 1. STRUCTURE FACTOR  (2D power spectrum, radially averaged)  -> Fig 1D/E
# ----------------------------------------------------------------------
def radial_power_spectrum(field):
    """Return (k, P(k)): radially-averaged power spectrum.
    k is in units of cycles per box; convert to spacing as 1/k."""
    f = field - np.mean(field)                 # remove DC so the peak isn't swamped by the mean
    F = np.fft.fftshift(np.fft.fft2(f))        # 2D FFT, zero-frequency centered
    power = np.abs(F)**2                        # power spectrum
    ny, nx = field.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)      # radial wavenumber index of each pixel
    r = r.astype(int)
    # radial average: mean power in each integer-radius ring
    tbin = np.bincount(r.ravel(), weights=power.ravel())
    nbin = np.bincount(r.ravel())
    radial = tbin / np.maximum(nbin, 1)
    k = np.arange(len(radial))
    return k[1:], radial[1:]                     # drop k=0 (the DC bin)

def structure_factor_metrics(field):
    """peak position (spacing), peak width (regular vs irregular), peak prominence (pattern vs noise)."""
    k, P = radial_power_spectrum(field)
    if len(P) == 0 or P.max() <= 0:
        return dict(spacing=np.nan, peak_width=np.nan, prominence=0.0, k_peak=np.nan)
    i = np.argmax(P)
    k_peak = k[i]
    spacing = (1.0 / k_peak) if k_peak > 0 else np.nan   # characteristic spacing (box units)
    # prominence: peak height above the median background, normalized
    background = np.median(P)
    prominence = (P[i] - background) / (P[i] + 1e-12)     # ~1 sharp pattern, ~0 noise
    # peak width: full width at half max around the peak (broad = irregular, sharp = regular)
    half = 0.5 * P[i]
    lo = i
    while lo > 0 and P[lo] > half: lo -= 1
    hi = i
    while hi < len(P) - 1 and P[hi] > half: hi += 1
    peak_width = k[hi] - k[lo]
    return dict(spacing=spacing, peak_width=peak_width, prominence=prominence, k_peak=k_peak)


# ----------------------------------------------------------------------
# 2. SPATIAL AUTOCORRELATION  (real-space length scale; cross-checks Fourier)
# ----------------------------------------------------------------------
def autocorrelation_2d(field):
    """Normalized 2D autocorrelation via FFT (Wiener-Khinchin). Center = zero lag = 1.0."""
    f = field - np.mean(field)
    F = np.fft.fft2(f)
    ac = np.fft.ifft2(np.abs(F)**2).real
    ac = np.fft.fftshift(ac)
    ac /= ac.max()                               # normalize so zero-lag = 1
    return ac

def autocorr_length(field):
    """Characteristic length = radius of the first ring/peak of the radially-averaged autocorrelation.
    For a periodic pattern this matches the structure-factor spacing (good cross-check)."""
    ac = autocorrelation_2d(field)
    radial = _radial_autocorr(ac)
    # first local MAX after the central decay = the pattern's repeat distance
    for d in range(1, len(radial) - 1):
        if radial[d] > radial[d-1] and radial[d] >= radial[d+1] and radial[d] > 0:
            return d
    return np.nan

def _radial_autocorr(ac):
    ny, nx = ac.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    tbin = np.bincount(r.ravel(), weights=ac.ravel())
    nbin = np.bincount(r.ravel())
    return tbin / np.maximum(nbin, 1)

def autocorr_coherence(field):
    """Order / coherence length: how many periods before the autocorrelation envelope dies out.
    A clean lattice stays correlated over many rings (high); a locally-periodic-but-globally-disordered
    pattern decorrelates fast (low). Independent of the structure-factor peak width, and a direct
    'how ordered' number. Returned as (decay length in pixels, n_periods of coherence)."""
    ac = autocorrelation_2d(field)
    radial = _radial_autocorr(ac)
    if len(radial) < 3:
        return dict(coherence_length=np.nan, n_coherent_periods=np.nan)
    # envelope = magnitude of the oscillating autocorrelation; find where it falls below 1/e
    env = np.abs(radial)
    thr = 1.0 / np.e
    coherence_length = np.nan
    for d in range(1, len(env)):
        if env[d] < thr:
            coherence_length = d
            break
    # express as number of pattern periods (coherence length / spacing)
    spacing = autocorr_length(field)
    n_periods = (coherence_length / spacing) if (spacing and not np.isnan(spacing)
                                                 and not np.isnan(coherence_length)) else np.nan
    return dict(coherence_length=coherence_length, n_coherent_periods=n_periods)


# ----------------------------------------------------------------------
# 3. FEATURE SEGMENTATION  -> feature size, inter-feature distance, density, shape
# ----------------------------------------------------------------------
def _choose_threshold(field):
    """Pick a black/white cutoff. Prefer Otsu (automatic, robust); fall back to mean+0.5 std."""
    try:
        from skimage.filters import threshold_otsu
        return threshold_otsu(field)
    except Exception:
        return np.mean(field) + 0.5 * np.std(field)

def feature_metrics(field):
    """Threshold into features (spots/stripes) and measure them.
    Returns feature size, inter-feature (nearest-neighbor centroid) distance, density,
    spacing regularity, and a MORPHOLOGY metric (eccentricity) for spots-vs-stripes-vs-maze."""
    thr = _choose_threshold(field)
    mask = field > thr
    labels, n = ndimage.label(mask)              # connected components = features
    if n == 0:
        return dict(n_features=0, feature_size=np.nan, interfeature_dist=np.nan,
                    density=0.0, spacing_regularity=np.nan,
                    eccentricity=np.nan, morphology="none")
    sizes = ndimage.sum(np.ones_like(field), labels, index=range(1, n+1))
    feature_size = float(np.median(sizes))
    centroids = np.array(ndimage.center_of_mass(np.ones_like(field), labels, index=range(1, n+1)))
    density = n / field.size
    # nearest-neighbor distance between feature centroids
    if n >= 2:
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        dists, _ = tree.query(centroids, k=2)    # k=2: self (0) + nearest neighbor
        nn = dists[:, 1]
        interfeature_dist = float(np.median(nn))
        spacing_regularity = float(np.std(nn) / np.mean(nn))  # low = regular lattice, high = disordered
    else:
        interfeature_dist = np.nan
        spacing_regularity = np.nan

    # --- MORPHOLOGY: spots vs stripes vs maze ---
    # eccentricity per feature: 0 = round (spot), ->1 = elongated (stripe/maze).
    # Prefer skimage.regionprops (gives it directly); fall back to a compactness estimate.
    ecc = _eccentricity(labels, n, sizes)
    if np.isnan(ecc):
        morphology = "unknown"
    elif ecc < 0.7:
        morphology = "spots"          # round, well-separated features
    elif n <= 3:
        morphology = "maze/stripes"   # few large elongated components = labyrinth or stripes
    else:
        morphology = "elongated"      # many elongated features
    return dict(n_features=n, feature_size=feature_size, interfeature_dist=interfeature_dist,
                density=density, spacing_regularity=spacing_regularity,
                eccentricity=ecc, morphology=morphology)

def _eccentricity(labels, n, sizes):
    """Median feature eccentricity. 0 = circular (spots), near 1 = elongated (stripes/maze)."""
    try:
        from skimage.measure import regionprops
        props = regionprops(labels)
        eccs = [p.eccentricity for p in props if p.area >= 3]  # ignore tiny specks
        return float(np.median(eccs)) if eccs else np.nan
    except Exception:
        # fallback: compactness on the largest few features. compactness=1 is round; lower = elongated.
        big = np.argsort(sizes)[::-1][:max(1, n // 5)] + 1
        vals = []
        for lab in big:
            area = sizes[lab - 1]
            perim = _perimeter(labels == lab)
            if perim > 0:
                compactness = 4 * np.pi * area / (perim ** 2)              # 1 round, <1 elongated
                vals.append(np.sqrt(max(0.0, 1.0 - min(1.0, compactness))))  # -> eccentricity-like
        return float(np.median(vals)) if vals else np.nan

def _perimeter(binary_feature):
    """Crude perimeter: count of feature pixels adjacent to background."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(binary_feature)
    return int(np.sum(binary_feature & ~eroded))


# ----------------------------------------------------------------------
# 4. A-B CROSS-CORRELATION  (activator vs inhibitor: co- or anti-located)
# ----------------------------------------------------------------------
def ab_crosscorrelation(field_A, field_B):
    """Zero-lag spatial Pearson correlation. +1 co-located, -1 anti-located, ~0 unrelated."""
    a = (field_A - field_A.mean()).ravel()
    b = (field_B - field_B.mean()).ravel()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------
# top-level: full descriptor for one run
# ----------------------------------------------------------------------
def describe_pattern(field_A, field_B=None, rel_std_threshold=0.05):
    if is_flat_or_noise(field_A, rel_std_threshold):
        return dict(regime="flat_or_noise")
    d = {}
    d.update(structure_factor_metrics(field_A))
    d["autocorr_length"] = autocorr_length(field_A)
    d.update(autocorr_coherence(field_A))
    d.update(feature_metrics(field_A))
    if field_B is not None:
        d["AB_xcorr"] = ab_crosscorrelation(field_A, field_B)
    return d


if __name__ == "__main__":
    # tiny self-test on synthetic fields so they can see it runs and the numbers make sense
    ny = nx = 128
    yy, xx = np.mgrid[0:ny, 0:nx]
    spots   = np.sin(2*np.pi*xx/16) * np.sin(2*np.pi*yy/16)          # regular, spacing ~16
    noise   = np.random.default_rng(0).normal(0.028, 0.001, (ny,nx))# low-variance noise
    stripes = np.sin(2*np.pi*xx/20)                                  # stripes, spacing ~20
    for name, f in [("regular spots", spots), ("noise", noise), ("stripes", stripes)]:
        print(name, "->")
        for k_, v_ in describe_pattern(f - f.min()).items():
            print(f"    {k_}: {v_}")
        print()

describe_pattern.__doc__ = """
Top-level function to extract a comprehensive set of pattern descriptors from a 2D field.
Returns a dictionary of metrics including:
- regime: "flat_or_noise" if the field is essentially structureless; otherwise not set
- spacing: characteristic pattern spacing from the structure factor (box units)
- peak_width: width of the structure factor peak (sharp = regular, broad = irregular)
- prominence: height of the structure factor peak above background (pattern vs noise)
- k_peak: wavenumber of the structure factor peak (cycles per box)
- autocorr_length: characteristic length from the first peak of the spatial autocorrelation
- coherence_length: how many pixels before the autocorrelation envelope decays (order/coherence)
- n_coherent_periods: coherence_length expressed in units of the pattern spacing
- n_features: number of segmented features (spots/stripes)
- feature_size: median size of the segmented features (pixels)
- interfeature_dist: median nearest-neighbor distance between feature centroids (pixels)
- density: number of features per pixel
- spacing_regularity: coefficient of variation of nearest-neighbor distances (low = regular lattice)
- eccentricity: median feature eccentricity (0 = round spots, near 1 = elongated stripes/maze)
- morphology: qualitative classification of feature shape ("spots", "maze/stripes", "elongated", "none", "unknown")
- AB_xcorr: if field_B is provided, the zero-lag spatial Pearson correlation between field_A and field_B (+1 co-located, -1 anti-located)
"""
