#!/usr/bin/env python3
"""
Batch compute connected-component size distributions on a hexagonal grid.

For every .npz file in an input folder:
- load a 2D array (default field: A_final)
- threshold it into a binary mask
- find connected components using hex-grid connectivity
- write one CSV per input file into input_folder/CSV/

Outputs per component:
- component_id
- size_hexagons
- approx_diameter_hexagons
- row_min, row_max, col_min, col_max
- bbox_height_hexes, bbox_width_hexes
- centroid_row, centroid_col

Default layout:
- even-r (row-offset, horizontal hex layout)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Area of a unit hexagon with side length = 1
HEX_AREA = 3.0 * math.sqrt(3.0) / 2.0


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Compute Otsu's threshold for a 1D or 2D numeric array."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("No finite values found for Otsu thresholding.")

    if np.allclose(x.min(), x.max()):
        return float(x.min())

    hist, bin_edges = np.histogram(x, bins=bins)
    hist = hist.astype(float)
    prob = hist / hist.sum()
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros_like(centers)
    valid = denom > 0
    sigma_b2[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]

    idx = int(np.argmax(sigma_b2))
    return float(centers[idx])


def get_neighbors(r: int, c: int, layout: str) -> List[Tuple[int, int]]:
    """Return the 6 neighbor coordinates for a cell in an offset hex grid."""
    if layout == "even-r":
        # Horizontal layout, even rows shifted right
        if r % 2 == 0:
            return [
                (r - 1, c),
                (r - 1, c - 1),
                (r, c - 1),
                (r, c + 1),
                (r + 1, c),
                (r + 1, c - 1),
            ]
        return [
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c),
            (r + 1, c + 1),
        ]

    if layout == "odd-r":
        # Horizontal layout, odd rows shifted right
        if r % 2 == 1:
            return [
                (r - 1, c),
                (r - 1, c - 1),
                (r, c - 1),
                (r, c + 1),
                (r + 1, c),
                (r + 1, c - 1),
            ]
        return [
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c),
            (r + 1, c + 1),
        ]

    if layout == "even-q":
        # Vertical layout, even columns shifted down
        if c % 2 == 0:
            return [
                (r - 1, c),
                (r - 1, c + 1),
                (r, c - 1),
                (r, c + 1),
                (r + 1, c),
                (r + 1, c + 1),
            ]
        return [
            (r - 1, c - 1),
            (r - 1, c),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
        ]

    if layout == "odd-q":
        # Vertical layout, odd columns shifted down
        if c % 2 == 1:
            return [
                (r - 1, c),
                (r - 1, c + 1),
                (r, c - 1),
                (r, c + 1),
                (r + 1, c),
                (r + 1, c + 1),
            ]
        return [
            (r - 1, c - 1),
            (r - 1, c),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
        ]

    raise ValueError(f"Unknown layout: {layout}")


def component_mask(mask: np.ndarray, layout: str) -> List[List[Tuple[int, int]]]:
    """Find connected components using 6-neighbor hex connectivity."""
    if mask.ndim != 2:
        raise ValueError("Expected a 2D array.")

    nrows, ncols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    for r in range(nrows):
        for c in range(ncols):
            if not mask[r, c] or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells: List[Tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))

                for nr, nc in get_neighbors(cr, cc, layout):
                    if (
                        0 <= nr < nrows
                        and 0 <= nc < ncols
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))

            components.append(cells)

    return components


def approx_diameter_hexagons(size_hexes: int) -> float:
    """
    Area-equivalent diameter expressed in hexagon units.

    A component with `size_hexes` unit hexagons has total area:
        size_hexes * HEX_AREA

    We return the diameter of a circle with the same area, divided by the
    characteristic hex scale. This is a useful approximate size measure.
    """
    area = size_hexes * HEX_AREA
    return 2.0 * math.sqrt(area / math.pi)


def process_file(
    npz_path: Path,
    out_csv: Path,
    field: str = "A_final",
    threshold: str = "auto",
    layout: str = "even-r",
    min_size: int = 2,
) -> None:
    """Process one .npz file and write a CSV of connected components."""
    data = np.load(npz_path, allow_pickle=True)

    if field not in data:
        raise KeyError(f"Field '{field}' not found in {npz_path}. Available keys: {list(data.files)}")

    arr = np.asarray(data[field], dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array for field '{field}', got shape {arr.shape}")

    if threshold == "auto":
        thr = otsu_threshold(arr)
    else:
        thr = float(threshold)

    mask = np.isfinite(arr) & (arr > thr)
    components = component_mask(mask, layout)
    components = [comp for comp in components if len(comp) >= min_size]

    rows = []
    for i, cells in enumerate(components, start=1):
        rr = np.array([r for r, _ in cells], dtype=int)
        cc = np.array([c for _, c in cells], dtype=int)
        size = len(cells)

        rows.append(
            {
                "component_id": i,
                "size_hexagons": size,
                "approx_diameter_hexagons": approx_diameter_hexagons(size),
                "row_min": int(rr.min()),
                "row_max": int(rr.max()),
                "col_min": int(cc.min()),
                "col_max": int(cc.max()),
                "bbox_height_hexes": int(rr.max() - rr.min() + 1),
                "bbox_width_hexes": int(cc.max() - cc.min() + 1),
                "centroid_row": float(rr.mean()),
                "centroid_col": float(cc.mean()),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            # Write a header even if nothing was found
            writer = csv.writer(f)
            writer.writerow(
                [
                    "component_id",
                    "size_hexagons",
                    "approx_diameter_hexagons",
                    "row_min",
                    "row_max",
                    "col_min",
                    "col_max",
                    "bbox_height_hexes",
                    "bbox_width_hexes",
                    "centroid_row",
                    "centroid_col",
                ]
            )

    print(f"Processed {npz_path.name} -> {out_csv.name} | components: {len(rows)} | threshold: {thr:.6g}")


def process_folder(
    input_folder: str | Path,
    field: str = "A_final",
    threshold: str = "auto",
    layout: str = "even-r",
    min_size: int = 2,
) -> None:
    """Process all .npz files in a folder and write CSVs into input_folder/CSV/."""
    input_folder = Path(input_folder)
    output_folder = input_folder / "CSV"
    output_folder.mkdir(exist_ok=True)

    npz_files = sorted(p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() == ".npz")

    if not npz_files:
        print(f"No .npz files found in {input_folder}")
        return

    for npz_path in npz_files:
        out_csv = output_folder / f"{npz_path.stem}.csv"
        process_file(
            npz_path=npz_path,
            out_csv=out_csv,
            field=field,
            threshold=threshold,
            layout=layout,
            min_size=min_size,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch compute connected-component size distributions on a hex grid from .npz files."
    )
    parser.add_argument("input_folder", help="Folder containing .npz files")
    parser.add_argument("--field", default="A_final", help="Array name inside each .npz file")
    parser.add_argument(
        "--threshold",
        default="auto",
        help="Threshold for feature segmentation. Use a number, or 'auto' for Otsu.",
    )
    parser.add_argument(
        "--layout",
        default="even-r",
        choices=["even-r", "odd-r", "even-q", "odd-q"],
        help="Hex-grid offset layout",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Discard components smaller than this many hexagons",
    )
    args = parser.parse_args()

    process_folder(
        input_folder=args.input_folder,
        field=args.field,
        threshold=args.threshold,
        layout=args.layout,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()
