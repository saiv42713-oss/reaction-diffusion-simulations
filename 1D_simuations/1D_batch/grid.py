import numpy as np
from itertools import product
from typing import Dict, Iterable, List, Tuple, Union, Any

ArrayLike = Union[Iterable[float], np.ndarray, Tuple[float, float, int]]

def _to_values(val: Union[ArrayLike, Dict[str, Tuple[float, float, int]], List[Any]]):
    """
    Accepted YAML forms:
      - [v1, v2, ...]                   -> explicit values
      - [start, stop, num]              -> linspace
      - {lin: [start, stop, num]}       -> linspace
      - {log: [exp_start, exp_stop, n]} -> logspace (10**exp)
      - list of any of the above        -> concatenate segments
    """
    # dict forms
    if isinstance(val, dict):
        if "lin" in val:
            a, b, n = val["lin"]
            return np.linspace(a, b, int(n))
        if "log" in val:
            a, b, n = val["log"]
            return np.logspace(a, b, int(n))
        raise ValueError(f"Unknown sweep dict: {val}")

    # list/tuple
    if isinstance(val, (list, tuple)):
        # Concatenate if it's a list-of-specs (e.g., [[...], {...}, ...])
        if any(isinstance(x, (list, tuple, dict)) for x in val):
            parts = [_to_values(x) for x in val]
            # optional: drop duplicate touching endpoints between segments
            out = np.concatenate(parts)
            return out
        # A single 3-item numeric spec => linspace
        if len(val) == 3 and all(isinstance(x, (int, float)) for x in val):
            a, b, n = val
            return np.linspace(a, b, int(n))
        # Otherwise treat as explicit values
        return np.array(list(val), dtype=float)

    # Fallback: explicit iterable
    return np.array(list(val), dtype=float)

def make_param_grid(
    base: Dict,
    sweeps: Dict[str, Union[ArrayLike, Dict[str, Tuple[float, float, int]]]],
    mode: str = "grid",
) -> List[Dict]:
    if not sweeps:
        return [base.copy()]

    keys = list(sweeps.keys())
    vals = [_to_values(sweeps[k]) for k in keys]
    param_list: List[Dict] = []

    if mode == "grid":
        for combo in product(*vals):
            p = base.copy()
            for k, v in zip(keys, combo):
                p[k] = float(v)
            param_list.append(p)
    elif mode == "zip":
        lengths = [len(v) for v in vals]
        if len(set(lengths)) != 1:
            raise ValueError(f"'zip' mode requires equal lengths, got {lengths}")
        for i in range(lengths[0]):
            p = base.copy()
            for k, vseq in zip(keys, vals):
                p[k] = float(vseq[i])
            param_list.append(p)
    else:
        raise ValueError("mode must be 'grid' or 'zip'")

    return param_list
