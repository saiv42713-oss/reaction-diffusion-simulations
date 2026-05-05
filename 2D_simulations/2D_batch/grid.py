import numpy as np
from itertools import product
from typing import Dict, Iterable, List, Tuple, Union, Any
import copy

ArrayLike = Union[Iterable[float], np.ndarray, Tuple[float, float, int]]


def _to_values(val: Union[ArrayLike, Dict[str, Tuple[float, float, int]], List[Any]]):
    """
    Accepted forms:
      - [v1, v2, ...]                   -> explicit values
      - [start, stop, num]              -> linspace
      - {lin: [start, stop, num]}       -> linspace
      - {log: [exp_start, exp_stop, n]}  -> logspace (10**exp)
      - list of any of the above        -> concatenate segments
    """
    if isinstance(val, dict):
        if "lin" in val:
            a, b, n = val["lin"]
            return np.linspace(a, b, int(n))
        if "log" in val:
            a, b, n = val["log"]
            return np.logspace(a, b, int(n))
        raise ValueError(f"Unknown sweep dict: {val}")

    if isinstance(val, (list, tuple, np.ndarray)):
        # list-of-specs
        if any(isinstance(x, (list, tuple, dict, np.ndarray)) for x in val):
            parts = [_to_values(x) for x in val]
            return np.concatenate([np.asarray(p, dtype=float) for p in parts])

        # 3-number shorthand => linspace
        if len(val) == 3 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in val):
            a, b, n = val
            return np.linspace(a, b, int(n))

        # explicit values
        return np.asarray(val, dtype=float)

    return np.asarray(list(val), dtype=float)


def _is_sweep_leaf(x):
    """
    A leaf sweep is something that can be converted to values directly.
    A nested sweep is a dict whose values are other sweep specs.
    """
    return not isinstance(x, dict) or ("lin" in x) or ("log" in x)


def _expand_nested_sweeps(prefix, sweep_dict):
    """
    Convert nested sweep structure into a list of (path, values).

    Example:
      {"p1": {"inh_prod_rate": [2, 5, 10]}}
    becomes:
      [(("p1", "inh_prod_rate"), array([2,5,10]))]
    """
    items = []
    for k, v in sweep_dict.items():
        new_prefix = prefix + (k,)
        if isinstance(v, dict) and not ("lin" in v or "log" in v):
            items.extend(_expand_nested_sweeps(new_prefix, v))
        else:
            items.append((new_prefix, _to_values(v)))
    return items


def _set_nested_value(d, path, value):
    """
    Set d[path[0]][path[1]]...[path[-1]] = value
    """
    cur = d
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = float(value)


def make_param_grid(
    base: Dict,
    sweeps: Dict[str, Any],
    mode: str = "grid",
) -> List[Dict]:
    if not sweeps:
        return [copy.deepcopy(base)]

    sweep_items = []
    for top_key, spec in sweeps.items():
        if isinstance(spec, dict) and not ("lin" in spec or "log" in spec):
            sweep_items.extend(_expand_nested_sweeps((top_key,), spec))
        else:
            sweep_items.append(((top_key,), _to_values(spec)))

    paths = [p for p, _ in sweep_items]
    vals = [v for _, v in sweep_items]

    param_list: List[Dict] = []

    if mode == "grid":
        for combo in product(*vals):
            p = copy.deepcopy(base)
            for path, v in zip(paths, combo):
                _set_nested_value(p, path, v)
            param_list.append(p)

    elif mode == "zip":
        lengths = [len(v) for v in vals]
        if len(set(lengths)) != 1:
            raise ValueError(f"'zip' mode requires equal lengths, got {lengths}")
        for i in range(lengths[0]):
            p = copy.deepcopy(base)
            for path, vseq in zip(paths, vals):
                _set_nested_value(p, path, vseq[i])
            param_list.append(p)

    else:
        raise ValueError("mode must be 'grid' or 'zip'")

    return param_list
