import json
import numpy as np
from typing import Dict, Any

def _to_json_list(x: Any):
    """Serialize lists/ndarrays to a compact JSON string for safe CSV storage."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return json.dumps([float(v) for v in x])
    return x

def write_constants_txt(constants: Dict[str, Any], path: str):
    with open(path, "w") as f:
        for k, v in constants.items():
            f.write(f"{k}\t{v}\n")
