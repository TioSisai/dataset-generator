import json
import numpy as np

def convert_for_json(x):
    """Convert numpy arrays and types to JSON-serializable format"""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.int64, np.int32)):
        return int(x)
    if isinstance(x, (np.float64, np.float32)):
        return float(x)
    if isinstance(x, list):
        return [convert_for_json(item) for item in x]
    return x