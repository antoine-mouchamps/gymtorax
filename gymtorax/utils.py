import json
import os

import numpy as np


def load_json_file(filename: str) -> dict:  # noqa: D103
    """Load a JSON file and return its contents as a dictionary.

    Args:
        filename (str): The name of the JSON file to load.

    Returns:
        dict: The contents of the JSON file.
    """
    json_path = os.path.join(os.path.dirname(__file__), filename)
    with open(json_path, "r") as f:  # noqa: UP015
        bounds = json.load(f)
    # Convert "inf" and "-inf" strings to np.inf and -np.inf
    for cat in bounds:
        for var, props in bounds[cat].items():
            for key in ["min", "max"]:
                if props[key] == "inf":
                    props[key] = np.inf
                elif props[key] == "-inf":
                    props[key] = -np.inf
    return bounds
