""" (Very) hacky way of reading, writing Goodfire controller params by reading / writing to a file 

NOTE: this means that you should not invoke multiple controllers simultaneously, as this will lead to race conditions.
"""

import json

from functools import lru_cache
from pathlib import Path
from typing import Any
from goodfire.controller.controller import Controller

curr_dir = Path(__file__).parent


@lru_cache(maxsize=1)
def _get_default_controller_params():
    return Controller().json()

def read_controller_params():
    if not (curr_dir / "controller.json").exists():
        write_controller_params(_get_default_controller_params())
    with open(curr_dir / "controller.json", "r") as f:
        return json.load(f)

def write_controller_params(
    controller_params: dict[str, Any], 
    path: Path | None = None,
):
    if path is None:
        path = curr_dir / "controller.json"
    with open(path, "w") as f:
        json.dump(controller_params, f)

def reset_controller_params():
    write_controller_params(_get_default_controller_params())
