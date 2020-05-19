#coding: utf-8
import os
import yaml
import json
from . import PARAMS_DIR_PATH

def load_params(path=None, name="chemvae"):
    # Load your parameter files.
    if path is not None:
        ext = os.path.splitext(path)[1]
        with open(path, mode="r") as f:
            if ext == ".json":
                params = json.load(f)
            elif ext == ".yml":
                params = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        params = {}

    # Rest of parameters are set as default.
    default_path = os.path.join(PARAMS_DIR_PATH, name+".yml")
    if os.path.exists(default_path):
        with open(default_path, mode="r") as f:
            default_params =yaml.load(f, Loader=yaml.SafeLoader)
    else:
        default_params = {}

    default_params.update(params)
    return default_params
