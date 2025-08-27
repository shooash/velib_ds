import os
import json
from itertools import product
from .general import DataFiles, log
from pathlib import Path

def load_grid_params() -> dict:
    """Load parameters for grid training"""
    # Copy default params if no params exist
    if not os.path.isfile(DataFiles.mlp_grid_config):
        with open(DataFiles.mlp_grid_default_config) as src:
            with open(DataFiles.mlp_grid_config, 'w') as tgt:
                tgt.write(src.read())
    # Load params from file
    with open(DataFiles.mlp_grid_config) as f:
        return json.load(f)
    

def gen_grid_params(params : list[dict]):
    result = []
    for p in params:
        result.append([dict(zip(p.keys, combo)) for combo in product(*p.values())])
    return result

def replace_links(params : list[dict]):
    '''Replace values starting with "@" by a linked value.
    E.g. {"wh" : 3, "wl" : "@wh"} => {"wh" : 3, "wl" : 3}
    '''
    for p in params:
        for k in p.keys():
            if isinstance(p[k], str) and p[k].startswith('@'):
                p[k] = p[p[k][1:]]

def get_complete_params():
    Path(DataFiles.mlp_grid_params_list).touch()
    with open(DataFiles.mlp_grid_params_list) as f:
        params_done = f.readlines()
    log(f'Loaded {len(params_done)} params marked as done.')
    return params_done

def run_grid():
    params = load_grid_params()
    params = gen_grid_params(params)
    params = replace_links(params)
    params_done = get_complete_params()
    params = [p for p in params if str(p) + '\n' not in params_done]
    log(f"There're {len(params)} params to process.")
    if not len(params):
        return(0)
    log('Loading MLPFlow')
    mlp = MLPFlow()