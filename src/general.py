import json
import os
import logging
from types import SimpleNamespace
from targets import TargetPath, targets

# Local Storage
@targets(__file__, ensure=True)
class Storage:
    home = TargetPath('..')
    local = TargetPath('../local')
    logs = TargetPath('../local/logs')
    data = TargetPath('../local/data')
    models = TargetPath('../local/models')
    stats = TargetPath('../local/stats')
    configs = TargetPath('../local/configs')
    shared_data = TargetPath('../data')

class DataFiles(SimpleNamespace):
    raw_velib = Storage.data('raw_velib.h5')
    raw_meteofrance = Storage.data('raw_meteofrance.h5')
    processed = Storage.data('processed.h5')
    processed_7d = Storage.data('processed_7d.h5')
    processed_stations = Storage.data('processed_stations.h5')
    train = Storage.data('train.h5')
    test = Storage.data('test.h5')
    mlp_grid_default_config = Storage.shared_data('mlp_grid_params.default.json')
    mlp_grid_config = Storage.configs('mlp_grid_params.json')
    mlp_grid_params_list = Storage.configs('mlp_grid_params.txt')
    default_transformer_config = Storage.shared_data('transformer_params.default.json')
    transformer_config = Storage.configs('transfromer_params.json')
    transformer = Storage.models('transformer.joblib')
    default_best_params = Storage.shared_data('best_params.default.json')
    best_params = Storage.configs('best_params.json')

LOGGER = None

def log(*args, sep=' ', end='\n'):
    global LOGGER
    if LOGGER is None:
        LOGGER = logging.getLogger('VelibDS')
        if not LOGGER.handlers:
            # Add console logger only if no output is already setup to avoid duplicated logs
            if not LOGGER.hasHandlers():
                LOGGER.setLevel(logging.DEBUG)
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                LOGGER.addHandler(console_handler)
            # Add file logger only if VelibDS loggers are not present
            try:
                file_handler = logging.FileHandler(Storage.logs('velibds.log'), mode='a')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                LOGGER.addHandler(file_handler)
            except:
                LOGGER.error('Unable to add File Logger for local/logs/VelibData.log.')
    msg_list = [str(a) for a in args]
    LOGGER.info(sep.join(msg_list))

def load_transformer_params() -> dict:
    """Load parameters for transformer"""
    # Copy default params if no params exist
    if not os.path.isfile(DataFiles.transformer_config):
        with open(DataFiles.default_transformer_config) as src:
            with open(DataFiles.transformer_config, 'w') as tgt:
                tgt.write(src.read())
    # Load params from file
    with open(DataFiles.transformer_config) as f:
        return json.load(f)
    
def load_best_params() -> dict:
    """Load best fitted parameters for MLPFlow"""
    if not os.path.isfile(DataFiles.best_params):
        with open(DataFiles.default_best_params) as src:
            with open(DataFiles.best_params, 'w') as tgt:
                tgt.write(src.read())
    # Load params from file
    with open(DataFiles.best_params) as f:
        return json.load(f)
