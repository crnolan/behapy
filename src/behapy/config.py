import json
from .pathutils import preprocess_config_path


def load_preprocess_config(bidsroot):
    with open(preprocess_config_path(bidsroot), 'r') as f:
        config = json.load(f)
    if 'method' not in config:
        config['method'] = 'detrend'
    if 'detrend_cutoff' not in config:
        config['detrend_cutoff'] = 0.05
    return config
