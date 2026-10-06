"""Configuration management with dot-notation access."""

import json


class DotDict(dict):
    """Dictionary subclass enabling dot-notation access to nested keys."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def _recursive_dotdict(obj):
    """Recursively convert dicts to DotDict."""
    if isinstance(obj, dict):
        return DotDict({k: _recursive_dotdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_recursive_dotdict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_recursive_dotdict(item) for item in obj)
    return obj


def get_exp_config(config_file: str) -> DotDict:
    """Load a JSON config file and return a DotDict."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return _recursive_dotdict(config)
