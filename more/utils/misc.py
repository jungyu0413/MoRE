"""Small shared helpers for reproducibility and JSON serialization."""

import random

import numpy as np
import torch


def reproducibility_settings(seed: int = 0):
    """Fix all random seeds and disable non-deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def round_floats(o, ndigits: int = 6):
    """Recursively round every float in a nested structure, for compact JSON."""
    if isinstance(o, float):
        return round(o, ndigits)
    if isinstance(o, dict):
        return {k: round_floats(v, ndigits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x, ndigits) for x in o]
    return o
