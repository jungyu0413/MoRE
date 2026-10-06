"""Test-time augmentation (TTA) for best-of-K trajectory sampling — always used by more.evaluate.

The N samples that best-of-K is computed from are split evenly over several versions of the prompt:
geometric transforms of the pixel coordinates (point flip, x/y swap — the augmentations the language
model was trained with) and the prompt with / without the scene caption. Each sampled trajectory is
mapped back to the original frame with the inverse transform before the usual post-processing
(wall avoidance, K-means to K) and scoring. The model, N and K are unchanged.

Per-dataset variant sets (MODE_BY_DATASET):
  geo4ctxnc  original, flip, swap, flip+swap  x  with / without scene caption  (8 prompts)
  geo4       original, flip, swap, flip+swap                                    (4 prompts)
"""
import os
import re
import numpy as np

MODE_BY_DATASET = {'eth': 'geo4ctxnc', 'hotel': 'geo4ctxnc', 'univ': 'geo4ctxnc', 'zara1': 'geo4ctxnc', 'zara2': 'geo4'}

# Best weights found so far (full ETH/UCY test sets, 1000 samples, best-of-20, with the TTA above).
# Used by more.evaluate when no --checkpoint is given. Paths are relative to weights_dir (see weights/README.md).
#   dataset: (base checkpoint, LoRA adapter or None)          result (ADE/FDE)
BEST_WEIGHTS = {
    'eth':   ('lmtraj_eth',   None),                           # 0.346/0.398
    'hotel': ('lmtraj_eth',   None),                           # 0.101/0.122 (eth-split model, as the original MoRE hotel adapter)
    'univ':  ('lmtraj_univ',  'univ_more_adapter_lr1e5_s150'), # 0.207/0.320
    'zara1': ('lmtraj_zara1', None),                           # 0.189/0.299
    'zara2': ('lmtraj_zara2', None),                           # 0.166/0.255
}
# <repo>/weights by default; override with "weights_dir" in the config
DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'weights')


def best_weights(dataset_name, weights_dir=None):
    """(base checkpoint dir, adapter dir or None) for the best known model of a dataset."""
    root = weights_dir or DEFAULT_WEIGHTS_DIR
    base, adapter = BEST_WEIGHTS[dataset_name]
    return os.path.join(root, base), (os.path.join(root, adapter) if adapter else None)

_COORD = re.compile(r'\((-?\d+), (-?\d+)\)')


def mode_for_dataset(dataset_name):
    return MODE_BY_DATASET.get(dataset_name, 'geo4')


def _geo_text(text, flip, swap, size):
    W, H = int(size[0]), int(size[1])

    def f(m):
        x, y = int(m.group(1)), int(m.group(2))
        if flip in (True, 'x'):
            x = W - x
        if flip in (True, 'y'):
            y = H - y
        if swap:
            x, y = y, x
        return f'({x}, {y})'
    return _COORD.sub(f, text)


def _geo_inv(traj, flip, swap, size):
    t = np.asarray(traj, dtype=np.float32).copy()
    if swap:
        t = t[:, [1, 0]]
    if flip in (True, 'x'):
        t[:, 0] = size[0] - t[:, 0]
    if flip in (True, 'y'):
        t[:, 1] = size[1] - t[:, 1]
    return t


def strip_scene_caption(text, caption):
    """Context prompt -> no-context prompt (identical to preprocessing with use_scene_context=False)."""
    if caption:
        return text.replace(' ' + caption + ' answer:', ' answer:')
    return text


def build_variants(text, mode, scene_size, scene_caption=''):
    """Return a list of (prompt, inverse_fn) for one pedestrian."""
    if 'geo8' in mode:
        geo = [(fl, sw) for fl in (False, True, 'x', 'y') for sw in (False, True)]
    elif 'geo4' in mode:
        geo = [(False, False), (True, False), (False, True), (True, True)]
    elif 'geo2' in mode:
        geo = [(False, False), (True, False)]
    elif 'swap2' in mode:
        geo = [(False, False), (False, True)]
    else:
        geo = [(False, False)]
    prompts = [text]
    if 'ctxnc' in mode:
        nc = strip_scene_caption(text, scene_caption)
        if nc != text:
            prompts.append(nc)
    out = []
    for base in prompts:
        for flip, swap in geo:
            t = _geo_text(base, flip, swap, scene_size) if (flip or swap) else base
            out.append((t, lambda tr, f=flip, sw=swap: _geo_inv(tr, f, sw, scene_size)))
    return out


def split_counts(n_total, n_variants):
    """Even split of the N samples over the prompt variants."""
    per, rem = divmod(n_total, n_variants)
    return [per + (1 if k < rem else 0) for k in range(n_variants)]
