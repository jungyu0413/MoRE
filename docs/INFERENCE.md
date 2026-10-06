# Inference: test-time augmentation (TTA)

Implemented in `more/inference/tta.py` and called from `more/evaluate.py`; applied to every stochastic
evaluation. Deterministic (beam-search) evaluation is unchanged.

For each test pedestrian the N samples are split evenly over a fixed set of prompt variants:

| dataset | variants | samples per variant (N = 1000) |
|---|---|---|
| eth, hotel, univ, zara1 | {original, point flip, x/y swap, flip + swap} × {with, without scene caption} | 125 |
| zara2 | {original, point flip, x/y swap, flip + swap} | 250 |

- Point flip maps pixel (x, y) to (W − x, H − y) and swap to (y, x), where W × H is the quarter-resolution
  scene image size (the pixel frame of the prompts). These are the geometric augmentations LMTraj is
  trained with. Each decoded trajectory is mapped back with the inverse transform.
- The no-caption prompt is built by removing the scene caption from the original prompt; this is
  identical to preprocessing with `use_scene_context=False`, so no extra data files are needed.
- The pooled N samples then go through the original post-processing (abnormal-motion filter, wall
  avoidance, K-means to K) and best-of-K scoring.

Effect on the same models (full test sets, 1000 samples, best-of-20, ADE/FDE):

| | ETH | HOTEL | UNIV | ZARA1 | ZARA2 |
|---|---|---|---|---|---|
| without TTA | 0.415/0.503 | 0.106/0.133 | 0.213/0.332* | 0.199/0.324 | 0.174/0.271 |
| with TTA | 0.346/0.398 | 0.101/0.122 | 0.207/0.320* | 0.189/0.299 | 0.166/0.255 |

\* UNIV with the MoRE LoRA adapter. All differences are significant in a paired per-pedestrian comparison.
