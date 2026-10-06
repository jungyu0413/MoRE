# Pretrained weights (`weights/`)

Pretrained weights will be released on the [release page](https://github.com/jungyu0413/MoRE/releases).
Download them and extract them into this folder (or set `weights_dir` in `configs/default.json`).

`more.evaluate` loads the weights for the requested dataset from here when no `--checkpoint` is given.
The folder each dataset uses is listed in `BEST_WEIGHTS` in [`more/inference/tta.py`](../more/inference/tta.py);
a dataset entry may combine a base checkpoint with a LoRA adapter, which is merged before evaluation.
