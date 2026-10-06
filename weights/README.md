# Evaluation weights (`weights/`)

`more.evaluate` uses these when no `--checkpoint` is given (table `BEST_WEIGHTS` in
`more/inference/tta.py`). They are not stored in git (≈800 MB); place them here or set `weights_dir`
in the config.

```
weights/
├── lmtraj_eth/                      # LMTraj-SUP, ETH split   (also used for HOTEL)
├── lmtraj_univ/                     # LMTraj-SUP, UNIV split
├── lmtraj_zara1/                    # LMTraj-SUP, ZARA1 split
├── lmtraj_zara2/                    # LMTraj-SUP, ZARA2 split
└── univ_more_adapter_lr1e5_s150/    # MoRE LoRA adapter for UNIV (applied on top of lmtraj_univ)
```

- `lmtraj_*`: the pretrained LMTraj-SUP models (pixel, multimodal) released with
  [LMTrajectory](https://github.com/InhwanBae/LMTrajectory/releases/tag/v1.0).
- `univ_more_adapter_lr1e5_s150`: MoRE refinement of `lmtraj_univ` (LoRA r=16, 150 steps, lr 1e-5).

Results with these weights and the built-in test-time augmentation
(full ETH/UCY test sets, 1000 samples, best-of-20, ADE/FDE in metres):

| ETH | HOTEL | UNIV | ZARA1 | ZARA2 | AVG |
|---|---|---|---|---|---|
| 0.346/0.398 | 0.101/0.122 | 0.207/0.320 | 0.189/0.299 | 0.166/0.255 | 0.202/0.279 |

HOTEL is evaluated with the ETH-split model, as in the original setup; that model has seen the HOTEL scene
during training, so HOTEL is not a strict leave-one-out number.
