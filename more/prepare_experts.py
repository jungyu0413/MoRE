"""Precompute expert model predictions and cache to disk.

This must be run before training to generate expert prediction files.

Usage:
    python -m more.precompute_experts --config_file configs/default.json --phase train
"""

import argparse
import os
import logging

import torch
from datasets import load_dataset

from more.utils.config import get_exp_config
from more.utils.expert_predict import precompute_expert_predictions
from more.experts.registry import load_all_experts

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute expert predictions")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--phase", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Override dataset_name from the config file (eth/hotel/univ/zara1/zara2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = get_exp_config(args.config_file)
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load expert models
    experts, extra = load_all_experts(cfg, device)
    logger.info(f"Loaded experts: {list(experts.keys())}")

    # Load dataset
    preprocessed_path = os.path.join(cfg.dataset_path, "preprocessed")
    use_scene_context = getattr(cfg, 'use_scene_context', True)
    suffix = "multimodal" if use_scene_context else "multimodal-nocontext"

    if args.phase == "train":
        fname = f"{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-{suffix}.json"
    else:
        fname = f"{cfg.dataset_name}-{args.phase}-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"

    fpath = os.path.join(preprocessed_path, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Dataset not found: {fpath}")

    raw_dataset = load_dataset("json", data_files={args.phase: fpath}, cache_dir=cfg.cache_dir)

    hyper_params = extra.get('singulartrajectory', (None,))[0] if 'singulartrajectory' in extra else None

    # Build raw_datasets dict expected by precompute function
    # Map 'ped_id' -> 'original_ped_id' if needed
    ds = raw_dataset[args.phase]
    if 'original_ped_id' not in ds.column_names and 'ped_id' in ds.column_names:
        ds = ds.rename_column('ped_id', 'original_ped_id')
    raw_datasets = {args.phase: ds}

    # Create a minimal accelerator for the precompute function
    from accelerate import Accelerator
    accelerator = Accelerator()

    precompute_expert_predictions(
        raw_datasets=raw_datasets,
        singulartrajectory_model=experts.get('singulartrajectory'),
        dmrgcn_model=experts.get('dmrgcn'),
        gpgraph_model=experts.get('gpgraph'),
        dataset_name=cfg.dataset_name,
        cfg=cfg,
        hyper_params=hyper_params,
        accelerator=accelerator,
        phase=args.phase,
        stgcnn_model=experts.get('stgcnn'),
        expert_traj_model=experts.get('expert_traj'),
    )

    logger.info("Expert predictions precomputed successfully.")


if __name__ == "__main__":
    main()
