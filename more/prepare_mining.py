"""Precompute hard sample mining entropies and cache to disk.

This can be run independently before training so that train.py
simply loads the cached .npy file instead of recomputing every time.

Usage:
    accelerate launch -m more.precompute_mining --config_file configs/default.json
"""

import argparse
import logging
import math
import os
import sys
from functools import partial

import numpy as np
import torch
import datasets
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from more.utils.config import get_exp_config
from more.data.preprocessing import preprocess_train_val
from more.data.collator import TrajectoryDataCollator
from more.mining import compute_sequence_entropy, select_hard_samples

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute hard sample mining")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Override dataset_name from the config file (eth/hotel/univ/zara1/zara2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    cfg = get_exp_config(args.config_file)
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    accelerator = Accelerator(mixed_precision=getattr(cfg, 'mixed_precision', None))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ── Dataset loading ──────────────────────────────────────
    preprocessed_path = os.path.join(cfg.dataset_path, "preprocessed")
    use_scene_context = getattr(cfg, 'use_scene_context', True)
    multimodal_suffix = "multimodal" if use_scene_context else "multimodal-nocontext"

    train_file = os.path.join(
        preprocessed_path,
        f"{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-{multimodal_suffix}.json",
    )
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")

    raw_datasets = load_dataset("json", data_files={"train": train_file}, cache_dir=cfg.cache_dir)

    # ── Model loading ────────────────────────────────────────
    base_model_path = cfg.pretrained_checkpoint_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name or base_model_path,
        use_fast=not cfg.use_slow_tokenizer,
        cache_dir=cfg.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path, trust_remote_code=False, cache_dir=cfg.cache_dir,
    )

    # Load LoRA adapter if exists
    tag = args.tag or getattr(cfg, 'checkpoint_name', None) or getattr(cfg, 'config_name', 'default')
    checkpoint_path = os.path.join(cfg.checkpoint_path, tag)
    lora_adapter_path = os.path.join(checkpoint_path, "lora_adapter")
    if getattr(cfg, 'use_lora', False) and PEFT_AVAILABLE:
        if os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=False)
            logger.info(f"LoRA adapter loaded from {lora_adapter_path}")

    model.eval()

    # ── Preprocessing ────────────────────────────────────────
    column_names = raw_datasets["train"].column_names
    padding = "max_length" if cfg.pad_to_max_length else False
    columns_to_keep = [c for c in ("obs_traj", "pred_traj", "scene_id", "scene",
                                    "scene_idx", "original_ped_id") if c in column_names]
    columns_to_remove = [c for c in column_names if c not in columns_to_keep]

    train_dataset = raw_datasets["train"].map(
        partial(
            preprocess_train_val, tokenizer=tokenizer,
            max_source_length=cfg.max_source_length, max_target_length=cfg.max_target_length,
            history_column=cfg.history_column, future_column=cfg.future_column,
            padding=padding, other_coord_system=None, other_coord_datasets=None,
            split_name="train",
        ),
        batched=True, with_indices=True,
        num_proc=cfg.preprocessing_num_workers, remove_columns=columns_to_remove,
        load_from_cache_file=not cfg.overwrite_cache,
    )

    label_pad_token_id = getattr(cfg, 'label_pad_token_id', -100)
    pad_to_multiple_of = getattr(cfg, 'pad_to_multiple_of', 8)
    data_collator = TrajectoryDataCollator(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    mining_dl = DataLoader(
        train_dataset, collate_fn=data_collator,
        batch_size=cfg.per_device_train_batch_size, shuffle=False,
    )

    model = accelerator.prepare(model)

    # ── Compute entropy (Eq. 5) ──────────────────────────────
    logger.info("Computing Shannon entropy for hard sample mining (Eq. 5)...")
    entropies, ent_indices = compute_sequence_entropy(
        model, tokenizer, mining_dl, accelerator,
        max_length=getattr(cfg, 'safe_max_length', 100),
        temperature=getattr(cfg, 'mining_temperature', 1.0),
    )

    # ── Select hard samples (Eq. 6) ─────────────────────────
    mining_pct = getattr(cfg, 'mining_percentile', 99.0)
    mined_indices, threshold = select_hard_samples(entropies, percentile=mining_pct)

    # ── Save to .npy ─────────────────────────────────────────
    cache_dir = getattr(cfg, 'expert_tmp_path', './expert_predictions')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"mining_entropies_{cfg.dataset_name}_p{mining_pct:.0f}.npy",
    )
    np.save(cache_file, {
        'entropies': entropies,
        'mined_indices': mined_indices,
        'threshold': threshold,
    })
    logger.info(
        f"Saved mining results to {cache_file}\n"
        f"  Total samples: {len(entropies)}\n"
        f"  Threshold γ_{mining_pct:.0f} = {threshold:.4f}\n"
        f"  Hard samples |M| = {len(mined_indices)} ({100 * len(mined_indices) / len(entropies):.1f}%)"
    )


if __name__ == "__main__":
    main()
