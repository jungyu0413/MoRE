"""MoRE: Main PPO training script for trajectory prediction.

Overall objective (Eq. 7):
  L_total = (1/|M|) Σ_{i∈M} [L_sup(i) + λ_RL · L_PPO(i)]

where M is the hard sample set from uncertainty-driven mining (Section 3.4).

Implementation notes (PPO mechanics, RLHF/TRL standard):
  • π_old is the policy AT ROLLOUT TIME (separate teacher-forced forward
    on the current π_θ), not π_ref.
  • π_ref is used ONLY for the KL penalty.
  • Token-level importance ratio + token-level clipped surrogate.
  • K = cfg.ppo_epochs PPO epochs over the same rollout
    (advantages and returns are frozen at rollout time).
  • Schulman approx_kl monitoring + optional ppo_target_kl early stop.
  • True per-step Shannon entropy on the policy distribution.
  • SFT/KD update and PPO update are run as separate optimizer steps.

Usage:
    accelerate launch -m more.train --config_file configs/default.json --tag my_run
"""

import json
import logging
import math
import os
import csv
import sys
from datetime import timedelta
from functools import partial

import numpy as np
import torch
import datasets
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
    get_scheduler, CONFIG_MAPPING,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from more.utils.nltoolkit import init_nltk
from more.data.converter import batch_text2traj
from more.data.homography import image2world, generate_homography
from more.data.preprocessing import preprocess_train_val, preprocess_test
from more.data.collator import TrajectoryDataCollator
from more.rl.ppo import (
    ValueHead, compute_advantage, masked_mean_pool,
    compute_ppo_loss_token_level,
    compute_kl_penalty_token_level,
    compute_entropy_token_level,
)
from more.rl.rewards import compute_reward_from_experts
from more.experts.predictions import load_expert_predictions
from more.mining import compute_sequence_entropy, select_hard_samples
from more.evaluate import run_test_evaluation

logger = get_logger(__name__)


def _print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    return trainable, total


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def trainval(cfg):
    init_nltk()

    # Checkpoint naming
    if cfg.checkpoint_name is None:
        cfg.checkpoint_name = getattr(cfg, 'run_tag', None) or f"rl-ppo-{cfg.dataset_name}"
    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)

    # ── Accelerator setup ────────────────────────────────────
    accel_kwargs = {}
    if cfg.use_logger:
        accel_kwargs["log_with"] = cfg.logger_type
        accel_kwargs["project_dir"] = checkpoint_path
    mixed_precision = getattr(cfg, 'mixed_precision', 'fp16')
    if mixed_precision != 'no':
        accel_kwargs["mixed_precision"] = mixed_precision

    ddp_timeout = getattr(cfg, 'ddp_timeout', 10800)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=ddp_timeout))],
        **accel_kwargs,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        set_seed(cfg.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
        try:
            with open(os.path.join(checkpoint_path, "run_config.json"), 'w') as f:
                json.dump(dict(cfg), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save run config: {e}")
    accelerator.wait_for_everyone()

    # ── Dataset loading ──────────────────────────────────────
    preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed")
    use_scene_context = getattr(cfg, 'use_scene_context', True)
    multimodal_suffix = "multimodal" if use_scene_context else "multimodal-nocontext"

    # Support prebuilt hard sample datasets (e.g., hard-top1.0)
    hard_sample_suffix = getattr(cfg, 'hard_sample_suffix', None)
    if hard_sample_suffix:
        train_dataset_name = f"{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-multimodal-{hard_sample_suffix}.json"
    else:
        train_dataset_name = f"{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-{multimodal_suffix}.json"
    val_dataset_name = f"{cfg.dataset_name}-val-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"
    test_dataset_name = f"{cfg.dataset_name}-test-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"

    data_files = {
        "train": os.path.join(preprocessed_dataset_path, train_dataset_name),
        "validation": os.path.join(preprocessed_dataset_path, val_dataset_name),
        "test": os.path.join(preprocessed_dataset_path, test_dataset_name),
    }
    for key, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}. Run preprocessing first.")

    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cfg.cache_dir)

    # Optional: other coordinate system
    other_coord_system = getattr(cfg, 'other_model_coordinate', None)
    other_coord_datasets = None
    if other_coord_system and other_coord_system != cfg.metric:
        other_files = {}
        for split, suffix in [("train", "-multimodal"), ("val", ""), ("test", "")]:
            fname = f"{cfg.dataset_name}-{split}-{cfg.obs_len}-{cfg.pred_len}-{other_coord_system}{suffix}.json"
            fpath = os.path.join(preprocessed_dataset_path, fname)
            if os.path.exists(fpath):
                other_files[split] = fpath
            else:
                other_coord_system = None
                break
        if other_coord_system and len(other_files) == 3:
            other_coord_datasets = load_dataset("json", data_files=other_files, cache_dir=cfg.cache_dir)
        else:
            other_coord_system = None

    # ── Model loading ────────────────────────────────────────
    test_only = getattr(cfg, 'test', False)
    model, tokenizer, config = _load_model(cfg, checkpoint_path)
    base_model_path = cfg.pretrained_checkpoint_path or cfg.model_name_or_path

    if not test_only and len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("config.decoder_start_token_id must be set.")

    # ── LoRA (optional, skip for test-only) ─────────────────
    use_lora = getattr(cfg, 'use_lora', False) and not test_only
    lora_r = lora_alpha = lora_dropout = None
    lora_target_modules = None

    if use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("Install peft: pip install peft")
        lora_r = getattr(cfg, 'lora_r', None) if getattr(cfg, 'lora_r', None) is not None else 16
        lora_alpha = getattr(cfg, 'lora_alpha', None) if getattr(cfg, 'lora_alpha', None) is not None else 32
        lora_dropout = getattr(cfg, 'lora_dropout', None) if getattr(cfg, 'lora_dropout', None) is not None else 0.1
        lora_target_modules = getattr(cfg, 'lora_target_modules', None) or ["q", "k", "v", "o", "wi", "wo"]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=lora_r,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_target_modules, bias="none", inference_mode=False,
        )
        lora_adapter_path = os.path.join(checkpoint_path, "lora_adapter")
        if os.path.exists(lora_adapter_path) and os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
        else:
            model = get_peft_model(model, lora_config)
        logger.info(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    trainable_params, all_params = _print_trainable_parameters(model)

    # ── Reference model π_ref (frozen, for KL penalty in Eq. 4) ──
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path, trust_remote_code=False, cache_dir=cfg.cache_dir,
    )
    ref_model.to(accelerator.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── Value head V(τ_obs) (Section 3.3) ────────────────────
    hidden_size = getattr(model.config, 'd_model', getattr(model.config, 'hidden_size', 512))
    value_head_dim = getattr(cfg, 'value_head_input_dim', 256)
    value_head = ValueHead(hidden_size, value_head_dim)
    value_head = value_head.to(next(model.parameters()).device)

    # ── Preprocessing ────────────────────────────────────────
    column_names = raw_datasets["train"].column_names
    padding = "max_length" if cfg.pad_to_max_length else False

    with accelerator.main_process_first():
        columns_to_keep = [c for c in ("obs_traj", "pred_traj", "scene_id", "scene",
                                        "scene_idx", "original_ped_id") if c in column_names]
        columns_to_remove = [c for c in column_names if c not in columns_to_keep]

        if not test_only:
            mk_func = lambda split: partial(
                preprocess_train_val, tokenizer=tokenizer,
                max_source_length=cfg.max_source_length, max_target_length=cfg.max_target_length,
                history_column=cfg.history_column, future_column=cfg.future_column,
                padding=padding, other_coord_system=other_coord_system,
                other_coord_datasets=other_coord_datasets, split_name=split,
            )
            train_dataset = raw_datasets["train"].map(
                mk_func("train"), batched=True, with_indices=True,
                num_proc=cfg.preprocessing_num_workers, remove_columns=columns_to_remove,
                load_from_cache_file=not cfg.overwrite_cache,
            )
            val_dataset = raw_datasets["validation"].map(
                mk_func("validation"), batched=True, with_indices=True,
                num_proc=cfg.preprocessing_num_workers, remove_columns=columns_to_remove,
                load_from_cache_file=not cfg.overwrite_cache,
            )
        else:
            train_dataset = val_dataset = None

        cols_remove_test = [c for c in column_names
                           if c not in ("obs_traj", "pred_traj", "original_ped_id")]
        test_dataset = raw_datasets["test"].map(
            partial(preprocess_test, tokenizer=tokenizer,
                    max_source_length=cfg.max_source_length, max_target_length=cfg.max_target_length,
                    history_column=cfg.history_column, future_column=cfg.future_column,
                    padding=padding),
            batched=True, num_proc=cfg.preprocessing_num_workers,
            remove_columns=cols_remove_test, load_from_cache_file=not cfg.overwrite_cache,
        )

    # ── Data loaders ─────────────────────────────────────────
    label_pad_token_id = getattr(cfg, 'label_pad_token_id', -100)
    pad_to_multiple_of = getattr(cfg, 'pad_to_multiple_of', 8)
    data_collator = TrajectoryDataCollator(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    if not test_only:
        dl_generator = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                      batch_size=cfg.per_device_train_batch_size,
                                      generator=dl_generator)
        eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator,
                                     batch_size=cfg.per_device_eval_batch_size)
    else:
        train_dataloader = eval_dataloader = None

    test_collator = TrajectoryDataCollator(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id,
    )
    test_batch_size = getattr(cfg, 'test_batch_size', 1)
    test_dataloader = DataLoader(test_dataset, collate_fn=test_collator,
                                 batch_size=test_batch_size, shuffle=False)

    # ── Trajectory dataloader (for postprocessing) ───────────
    from more.utils.dataloader import get_dataloader

    traj_dl = get_dataloader(os.path.join(cfg.dataset_path, cfg.dataset_name),
                             'test', cfg.obs_len, cfg.pred_len, batch_size=1e8)
    all_obs = traj_dl.dataset.obs_traj.numpy()
    all_gts = traj_dl.dataset.pred_traj.numpy()
    homography = traj_dl.dataset.homography
    scene_id_map = traj_dl.dataset.scene_id
    scene_map = traj_dl.dataset.scene_map
    seq_start_end = traj_dl.dataset.seq_start_end

    # Also load train homography for RL reward computation
    train_traj_dl = get_dataloader(os.path.join(cfg.dataset_path, cfg.dataset_name),
                                   'train', cfg.obs_len, cfg.pred_len, batch_size=1e8)
    if hasattr(train_traj_dl.dataset, 'homography'):
        for k, v in train_traj_dl.dataset.homography.items():
            if k not in homography:
                homography[k] = v
    del train_traj_dl

    image_scale_down = getattr(cfg, 'image_scale_down', 0.25)
    for k in homography:
        homography[k] = homography[k].copy() @ generate_homography(scale=image_scale_down)

    # ── Optimizer & scheduler ────────────────────────────────
    # Policy (LoRA on pretrained T5) and value head (random-init MLP) get
    # separate LRs: the critic typically needs a higher LR than the actor
    # to keep up with reward signals (RLHF/TRL convention).
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    policy_lr = cfg.learning_rate
    value_lr = getattr(cfg, 'value_lr', cfg.learning_rate)
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": cfg.weight_decay, "lr": policy_lr},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0, "lr": policy_lr},
    ]
    if value_head is not None:
        param_groups += [
            {"params": [p for n, p in value_head.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": cfg.weight_decay, "lr": value_lr},
            {"params": [p for n, p in value_head.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": value_lr},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate)

    if not test_only:
        num_update_steps = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            cfg.max_train_steps = cfg.num_train_epochs * num_update_steps

        # Each rl_step_frequency-th batch triggers ppo_epochs extra
        # optimizer.step()s (one SFT step + K PPO steps). Extend the LR
        # schedule horizon accordingly so it doesn't decay to zero early.
        rl_step_freq_init = getattr(cfg, 'rl_step_frequency', 100)
        ppo_epochs_init = getattr(cfg, 'ppo_epochs', 4)
        rl_extra_steps = (cfg.max_train_steps // max(rl_step_freq_init, 1)) * ppo_epochs_init
        scheduler_total_steps = cfg.max_train_steps + rl_extra_steps

        lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler_type, optimizer=optimizer,
            num_warmup_steps=cfg.num_warmup_steps * cfg.gradient_accumulation_steps,
            num_training_steps=scheduler_total_steps * cfg.gradient_accumulation_steps,
        )
        if value_head is not None:
            model, value_head, optimizer, train_dataloader, eval_dataloader, lr_scheduler = \
                accelerator.prepare(model, value_head, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
        else:
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = \
                accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
        num_update_steps = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps)
    else:
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=1)
        if value_head is not None:
            model, value_head, optimizer, lr_scheduler = accelerator.prepare(model, value_head, optimizer, lr_scheduler)
        else:
            model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # ── Load expert predictions ──────────────────────────────
    expert_predictions = None
    if not test_only:
        expert_tmp_path = getattr(cfg, 'expert_tmp_path', None)
        if expert_tmp_path is None:
            raise ValueError("expert_tmp_path must be set in config.")
        expert_predictions = load_expert_predictions(expert_tmp_path, cfg.dataset_name, 'train')
        if expert_predictions is None:
            raise FileNotFoundError(
                f"Expert predictions not found at {expert_tmp_path}. "
                f"Run precompute_experts.py first."
            )
        logger.info(f"Expert predictions loaded: {list(expert_predictions.keys())}")

    # ── Hard sample mining (Section 3.4, Eq. 5-6) ───────────
    mined_indices = None
    if not test_only and getattr(cfg, 'use_hard_mining', False):
        mining_cache_dir = getattr(cfg, 'expert_tmp_path', './expert_predictions')
        mining_pct = getattr(cfg, 'mining_percentile', 99.0)
        mining_cache_file = os.path.join(
            mining_cache_dir,
            f"mining_entropies_{cfg.dataset_name}_p{mining_pct:.0f}.npy",
        )

        if os.path.exists(mining_cache_file):
            # Load cached entropies
            cached = np.load(mining_cache_file, allow_pickle=True).item()
            entropies = cached['entropies']
            mined_indices = cached['mined_indices']
            threshold = cached['threshold']
            logger.info(
                f"Loaded cached mining results from {mining_cache_file} "
                f"(γ_{mining_pct:.0f} = {threshold:.4f}, |M| = {len(mined_indices)})"
            )
        else:
            # Compute entropies from scratch
            logger.info("Computing Shannon entropy for hard sample mining (Eq. 5)...")
            mining_dl = DataLoader(train_dataset, collate_fn=data_collator,
                                   batch_size=cfg.per_device_train_batch_size, shuffle=False)
            entropies, ent_indices = compute_sequence_entropy(
                model, tokenizer, mining_dl, accelerator,
                max_length=getattr(cfg, 'safe_max_length', 100),
                temperature=getattr(cfg, 'mining_temperature', 1.0),
            )
            mined_indices, threshold = select_hard_samples(entropies, percentile=mining_pct)
            logger.info(f"Mining threshold γ_{mining_pct:.0f} = {threshold:.4f}")

            # Cache to disk
            os.makedirs(mining_cache_dir, exist_ok=True)
            np.save(mining_cache_file, {
                'entropies': entropies,
                'mined_indices': mined_indices,
                'threshold': threshold,
            })
            logger.info(f"Mining results cached to {mining_cache_file}")
            del mining_dl

        # Filter training dataset to hard samples M
        train_dataset = train_dataset.select(mined_indices.tolist())
        logger.info(f"Training on |M| = {len(train_dataset)} hard samples (from {len(entropies)} total)")
        # Recreate DataLoader with mined subset
        dl_generator = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                      batch_size=cfg.per_device_train_batch_size,
                                      generator=dl_generator)
        train_dataloader = accelerator.prepare(train_dataloader)
        # Recalculate training steps based on new dataset size
        num_update_steps = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps)

    # ── Best tracking ────────────────────────────────────────
    best_tracker = {'combined': float('inf'), 'ade': float('inf'), 'fde': float('inf'), 'epoch': 0}

    # Shared kwargs for run_test_evaluation calls
    eval_kwargs = dict(
        model=model, tokenizer=tokenizer, value_head=value_head,
        accelerator=accelerator, cfg=cfg, test_dataloader=test_dataloader,
        homography=homography, all_obs=all_obs, all_gts=all_gts,
        scene_id_map=scene_id_map, scene_map=scene_map,
        seq_start_end=seq_start_end, traj_dl=traj_dl,
        checkpoint_path=checkpoint_path, use_lora=use_lora,
        best_tracker=best_tracker,
    )

    # ── Training loop ────────────────────────────────────────
    # Eq. 7: L_total = (1/|M|) Σ_{i∈M} [L_sup(i) + λ_RL · L_PPO(i)]
    if not test_only:
        rl_step_freq = getattr(cfg, 'rl_step_frequency', 100)
        eval_at_steps = set(getattr(cfg, 'eval_at_steps', None) or [])
        eval_at_steps_only = getattr(cfg, 'eval_at_steps_only', False)
        max_eval_step = max(eval_at_steps) if eval_at_steps else None

        progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        ppo_epochs = getattr(cfg, 'ppo_epochs', 4)
        rl_weight = getattr(cfg, 'rl_loss_weight', 0.5)  # λ_RL

        for epoch in range(cfg.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                # ── Common batch metadata ────────────────────────────
                ped_id_tensor = batch.get('original_ped_id')
                if ped_id_tensor is None:
                    batch_ped_ids = np.arange(len(batch['input_ids']), dtype=np.float64)
                else:
                    batch_ped_ids = ped_id_tensor.cpu().numpy().astype(float)
                batch_scene_ids = batch.get('scene_id')
                if batch_scene_ids is None:
                    scene_idx = batch.get('scene_idx')
                    if scene_idx is not None:
                        from more.data.collator import IDX_TO_SCENE
                        idx_list = scene_idx.tolist() if hasattr(scene_idx, 'tolist') else list(scene_idx)
                        batch_scene_ids = [IDX_TO_SCENE.get(int(i), 'unknown') for i in idx_list]
                expert_preds = _gather_batch_expert_preds(
                    cfg, expert_predictions, batch_ped_ids, accelerator.device,
                    batch_scene_ids=batch_scene_ids,
                )
                batch_gt_meter = _get_gt_meter(batch, cfg, homography)

                # ──────────────────────────────────────────────────────
                # (A) SFT update — every batch
                # ──────────────────────────────────────────────────────
                with accelerator.accumulate(model):
                    model_inputs = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'labels': batch['labels'],
                    }
                    outputs = model(**model_inputs)
                    sft_loss = outputs.loss  # L_sup
                    supervised_loss = sft_loss  # alias for downstream logging

                    accelerator.backward(sft_loss)

                    if accelerator.sync_gradients and getattr(cfg, 'max_grad_norm', None):
                        params_to_clip = list(model.parameters()) + list(value_head.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, max_norm=cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # ──────────────────────────────────────────────────────
                # (B) PPO update — every rl_step_frequency batches
                # ──────────────────────────────────────────────────────
                if step % rl_step_freq == 0:
                    ppo_metrics = _ppo_update(
                        model=model, ref_model=ref_model, value_head=value_head,
                        batch=batch, tokenizer=tokenizer,
                        expert_preds=expert_preds, batch_gt_meter=batch_gt_meter,
                        homography=homography, seq_start_end=seq_start_end,
                        scene_id_map=scene_id_map, cfg=cfg, accelerator=accelerator,
                        step=step, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        rl_weight=rl_weight, ppo_epochs=ppo_epochs,
                    )
                    # Append per-step rl history to disk every 10 steps
                    if accelerator.is_main_process and step % 10 == 0 and ppo_metrics is not None:
                        rl_hist_path = os.path.join(checkpoint_path, 'rl_history.txt')
                        write_header = not os.path.exists(rl_hist_path)
                        with open(rl_hist_path, 'a') as fp:
                            if write_header:
                                fp.write("step\tepoch\tR_mean\tL_sup\tL_PPO\tpg\tvf\tkl\tapprox_kl\tclip_frac\tepochs_run\n")
                            fp.write(
                                f"{step}\t{epoch+1}\t{ppo_metrics['reward_mean']:.4f}\t"
                                f"{supervised_loss.item():.4f}\t{ppo_metrics['rl_loss']:.4f}\t"
                                f"{ppo_metrics['policy_loss']:.4f}\t{ppo_metrics['value_loss']:.4f}\t"
                                f"{ppo_metrics['kl_div']:.4f}\t{ppo_metrics['approx_kl']:.4f}\t"
                                f"{ppo_metrics['clip_frac']:.4f}\t{ppo_metrics['epochs_run']}\n"
                            )
                    if accelerator.is_local_main_process and step % 10 == 0 and ppo_metrics is not None:
                        logger.info(
                            f"[Step {step}] L_sup={supervised_loss.item():.4f} "
                            f"L_PPO={ppo_metrics['rl_loss']:.4f} "
                            f"pg={ppo_metrics['policy_loss']:.4f} "
                            f"vf={ppo_metrics['value_loss']:.4f} "
                            f"kl={ppo_metrics['kl_div']:.4f} "
                            f"approx_kl={ppo_metrics['approx_kl']:.4f} "
                            f"clip_frac={ppo_metrics['clip_frac']:.3f} "
                            f"R_mean={ppo_metrics['reward_mean']:.3f} "
                            f"epochs={ppo_metrics['epochs_run']}"
                        )

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    should_eval = (
                        (eval_at_steps and completed_steps in eval_at_steps)
                        or (not eval_at_steps and completed_steps == rl_step_freq)
                    )
                    if should_eval:
                        run_test_evaluation(**eval_kwargs, epoch=epoch + 1, step=completed_steps)
                        model.train()

                    if eval_at_steps_only and max_eval_step and completed_steps >= max_eval_step:
                        break

            if eval_at_steps_only and max_eval_step and completed_steps >= max_eval_step:
                break

            run_test_evaluation(**eval_kwargs, epoch=epoch + 1, step=completed_steps)
    else:
        run_test_evaluation(**eval_kwargs, is_final=True)


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def _load_model(cfg, checkpoint_path):
    """Load model/tokenizer from pretrained checkpoint or config."""
    for path in [getattr(cfg, 'pretrained_checkpoint_path', None), checkpoint_path, cfg.model_name_or_path]:
        if path and os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            config = AutoConfig.from_pretrained(path, trust_remote_code=False, cache_dir=cfg.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=False, cache_dir=cfg.cache_dir,
                use_fast=not cfg.use_slow_tokenizer,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                path, config=config, trust_remote_code=False, cache_dir=cfg.cache_dir,
            )
            return model, tokenizer, config

    if cfg.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name, trust_remote_code=False, cache_dir=cfg.cache_dir,
            use_fast=not cfg.use_slow_tokenizer,
        )
    else:
        raise ValueError("tokenizer_name is required.")

    if cfg.model_config_name:
        config = AutoConfig.from_pretrained(cfg.model_config_name, trust_remote_code=False, cache_dir=cfg.cache_dir)
    else:
        config = CONFIG_MAPPING[cfg.model_type]()

    model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=False)
    return model, tokenizer, config


def _gather_batch_expert_preds(cfg, expert_predictions, batch_ped_ids, device,
                               batch_scene_ids=None):
    """Look up precomputed expert predictions {S_k} for a batch of pedestrian IDs.

    Supports both plain ped_id keys and (ped_id, scene) tuple keys.
    """
    # K=5 expert mapping (Section 3.2)
    EXPERT_MAP = {
        'sig': ('use_singulartrajectory', 'singular'),   # (iv) motion pattern
        'dmr': ('use_dmrgcn', 'dmrgcn'),                 # (ii) structural relation
        'gpg': ('use_gpgraph', 'gpgraph'),               # (iii) social grouping
        'stg': ('use_stgcnn', 'stgcnn'),                 # (i) local interaction
        'exp': ('use_expert_traj', 'expert_traj'),        # (v) goal expert
    }
    result = {}
    for short_name, (flag, pred_key) in EXPERT_MAP.items():
        if not getattr(cfg, flag, False) or pred_key not in expert_predictions:
            continue
        preds = []
        expert_dict = expert_predictions[pred_key]
        # Detect key format
        _uses_tuple_keys = False
        if expert_dict:
            first_key = next(iter(expert_dict))
            _uses_tuple_keys = isinstance(first_key, tuple)

        for i, pid in enumerate(batch_ped_ids):
            pred = None
            if _uses_tuple_keys and batch_scene_ids is not None:
                scene = batch_scene_ids[i] if isinstance(batch_scene_ids, list) else batch_scene_ids[i].item() if hasattr(batch_scene_ids[i], 'item') else str(batch_scene_ids[i])
                pred = expert_dict.get((float(pid), scene))
                if pred is None:
                    pred = expert_dict.get((pid, scene))
            else:
                # Fallback: plain ped_id keys
                pred = expert_dict.get(pid)
                if pred is None:
                    pred = expert_dict.get(int(pid))
                if pred is None:
                    pred = expert_dict.get(float(pid))
            if pred is not None:
                preds.append(
                    pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
                )
        if preds:
            result[short_name] = torch.from_numpy(np.stack(preds)).to(device)
    return result


def _get_gt_meter(batch, cfg, homography):
    """Extract ground-truth trajectory S_GT in meter coordinates from batch."""
    pred_traj_other = batch.get('pred_traj_other')
    pred_traj = batch.get('pred_traj')
    scene_ids = batch.get('scene_id')

    if pred_traj_other is not None:
        if isinstance(pred_traj_other, torch.Tensor):
            gt = pred_traj_other.cpu().numpy().astype(np.float32)
        else:
            gt = np.array(pred_traj_other, dtype=np.float32)
        if gt.ndim == 2:
            gt = gt.reshape(-1, cfg.pred_len, 2)
        return gt

    if pred_traj is not None:
        gt_list = []
        for i in range(len(pred_traj)):
            gt_pix = pred_traj[i]
            if isinstance(gt_pix, torch.Tensor):
                gt_pix = gt_pix.cpu().numpy()
            else:
                gt_pix = np.array(gt_pix)
            if gt_pix.ndim == 1:
                gt_pix = gt_pix.reshape(-1, 2)

            sid = None
            if scene_ids is not None:
                sid = scene_ids[i].item() if isinstance(scene_ids, torch.Tensor) else scene_ids[i]

            if cfg.metric == 'pixel' and sid and sid in homography:
                from more.data.homography import image2world
                gt_list.append(image2world(gt_pix, homography[sid]))
            else:
                gt_list.append(gt_pix)
        return np.stack(gt_list)

    return None


def _ppo_rollout(model, ref_model, value_head, batch, tokenizer,
                 expert_preds, batch_gt_meter, homography,
                 seq_start_end, scene_id_map, cfg, accelerator, step):
    """Sample one rollout and compute frozen quantities for PPO updates.

    Returns a dict whose tensors are detached and reused across PPO epochs:
      decoder_input_ids, labels_rl, mask_f
      old_token_lp  — log π_θ(a_t) at sampling time          (no_grad)
      ref_token_lp  — log π_ref(a_t)                          (no_grad)
      advantages    — A = R - V(τ_obs)                        (no_grad)
      returns       — R                                       (no_grad)
      raw_reward_mean — for logging
    """
    # NOTE: model.eval() is set by the caller (_ppo_update) for the entire
    # PPO phase so that dropout is deterministic across the rollout forward
    # and the K subsequent gradient passes — without this, r_t = exp(new−old)
    # is biased by independent dropout masks even at PPO epoch 0.
    unwrapped = accelerator.unwrap_model(model)

    # ── 1. Sample from current policy π_θ ─────────────────────
    # Disable autocast during generation to avoid NaN from bf16 logit overflow.
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        generated = unwrapped.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            max_length=getattr(cfg, 'safe_max_length', 100),
            num_beams=1, do_sample=True,
            temperature=getattr(cfg, 'rl_temperature', 1.0),
            top_k=getattr(cfg, 'rl_top_k', 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoder_input_ids = generated[:, :-1]
    labels_rl = generated[:, 1:]
    labels_mask = (labels_rl != tokenizer.pad_token_id).long()
    mask_f = labels_mask.float()

    # ── 2. log π_old from CURRENT π_θ (rollout-time policy) ───
    # Teacher-forced forward, no_grad. Cast logits to fp32 for log_softmax stability.
    with torch.no_grad():
        old_out = unwrapped(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=labels_mask,
        )
        old_lp_full = torch.nn.functional.log_softmax(old_out.logits.float(), dim=-1)
        old_token_lp = old_lp_full.gather(-1, labels_rl.unsqueeze(-1)).squeeze(-1)
        old_token_lp = old_token_lp * mask_f

    # ── 3. log π_ref from frozen reference model (KL target) ──
    with torch.no_grad():
        ref_out = ref_model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=labels_mask,
        )
        ref_lp_full = torch.nn.functional.log_softmax(ref_out.logits.float(), dim=-1)
        ref_token_lp = ref_lp_full.gather(-1, labels_rl.unsqueeze(-1)).squeeze(-1)
        ref_token_lp = ref_token_lp * mask_f

    # ── 4. Decode and compute composite reward R (Eq. 3) ─────
    gen_np = generated.cpu().numpy()
    if cfg.use_slow_tokenizer and hasattr(tokenizer, 'sp_model'):
        filtered = np.where(gen_np >= tokenizer.sp_model.get_piece_size(), 0, gen_np)
        decoded = [tokenizer.sp_model.decode(t.tolist()) for t in filtered]
    else:
        decoded = tokenizer.batch_decode(gen_np, skip_special_tokens=True)
    decoded = [p.strip() for p in decoded]

    expert_weights = {
        'sig': getattr(cfg, 'expert_weight_sig', 0.2),
        'dmr': getattr(cfg, 'expert_weight_dmr', 0.2),
        'gpg': getattr(cfg, 'expert_weight_gpg', 0.2),
        'stg': getattr(cfg, 'expert_weight_stg', 0.2),
        'exp': getattr(cfg, 'expert_weight_exp', 0.2),
    }
    rewards = compute_reward_from_experts(
        decoded_preds=decoded, batch_scene_ids=batch.get("scene_id"),
        homography=homography, expert_preds_dict=expert_preds,
        pred_len=cfg.pred_len, device=accelerator.device,
        gt_traj_meter=batch_gt_meter if getattr(cfg, 'use_gt_reward', True) else None,
        uwo_lambda=getattr(cfg, 'rl_uwo_lambda', 0.5),
        ensemble_weight=getattr(cfg, 'rl_ensemble_weight', 0.5),
        reward_fail_penalty=getattr(cfg, 'reward_fail_penalty', -10.0),
        seq_start_end=seq_start_end, scene_id=scene_id_map,
        step=step, cfg=cfg, accelerator=accelerator,
        gt_weight=getattr(cfg, 'rl_gt_weight', 1.0),
        expert_weights=expert_weights,
    )
    raw_reward_mean = rewards.mean().item()


    # Clamp raw rewards (no z-score here — advantages are normalized below).
    rewards = torch.clamp(
        rewards,
        min=getattr(cfg, 'reward_clamp_min', -10.0),
        max=getattr(cfg, 'reward_clamp_max', 0.0),
    )

    # ── 5. V_rollout for advantage (frozen) ───────────────────
    # Reuse encoder hidden states from the old_out forward to avoid an
    # extra encoder pass.
    with torch.no_grad():
        pooled = masked_mean_pool(
            old_out.encoder_last_hidden_state, batch["attention_mask"],
        )
        values_rollout = value_head(pooled.float())

    # Returns target = raw R; advantages get z-score normalized + clamped.
    # PPO theory needs the *advantage* magnitude to be controlled (it
    # directly scales the policy gradient); normalizing rewards before the
    # subtraction propagates V's noise into the advantage scale.
    advantages, returns = compute_advantage(
        rewards=rewards, values=values_rollout,
    )
    norm_clamp = getattr(cfg, 'reward_norm_clamp', 5.0)
    # Center-only advantage normalization (no std rescale): when reward
    # distribution is naturally tight, z-scoring blows up sub-unit noise to
    # ~1.0 scale and wipes out the actual signal. Centering preserves the
    # relative ordering and magnitude.
    advantages = torch.clamp(advantages - advantages.mean(), -norm_clamp, norm_clamp)

    return {
        'decoder_input_ids': decoder_input_ids.detach(),
        'labels_rl': labels_rl.detach(),
        'mask_f': mask_f.detach(),
        'old_token_lp': old_token_lp.detach(),
        'ref_token_lp': ref_token_lp.detach(),
        'advantages': advantages,
        'returns': returns,
        'raw_reward_mean': raw_reward_mean,
    }


def _ppo_step_loss(model, value_head, batch, rollout, cfg):
    """One PPO epoch over a frozen rollout: returns scalar loss + metrics."""
    mask_f = rollout['mask_f']
    labels_rl = rollout['labels_rl']
    decoder_input_ids = rollout['decoder_input_ids']

    # Forward pass with grad — gives both new log-probs and encoder pool.
    out = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        decoder_input_ids=decoder_input_ids, decoder_attention_mask=mask_f.long(),
    )
    new_lp_full = torch.nn.functional.log_softmax(out.logits.float(), dim=-1)
    new_token_lp = new_lp_full.gather(-1, labels_rl.unsqueeze(-1)).squeeze(-1)
    new_token_lp = new_token_lp * mask_f

    # Token-level clipped surrogate
    policy_loss, ratio, clip_frac, approx_kl = compute_ppo_loss_token_level(
        new_token_lp=new_token_lp,
        old_token_lp=rollout['old_token_lp'],
        advantages=rollout['advantages'],
        mask=mask_f,
        cliprange=getattr(cfg, 'ppo_cliprange', 0.2),
    )

    # KL(π_θ || π_ref) penalty (token-level mean)
    kl_div = compute_kl_penalty_token_level(
        new_token_lp=new_token_lp,
        ref_token_lp=rollout['ref_token_lp'],
        mask=mask_f,
    )

    # Value loss: (V - R)²
    # Encoder hidden states are detached so the value regression gradient
    # does not flow back into the LoRA-adapted policy backbone.
    pooled = masked_mean_pool(out.encoder_last_hidden_state.detach(), batch["attention_mask"])
    values = value_head(pooled.float())
    value_loss = (values - rollout['returns']).pow(2).mean()

    # True per-step Shannon entropy on policy distribution
    entropy = compute_entropy_token_level(new_lp_full, mask_f)

    kl_beta = getattr(cfg, 'kl_beta', 0.1)
    vf_coef = getattr(cfg, 'ppo_vf_coef', 0.5)
    ent_coef = getattr(cfg, 'ppo_entropy_coef', 0.01)

    rl_loss = policy_loss + kl_beta * kl_div + vf_coef * value_loss - ent_coef * entropy

    metrics = {
        'rl_loss': float(rl_loss.detach().item()),
        'policy_loss': float(policy_loss.detach().item()),
        'value_loss': float(value_loss.detach().item()),
        'kl_div': float(kl_div.detach().item()),
        'approx_kl': float(approx_kl.detach().item()),
        'clip_frac': float(clip_frac.detach().item()),
        'entropy': float(entropy.detach().item()),
    }
    return rl_loss, metrics


def _ppo_update(model, ref_model, value_head, batch, tokenizer,
                expert_preds, batch_gt_meter, homography,
                seq_start_end, scene_id_map, cfg, accelerator, step,
                optimizer, lr_scheduler, rl_weight=0.5, ppo_epochs=4):
    """Full PPO update: rollout + K epochs of policy/value updates.

    Each PPO epoch is its own optimizer.step() (RLHF/TRL style).
    Optional early-stop via cfg.ppo_target_kl on Schulman approx_kl.

    The entire PPO phase (rollout + K updates) is run with the policy in
    eval() mode so that dropout is deterministic. Otherwise the IS ratio
    r_t = exp(log π_θ_new − log π_θ_old) reflects dropout randomness on top
    of true policy change, biasing the clipped surrogate even at epoch 0.
    """
    was_training = model.training
    model.eval()
    try:
        rollout = _ppo_rollout(
            model, ref_model, value_head, batch, tokenizer,
            expert_preds, batch_gt_meter, homography,
            seq_start_end, scene_id_map, cfg, accelerator, step,
        )

        target_kl = getattr(cfg, 'ppo_target_kl', None)
        last_metrics = None
        epochs_run = 0

        for k in range(ppo_epochs):
            rl_loss, metrics = _ppo_step_loss(model, value_head, batch, rollout, cfg)
            accelerator.backward(rl_weight * rl_loss)

            if accelerator.sync_gradients and getattr(cfg, 'max_grad_norm', None):
                params_to_clip = list(model.parameters()) + list(value_head.parameters())
                accelerator.clip_grad_norm_(params_to_clip, max_norm=cfg.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            last_metrics = metrics
            epochs_run = k + 1

            if target_kl is not None and metrics['approx_kl'] > float(target_kl):
                break

        if last_metrics is not None:
            last_metrics['epochs_run'] = epochs_run
            last_metrics['reward_mean'] = rollout['raw_reward_mean']

        # Cleanup rollout tensors
        del rollout
    finally:
        if was_training:
            model.train()
    return last_metrics


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from more.utils.config import get_exp_config

    parser = argparse.ArgumentParser(description="MoRE: PPO Training for Trajectory Prediction")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Override dataset_name from the config file (eth/hotel/univ/zara1/zara2)")
    args = parser.parse_args()

    cfg = get_exp_config(args.config_file)
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.tag:
        cfg.checkpoint_name = args.tag
        cfg.run_tag = args.tag

    trainval(cfg)
