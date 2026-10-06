"""MoRE: Evaluation logic for trajectory prediction.

Provides run_test_evaluation() used during training and standalone evaluation.

Usage (standalone):
    accelerate launch -m more.evaluate --config_file configs/default.json --checkpoint best_model
"""

import csv
import json
import logging
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from more.data.converter import batch_text2traj
from more.data.homography import image2world
from more.data.postprocessor import postprocess_trajectory
from more.inference.tta import mode_for_dataset, build_variants, split_counts

logger = logging.getLogger(__name__)


def run_test_evaluation(
    model, tokenizer, value_head, accelerator, cfg,
    test_dataloader, homography, all_obs, all_gts,
    scene_id_map, scene_map, seq_start_end, traj_dl,
    checkpoint_path, use_lora, best_tracker,
    epoch=None, step=None, is_final=False,
):
    """Run test evaluation under DDP.

    test_dataloader is NOT prepared by accelerator.
    Each process handles batch_idx % num_processes == process_index.
    Single gather at the end.
    """
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    device = accelerator.device
    test_num_samples = cfg.test_num_samples if not cfg.deterministic else 1

    total_batches = len(test_dataloader)

    local_preds_list = []
    local_indices_list = []

    # ── Test-time augmentation (always on for stochastic evaluation) ──
    if not cfg.deterministic:
        tta_mode = mode_for_dataset(cfg.dataset_name)
        import json as _json
        _test_json = os.path.join(cfg.dataset_path, "preprocessed",
                                  f"{cfg.dataset_name}-test-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json")
        tta_prompts = [_json.loads(l)['observation'] for l in open(_test_json)]
        assert len(tta_prompts) == len(all_gts), (len(tta_prompts), len(all_gts))
        _sd = getattr(cfg, 'image_scale_down', 0.25)
        tta_size = {k: (np.array(v.size) * _sd).astype(int).tolist()
                    for k, v in traj_dl.dataset.scene_img.items() if v is not None}
        tta_caption = traj_dl.dataset.scene_desc
        piece_size = tokenizer.sp_model.get_piece_size() if hasattr(tokenizer, 'sp_model') else None
        logger.info(f"TTA: mode={tta_mode}")

    if accelerator.is_local_main_process:
        pbar = tqdm(total=total_batches, desc=f"Evaluating (GPU {accelerator.process_index})", unit="batch")

    for batch_idx, batch in enumerate(test_dataloader):
        # Each process handles its own subset of batches
        if batch_idx % accelerator.num_processes != accelerator.process_index:
            if accelerator.is_local_main_process:
                pbar.update(1)
            continue

        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if cfg.deterministic:
                generated_tokens = unwrapped.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_length=cfg.max_target_length, num_beams=cfg.num_beams,
                )
            else:
                traj_data = _tta_sample(unwrapped, tokenizer, cfg, device, batch_idx, tta_mode, tta_prompts,
                                        tta_size, tta_caption, scene_id_map, all_obs, test_num_samples, piece_size)
                local_preds_list.append(traj_data)
                local_indices_list.append(batch_idx)
                if accelerator.is_local_main_process:
                    pbar.update(1)
                continue

            generated_tokens = generated_tokens.cpu().numpy()

            if cfg.use_slow_tokenizer and hasattr(tokenizer, 'sp_model'):
                filtered = np.where(generated_tokens >= tokenizer.sp_model.get_piece_size(), 0, generated_tokens)
                decoded = [tokenizer.sp_model.decode(t.tolist()) for t in filtered]
            else:
                decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded = [p.strip() for p in decoded]
            traj_data = batch_text2traj(decoded, frame=cfg.pred_len, dim=2)

            # Handle None (failed parse) — batch_size=1, so batch_idx == ped_idx
            for i in range(len(traj_data)):
                if traj_data[i] is None:
                    if batch_idx < len(all_obs):
                        traj_data[i] = np.tile(all_obs[batch_idx, -1], (cfg.pred_len, 1))
                    else:
                        traj_data[i] = np.zeros((cfg.pred_len, 2))

            if cfg.deterministic:
                traj_data = np.stack(traj_data, axis=0).reshape(-1, 1, cfg.pred_len, 2)
            else:
                traj_data = np.stack(traj_data, axis=0).reshape(-1, test_num_samples, cfg.pred_len, 2)

            # batch_size=1, store first (only) pedestrian
            local_preds_list.append(traj_data[0])
            local_indices_list.append(batch_idx)

        if accelerator.is_local_main_process:
            pbar.update(1)

    if accelerator.is_local_main_process:
        pbar.close()

    # ── Single gather at the end ─────────────────────────────
    accelerator.wait_for_everyone()

    if len(local_preds_list) > 0:
        local_preds = np.stack(local_preds_list, axis=0)
        local_indices = np.array(local_indices_list, dtype=np.int64)
    else:
        local_preds = np.zeros((0, test_num_samples, cfg.pred_len, 2), dtype=np.float32)
        local_indices = np.array([], dtype=np.int64)

    local_preds_t = torch.from_numpy(local_preds.astype(np.float32)).to(device)
    local_indices_t = torch.from_numpy(local_indices).to(device)

    # Pad to same size for NCCL gather
    local_n = torch.tensor([len(local_indices)], device=device)
    all_n = accelerator.gather(local_n)
    max_n = int(all_n.max().item())

    if len(local_indices) < max_n:
        pad = max_n - len(local_indices)
        local_preds_t = torch.cat([local_preds_t,
            torch.zeros(pad, *local_preds_t.shape[1:], device=device)])
        local_indices_t = torch.cat([local_indices_t,
            torch.full((pad,), -1, dtype=local_indices_t.dtype, device=device)])

    accelerator.wait_for_everyone()
    gathered_preds = accelerator.gather(local_preds_t)
    gathered_indices = accelerator.gather(local_indices_t)

    if accelerator.is_main_process:
        valid = gathered_indices >= 0
        gathered_preds = gathered_preds[valid]
        gathered_indices = gathered_indices[valid]
        order = torch.argsort(gathered_indices)
        gathered_preds = gathered_preds[order].cpu().numpy()
        gathered_indices = gathered_indices[order].cpu().numpy()

        total = len(all_gts)
        all_preds = np.zeros((total, test_num_samples, cfg.pred_len, 2), dtype=np.float32)
        for i, idx in enumerate(gathered_indices):
            if 0 <= idx < total:
                all_preds[int(idx)] = gathered_preds[i]

        all_preds = postprocess_trajectory(all_preds, all_obs, seq_start_end,
                                           scene_id_map, homography, scene_map, cfg)

        ADE, FDE = [], []
        for i in range(len(all_preds)):
            pred, gt = all_preds[i], all_gts[i]
            if cfg.metric == 'pixel' and scene_id_map[i] in homography:
                H = homography[scene_id_map[i]]
                orig_shape = pred.shape
                pred_2d = pred.reshape(-1, 2)
                pred_2d = image2world(pred_2d, H)
                pred = pred_2d.reshape(orig_shape)
                gt = traj_dl.dataset.pred_traj[i].numpy()
            diff = np.linalg.norm(pred - gt, ord=2, axis=-1)
            ADE.append(np.mean(diff, axis=-1).min())
            FDE.append(diff[:, -1].min())

        current_ade = np.mean(ADE)
        current_fde = np.mean(FDE)
        current_combined = (current_ade + current_fde) / 2

        label = f"Step {step}" if step else (f"Epoch {epoch}" if epoch else "Final")
        logger.info(f"{'='*60}")
        logger.info(f"{label} | ADE: {current_ade:.6f} | FDE: {current_fde:.6f} | Combined: {current_combined:.6f}")
        logger.info(f"{'='*60}")

        # Append to metrics history TXT
        history_path = os.path.join(checkpoint_path, "metrics_history.txt")
        write_header = not os.path.exists(history_path)
        with open(history_path, 'a') as f:
            if write_header:
                f.write("epoch\tstep\tADE\tFDE\tcombined\n")
            f.write(f"{epoch if epoch else ''}\t{step if step else ''}\t"
                    f"{current_ade:.6f}\t{current_fde:.6f}\t{current_combined:.6f}\n")

        # Save predictions
        subdir = f"step_{step}" if step else (f"epoch_{epoch}" if epoch else "final")
        save_dir = os.path.join(checkpoint_path, "predictions", subdir)
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "per_pedestrian_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ped_idx', 'ADE', 'FDE'])
            for i in range(len(ADE)):
                writer.writerow([i, f"{ADE[i]:.6f}", f"{FDE[i]:.6f}"])

        # Save full trajectories (all K samples + best + GT + obs in world coords)
        try:
            obs_world = np.zeros_like(all_obs)
            pred_world = np.zeros((len(all_preds), cfg.pred_len, 2), dtype=np.float32)
            all_preds_world = np.zeros_like(all_preds)
            gt_world = np.zeros((len(all_gts), cfg.pred_len, 2), dtype=np.float32)
            for i in range(len(all_preds)):
                pred = all_preds[i]
                obs = all_obs[i]
                gt = all_gts[i]
                if cfg.metric == 'pixel' and scene_id_map[i] in homography:
                    H = homography[scene_id_map[i]]
                    pred_world_i = image2world(pred.reshape(-1, 2), H).reshape(pred.shape)
                    obs_world_i = image2world(obs.reshape(-1, 2), H).reshape(obs.shape)
                    gt_world_i = traj_dl.dataset.pred_traj[i].numpy()
                else:
                    pred_world_i, obs_world_i, gt_world_i = pred, obs, gt
                all_preds_world[i] = pred_world_i
                obs_world[i] = obs_world_i
                gt_world[i] = gt_world_i
                diff = np.linalg.norm(pred_world_i - gt_world_i, axis=-1).mean(axis=-1)
                pred_world[i] = pred_world_i[int(diff.argmin())]
            np.save(os.path.join(save_dir, "all_preds.npy"), all_preds_world)
            np.save(os.path.join(save_dir, "pred_traj_world.npy"), pred_world)
            np.save(os.path.join(save_dir, "gt_traj_world.npy"), gt_world)
            np.save(os.path.join(save_dir, "all_obs.npy"), obs_world)
            logger.info(f"Saved trajectories to {save_dir}")
        except Exception as e:
            logger.warning(f"Failed to save trajectories: {e}")

        # Save model checkpoint at every eval point (LoRA adapter + value head)
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            if use_lora:
                unwrapped_model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
            else:
                unwrapped_model.save_pretrained(save_dir)
            if value_head is not None:
                torch.save(accelerator.unwrap_model(value_head).state_dict(),
                           os.path.join(save_dir, "value_head.pt"))
            with open(os.path.join(save_dir, "results.json"), "w") as f:
                json.dump({"ade": float(current_ade), "fde": float(current_fde),
                           "combined": float(current_combined),
                           "step": step, "epoch": epoch, "use_lora": use_lora}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save step checkpoint: {e}")

        # Best model saving
        if current_combined < best_tracker['combined']:
            best_tracker['combined'] = current_combined
            best_tracker['ade'] = current_ade
            best_tracker['fde'] = current_fde
            best_tracker['epoch'] = epoch or 0
            save_path = os.path.join(checkpoint_path, "best_model")
            os.makedirs(save_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            if use_lora:
                unwrapped_model.save_pretrained(os.path.join(save_path, "lora_adapter"))
            else:
                unwrapped_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            if value_head is not None:
                torch.save(accelerator.unwrap_model(value_head).state_dict(),
                           os.path.join(save_path, "value_head.pt"))
            with open(os.path.join(save_path, "results.json"), "w") as f:
                json.dump({"ade": float(current_ade), "fde": float(current_fde),
                           "epoch": epoch, "use_lora": use_lora}, f, indent=2)
            logger.info(f"New best model saved to {save_path}")

        # Epoch/step checkpoint
        if epoch is not None:
            ep_path = os.path.join(checkpoint_path, f"epoch_{epoch}")
            os.makedirs(ep_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            if use_lora:
                unwrapped_model.save_pretrained(os.path.join(ep_path, "lora_adapter"))
            else:
                unwrapped_model.save_pretrained(ep_path)
            tokenizer.save_pretrained(ep_path)

    accelerator.wait_for_everyone()
    unwrapped.train()


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def _tta_sample(model, tokenizer, cfg, device, idx, mode, prompts, sizes, captions, scene_id_map, all_obs,
                n_total, piece_size):
    """Sample n_total trajectories for pedestrian idx, split over the TTA prompt variants, in the original frame."""
    sid = scene_id_map[idx]
    variants = build_variants(prompts[idx], mode, sizes.get(sid, [0, 0]), captions.get(sid, ''))
    counts = split_counts(n_total, len(variants))
    enc = tokenizer([v[0] for v in variants], max_length=cfg.max_source_length, padding=True,
                    truncation=True, return_tensors='pt')
    gen = model.generate(input_ids=enc.input_ids.to(device), attention_mask=enc.attention_mask.to(device),
                         max_length=cfg.max_target_length, do_sample=True, num_return_sequences=max(counts),
                         temperature=cfg.temperature, top_k=cfg.top_k)
    gen = gen.cpu().numpy().reshape(len(variants), max(counts), -1)
    out = []
    for k, (_, inverse) in enumerate(variants):
        g = gen[k, :counts[k]]
        if piece_size is not None:
            g = np.where(g >= piece_size, 0, g)
            decoded = [tokenizer.sp_model.decode(t.tolist()).strip() for t in g]
        else:
            decoded = [d.strip() for d in tokenizer.batch_decode(g, skip_special_tokens=True)]
        for t in batch_text2traj(decoded, frame=cfg.pred_len, dim=2):
            # failed parses fall back to a static trajectory, as in the original evaluation
            out.append(np.tile(all_obs[idx, -1], (cfg.pred_len, 1)) if t is None else inverse(t))
    return np.stack(out, 0).astype(np.float32)


def _merge_lora(base_dir, adapter_dir):
    """Merge a LoRA adapter into its base model (test mode in more.train does not load adapters)."""
    import dataclasses, shutil, tempfile
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel, LoraConfig
    out = tempfile.mkdtemp(prefix='more_merged_')
    known = {f.name for f in dataclasses.fields(LoraConfig)}
    acfg = json.load(open(os.path.join(adapter_dir, 'adapter_config.json')))
    ad = os.path.join(out, 'adapter'); os.makedirs(ad)
    json.dump({k: v for k, v in acfg.items() if k in known}, open(os.path.join(ad, 'adapter_config.json'), 'w'))
    for f in os.listdir(adapter_dir):
        if f.startswith('adapter_model'):
            shutil.copy(os.path.join(adapter_dir, f), ad)
    model = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(base_dir), ad).merge_and_unload()
    model.save_pretrained(out); AutoTokenizer.from_pretrained(base_dir, use_fast=False).save_pretrained(out)
    shutil.rmtree(ad)
    return out


def main():
    import argparse
    from more.utils.config import get_exp_config
    from more.train import trainval

    parser = argparse.ArgumentParser(description="MoRE: Evaluate trajectory prediction model")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model to evaluate. If omitted (and the config has no pretrained_checkpoint_path), "
                             "the best known weights for the dataset are used (more/inference/tta.py BEST_WEIGHTS)")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Override dataset_name from the config file (eth/hotel/univ/zara1/zara2)")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA adapter directory to merge into the checkpoint before evaluation")
    parser.add_argument("--no_fast_attention", action="store_true",
                        help="Disable the copy-free T5 attention used for faster sampling (outputs are identical)")
    args = parser.parse_args()

    cfg = get_exp_config(args.config_file)
    cfg.test = True  # Force test-only mode

    if args.dataset_name:
        cfg.dataset_name = args.dataset_name

    if args.tag:
        cfg.checkpoint_name = args.tag
    if args.checkpoint:
        cfg.pretrained_checkpoint_path = args.checkpoint
    if not cfg.get('pretrained_checkpoint_path'):
        from more.inference.tta import best_weights
        base, adapter = best_weights(cfg.dataset_name, cfg.get('weights_dir'))
        cfg.pretrained_checkpoint_path = base
        if adapter and not args.lora:
            args.lora = adapter
        print(f"[MoRE] No checkpoint given: using best weights for {cfg.dataset_name}: {base}" + (f" + {adapter}" if adapter else ""))
    if not args.no_fast_attention:
        from more.inference import fast_t5
        fast_t5.enable()
    if args.lora:
        cfg.pretrained_checkpoint_path = _merge_lora(cfg.pretrained_checkpoint_path, args.lora)

    trainval(cfg)


if __name__ == "__main__":
    main()
