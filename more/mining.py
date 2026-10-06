"""Uncertainty-Driven Mining (Section 3.4).

Difficulty scoring via Shannon entropy over the vocabulary space (Eq. 5):

  S_i = -(1/L_i) Σ_{k=1}^{L_i} Σ_{v∈V} π(v | τ_obs, τ_{<k}) log π(v | τ_obs, τ_{<k})

      = (1/L_i) Σ_{k=1}^{L_i} H[π(· | τ_obs, τ_{<k})]

where H[·] is the Shannon entropy of the per-step token distribution.

Hard sample selection with threshold γ_p (Eq. 6):
  M = {(τ_obs_i, τ_gt_i) | S_i ≥ γ_p}

where γ_p is the p-th percentile of {S_i}_{i=1}^{N}.
"""

import logging

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_sequence_entropy(
    model,
    tokenizer,
    dataloader,
    accelerator,
    max_length: int = 100,
    temperature: float = 1.0,
):
    """Compute Shannon entropy difficulty score S_i for each sample (Eq. 5).

    S_i = (1/L_i) Σ_{k=1}^{L_i} H_k

    where H_k = -Σ_{v∈V} π(v | τ_obs, τ_{<k}) log π(v | τ_obs, τ_{<k})
    is the per-step Shannon entropy over the vocabulary V.

    Args:
        model: The seq2seq language model π_θ.
        tokenizer: Tokenizer for the model.
        dataloader: DataLoader yielding batches with input_ids, attention_mask, labels.
        accelerator: Accelerate instance.
        max_length: Maximum generation length for auto-regressive mode.
        temperature: Temperature τ for softmax (default 1.0 = standard entropy).

    Returns:
        entropies: np.ndarray of shape (N,) with per-sample entropy scores S_i.
        indices: np.ndarray of shape (N,) with sample indices.
    """
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    device = accelerator.device

    all_entropies = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader, desc="Computing entropy (Eq. 5)",
            disable=not accelerator.is_local_main_process,
        )):
            input_ids = batch["input_ids"].to(device)       # τ_obs tokenized
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")

            if labels is not None:
                labels = labels.to(device)
                # Teacher-forced: compute H_k at each decoding position k
                outputs = unwrapped(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = outputs.logits  # (B, L_i, |V|)
                if temperature != 1.0:
                    logits = logits / temperature

                # π(v | τ_obs, τ_{<k}) = softmax(logits_k / τ)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)

                # H_k = -Σ_{v∈V} π(v) log π(v)  (per-position Shannon entropy)
                token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, L_i)

                # Mask padding positions (labels == -100)
                label_mask = (labels != -100).float()
                seq_lengths = label_mask.sum(dim=-1).clamp(min=1)  # L_i for each sample

                # Eq. 5: S_i = (1/L_i) Σ_{k=1}^{L_i} H_k
                seq_entropy = (token_entropy * label_mask).sum(dim=-1) / seq_lengths
            else:
                # Auto-regressive: generate and compute H_k step by step
                generated = unwrapped.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # generated.scores: tuple of (B, |V|) logits, one per step k
                if generated.scores:
                    step_entropies = []
                    for score in generated.scores:
                        if temperature != 1.0:
                            score = score / temperature
                        lp = torch.nn.functional.log_softmax(score, dim=-1)
                        p = torch.exp(lp)
                        # H_k = -Σ_v π(v) log π(v)
                        h = -(p * lp).sum(dim=-1)  # (B,)
                        step_entropies.append(h)
                    stacked = torch.stack(step_entropies, dim=1)  # (B, T)
                    # S_i = (1/L_i) Σ_k H_k
                    seq_entropy = stacked.mean(dim=1)
                else:
                    seq_entropy = torch.zeros(input_ids.shape[0], device=device)

            all_entropies.append(seq_entropy.cpu().numpy())

    # Build indices from actual accumulated counts to handle variable-size last batch
    all_indices = []
    cumulative = 0
    for ent in all_entropies:
        batch_size = len(ent)
        all_indices.append(np.arange(cumulative, cumulative + batch_size))
        cumulative += batch_size

    entropies = np.concatenate(all_entropies)
    indices = np.concatenate(all_indices)
    return entropies, indices


def select_hard_samples(
    entropies: np.ndarray,
    percentile: float = 70.0,
):
    """Select hard samples M = {i | S_i >= γ_p} (Eq. 6).

    γ_p is the p-th percentile of the difficulty scores {S_i}_{i=1}^{N}.
    Samples with higher entropy (more uncertain predictions) are considered harder.

    Args:
        entropies: Per-sample entropy scores {S_i}, shape (N,).
        percentile: p, the percentile threshold for γ_p.

    Returns:
        selected_indices: np.ndarray of indices i where S_i >= γ_p.
        threshold: γ_p, the computed threshold value.
    """
    # γ_p = p-th percentile of {S_i}
    threshold = np.percentile(entropies, percentile)

    # M = {i | S_i >= γ_p}  (Eq. 6)
    selected_indices = np.where(entropies >= threshold)[0]

    logger.info(
        f"Hard sample mining (Eq. 6): γ_{percentile:.0f} = {threshold:.4f}, "
        f"selected |M| = {len(selected_indices)}/{len(entropies)} samples "
        f"({100 * len(selected_indices) / len(entropies):.1f}%)"
    )
    return selected_indices, threshold
