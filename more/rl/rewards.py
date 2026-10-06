"""Multi-Expert Consensus Reward Modeling (Section 3.2).

Implements the composite reward function:
  R = R_GT + λ_exp · R_exp                                              (Eq. 3)

where R_exp uses Uncertainty-Weighted Optimization (UWO) [Coste et al., ICLR 2024]
to aggregate rewards from K=5 heterogeneous expert models.

  Expert reward:   R_k = -MSE(Ŝ, S_k) = -(1/T) Σ_t ||ŝ_t - s_k_t||²   (Eq. 1)
  UWO consensus:   R_exp = μ - λ_uwo · σ                                 (Eq. 2)
                   where μ = (1/K) Σ_k R_k,  σ = √(Var(R_1,...,R_K))
  GT reward:       R_GT = -MSE(Ŝ, S_GT) = -(1/T) Σ_t ||ŝ_t - s_gt_t||² (Eq. 3)
"""

import numpy as np
import torch

from more.data.converter import batch_text2traj
from more.data.homography import image2world

# K=5 heterogeneous expert models (Section 3.2):
#   (i)   local interaction   → stg (Social-STGCNN)
#   (ii)  structural relation → dmr (DMRGCN)
#   (iii) social grouping     → gpg (GPGraph)
#   (iv)  motion pattern      → sig (SingularTrajectory)
#   (v)   goal expert         → exp (ExpertTraj / Goal-Example)
EXPERT_NAMES = ('stg', 'dmr', 'gpg', 'sig', 'exp')


def compute_reward_from_experts(
    decoded_preds,
    batch_scene_ids,
    homography,
    expert_preds_dict,
    pred_len,
    device,
    gt_traj_meter=None,
    uwo_lambda=0.5,
    ensemble_weight=0.5,
    reward_fail_penalty=-10.0,
    seq_start_end=None,
    scene_id=None,
    step=None,
    cfg=None,
    accelerator=None,
    expert_weights=None,
    gt_weight=1.0,
):
    """Compute composite reward R = R_GT + λ_exp · R_exp (Eq. 3).

    Args:
        decoded_preds: List[str] - Decoded text predictions from the policy π_θ.
        batch_scene_ids: Scene IDs for coordinate conversion.
        homography: Dict mapping scene_id -> (3,3) homography matrix.
        expert_preds_dict: Dict[str, Tensor] - Expert predictions {S_k} keyed by
            short name in EXPERT_NAMES.
        pred_len: T_pred, prediction horizon length.
        device: torch device.
        gt_traj_meter: S_GT, ground-truth trajectories in meter coordinates.
        uwo_lambda: λ_uwo, uncertainty penalty coefficient (Eq. 2).
        ensemble_weight: λ_exp, expert ensemble weight (Eq. 3).
        reward_fail_penalty: Penalty value for unparseable trajectories.

    Returns:
        rewards: Tensor of shape (batch_size,), R for each sample.
    """
    policy_traj_pixel_list = batch_text2traj(decoded_preds, frame=pred_len, dim=2)
    batch_size = len(policy_traj_pixel_list)

    # Resolve scene IDs for homography lookup
    if batch_scene_ids is None and seq_start_end is not None:
        batch_scene_ids = []
        for pid in range(batch_size):
            global_idx = step * cfg.per_device_train_batch_size * accelerator.num_processes + pid
            scene_id_val = None
            for s_id, (s, e) in enumerate(seq_start_end):
                if s <= global_idx < e:
                    scene_id_val = scene_id[s]
                    break
            batch_scene_ids.append(scene_id_val if scene_id_val is not None else (scene_id[0] if len(seq_start_end) > 0 else None))
    elif batch_scene_ids is not None:
        if isinstance(batch_scene_ids, torch.Tensor):
            batch_scene_ids = [batch_scene_ids[i].item() for i in range(batch_size)]
        else:
            batch_scene_ids = list(batch_scene_ids)
    else:
        batch_scene_ids = [None] * batch_size

    needs_homography = cfg is not None and getattr(cfg, 'other_model_coordinate', None) == 'meter'

    rewards_list = []
    for bid in range(batch_size):
        # Failed trajectory parsing → penalty
        if policy_traj_pixel_list[bid] is None:
            rewards_list.append(reward_fail_penalty)
            continue

        # Convert policy trajectory Ŝ to meter coordinates
        if needs_homography:
            H = homography[batch_scene_ids[bid]]
            policy_traj_meter = image2world(policy_traj_pixel_list[bid], H)
        else:
            policy_traj_meter = policy_traj_pixel_list[bid]

        # Ŝ: policy predicted trajectory, shape (T_pred, 2)
        policy_traj = torch.from_numpy(
            np.asarray(policy_traj_meter, dtype=np.float32)
        ).to(device)

        # ── Expert rewards: R_k = -MSE(Ŝ, S_k)  (Eq. 1) ──────────────
        # R_k = -(1/T) Σ_{t=1}^{T} ||ŝ_t - s_k_t||²
        expert_rewards = []
        expert_w_list = []
        for ename in EXPERT_NAMES:
            epreds = expert_preds_dict.get(ename)
            if epreds is None or bid >= len(epreds):
                continue
            expert_pred = epreds[bid]  # S_k for expert k
            if isinstance(expert_pred, torch.Tensor):
                if expert_pred.ndim == 3:
                    expert_pred = expert_pred[0]
                expert_tensor = expert_pred.to(device)
            else:
                ep_np = np.asarray(expert_pred)
                if ep_np.ndim == 3:
                    ep_np = ep_np[0]
                expert_tensor = torch.from_numpy(ep_np.astype(np.float32)).to(device)

            sq_dist = torch.sum((policy_traj - expert_tensor) ** 2, dim=-1)
            mse = torch.mean(sq_dist)
            expert_rewards.append(-mse)
            # Per-expert weight for the weighted UWO consensus
            if expert_weights and ename in expert_weights:
                expert_w_list.append(expert_weights[ename])
            else:
                expert_w_list.append(1.0)

        # No expert predictions available → penalty
        if len(expert_rewards) == 0:
            rewards_list.append(reward_fail_penalty)
            continue

        # ── Weighted UWO consensus: R_exp = μ - λ_uwo · σ  (Eq. 2) ──
        # μ = Σ(w_k · R_k),  σ = √(Σ(w_k · (R_k - μ)²))
        R_stack = torch.stack(expert_rewards)
        w_tensor = torch.tensor(expert_w_list, dtype=torch.float32, device=device)
        w_tensor = w_tensor / (w_tensor.sum() + 1e-8)  # normalize
        mu = torch.sum(w_tensor * R_stack)
        sigma = torch.sqrt(torch.sum(w_tensor * (R_stack - mu) ** 2) + 1e-8)
        R_exp = mu - float(uwo_lambda) * sigma

        # ── GT reward: R_GT = -MSE(Ŝ, S_GT)  (Eq. 3) ─────────────────
        # R_GT = -(1/T) Σ_{t=1}^{T} ||ŝ_t - s_gt_t||²
        R_gt_val = 0.0
        if gt_traj_meter is not None and bid < len(gt_traj_meter):
            gt = gt_traj_meter[bid]  # S_GT
            gt_tensor = (
                gt.to(device) if isinstance(gt, torch.Tensor)
                else torch.from_numpy(np.asarray(gt, dtype=np.float32)).to(device)
            )
            sq_dist_gt = torch.sum((policy_traj - gt_tensor) ** 2, dim=-1)
            R_gt_val = -torch.mean(sq_dist_gt).item()
        elif gt_traj_meter is not None:
            # GT expected but missing for this sample → penalty
            R_gt_val = reward_fail_penalty

        # ── Composite reward: R = w_gt · R_GT + λ_exp · R_exp  (Eq. 3) ──────
        R = gt_weight * R_gt_val + ensemble_weight * R_exp.item()
        rewards_list.append(R)

    return torch.tensor(rewards_list, dtype=torch.float32, device=device)
