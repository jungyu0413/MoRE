"""PPO (Proximal Policy Optimization) for trajectory prediction.

Implementation notes:
  • Token-level importance-sampling ratio (RLHF/TRL standard).
  • π_old is the policy at rollout time (separate from π_ref).
  • Schulman approx_kl for monitoring / early-stop.
  • True per-step Shannon entropy (-Σ_v π log π) instead of the MC
    estimator at the sampled token.
  • Attention-masked encoder pooling for V(τ_obs).

Clipped surrogate (token-level, Schulman et al. 2017):
  L_clip = -E_t[ min( r_t · A,  clip(r_t, 1-ε, 1+ε) · A ) ]
    r_t = π_θ(a_t | s_t) / π_old(a_t | s_t)
    A   = R - V(τ_obs)   (frozen at rollout time)

KL penalty (RLHF style):
  D_KL(π_θ || π_ref) ≈ E_t[ log π_θ(a_t) - log π_ref(a_t) ]

Value function V(τ_obs):
  Two-layer MLP on attention-masked mean-pooled encoder hidden states.
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """V(τ_obs) — two-layer MLP on mean-pooled encoder output."""

    def __init__(self, hidden_size: int, intermediate_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-masked mean pool over the source sequence.

    h_pool = (Σ_l m_l · h_l) / (Σ_l m_l)
    """
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def compute_advantage(rewards: torch.Tensor, values: torch.Tensor):
    """Single-step advantage A = R - V (sequence-level reward).

    Both advantages and returns are detached so they remain frozen across
    PPO epochs (TRL/OpenAI standard).
    """
    advantages = (rewards - values).detach()
    returns = rewards.detach()
    return advantages, returns


def compute_ppo_loss_token_level(
    new_token_lp: torch.Tensor,    # (B, T) with grad
    old_token_lp: torch.Tensor,    # (B, T) detached
    advantages: torch.Tensor,      # (B,)   detached, broadcast over T
    mask: torch.Tensor,            # (B, T) float, 1 for valid token
    cliprange: float = 0.2,
):
    """Token-level clipped PPO surrogate.

    L_clip = -mean_t[ min(r_t · A, clip(r_t, 1±ε) · A) ]
      r_t = exp(log π_θ(a_t) - log π_old(a_t))

    The advantage is sequence-level (one scalar per sample) and is broadcast
    across the token dimension. The mean is taken over *valid* tokens only.
    """
    log_ratio = (new_token_lp - old_token_lp) * mask
    ratio = torch.exp(log_ratio)

    adv = advantages.detach().unsqueeze(-1).expand_as(ratio)

    pg_unclipped = ratio * adv
    pg_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * adv
    pg_per_token = -torch.min(pg_unclipped, pg_clipped)

    valid = mask.sum().clamp(min=1.0)
    policy_loss = (pg_per_token * mask).sum() / valid

    with torch.no_grad():
        clipped = ((ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))).float()
        clip_frac = (clipped * mask).sum() / valid
        # Schulman's low-variance estimator: KL ≈ E[(r-1) - log r]
        approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / valid

    return policy_loss, ratio, clip_frac, approx_kl


def compute_kl_penalty_token_level(
    new_token_lp: torch.Tensor,    # (B, T) with grad
    ref_token_lp: torch.Tensor,    # (B, T) detached
    mask: torch.Tensor,            # (B, T) float
):
    """E[KL(π_θ || π_ref)] estimator over valid tokens.

    Schulman's low-variance k3 estimator (always non-negative, unbiased):
      KL ≈ mean_t[ (r_t - 1) - log r_t ],   r_t = π_θ(a_t)/π_ref(a_t)
    Compared to the naive MC log-ratio (which can be negative on a single
    sample), this has the same expectation but markedly lower variance.
    """
    log_ratio = (new_token_lp - ref_token_lp) * mask
    ratio = torch.exp(log_ratio)
    kl_per_token = (ratio - 1.0) - log_ratio
    valid = mask.sum().clamp(min=1.0)
    return (kl_per_token * mask).sum() / valid


def compute_entropy_token_level(
    log_probs_full: torch.Tensor,  # (B, T, V) with grad
    mask: torch.Tensor,            # (B, T) float
):
    """True per-step Shannon entropy averaged over valid tokens.

    H_t = -Σ_v π(v|·) log π(v|·)
    """
    p = torch.exp(log_probs_full)
    h_per_step = -(p * log_probs_full).sum(dim=-1)  # (B, T)
    valid = mask.sum().clamp(min=1.0)
    return (h_per_step * mask).sum() / valid
