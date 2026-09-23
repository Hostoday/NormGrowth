"""Utilities for efficacy-matched inner-loop scheduling experiments.

The helpers in this module are deliberately independent of a particular
editor.  They operate on the teacher-forced log probabilities already
computed by ``compute_z`` and therefore do not change tokenization or the
evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


VALID_INNER_MARGIN_SCHEDULES = {
    "legacy",
    "fixed_stop",
    "delayed_rgr",
    "two_stage",
}


@dataclass(frozen=True)
class TargetMarginStats:
    """Differentiable target-vs-best-alternative margin statistics."""

    token_margin: torch.Tensor
    mask: torch.Tensor
    prompt_mean: torch.Tensor
    prompt_min: torch.Tensor
    mean: torch.Tensor
    minimum: torch.Tensor

    def hinge_floor(self, threshold: float) -> torch.Tensor:
        """Mean hinge needed to keep every supervised token above a floor."""

        deficit = torch.relu(
            torch.as_tensor(
                threshold,
                device=self.token_margin.device,
                dtype=self.token_margin.dtype,
            )
            - self.token_margin
        )
        return (deficit * self.mask).sum() / self.mask.sum().clamp_min(1.0)


def target_margin_stats(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
) -> TargetMarginStats:
    """Compute target-token margins without constructing a one-hot tensor.

    ``log_probs`` has shape ``[batch, sequence, vocabulary]`` and ``targets``
    has shape ``[batch, sequence]`` with ``-100`` at unsupervised positions.
    The competitor is the highest-log-probability token other than the target
    at each supervised position.  Log-probability differences equal logit
    differences, so this is the usual target-vs-best-alternative margin.
    """

    if log_probs.ndim != 3:
        raise ValueError(
            f"log_probs must have rank 3, got shape={tuple(log_probs.shape)}"
        )
    if targets.ndim != 2 or tuple(targets.shape) != tuple(log_probs.shape[:2]):
        raise ValueError(
            "targets must have shape [batch, sequence] matching log_probs; "
            f"got targets={tuple(targets.shape)} log_probs={tuple(log_probs.shape)}"
        )
    if log_probs.shape[-1] < 2:
        raise ValueError("target margins require a vocabulary of at least two")

    # In model-parallel runs the rewrite targets are created on the embedding
    # device, while ``log_probs`` can live on the (later) loss-layer device.
    # ``gather`` requires both tensors to be colocated.  Moving the small index
    # tensor here keeps callers simple and does not affect its integer values.
    targets = targets.to(device=log_probs.device)
    mask = targets.ne(-100)
    if not bool(mask.any()):
        raise ValueError("targets contain no supervised tokens")
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    target_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=safe_targets.unsqueeze(2),
    ).squeeze(2)

    top_values, top_indices = torch.topk(log_probs, k=2, dim=2)
    best_other = torch.where(
        top_indices[..., 0].eq(safe_targets),
        top_values[..., 1],
        top_values[..., 0],
    )
    token_margin = target_log_probs - best_other
    float_mask = mask.to(dtype=token_margin.dtype)
    prompt_counts = float_mask.sum(dim=1).clamp_min(1.0)
    prompt_mean = (token_margin * float_mask).sum(dim=1) / prompt_counts
    prompt_min = token_margin.masked_fill(~mask, torch.inf).min(dim=1).values

    return TargetMarginStats(
        token_margin=token_margin,
        mask=float_mask,
        prompt_mean=prompt_mean,
        prompt_min=prompt_min,
        mean=prompt_mean.mean(),
        minimum=prompt_min.min(),
    )


def validate_inner_margin_schedule(
    mode: str,
    threshold: float,
    refinement_steps: int,
    hinge_weight: float,
) -> Dict[str, object]:
    """Normalize and validate an opt-in inner-loop schedule configuration."""

    normalized = str(mode).strip().lower()
    if normalized not in VALID_INNER_MARGIN_SCHEDULES:
        raise ValueError(
            f"Unknown inner margin schedule {mode!r}; expected one of "
            f"{sorted(VALID_INNER_MARGIN_SCHEDULES)}"
        )
    if not torch.isfinite(torch.tensor(float(threshold))):
        raise ValueError("inner target-margin threshold must be finite")
    if int(refinement_steps) < 0:
        raise ValueError("inner refinement steps must be non-negative")
    if normalized in {"delayed_rgr", "two_stage"} and int(refinement_steps) <= 0:
        raise ValueError(f"{normalized} requires positive refinement steps")
    if not torch.isfinite(torch.tensor(float(hinge_weight))):
        raise ValueError("inner margin hinge weight must be finite")
    if float(hinge_weight) < 0.0:
        raise ValueError("inner margin hinge weight must be non-negative")
    if normalized == "two_stage" and float(hinge_weight) <= 0.0:
        raise ValueError("two_stage requires a positive margin hinge weight")
    return {
        "mode": normalized,
        "threshold": float(threshold),
        "refinement_steps": int(refinement_steps),
        "hinge_weight": float(hinge_weight),
    }
