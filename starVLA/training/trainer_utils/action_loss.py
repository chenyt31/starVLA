"""Shared action-horizon masking helpers."""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def build_action_valid_mask(
    examples: List[dict],
    *,
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a [B, H] mask for real (non-padded) action targets."""

    masks = []
    for example in examples:
        raw_mask = example.get("action_valid_mask")
        if raw_mask is None:
            mask = np.ones(horizon, dtype=np.float32)
        else:
            mask = np.asarray(raw_mask, dtype=np.float32).reshape(-1)
            if len(mask) != horizon:
                raise ValueError(
                    "action_valid_mask must match action_horizon: "
                    f"got {len(mask)} for horizon {horizon}"
                )
        masks.append(mask)
    return torch.as_tensor(np.stack(masks, axis=0), device=device, dtype=dtype)


def compute_masked_action_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    action_valid_mask: torch.Tensor,
    dimension_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one L1 loss per sample, ignoring only boundary padding.

    ``dimension_weights`` optionally weights action coordinates before their
    per-frame reduction.  Its mean scale is normalized away, so all-ones
    remains exactly equivalent to the historical unweighted loss.
    """

    if predictions.shape != targets.shape:
        raise ValueError(f"prediction/target shape mismatch: {predictions.shape} vs {targets.shape}")
    if action_valid_mask.shape != predictions.shape[:2]:
        raise ValueError(
            "action_valid_mask must have shape [B, H]: "
            f"got {action_valid_mask.shape} for predictions {predictions.shape}"
        )
    absolute_error = torch.abs(predictions - targets)
    if dimension_weights is None:
        per_step_loss = absolute_error.mean(dim=-1)
    else:
        weights = torch.as_tensor(
            dimension_weights, device=absolute_error.device, dtype=absolute_error.dtype
        ).reshape(-1)
        if weights.shape[0] != predictions.shape[-1]:
            raise ValueError(
                "dimension_weights must have one value per action dimension: "
                f"got {weights.shape[0]} for action dim {predictions.shape[-1]}"
            )
        if not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("dimension_weights must be finite, non-negative, and sum to > 0")
        per_step_loss = (absolute_error * weights).sum(dim=-1) / weights.sum()
    valid = action_valid_mask.to(dtype=per_step_loss.dtype)
    denominator = valid.sum(dim=1).clamp_min(1.0)
    return (per_step_loss * valid).sum(dim=1) / denominator
