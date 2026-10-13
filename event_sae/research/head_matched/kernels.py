"""Original-forward matched edits and explicit post-cast norm diagnostics.

Norm matching compares two counterfactual edits to THE SAME incoming hidden
row. It does not equate different closed-loop trajectories. A decoder control
borrows the anchor's magnitude/support; it is not ablation of its own latent.
"""
from __future__ import annotations

import math

import torch

from event_sae.research.head_followup.interventions import apply_edit


def validate_norm_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"atol", "rtol"}:
        raise ValueError("norm_policy must explicitly contain only atol and rtol")
    for key, value in policy.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"norm_policy.{key} must be finite and nonnegative")
    return {key: float(policy[key]) for key in ("atol", "rtol")}


@torch.inference_mode()
def matched_edit(hidden, sae, feature_id, *, mode, control_direction=None, norm_policy):
    """Return edited rows plus evidence; the hook aborts any ineligible result.

The operation itself uses the frozen scorer's ``apply_edit`` arithmetic. Every
SAE encoding contains all rows from one original forward; callers must never
split a BatchTopK forward into separately encoded rows. Diagnostics remain
tensors until the evidence writer serializes them, without retaining h or z.
"""
    policy = validate_norm_policy(norm_policy)
    if mode not in {"identity", "anchor", "matched_decoder"}:
        raise ValueError("unsupported matched edit mode")
    if (mode == "matched_decoder") != (control_direction is not None):
        raise ValueError("a control direction is required only for matched_decoder")
    if hidden.ndim != 2 or hidden.shape[0] < 1:
        raise ValueError("nonempty [original-forward rows, hidden width] required")
    # apply_edit/feature_delta validates finite FP32 nonnegative latents.
    z = sae.encode(hidden.float())
    alpha = 1. if mode == "identity" else 0.
    anchor, anchor_metrics = apply_edit(hidden, sae, feature_id, alpha, encoded=z)
    intended = anchor_metrics["intended_norm"]
    anchor_actual = anchor_metrics["realized_norm"]
    active = z[:, feature_id] > 0
    if mode == "matched_decoder":
        vector = torch.as_tensor(control_direction, device=hidden.device, dtype=torch.float32)
        # apply_edit performs the authoritative direction validation.
        result, metrics = apply_edit(hidden, sae, feature_id, alpha, direction=vector, encoded=z)
        unit = vector / vector.norm()
        pre_cast_control = (intended[:, None] * unit[None, :]).norm(dim=-1)
    else:
        result, metrics = anchor, anchor_metrics
        pre_cast_control = intended.clone()
    actual = metrics["realized_norm"]
    pair_error = (actual - anchor_actual).abs()
    anchor_error = (anchor_actual - intended).abs()
    control_error = (actual - intended).abs()
    pre_error = (pre_cast_control - intended).abs()
    # FP32 unit-vector arithmetic is not bit-exact equality; this tight bound
    # only checks numerical construction and never relaxes the BF16 policy.
    # Evaluate the policy in FP64 on the recorded FP32 norm values, exactly as
    # the independent CPU journal does. This avoids a float32-rounded bound
    # approving a marginal row that the journal would subsequently reject.
    intended64, anchor64, actual64 = intended.double(), anchor_actual.double(), actual.double()
    pre_valid = (pre_cast_control.double() - intended64).abs() <= 1e-6 + 1e-5 * intended64
    valid = (pre_valid & ((anchor64 - intended64).abs() <= policy["atol"] + policy["rtol"] * intended64)
             & ((actual64 - intended64).abs() <= policy["atol"] + policy["rtol"] * intended64)
             & ((actual64 - anchor64).abs() <= policy["atol"] + policy["rtol"] * anchor64))
    off_support = (~active) & ((anchor_actual > 0) | (actual > 0) | (intended > 0))
    finite = torch.isfinite(torch.stack((intended, anchor_actual, actual, pre_cast_control), -1)).all(dim=-1)
    valid &= finite & ~off_support
    if mode == "identity":
        valid &= (actual == 0) & (intended == 0)
    relative = torch.where(anchor_actual > 0, pair_error / anchor_actual.clamp_min(torch.finfo(torch.float32).tiny),
                           torch.zeros_like(pair_error))
    return result, {
        "anchor_active": active, "intended_norm": intended,
        "anchor_realized_norm": anchor_actual, "control_realized_norm": actual,
        "control_pre_cast_norm": pre_cast_control,
        "pair_norm_abs_error": pair_error, "pair_norm_rel_error": relative,
        "anchor_cast_abs_error": anchor_error, "control_cast_abs_error": control_error,
        "pre_cast_abs_error": pre_error, "eligible": valid,
        "off_support": off_support, "finite": finite,
        "norm_scope": "per_original_forward_row_same_incoming_hidden",
        "cross_trajectory_norm_match": False,
    }
