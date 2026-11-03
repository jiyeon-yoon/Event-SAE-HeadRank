"""Smooth final-head Taylor kernels, separate from the native BF16 replay.

The reference policy still chooses each fixed target token using its original
arithmetic.  These functions differentiate a continuous FP32 surrogate of the
same final RMSNorm and unembedding weights.  They do not differentiate BF16
rounding, regenerate an action, or replace a residual with its SAE reconstruction.

Float64 inputs are retained for numerical verification.  All other floating
inputs are evaluated in FP32.  No decoder direction or hidden vector is
unit-normalized.  Model weights are never changed in place.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _floating(value: torch.Tensor, name: str, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and floating point")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} cannot have an empty dimension")


def _common(values: list[torch.Tensor]) -> list[torch.Tensor]:
    if any(value.device != values[0].device for value in values):
        raise ValueError("all floating inputs must be on the same device")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32
    return [value.to(dtype=dtype) for value in values]


def _head_inputs(hidden, norm_weight, head_weight, eps, bias):
    _floating(hidden, "hidden", 2)
    _floating(norm_weight, "norm_weight", 1)
    _floating(head_weight, "head_weight", 2)
    if norm_weight.shape[0] != hidden.shape[1] or head_weight.shape[1] != hidden.shape[1]:
        raise ValueError("hidden, norm_weight and head_weight widths must agree")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    tensors = [hidden, norm_weight, head_weight]
    if bias is not None:
        _floating(bias, "bias", 1)
        if bias.shape[0] != head_weight.shape[0]:
            raise ValueError("bias must have one value per vocabulary token")
        tensors.append(bias)
    converted = _common(tensors)
    return (*converted[:3], converted[3] if bias is not None else None)


def _targets(target_ids, hidden, vocabulary_size):
    if (not isinstance(target_ids, torch.Tensor) or target_ids.ndim != 1
            or target_ids.shape[0] != hidden.shape[0]
            or target_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError("target_ids must be a batch-length integer tensor")
    if target_ids.device != hidden.device:
        raise ValueError("target_ids must be on the hidden device")
    if bool(((target_ids < 0) | (target_ids >= vocabulary_size)).any()):
        raise ValueError("target_ids are outside the full vocabulary")


def _gradient_from_logits(hidden, norm_weight, head_weight, inverse_rms, target_ids, logits):
    logit_gradient = logits.softmax(dim=-1)
    logit_gradient = logit_gradient.scatter_add(
        1, target_ids.to(dtype=torch.int64).unsqueeze(1),
        -torch.ones((hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device),
    )
    normalized_gradient = (logit_gradient @ head_weight) * norm_weight
    gradient = (inverse_rms * normalized_gradient
                - hidden * inverse_rms.pow(3)
                * (normalized_gradient * hidden).mean(dim=-1, keepdim=True))
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("smooth head produced a nonfinite hidden gradient")
    return gradient


class PreparedTaylorHead:
    """Validate/convert frozen weights once and reuse logits for the gradient.

    ``evaluate(hidden, target_ids)`` returns ``(logits, gradient)``.  Weights
    remain on their original common device and use FP32 unless any weight is
    FP64.  FP64 verification therefore requires constructing with FP64 weights;
    a FP64 hidden passed to a FP32 prepared head is rejected, not downcast.

    Prepared tensors are immutable by contract.  Evaluation validates each new
    hidden and target, but never rescans or converts the full unembedding.
    """

    def __init__(self, norm_weight, head_weight, eps, bias=None):
        _floating(norm_weight, "norm_weight", 1)
        dummy_hidden = norm_weight.new_zeros((1, norm_weight.shape[0]))
        _, norm_weight, head_weight, bias = _head_inputs(
            dummy_hidden, norm_weight, head_weight, eps, bias
        )
        self.norm_weight = norm_weight.detach()
        self.head_weight = head_weight.detach()
        self.bias = None if bias is None else bias.detach()
        self.eps = eps
        self.dtype = self.head_weight.dtype
        self.device = self.head_weight.device

    def evaluate(self, hidden, target_ids):
        _floating(hidden, "hidden", 2)
        if hidden.shape[1] != self.head_weight.shape[1]:
            raise ValueError("hidden width must match the prepared head")
        if hidden.device != self.device:
            raise ValueError("hidden must be on the prepared head device")
        if hidden.dtype == torch.float64 and self.dtype != torch.float64:
            raise ValueError("FP64 hidden requires FP64 prepared weights")
        hidden = hidden.to(dtype=self.dtype)
        _targets(target_ids, hidden, self.head_weight.shape[0])
        inverse_rms = torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + self.eps)
        logits = F.linear((hidden * inverse_rms) * self.norm_weight, self.head_weight, self.bias)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("smooth head produced nonfinite logits")
        gradient = _gradient_from_logits(hidden, self.norm_weight, self.head_weight,
                                         inverse_rms, target_ids, logits)
        return logits, gradient


def smooth_head_logits(hidden, norm_weight, head_weight, eps, bias=None):
    """Return continuous final-head logits, without a post-normalization BF16 cast.

    Args have shapes ``[B,D]``, ``[D]``, ``[V,D]`` and optional ``[V]``.
    Unlike the native output-head bundle, this function allows autograd.
    """
    hidden, norm_weight, head_weight, bias = _head_inputs(
        hidden, norm_weight, head_weight, eps, bias
    )
    inverse_rms = torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + eps)
    logits = F.linear((hidden * inverse_rms) * norm_weight, head_weight, bias)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("smooth head produced nonfinite logits")
    return logits


def token_nll_gradient(hidden, norm_weight, head_weight, eps, target_ids, bias=None):
    """Analytically differentiate fixed-token full-vocabulary negative log likelihood.

    ``target_ids`` is an integer ``[B]`` tensor: the *native reference* target,
    not a target reselected by the FP32 surrogate or by an edited condition.
    The returned ``[B,D]`` gradient is per example, with no batch reduction.
    """
    hidden, norm_weight, head_weight, bias = _head_inputs(
        hidden, norm_weight, head_weight, eps, bias
    )
    _targets(target_ids, hidden, head_weight.shape[0])
    inverse_rms = torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + eps)
    logits = F.linear((hidden * inverse_rms) * norm_weight, head_weight, bias)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("smooth head produced nonfinite logits")
    return _gradient_from_logits(hidden, norm_weight, head_weight, inverse_rms, target_ids, logits)


def feature_taylor_scores(gradient, directions, activations):
    """Return ``abs(z_ni * <grad_n, d_i>)`` for each example and feature.

    Shapes are ``gradient[B,D]``, ``directions[K,D]`` and ``activations[B,K]``.
    The caller averages these nonnegative *per-example* values.  Averaging
    signed gradients before taking the absolute value is a different metric.
    ``directions`` must be the actual SAE decoder vectors, not unit directions.
    """
    _floating(gradient, "gradient", 2)
    _floating(directions, "directions", 2)
    _floating(activations, "activations", 2)
    if (directions.shape[1] != gradient.shape[1]
            or activations.shape != (gradient.shape[0], directions.shape[0])):
        raise ValueError("gradient, directions and activations shapes do not agree")
    gradient, directions, activations = _common([gradient, directions, activations])
    scores = (activations * (gradient @ directions.T)).abs()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Taylor contribution produced nonfinite values")
    return scores
