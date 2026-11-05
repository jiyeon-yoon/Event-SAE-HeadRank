"""Taylor importance at the frozen residual, not at an SAE reconstruction.

Targets are the current validated native-head greedy vocabulary IDs at each
saved prefix. They are not asserted to be historical generated action tokens.
The derivative uses a continuous FP32 final-head surrogate; it is not a
derivative of BF16 rounding and is not a task-success gradient.
"""
from __future__ import annotations

import math
import time

import torch
from torch.nn import functional as F

from event_sae.research.head_rank.scoring import _parity, query_index
from event_sae.research.output_head.sensitivity import (
    _validate_sparse_cache, aggregation_weights, edit_feature_reference, full_vocab_kl,
)
from .kernels import PreparedTaylorHead, feature_taylor_scores


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decoder_vectors(sae, feature_ids, dict_size, hidden_dim, *, device, block_size=32):
    """Actual decoder columns, bias removed, NEVER unit-normalized.

The pinned BatchTopK decoder is nn.Linear(F,D,bias=False) plus b_dec.
Read the declared weight columns directly. decode(unit)-decode(0) is unsafe:
adding/subtracting a large bias in FP32 can erase small column coefficients.
The public decode API is checked against the declared dense graph below.
"""
    decoder, bias = getattr(sae, "decoder", None), getattr(sae, "b_dec", None)
    if (type(decoder) is not torch.nn.Linear or decoder.bias is not None
            or decoder.in_features != dict_size or decoder.out_features != hidden_dim):
        raise ValueError("Taylor requires the verified bias-free nn.Linear BatchTopK decoder")
    if (not isinstance(bias, torch.Tensor) or bias.shape != (hidden_dim,)
            or bias.dtype != torch.float32 or decoder.weight.dtype != torch.float32
            or bias.device != torch.device(device) or decoder.weight.device != torch.device(device)
            or not torch.isfinite(decoder.weight).all() or not torch.isfinite(bias).all()):
        raise ValueError("SAE decoder weights/b_dec must be finite FP32 on the scoring device")
    if (not feature_ids or len(set(feature_ids)) != len(feature_ids)
            or any(type(fid) is not int or not 0 <= fid < dict_size for fid in feature_ids)):
        raise ValueError("Invalid decoder feature columns")
    positions = torch.tensor(feature_ids, dtype=torch.int64, device=device)
    vectors = decoder.weight.detach().T.index_select(0, positions).contiguous()
    return vectors, bias.detach()[None, :]


@torch.inference_mode()
def check_decoder_affine(sae, z, row_values, directions, intercept):
    """Separate real decoder mismatch from dense/sparse FP32 summation error.

1. Compare public decode with the SAME dense FP32 graph (including bias).
2. Check selected-column equivalence in FP64, with unchanged strict tolerances.
3. Record, but do not misclassify, differently ordered FP32 sparse summation.
No native-head/edit parity tolerance is modified.
"""
    actual = sae.decode(z)
    declared = F.linear(z, sae.decoder.weight) + intercept
    if (actual.shape != declared.shape or not torch.isfinite(actual).all()
            or not torch.equal(actual, declared)):
        raise ValueError("SAE decode differs from its declared affine dense graph")
    sparse = row_values @ directions + intercept
    dense64 = F.linear(z.double(), sae.decoder.weight.double()) + intercept.double()
    sparse64 = row_values.double() @ directions.double() + intercept.double()
    if not torch.allclose(dense64, sparse64, rtol=1e-10, atol=1e-10):
        raise ValueError("SAE decoder column mapping is not consistent in FP64")
    return {"fp32_sparse_vs_native_max_abs_error": float((actual - sparse).abs().max()),
            "fp64_column_max_abs_error": float((dense64 - sparse64).abs().max())}


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@torch.inference_mode()
def score_cache(cache, head, sae, *, max_readouts=20000, max_active_pairs=1000000,
                current_identity=None, parity_reports=None, progress=None):
    """One smooth hidden gradient per readout, with inactive zeros in denominators.

No feature screening, re-encoding, outcome loading, editing or generation.
The active universe is from the entire provided cache (whole-episode smoke
subsets are separately and explicitly marked by the workflow).
"""
    hidden, records, ids, values, width = _validate_sparse_cache(cache)
    if cache.get("encoder_grouping") != "original_forward":
        raise ValueError("Taylor requires lossless original_forward latents")
    synthetic = _parity(cache, head, current_identity, parity_reports)
    query_index(records)  # Complete seven-position queries before any head call.
    _integer(max_readouts, "max_readouts")
    _integer(max_active_pairs, "max_active_pairs")
    pairs = sum(len(row) for row in ids)
    if len(records) > max_readouts or pairs > max_active_pairs:
        raise ValueError("Taylor readout/active-pair budget exceeded before model execution")
    active_ids = sorted({int(fid) for row in ids for fid in row.tolist()})
    if not active_ids:
        raise ValueError("Taylor requires at least one observed active feature")
    if hidden.shape[1] != head.head_weight.shape[1]:
        raise ValueError("Readout and head hidden widths differ")
    device = head.head_weight.device
    _sync(device)
    started = time.perf_counter()
    vectors, intercept = decoder_vectors(sae, active_ids, width, hidden.shape[1], device=device)
    # Freeze a single FP32 copy, no per-readout scan/conversion of the large W.
    smooth = PreparedTaylorHead(head.norm.weight, head.head_weight,
                                head.norm.variance_epsilon, head.head_bias)
    _sync(device)
    preparation_seconds = time.perf_counter() - started
    feature_index = {fid: index for index, fid in enumerate(active_ids)}
    weights, _ = aggregation_weights(records)
    totals = torch.zeros(len(active_ids), dtype=torch.float64)
    counts = torch.zeros(len(active_ids), dtype=torch.int64)
    frequency = torch.zeros(len(active_ids), dtype=torch.float64)
    edit_norm = torch.zeros(len(active_ids), dtype=torch.float64)
    max_affine_error, max_fp64_column_error, affine_checks = 0., 0., 0
    max_logit_error, target_mismatches, targets_outside_canonical = 0., 0, 0
    executed = 0
    _sync(device)
    scoring_started = time.perf_counter()
    for index, record in enumerate(records):
        h = hidden[index:index + 1].to(device=device, dtype=head.hidden_dtype)
        native_logits = head(h)
        target = native_logits.argmax(-1)
        smooth_logits, gradient = smooth.evaluate(h.float(), target)
        max_logit_error = max(max_logit_error, float((native_logits.float() - smooth_logits).abs().max()))
        target_mismatches += int((smooth_logits.argmax(-1) != target).sum())
        if native_logits.shape[-1] == 32064:
            # Diagnostic for the documented pinned encoder support, NOT a
            # physical-action mapping or validation of generated sequences.
            targets_outside_canonical += int(((target < 31744) | (target >= 32000)).sum())
        row_ids = ids[index].tolist()
        positions = [feature_index[fid] for fid in row_ids]
        if positions:
            positions_device = torch.tensor(positions, device=device, dtype=torch.int64)
            directions = vectors[positions_device]
            z_values = values[index].to(device)[None, :]
            scores = feature_taylor_scores(gradient, directions, z_values)[0].double().cpu()
            totals[positions] += weights[index] * scores
            frequency[positions] += weights[index]
            edit_norm[positions] += weights[index] * (z_values[0] * directions.norm(dim=-1)).double().cpu()
            counts[positions] += 1
            executed += len(positions)
            # Deterministic actual-value checks, without re-encoding or using
            # decoder reconstruction in place of the original residual.
            if index < 7 or (index + 1) % 1000 == 0:
                z = torch.zeros((1, width), dtype=torch.float32, device=device)
                z[0, ids[index].to(device)] = values[index].to(device)
                audit = check_decoder_affine(sae, z, z_values, directions, intercept)
                max_affine_error = max(max_affine_error, audit["fp32_sparse_vs_native_max_abs_error"])
                max_fp64_column_error = max(max_fp64_column_error, audit["fp64_column_max_abs_error"])
                affine_checks += 1
        if progress and ((index + 1) % 100 == 0 or index + 1 == len(records)):
            _sync(device)
            elapsed = time.perf_counter() - scoring_started
            progress({"readouts": index + 1, "total_readouts": len(records),
                      "executed_pairs": executed, "total_pairs": pairs,
                      "elapsed_seconds": elapsed,
                      "estimated_remaining_seconds": elapsed * (len(records) - index - 1) / (index + 1)})
    _sync(device)
    scoring_seconds = time.perf_counter() - scoring_started
    if executed != pairs:
        raise RuntimeError("Taylor executed-pair count differs from preflight")
    rows = [{"feature_id": fid, "taylor_abs_nll": float(totals[i]),
             "num_active_readouts": int(counts[i]), "num_readouts": len(records),
             "sample_activation_frequency": float(frequency[i]),
             "sample_intended_edit_norm": float(edit_norm[i])}
            for i, fid in enumerate(active_ids)]
    if any(not math.isfinite(row["taylor_abs_nll"]) for row in rows):
        raise ValueError("Nonfinite aggregated Taylor score")
    return {"schema_version": "head_taylor_scores_v1", "synthetic": synthetic, "features": rows,
            "scored_feature_ids": active_ids, "dictionary_size": width,
            "aggregation": "equal_task_episode_step_dimension_inactive_zero",
            "definition": "mean_abs(z_j * dot(d_j, grad_h[-log p_fp32(y_native|h_raw)]))",
            "target_definition": "current_validated_native_head_full_vocab_greedy_at_frozen_prefix",
            "gradient_definition": "continuous_fp32_rmsnorm_and_unembedding_same_frozen_weights",
            "not_claimed": ["exact derivative of BF16 rounding", "historical generated token identity",
                            "task loss or success gradient", "physical-action error", "novel Taylor principle"],
            "diagnostics": {"device": str(device), "readouts": len(records), "active_pairs": pairs,
                            "gradient_head_calls": len(records), "native_target_head_calls": len(records),
                            "native_vs_fp32_max_abs_logit_error": max_logit_error,
                            "native_vs_fp32_argmax_mismatch_rate": target_mismatches / len(records),
                            "native_targets_outside_documented_256_support": targets_outside_canonical
                                if head.head_weight.shape[0] == 32064 else None,
                            "max_decoder_affine_abs_error": max_affine_error,
                            "decoder_vector_source": "declared_nn_linear_weight_columns_no_bias_subtraction",
                            "decoder_affine_check": "exact_native_vs_dense_fp32_graph_and_fp64_column_equivalence",
                            "decoder_affine_checks": affine_checks,
                            "max_decoder_fp64_column_abs_error": max_fp64_column_error,
                            "preparation_seconds": preparation_seconds, "scoring_seconds": scoring_seconds,
                            "total_seconds": preparation_seconds + scoring_seconds},
            "new_rollouts": 0, "sae_training": False, "model_regeneration": False}


@torch.inference_mode()
def benchmark_exact_kl(cache, head, sae, *, max_active_pairs=100000, progress=None):
    """Same-device isolated-row exact KL timing on the SAME smoke readouts.

This is a timing/reference diagnostic, not a new mean/tail ranking. It never
compares a new GPU's timing directly to an old archive's elapsed time.
"""
    hidden, records, ids, values, width = _validate_sparse_cache(cache)
    query_index(records)
    pairs = sum(len(row) for row in ids)
    _integer(max_active_pairs, "max_active_pairs")
    if pairs > max_active_pairs:
        raise ValueError("Exact KL benchmark pair budget exceeded")
    device = head.head_weight.device
    _sync(device)
    started = time.perf_counter()
    checksum, done = 0., 0
    for index in range(len(records)):
        h = hidden[index:index + 1].to(device=device, dtype=head.hidden_dtype)
        z = torch.zeros((1, width), dtype=torch.float32, device=device)
        z[0, ids[index].to(device)] = values[index].to(device)
        base = head(h)
        for fid in ids[index].tolist():
            changed = edit_feature_reference(h, sae, fid, 0., encoded=z)
            checksum += float(full_vocab_kl(base, head(changed))[0])
            done += 1
        if progress and ((index + 1) % 100 == 0 or index + 1 == len(records)):
            progress({"benchmark": "exact_head_kl", "readouts": index + 1,
                      "total_readouts": len(records), "executed_pairs": done, "total_pairs": pairs})
    _sync(device)
    return {"seconds": time.perf_counter() - started, "active_pairs": done,
            "readouts": len(records), "kl_checksum": checksum,
            "scope": "same_device_same_readouts_reference_only_not_behavioral_outcomes"}
