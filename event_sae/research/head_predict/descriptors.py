"""Task-by-feature fixed-prefix responses, not physical harm predictions.

Action-bin descriptors condition on the verified canonical action-token support.
They describe existing autoregressive positions; no token is regenerated and no
physical unnormalization, simulator, or outcome label participates in scoring.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from event_sae.research.head_followup.interventions import apply_edit
from event_sae.research.head_followup.metrics import action_metrics
from event_sae.research.head_followup.stage1 import validate_mapping
from event_sae.research.head_rank.scoring import _feature_ids, _integer, _parity, query_index
from event_sae.research.output_head.sensitivity import _validate_sparse_cache


def action_bin_mapping(mapping, head_manifest, vocab_size, *, synthetic=False):
    """Resolve token IDs to native normalized bin centers, including aliases.

    The pinned OpenVLA decoder builds 256 EDGES and hence 255 centers, then
    indexes ``clip(base_vocab_size - token_id - 1, 0, 254)``. In particular,
    IDs 31744 and 31745 share the last center: assigning a linspace directly
    to the 256 token IDs would both reverse the sign and lose that alias.

    CPU fixtures must explicitly declare ``synthetic: true`` and provide a
    ``token_id_to_bin_center`` object with string token-ID keys. Production
    never accepts that escape hatch and requires the verified Stage 1 mapping.
    """
    if not isinstance(mapping, dict):
        raise ValueError("action mapping must be an object")
    ids = mapping.get("action_token_ids")
    if (not isinstance(ids, list) or not ids
            or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("invalid action token mapping")
    ids = sorted(ids)  # Canonical tie-breaking is independent of mapping order.
    if synthetic and mapping.get("synthetic") is True:
        explicit = mapping.get("token_id_to_bin_center")
        if not isinstance(explicit, dict) or set(explicit) != {str(token) for token in ids}:
            raise ValueError("synthetic mapping requires exact token_id_to_bin_center keys")
        centers = [explicit[str(token)] for token in ids]
        rule = "explicit_synthetic_token_id_to_bin_center_not_policy_evidence"
    else:
        validate_mapping(mapping, head_manifest)
        if (mapping.get("synthetic", False) is not False or mapping.get("verified") is not True
                or mapping.get("base_vocab_size") != 32000 or type(mapping.get("base_vocab_size")) is not int
                or mapping.get("n_action_bins") != 256 or type(mapping.get("n_action_bins")) is not int
                or vocab_size != 32064 or mapping.get("head_vocab_size") != vocab_size
                or head_manifest.get("vocab_size") != vocab_size
                or mapping.get("tokenizer_sha256") != mapping.get("evidence", {}).get("tokenizer.json")):
            raise ValueError("verified native action decode dimensions/provenance mismatch")
        # Exactly the pinned decoder's edge->center construction in FP64.
        edges = np.linspace(-1., 1., mapping["n_action_bins"])
        native_centers = (edges[:-1] + edges[1:]) / 2
        indices = [min(max(mapping["base_vocab_size"] - token - 1, 0), len(native_centers) - 1)
                   for token in ids]
        centers = [float(native_centers[index]) for index in indices]
        rule = "center[clip(base_vocab_size-token_id-1,0,n_action_bins-2)]"
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not -1 <= value <= 1 for value in centers):
        raise ValueError("action bin centers must be finite normalized values in [-1,1]")
    return ids, [float(value) for value in centers], {
        "token_id_to_bin_center": {str(token): float(value) for token, value in zip(ids, centers)},
        "decode_rule": rule, "num_action_tokens": len(ids),
        "num_distinct_bin_centers": len(set(centers)),
        "conditional_argmax_scope": "token_identity_including_endpoint_aliases_not_physical_action_identity",
    }


@torch.inference_mode()
def score_descriptor_cache(cache, head, sae, mapping, *, feature_ids=None, max_pairs=5000000,
                           current_identity=None, parity_reports=None, progress=None):
    """Score exact original-forward sparse latents with all inactive zeros.

    Scalar means divide by every sampled readout of the task. Each of the
    seven array entries divides by every sampled query of the task, not by
    active positions. Balanced episode query counts make these equal-episode
    means. ``absolute_action_bin_shift`` is mean(abs(per-readout shift)), not
    abs(mean(shift)); signed shift is E[edited bin] minus E[baseline bin].
    ``mean_edit_norm`` uses realized post-cast hidden differences.

    This low-level kernel defaults to all observed active features. The
    public file workflow defaults to the fixed historical development panel.
    """
    hidden, records, ids, values, width = _validate_sparse_cache(cache)
    if cache.get("encoder_grouping") != "original_forward":
        raise ValueError("descriptors require original_forward sparse latents")
    groups, queries, _ = query_index(records)
    _integer(max_pairs, "max_pairs", 1)
    observed = sorted({int(fid) for row in ids for fid in row.tolist()})
    requested = observed if feature_ids is None else feature_ids
    if isinstance(requested, (set, frozenset)):
        requested = sorted(requested)
    selected = _feature_ids(requested, width, "feature_ids")
    selected_set = set(selected)
    per_row = [sorted(fid for fid in row.tolist() if fid in selected_set) for row in ids]
    planned = sum(map(len, per_row))
    if planned > max_pairs:
        raise ValueError(f"descriptor pair budget exceeded: {planned} > {max_pairs}")
    synthetic = _parity(cache, head, current_identity, parity_reports)
    token_ids, centers, decode = action_bin_mapping(
        mapping, head.manifest, head.head_weight.shape[0], synthetic=synthetic)
    # All shape, sample, budget, mapping and parity checks precede head/decode.
    task_queries = Counter(query["task_id"] for query in queries)
    tasks = sorted(task_queries)
    totals = {(fid, task): {"scalar": [0.] * 6, "positions": [[0.] * 7 for _ in range(6)], "active": 0}
              for fid in selected for task in tasks}
    device = head.head_weight.device
    center_tensor = torch.tensor(centers, dtype=torch.float32, device=device)
    executed = baseline_calls = 0

    def report(completed):
        if progress and (completed % 100 == 0 or completed == len(records)):
            progress({"readouts": completed, "total_readouts": len(records),
                      "executed_pairs": executed, "total_pairs": planned})

    for completed, index in enumerate((row for group in groups for row in group), 1):
        features = per_row[index]
        if not features:
            report(completed)
            continue
        h = hidden[index:index + 1].to(device=device, dtype=head.hidden_dtype)
        z = torch.zeros((1, width), dtype=torch.float32, device=device)
        z[0, ids[index].to(device)] = values[index].to(device)
        base = head(h)
        baseline_calls += 1
        if (not isinstance(base, torch.Tensor) or base.shape != (1, head.head_weight.shape[0])
                or not torch.isfinite(base).all()):
            raise ValueError("baseline head must return finite [1, full vocabulary] logits")
        base_expected = (base[:, token_ids].float().softmax(-1) * center_tensor).sum(-1)
        task, dim = records[index]["task_id"], records[index]["action_dim_index"]
        for fid in features:
            edited, diagnostic = apply_edit(h, sae, fid, 0., encoded=z)
            changed = head(edited)
            metric = action_metrics(base, changed, token_ids)
            signed = float(((changed[:, token_ids].float().softmax(-1) * center_tensor).sum(-1)
                            - base_expected)[0])
            scalar = [float(metric["full_vocab_kl"][0]), float(diagnostic["realized_norm"][0]),
                      float(base.argmax(-1)[0] != changed.argmax(-1)[0]),
                      float(diagnostic["intended_norm"][0]), float(z[0, fid]),
                      float(diagnostic["cast_norm_abs_error"][0])]
            position = [signed, abs(signed), float(metric["conditional_action_argmax_flip"][0]),
                        float(metric["action_conditional_kl"][0]), float(metric["action_mass_abs_change"][0]),
                        float(metric["edited_action_mass"][0] - metric["base_action_mass"][0])]
            if any(not math.isfinite(value) for value in scalar + position):
                raise ValueError("descriptor response contains non-finite values")
            total = totals[fid, task]
            for number, value in enumerate(scalar):
                total["scalar"][number] += value
            for number, value in enumerate(position):
                total["positions"][number][dim] += value
            total["active"] += 1
            executed += 1
        report(completed)
    if executed != planned:
        raise RuntimeError("descriptor executed pairs differ from preflight")
    scalar_keys = ("mean_head_kl", "mean_edit_norm", "full_vocab_argmax_flip_rate",
                   "mean_intended_edit_norm", "mean_readout_activation", "mean_cast_norm_abs_error")
    position_keys = ("signed_action_bin_shift", "absolute_action_bin_shift", "action_argmax_flip",
                     "action_conditional_kl", "action_mass_abs_change", "signed_action_mass_change")
    rows = []
    for fid in selected:
        for task in tasks:
            total, n = totals[fid, task], task_queries[task]
            row = {"feature_id": fid, "task_id": task, "num_queries": n, "num_readouts": n * 7,
                   "num_active_readouts": total["active"], "readout_activation_frequency": total["active"] / (n * 7),
                   **{key: value / (n * 7) for key, value in zip(scalar_keys, total["scalar"])},
                   **{key: [value / n for value in vector] for key, vector in zip(position_keys, total["positions"])}}
            if any(not math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                raise ValueError("non-finite descriptor aggregate")
            if any(not math.isfinite(value) for key in position_keys for value in row[key]):
                raise ValueError("non-finite descriptor position aggregate")
            rows.append(row)
    return {
        "schema_version": "head_response_descriptors_v1", "synthetic": synthetic,
        "rows": rows, "feature_ids": selected, "task_ids": tasks, "dictionary_size": width,
        "query_metadata": queries, "action_dimension_order": list(range(7)),
        "source_cache_signature": cache.get("manifest", {}).get("cache_signature"),
        "action_mapping": mapping, "action_bin_decode": decode,
        "aggregation": "per_task_equal_episode_query_means_including_all_inactive_zeros",
        "scalar_denominator": "all_task_readouts", "position_denominator": "all_task_queries_per_position",
        "signed_shift_definition": "E[edited_normalized_bin|action_tokens]-E[baseline_normalized_bin|action_tokens]",
        "absolute_shift_definition": "mean_of_absolute_per_readout_expected_bin_shift_not_absolute_of_mean",
        "edit_norm_definition": "realized_post_hidden_dtype_cast_L2",
        "metric_reduction_dtype": "float32",
        "response_scope": "fixed_prefix_token_bin_response_not_physical_harm_or_rollout_prediction",
        "diagnostics": {"planned_pairs": planned, "executed_pairs": executed,
                        "baseline_head_calls": baseline_calls, "num_readouts": len(records)},
        "new_rollouts": 0, "model_regeneration": False, "model_download": False, "sae_training": False,
        "outcome_labels_used": False,
    }
