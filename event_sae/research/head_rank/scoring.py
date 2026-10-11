"""Fixed-prefix output-head ranking hypotheses; never a rollout outcome model.

Seven autoregressive *positions already in the cache* form an action query;
this module does not regenerate tokens. All inactive query/feature pairs are
zero, including in tail denominators. Tail means the largest sampled 10% per
task, not the worst 10% of an entire trajectory or task failure distribution.

Direction controls borrow the anchor edit's magnitude and active support.
They are not natural ablations of the control features. Norm equality is
exact only before casting the hidden state. By default a realized norm must
be within 5% + 1e-5 of the intended norm, and each realized control norm must
also agree with the realized anchor norm within the same tolerance. Any
invalid active row makes the entire feature's direction-excess score null;
invalid controls are never silently discarded or replaced.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from statistics import median

import torch

from event_sae.research.head_followup.interventions import apply_edit, decoder_direction
from event_sae.research.output_head.head import (
    EDIT_PARITY_CHECKS, HEAD_PARITY_CHECKS, validate_parity_report,
)
from event_sae.research.output_head.sensitivity import _validate_sparse_cache, full_vocab_kl


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _feature_ids(values, width, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty feature-ID list")
    if any(isinstance(fid, bool) or not isinstance(fid, int) or not 0 <= fid < width for fid in values):
        raise ValueError(f"{name} contains invalid feature IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate feature IDs")
    return sorted(values)


def query_index(records):
    """Validate complete seven-position queries and equal episode sampling.

    Different tasks may have different numbers of episodes. Within each task
    all episodes must have the same number of sampled action queries, so a
    within-task query mean is also an equal-episode mean. Keys/order are
    canonical and therefore independent of cache row ordering.
    """
    groups = defaultdict(list)
    episodes = defaultdict(lambda: defaultdict(set))
    if not records:
        raise ValueError("at least one readout is required")
    for index, record in enumerate(records):
        task = _integer(record.get("task_id"), "task_id")
        step = _integer(record.get("step_in_episode"), "step_in_episode")
        dimension = _integer(record.get("action_dim_index"), "action_dim_index")
        episode = record.get("episode_uid")
        if not isinstance(episode, str) or not episode:
            raise ValueError("episode_uid must be a nonempty string")
        key = (task, episode, step)
        groups[key].append((dimension, index))
        episodes[task][episode].add(step)
    keys = sorted(groups)
    indices, metadata = [], []
    row_to_query = [None] * len(records)
    for number, key in enumerate(keys):
        rows = sorted(groups[key])
        if [dimension for dimension, _ in rows] != list(range(7)):
            raise ValueError(f"each action query requires exactly dimensions 0..6: {key}")
        indices.append([index for _, index in rows])
        metadata.append({"query_index": number, "task_id": key[0], "episode_uid": key[1],
                         "step_in_episode": key[2], "num_action_positions": 7})
        for _, index in rows:
            row_to_query[index] = number
    for task, task_episodes in episodes.items():
        if len({len(steps) for steps in task_episodes.values()}) != 1:
            raise ValueError(f"unequal query counts per episode within task {task}; use a balanced sample")
    return indices, metadata, row_to_query


def select_controls(active_ids, anchor, count, seed):
    """Stable per-anchor controls, independent of scoring-panel/row order."""
    pool = sorted(fid for fid in active_ids if fid != anchor)
    if len(pool) < count:
        raise ValueError("not enough distinct active decoder directions for control_count")
    digest = hashlib.sha256(f"head_rank_decoder_controls_v1:{seed}:{anchor}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big")).sample(pool, count)


def aggregate_queries(query_values, query_metadata, tail_fraction):
    """Equal-task mean and fixed within-task upper-tail mean, zeros included."""
    if len(query_values) != len(query_metadata) or not query_values:
        raise ValueError("query values and metadata must have equal nonzero length")
    if any(not math.isfinite(float(value)) for value in query_values):
        raise ValueError("query values must be finite")
    if (isinstance(tail_fraction, bool) or not isinstance(tail_fraction, (int, float))
            or not math.isfinite(tail_fraction) or not 0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must be in (0, 1]")
    by_task = defaultdict(list)
    for value, meta in zip(query_values, query_metadata):
        by_task[meta["task_id"]].append(float(value))
    tasks = []
    for task, values in sorted(by_task.items()):
        tail_n = max(1, math.ceil(tail_fraction * len(values)))
        tasks.append({"task_id": task, "num_queries": len(values), "num_tail_queries": tail_n,
                      "mean": math.fsum(values) / len(values),
                      "tail": math.fsum(sorted(values, reverse=True)[:tail_n]) / tail_n})
    return {"mean": math.fsum(row["mean"] for row in tasks) / len(tasks),
            "tail": math.fsum(row["tail"] for row in tasks) / len(tasks), "tasks": tasks}


def _parity(cache, head, current_identity, parity_reports):
    synthetic = cache.get("synthetic") is True or cache.get("manifest", {}).get("synthetic") is True
    if not synthetic:
        if not current_identity or not parity_reports:
            raise ValueError("current head/edit runtime parity is required")
        for key, expected in {
            "head_weights_hash": head.manifest.get("weights_sha256"),
            "model_revision": head.manifest.get("model_revision"),
            "code_revision": head.manifest.get("code_revision"),
            "layer_idx": head.manifest.get("target_layer"),
            "encoder_grouping": "original_forward", "pair_batch_size": 1,
            "reduction_dtype": "float32", "negative_tolerance": 1e-6, "alpha": 0.0,
        }.items():
            if expected is None or current_identity.get(key) != expected:
                raise ValueError("parity identity differs from actual head-rank scorer inputs")
        validate_parity_report(parity_reports.get("head", {}), current_identity, HEAD_PARITY_CHECKS)
        validate_parity_report(parity_reports.get("edit", {}), current_identity, EDIT_PARITY_CHECKS)
    return synthetic


@torch.inference_mode()
def score_cache(cache, head, sae, *, feature_ids=None, control_count=4, seed=1729,
                tail_fraction=.1, max_pairs=5000000, current_identity=None,
                parity_reports=None, progress=None, norm_rtol=.05, norm_atol=1e-5):
    """Score exact sparse cached latents without SAE re-encoding or rollouts.

    A pair is one active anchor/readout plus one real or control edited-head
    evaluation. Budget preflight includes all controls and precedes any head
    or decoder call. Baseline-head calls are recorded separately.

    ``cache['control_pool_feature_ids']`` may preserve the full-cache active
    pool when the caller takes a balanced smoke subset. The caller must bind
    that declared pool to the full source-cache provenance.
    """
    hidden, records, ids, values, width = _validate_sparse_cache(cache)
    if cache.get("encoder_grouping") != "original_forward":
        raise ValueError("head-rank requires original_forward sparse latents")
    synthetic = _parity(cache, head, current_identity, parity_reports)
    _integer(control_count, "control_count", 1)
    _integer(seed, "seed")
    _integer(max_pairs, "max_pairs", 1)
    for name, value in (("norm_rtol", norm_rtol), ("norm_atol", norm_atol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    groups, queries, row_to_query = query_index(records)
    # Validate tail configuration before the first model/decoder invocation.
    aggregate_queries([0.] * len(queries), queries, tail_fraction)
    observed_active = sorted({fid for row in ids for fid in row.tolist()})
    selected = _feature_ids(feature_ids if feature_ids is not None else observed_active, width, "feature_ids")
    if "control_pool_feature_ids" in cache:
        pool = _feature_ids(cache["control_pool_feature_ids"], width, "control_pool_feature_ids")
        if not set(observed_active) <= set(pool):
            raise ValueError("control pool must contain every observed active feature")
        pool_source = "declared_full_cache_active_pool"
    else:
        pool, pool_source = observed_active, "current_cache_active_pool"
    selected_set = set(selected)
    selected_per_row = [[fid for fid in row.tolist() if fid in selected_set] for row in ids]
    active_pairs = sum(len(row) for row in selected_per_row)
    pair_budget = active_pairs * (control_count + 1)
    if pair_budget > max_pairs:
        raise ValueError(f"head-rank pair budget exceeded: {pair_budget} > {max_pairs}")
    active_selected = set(observed_active) & selected_set
    controls = {fid: select_controls(pool, fid, control_count, seed) if fid in active_selected else []
                for fid in selected}
    device = head.head_weight.device
    directions = {}
    accumulated = {fid: {} for fid in selected}
    counts = {fid: 0 for fid in selected}
    invalid_counts = {fid: 0 for fid in selected}
    invalid_control_counts = {fid: 0 for fid in selected}
    max_cast_error, max_pair_norm_error, executed, baseline_calls = 0., 0., 0, 0

    def report_progress(completed):
        if progress and (completed % 100 == 0 or completed == len(records)):
            progress({"readouts": completed, "total_readouts": len(records),
                      "executed_pairs": executed, "total_pairs": pair_budget})

    for completed, index in enumerate((row for group in groups for row in group), 1):
        anchors = selected_per_row[index]
        if not anchors:
            report_progress(completed)
            continue
        h = hidden[index:index + 1].to(device=device, dtype=head.hidden_dtype)
        z = torch.zeros((1, width), dtype=torch.float32, device=device)
        z[0, ids[index].to(device)] = values[index].to(device)
        base = head(h)
        baseline_calls += 1
        query = row_to_query[index]
        for fid in anchors:
            edited, actual = apply_edit(h, sae, fid, 0., encoded=z)
            real_kl = float(full_vocab_kl(base, head(edited))[0])
            intended = float(actual["intended_norm"][0])
            realized = float(actual["realized_norm"][0])
            tolerance = norm_atol + norm_rtol * intended
            real_error = abs(realized - intended)
            valid = real_error <= tolerance
            max_cast_error = max(max_cast_error, real_error)
            control_kls, control_norms, bad_controls = [], [], 0
            executed += 1
            for control in controls[fid]:
                if control not in directions:
                    directions[control] = -decoder_direction(sae, control, width, device=device)
                changed, diagnostic = apply_edit(h, sae, fid, 0., direction=directions[control], encoded=z)
                control_kls.append(float(full_vocab_kl(base, head(changed))[0]))
                control_intended = float(diagnostic["intended_norm"][0])
                control_realized = float(diagnostic["realized_norm"][0])
                control_norms.append(control_realized)
                cast_error = abs(control_realized - control_intended)
                pair_error = abs(control_realized - realized)
                # Both controls borrow the same intended anchor norm, but
                # realized casting error can differ with decoder direction.
                control_valid = (abs(control_intended - intended) <= tolerance and cast_error <= tolerance
                                 and pair_error <= norm_atol + norm_rtol * realized)
                bad_controls += int(not control_valid)
                valid = valid and control_valid
                max_cast_error = max(max_cast_error, cast_error)
                max_pair_norm_error = max(max_pair_norm_error, pair_error)
                executed += 1
            counts[fid] += 1
            invalid_counts[fid] += int(not valid)
            invalid_control_counts[fid] += bad_controls
            record = accumulated[fid].setdefault(query, {
                "real": [], "controls": [[] for _ in controls[fid]], "invalid": 0,
                "intended_norms": [], "realized_norms": [],
                "control_realized_norms": [[] for _ in controls[fid]],
            })
            record["real"].append(real_kl)
            record["invalid"] += int(not valid)
            record["intended_norms"].append(intended)
            record["realized_norms"].append(realized)
            for number, value in enumerate(control_kls):
                record["controls"][number].append(value)
                record["control_realized_norms"][number].append(control_norms[number])
        report_progress(completed)
    if executed != pair_budget:
        raise RuntimeError("head-rank executed-pair count differs from preflight")
    feature_scores, sparse_queries = [], {}
    for fid in selected:
        real_values, excess_values = [0.] * len(queries), [0.] * len(queries)
        intended_values, realized_values = [0.] * len(queries), [0.] * len(queries)
        sparse = []
        for number, row in sorted(accumulated[fid].items()):
            # Seven positions are the denominator even when the feature is
            # absent in some positions. Never divide by active positions.
            real_values[number] = math.fsum(row["real"]) / 7
            ctrl_values = [math.fsum(values) / 7 for values in row["controls"]]
            excess_values[number] = real_values[number] - median(ctrl_values)
            intended_values[number] = math.fsum(row["intended_norms"]) / 7
            realized_values[number] = math.fsum(row["realized_norms"]) / 7
            sparse.append({"query_index": number, "mean_head_kl": real_values[number],
                           "control_query_kls": ctrl_values,
                           "direction_excess_kl": excess_values[number] if not row["invalid"] else None,
                           "num_active_readouts": len(row["real"]), "norm_invalid_readouts": row["invalid"],
                           "mean_intended_edit_norm": intended_values[number],
                           "mean_realized_edit_norm": realized_values[number],
                           "control_mean_realized_norms": [math.fsum(values) / 7 for values in row["control_realized_norms"]]})
        real = aggregate_queries(real_values, queries, tail_fraction)
        excess = aggregate_queries(excess_values, queries, tail_fraction)["mean"]
        feature_scores.append({
            "feature_id": fid, "mean_head_kl": real["mean"], "tail_head_kl": real["tail"],
            "direction_excess_kl": excess if not invalid_counts[fid] else None,
            "direction_excess_status": "invalid_norm_match" if invalid_counts[fid] else "ok",
            "num_active_readouts": counts[fid], "num_readouts": len(records),
            "num_active_queries": len(sparse), "num_queries": len(queries),
            "norm_invalid_readouts": invalid_counts[fid], "norm_invalid_controls": invalid_control_counts[fid],
            "mean_intended_edit_norm": aggregate_queries(intended_values, queries, tail_fraction)["mean"],
            "mean_realized_edit_norm": aggregate_queries(realized_values, queries, tail_fraction)["mean"],
            "control_feature_ids": controls[fid], "task_scores": real["tasks"],
        })
        sparse_queries[str(fid)] = sparse
    return {
        "schema_version": "head_rank_scores_v1", "synthetic": synthetic,
        "feature_scores": feature_scores, "feature_ids": selected, "dictionary_size": width,
        "control_feature_ids": {str(fid): controls[fid] for fid in selected},
        "control_pool_feature_ids": pool, "control_pool_source": pool_source,
        "query_metadata": queries, "feature_query_scores": sparse_queries,
        "sparse_query_semantics": "unlisted_feature_query_pairs_are_exactly_inactive_zero",
        "aggregation": "seven_positions_mean_then_equal_task_query_mean_balanced_episodes",
        "tail_aggregation": "per_task_top_ceil_fraction_query_mean_then_equal_task_mean_including_inactive_zeros",
        "excess_aggregation": "per_query_real_mean_minus_median_of_control_query_means_then_equal_task_query_mean",
        "control_count": control_count, "control_seed": seed, "tail_fraction": float(tail_fraction),
        "norm_match": {"scope": "per_row_same_hidden_pre_cast_only", "rtol": norm_rtol, "atol": norm_atol,
                       "invalid_policy": "null_entire_feature_excess_if_any_active_row_invalid"},
        "diagnostics": {"active_pairs": active_pairs, "planned_pairs": pair_budget, "executed_pairs": executed,
                        "baseline_head_calls": baseline_calls, "max_cast_norm_abs_error": max_cast_error,
                        "max_realized_control_anchor_norm_abs_error": max_pair_norm_error,
                        "num_excess_eligible_features": sum(not invalid_counts[fid] for fid in selected),
                        "num_norm_invalid_readouts": sum(invalid_counts.values()),
                        "num_norm_invalid_controls": sum(invalid_control_counts.values()),
                        "device": str(device), "hidden_dtype": str(head.hidden_dtype),
                        "encoder_grouping": "original_forward", "pair_batch_size": 1},
        "source_cache_signature": cache.get("manifest", {}).get("cache_signature"),
        "new_rollouts": 0, "model_regeneration": False,
        "scientific_scope": "fixed_prefix_output_sensitivity_hypotheses_not_calibrated_success_drop_predictions",
    }
