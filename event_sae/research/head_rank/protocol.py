"""Outcome-blind feature-panel planning; this module never executes a rollout.

Hash verification detects accidental modification, not whether a human has seen
an outcome.  The known-evaluated registry must therefore also include results
disclosed by collaborators before a panel is frozen.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import math
import random
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "head_rank_blind_panel_v1"
DICTIONARY_SIZE = 32768
METRICS = ("mean_head_kl", "tail_head_kl", "direction_excess_kl", "event_max")
HISTORICAL_CONDITION_HOURS = 40.87710964816312
HISTORICAL_ROLLOUTS = 3600

# Union of the old two-task pilot and full-ten-task feature interventions.
# Keeping only the full10 panel would incorrectly call six pilot features new.
KNOWN_EVALUATED_IDS = frozenset({
    2723, 2875, 3165, 4449, 5169, 5632, 6553, 7024, 12409, 13724,
    14146, 17857, 18471, 22257, 22593, 24360, 24830, 25653, 27955,
    28491, 28858, 30729,
})


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _feature_id(value: Any) -> int:
    value = _integer(value, "feature_id")
    if value >= DICTIONARY_SIZE:
        raise ValueError(f"feature_id must be < {DICTIONARY_SIZE}")
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Plan fields must contain finite, JSON-serializable values") from exc


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def estimate_budget(feature_count: int, task_count: int = 10,
                    cases_per_task: int = 5) -> dict[str, Any]:
    """Historical sequential RTX4090 extrapolation, not a wall-time promise."""
    feature_count = _integer(feature_count, "feature_count", minimum=1)
    task_count = _integer(task_count, "task_count", minimum=1)
    cases_per_task = _integer(cases_per_task, "cases_per_task", minimum=1)
    conditions = feature_count + 2
    cases = task_count * cases_per_task
    total = conditions * cases
    return {
        "feature_conditions": feature_count,
        "control_conditions": ["raw", "identity"],
        "total_conditions": conditions,
        "task_count": task_count,
        "cases_per_task": cases_per_task,
        "cases_per_condition": cases,
        "feature_rollouts": feature_count * cases,
        "control_rollouts": 2 * cases,
        "total_rollouts": total,
        "historical_rollouts": HISTORICAL_ROLLOUTS,
        "historical_condition_hours": HISTORICAL_CONDITION_HOURS,
        "historical_seconds_per_rollout": HISTORICAL_CONDITION_HOURS * 3600 / HISTORICAL_ROLLOUTS,
        "estimated_condition_hours": total * HISTORICAL_CONDITION_HOURS / HISTORICAL_ROLLOUTS,
        "basis": "linear_extrapolation_of_recorded_sequential_rtx4090_condition_time",
        "excludes": ["offline_scoring", "downloads", "setup", "runtime_validation", "retries"],
        "wall_time_guaranteed": False,
        "statistical_power_established": False,
        "execute": False,
    }


def _score_rows(scores: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(scores, Mapping):
        raise ValueError("scores must be a mapping")
    if scores.get("synthetic") is not False:
        raise ValueError("Blind planning requires explicitly non-synthetic scores")
    for key in ("status", "sampling_mode", "mode", "phase", "score_phase", "purpose", "feature_set"):
        value = str(scores.get(key, "")).lower()
        if any(word in value for word in ("smoke", "development", "partial", "pilot")):
            raise ValueError("Development/smoke/partial scores cannot freeze a blind panel")
    if any(scores.get(key) is True for key in ("smoke", "development", "development_only", "partial")):
        raise ValueError("Development/smoke/partial scores cannot freeze a blind panel")
    if scores.get("complete") is False:
        raise ValueError("Incomplete scoring cannot freeze a blind panel")
    if scores.get("dictionary_size", DICTIONARY_SIZE) != DICTIONARY_SIZE:
        raise ValueError("Unexpected dictionary_size")
    rows = scores.get("feature_scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError("feature_scores must be a nonempty list")
    normalized, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every feature_scores row must be a mapping")
        fid = _feature_id(row.get("feature_id"))
        if fid in seen:
            raise ValueError(f"Duplicate feature_id: {fid}")
        seen.add(fid)
        item: dict[str, Any] = {"feature_id": fid}
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Feature {fid}: {metric} must be finite; no silent eligibility filtering")
            item[metric] = float(value)
        normalized.append(item)
    return sorted(normalized, key=lambda item: item["feature_id"])


def _cases(evaluation_cases: Sequence[Mapping[str, Any]] | None,
           task_ids: list[int], cases_per_task: int) -> list[dict[str, Any]] | None:
    if evaluation_cases is None:
        return None
    if not isinstance(evaluation_cases, (list, tuple)) or not evaluation_cases:
        raise ValueError("evaluation_cases must be a nonempty list")
    normalized, seen_indices, seen_states = [], set(), set()
    counts: Counter[int] = Counter()
    suite_name = None
    for case in evaluation_cases:
        if not isinstance(case, Mapping):
            raise ValueError("Each evaluation case must be a mapping")
        suite = case.get("suite")
        if not isinstance(suite, str) or not suite.strip():
            raise ValueError("Evaluation case suite is required")
        if suite_name is not None and suite != suite_name:
            raise ValueError("All evaluation cases must use one fixed suite")
        suite_name = suite
        task = _integer(case.get("task_id"), "task_id")
        if task not in task_ids:
            raise ValueError("Evaluation case task_id is outside planned task_ids")
        index = _integer(case.get("task_episode_idx"), "task_episode_idx")
        sha = case.get("initial_state_sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Evaluation initial_state_sha256 must be a lowercase SHA256 digest")
        identity, state_identity = (suite, task, index), (suite, task, sha)
        if identity in seen_indices or state_identity in seen_states:
            raise ValueError("Duplicate/overlapping evaluation case or initial state")
        seen_indices.add(identity)
        seen_states.add(state_identity)
        item = {"suite": suite, "task_id": task, "task_episode_idx": index,
                "initial_state_sha256": sha}
        if "seed" in case:
            item["seed"] = _integer(case["seed"], "seed")
        normalized.append(item)
        counts[task] += 1
    if counts != Counter({task: cases_per_task for task in task_ids}):
        raise ValueError("evaluation_cases must have exactly cases_per_task cases for every planned task")
    return sorted(normalized, key=lambda case: (case["task_id"], case["task_episode_idx"]))


def plan_panel(scores: Mapping[str, Any], *, additional_evaluated_ids=(),
               feature_count: int = 8, seed: int = 1729, top_k: int = 3,
               cases_per_task: int = 5, task_ids=range(10),
               primary_metric: str = "tail_head_kl", reference_bindings=None,
               evaluation_cases=None) -> dict[str, Any]:
    """Freeze outcome-blind rankings and a budgeted, multi-method feature panel.

    No behavior labels are accepted or used.  This predicts importance ordering,
    not calibrated percentage-point success drops.  Supplying evaluation cases
    freezes their identities, but never authorizes a paid runtime execution.
    """
    feature_count = _integer(feature_count, "feature_count", minimum=1)
    seed = _integer(seed, "seed")
    top_k = _integer(top_k, "top_k", minimum=1)
    cases_per_task = _integer(cases_per_task, "cases_per_task", minimum=1)
    tasks = [_integer(task, "task_id") for task in task_ids]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("task_ids must be nonempty and unique")
    tasks = sorted(tasks)
    if primary_metric not in ("tail_head_kl", "direction_excess_kl"):
        raise ValueError("primary_metric must be tail_head_kl or direction_excess_kl")
    if reference_bindings is not None and not isinstance(reference_bindings, Mapping):
        raise ValueError("reference_bindings must be a mapping")
    bindings = deepcopy(dict(reference_bindings or {}))
    _canonical(bindings)
    rows = _score_rows(scores)
    additional = {_feature_id(fid) for fid in additional_evaluated_ids}
    excluded = KNOWN_EVALUATED_IDS | additional
    eligible = [row for row in rows if row["feature_id"] not in excluded]
    if len(eligible) < feature_count or len(eligible) < top_k:
        raise ValueError("Insufficient eligible scored features for requested panel/top_k")
    ranks = {metric: [row["feature_id"] for row in sorted(
        eligible, key=lambda item: (-item[metric], item["feature_id"]))] for metric in METRICS}
    positions = {metric: {fid: rank for rank, fid in enumerate(order, 1)}
                 for metric, order in ranks.items()}
    priority = list(dict.fromkeys((primary_metric, "mean_head_kl", "event_max",
                                  "direction_excess_kl", "tail_head_kl")))
    reserved = min(2, feature_count // 4)
    round_robin_count = feature_count - 2 * reserved
    panel, trace = [], []
    selected = set()
    # Every scorer gets its turn; duplicates consume no slot and no method is
    # backfilled using observed successes or another downstream outcome.
    for metric in itertools.cycle(priority):
        fid = next(fid for fid in ranks[metric] if fid not in selected)
        selected.add(fid)
        panel.append(fid)
        trace.append({"feature_id": fid, "source": "rank_round_robin", "metric": metric})
        if len(panel) == round_robin_count:
            break
    rng = random.Random(seed)
    tie_order = sorted(row["feature_id"] for row in eligible if row["feature_id"] not in selected)
    rng.shuffle(tie_order)
    tiebreak = {fid: index for index, fid in enumerate(tie_order)}
    spread = {fid: max(positions[metric][fid] for metric in METRICS)
                   - min(positions[metric][fid] for metric in METRICS) for fid in tie_order}
    disagreements = sorted(tie_order, key=lambda fid: (-spread[fid], tiebreak[fid]))[:reserved]
    for fid in disagreements:
        selected.add(fid)
        panel.append(fid)
        trace.append({"feature_id": fid, "source": "symmetric_rank_disagreement",
                      "max_minus_min_rank": spread[fid]})
    random_pool = sorted(row["feature_id"] for row in eligible if row["feature_id"] not in selected)
    rng.shuffle(random_pool)
    for fid in random_pool[:reserved]:
        selected.add(fid)
        panel.append(fid)
        trace.append({"feature_id": fid, "source": "seeded_random_remainder"})
    cases = _cases(evaluation_cases, tasks, cases_per_task)
    topks = {}
    for metric, order in ranks.items():
        wanted = order[:top_k]
        present = [fid for fid in wanted if fid in selected]
        topks[metric] = {
            "feature_ids": wanted, "included_feature_ids": present,
            "missing_feature_ids": [fid for fid in wanted if fid not in selected],
            "complete_top_k_in_panel": len(present) == top_k,
            "coverage": len(present) / top_k,
            "substitution_permitted": False,
        }
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "needs_eval_manifest" if cases is None else "predictions_frozen_not_runtime_approved",
        "synthetic": False, "execute": False, "runtime_approved": False,
        "prediction_target": "rank_of_signed_raw_minus_ablation_success_rate_drop",
        "calibrated_success_drop_prediction": False,
        "primary_metric": primary_metric,
        "secondary_metrics": [metric for metric in METRICS if metric != primary_metric],
        "source_bindings": bindings,
        "normalized_score_rows_sha256": fingerprint(rows),
        "dictionary_size": DICTIONARY_SIZE,
        "scored_feature_count": len(rows),
        "scored_feature_ids": [row["feature_id"] for row in rows],
        "known_evaluated_ids": sorted(KNOWN_EVALUATED_IDS),
        "additional_evaluated_ids": sorted(additional),
        "excluded_scored_ids": sorted(row["feature_id"] for row in rows if row["feature_id"] in excluded),
        "eligible_feature_count": len(eligible), "eligible_feature_scores": eligible,
        "selection": {
            "version": "balanced_rank_disagreement_random_v1", "seed": seed,
            "metric_priority": priority, "round_robin_slots": round_robin_count,
            "disagreement_slots": reserved, "random_slots": reserved,
            "rank_ties": "ascending_feature_id",
            "disagreement": "largest_max_minus_min_rank_across_all_metrics; seeded_ties",
            "random": "seeded_uniform_shuffle_of_remaining_eligible_features",
            "outcome_labels_used": False, "trace": trace,
        },
        "panel_feature_ids": panel,
        "panel_rankings": {metric: [fid for fid in order if fid in selected] for metric, order in ranks.items()},
        "top_k": top_k, "top_k_availability": topks,
        "task_ids": tasks, "cases_per_task": cases_per_task,
        "evaluation_cases": cases,
        "evaluation_cases_sha256": None if cases is None else fingerprint(cases),
        "budget": estimate_budget(feature_count, len(tasks), cases_per_task),
        "analysis_rules": {
            "outcome": "mean_paired_raw_success_minus_feature_success; retain_negative_drops",
            "primary_statistic": "spearman_on_this_fixed_feature_panel",
            "top_k": "mean_signed_drop_only_when_all_original_top_k_features_are_measured",
            "uncertainty": "shared_case_resampling_within_fixed_tasks_across_all_conditions",
            "comparators": ["mean_head_kl", "event_max"],
            "secondary_method": "exploratory_unless_multiplicity_is_predeclared",
            "no_global_dictionary_recall_claim": True,
            "no_held_out_task_or_model_generalization_claim": True,
            "pilot_inspection_rule": "If used to change scores/protocol, these features become development; use fresh features for confirmation",
        },
        "limitations": [
            "Known-evaluated exclusion requires honest collaborator disclosure; hashes do not prove blindness",
            "Selected-panel ranking is not dictionary-wide behavioral prediction",
            "No statistical power or method improvement is guaranteed",
            "Raw and identity controls must be paired on the exact same evaluation cases",
            "Runtime execution requires separate implementation, safety validation and approval",
        ],
    }
    plan["plan_hash"] = fingerprint(plan)
    return plan


def verify_plan(plan: Mapping[str, Any]) -> None:
    """Validate both the content hash and deterministic planning semantics."""
    if not isinstance(plan, Mapping) or plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported blind panel schema")
    payload = deepcopy(dict(plan))
    expected = payload.pop("plan_hash", None)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Missing or malformed plan_hash")
    if fingerprint(payload) != expected:
        raise ValueError("Plan hash mismatch; predictions have been modified")
    if plan.get("execute") is not False or plan.get("runtime_approved") is not False:
        raise ValueError("A blind prediction plan is not runtime approval")
    if plan.get("synthetic") is not False:
        raise ValueError("Synthetic plans are not blind predictions")
    if plan.get("known_evaluated_ids") != sorted(KNOWN_EVALUATED_IDS):
        raise ValueError("Known-evaluated registry does not match this protocol")
    if plan.get("dictionary_size") != DICTIONARY_SIZE:
        raise ValueError("Unexpected dictionary_size")
    panel = plan.get("panel_feature_ids", [])
    if not panel or len(panel) != len(set(panel)):
        raise ValueError("Panel must contain distinct features")
    blocked = KNOWN_EVALUATED_IDS | set(plan.get("additional_evaluated_ids", []))
    if blocked.intersection(panel):
        raise ValueError("Previously evaluated feature included in blind panel")
    # Rebuild using only frozen eligible scores, never behavior outcomes. The
    # original excluded-score values are intentionally not needed to replay
    # selection; their original complete document is bound by its score hash.
    replay = plan_panel(
        {"feature_scores": plan["eligible_feature_scores"], "synthetic": False},
        additional_evaluated_ids=plan["additional_evaluated_ids"], feature_count=len(panel),
        seed=plan["selection"]["seed"], top_k=plan["top_k"],
        cases_per_task=plan["cases_per_task"], task_ids=plan["task_ids"],
        primary_metric=plan["primary_metric"], reference_bindings=plan["source_bindings"],
        evaluation_cases=plan["evaluation_cases"],
    )
    for key in ("status", "panel_feature_ids", "panel_rankings", "top_k_availability",
                "selection", "budget", "evaluation_cases_sha256", "analysis_rules",
                "eligible_feature_count", "eligible_feature_scores"):
        if replay[key] != plan.get(key):
            raise ValueError(f"Plan semantic verification failed: {key}")
