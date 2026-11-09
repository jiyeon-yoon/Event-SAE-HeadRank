"""Frozen full-population Top-5 protocol, with no retrospective exclusions.

This is a protocol-alignment study on the inherited Spatial-500 collection.
Discovery and evaluation deliberately use the same 500 task/state identities.
It is not an independently held-out test, nor a calibrated failure predictor.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any

from event_sae.research.output_head.provenance import fingerprint

SCHEMA = "team500_plan_v1"
SCORE_SCHEMA = "team500_scores_v1"
MODEL_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
SAE_SHA = "18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6"
BASELINE_COMMIT = "56e9f012aa88532f9880259b371d9300a8013b9e"
METRICS = ("mean_head_kl", "tail_head_kl", "event_max")
UNIVERSE = "all_dictionary_features_no_known_feature_exclusion"
SEED_PROTOCOL = "baseline_global_seed0_condition_start_env_seed0_per_task_v1"
INTERPRETATION = "retrospective_full500_overlap_protocol_alignment_not_held_out"
DICT_SIZE = 32768


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _int(value: Any, field: str, low=0) -> int:
    if type(value) is not int or value < low:
        raise ValueError(f"{field} must be an integer >= {low}")
    return value


def ordered_cases(cases: list[dict]) -> list[dict]:
    """Retain the exact final global-RNG order; never silently sort a plan."""
    if not isinstance(cases, list) or len(cases) != 500:
        raise ValueError("Team500 requires exactly 500 evaluation cases")
    expected = [(task, trial) for task in range(10) for trial in range(50)]
    keys = []
    result = []
    for row in cases:
        if not isinstance(row, dict) or row.get("suite") != "libero_spatial":
            raise ValueError("Every case must be LIBERO-Spatial")
        task = _int(row.get("task_id"), "task_id")
        trial = _int(row.get("task_episode_idx"), "task_episode_idx")
        if row.get("seed", 0) != 0 or type(row.get("seed", 0)) is not int:
            raise ValueError("Case-specific hashed seeds are incompatible with Baseline global seed 0")
        state = _sha(row.get("initial_state_sha256"), "initial_state_sha256")
        keys.append((task, trial))
        result.append({"suite": "libero_spatial", "task_id": task,
                       "task_episode_idx": trial, "initial_state_sha256": state})
    if keys != expected:
        raise ValueError("Cases must be ordered task 0..9 then original trial 0..49, without skipping")
    for task in range(10):
        if len({row["initial_state_sha256"] for row in result[task * 50:(task + 1) * 50]}) != 50:
            raise ValueError("Duplicate initial states within a task")
    return result


def validate_runtime_config(config: dict) -> None:
    """Validate the policy contract; on-device identities are checked separately."""
    if not isinstance(config, dict) or config.get("schema_version") != "team500_runtime_v1":
        raise ValueError("Require a Team500 runtime configuration")
    base = config.get("baseline_config", {})
    model, env = base.get("model", {}), base.get("env", {})
    allowed = {
        "model": {"family", "checkpoint", "revision", "code_revision", "center_crop", "load_in_8bit", "load_in_4bit"},
        "env": {"task_suite_name", "seed", "num_steps_wait", "num_trials_per_task", "task_ids"},
        "logging": {"root_dir", "save_video", "save_actions", "save_prompt_records", "save_trajectory_records"},
        "sae_collect": {"enabled", "mode", "layer_idxs", "flush_every", "sae_checkpoint", "topk", "rows_per_shard"},
    }
    if set(base) != set(allowed) or any(not isinstance(base[key], dict)
                                       or not set(base[key]) <= keys for key, keys in allowed.items()):
        raise ValueError("Only fields supported by the pinned stock Baseline dataclasses are permitted")
    if (model.get("family") != "openvla" or model.get("revision") != MODEL_REVISION
            or model.get("code_revision") != CODE_REVISION or model.get("center_crop") is not True
            or model.get("load_in_8bit") is not False or model.get("load_in_4bit") is not False):
        raise ValueError("Baseline policy must be pinned unquantized OpenVLA with center crop")
    if not isinstance(model.get("checkpoint"), str) or not model["checkpoint"].startswith("/"):
        raise ValueError("Policy checkpoint must be an existing-local-snapshot path")
    if (env.get("task_suite_name") != "libero_spatial" or type(env.get("seed")) is not int
            or env["seed"] != 0 or type(env.get("num_steps_wait")) is not int or env["num_steps_wait"] != 10
            or type(env.get("num_trials_per_task")) is not int or env["num_trials_per_task"] != 50):
        raise ValueError("Runtime must use Spatial, global seed 0, wait 10 and 50 trials/task")
    if "per_episode_seed" in env or "trial_indices" in env:
        raise ValueError("No episode reseeding or trial selection overrides are permitted")
    if env.get("task_ids") not in (None, "", "all", list(range(10))):
        raise ValueError("All tasks must execute in natural 0..9 order")
    if base.get("sae_collect", {}).get("enabled") is not False:
        raise ValueError("No new activation collection is authorized")
    logging = base.get("logging", {})
    if logging.get("save_actions") is not True or logging.get("save_prompt_records") is not True:
        raise ValueError("Actions and per-step evidence must be recorded")
    if logging.get("save_video", False) is not False:
        raise ValueError("Team500 budget excludes video recording")
    if config.get("expected_attention_backend") not in ("sdpa", "flash_attention_2"):
        raise ValueError("Freeze an explicit expected actual attention backend")
    if config.get("expected_unnorm_key") != "libero_spatial":
        raise ValueError("Freeze the actual libero_spatial action normalization key")
    files = config.get("model_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Runtime must bind snapshot file content hashes")
    for name, value in files.items():
        if not isinstance(name, str) or name.startswith("/") or ".." in name.split("/"):
            raise ValueError("Snapshot manifest paths must be confined and relative")
        _sha(value, "model file SHA")


def validate_scores(scores: dict) -> None:
    if (not isinstance(scores, dict) or scores.get("schema_version") != SCORE_SCHEMA
            or scores.get("status") != "scored" or scores.get("sampling_mode") != "full500"
            or scores.get("development_only") is not False or scores.get("synthetic") is not False):
        raise ValueError("Full real Team500 scores are required; smoke/250 scores cannot authorize a plan")
    if scores.get("score_hash") != fingerprint({k: v for k, v in scores.items() if k != "score_hash"}):
        raise ValueError("Score artifact integrity hash mismatch")
    scope = scores.get("scope", {})
    expected_scope = {"task_ids": list(range(10)), "discovery_episodes": 500,
                      "episodes_per_task": 50, "candidate_universe": UNIVERSE,
                      "dictionary_size": DICT_SIZE, "sae_sha256": SAE_SHA,
                      "discovery_evaluation_state_overlap": True,
                      "sae_training_evaluation_overlap": True,
                      "historical_state_identity_verified": False,
                      "historical_runtime_flags_verified": False}
    if any(scope.get(k) != v for k, v in expected_scope.items()):
        raise ValueError("Scores do not describe the exact inherited full500 population/scope")
    for field in ("discovery_episodes", "episodes_per_task", "dictionary_size"):
        _int(scope[field], field, 1)
    _sha(scope.get("discovery_manifest_hash"), "discovery_manifest_hash")
    _sha(scope.get("sample_manifest_hash"), "sample_manifest_hash")
    _sha(scope.get("source_index_sha256"), "source_index_sha256")
    _sha(scope.get("discovery_cases_hash"), "discovery_cases_hash")
    rows = scores.get("feature_scores")
    if not isinstance(rows, list) or len(rows) != DICT_SIZE:
        raise ValueError("All 32,768 dictionary features must retain scores")
    if (any(not isinstance(row, dict) or type(row.get("feature_id")) is not int for row in rows)
            or [row.get("feature_id") for row in rows] != list(range(DICT_SIZE))):
        raise ValueError("Feature scores must contain each dictionary ID exactly once in ascending order")
    for row in rows:
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing/nonfinite score for {metric}")
            if metric != "event_max" and value < -1e-6:
                raise ValueError("KL cannot be negative")


def build_plan(scores: dict, cases: list[dict], runtime_config: dict, *,
               score_file_sha256: str | None = None, reference_bindings: dict | None = None) -> dict:
    validate_scores(scores)
    canonical = ordered_cases(cases)
    if scores["scope"]["discovery_cases_hash"] != fingerprint(canonical):
        raise ValueError("Evaluation cases differ from the full500 discovery current-state registry")
    validate_runtime_config(runtime_config)
    if score_file_sha256 is not None:
        _sha(score_file_sha256, "score_file_sha256")
    rows = scores["feature_scores"]
    methods = {metric: [row["feature_id"] for row in sorted(
        rows, key=lambda r: (-float(r[metric]), r["feature_id"]))[:5]] for metric in METRICS}
    feature_ids = sorted(set().union(*map(set, methods.values())))
    anchor = methods["mean_head_kl"][0]
    conditions = [{"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.0},
                  {"condition_id": f"identity-{anchor}", "mode": "identity",
                   "feature_id": anchor, "alpha": 1.0}]
    conditions += [{"condition_id": f"feature-{fid}", "mode": "intervention",
                    "feature_id": fid, "alpha": 0.0} for fid in feature_ids]
    result = {
        "schema_version": SCHEMA, "status": "predictions_frozen_not_runtime_approved",
        "top_k": 5, "methods": methods, "feature_ids": feature_ids,
        "conditions": conditions, "cases": canonical, "cases_hash": fingerprint(canonical),
        "runtime_config": copy.deepcopy(runtime_config), "runtime_config_hash": fingerprint(runtime_config),
        "scores_hash": scores["score_hash"], "score_file_sha256": score_file_sha256,
        "source_scope": copy.deepcopy(scores["scope"]), "reference_bindings": reference_bindings or {},
        "candidate_universe": UNIVERSE, "known_feature_exclusion": False,
        "seed_protocol": SEED_PROTOCOL, "baseline_commit": BASELINE_COMMIT,
        "interpretation": INTERPRETATION, "selection_evaluation_overlap": True,
        "evaluation_labels_previously_used": True, "total_rollouts": len(conditions) * 500,
        "num_evaluation_cases": 500, "new_rollouts": 0, "execute": False,
        "resume_policy": "restart_entire_incomplete_condition_from_seed0_no_case_skipping",
        "prediction_target": "signed_success_rate_loss_not_semantic_feature_meaning",
        "topk_effect": "mean_of_separate_single_feature_removals_not_simultaneous_removal",
        "budget_status": "requires_exact_execution_approval_after_conformance_gate",
    }
    result["plan_hash"] = fingerprint(result)
    verify_plan(result)
    return result


def verify_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported Team500 plan")
    if plan.get("plan_hash") != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("Frozen plan integrity hash mismatch")
    cases = ordered_cases(plan.get("cases"))
    if plan.get("cases_hash") != fingerprint(cases):
        raise ValueError("Plan case order/state binding differs")
    validate_runtime_config(plan.get("runtime_config"))
    if plan.get("runtime_config_hash") != fingerprint(plan["runtime_config"]):
        raise ValueError("Runtime configuration binding mismatch")
    if (plan.get("top_k") != 5 or plan.get("candidate_universe") != UNIVERSE
            or plan.get("known_feature_exclusion") is not False
            or plan.get("baseline_commit") != BASELINE_COMMIT
            or plan.get("seed_protocol") != SEED_PROTOCOL
            or plan.get("interpretation") != INTERPRETATION
            or plan.get("selection_evaluation_overlap") is not True):
        raise ValueError("Plan protocol settings differ from frozen full500 Top-5 design")
    methods = plan.get("methods", {})
    if set(methods) != set(METRICS):
        raise ValueError("Mean, Tail and Event-max must all be present")
    for ids in methods.values():
        if (not isinstance(ids, list) or len(ids) != 5 or len(set(ids)) != 5
                or any(type(fid) is not int or not 0 <= fid < DICT_SIZE for fid in ids)):
            raise ValueError("Every method must retain exactly its original five feature IDs")
    features = sorted(set().union(*map(set, methods.values())))
    anchor = methods["mean_head_kl"][0]
    expected = [{"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.0},
                {"condition_id": f"identity-{anchor}", "mode": "identity", "feature_id": anchor, "alpha": 1.0}]
    expected += [{"condition_id": f"feature-{fid}", "mode": "intervention", "feature_id": fid, "alpha": 0.0}
                 for fid in features]
    if plan.get("feature_ids") != features or plan.get("conditions") != expected:
        raise ValueError("Conditions differ from the exact Top-5 union plus raw/identity")
    if plan.get("num_evaluation_cases") != 500 or plan.get("total_rollouts") != len(expected) * 500:
        raise ValueError("Plan rollout count is inconsistent")
    if plan.get("resume_policy") != "restart_entire_incomplete_condition_from_seed0_no_case_skipping":
        raise ValueError("Case-skipping resume is forbidden under global RNG")
    _sha(plan.get("scores_hash"), "scores_hash")


def verify_plan_scores(plan: dict, scores: dict) -> None:
    """Recompute selection from score values, not only a self-consistent plan hash."""
    verify_plan(plan)
    rebuilt = build_plan(scores, plan["cases"], plan["runtime_config"],
                         score_file_sha256=plan.get("score_file_sha256"),
                         reference_bindings=plan.get("reference_bindings"))
    if rebuilt != plan:
        raise ValueError("Plan rankings differ from the source score artifact")
