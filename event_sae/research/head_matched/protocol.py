"""Frozen, retrospective matched-direction diagnostic; never runs a model.

Decoder controls borrow the anchor's support and magnitude. They are NOT
ablations of the control feature. This protocol does not certify equal edit
budgets on diverging trajectories, nor prospective new-feature prediction.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import random

from event_sae.research.output_head.provenance import fingerprint, read_json, sha256_file, atomic_write_json
from event_sae.research.head_rank.workflow import safe_output, validate_full_score_population

NORM_POLICY = {"atol": 1e-5, "rtol": .05}
ANALYSIS = {"bootstrap_replicates": 2000, "bootstrap_seed": 1729, "confidence_level": .95}
SCOPE = "retrospective_single_anchor_direction_diagnostic_not_unseen_feature_validation"


def signed(document, field):
    if not isinstance(document, dict) or document.get(field) != fingerprint({k: v for k, v in document.items() if k != field}):
        raise ValueError(f"Missing/changed {field}")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def conditions(anchor, controls):
    integer(anchor, "anchor")
    if anchor >= 32768 or len(controls) != 4 or len(set(controls)) != 4 or anchor in controls:
        raise ValueError("One anchor and all four distinct preselected decoder controls are required")
    for control in controls:
        if integer(control, "control") >= 32768:
            raise ValueError("Invalid control feature")
    rows = [("raw", "raw", None, 1., None),
            (f"identity-{anchor}", "identity", anchor, 1., None),
            (f"anchor-{anchor}", "anchor", anchor, 0., None)]
    rows.extend((f"decoder-{anchor}-control-{c}", "matched_decoder", anchor, 0., c) for c in controls)
    return [{"condition_id": name, "mode": mode, "feature_id": fid, "alpha": alpha,
             "control_feature_id": control, "norm_policy": dict(NORM_POLICY)}
            for name, mode, fid, alpha, control in rows]


def direction_scores(scores, anchor):
    """Replay all five equal-task query means; inactive queries remain zero."""
    signed(scores, "score_hash")
    if scores.get("synthetic") is not False:
        raise ValueError("Real scores required")
    rows = {r["feature_id"]: r for r in scores["feature_scores"]}
    row = rows.get(anchor)
    if row is None or row.get("direction_excess_status") != "ok" or row.get("norm_invalid_readouts") != 0 or row.get("norm_invalid_controls") != 0:
        raise ValueError("Anchor needs complete, norm-eligible stored control scores")
    controls = row["control_feature_ids"]
    labels = [f"anchor-{anchor}", *[f"decoder-{anchor}-control-{c}" for c in controls]]
    meta = scores["query_metadata"]
    if [q["query_index"] for q in meta] != list(range(len(meta))):
        raise ValueError("Noncanonical query metadata")
    groups = defaultdict(list)
    for q in meta:
        groups[q["task_id"]].append(q["query_index"])
    values = [[0.] * len(meta) for _ in labels]
    seen = set()
    for q in scores["feature_query_scores"][str(anchor)]:
        index = integer(q["query_index"], "query_index")
        if index in seen or index >= len(meta) or len(q["control_query_kls"]) != 4:
            raise ValueError("Malformed/duplicate control query")
        seen.add(index)
        for column, value in zip(values, [q["mean_head_kl"], *q["control_query_kls"]]):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Control KL must be finite and nonnegative")
            column[index] = value
    output = []
    for name, column in zip(labels, values):
        tasks = {str(t): math.fsum(column[q] for q in queries) / len(queries) for t, queries in sorted(groups.items())}
        output.append({"condition_id": name, "mean_head_kl": math.fsum(tasks.values()) / len(tasks), "task_mean_head_kl": tasks})
    if not math.isclose(output[0]["mean_head_kl"], row["mean_head_kl"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Stored anchor score does not replay from complete queries")
    return output


def build_plan(scores, parent, *, anchor=30452, cases_per_task=2, seed=1729, source_bindings=None):
    from event_sae.research.head_rank.runtime import verify_experiment
    verify_experiment(parent)
    signed(scores, "score_hash")
    validate_full_score_population(scores)
    if (scores.get("development_only") is not False or scores.get("feature_set") != "all-active"
            or scores.get("norm_match", {}).get("atol") != NORM_POLICY["atol"]
            or scores.get("norm_match", {}).get("rtol") != NORM_POLICY["rtol"]):
        raise ValueError("Full, norm-matched discovery scores are required")
    if scores["score_hash"] != parent["prediction_plan"]["source_bindings"]["score_hash"]:
        raise ValueError("Parent evaluation and frozen scores differ")
    integer(cases_per_task, "cases_per_task", 1)
    integer(seed, "seed")
    if anchor not in parent["prediction_plan"]["panel_feature_ids"]:
        raise ValueError("Diagnostic anchor must belong to the already evaluated panel")
    source = next(row for row in scores["feature_scores"] if row["feature_id"] == anchor)
    chosen = []
    for task in range(10):
        pool = sorted((r for r in parent["cases"] if r["task_id"] == task), key=lambda r: r["task_episode_idx"])
        if not pool or len(pool) < cases_per_task:
            raise ValueError("Every one of the ten fixed tasks needs enough parent cases")
        chosen.extend(random.Random(f"head-matched-cases-v1:{seed}:{task}").sample(pool, cases_per_task))
    chosen.sort(key=lambda r: (r["task_id"], r["task_episode_idx"]))
    conds = conditions(anchor, source["control_feature_ids"])
    doc = {"schema_version": "head_matched_plan_v1", "status": "design_frozen_not_runtime_approved",
           "anchor_feature_id": anchor, "control_feature_ids": source["control_feature_ids"],
           "conditions": conds, "cases": chosen, "task_ids": list(range(10)),
           "cases_per_task": cases_per_task, "case_selection_seed": seed,
           "total_rollouts": len(conds) * len(chosen),
           "purpose": "smoke_diagnostic_no_power_claim" if cases_per_task < 20 else "fixed_task_retrospective_diagnostic_no_power_guarantee",
           "scientific_scope": SCOPE, "norm_scope": "per_row_same_current_hidden_not_equal_trajectory_budget",
           "prediction_scope": "fixed_prefix_full_vocabulary_equal_task_mean_not_physical_action_error",
           "primary_endpoint": "spearman_of_five_frozen_direction_KLs_vs_signed_success_rate_loss",
           "secondary_endpoint": "anchor_signed_loss_minus_four_controls_mean_signed_loss",
           "archived_predictions": direction_scores(scores, anchor),
           "final_predictions": "require_current_device_scoring_before_any_behavior_outcomes",
           "source_score_hash": scores["score_hash"], "source_scope": scores["scope"],
           "source_query_metadata_hash": fingerprint(scores["query_metadata"]),
           "source_readouts": scores["scoring_sample"]["readouts_used"],
           "source_identity": scores["identity"], "source_implementation": scores["implementation"],
           "parent_content_hashes": {k: v["sha256"] for k, v in parent["sources"].items()
                                     if k.startswith("snapshot/") or k in
                                     ("base_eval_config", "sae", "sae_config", "head_weights", "head_manifest", "generation_manifest")},
           "source_reference": {k: scores[k] for k in ("source_scores_sha256", "source_sample_sha256", "source_cache_manifest_sha256")},
           "parent_execution_hash": parent["execution_hash"], "base_seed": parent["base_seed"],
           "parent_split_evidence": parent["split_evidence"], "analysis": dict(ANALYSIS),
           "source_bindings": deepcopy(source_bindings or {}), "synthetic": False,
           "new_rollouts": 0, "model_execution": False, "automatic_reuse_of_old_outcomes": False}
    doc["plan_hash"] = fingerprint(doc)
    verify_plan(doc)
    return doc


def verify_plan(plan):
    from event_sae.research.head_rank.native import episode_seed
    signed(plan, "plan_hash")
    if (plan.get("schema_version") != "head_matched_plan_v1" or plan.get("scientific_scope") != SCOPE
            or plan.get("synthetic") is not False or plan.get("automatic_reuse_of_old_outcomes") is not False
            or plan.get("analysis") != ANALYSIS or plan.get("task_ids") != list(range(10))
            or plan.get("norm_scope") != "per_row_same_current_hidden_not_equal_trajectory_budget"
            or plan.get("prediction_scope") != "fixed_prefix_full_vocabulary_equal_task_mean_not_physical_action_error"
            or plan.get("final_predictions") != "require_current_device_scoring_before_any_behavior_outcomes"):
        raise ValueError("Unsupported matched diagnostic plan")
    if plan["conditions"] != conditions(plan["anchor_feature_id"], plan["control_feature_ids"]):
        raise ValueError("Conditions differ from fixed anchor/four controls")
    n = integer(plan["cases_per_task"], "cases_per_task", 1)
    if integer(plan["source_readouts"], "source_readouts", 1) % 7:
        raise ValueError("Source must contain complete seven-position readouts")
    counts, seen, states = defaultdict(int), set(), set()
    for row in plan["cases"]:
        task, trial = integer(row["task_id"], "task"), integer(row["task_episode_idx"], "trial")
        if task not in range(10) or row["suite"] != "libero_spatial" or row["seed"] != episode_seed(plan["base_seed"], task, trial):
            raise ValueError("Case task/suite/seed differs from native fixed protocol")
        digest = row["initial_state_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid state hash")
        if (task, trial) in seen or (task, digest) in states:
            raise ValueError("Duplicate case/state")
        seen.add((task, trial)); states.add((task, digest)); counts[task] += 1
    if dict(counts) != {t: n for t in range(10)} or plan["total_rollouts"] != 7 * 10 * n:
        raise ValueError("Case counts/budget differ")
    if [r["condition_id"] for r in plan["archived_predictions"]] != [r["condition_id"] for r in plan["conditions"][2:]]:
        raise ValueError("Frozen prediction conditions differ")
    hashes = plan.get("parent_content_hashes", {})
    if len([k for k in hashes if k.startswith("snapshot/") and k.endswith(".safetensors")]) != 4:
        raise ValueError("Parent must bind all four complete model shards by content")
    for key in ("base_eval_config", "sae", "sae_config", "head_weights", "head_manifest", "generation_manifest"):
        if key not in hashes:
            raise ValueError(f"Parent content binding missing: {key}")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in hashes.values()):
        raise ValueError("Malformed parent content hash")


def plan_workflow(scores_path, parent_path, output, **kwargs):
    source = {"scores": {"path": str(Path(scores_path).resolve()), "sha256": sha256_file(scores_path)},
              "parent_execution": {"path": str(Path(parent_path).resolve()), "sha256": sha256_file(parent_path)}}
    plan = build_plan(read_json(scores_path), read_json(parent_path), source_bindings=source, **kwargs)
    target = safe_output(output, [scores_path, parent_path])
    atomic_write_json(target, plan)
    return {"status": plan["status"], "output": str(target), "plan_hash": plan["plan_hash"],
            "conditions": len(plan["conditions"]), "total_rollouts": plan["total_rollouts"],
            "purpose": plan["purpose"], "new_rollouts": 0, "model_execution": False}
