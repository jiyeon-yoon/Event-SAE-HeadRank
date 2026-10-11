"""Freeze auxiliary predictions before Tail rollouts and evaluate the same panel.

The primary Tail protocol and candidate selection are unchanged. This is a
secondary, selected-panel prediction test, not dictionary-wide validation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from event_sae.research.output_head.provenance import (
    atomic_write_json, fingerprint, read_json, sha256_file,
)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ids(values, name):
    if (not isinstance(values, list) or len(set(values)) != len(values)
            or any(type(v) is not int or not 0 <= v < 32768 for v in values)):
        raise ValueError(f"Invalid {name}")
    return set(values)


def validate_prediction_attachment(plan):
    """Pure semantic validation; never reads future outcomes or local files."""
    bindings = plan.get("source_bindings", {})
    attachment = bindings.get("calibrated_predictions")
    if attachment is None:
        return None
    if not isinstance(attachment, dict) or attachment.get("schema_version") != "head_predict_plan_attachment_v1":
        raise ValueError("Unsupported calibrated prediction attachment")
    artifact = attachment.get("predictions")
    if (not isinstance(artifact, dict)
            or artifact.get("schema_version") != "head_predict_disjoint_predictions_v1"
            or artifact.get("prediction_hash") != fingerprint({k: v for k, v in artifact.items() if k != "prediction_hash"})):
        raise ValueError("Prediction artifact hash/schema mismatch")
    if (artifact.get("outcome_labels_read") is not False
            or artifact.get("no_new_feature_outcomes_used") is not True
            or artifact.get("rollouts_authorized") is not False
            or type(artifact.get("new_rollouts")) is not int or artifact["new_rollouts"] != 0):
        raise ValueError("Attachment must be outcome-free offline predictions, not rollout authorization")
    for key in ("model_hash", "model_sha256", "registry_sha256"):
        if not _digest(artifact.get(key)):
            raise ValueError(f"Missing prediction source digest: {key}")
    model = attachment.get("model")
    if (not isinstance(model, dict) or model.get("schema_version") != "head_predict_calibrated_model_v1"
            or model.get("model_hash") != fingerprint({k: v for k, v in model.items() if k != "model_hash"})
            or model.get("model_hash") != artifact["model_hash"]
            or attachment.get("model_file_sha256") != artifact["model_sha256"]
            or model.get("requires_disjoint_inference_feature_ids") is not True
            or model.get("training_feature_ids") != artifact.get("training_feature_ids")
            or artifact.get("predictor") not in model.get("models", {})):
        raise ValueError("Embedded model/training IDs do not match the frozen prediction artifact")
    if not _digest(attachment.get("prediction_file_sha256")):
        raise ValueError("Missing original prediction file digest")
    source = artifact.get("source_bindings", {})
    if (source.get("source_score_hash") != bindings.get("score_hash")
            or source.get("source_scores_sha256") != bindings.get("scores_sha256")
            or source.get("scope") != bindings.get("score_scope")
            or artifact["registry_sha256"] != bindings.get("registry_sha256")):
        raise ValueError("Prediction source scores/scope/registry differ from the frozen Tail plan")
    model_source = model.get("source_bindings", {})
    producer = artifact.get("implementation")
    files = producer.get("files") if isinstance(producer, dict) else None
    if (not isinstance(files, dict) or not files
            or any(not isinstance(name, str) or not name or not _digest(digest) for name, digest in files.items())
            or producer != model_source.get("implementation")):
        raise ValueError("Model and prediction producer implementations are missing or differ")
    for key in ("source_score_hash", "source_scores_sha256", "scope"):
        if model_source.get(key) != source.get(key):
            raise ValueError("Embedded model and prediction source provenance differ")
    panel = _ids(plan.get("panel_feature_ids"), "plan panel")
    eligible = _ids(artifact.get("eligible_feature_ids"), "prediction eligible IDs")
    training = _ids(artifact.get("training_feature_ids"), "prediction training IDs")
    excluded = _ids(artifact.get("excluded_feature_ids"), "prediction excluded IDs")
    known = set(plan.get("known_evaluated_ids", [])) | set(plan.get("additional_evaluated_ids", []))
    if not training or not training <= excluded or not known <= excluded or panel & excluded or not panel <= eligible:
        raise ValueError("Prediction panel includes training/known features or lacks predictions")
    if eligible & excluded:
        raise ValueError("Prediction eligible and excluded feature IDs overlap")
    scored = _ids(plan.get("scored_feature_ids"), "scored feature IDs")
    population = artifact.get("selection_scope")
    if (not eligible <= scored or population not in ("explicit_feature_subset", "all_eligible_disjoint_features")
            or population == "explicit_feature_subset" and eligible != panel
            or population == "all_eligible_disjoint_features" and eligible != scored - excluded):
        raise ValueError("Prediction population is incomplete or differs from declared subset/all-eligible scope")
    tasks = plan.get("task_ids", [])
    if source.get("scope", {}).get("task_ids") != tasks or model.get("task_ids") != tasks:
        raise ValueError("Prediction tasks differ from the frozen plan")
    rows = artifact.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("Prediction task rows are missing")
    by_pair = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid task prediction row")
        fid, task, value = row.get("feature_id"), row.get("task_id"), row.get("predicted_drop")
        if (type(fid) is not int or type(task) is not int or fid not in eligible or task not in tasks
                or isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not -1 <= value <= 1 or (fid, task) in by_pair):
            raise ValueError("Invalid, duplicate or unbounded task prediction")
        by_pair[fid, task] = float(value)
    if set(by_pair) != {(fid, task) for fid in eligible for task in tasks}:
        raise ValueError("Incomplete prediction feature/task coverage")
    aggregate = artifact.get("feature_predictions", [])
    if (not isinstance(aggregate, list) or len(aggregate) != len(eligible)
            or any(not isinstance(r, dict) for r in aggregate)
            or {r.get("feature_id") for r in aggregate} != eligible):
        raise ValueError("Feature prediction coverage mismatch")
    by_feature = {}
    for row in aggregate:
        fid, value = row["feature_id"], row.get("predicted_drop")
        expected = math.fsum(by_pair[fid, task] for task in tasks) / len(tasks)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Aggregate predicted drop differs from equal-task mean")
        by_feature[fid] = float(value)
    if attachment.get("panel_feature_ids") != sorted(panel) or attachment.get("role") != "secondary_fixed_panel_prediction":
        raise ValueError("Attachment panel/analysis role mismatch")
    return {"predictor": artifact["predictor"], "prediction_hash": artifact["prediction_hash"],
            "model_hash": artifact["model_hash"], "by_feature": by_feature, "by_task": by_pair}


def attach_predictions(plan_path, predictions_path, model_path, output):
    from event_sae.research.head_rank.protocol import fingerprint as plan_fingerprint, verify_plan
    from event_sae.research.head_rank.workflow import implementation_identity, safe_output
    from .calibration_workflow import implementation_identity as producer_identity

    original, artifact, model = read_json(plan_path), read_json(predictions_path), read_json(model_path)
    verify_plan(original)
    if original.get("status") != "predictions_frozen_not_runtime_approved":
        raise ValueError("Freeze evaluation cases before attaching predictions")
    if original["source_bindings"].get("implementation") != implementation_identity():
        raise ValueError("Planning source changed; create a new reviewed plan before attaching predictions")
    if "calibrated_predictions" in original["source_bindings"]:
        raise ValueError("Predictions already frozen in this plan; do not replace them")
    if artifact.get("implementation") != producer_identity():
        raise ValueError("Prediction producer implementation changed; recalibrate/predict before attaching")
    if sha256_file(model_path) != artifact.get("model_sha256"):
        raise ValueError("Provided model file differs from the model used to predict")
    target = safe_output(output, [plan_path, predictions_path, model_path])
    plan = deepcopy(original)
    plan["source_bindings"]["calibrated_predictions"] = {
        "schema_version": "head_predict_plan_attachment_v1",
        "role": "secondary_fixed_panel_prediction", "panel_feature_ids": sorted(plan["panel_feature_ids"]),
        "prediction_file_sha256": sha256_file(predictions_path), "predictions": artifact,
        "model": model, "model_file_sha256": sha256_file(model_path),
    }
    validate_prediction_attachment(plan)
    plan["plan_hash"] = plan_fingerprint({k: v for k, v in plan.items() if k != "plan_hash"})
    verify_plan(plan)
    atomic_write_json(target, plan)
    return {"status": "predictions_attached_not_runtime_approved", "output": str(target),
            "plan_hash": plan["plan_hash"], "primary_metric": plan["primary_metric"],
            "secondary_predictor": artifact["predictor"], "total_rollouts": plan["budget"]["total_rollouts"],
            "new_rollouts": 0}


def evaluate_prediction_attachment(plan, effects):
    validated = validate_prediction_attachment(plan)
    if validated is None:
        return None
    from event_sae.research.output_head.results import spearman_correlation

    panel = sorted(plan["panel_feature_ids"])
    by_id = {row["feature_id"]: row for row in effects}
    if len(by_id) != len(effects) or set(by_id) != set(panel):
        raise ValueError("Auxiliary evaluation requires the exact complete frozen feature panel")
    predicted = [validated["by_feature"][fid] for fid in panel]
    actual = [by_id[fid]["drop"] for fid in panel]
    rows = [{"feature_id": fid, "predicted_drop": p, "observed_drop": y, "error": p - y}
            for fid, p, y in zip(panel, predicted, actual)]
    return {"predictor": validated["predictor"], "prediction_hash": validated["prediction_hash"],
            "model_hash": validated["model_hash"], "feature_ids": panel, "predictions": rows,
            "mae": math.fsum(abs(p - y) for p, y in zip(predicted, actual)) / len(panel),
            "spearman": spearman_correlation(predicted, actual), "role": "secondary_fixed_panel_prediction",
            "predictions_bound_before_execution": True, "trained_on_evaluation_features": False,
            "scope": "Tail-selected fixed panel, not a global dictionary accuracy or superiority claim"}
