"""Immutable, label-separated local workflows for developmental calibration.

This module uses JSON and NumPy only. It never starts rollouts or reads edited
outcomes in predict-only mode. A prediction artifact is not an execution plan.
"""
from __future__ import annotations

import math
from pathlib import Path

from event_sae.research.output_head.provenance import (
    atomic_write_json, atomic_write_text, fingerprint, read_json, sha256_file,
)
from event_sae.research.head_rank.workflow import (
    DEVELOPMENT_FEATURES, safe_output, validate_full_score_population,
    implementation_identity as headrank_implementation_identity,
)
from event_sae.research.head_rank.compatibility import validate_scoring_implementation
from event_sae.research.head_rank.protocol import KNOWN_EVALUATED_IDS
from .calibration import (
    ACTION_COLUMNS, HEAD_COLUMNS, calibrate_rows, clip_drop, descriptor_matrix,
    finite, integer, paired_task_labels, predict_ridge, unique_rows,
)


SCOPE_FIELDS = ("dictionary_size", "discovery_manifest_hash", "sample_manifest_hash",
                "split_manifest_hash", "sae_sha256", "task_ids")
FORBIDDEN_OUTCOME_FIELDS = frozenset({"target_drop", "drop", "sr_edit", "edit_sr", "raw_success",
                                    "edit_success", "delta_sr", "outcome_switch_rate"})


def implementation_identity():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).with_name("calibration.py"), Path(__file__),
             root / "event_sae/research/output_head/provenance.py",
             root / "event_sae/research/head_rank/workflow.py",
             root / "event_sae/research/head_rank/protocol.py",
             root / "event_sae/research/head_rank/compatibility.py"]
    return {"files": {str(path.relative_to(root)): sha256_file(path) for path in paths}}


def verify_hash(payload, field):
    if not isinstance(payload, dict) or payload.get(field) != fingerprint({k: v for k, v in payload.items() if k != field}):
        raise ValueError(f"Missing or mismatched {field}")


def _scope_check(actual, expected):
    if any(expected.get(key) is None or actual.get(key) != expected[key] for key in SCOPE_FIELDS):
        raise ValueError("Scope mismatch: dictionary/sample/SAE/discovery/split/tasks must match")
    tasks = expected["task_ids"]
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Scope requires unique task IDs")
    for task in tasks:
        integer(task, "task_id")


def _load_inputs(scores_path, reference_result):
    """No outcomes are read here; shared by fit and predict-only workflows."""
    scores_path, root = Path(scores_path), Path(reference_result)
    reference_path, sample_path = root / "scores/scores.json", root / "sample_manifest.json"
    source_sha, reference_sha, sample_sha = sha256_file(scores_path), sha256_file(reference_path), sha256_file(sample_path)
    scores = read_json(scores_path)
    verify_hash(scores, "score_hash")
    validate_full_score_population(scores)
    compatibility = validate_scoring_implementation(scores.get("implementation"), headrank_implementation_identity())
    if scores.get("synthetic") is not False or scores.get("development_only") is not False or scores.get("feature_set") != "all-active":
        raise ValueError("Require real full all-active source scores")
    if scores.get("source_scores_sha256") != reference_sha or scores.get("source_sample_sha256") != sample_sha:
        raise ValueError("Source scores/reference/sample file hash binding mismatch")
    reference, sample = read_json(reference_path), read_json(sample_path)
    _scope_check(reference.get("scope", {}), scores.get("scope", {}))
    width = integer(scores["scope"].get("dictionary_size"), "dictionary_size")
    if width < 1 or any(integer(fid, "source feature_id") >= width for fid in scores["feature_ids"]):
        raise ValueError("Source feature ID outside declared dictionary")
    if reference.get("synthetic") is not False or sample.get("synthetic") is not False:
        raise ValueError("Reference/sample must be real rather than synthetic")
    if (sample.get("sample_hash") != scores["scope"]["sample_manifest_hash"]
            or sample.get("discovery_manifest_hash") != scores["scope"]["discovery_manifest_hash"]
            or sample.get("split_manifest_hash") != scores["scope"]["split_manifest_hash"]
            or fingerprint(sample.get("readouts")) != scores.get("readout_record_hash")):
        raise ValueError("Sample manifest scope/readout identity mismatch")
    if reference.get("identity_hash") != fingerprint(reference.get("identity")):
        raise ValueError("Reference numerical identity hash mismatch")
    identity = scores.get("identity", {})
    for field in ("model_checkpoint", "model_revision", "code_revision", "layer_idx", "alpha", "capture_target",
                  "head_weights_hash", "sae_checkpoint_hash", "score_definition"):
        if identity.get(field) is None or identity[field] != reference["identity"].get(field):
            raise ValueError(f"HeadRank/reference numerical identity mismatch: {field}")
    if identity["alpha"] != 0 or identity["capture_target"] != "post_mlp_residual":
        raise ValueError("Calibration requires complete post-MLP residual ablation")
    if scores.get("identity", {}).get("sae_checkpoint_hash") != scores["scope"]["sae_sha256"]:
        raise ValueError("Source numerical SAE identity mismatch")
    event_scope = scores.get("event_max_binding", {}).get("scope", {})
    for name in ("dictionary_size", "sae_sha256", "discovery_manifest_hash", "split_manifest_hash", "task_ids"):
        if event_scope.get(name) != scores["scope"][name]:
            raise ValueError("Event-max discovery scope mismatch")
    bindings = {"source_scores_sha256": source_sha, "source_score_hash": scores["score_hash"],
                "reference_scores_sha256": reference_sha, "sample_manifest_sha256": sample_sha,
                "scope": scores["scope"], "readout_record_hash": scores["readout_record_hash"],
                "event_max_binding": scores["event_max_binding"], "scorer_compatibility": compatibility}
    return scores, reference, bindings


def _load_descriptors(path, bindings, *, source_scores, reference_scores):
    if path is None:
        return None, None
    if source_scores is None or reference_scores is None:
        raise ValueError("Response validation requires complete source/reference score metadata")
    descriptor_sha = sha256_file(path)
    payload = read_json(path)
    verify_hash(payload, "descriptor_hash")
    if (payload.get("schema_version") != "head_response_descriptors_v1"
            or payload.get("synthetic") is not False or payload.get("full_readouts") is not True
            or payload.get("sampling_mode") != "full"):
        raise ValueError("Calibration requires real full-readout response descriptors, not smoke data")
    _scope_check(payload.get("scope", {}), bindings["scope"])
    if (payload.get("source_sample_sha256") != bindings["sample_manifest_sha256"]
            or payload.get("readout_record_hash") != bindings["readout_record_hash"]):
        raise ValueError("Response descriptors use a different sample/readout population")
    # Native descriptor extraction binds the frozen reference, which predates
    # score_hash. Do not manufacture a full-score hash for that older artifact.
    if payload.get("reference_scores_sha256") != bindings["reference_scores_sha256"]:
        raise ValueError("Descriptor/reference source hash mismatch")
    unique_rows(payload.get("rows"), "response descriptor")
    for row in payload["rows"]:
        if FORBIDDEN_OUTCOME_FIELDS & set(row):
            raise ValueError("Response descriptor file must not contain outcomes")
    from .descriptor_workflow import descriptor_implementation_identity
    if payload.get("implementation") != descriptor_implementation_identity():
        raise ValueError("Response descriptor implementation changed after extraction")
    mapping_hash = payload.get("action_mapping_sha256")
    if not isinstance(mapping_hash, str) or len(mapping_hash) != 64:
        raise ValueError("Response descriptors require a bound action mapping")
    from .descriptor_workflow import validate_descriptor_payload
    validate_descriptor_payload(payload, source_scores=source_scores,
                                reference_scores=reference_scores, bindings=bindings)
    return payload, {"sha256": descriptor_sha, "descriptor_hash": payload["descriptor_hash"],
                     "scope": payload["scope"], "schema_version": payload["schema_version"],
                     "implementation": payload["implementation"], "action_mapping_sha256": mapping_hash,
                     "action_bin_decode": payload.get("action_bin_decode")}


def build_descriptor_rows(scores, reference, feature_ids, *, response=None):
    """Task-aware joins; Event-max alone is a feature-level discovery aggregate."""
    tasks = scores["scope"]["task_ids"]
    rows = unique_rows(reference.get("feature_task_scores"), "reference descriptor")
    source = {r["feature_id"]: r for r in scores["feature_scores"]}
    response_rows = unique_rows(response["rows"], "response descriptor") if response is not None else None
    wanted = {(fid, task) for fid in feature_ids for task in tasks}
    if not wanted.issubset(rows):
        raise ValueError("Missing reference descriptor feature/task rows")
    if response_rows is not None and not wanted.issubset(response_rows):
        raise ValueError("Missing response descriptor feature/task rows; no silent row exclusion")
    if response_rows is not None:
        if any(fid not in source or task not in tasks for fid, task in response_rows):
            raise ValueError("Response descriptors outside frozen feature/task population")
        response_ids = {fid for fid, _ in response_rows}
        if set(response_rows) != {(fid, task) for fid in response_ids for task in tasks}:
            raise ValueError("Partial feature/task coverage in response descriptors")
    output = []
    for fid in sorted(feature_ids):
        if fid not in source:
            raise ValueError("Requested feature is absent from source scores")
        task_scores = unique_rows([{**r, "feature_id": fid} for r in source[fid].get("task_scores", [])], "HeadRank task score")
        if set(task_scores) != {(fid, task) for task in tasks}:
            raise ValueError("HeadRank task scope mismatch")
        for task in sorted(tasks):
            ref = rows[(fid, task)]
            if ref.get("alpha") != 0:
                raise ValueError("Descriptors require alpha=0 complete removal")
            row = {"feature_id": fid, "task_id": task,
                   "event_max": finite(source[fid].get("event_max"), "event_max"),
                   "mean_head_kl": finite(task_scores[(fid, task)].get("mean"), "mean_head_kl")}
            legacy_kl = finite(ref.get("head_full_vocab_kl_fixed_prefix"), "reference KL")
            if not math.isclose(row["mean_head_kl"], legacy_kl, rel_tol=1e-6, abs_tol=1e-10):
                raise ValueError("Full-score/reference per-task KL mismatch")
            for name in HEAD_COLUMNS[1:]:
                row[name] = finite(ref.get(name), name)
            if any(row[name] < 0 for name in ("mean_head_kl", "mean_edit_norm", "event_max")):
                raise ValueError("Magnitude descriptors must be nonnegative")
            if any(not 0 <= row[name] <= 1 for name in ("full_vocab_argmax_flip_rate", "readout_activation_frequency")):
                raise ValueError("Frequency descriptors must be in [0, 1]")
            if response_rows is not None:
                richer = response_rows[(fid, task)]
                # Keep the matched baseline columns fixed: adding richer input
                # must not silently change the equal-budget baseline designs.
                for name in HEAD_COLUMNS:
                    finite(richer.get(name), name)
                for name in ("signed_action_bin_shift", "absolute_action_bin_shift", "action_argmax_flip"):
                    values = richer.get(name)
                    if not isinstance(values, list) or len(values) != 7:
                        raise ValueError(f"{name} requires all seven ordered action dimensions")
                    for index, value in enumerate(values):
                        value = finite(value, name)
                        if name == "absolute_action_bin_shift" and value < 0:
                            raise ValueError("Absolute action shifts must be nonnegative")
                        if name == "action_argmax_flip" and not 0 <= value <= 1:
                            raise ValueError("Action argmax flip frequencies must be in [0, 1]")
                        row[f"{name}_{index}"] = value
            output.append(row)
    return output


def _markdown(report):
    lines = ["# Historical signed-drop calibration — development diagnostic", "",
             "Target: raw SR − ablation SR. Positive means harm; negative means repair.", "",
             f"Independent feature units: {report['independent_feature_units']}; feature-task rows: {report['feature_task_rows']}.",
             "All rows of a held-out feature are excluded from fitting and standardization.", "",
             "| Predictor | LOFO feature-mean MAE | LOFO Spearman | Noncatastrophic-only LOFO MAE |",
             "|---|---:|---:|---:|"]
    for name, data in report["models"].items():
        overall = data["all_features"]
        rho = "undefined (constant)" if overall["spearman"] is None else f"{overall['spearman']:.6f}"
        subset = data.get("noncatastrophic_only_lofo", {}).get("feature_mean_mae")
        lines.append(f"| {name} | {overall['feature_mean_mae']:.6f} | {rho} | {subset if subset is not None else 'unavailable'} |")
    lines += ["", "MAE is in success-rate units (multiply by 100 for percentage points).",
              "Noncatastrophic-only LOFO removes 7024 and 24830 from both training and evaluation.",
              "The JSON also reports the full-LOFO noncatastrophic subset and joint holdout of both fixed IDs.", "",
              "No confirmatory p-values, automatic rollout gate, or published-new-method claim. Labels have been repeatedly inspected.",
              "The final model is a separate fit on all historical development labels; its fitted values are not held-out evidence.",
              "Predict-only requires disjoint features, excludes the historical known-22 and reviewed team registry, and cannot authorize rollouts.", ""]
    return "\n".join(lines)


def calibration_workflow(scores_path, reference_result, output, *, descriptors_path=None, ridge_alpha=1.0):
    scores, reference, bindings = _load_inputs(scores_path, reference_result)
    root = Path(reference_result)
    labels_path = root / "analysis/paired_effects.json"
    target = safe_output(output, [scores_path, root, descriptors_path, *scores.get("source_readonly_roots", [])])
    implementation = implementation_identity()
    label_sha = sha256_file(labels_path)
    paired = read_json(labels_path)
    if paired.get("synthetic") is not False or paired.get("protocol", {}).get("sae_sha256") != scores["scope"]["sae_sha256"]:
        raise ValueError("Historical label synthetic/SAE identity mismatch")
    label_binding = _validate_label_binding(paired, scores)
    label_rows = paired_task_labels(paired, task_ids=scores["scope"]["task_ids"], feature_ids=DEVELOPMENT_FEATURES)
    response, response_binding = _load_descriptors(descriptors_path, bindings, source_scores=scores, reference_scores=reference)
    design = build_descriptor_rows(scores, reference, sorted({r["feature_id"] for r in label_rows}), response=response)
    report, model = calibrate_rows(design, label_rows, ridge_alpha=ridge_alpha, with_response=response is not None)
    bindings.update(labels_sha256=label_sha, label_protocol_id=paired["protocol_id"],
                    label_semantics=label_binding, descriptors=response_binding, implementation=implementation)
    report.update(schema_version="head_predict_calibration_report_v1", status="development_diagnostic",
                  source_bindings=bindings, target_pairing={"paired_cases_per_feature": paired["num_shared_cases"],
                  "total_paired_rows": len(paired["paired_rows"]), "num_independent_feature_labels": len(DEVELOPMENT_FEATURES),
                  "identical_raw_cases_and_outcomes_verified": True},
                  training_target_rows=label_rows,
                  evaluation_labels_previously_used=paired["protocol"].get("evaluation_labels_previously_used"),
                  new_rollouts=0, model_download=False, sae_training=False)
    model.update(schema_version="head_predict_calibrated_model_v1", source_bindings=bindings,
                 diagnostic_report_schema=report["schema_version"],
                 known_evaluated_feature_ids=sorted(KNOWN_EVALUATED_IDS),
                 rollout_authorization=False, prediction_semantics="signed_drop_clipped_by_shared_historical_raw_SR")
    model["model_hash"] = fingerprint(model)
    report["model_hash"] = model["model_hash"]
    report["report_hash"] = fingerprint(report)
    if implementation != implementation_identity():
        raise ValueError("Calibration source changed during execution")
    _verify_inputs_unchanged(scores_path, root, descriptors_path, bindings)
    if sha256_file(labels_path) != label_sha:
        raise ValueError("Historical label artifact changed during execution")
    target.mkdir(parents=True, exist_ok=False)
    atomic_write_json(target / "report.json", report)
    atomic_write_json(target / "model.json", model)
    atomic_write_text(target / "report.md", _markdown(report))
    return {"status": report["status"], "output": str(target), "report_hash": report["report_hash"],
            "model_hash": model["model_hash"], "num_training_features": len(model["training_feature_ids"]),
            "gate": report["gate"], "new_rollouts": 0}


def predict_workflow(model_path, scores_path, reference_result, output, *, registry,
                     descriptors_path=None, predictor="head_ridge", feature_ids=None):
    """Predict all eligible disjoint features, WITHOUT reading a label artifact.

    The reviewed registry is a list of known feature IDs, never outcome values.
    This is offline prediction only and does not select or run a robot panel.
    """
    model_sha, registry_sha = sha256_file(model_path), sha256_file(registry)
    model = read_json(model_path)
    verify_hash(model, "model_hash")
    if (model.get("schema_version") != "head_predict_calibrated_model_v1"
            or model.get("requires_disjoint_inference_feature_ids") is not True
            or model.get("source_bindings", {}).get("implementation") != implementation_identity()):
        raise ValueError("Incompatible model schema, disjointness guard or implementation")
    scores, reference, bindings = _load_inputs(scores_path, reference_result)
    for name, value in bindings.items():
        if model["source_bindings"].get(name) != value:
            raise ValueError(f"Model/prediction source binding mismatch: {name}")
    if predictor not in model.get("models", {}):
        raise ValueError("Predictor not present in the frozen model artifact")
    fit = model["models"][predictor]
    needs_response = bool(set(fit.get("input_columns", [])) & set(ACTION_COLUMNS))
    if needs_response and descriptors_path is None:
        raise ValueError("Response model inference requires bound response descriptors")
    response, response_binding = _load_descriptors(descriptors_path, bindings, source_scores=scores, reference_scores=reference)
    if needs_response:
        fitted_binding = model["source_bindings"].get("descriptors") or {}
        for field in ("schema_version", "scope", "implementation", "action_mapping_sha256", "action_bin_decode"):
            if fitted_binding.get(field) != response_binding[field]:
                raise ValueError(f"Response descriptor semantics differ from training: {field}")
    registry_payload = read_json(registry)
    known = registry_payload.get("feature_ids")
    if registry_payload.get("reviewed_for_team_prior_outcomes") is not True or not isinstance(known, list):
        raise ValueError("Explicit team-reviewed known-feature registry is required")
    if len(set(known)) != len(known):
        raise ValueError("Duplicate registry feature IDs")
    for fid in known:
        if integer(fid, "registry feature_id") >= scores["scope"]["dictionary_size"]:
            raise ValueError("Registry feature ID outside dictionary")
    training = model.get("training_feature_ids")
    if not isinstance(training, list) or not training or len(set(training)) != len(training):
        raise ValueError("Model must retain distinct training feature IDs")
    for fid in training:
        integer(fid, "training feature_id")
    if model.get("known_evaluated_feature_ids") != sorted(KNOWN_EVALUATED_IDS):
        raise ValueError("Historical known-feature exclusion registry changed")
    excluded = set(training) | set(known) | KNOWN_EVALUATED_IDS
    eligible = sorted(row["feature_id"] for row in scores["feature_scores"]
                      if row["feature_id"] not in excluded and row.get("direction_excess_status") == "ok")
    if feature_ids is not None:
        if not isinstance(feature_ids, (list, tuple)) or not feature_ids or len(set(feature_ids)) != len(feature_ids):
            raise ValueError("Requested prediction feature IDs must be a distinct nonempty list")
        for fid in feature_ids:
            integer(fid, "requested feature_id")
        if not set(feature_ids).issubset(eligible):
            raise ValueError("Requested prediction features are absent, ineligible or previously evaluated/training IDs")
        eligible = sorted(feature_ids)
    if not eligible:
        raise ValueError("No eligible disjoint features remain")
    design = build_descriptor_rows(scores, reference, eligible, response=response if needs_response else None)
    import numpy as np
    values = predict_ridge(fit, descriptor_matrix(design, fit["input_columns"])) if fit["kind"] == "ridge" else np.zeros(len(design))
    if model.get("task_ids") != scores["scope"]["task_ids"]:
        raise ValueError("Model task scope mismatch")
    raw = [finite(model.get("raw_sr_by_task", {}).get(str(row["task_id"])), "training raw SR") for row in design]
    clipped = clip_drop(values, raw)
    rows = [{"feature_id": row["feature_id"], "task_id": row["task_id"],
             "predicted_drop": float(clipped[i]), "unclipped_predicted_drop": float(values[i])}
            for i, row in enumerate(design)]
    by_id = {fid: [] for fid in eligible}
    for row in rows:
        by_id[row["feature_id"]].append(row["predicted_drop"])
    result = {"schema_version": "head_predict_disjoint_predictions_v1", "status": "offline_predictions_only",
              "predictor": predictor, "model_sha256": model_sha, "model_hash": model["model_hash"],
              "source_bindings": bindings, "descriptors_binding": response_binding,
              "registry_sha256": registry_sha, "training_feature_ids": training,
              "excluded_feature_ids": sorted(excluded), "eligible_feature_ids": eligible,
              "selection_scope": "explicit_feature_subset" if feature_ids is not None else "all_eligible_disjoint_features",
              "predictions": rows, "feature_predictions": [{"feature_id": fid, "predicted_drop": float(np.mean(by_id[fid]))} for fid in eligible],
              "outcome_labels_read": False, "no_new_feature_outcomes_used": True, "new_rollouts": 0,
              "rollouts_authorized": False, "clipping_raw_sr_source": "shared_historical_training_baseline",
              "developmental_model_not_independent_validation": True,
              "implementation": implementation_identity()}
    result["prediction_hash"] = fingerprint(result)
    target = safe_output(output, [model_path, scores_path, reference_result, registry, descriptors_path,
                                  *scores.get("source_readonly_roots", [])])
    _verify_inputs_unchanged(scores_path, Path(reference_result), descriptors_path,
                             {**bindings, "descriptors": response_binding})
    if sha256_file(model_path) != model_sha or sha256_file(registry) != registry_sha:
        raise ValueError("Model or registry changed during prediction")
    if model["source_bindings"]["implementation"] != implementation_identity():
        raise ValueError("Predictor source changed during execution")
    atomic_write_json(target, result)
    return {"status": result["status"], "output": str(target), "prediction_hash": result["prediction_hash"],
            "num_eligible_features": len(eligible), "rollouts_authorized": False, "new_rollouts": 0}


def _verify_inputs_unchanged(scores_path, root, descriptors_path, bindings):
    paths = {"source_scores_sha256": scores_path, "reference_scores_sha256": root / "scores/scores.json",
             "sample_manifest_sha256": root / "sample_manifest.json"}
    if any(sha256_file(path) != bindings[name] for name, path in paths.items()):
        raise ValueError("Scoring inputs changed during execution")
    if descriptors_path is not None and sha256_file(descriptors_path) != bindings["descriptors"]["sha256"]:
        raise ValueError("Response descriptor artifact changed during execution")


def _validate_label_binding(paired, scores):
    """Use recorded legacy evidence without inventing missing hook attestation."""
    protocol, identity = paired["protocol"], scores["identity"]
    matched = {}
    for label_field, score_field in (("model_checkpoint", "model_checkpoint"), ("model_revision", "model_revision"),
                                     ("model_code_revision", "code_revision"), ("layer_idx", "layer_idx"),
                                     ("alpha", "alpha"), ("sae_sha256", "sae_checkpoint_hash")):
        if protocol.get(label_field) is None or protocol[label_field] != identity[score_field]:
            raise ValueError(f"Label/scoring intervention identity mismatch: {label_field}")
        matched[label_field] = protocol[label_field]
    if protocol.get("hook_start_step") != 0 or protocol.get("primary_label") != "success_rate_drop":
        raise ValueError("Labels require whole-trajectory intervention and signed success-rate drop")
    if protocol.get("suite") != identity.get("unnorm_key"):
        raise ValueError("Label suite differs from the scored action normalization suite")
    if protocol.get("state_identity_scope") != scores["scope"].get("state_identity_scope"):
        raise ValueError("Historical state identity limitations differ between labels and scores")
    unrecorded = []
    for name in ("capture_target", "head_weights_hash"):
        if name in protocol:
            if protocol[name] != identity[name]:
                raise ValueError(f"Label/scoring intervention identity mismatch: {name}")
            matched[name] = protocol[name]
        else:
            unrecorded.append(name)
    return {"matched_recorded_fields": matched, "hook_start_step": 0,
            "legacy_protocol_unrecorded_fields": unrecorded,
            "historical_capture_target_independently_attested": "capture_target" in protocol,
            "historical_head_weights_independently_attested": "head_weights_hash" in protocol,
            "source_scoring_capture_target": identity["capture_target"],
            "source_scoring_head_weights_hash": identity["head_weights_hash"],
            "historical_runtime_flags_verified": scores["scope"].get("state_identity_scope", {}).get("historical_runtime_flags_verified", False)}
