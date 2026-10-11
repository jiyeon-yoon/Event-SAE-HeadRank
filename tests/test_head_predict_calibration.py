"""CPU-only regression/leakage/integrity tests; no model or simulator needed."""
import copy
import json

import numpy as np
import pytest

from event_sae.research.head_predict.calibration import (
    calibrate_rows, clip_drop, descriptor_matrix, fit_ridge, paired_task_labels,
    predict_ridge, feature_metrics, ACTION_COLUMNS,
)
from event_sae.research.head_predict.calibration_workflow import (
    _load_descriptors, _validate_label_binding, build_descriptor_rows, calibration_workflow, predict_workflow,
)
from event_sae.research.head_rank.workflow import DEVELOPMENT_FEATURES
from event_sae.research.head_rank.workflow import implementation_identity as headrank_implementation_identity
from event_sae.research.output_head.provenance import fingerprint, sha256_file


def descriptor_rows(features=(1, 2, 3, 7024, 24830), tasks=(0, 1)):
    return [{"feature_id": fid, "task_id": task, "mean_head_kl": float(i + task),
             "event_max": float(i * 2), "mean_edit_norm": float(i + 1),
             "full_vocab_argmax_flip_rate": .1 * i, "readout_activation_frequency": .1 * task}
            for i, fid in enumerate(features) for task in tasks]


def labels(features=(1, 2, 3, 7024, 24830), tasks=(0, 1)):
    return [{"feature_id": fid, "task_id": task, "target_drop": .02 * (i - 1), "raw_sr": .5}
            for i, fid in enumerate(features) for task in tasks]


def paired_fixture(features=(1, 2, 3), tasks=(0, 1)):
    cases = [{"suite": "test", "task_id": task, "task_episode_idx": case,
              "initial_state_sha256": f"{task * 10 + case:064x}"}
             for task in tasks for case in range(2)]
    paired_rows, effects = [], []
    for index, fid in enumerate(features):
        task_effects, harm, repair = [], 0, 0
        for task in tasks:
            values = []
            for case in [c for c in cases if c["task_id"] == task]:
                raw = case["task_episode_idx"] == 0
                # feature 1 harms one raw success AND repairs a raw failure.
                edit = not raw if index == 0 else (False if index == 1 else raw)
                values.append((raw, edit))
                paired_rows.append({**case, "seed": 0, "protocol_id": "protocol", "feature_id": fid,
                                    "raw_success": raw, "edit_success": edit, "drop": int(raw) - int(edit)})
                harm += int(raw and not edit)
                repair += int(not raw and edit)
            sr_raw, sr_edit = .5, sum(b for _, b in values) / 2
            task_effects.append({"task_id": task, "suite": "test", "sr_raw": sr_raw, "sr_edit": sr_edit,
                                 "drop": sr_raw - sr_edit, "num_valid_pairs": 2})
        effects.append({"feature_id": fid, "alpha": 0., "protocol_id": "protocol", "sr_raw": .5,
                        "sr_edit": float(np.mean([r["sr_edit"] for r in task_effects])),
                        "drop": float(np.mean([r["drop"] for r in task_effects])),
                        "num_valid_pairs": len(cases), "n_success_to_failure": harm,
                        "n_failure_to_success": repair, "task_effects": task_effects})
    return {"feature_ids": list(features), "protocol_id": "protocol", "num_shared_cases": len(cases),
            "protocol": {"alpha": 0., "allow_missing_pairs": False, "protocol_id": "protocol",
                         "expected_feature_ids": list(features), "eval_cases": cases, "suite": "test",
                         "model_checkpoint": "test-model", "model_revision": "model-revision",
                         "model_code_revision": "code-revision", "layer_idx": 31,
                         "hook_start_step": 0, "primary_label": "success_rate_drop",
                         "sae_sha256": "s" * 64, "evaluation_labels_previously_used": True},
            "paired_rows": paired_rows, "feature_effects": effects, "synthetic": False}


def test_pairing_target_distinguishes_harm_and_repair_not_switch_rate():
    result = paired_task_labels(paired_fixture(), task_ids=[0, 1])
    assert result[0]["target_drop"] == 0
    assert result[0]["n_success_to_failure"] == result[0]["n_failure_to_success"] == 1
    assert result[2]["target_drop"] == .5


@pytest.mark.parametrize("change", ["duplicate", "raw_mismatch", "state_mismatch", "missing", "sign", "summary", "protocol"])
def test_pairing_rejects_inconsistent_or_missing_cases(change):
    paired = paired_fixture()
    if change == "duplicate": paired["paired_rows"].append(copy.deepcopy(paired["paired_rows"][0]))
    elif change == "raw_mismatch":
        paired["paired_rows"][0]["raw_success"] = False
        paired["paired_rows"][0]["drop"] = 0
    elif change == "state_mismatch": paired["paired_rows"][0]["initial_state_sha256"] = "a" * 64
    elif change == "missing": paired["paired_rows"].pop()
    elif change == "sign": paired["paired_rows"][0]["drop"] *= -1
    elif change == "summary": paired["feature_effects"][0]["drop"] = .5
    elif change == "protocol": paired["protocol"]["alpha"] = 1
    with pytest.raises(ValueError): paired_task_labels(paired, task_ids=[0, 1])


def test_lofo_excludes_entire_feature_and_standardizes_training_only():
    rows, targets = descriptor_rows(), labels()
    for row in rows:
        if row["feature_id"] == 1: row["mean_head_kl"] = 100000.
    report, model = calibrate_rows(rows, targets)
    fold = report["models"]["kl_ridge"]["folds"][0]
    assert fold["held_out_feature_ids"] == [1]
    assert 1 not in fold["training_feature_ids"]
    assert fold["test_rows"] == 2 and fold["training_rows"] == 8
    assert fold["fit"]["mean"] == [3.0]
    assert model["models"]["kl_ridge"]["mean"][0] > 1000
    for info in report["models"].values():
        for fold in info["folds"]:
            assert not set(fold["training_feature_ids"]) & set(fold["held_out_feature_ids"])
        holdout = info["joint_catastrophic_holdout"]["fold"]
        assert holdout["held_out_feature_ids"] == [7024, 24830]
        assert holdout["training_feature_ids"] == [1, 2, 3]
        assert info["noncatastrophic_only_lofo"]["num_features"] == 3


def test_task_join_is_by_keys_not_position_and_missing_duplicate_error():
    rows, targets = descriptor_rows(), labels()
    first, _ = calibrate_rows(rows, targets)
    second, _ = calibrate_rows(list(reversed(rows)), targets[2:] + targets[:2])
    assert first == second
    with pytest.raises(ValueError, match="missing or extra"):
        calibrate_rows(rows[:-1], targets)
    with pytest.raises(ValueError, match="Duplicate"):
        calibrate_rows(rows + rows[:1], targets)


def test_no_feature_or_task_ids_or_training_targets_as_covariates():
    rows = descriptor_rows()
    # Even malicious extra label-like columns cannot enter the explicit design.
    polluted = [{**r, "target_drop": 1e100, "edit_success": True} for r in rows]
    report, model = calibrate_rows(rows, labels())
    report2, model2 = calibrate_rows(polluted, labels())
    assert report == report2 and model == model2
    for name in ("feature_id", "task_id", "target_drop", "raw_sr"):
        with pytest.raises(ValueError, match="forbidden"):
            descriptor_matrix(polluted, [name])


def test_ridge_constant_and_nonfinite_policy_and_clip_bounds():
    fit = fit_ridge([[2, 0], [2, 1], [2, 2]], [0., 1., 2.], input_columns=["mean_edit_norm", "mean_head_kl"])
    assert fit["constant_columns"] == [0] and fit["coef"][0] == 0 and fit["scale"][0] == 1
    np.testing.assert_allclose(predict_ridge(fit, [[999999, 1]]), [1])
    np.testing.assert_allclose(clip_drop([9, -9, 4, -4], [.8, .8, 0, 1]), [.8, -.2, 0, 0], rtol=0, atol=1e-15)
    with pytest.raises(ValueError): fit_ridge([[float("nan")]], [0])
    with pytest.raises(ValueError): fit_ridge([[1]], [0], ridge_alpha=0)
    with pytest.raises(ValueError): clip_drop([0], [1.1])


def test_zero_baseline_undefined_correlation_is_null():
    report, _ = calibrate_rows(descriptor_rows(), labels())
    assert report["models"]["zero"]["all_features"]["spearman"] is None
    assert report["gate"]["rollouts_authorized"] is False
    assert report["hyperparameter_search"] is False


def test_exact_harm_repair_cancellation_preserves_zero_target_rank_tie():
    values = [-.05, .1, -.1, -.05, .05, 0, 0, .05, .1, -.1]
    result = feature_metrics([{"feature_id": 18471, "target_drop": value, "predicted_drop": 0.}
                              for value in values])
    assert result["per_feature"][0]["target_drop"] == 0.0


def test_response_and_incremental_models_use_the_same_folds_and_baselines():
    rows = [{**row, **{name: 0. for name in ACTION_COLUMNS}} for row in descriptor_rows()]
    baseline, _ = calibrate_rows(rows, labels())
    richer, model = calibrate_rows(rows, labels(), with_response=True)
    for name in baseline["models"]:
        assert baseline["models"][name] == richer["models"][name]
    assert len(model["models"]["response_ridge"]["input_columns"]) == 25
    assert len(model["models"]["event_plus_head_ridge"]["input_columns"]) == 5
    assert richer["models"]["response_ridge"]["all_features"] == richer["models"]["head_ridge"]["all_features"]
    for model_data in richer["models"].values():
        assert [fold["training_feature_ids"] for fold in model_data["folds"]] == [
            fold["training_feature_ids"] for fold in baseline["models"]["head_ridge"]["folds"]]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def workflow_fixture(tmp_path):
    root = tmp_path / "reference"
    features = sorted([*DEVELOPMENT_FEATURES, 900])
    scope = {"dictionary_size": 32768, "task_ids": [0, 1], "sample_manifest_hash": "m" * 64,
             "sae_sha256": "s" * 64, "discovery_manifest_hash": "d" * 64, "split_manifest_hash": "p" * 64}
    readouts = [{"task_id": task, "index": index} for task in range(2) for index in range(7)]
    sample = {"synthetic": False, "readouts": readouts, "sample_hash": scope["sample_manifest_hash"],
              "discovery_manifest_hash": scope["discovery_manifest_hash"], "split_manifest_hash": scope["split_manifest_hash"]}
    ref_rows = [{**row, "alpha": 0., "head_full_vocab_kl_fixed_prefix": row["mean_head_kl"]}
                for row in descriptor_rows(features)]
    # Keep frequencies valid for the larger fixture.
    for row in ref_rows: row["full_vocab_argmax_flip_rate"] /= len(features)
    identity = {"model_checkpoint": "test-model", "model_revision": "model-revision", "code_revision": "code-revision",
                "layer_idx": 31, "alpha": 0., "capture_target": "post_mlp_residual", "head_weights_hash": "h" * 64,
                "sae_checkpoint_hash": scope["sae_sha256"], "score_definition": "head_full_vocab_kl_fixed_prefix", "unnorm_key": "test"}
    reference = {"feature_task_scores": ref_rows, "scope": scope, "synthetic": False,
                 "identity": identity, "identity_hash": fingerprint(identity)}
    write_json(root / "scores/scores.json", reference)
    write_json(root / "sample_manifest.json", sample)
    write_json(root / "analysis/paired_effects.json", paired_fixture(DEVELOPMENT_FEATURES))
    scores = {"schema_version": "head_rank_scores_v1", "sampling_mode": "full", "synthetic": False,
              "development_only": False, "feature_set": "all-active", "scope": scope,
              "feature_ids": features, "scored_feature_ids": features, "frozen_active_feature_ids": features,
              "identity": identity, "event_max_binding": {"scope": scope}, "implementation": headrank_implementation_identity(),
              "readout_record_hash": fingerprint(readouts), "source_scores_sha256": sha256_file(root / "scores/scores.json"),
              "source_sample_sha256": sha256_file(root / "sample_manifest.json"),
              "query_metadata": [{"query_index": i, "task_id": i, "episode_uid": str(i), "step_in_episode": 0,
                                  "num_action_positions": 7} for i in range(2)],
              "scoring_sample": {"readouts_used": 14, "discovery_episodes_used": 2, "event_max_matches_scoring_episode_population": True},
              "feature_scores": [{"feature_id": fid, "event_max": i * 2., "num_queries": 2, "num_readouts": 14,
                                  "direction_excess_status": "ok", "task_scores": [{"task_id": t, "mean": float(i + t)} for t in range(2)]}
                                 for i, fid in enumerate(features)]}
    scores["score_hash"] = fingerprint(scores)
    score_path = tmp_path / "source_scores.json"
    registry = tmp_path / "registry.json"
    write_json(score_path, scores)
    write_json(registry, {"reviewed_for_team_prior_outcomes": True, "feature_ids": []})
    return root, score_path, registry


def test_response_missing_task_is_rejected_not_silently_dropped(tmp_path):
    root, source, _ = workflow_fixture(tmp_path)
    scores = json.loads(source.read_text())
    reference = json.loads((root / "scores/scores.json").read_text())
    response = {"rows": [{**row, **{name: [0.] * 7 for name in (
        "signed_action_bin_shift", "absolute_action_bin_shift", "action_argmax_flip")}}
        for row in reference["feature_task_scores"]]}
    before = build_descriptor_rows(scores, reference, list(DEVELOPMENT_FEATURES), response=response)
    assert len(before) == 32
    response["rows"].pop()
    with pytest.raises(ValueError, match="Missing response"):
        build_descriptor_rows(scores, reference, list(DEVELOPMENT_FEATURES), response=response)


@pytest.mark.parametrize("field", ["model_checkpoint", "model_revision", "model_code_revision", "layer_idx",
                                  "alpha", "capture_target", "head_weights_hash", "hook_start_step", "suite"])
def test_historical_label_binding_checks_recorded_evidence(field, tmp_path):
    root, source, _ = workflow_fixture(tmp_path)
    scores = json.loads(source.read_text())
    paired = json.loads((root / "analysis/paired_effects.json").read_text())
    evidence = _validate_label_binding(paired, scores)
    assert evidence["legacy_protocol_unrecorded_fields"] == ["capture_target", "head_weights_hash"]
    assert evidence["historical_capture_target_independently_attested"] is False
    paired["protocol"][field] = "incorrect"
    with pytest.raises(ValueError): _validate_label_binding(paired, scores)


def test_immutable_workflow_and_predict_never_reads_outcomes(tmp_path, monkeypatch):
    root, source, registry = workflow_fixture(tmp_path)
    output = tmp_path / "fit"
    result = calibration_workflow(source, root, output)
    assert result["num_training_features"] == 16 and result["new_rollouts"] == 0
    assert (output / "report.md").exists()
    with pytest.raises(FileExistsError): calibration_workflow(source, root, output)
    from event_sae.research.head_predict import calibration_workflow as module
    real_read = module.read_json
    def no_outcomes(path):
        assert "paired_effects" not in str(path)
        return real_read(path)
    monkeypatch.setattr(module, "read_json", no_outcomes)
    destination = tmp_path / "predictions.json"
    predict_workflow(output / "model.json", source, root, destination, registry=registry, feature_ids=[900])
    predicted = json.loads(destination.read_text())
    assert predicted["eligible_feature_ids"] == [900]
    assert predicted["outcome_labels_read"] is False and predicted["rollouts_authorized"] is False
    assert not set(predicted["eligible_feature_ids"]) & set(predicted["training_feature_ids"])
    assert not any("target_drop" in row or "edit_sr" in row for row in predicted["predictions"])
    with pytest.raises(ValueError, match="previously evaluated"):
        predict_workflow(output / "model.json", source, root, tmp_path / "bad.json", registry=registry, feature_ids=[2723])


@pytest.mark.parametrize("target", ["scores", "sample", "reference", "model"])
def test_artifact_tampering_rejected(tmp_path, target):
    root, source, registry = workflow_fixture(tmp_path)
    fit = tmp_path / "fit"
    calibration_workflow(source, root, fit)
    paths = {"scores": source, "sample": root / "sample_manifest.json",
             "reference": root / "scores/scores.json", "model": fit / "model.json"}
    payload = json.loads(paths[target].read_text())
    payload["tampered"] = True
    write_json(paths[target], payload)
    with pytest.raises(ValueError):
        predict_workflow(fit / "model.json", source, root, tmp_path / "prediction.json", registry=registry)


def test_response_descriptors_are_hash_bound_and_outcomes_forbidden(tmp_path):
    from event_sae.research.head_predict.descriptor_workflow import descriptor_implementation_identity
    from test_head_predict_descriptors import semantic_payload_fixture
    payload, source, reference, bindings = semantic_payload_fixture()
    payload["implementation"] = descriptor_implementation_identity()
    payload["descriptor_hash"] = fingerprint({k: v for k, v in payload.items() if k != "descriptor_hash"})
    kwargs = {"source_scores": source, "reference_scores": reference}
    path = tmp_path / "response.json"
    write_json(path, payload)
    assert _load_descriptors(path, bindings, **kwargs)[1]["descriptor_hash"] == payload["descriptor_hash"]
    with pytest.raises(ValueError, match="requires complete"):
        _load_descriptors(path, bindings, source_scores=None, reference_scores=None)
    payload["rows"][0]["target_drop"] = .5
    write_json(path, payload)
    with pytest.raises(ValueError, match="descriptor_hash"): _load_descriptors(path, bindings, **kwargs)
    payload["descriptor_hash"] = fingerprint({k: v for k, v in payload.items() if k != "descriptor_hash"})
    write_json(path, payload)
    with pytest.raises(ValueError, match="outcomes"): _load_descriptors(path, bindings, **kwargs)
