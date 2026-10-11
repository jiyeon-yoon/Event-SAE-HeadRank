"""Auxiliary prediction freezing uses synthetic CPU contracts, never real outcomes."""
from copy import deepcopy
import json

import pytest

from event_sae.research.head_predict.evaluation import (
    attach_predictions, evaluate_prediction_attachment, validate_prediction_attachment,
)
from event_sae.research.head_rank.protocol import (
    KNOWN_EVALUATED_IDS, fingerprint as plan_fingerprint, plan_panel, verify_plan,
)
from event_sae.research.head_rank.workflow import implementation_identity
from event_sae.research.head_predict.calibration_workflow import implementation_identity as producer_identity
from event_sae.research.output_head.provenance import fingerprint, sha256_file


def seal(doc, key):
    doc[key] = fingerprint({k: v for k, v in doc.items() if k != key})
    return doc


def fixture(plan=None):
    if plan is None:
        rows = [{"feature_id": fid, "mean_head_kl": float(fid), "tail_head_kl": float(15 - fid),
                 "event_max": float(fid % 3), "direction_excess_kl": float(fid % 5)} for fid in range(16)]
        cases = [{"suite": "libero_spatial", "task_id": t, "task_episode_idx": 0,
                  "initial_state_sha256": str(t + 1) * 64} for t in (0, 1)]
        plan = plan_panel({"feature_scores": rows, "synthetic": False}, feature_count=4, top_k=1,
            task_ids=[0, 1], cases_per_task=1, selection_strategy="topk_union", evaluation_cases=cases,
            reference_bindings={"implementation": implementation_identity(), "score_hash": "a" * 64,
                "scores_sha256": "b" * 64, "registry_sha256": "c" * 64, "score_scope": {"task_ids": [0, 1]}})
    bindings = plan["source_bindings"]
    source = {"source_score_hash": bindings["score_hash"], "source_scores_sha256": bindings["scores_sha256"],
              "scope": bindings["score_scope"], "implementation": producer_identity()}
    panel = sorted(plan["panel_feature_ids"])
    training = [7024, 24830]
    model = seal({"schema_version": "head_predict_calibrated_model_v1", "training_feature_ids": training,
        "task_ids": plan["task_ids"], "requires_disjoint_inference_feature_ids": True,
        "source_bindings": source, "models": {"head_ridge": {"kind": "fixture_no_real_model"}}}, "model_hash")
    task_rows = [{"feature_id": fid, "task_id": t, "predicted_drop": i * .1, "unclipped_predicted_drop": i * .1}
                 for i, fid in enumerate(panel) for t in plan["task_ids"]]
    artifact = seal({"schema_version": "head_predict_disjoint_predictions_v1", "predictor": "head_ridge",
        "source_bindings": source, "registry_sha256": bindings["registry_sha256"],
        "model_hash": model["model_hash"], "model_sha256": "d" * 64, "training_feature_ids": training,
        "excluded_feature_ids": sorted(KNOWN_EVALUATED_IDS | set(plan.get("additional_evaluated_ids", []))),
        "eligible_feature_ids": panel, "selection_scope": "explicit_feature_subset", "predictions": task_rows,
        "feature_predictions": [{"feature_id": fid, "predicted_drop": i * .1} for i, fid in enumerate(panel)],
        "outcome_labels_read": False, "no_new_feature_outcomes_used": True, "new_rollouts": 0,
        "rollouts_authorized": False, "implementation": producer_identity()}, "prediction_hash")
    return plan, artifact, model


def embedded(plan, artifact, model):
    result = deepcopy(plan)
    result["source_bindings"]["calibrated_predictions"] = {
        "schema_version": "head_predict_plan_attachment_v1", "role": "secondary_fixed_panel_prediction",
        "panel_feature_ids": sorted(plan["panel_feature_ids"]), "prediction_file_sha256": "e" * 64,
        "model_file_sha256": artifact["model_sha256"], "model": model, "predictions": artifact}
    result["plan_hash"] = plan_fingerprint({k: v for k, v in result.items() if k != "plan_hash"})
    return result


def test_no_attachment_preserves_legacy_analysis():
    plan, _, _ = fixture()
    assert validate_prediction_attachment(plan) is None
    assert evaluate_prediction_attachment(plan, []) is None


def test_embedded_predictions_change_hash_not_selection_budget_or_primary():
    original, artifact, model = fixture()
    plan = embedded(original, artifact, model)
    verify_plan(plan)
    assert validate_prediction_attachment(plan)["predictor"] == "head_ridge"
    for name in ("panel_feature_ids", "budget", "primary_metric", "selection", "evaluation_cases"):
        assert plan[name] == original[name]
    assert plan["plan_hash"] != original["plan_hash"]
    effects = [{"feature_id": r["feature_id"], "drop": r["predicted_drop"] - .05} for r in artifact["feature_predictions"]]
    result = evaluate_prediction_attachment(plan, effects)
    assert result["mae"] == pytest.approx(.05)
    assert result["spearman"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", ["training", "registry", "scope", "score_hash", "aggregate", "duplicate", "nan",
                                  "model_training", "model_hash", "population", "row_type",
                                  "producer_empty", "producer_mismatch"])
def test_rehashed_semantic_mutations_are_rejected(bad):
    plan, artifact, model = fixture()
    if bad == "training":
        artifact["training_feature_ids"] = [plan["panel_feature_ids"][0]]
    elif bad == "registry": artifact["registry_sha256"] = "f" * 64
    elif bad == "scope": artifact["source_bindings"] = {**artifact["source_bindings"], "scope": {"task_ids": [9]}}
    elif bad == "score_hash": artifact["source_bindings"] = {**artifact["source_bindings"], "source_score_hash": "f" * 64}
    elif bad == "aggregate": artifact["feature_predictions"][0]["predicted_drop"] += .1
    elif bad == "duplicate": artifact["predictions"][1] = artifact["predictions"][0]
    elif bad == "nan": artifact["predictions"][0]["predicted_drop"] = 1.1
    elif bad == "model_training": model["training_feature_ids"] = [5]
    elif bad == "model_hash": artifact["model_hash"] = "f" * 64
    elif bad == "population": artifact["selection_scope"] = "all_eligible_disjoint_features"
    elif bad == "row_type": artifact["feature_predictions"][0] = None
    elif bad == "producer_empty": artifact["implementation"] = {}
    elif bad == "producer_mismatch": artifact["implementation"] = {"files": {"changed.py": "a" * 64}}
    seal(model, "model_hash")
    seal(artifact, "prediction_hash")
    with pytest.raises(ValueError): validate_prediction_attachment(embedded(plan, artifact, model))


def test_attach_does_not_overwrite_and_checks_actual_model_file(tmp_path):
    plan, artifact, model = fixture()
    p, a, m = (tmp_path / name for name in ("plan.json", "predictions.json", "model.json"))
    m.write_text(json.dumps(model))
    artifact["model_sha256"] = sha256_file(m)
    seal(artifact, "prediction_hash")
    p.write_text(json.dumps(plan))
    a.write_text(json.dumps(artifact))
    output = tmp_path / "attached.json"
    result = attach_predictions(p, a, m, output)
    assert result["new_rollouts"] == 0 and result["total_rollouts"] == plan["budget"]["total_rollouts"]
    assert json.loads(p.read_text()) == plan
    with pytest.raises(FileExistsError): attach_predictions(p, a, m, output)
    m.write_text(json.dumps({**model, "changed": True}))
    with pytest.raises(ValueError, match="model file"): attach_predictions(p, a, m, tmp_path / "other.json")


def test_attach_rejects_changed_calibrator_before_writing(tmp_path, monkeypatch):
    from event_sae.research.head_predict import calibration_workflow
    plan, artifact, model = fixture()
    p, a, m = (tmp_path / name for name in ("plan.json", "predictions.json", "model.json"))
    m.write_text(json.dumps(model))
    artifact["model_sha256"] = sha256_file(m)
    seal(artifact, "prediction_hash")
    p.write_text(json.dumps(plan))
    a.write_text(json.dumps(artifact))
    monkeypatch.setattr(calibration_workflow, "implementation_identity", lambda: {"files": {"changed.py": "a" * 64}})
    with pytest.raises(ValueError, match="producer implementation changed"):
        attach_predictions(p, a, m, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_calibrated_auxiliary_full_fake_runtime_and_analysis(tmp_path, monkeypatch):
    from test_head_rank_runtime import fixture_files, CPUFakeBackend, CODE
    from event_sae.research.head_rank import native, runtime
    from event_sae.research.head_rank.analysis import analyze_experiment

    files = fixture_files(tmp_path, monkeypatch)
    plan, artifact, model = fixture(files["plan"])
    plan = embedded(plan, artifact, model)
    files["kwargs"]["plan_path"].write_text(json.dumps(plan))
    prepared = runtime.prepare_execution(**files["kwargs"])
    monkeypatch.setattr(native, "NativeBackend", CPUFakeBackend)
    result = runtime.run_execution(files["kwargs"]["output"], approved_hash=prepared["execution_hash"],
                                   expected_revision=CODE["commit"])
    assert result["status"] == "completed"
    target = tmp_path / "analysis"
    analyze_experiment(files["kwargs"]["output"], files["kwargs"]["run_output"], target)
    report = json.loads((target / "analysis.json").read_text())
    assert report["calibrated_prediction_comparison"]["model_hash"] == model["model_hash"]
    assert "Frozen calibrated predictor" in (target / "report.md").read_text()
