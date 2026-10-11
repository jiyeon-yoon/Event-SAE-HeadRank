"""Real contract/journal integration with fake CPU policy, never GPU execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from event_sae.research.head_rank import native, runtime
from event_sae.research.head_rank.analysis import analyze_experiment
from event_sae.research.head_rank.protocol import plan_panel
from event_sae.research.output_head import workflow as old_workflow
from event_sae.research.output_head.config import DEFAULTS
from event_sae.research.output_head.provenance import fingerprint, read_json, sha256_file
from event_sae.research.output_head.splits import build_followup_split


CODE = {"commit": "a" * 40, "dirty": False}
IMPLEMENTATION = {"files": {"fake_cpu_fixture.py": "b" * 64}}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def seal(doc, field):
    doc.pop(field, None)
    doc[field] = fingerprint(doc)
    return doc


def fixture_files(tmp_path, monkeypatch):
    """Small real JSON/hash files; model, parity and Git calls are mocked."""
    tmp_path = tmp_path.resolve()
    inputs, reference = tmp_path / "inputs", tmp_path / "reference"
    source_episodes = []
    for task in (0, 1):
        for trial in range(6):
            digest = hashlib.sha256(f"state-{task}-{trial}".encode()).hexdigest()
            source_episodes.append({"suite": "libero_spatial", "task_id": task,
                "task_episode_idx": trial, "episode_uid": f"episode-{task}-{trial}",
                "initial_state_sha256": digest, "current_runtime_state_sha256": digest,
                "historical_initial_state_sha256": None,
                "initial_state_hash_provenance": "synthetic_cpu_test_registry"})
    split = build_followup_split({"episodes": source_episodes,
        "historical_state_identity_verified": False}, {"discovery_per_task": 2,
        "validation_per_task": 1, "evaluation_per_task": 3, "split_seed": 42,
        "evaluation_labels_previously_used": None})
    split_path = write(inputs / "split.json", split)
    evaluation_path = inputs / "evaluation.json"
    runtime.select_cases(split_path, evaluation_path, cases_per_task=1)
    evaluation = read_json(evaluation_path)
    sample = {"discovery_episodes": split["splits"]["discovery"],
              "split_manifest_hash": split["manifest_hash"],
              "discovery_manifest_hash": "d" * 64, "sample_hash": "e" * 64}
    sample_path = write(reference / "sample_manifest.json", sample)
    old_scores = write(reference / "scores/scores.json", {"fixture_only": True})
    cache_manifest = write(reference / "readouts/manifest.json", {"fixture_only": True})
    weights = write(inputs / "sae/ae.pt", {"not_real_weights": True})
    sae_config = write(inputs / "sae/config.json", {"fixture_only": True})
    head = inputs / "head"
    write(head / "output_head.safetensors", {"not_real_weights": True})
    write(head / "output_head_manifest.json", {"fixture_only": True})
    snapshot = inputs / "snapshot"
    write(snapshot / "model-1.safetensors", {"not_real_weights": True})
    write(snapshot / "config.json", {"fixture_only": True})
    parity_root = tmp_path / "parity"
    write(parity_root / "validation/runtime_parity.json", {"fake_parity_fixture": True})
    write(parity_root / "validation/edit_parity.json", {"fake_parity_fixture": True})
    cfg = deepcopy(DEFAULTS)
    cfg["inputs"].update(local_model_snapshot=str(snapshot), sae_checkpoint=str(weights),
                         generation_manifest=str(write(inputs / "generation.json", {"fixture_only": True})))
    cfg["output"]["root_dir"] = str(parity_root)
    cfg["sampling"].update(mode="followup", task_ids=[0, 1], split_manifest=str(split_path))
    cfg["rollout"].update(env_seed=42, base_eval_config=str(write(inputs / "base.json", {"fixture_only": True})),
                           max_total_rollouts=100)
    config_path = inputs / "head.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    identity = {"sae_checkpoint_hash": sha256_file(weights), "device": "cpu_test_no_model"}
    policy = {"fixture_policy": "no_model_no_simulator"}
    monkeypatch.setattr(runtime, "git_identity", lambda _: deepcopy(CODE))
    monkeypatch.setattr(runtime, "implementation_identity", lambda: deepcopy(IMPLEMENTATION))
    monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: deepcopy(identity))
    parity_calls = []
    monkeypatch.setattr(old_workflow, "load_parity", lambda cfg, identity: parity_calls.append((cfg, identity)))
    monkeypatch.setattr(old_workflow, "_sae_paths", lambda _: (weights, sae_config))
    monkeypatch.setattr(old_workflow, "_head_path", lambda _: head)
    monkeypatch.setattr(native, "effective_policy", lambda _: deepcopy(policy))
    scope = {"dictionary_size": 32768, "task_ids": [0, 1], "sae_sha256": sha256_file(weights),
             "split_manifest_hash": split["manifest_hash"],
             "discovery_manifest_hash": sample["discovery_manifest_hash"], "sample_manifest_hash": sample["sample_hash"]}
    queries = [{"query_index": index, "task_id": row["task_id"], "episode_uid": row["episode_uid"],
                "step_in_episode": 0, "num_action_positions": 7}
               for index, row in enumerate(sample["discovery_episodes"])]
    score_rows = [{"feature_id": fid, "mean_head_kl": float(fid), "tail_head_kl": float(30 - fid),
                   "direction_excess_kl": float(fid % 7), "event_max": float((fid * 3) % 17),
                   "num_readouts": 7 * len(queries), "num_queries": len(queries)} for fid in range(20)]
    scores = {"schema_version": "head_rank_scores_v1", "feature_scores": score_rows,
        "feature_ids": list(range(20)), "scored_feature_ids": list(range(20)),
        "frozen_active_feature_ids": list(range(20)), "synthetic": False, "status": "scored",
        "sampling_mode": "full", "development_only": False, "feature_set": "all-active",
        "scope": scope, "implementation": IMPLEMENTATION, "identity": identity,
        "event_max_binding": {"source_sha256": "f" * 64}, "query_metadata": queries,
        "scoring_sample": {"readouts_used": 7 * len(queries), "discovery_episodes_used": len(queries),
                           "event_max_matches_scoring_episode_population": True},
        "source_sample_sha256": sha256_file(sample_path), "source_scores_sha256": sha256_file(old_scores),
        "source_cache_manifest_sha256": sha256_file(cache_manifest), "source_readonly_roots": [str(reference)]}
    scores_path = write(inputs / "scores.json", seal(scores, "score_hash"))
    registry_path = write(inputs / "registry.json", {"reviewed_for_team_prior_outcomes": True, "feature_ids": []})
    bindings = {"scores_sha256": sha256_file(scores_path), "score_hash": scores["score_hash"],
        "registry_sha256": sha256_file(registry_path), "eval_manifest_sha256": sha256_file(evaluation_path),
        "implementation": IMPLEMENTATION, "event_max_binding": scores["event_max_binding"],
        "score_scope": scope, "evaluation_label_history": evaluation["evaluation_labels_previously_used"]}
    plan = plan_panel(scores, feature_count=4, task_ids=[0, 1], cases_per_task=1,
                       reference_bindings=bindings, evaluation_cases=evaluation["cases"])
    plan_path = write(inputs / "plan.json", plan)
    kwargs = {"config": config_path, "plan_path": plan_path, "scores_path": scores_path,
        "registry_path": registry_path, "eval_manifest": evaluation_path, "reference_result": reference,
        "split_path": split_path, "output": tmp_path / "execution.json", "run_output": tmp_path / "runs"}
    return {"kwargs": kwargs, "cfg": cfg, "identity": identity, "policy": policy, "plan": plan,
            "scores": scores, "evaluation": evaluation, "split": split, "sample": sample,
            "weights": weights, "snapshot": snapshot, "parity_calls": parity_calls}


def prepared(tmp_path, monkeypatch):
    fixture = fixture_files(tmp_path, monkeypatch)
    result = runtime.prepare_execution(**fixture["kwargs"])
    return fixture, result, read_json(fixture["kwargs"]["output"])


class CPUFakeBackend:
    instances = []

    def __init__(self, cfg, identity):
        self.calls, self.closed = [], False
        self.validated = None
        self.instances.append(self)

    def validate_cases(self, cases):
        self.validated = deepcopy(cases)

    def run_case(self, condition, case, directory):
        assert case in self.validated
        assert (directory / "started.json").is_file()
        self.calls.append((condition, case))
        raw_success = case["task_id"] == 0
        success = raw_success if condition["mode"] != "intervention" else condition["feature_id"] % 2 == 0
        episode = {**case, "success": success, "caught_exception": None, "num_actions": 1}
        metrics, evidence = None, None
        if condition["mode"] != "raw":
            metrics = {"feature_id": condition["feature_id"], "alpha": condition["alpha"],
                "num_forwards": 7, "num_environment_steps": 1, "active_feature_values": 0,
                "post_intervention_active_feature_values": 0, "max_feature_activation": 0.,
                "post_intervention_max_feature_activation": 0., "absolute_feature_delta": 0.}
            path = write(directory / "native/hook.jsonl", {"feature_id": condition["feature_id"],
                "alpha": condition["alpha"], "task_id": case["task_id"], "task_episode_idx": case["task_episode_idx"],
                "step_in_episode": 0, "first_forward_idx": 1})
            evidence = {"path": str(path), "sha256": sha256_file(path)}
        return {"episode": episode, "actions": [[float(case["task_id"])] * 7],
                "hook_metrics": metrics, "hook_evidence": evidence}

    def close(self):
        self.closed = True


def test_prepare_is_model_free_and_freezes_all_sources(tmp_path, monkeypatch):
    fixture, result, experiment = prepared(tmp_path, monkeypatch)
    assert result["status"] == "prepared_not_executed"
    assert result["new_rollouts"] == 0
    assert experiment["total_rollouts"] == 12
    assert experiment["max_attempts"] == 12
    assert len(fixture["parity_calls"]) == 1
    assert experiment["split_evidence"]["historical_state_identity_verified"] is False
    assert not fixture["kwargs"]["run_output"].exists()
    assert {"sae", "head_weights", "snapshot/model-1.safetensors", "prediction_plan", "scores", "split"} <= set(experiment["sources"])
    runtime.verify_experiment(experiment)


def legacy_v2_fixture(tmp_path, monkeypatch):
    from event_sae.research.head_rank import workflow
    from event_sae.research.head_rank.compatibility import LEGACY_SCORER_IMPLEMENTATION
    fixture = fixture_files(tmp_path, monkeypatch)
    kwargs = fixture["kwargs"]
    current = workflow.implementation_identity()
    monkeypatch.setattr(runtime, "implementation_identity", lambda: deepcopy(current))
    # Only this synthetic test score fixture is resealed; the real legacy input
    # remains read-only and is never rewritten by the workflow.
    fixture["scores"]["implementation"] = deepcopy(LEGACY_SCORER_IMPLEMENTATION)
    write(kwargs["scores_path"], seal(fixture["scores"], "score_hash"))
    before = sha256_file(kwargs["scores_path"])
    kwargs["plan_path"] = tmp_path / "legacy-v2-plan.json"
    workflow.plan_workflow(kwargs["scores_path"], kwargs["plan_path"],
        registry=kwargs["registry_path"], eval_manifest=kwargs["eval_manifest"],
        feature_count=16, cases_per_task=1, selection_strategy="topk_union")
    assert sha256_file(kwargs["scores_path"]) == before
    fixture["plan"] = read_json(kwargs["plan_path"])
    return fixture


def test_pinned_legacy_v2_prepare_and_cpu_fake_runtime(tmp_path, monkeypatch):
    fixture = legacy_v2_fixture(tmp_path, monkeypatch)
    result = runtime.prepare_execution(**fixture["kwargs"])
    assert result["total_rollouts"] == 36  # 16 + raw + identity, 2 fake cases.
    exp = read_json(fixture["kwargs"]["output"])
    bindings = exp["prediction_plan"]["source_bindings"]
    assert bindings["scoring_implementation"] != exp["implementation"]
    assert bindings["scorer_compatibility"]["mode"] == "pinned_legacy_v1"
    assert len(fixture["parity_calls"]) == 1
    monkeypatch.setattr(native, "NativeBackend", CPUFakeBackend)
    CPUFakeBackend.instances.clear()
    result = runtime.run_execution(fixture["kwargs"]["output"],
        approved_hash=exp["execution_hash"], expected_revision=CODE["commit"])
    assert result["status"] == "completed" and result["new_attempts"] == 36
    assert len(CPUFakeBackend.instances[-1].calls) == 36
    assert len(fixture["parity_calls"]) == 2


@pytest.mark.parametrize("change", ["dirty", "identity", "parity", "registry", "score", "binding"])
def test_legacy_exception_never_bypasses_existing_runtime_gates(tmp_path, monkeypatch, change):
    fixture = legacy_v2_fixture(tmp_path, monkeypatch)
    kwargs = fixture["kwargs"]
    if change == "dirty":
        monkeypatch.setattr(runtime, "git_identity", lambda _: {**CODE, "dirty": True})
    elif change == "identity":
        monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: {"other_device": True})
    elif change == "parity":
        def stale_parity(*args):
            raise ValueError("Current numerical parity must be freshly validated")
        monkeypatch.setattr(old_workflow, "load_parity", stale_parity)
    elif change == "registry":
        write(kwargs["registry_path"], {"reviewed_for_team_prior_outcomes": False, "feature_ids": []})
    elif change == "score":
        modified = deepcopy(fixture["scores"])
        modified["feature_scores"][0]["tail_head_kl"] += 1
        write(kwargs["scores_path"], seal(modified, "score_hash"))
    else:
        modified = deepcopy(fixture["plan"])
        modified["source_bindings"]["scoring_implementation"] = {"unsupported": True}
        write(kwargs["plan_path"], seal(modified, "plan_hash"))
    with pytest.raises(ValueError):
        runtime.prepare_execution(**kwargs)
    assert not kwargs["output"].exists() and not kwargs["run_output"].exists()


def test_full10_preset_never_shrinks_to_small_fixture_scope(tmp_path, monkeypatch):
    from event_sae.research.head_rank.workflow import plan_workflow
    fixture = legacy_v2_fixture(tmp_path, monkeypatch)
    kwargs = fixture["kwargs"]
    target = tmp_path / "invalid-full10-plan.json"
    with pytest.raises(ValueError, match="Full10 requires"):
        plan_workflow(kwargs["scores_path"], target, registry=kwargs["registry_path"],
                      feature_count=16, cases_per_task=20, selection_strategy="topk_union",
                      require_full10=True)
    assert not target.exists()


def test_legacy_plan_workflow_keeps_unreviewed_registry_blocked(tmp_path, monkeypatch):
    from event_sae.research.head_rank.workflow import plan_workflow
    fixture = legacy_v2_fixture(tmp_path, monkeypatch)
    kwargs = fixture["kwargs"]
    write(kwargs["registry_path"], {"reviewed_for_team_prior_outcomes": False, "feature_ids": []})
    target = tmp_path / "unreviewed-plan.json"
    with pytest.raises(ValueError, match="with your team"):
        plan_workflow(kwargs["scores_path"], target, registry=kwargs["registry_path"],
                      feature_count=16, cases_per_task=1, selection_strategy="topk_union")
    assert not target.exists()


def test_prepare_execute_resume_analyze_real_journal_with_cpu_backend(tmp_path, monkeypatch):
    fixture, result, experiment = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(native, "NativeBackend", CPUFakeBackend)
    CPUFakeBackend.instances.clear()
    run = runtime.run_execution(fixture["kwargs"]["output"], approved_hash=experiment["execution_hash"], expected_revision=CODE["commit"])
    assert run["status"] == "completed" and run["new_attempts"] == 12
    assert CPUFakeBackend.instances[-1].closed
    assert len(CPUFakeBackend.instances[-1].calls) == 12
    before = {path: sha256_file(path) for path in fixture["kwargs"]["run_output"].rglob("result.json")}
    resumed = runtime.run_execution(fixture["kwargs"]["output"], approved_hash=experiment["execution_hash"], expected_revision=CODE["commit"])
    assert resumed["new_rollouts"] == 0
    assert len(CPUFakeBackend.instances) == 1
    assert all(sha256_file(path) == digest for path, digest in before.items())
    output = tmp_path / "analysis"
    analyzed = analyze_experiment(fixture["kwargs"]["output"], fixture["kwargs"]["run_output"], output)
    assert analyzed["status"] == "analyzed" and analyzed["new_rollouts"] == 0
    data = read_json(output / "analysis.json")
    assert data["identity_validation"]["status"] == "passed"
    assert data["num_cases_per_feature"] == 2
    assert len(data["feature_effects"]) == 4
    assert "analysis_implementation" in data
    assert (output / "report.md").is_file()
    # Provenance paths may be unavailable after archival; analysis uses frozen
    # contracts and receipts, never attempts to load the original model.
    fixture["weights"].unlink()
    second = analyze_experiment(fixture["kwargs"]["output"], fixture["kwargs"]["run_output"], tmp_path / "archived-analysis")
    assert second["status"] == "analyzed"


@pytest.mark.parametrize("change", ["alpha", "seed", "panel", "budget", "cfg", "code_dirty", "analysis", "overlap", "implementation"])
def test_rehashed_semantic_contract_tampering_rejected(tmp_path, monkeypatch, change):
    _, _, exp = prepared(tmp_path, monkeypatch)
    if change == "alpha":
        exp["conditions"][-1]["alpha"] = .5
    elif change == "seed":
        exp["cases"][0]["seed"] += 1
    elif change == "panel":
        exp["conditions"][-1]["feature_id"] = 999
    elif change == "budget":
        exp["max_attempts"] += 1
    elif change == "cfg":
        exp["config"]["model"]["allow_download"] = True
        exp["config_hash"] = fingerprint(exp["config"])
    elif change == "code_dirty":
        exp["code_identity"]["dirty"] = True
    elif change == "analysis":
        exp["analysis"]["bootstrap_replicates"] = 2
    elif change == "overlap":
        exp["split_evidence"]["selection_eval_overlap"] = True
    elif change == "implementation":
        exp["implementation"] = {"different": True}
    seal(exp, "execution_hash")
    with pytest.raises(ValueError):
        runtime.verify_experiment(exp)


@pytest.mark.parametrize("approval,revision", [("", CODE["commit"]), ("b" * 64, CODE["commit"]), (None, "b" * 40)])
def test_run_explicit_approval_and_revision_gate_before_backend(tmp_path, monkeypatch, approval, revision):
    fixture, _, exp = prepared(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU backend must not be constructed")
    monkeypatch.setattr(native, "NativeBackend", forbidden)
    actual = exp["execution_hash"] if approval is None else approval
    with pytest.raises(ValueError):
        runtime.run_execution(fixture["kwargs"]["output"], approved_hash=actual, expected_revision=revision)
    assert not fixture["kwargs"]["run_output"].exists()


@pytest.mark.parametrize("change", ["dirty", "revision", "implementation", "source", "identity", "policy"])
def test_run_current_provenance_changes_block_before_backend(tmp_path, monkeypatch, change):
    fixture, _, exp = prepared(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU backend must not be constructed")
    monkeypatch.setattr(native, "NativeBackend", forbidden)
    if change == "dirty":
        monkeypatch.setattr(runtime, "git_identity", lambda _: {**CODE, "dirty": True})
    elif change == "revision":
        monkeypatch.setattr(runtime, "git_identity", lambda _: {**CODE, "commit": "b" * 40})
    elif change == "implementation":
        monkeypatch.setattr(runtime, "implementation_identity", lambda: {"different": True})
    elif change == "source":
        write(fixture["weights"], {"modified": True})
    elif change == "identity":
        monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: {**fixture["identity"], "device": "different"})
    elif change == "policy":
        monkeypatch.setattr(native, "effective_policy", lambda _: {"different": True})
    with pytest.raises(ValueError):
        runtime.run_execution(fixture["kwargs"]["output"], approved_hash=exp["execution_hash"], expected_revision=CODE["commit"])
    assert not fixture["kwargs"]["run_output"].exists()


def test_dirty_preparation_has_no_outputs_or_model_calls(tmp_path, monkeypatch):
    fixture = fixture_files(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "git_identity", lambda _: {**CODE, "dirty": True})
    with pytest.raises(ValueError, match="clean commit"):
        runtime.prepare_execution(**fixture["kwargs"])
    assert not fixture["kwargs"]["output"].exists()
    assert not fixture["parity_calls"]


@pytest.mark.parametrize("change", ["registry", "scores", "reference", "eval_history", "runtime_identity"])
def test_preparation_provenance_mismatch_rejected(tmp_path, monkeypatch, change):
    fixture = fixture_files(tmp_path, monkeypatch)
    kwargs = fixture["kwargs"]
    if change == "registry":
        write(kwargs["registry_path"], {"reviewed_for_team_prior_outcomes": False, "feature_ids": []})
    elif change == "scores":
        altered = deepcopy(fixture["scores"])
        altered["feature_scores"][0]["tail_head_kl"] += .1
        write(kwargs["scores_path"], altered)
    elif change == "reference":
        write(kwargs["reference_result"] / "sample_manifest.json", {"changed": True})
    elif change == "eval_history":
        altered = deepcopy(fixture["evaluation"])
        altered["evaluation_labels_previously_used"] = False
        write(kwargs["eval_manifest"], altered)
    elif change == "runtime_identity":
        monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: {"different": True})
    with pytest.raises(ValueError):
        runtime.prepare_execution(**kwargs)
    assert not kwargs["output"].exists()


def test_select_cases_deterministic_heldout_only_and_no_overwrite(tmp_path, monkeypatch):
    fixture = fixture_files(tmp_path, monkeypatch)
    split_path = fixture["kwargs"]["split_path"]
    path = tmp_path / "same-cases.json"
    runtime.select_cases(split_path, path, cases_per_task=1)
    assert read_json(path) == fixture["evaluation"]
    allowed = {(r["task_id"], r["task_episode_idx"]) for r in fixture["split"]["splits"]["evaluation"]}
    assert all((r["task_id"], r["task_episode_idx"]) in allowed for r in read_json(path)["cases"])
    with pytest.raises(FileExistsError):
        runtime.select_cases(split_path, path, cases_per_task=1)
    with pytest.raises(ValueError, match="exceeds"):
        runtime.select_cases(split_path, tmp_path / "too-many.json", cases_per_task=4)


def test_split_selection_rejects_discovery_member_and_hash_mismatch(tmp_path, monkeypatch):
    fixture = fixture_files(tmp_path, monkeypatch)
    cfg, split, sample, scores = fixture["cfg"], fixture["split"], fixture["sample"], fixture["scores"]
    plan, evaluation = deepcopy(fixture["plan"]), deepcopy(fixture["evaluation"])
    plan["evaluation_cases"][0] = deepcopy(split["splits"]["discovery"][0])
    with pytest.raises(ValueError, match="forbidden split"):
        runtime._check_split(cfg, split, sample, scores, plan, evaluation)
    scores = deepcopy(scores)
    scores["scope"]["sample_manifest_hash"] = "f" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        runtime._check_split(cfg, split, sample, scores, fixture["plan"], evaluation)


def test_retry_cap_and_original_source_output_protection(tmp_path, monkeypatch):
    fixture = fixture_files(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="rollout cap"):
        runtime.prepare_execution(**fixture["kwargs"], retry_budget=1, max_rollouts=12)
    kwargs = dict(fixture["kwargs"], run_output=fixture["snapshot"] / "must-not-write")
    with pytest.raises(ValueError, match="overlap"):
        runtime.prepare_execution(**kwargs)
    assert not kwargs["run_output"].exists()


def test_contract_cannot_be_silently_overwritten(tmp_path, monkeypatch):
    fixture, _, _ = prepared(tmp_path, monkeypatch)
    original = sha256_file(fixture["kwargs"]["output"])
    with pytest.raises(FileExistsError):
        runtime.prepare_execution(**fixture["kwargs"])
    assert sha256_file(fixture["kwargs"]["output"]) == original


def test_analysis_output_cannot_pollute_frozen_snapshot(tmp_path, monkeypatch):
    fixture, _, _ = prepared(tmp_path, monkeypatch)
    target = fixture["snapshot"] / "analysis-must-not-write"
    with pytest.raises(ValueError, match="overlap"):
        analyze_experiment(fixture["kwargs"]["output"], fixture["kwargs"]["run_output"], target)
    assert not target.exists()


def test_rehashed_partial_score_population_cannot_prepare_execution(tmp_path, monkeypatch):
    fixture = fixture_files(tmp_path, monkeypatch)
    scores = deepcopy(fixture["scores"])
    scores["feature_scores"].pop()
    write(fixture["kwargs"]["scores_path"], seal(scores, "score_hash"))
    with pytest.raises(ValueError, match="population"):
        runtime.prepare_execution(**fixture["kwargs"])
    assert not fixture["kwargs"]["output"].exists()
