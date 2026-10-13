"""CPU-only contract/authorization tests; fake content is never a GPU proof."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch  # Keep native OpenMP initialization consistent with native tests.

from event_sae.research.head_matched import runtime, workflow
from event_sae.research.head_matched.protocol import ANALYSIS, NORM_POLICY, SCOPE, conditions, direction_scores
from event_sae.research.head_rank.native import episode_seed
from event_sae.research.output_head.config import DEFAULTS
from event_sae.research.output_head.provenance import fingerprint, sha256_file
from test_head_matched_journal import hook_payload


def seal(document, field):
    document[field] = fingerprint({key: value for key, value in document.items() if key != field})
    return document


@pytest.fixture
def contract_fixture(tmp_path):
    def make_file(relative):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic unit-test bytes: {relative}")
        return {"path": str(path), "sha256": sha256_file(path)}
    source_paths = {
        "config": "config.yaml", "sae": "sae/ae.pt", "sae_config": "sae/config.json",
        "head_weights": "head/output_head.safetensors", "head_manifest": "head/output_head_manifest.json",
        "base_eval_config": "base.yaml", "generation_manifest": "generation.json",
        "head_parity": "runtime/validation/runtime_parity.json", "edit_parity": "runtime/validation/edit_parity.json",
        "runtime_inputs": "calls.pt",
        **{f"snapshot/model-{n:05d}-of-00004.safetensors": f"snapshot/model-{n:05d}-of-00004.safetensors" for n in range(1, 5)},
    }
    sources = {name: make_file(path) for name, path in source_paths.items()}
    identity = {"sae_checkpoint_hash": sources["sae"]["sha256"], "device": "cpu", "runtime": "fake"}
    impl, code = {"files": {"fixture.py": "a" * 64}}, {"commit": "b" * 40, "dirty": False}
    conds = conditions(9, [12, 13, 14, 15])
    cases = [{"suite": "libero_spatial", "task_id": t, "task_episode_idx": 1,
              "initial_state_sha256": f"{t + 1:064x}", "seed": episode_seed(0, t, 1)} for t in range(10)]
    metadata = [{"query_index": t, "task_id": t} for t in range(10)]
    raw = {"schema_version": "head_rank_scores_v1", "synthetic": False, "feature_ids": [9],
        "control_count": 4, "control_feature_ids": {"9": [12, 13, 14, 15]}, "norm_match": NORM_POLICY,
        "diagnostics": {"num_norm_invalid_readouts": 0, "num_norm_invalid_controls": 0},
        "feature_scores": [{"feature_id": 9, "mean_head_kl": 1., "control_feature_ids": [12, 13, 14, 15],
            "direction_excess_status": "ok", "norm_invalid_readouts": 0, "norm_invalid_controls": 0, "num_readouts": 70}],
        "query_metadata": metadata, "feature_query_scores": {"9": [
            {"query_index": t, "mean_head_kl": 1., "control_query_kls": [.5, .8, 1.1, 1.5]} for t in range(10)]}}
    seal(raw, "score_hash")
    predictions = direction_scores(raw, 9)
    parent_keys = [name for name in sources if name.startswith("snapshot/")] + [
        "sae", "sae_config", "head_weights", "head_manifest", "base_eval_config", "generation_manifest"]
    plan = {"schema_version": "head_matched_plan_v1", "scientific_scope": SCOPE,
        "synthetic": False, "automatic_reuse_of_old_outcomes": False, "analysis": ANALYSIS,
        "anchor_feature_id": 9, "control_feature_ids": [12, 13, 14, 15], "conditions": conds,
        "cases_per_task": 1, "base_seed": 0, "cases": cases, "total_rollouts": 70,
        "source_readouts": 70, "source_query_metadata_hash": fingerprint(metadata),
        "task_ids": list(range(10)), "norm_scope": "per_row_same_current_hidden_not_equal_trajectory_budget",
        "prediction_scope": "fixed_prefix_full_vocabulary_equal_task_mean_not_physical_action_error",
        "final_predictions": "require_current_device_scoring_before_any_behavior_outcomes",
        "archived_predictions": predictions, "parent_content_hashes": {name: sources[name]["sha256"] for name in parent_keys},
        "source_bindings": {"reference": sources["config"]}}
    seal(plan, "plan_hash")
    score = {"schema_version": "head_matched_scores_v1", "status": "scored", "plan_hash": plan["plan_hash"],
        "identity": identity, "implementation": impl, "synthetic": False, "norm_policy": NORM_POLICY,
        "source_bindings": {"runtime_inputs": sources["runtime_inputs"]}, "predictions": predictions, "score_result": raw}
    seal(score, "score_hash")
    summaries, evidence, observations = {}, {}, []
    validation_sources = {"runtime_inputs": sources["runtime_inputs"]}
    for condition in conds:
        name = condition["condition_id"]
        for i in range(7):
            observations.append({"condition_id": name, "mode": condition["mode"], "call_index": 0,
                "row_count": 3 if i == 0 else 1, "status": "passed", "hidden_exact_equal": True,
                "grouped_logits": {"status": "passed"}, "active_anchor": condition["mode"] != "raw"})
        if condition["mode"] != "raw":
            summary, _ = hook_payload(condition, cases[0])
            summary["direction_binding"]["sae_checkpoint_hash"] = identity["sae_checkpoint_hash"]
            summary["direction_binding_hash"] = fingerprint(summary["direction_binding"])
            summary["num_rows"] = summary["active_rows"] = 9
            summary["edited_rows"] = 0 if condition["mode"] == "identity" else 9
            summaries[name] = summary
            relative = f"{name}/matched_records.jsonl"
            binding = make_file(f"evidence/{relative}")
            evidence[name] = {"path": relative, "sha256": binding["sha256"]}
            validation_sources[f"evidence/{relative}"] = binding
    details = {"schema_version": "head_matched_runtime_v1", "status": "passed", "synthetic": False,
        "identity": identity, "conditions_hash": fingerprint(conds), "conditions": [row["condition_id"] for row in conds],
        "raw_identity_action_match": True, "fixture_sha256": sources["runtime_inputs"]["sha256"],
        "observations": observations, "model_action_queries": 7,
        "action_trace_hashes": {row["condition_id"]: "c" * 64 for row in conds},
        "hook_summaries": summaries, "hook_evidence": evidence}
    validation = {"schema_version": "head_matched_validation_v1", "status": "passed", "plan_hash": plan["plan_hash"],
        "identity": identity, "implementation": impl, "synthetic": False, "source_bindings": validation_sources, "details": details}
    seal(validation, "validation_hash")
    for name, document in (("plan", plan), ("scores", score), ("validation", validation)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        sources[name] = {"path": str(path), "sha256": sha256_file(path)}
    for label, document in (("scores", score), ("validation", validation)):
        sources.update({f"{label}_input/{key}": value for key, value in document["source_bindings"].items()})
    sources["runtime_inputs/calls.pt"] = sources["runtime_inputs"]
    cfg = copy.deepcopy(DEFAULTS)
    cfg["inputs"].update(local_model_snapshot=str(tmp_path / "snapshot"), sae_checkpoint=str(tmp_path / "sae/ae.pt"),
        output_head_bundle=str(tmp_path / "head"), runtime_inputs=str(tmp_path / "calls.pt"), generation_manifest=str(tmp_path / "generation.json"))
    cfg["output"]["root_dir"] = str(tmp_path / "runtime")
    cfg["rollout"]["base_eval_config"] = str(tmp_path / "base.yaml")
    cfg["rollout"]["max_total_rollouts"] = 70
    policy = {"fixture": "no native model"}
    doc = runtime.build_experiment(plan, score, validation, cfg=cfg, identity=identity, implementation=impl,
        code_identity=code, policy=policy, sources=sources, output_root=tmp_path / "runs")
    return SimpleNamespace(root=tmp_path, document=doc, cfg=cfg, plan=plan, scores=score, validation=validation,
        sources=sources, code=code, implementation=impl, identity=identity, policy=policy)


def test_complete_pure_contract_and_portable_status(contract_fixture):
    f = contract_fixture
    runtime.verify_experiment(f.document)
    path = f.root / "execution.json"
    path.write_text(json.dumps(f.document))
    assert runtime.status(path, run_root=f.root / "restored-run")["status"] == "not_started"
    assert not (f.root / "restored-run").exists()


@pytest.mark.parametrize("kind", ["plan", "predictions", "norm", "budget", "source_hash", "parent_content", "dirty", "missing_validation_body", "failed_forward", "inactive", "identity_action", "missing_direction", "raw_scores", "source_input"])
def test_resealed_bad_contracts_rejected(contract_fixture, kind):
    doc = copy.deepcopy(contract_fixture.document)
    if kind == "plan": doc["conditions"][2]["feature_id"] += 1
    elif kind == "predictions": doc["predictions"][0]["mean_head_kl"] = 99.
    elif kind == "norm": doc["norm_policy"] = {"atol": 1., "rtol": 1.}
    elif kind == "budget": doc["max_attempts"] += 1
    elif kind == "source_hash": doc["sources"]["sae"]["sha256"] = "x"
    elif kind == "parent_content": doc["sources"]["sae"]["sha256"] = "f" * 64
    elif kind == "dirty": doc["code_identity"]["dirty"] = True
    elif kind == "missing_validation_body": doc["validation"].pop("details")
    elif kind == "failed_forward": doc["validation"]["details"]["observations"][0]["grouped_logits"]["status"] = "failed"
    elif kind == "inactive":
        for row in doc["validation"]["details"]["observations"]: row["active_anchor"] = False
    elif kind == "identity_action": doc["validation"]["details"]["action_trace_hashes"]["raw"] = "f" * 64
    elif kind == "missing_direction": doc["validation"]["details"]["hook_summaries"].pop("anchor-9")
    elif kind == "raw_scores":
        doc["scores"]["score_result"]["feature_query_scores"]["9"][0]["control_query_kls"][0] = 999.
        seal(doc["scores"]["score_result"], "score_hash")
    else: doc["sources"].pop("validation_input/runtime_inputs")
    seal(doc["scores"], "score_hash")
    seal(doc["validation"], "validation_hash")
    seal(doc, "execution_hash")
    with pytest.raises((ValueError, KeyError)):
        runtime.verify_experiment(doc)


def test_file_content_change_guard(contract_fixture):
    f = contract_fixture
    runtime._check_sources(f.sources)
    Path(f.sources["runtime_inputs"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="source missing or changed"):
        runtime._check_sources(f.sources)


def test_explicit_authorization_before_live_checks(contract_fixture, monkeypatch):
    f = contract_fixture
    path = f.root / "execution.json"
    path.write_text(json.dumps(f.document))
    monkeypatch.setattr(runtime, "git_identity", lambda *a: pytest.fail("must not inspect source before approval"))
    with pytest.raises(ValueError, match="Explicit approval"):
        runtime.run_execution(path, approved_hash="wrong", expected_revision=f.code["commit"])
    with pytest.raises(ValueError, match="Expected code"):
        runtime.run_execution(path, approved_hash=f.document["execution_hash"], expected_revision="wrong")


def test_completed_runtime_does_not_import_or_load_model(contract_fixture, monkeypatch):
    from event_sae.research.head_matched import journal
    f = contract_fixture
    path = f.root / "execution.json"
    path.write_text(json.dumps(f.document))
    monkeypatch.setattr(journal, "inspect", lambda *a: {"status": "completed", "completed_rollouts": 70})
    monkeypatch.setattr(runtime, "git_identity", lambda *a: pytest.fail("no current checkout required for completion"))
    result = runtime.run_execution(path, approved_hash=f.document["execution_hash"], expected_revision=f.code["commit"])
    assert result["new_rollouts"] == 0 and result["model_execution"] is False


def prepare_mocks(f, monkeypatch):
    from event_sae.research.output_head import workflow as old, config as old_config
    from event_sae.research.head_rank import native
    monkeypatch.setattr(runtime, "git_identity", lambda *a: f.code)
    monkeypatch.setattr(runtime, "_processor_preflight", lambda *a: None)
    monkeypatch.setattr(workflow, "implementation_identity", lambda: f.implementation)
    monkeypatch.setattr(workflow, "check_runtime_bindings", lambda *a, **kw: {k: f.sources[k] for k in f.plan["parent_content_hashes"]})
    monkeypatch.setattr(old_config, "load_config", lambda *a: copy.deepcopy(f.cfg))
    monkeypatch.setattr(old, "numerical_identity", lambda *a: f.identity)
    monkeypatch.setattr(old, "load_parity", lambda *a: {})
    monkeypatch.setattr(old, "_sae_paths", lambda *a: (f.root / "sae/ae.pt", f.root / "sae/config.json"))
    monkeypatch.setattr(old, "_head_path", lambda *a: f.root / "head")
    monkeypatch.setattr(native, "effective_policy", lambda *a: f.policy)


def test_prepare_content_binds_all_shards_and_refuses_overwrite(contract_fixture, monkeypatch):
    f = contract_fixture
    prepare_mocks(f, monkeypatch)
    kwargs = dict(config=f.root / "config.yaml", plan_file=f.root / "plan.json", scores_file=f.root / "scores.json",
                  validation_file=f.root / "validation.json", output=f.root / "execution.json", run_output=f.root / "new-runs", max_rollouts=70)
    result = runtime.prepare_execution(**kwargs)
    assert result["new_rollouts"] == 0 and result["total_rollouts"] == 70
    doc = json.loads((f.root / "execution.json").read_text())
    assert len([name for name in doc["sources"] if name.startswith("snapshot/")]) == 4
    runtime.verify_experiment(doc)
    assert not (f.root / "new-runs").exists()
    with pytest.raises(FileExistsError):
        runtime.prepare_execution(**kwargs)


def test_prepare_dirty_code_and_budget_gate(contract_fixture, monkeypatch):
    f = contract_fixture
    prepare_mocks(f, monkeypatch)
    kwargs = dict(config=f.root / "config.yaml", plan_file=f.root / "plan.json", scores_file=f.root / "scores.json",
                  validation_file=f.root / "validation.json", output=f.root / "execution.json", run_output=f.root / "new-runs")
    monkeypatch.setattr(runtime, "git_identity", lambda *a: {**f.code, "dirty": True})
    with pytest.raises(ValueError, match="clean commit"):
        runtime.prepare_execution(**kwargs)
    monkeypatch.setattr(runtime, "git_identity", lambda *a: f.code)
    with pytest.raises(ValueError, match="exceed explicit rollout cap"):
        runtime.prepare_execution(**kwargs, max_rollouts=69)
    with pytest.raises(ValueError, match="exceed explicit rollout cap"):
        runtime.prepare_execution(**kwargs, max_rollouts=70, retry_allowance=1)
