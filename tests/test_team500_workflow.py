"""Setup/CLI checks do not require LIBERO or any GPU."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from event_sae.research.team500 import workflow


def registry_fixture():
    return {"suite": "libero_spatial", "episodes": [
        {"task_id": task, "task_episode_idx": trial,
         "initial_state_sha256": f"{task * 50 + trial:064x}"}
        for task in range(10) for trial in range(50)]}


def test_full500_order_is_exact_not_randomized():
    registry = registry_fixture()
    registry["episodes"].reverse()
    cases = workflow.ordered_registry_cases(registry)
    assert len(cases) == 500
    assert [(x["task_id"], x["task_episode_idx"]) for x in cases] == [
        (t, i) for t in range(10) for i in range(50)]
    assert "seed" not in cases[0]


def test_missing_duplicate_population_fails():
    registry = registry_fixture()
    registry["episodes"][-1] = registry["episodes"][0]
    with pytest.raises(ValueError, match="exactly"):
        workflow.ordered_registry_cases(registry)


def test_setup_explicit_simulator_permission_first(tmp_path):
    with pytest.raises(ValueError, match="allow-libero"):
        workflow.setup(tmp_path, tmp_path / "out", tmp_path / "baseline")


def test_prepare_no_execute_never_opens_inputs():
    result = workflow.prepare_cache("/missing/config.yaml", execute=False)
    assert result["status"] == "not_executed"


def test_write_once_no_overwrite(tmp_path):
    target = tmp_path / "test.json"
    workflow.write_once(target, {"a": 1})
    workflow.write_once(target, {"a": 1})
    with pytest.raises(ValueError, match="differs"):
        workflow.write_once(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}


def test_cli_help_and_dry_prepare_do_not_import_torch():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts/openvla/headteam500.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "preflight" in result.stdout
    result = subprocess.run([sys.executable, str(script), "prepare-cache", "--head-config", "/absent"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["policy_execution"] is False


def test_setup_freezes_stock_config_and_preserves_history(tmp_path, monkeypatch):
    """Exercise orchestration without a simulator, model or real checkpoints."""
    import yaml
    from scripts.openvla import headfull
    from event_sae.research.output_head import state_registry
    from event_sae.research.output_head.provenance import fingerprint, sha256_file
    from event_sae.research.team500 import native
    from event_sae.research.team500.protocol import validate_runtime_config

    run = tmp_path / "new-run"
    p = workflow.paths(tmp_path, run)
    p["sae"].parent.mkdir(parents=True)
    p["sae"].write_bytes(b"CPU fixture: not a real SAE")
    p["event_scores"].parent.mkdir(parents=True)
    p["event_scores"].write_bytes(b"CPU fixture: not public Event scores")
    monkeypatch.setattr(headfull, "check_merged", lambda paths: None)
    monkeypatch.setattr(headfull, "SAE_SHA", sha256_file(p["sae"]))
    monkeypatch.setattr(workflow, "PUBLIC_EVENT_SHA", sha256_file(p["event_scores"]))
    monkeypatch.setattr(workflow, "snapshot_hashes", lambda path: {"config.json": "a" * 64})
    monkeypatch.setattr(workflow, "remote_code_hashes", lambda: {"test.py": "b" * 64})
    monkeypatch.setattr(native, "verify_baseline", lambda path: {"revision": workflow.BASELINE_COMMIT})
    registry = {**registry_fixture(), "schema_version": state_registry.SCHEMA,
                "dataset_id": "CPU_fixture_only", "historical_state_identity_verified": False,
                "structural_audit": {"all_action_queries_complete": True, "layer_idx": 31, "action_dim": 7},
                "source_files": {"activation_index": {"sha256": "c" * 64}}}
    registry["manifest_hash"] = fingerprint(registry)
    monkeypatch.setattr(state_registry, "build_state_registry", lambda *a, **kw: registry)
    result = workflow.setup(tmp_path, run, tmp_path / "baseline", allow_libero=True)
    assert result["selection_episodes"] == result["evaluation_cases"] == 500
    assert result["selection_evaluation_state_overlap"] is True
    runtime = json.loads(Path(result["runtime_config"]).read_text())
    validate_runtime_config(runtime)
    assert "per_episode_seed" not in runtime["baseline_config"]["env"]
    assert "trial_indices" not in runtime["baseline_config"]["env"]
    assert runtime["baseline_config"]["env"]["num_trials_per_task"] == 50
    generation = json.loads((run / "inputs/generation.json").read_text())
    assert generation["evidence"]["historical_runtime_flags_verified"] is False
    before = {x: x.read_bytes() for x in run.rglob("*") if x.is_file()}
    assert workflow.setup(tmp_path, run, tmp_path / "baseline", allow_libero=True) == result
    assert {x: x.read_bytes() for x in run.rglob("*") if x.is_file()} == before
    # Later numerical fixture capture may add paths, but not change behavior.
    head_path = Path(result["head_config"])
    head = yaml.safe_load(head_path.read_text())
    head["inputs"]["runtime_inputs"] = str(run / "inputs/runtime.pt")
    head_path.write_text(yaml.safe_dump(head))
    workflow.setup(tmp_path, run, tmp_path / "baseline", allow_libero=True)
    assert yaml.safe_load(head_path.read_text())["inputs"]["runtime_inputs"].endswith("runtime.pt")


def test_cli_plan_and_verification_bind_actual_score_bytes(tmp_path):
    from test_team500_protocol import scores_fixture, cases_fixture, runtime_fixture

    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts/openvla/headteam500.py"
    for name, payload in (("scores", scores_fixture()), ("cases", {"cases": cases_fixture()}),
                          ("runtime", runtime_fixture())):
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    plan = tmp_path / "plan.json"
    result = subprocess.run([sys.executable, str(script), "plan",
        "--scores", str(tmp_path / "scores.json"), "--cases", str(tmp_path / "cases.json"),
        "--runtime-config", str(tmp_path / "runtime.json"), "--output", str(plan)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["total_rollouts"] == 5000
    result = subprocess.run([sys.executable, str(script), "verify-plan", "--plan", str(plan)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    moved = tmp_path / "relocated-scores.json"
    (tmp_path / "scores.json").rename(moved)
    result = subprocess.run([sys.executable, str(script), "verify-plan", "--plan", str(plan),
                            "--scores", str(moved)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    moved.write_text(moved.read_text() + "\n")
    result = subprocess.run([sys.executable, str(script), "verify-plan", "--plan", str(plan),
                            "--scores", str(moved)], capture_output=True, text=True)
    assert result.returncode == 2 and "score file differs" in result.stderr
