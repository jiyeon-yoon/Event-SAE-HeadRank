"""Mocked CPU contract tests. These do not certify a GPU or simulator."""
from copy import deepcopy
import hashlib
import json

import pytest

from event_sae.research.output_head.provenance import atomic_write_json, fingerprint, sha256_file
from event_sae.research.team500 import execution as ex


def make_plan():
    cases = [{"suite": "libero_spatial", "task_id": task, "task_episode_idx": trial,
              "initial_state_sha256": hashlib.sha256(f"{task}:{trial}".encode()).hexdigest()}
             for task in range(10) for trial in range(50)]
    methods = {"mean_head_kl": [1, 2, 3, 4, 5], "tail_head_kl": [2, 1, 3, 4, 5],
               "event_max": [1, 2, 3, 4, 6]}
    conditions = [{"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.},
                  {"condition_id": "identity-1", "mode": "identity", "feature_id": 1, "alpha": 1.}]
    conditions += [{"condition_id": f"feature-{fid}", "mode": "intervention", "feature_id": fid,
                    "alpha": 0.} for fid in range(1, 7)]
    config = {"schema_version": "mock_cpu_config_not_a_native_runtime", "baseline_config": {}}
    value = {"schema_version": "fixture", "cases": cases, "conditions": conditions,
             "feature_ids": list(range(1, 7)), "top_k": 5, "methods": methods,
             "runtime_config": config, "runtime_config_hash": fingerprint(config),
             "scores_hash": "c" * 64, "score_file_sha256": None, "total_rollouts": 4000}
    value["plan_hash"] = fingerprint(value)
    return value


def mock_receipt(config, condition, cases, output, baseline_root, reference=False,
                 expected_runtime_identity=None):
    episodes = []
    for case in cases:
        action = [float(case["task_id"]), float(case["task_episode_idx"]), 0., 0., 0., 0., -1.]
        if condition["mode"] == "intervention":
            action[2] = .01 * condition["feature_id"]
        episodes.append({**case, "success": case["task_episode_idx"] % 5 != 0,
                         "num_actions": 2, "actions": [action, list(action)], "caught_exception": None})
    hook = None if condition["mode"] == "raw" else {
        "feature_id": condition["feature_id"], "alpha": condition["alpha"], "num_forwards": 14,
        "active_feature_values": 4,
        "absolute_feature_delta": 0. if condition["alpha"] == 1 else 1.,
        "post_intervention_active_feature_values": 0,
        "post_intervention_max_feature_activation": 0.}
    return {"status": "completed", "synthetic": False, "condition": deepcopy(condition),
            "case_manifest_hash": fingerprint(cases), "episodes": episodes,
            "runtime_identity": {"gpu": "MOCKED_CPU_TEST_ONLY", "actual_backend": "mock"},
            "hook_metrics": hook, "elapsed_seconds": .01}


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    plan = make_plan()
    path = tmp_path / "plan.json"
    atomic_write_json(path, plan)
    code = {"commit": "a" * 40, "dirty": False, "files": {"mock_cpu": "b" * 64}}
    monkeypatch.setattr(ex, "_load_plan", lambda path, scores_path=None: deepcopy(plan))
    monkeypatch.setattr(ex, "_code_identity", lambda: deepcopy(code))
    monkeypatch.setattr(ex, "_native_call", mock_receipt)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    return {"plan": plan, "plan_path": path, "code": code, "baseline_root": baseline,
            "output_root": tmp_path / "runs", "preflight_root": tmp_path / "preflight"}


def approve(fixture):
    return {"approved_plan_hash": fixture["plan"]["plan_hash"],
            "expected_code_revision": fixture["code"]["commit"], "execute": True}


def certify(fixture):
    value = ex.preflight(fixture["plan_path"], fixture["preflight_root"], fixture["baseline_root"],
                         **approve(fixture))
    return value["preflight"]


def run(fixture, **kwargs):
    return ex.run_experiment(fixture["plan_path"], fixture["output_root"], fixture["baseline_root"],
        preflight_path=fixture["preflight_root"] / "preflight.json", **approve(fixture), **kwargs)


def test_preview_has_no_backend_or_writes(run_fixture, monkeypatch):
    monkeypatch.setattr(ex, "_native_call", lambda *a, **k: pytest.fail("must not execute"))
    value = ex.preflight(run_fixture["plan_path"], run_fixture["preflight_root"], run_fixture["baseline_root"])
    assert value["preflight_rollouts"] == 10
    assert not run_fixture["preflight_root"].exists()
    value = ex.run_experiment(run_fixture["plan_path"], run_fixture["output_root"], run_fixture["baseline_root"])
    assert value["total_rollouts"] == 4000 and not run_fixture["output_root"].exists()


def test_mocked_preflight_uses_final_plan_first_two_cases_only(run_fixture, monkeypatch):
    calls = []
    def call(config, condition, cases, *args, **kwargs):
        calls.append((deepcopy(cases), condition["mode"], kwargs.get("reference", False)))
        return mock_receipt(config, condition, cases, *args, **kwargs)
    monkeypatch.setattr(ex, "_native_call", call)
    certificate = json.load(open(certify(run_fixture)))
    assert certificate["schema_version"] == "team500_preflight_v2"
    assert certificate["completed_rollouts"] == certificate["max_rollouts"] == 10
    assert [len(c[0]) for c in calls] == [2, 2, 2, 2, 2]
    assert [(r["task_id"], r["task_episode_idx"]) for r in calls[0][0]] == [(0, 0), (0, 1)]
    assert all(c[0] == run_fixture["plan"]["cases"][:2] for c in calls)
    assert calls[0][2] is True and all(c[2] is False for c in calls[1:])


def test_old_402_budget_is_rejected_before_native_work(run_fixture, monkeypatch):
    monkeypatch.setattr(ex, "_native_call", lambda *a, **k: pytest.fail("old budget must not execute"))
    with pytest.raises(ValueError, match="budget"):
        ex.preflight(run_fixture["plan_path"], run_fixture["preflight_root"],
                     run_fixture["baseline_root"], max_rollouts=402, **approve(run_fixture))
    assert not run_fixture["preflight_root"].exists()


def test_old_v1_402_certificate_cannot_authorize_new_execution(run_fixture, monkeypatch):
    old = json.load(open(certify(run_fixture)))
    old.update(schema_version="team500_preflight_v1", completed_rollouts=402, max_rollouts=402,
               prefix_manifest_hash=fingerprint(run_fixture["plan"]["cases"][:100]))
    old["preflight_hash"] = fingerprint({k: v for k, v in old.items() if k != "preflight_hash"})
    path = run_fixture["preflight_root"] / "old-certificate.json"
    atomic_write_json(path, old)
    monkeypatch.setattr(ex, "_native_call", lambda *a, **k: pytest.fail("old certificate must not execute"))
    with pytest.raises(ValueError, match="Preflight is missing, stale"):
        ex.run_experiment(run_fixture["plan_path"], run_fixture["output_root"],
            run_fixture["baseline_root"], preflight_path=path, **approve(run_fixture))
    assert not run_fixture["output_root"].exists()


def test_exact_actions_not_just_success_rate_gate(run_fixture, monkeypatch):
    def call(*args, **kwargs):
        receipt = mock_receipt(*args, **kwargs)
        if not kwargs.get("reference", False):
            receipt["episodes"][0]["actions"][0][0] += .000001
        return receipt
    monkeypatch.setattr(ex, "_native_call", call)
    with pytest.raises(ValueError, match="Exact native actions"):
        certify(run_fixture)
    assert not (run_fixture["preflight_root"] / "preflight.json").exists()


def test_missing_preflight_and_dirty_revision_block_before_backend(run_fixture, monkeypatch):
    with pytest.raises(ValueError, match="preflight"):
        ex.run_experiment(run_fixture["plan_path"], run_fixture["output_root"],
                          run_fixture["baseline_root"], **approve(run_fixture))
    code = deepcopy(run_fixture["code"])
    code["dirty"] = True
    monkeypatch.setattr(ex, "_code_identity", lambda: code)
    with pytest.raises(ValueError, match="clean"):
        certify(run_fixture)


def test_no_feature_before_full_raw_identity_gate_and_safe_pause(run_fixture, monkeypatch):
    certify(run_fixture)
    calls = []
    def call(config, condition, cases, *args, **kwargs):
        calls.append(condition["condition_id"])
        assert len(cases) == 500
        return mock_receipt(config, condition, cases, *args, **kwargs)
    monkeypatch.setattr(ex, "_native_call", call)
    paused = run(run_fixture, max_new_conditions=2)
    assert calls == ["raw", "identity-1"] and paused["completed_rollouts"] == 1000
    done = run(run_fixture)
    assert done["status"] == "completed" and done["completed_rollouts"] == 4000
    assert calls == [c["condition_id"] for c in run_fixture["plan"]["conditions"]]
    assert ex.status(run_fixture["plan_path"], run_fixture["output_root"])["pending_rollouts"] == 0


def test_small_preflight_never_substitutes_for_full500_raw_controls(run_fixture, monkeypatch):
    certify(run_fixture)
    calls = []
    def partial(config, condition, cases, *args, **kwargs):
        calls.append((condition["condition_id"], len(cases)))
        receipt = mock_receipt(config, condition, cases, *args, **kwargs)
        receipt["episodes"] = receipt["episodes"][:2]
        return receipt
    monkeypatch.setattr(ex, "_native_call", partial)
    with pytest.raises(ValueError, match="episode count"):
        run(run_fixture)
    assert calls == [("raw", 500)]
    state = ex.status(run_fixture["plan_path"], run_fixture["output_root"])
    assert state["completed_conditions"] == 0 and state["completed_rollouts"] == 0


def test_failed_identity_cannot_launch_feature_or_count_task_failure(run_fixture, monkeypatch):
    certify(run_fixture)
    calls = []
    def call(config, condition, cases, *args, **kwargs):
        calls.append(condition["mode"])
        receipt = mock_receipt(config, condition, cases, *args, **kwargs)
        if condition["mode"] == "identity":
            receipt["episodes"][0]["success"] = not receipt["episodes"][0]["success"]
        return receipt
    monkeypatch.setattr(ex, "_native_call", call)
    with pytest.raises(ValueError, match="Exact native actions"):
        run(run_fixture)
    assert calls == ["raw", "identity"]
    status = ex.status(run_fixture["plan_path"], run_fixture["output_root"])
    assert status["completed_rollouts"] == 500 and status["incomplete_condition_attempts"] == 1
    failures = list(run_fixture["output_root"].glob("attempts/*/*/failure.json"))
    assert json.load(open(failures[0]))["status"] == "failed_not_a_task_failure"


def test_retry_restarts_entire_condition_and_consumes_declared_reserve(run_fixture, monkeypatch):
    certify(run_fixture)
    calls = []
    failed = False
    def call(config, condition, cases, *args, **kwargs):
        nonlocal failed
        calls.append((condition["condition_id"], len(cases), cases[0]["task_episode_idx"]))
        if not failed:
            failed = True
            raise RuntimeError("interrupted after unknown work")
        return mock_receipt(config, condition, cases, *args, **kwargs)
    monkeypatch.setattr(ex, "_native_call", call)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(run_fixture, retry_budget_conditions=1)
    with pytest.raises(ValueError, match="retry-incomplete"):
        run(run_fixture, retry_budget_conditions=1)
    done = run(run_fixture, retry_budget_conditions=1, retry_incomplete=True)
    assert done["reserved_rollout_attempts_used"] == 4500
    assert calls[:2] == [("raw", 500, 0), ("raw", 500, 0)]
    assert ex.status(run_fixture["plan_path"], run_fixture["output_root"])["historical_incomplete_attempts"] == 1


def test_runtime_change_or_tampered_preflight_stops_execution(run_fixture, monkeypatch):
    cert = certify(run_fixture)
    payload = json.load(open(cert))
    payload["checks"]["raw_identity"]["cases"] = 99
    # Test corruption via the normal non-overwriting utility in a different file.
    corrupted = run_fixture["preflight_root"] / "corrupt.json"
    atomic_write_json(corrupted, payload)
    with pytest.raises(ValueError, match="preflight_hash"):
        ex.run_experiment(run_fixture["plan_path"], run_fixture["output_root"], run_fixture["baseline_root"],
            preflight_path=corrupted, **approve(run_fixture))
    def call(*args, **kwargs):
        result = mock_receipt(*args, **kwargs)
        result["runtime_identity"]["actual_backend"] = "changed"
        return result
    monkeypatch.setattr(ex, "_native_call", call)
    with pytest.raises(ValueError, match="runtime changed"):
        run(run_fixture)


def test_exceptions_and_nonfinite_actions_never_count_as_task_failure(run_fixture):
    plan = run_fixture["plan"]
    receipt = mock_receipt(plan["runtime_config"], plan["conditions"][0], plan["cases"], None, None)
    receipt["episodes"][0]["caught_exception"] = "EGL failure"
    with pytest.raises(ValueError, match="exception"):
        ex.validate_receipt(receipt, plan["conditions"][0], plan["cases"])
    receipt["episodes"][0]["caught_exception"] = None
    receipt["episodes"][0]["actions"][0][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        ex.validate_receipt(receipt, plan["conditions"][0], plan["cases"])


def test_inactive_suppression_smoke_cannot_authorize_full_run(run_fixture, monkeypatch):
    def call(*args, **kwargs):
        result = mock_receipt(*args, **kwargs)
        if result["condition"]["alpha"] == 0:
            result["hook_metrics"]["active_feature_values"] = 0
            result["hook_metrics"]["absolute_feature_delta"] = 0.
        return result
    monkeypatch.setattr(ex, "_native_call", call)
    with pytest.raises(ValueError, match="active target"):
        certify(run_fixture)
    assert not (run_fixture["preflight_root"] / "preflight.json").exists()


def test_real_plan_loader_rebuilds_top5_from_bound_scores(tmp_path):
    from test_team500_protocol import scores_fixture, cases_fixture, runtime_fixture
    from event_sae.research.team500.protocol import build_plan
    scores = scores_fixture()
    score_path = tmp_path / "scores.json"
    atomic_write_json(score_path, scores)
    plan = build_plan(scores, cases_fixture(), runtime_fixture(),
        score_file_sha256=sha256_file(score_path), reference_bindings={"scores_path": str(score_path)})
    path = tmp_path / "plan.json"
    atomic_write_json(path, plan)
    assert ex._load_plan(path) == plan
    changed = deepcopy(plan)
    changed["methods"]["mean_head_kl"][1:3] = changed["methods"]["mean_head_kl"][1:3][::-1]
    changed["plan_hash"] = fingerprint({k: v for k, v in changed.items() if k != "plan_hash"})
    other = tmp_path / "changed-plan.json"
    atomic_write_json(other, changed)
    with pytest.raises(ValueError, match="rankings differ"):
        ex._load_plan(other)


def test_certificate_recomputes_checks_even_if_outer_hash_is_recomputed(run_fixture):
    certificate = json.load(open(certify(run_fixture)))
    certificate["checks"]["raw_identity"]["cases"] = 99
    certificate["preflight_hash"] = fingerprint({k: v for k, v in certificate.items() if k != "preflight_hash"})
    changed = run_fixture["preflight_root"] / "changed-certificate.json"
    atomic_write_json(changed, certificate)
    with pytest.raises(ValueError, match="checks do not match"):
        ex.run_experiment(run_fixture["plan_path"], run_fixture["output_root"], run_fixture["baseline_root"],
            preflight_path=changed, **approve(run_fixture))
