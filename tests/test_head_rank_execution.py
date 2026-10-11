"""Fake-backend case journaling tests; no GPU/model/robot is executed."""
import copy
import fcntl
import json
from pathlib import Path

import pytest

from event_sae.research.head_rank.execution import execute_experiment, load_completed_results, inspect_experiment
from event_sae.research.output_head.provenance import fingerprint, sha256_file


def experiment(reserve=0):
    conditions = [
        {"condition_id": "feature-15", "mode": "intervention", "feature_id": 15, "alpha": 0.},
        {"condition_id": "identity-9", "mode": "identity", "feature_id": 9, "alpha": 1.},
        {"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.},
        {"condition_id": "feature-9", "mode": "intervention", "feature_id": 9, "alpha": 0.},
    ]
    cases = [{"suite": "libero_spatial", "task_id": t, "task_episode_idx": i,
              "initial_state_sha256": str(n) * 64, "seed": seed}
             for t, i, n, seed in [(7, 29, 1, 42029), (2, 31, 2, 92031)]]
    result = {"schema_version": "head_rank_execution_v1", "plan_hash": "a" * 64,
              "conditions": conditions, "cases": cases, "total_rollouts": 8,
              "max_attempts": 8 + reserve, "code_identity": {"commit": "b" * 40, "dirty": False},
              "numerical_identity": {"device": "fake"}, "policy_config_hash": "c" * 64}
    result["execution_hash"] = fingerprint(result)
    return result


def seal(payload, field):
    payload[field] = fingerprint({k: v for k, v in payload.items() if k != field})
    return payload


class FakeBackend:
    def __init__(self, *, fail=None, identity_change=None, malformed=None):
        self.calls, self.validated = [], []
        self.fail, self.identity_change, self.malformed = fail, identity_change, malformed

    def validate_cases(self, cases):
        assert not self.calls
        self.validated.append(copy.deepcopy(cases))

    def run_case(self, condition, case, directory):
        assert self.validated
        assert (directory / "started.json").is_file()
        self.calls.append((copy.deepcopy(condition), copy.deepcopy(case), directory))
        if self.fail:
            raise self.fail
        action = [case["seed"] / 1e6] * 7
        success = case["task_id"] == 7
        if condition["mode"] == "identity" and self.identity_change == "action":
            action[0] += .001
        if condition["mode"] == "identity" and self.identity_change == "outcome":
            success = not success
        if condition["mode"] == "intervention":
            success = condition["feature_id"] == 9
            action[0] += .05
        episode = {**case, "success": success, "caught_exception": None, "num_actions": 1}
        metrics, evidence = None, None
        if condition["mode"] != "raw":
            metrics = {"feature_id": condition["feature_id"], "alpha": condition["alpha"],
                       "num_forwards": 7, "num_environment_steps": 1, "active_feature_values": 0,
                       "post_intervention_active_feature_values": 0, "max_feature_activation": 0.,
                       "post_intervention_max_feature_activation": 0., "absolute_feature_delta": 0.}
            path = directory / "native" / "hook.jsonl"
            path.parent.mkdir()
            path.write_text(json.dumps({"feature_id": condition["feature_id"], "alpha": condition["alpha"],
                "task_id": case["task_id"], "task_episode_idx": case["task_episode_idx"],
                "step_in_episode": 0, "first_forward_idx": 1}) + "\n")
            evidence = {"path": str(path), "sha256": sha256_file(path)}
        result = {"episode": episode, "actions": [action], "hook_metrics": metrics, "hook_evidence": evidence}
        if self.malformed:
            self.malformed(result, condition, case, directory)
        return result


def run(exp, root, backend, **kwargs):
    return execute_experiment(exp, root, backend, approved_hash=exp["execution_hash"], **kwargs)


def test_routing_exact_cases_and_seeds_raw_identity_all_first(tmp_path):
    exp, backend = experiment(), FakeBackend()
    result = run(exp, tmp_path / "runs", backend)
    assert result["status"] == "completed" and result["new_attempts"] == 8
    assert [c[0]["condition_id"] for c in backend.calls] == ["raw", "identity-9", "raw", "identity-9", "feature-15", "feature-15", "feature-9", "feature-9"]
    assert [c[1] for c in backend.calls] == [case for case in exp["cases"] for _ in range(2)] + exp["cases"] * 2
    assert backend.validated == [exp["cases"]]
    results = load_completed_results(exp, tmp_path / "runs")
    assert set(results) == {c["condition_id"] for c in exp["conditions"]}
    assert all(len(rows) == 2 for rows in results.values())
    assert results["feature-9"][0]["result"]["hook_evidence"]["path"] == "native/hook.jsonl"
    assert all(row["result"]["hook_metrics"]["active_feature_values"] == 0 for row in results["feature-9"])


def test_completed_resume_never_calls_backend_or_gpu(tmp_path):
    exp = experiment()
    run(exp, tmp_path / "runs", FakeBackend())
    class NoBackend:
        def __getattr__(self, _):
            raise AssertionError("complete resume cannot inspect native backend")
    result = run(exp, tmp_path / "runs", NoBackend())
    assert result["new_attempts"] == 0 and result["model_execution"] is False


def test_clean_boundary_cap_and_resume_preserves_prior_receipts(tmp_path):
    exp, first = experiment(), FakeBackend()
    result = run(exp, tmp_path / "runs", first, max_new_attempts=3)
    assert result["status"] == "paused" and len(first.calls) == 3
    hashes = {p: sha256_file(p) for p in (tmp_path / "runs").rglob("result.json")}
    with pytest.raises(ValueError, match="incomplete"):
        load_completed_results(exp, tmp_path / "runs")
    second = FakeBackend()
    result = run(exp, tmp_path / "runs", second)
    assert result["status"] == "completed" and len(second.calls) == 5
    assert all(sha256_file(p) == value for p, value in hashes.items())


def test_zero_cap_does_not_validate_backend(tmp_path):
    exp, backend = experiment(), FakeBackend()
    assert run(exp, tmp_path / "runs", backend, max_new_attempts=0)["status"] == "paused"
    assert not backend.calls and not backend.validated


def test_two_attempt_cap_is_one_identity_pair_not_two_raw_cases(tmp_path):
    exp, backend = experiment(), FakeBackend()
    result = run(exp, tmp_path / "runs", backend, max_new_attempts=2)
    assert result["status"] == "paused" and result["completed_rollouts"] == 2
    assert [call[0]["mode"] for call in backend.calls] == ["raw", "identity"]
    assert backend.calls[0][1] == backend.calls[1][1] == exp["cases"][0]


@pytest.mark.parametrize("change", ["action", "outcome"])
def test_identity_mismatch_blocks_every_feature_call_and_resume(tmp_path, change):
    exp, backend = experiment(), FakeBackend(identity_change=change)
    with pytest.raises(ValueError, match="parity"):
        run(exp, tmp_path / "runs", backend)
    assert all(row[0]["mode"] != "intervention" for row in backend.calls)
    with pytest.raises(ValueError, match="parity"):
        run(exp, tmp_path / "runs", FakeBackend(), retry_incomplete=True)


def test_exception_receipt_is_not_an_outcome_and_explicit_reserved_retry(tmp_path):
    exp = experiment(reserve=1)
    with pytest.raises(RuntimeError, match="no task failure"):
        run(exp, tmp_path / "runs", FakeBackend(fail=RuntimeError("simulator exploded")))
    files = list((tmp_path / "runs").rglob("result.json"))
    assert len(files) == 1
    failed = json.loads(files[0].read_text())
    assert failed["status"] == "failed" and "result" not in failed
    with pytest.raises(ValueError, match="explicit retry"):
        run(exp, tmp_path / "runs", FakeBackend())
    result = run(exp, tmp_path / "runs", FakeBackend(), retry_incomplete=True)
    assert result["status"] == "completed" and result["attempts_used"] == 9
    assert json.loads(files[0].read_text()) == failed


def test_retry_cannot_spend_unapproved_reserve(tmp_path):
    exp = experiment()
    with pytest.raises(RuntimeError):
        run(exp, tmp_path / "runs", FakeBackend(fail=RuntimeError("bad")))
    backend = FakeBackend()
    with pytest.raises(ValueError, match="retry reserve"):
        run(exp, tmp_path / "runs", backend, retry_incomplete=True)
    assert not backend.calls and not backend.validated


def test_interrupted_started_counts_budget_and_lock_released(tmp_path):
    exp = experiment(reserve=1)
    with pytest.raises(KeyboardInterrupt):
        run(exp, tmp_path / "runs", FakeBackend(fail=KeyboardInterrupt()))
    assert len(list((tmp_path / "runs").rglob("started.json"))) == 1
    assert not list((tmp_path / "runs").rglob("result.json"))
    with pytest.raises(ValueError, match="explicit retry"):
        run(exp, tmp_path / "runs", FakeBackend())
    assert run(exp, tmp_path / "runs", FakeBackend(), retry_incomplete=True)["attempts_used"] == 9


@pytest.mark.parametrize("mutation", ["success", "caught", "seed", "task", "action_nan", "action_shape", "count", "raw_hook"])
def test_bad_backend_payload_stops_and_writes_failed_receipt(tmp_path, mutation):
    def mutate(result, *_):
        if mutation == "success": result["episode"]["success"] = 1
        elif mutation == "caught": result["episode"]["caught_exception"] = "crash"
        elif mutation == "seed": result["episode"]["seed"] += 1
        elif mutation == "task": result["episode"]["task_id"] = 8
        elif mutation == "action_nan": result["actions"][0][0] = float("nan")
        elif mutation == "action_shape": result["actions"][0].pop()
        elif mutation == "count": result["episode"]["num_actions"] = 2
        else: result["hook_metrics"] = {}
    backend = FakeBackend(malformed=mutate)
    with pytest.raises(RuntimeError):
        run(experiment(), tmp_path / "runs", backend)
    assert len(backend.calls) == 1
    assert json.loads(next((tmp_path / "runs").rglob("result.json")).read_text())["status"] == "failed"


@pytest.mark.parametrize("mutation", ["feature", "alpha", "forward", "post_active", "post_max", "evidence", "outside", "record", "step", "first_forward"])
def test_hook_metrics_and_evidence_verified(tmp_path, mutation):
    def mutate(result, condition, case, directory):
        if condition["mode"] != "intervention": return
        metrics = result["hook_metrics"]
        if mutation == "feature": metrics["feature_id"] += 1
        elif mutation == "alpha": metrics["alpha"] = .5
        elif mutation == "forward": metrics["num_forwards"] = 0
        elif mutation == "post_active": metrics["post_intervention_active_feature_values"] = 1
        elif mutation == "post_max": metrics["post_intervention_max_feature_activation"] = .1
        elif mutation == "evidence": result["hook_evidence"]["sha256"] = "0" * 64
        elif mutation == "outside": result["hook_evidence"]["path"] = "../elsewhere.jsonl"
        elif mutation in ("record", "step", "first_forward"):
            path = Path(result["hook_evidence"]["path"])
            row = json.loads(path.read_text())
            if mutation == "record": row["task_episode_idx"] += 1
            elif mutation == "step": row["step_in_episode"] = 1
            else: row["first_forward_idx"] = 2
            path.write_text(json.dumps(row) + "\n")
            result["hook_evidence"]["sha256"] = sha256_file(path)
    backend = FakeBackend(malformed=mutate)
    with pytest.raises(RuntimeError):
        run(experiment(), tmp_path / "runs", backend)
    assert len(backend.calls) == 5


@pytest.mark.parametrize("mutation", ["hash", "seed_rehash", "hook_evidence", "started"])
def test_resume_rejects_tampered_completed_results(tmp_path, mutation):
    exp = experiment()
    run(exp, tmp_path / "runs", FakeBackend())
    receipt_path = next((tmp_path / "runs" / "attempts" / "feature-9").rglob("result.json"))
    receipt = json.loads(receipt_path.read_text())
    if mutation == "hook_evidence":
        (receipt_path.parent / receipt["result"]["hook_evidence"]["path"]).write_text("changed")
    elif mutation == "started":
        path = receipt_path.parent / "started.json"
        data = json.loads(path.read_text())
        data["case"]["seed"] += 1
        path.write_text(json.dumps(seal(data, "started_hash")))
    else:
        receipt["result"]["episode"]["seed"] += 1
        if mutation == "seed_rehash": seal(receipt, "receipt_hash")
        receipt_path.write_text(json.dumps(receipt))
    backend = FakeBackend()
    with pytest.raises(ValueError): run(exp, tmp_path / "runs", backend)
    assert not backend.calls


def test_exclusive_lock_and_symlink_rejection(tmp_path):
    exp = experiment()
    root = tmp_path / "runs"
    root.mkdir()
    with (root / ".execution.lock").open("w") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="busy/locked"):
            run(exp, root, FakeBackend())
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    with pytest.raises(ValueError, match="Symlink"):
        run(exp, alias, FakeBackend())
    assert run(exp, root, FakeBackend())["status"] == "completed"


def test_stale_approval_and_changed_experiment_do_not_execute(tmp_path):
    exp, backend = experiment(), FakeBackend()
    with pytest.raises(ValueError, match="Approved"):
        execute_experiment(exp, tmp_path / "runs", backend, approved_hash="0" * 64)
    assert not (tmp_path / "runs").exists()
    run(exp, tmp_path / "runs", backend, max_new_attempts=0)
    exp["max_attempts"] += 1
    seal(exp, "execution_hash")
    with pytest.raises(ValueError, match="different/modified"):
        run(exp, tmp_path / "runs", FakeBackend())


def test_unknown_journal_entries_rejected(tmp_path):
    exp = experiment()
    run(exp, tmp_path / "runs", FakeBackend(), max_new_attempts=1)
    (tmp_path / "runs" / "attempts" / "feature-999").mkdir()
    with pytest.raises(ValueError, match="Unknown condition"):
        run(exp, tmp_path / "runs", FakeBackend())


def test_cpu_inspection_never_misclassifies_corruption_as_incomplete(tmp_path):
    exp, root = experiment(), tmp_path / "runs"
    assert inspect_experiment(exp, root)["status"] == "not_started"
    assert not root.exists()
    run(exp, root, FakeBackend(), max_new_attempts=1)
    partial = inspect_experiment(exp, root)
    assert partial["status"] == "incomplete" and partial["pending_rollouts"] == 7
    run(exp, root, FakeBackend())
    assert inspect_experiment(exp, root)["status"] == "completed"
    next(root.rglob("result.json")).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_experiment(exp, root)


def test_empty_output_or_only_leftover_lock_can_start_without_overwriting(tmp_path):
    exp, root = experiment(), tmp_path / "runs"
    root.mkdir()
    assert inspect_experiment(exp, root)["status"] == "not_started"
    assert set(p.name for p in root.iterdir()) == {".execution.lock"}
    assert inspect_experiment(exp, root)["status"] == "not_started"
    assert run(exp, root, FakeBackend(), max_new_attempts=0)["status"] == "paused"
    assert (root / "experiment.json").is_file()


@pytest.mark.parametrize("existing", ["note.txt", "attempts"])
def test_unrelated_output_or_orphan_attempts_are_never_initialized(tmp_path, existing):
    exp, root = experiment(), tmp_path / "runs"
    root.mkdir()
    source = root / existing
    if existing == "attempts": source.mkdir()
    else: source.write_text("must preserve this unrelated file")
    for operation in (lambda: inspect_experiment(exp, root), lambda: run(exp, root, FakeBackend())):
        with pytest.raises(ValueError, match="Unrelated files or orphan"):
            operation()
    assert not (root / "experiment.json").exists()
    assert source.exists()
