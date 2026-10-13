"""No-GPU adversarial tests for the isolated norm-matched case journal."""
import copy
import fcntl
import json

import pytest

from event_sae.research.head_matched.journal import execute, inspect, load_results
from event_sae.research.output_head.provenance import fingerprint, sha256_file
from test_head_rank_native import fixture_runtime


def experiment(reserve=0):
    policy = {"atol": 1e-5, "rtol": .05}
    conditions = [
        {"condition_id": "control-12", "mode": "matched_decoder", "feature_id": 9, "alpha": 0,
         "control_feature_id": 12, "norm_policy": policy},
        {"condition_id": "anchor-9", "mode": "anchor", "feature_id": 9, "alpha": 0, "norm_policy": policy},
        {"condition_id": "identity-9", "mode": "identity", "feature_id": 9, "alpha": 1, "norm_policy": policy},
        {"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1},
    ]
    cases = [{"suite": "libero_spatial", "task_id": task, "task_episode_idx": 3,
              "initial_state_sha256": str(task + 1) * 64, "seed": task + 100}
             for task in (0, 1)]
    result = {"schema_version": "head_matched_execution_v1", "plan_hash": "a" * 64,
              "conditions": conditions, "cases": cases, "total_rollouts": 8,
              "max_attempts": 8 + reserve, "identity": {"sae_checkpoint_hash": "b" * 64},
              "validation": {"details": {"hook_summaries": {
                  row["condition_id"]: hook_payload(row, cases[0])[0] for row in conditions if row["mode"] != "raw"}}}}
    result["execution_hash"] = fingerprint(result)
    return result


def hook_payload(condition, case):
    mode, fid = condition["mode"], condition["feature_id"]
    cid, value = condition.get("control_feature_id"), 0. if mode == "identity" else 1.
    binding = {"mode": mode, "anchor_feature_id": fid, "sae_checkpoint_hash": "b" * 64,
               "support": "current_original_forward_anchor_activation",
               "norm_scope": "per_original_forward_row_same_incoming_hidden"}
    if mode == "matched_decoder":
        binding.update(control_feature_id=cid, sign="negative_decoder_unit", direction_fp32_sha256="c" * 64)
    shared = {"condition_id": condition["condition_id"], "mode": mode, "feature_id": fid,
              "control_feature_id": cid, "norm_policy": condition["norm_policy"],
              "encoder_grouping": "original_forward"}
    metrics = {"schema_version": "head_matched_hook_v1", **shared, "alpha": condition["alpha"],
        "direction_binding": binding, "direction_binding_hash": fingerprint(binding),
        "num_forwards": 7, "num_rows": 7, "active_rows": 7, "edited_rows": 0 if mode == "identity" else 7,
        "norm_invalid_rows": 0, "off_support_rows": 0, "cross_trajectory_norm_match": False,
        "control_is_feature_ablation": False,
        **{f"max_{name}": 0. for name in ("pair_norm_abs_error", "pair_norm_rel_error", "anchor_cast_abs_error", "control_cast_abs_error")}}
    records = []
    for index in range(7):
        records.append({"schema_version": "head_matched_forward_v1", **shared,
            "task_id": case["task_id"], "task_episode_idx": case["task_episode_idx"],
            "step_in_episode": 0, "forward_index": index + 1, "num_rows": 1,
            "direction_binding_hash": fingerprint(binding), "norm_invalid_rows": 0, "off_support_rows": 0,
            "anchor_active": [True], "eligible": [True], "finite": [True], "off_support": [False],
            "norm_scope": "per_original_forward_row_same_incoming_hidden", "cross_trajectory_norm_match": False,
            **{name: [value] for name in ("intended_norm", "anchor_realized_norm", "control_realized_norm", "control_pre_cast_norm")},
            **{name: [0.] for name in ("pair_norm_abs_error", "pair_norm_rel_error", "anchor_cast_abs_error", "control_cast_abs_error", "pre_cast_abs_error")}})
    return metrics, records


class FakeBackend:
    def __init__(self, *, fail=None, mutate=None, hook_mutate=None, identity_change=False):
        self.fail, self.mutate, self.hook_mutate = fail, mutate, hook_mutate
        self.identity_change = identity_change
        self.calls, self.validated = [], []

    def validate_cases(self, cases):
        assert not self.calls
        self.validated.append(copy.deepcopy(cases))

    def run_case(self, condition, case, directory):
        assert self.validated and (directory / "started.json").is_file()
        self.calls.append((copy.deepcopy(condition), copy.deepcopy(case)))
        if self.fail:
            raise self.fail
        success = case["task_id"] == 0
        action = [0.] * 7
        if condition["mode"] in ("anchor", "matched_decoder"):
            action[0] = .2
        if condition["mode"] == "identity" and self.identity_change:
            action[0] = .1
        metrics = evidence = None
        if condition["mode"] != "raw":
            metrics, records = hook_payload(condition, case)
            if self.hook_mutate:
                self.hook_mutate(metrics, records, condition)
            path = directory / "native" / "matched_records.jsonl"
            path.parent.mkdir()
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            evidence = {"path": str(path), "sha256": sha256_file(path)}
        result = {"episode": {**case, "success": success, "caught_exception": None, "num_actions": 1},
                  "actions": [action], "hook_metrics": metrics, "hook_evidence": evidence}
        if self.mutate:
            self.mutate(result, condition, directory)
        return result


def run(exp, root, backend, **kwargs):
    return execute(exp, root, backend, approved_hash=exp["execution_hash"], **kwargs)


def test_casewise_first_four_cover_every_mode_and_resume(tmp_path):
    exp, backend = experiment(), FakeBackend()
    out = run(exp, tmp_path / "run", backend, max_new_attempts=4)
    assert out["status"] == "paused" and out["completed_rollouts"] == 4
    assert [row[0]["mode"] for row in backend.calls] == ["raw", "identity", "anchor", "matched_decoder"]
    assert all(row[1] == exp["cases"][0] for row in backend.calls)
    hashes = {p: sha256_file(p) for p in (tmp_path / "run").rglob("result.json")}
    with pytest.raises(ValueError, match="Incomplete"):
        load_results(exp, tmp_path / "run")
    second = FakeBackend()
    assert run(exp, tmp_path / "run", second)["status"] == "completed"
    assert len(second.calls) == 4
    assert all(sha256_file(p) == h for p, h in hashes.items())
    results = load_results(exp, tmp_path / "run")
    assert all(len(rows) == 2 for rows in results.values())
    assert results["control-12"][0]["result"]["hook_evidence"]["path"] == "native/matched_records.jsonl"
    assert inspect(exp, tmp_path / "run")["completed_rollouts"] == 8


def test_completed_and_zero_cap_never_load_model(tmp_path):
    exp = experiment()
    backend = FakeBackend()
    assert run(exp, tmp_path / "run", backend, max_new_attempts=0)["new_attempts"] == 0
    assert not backend.calls and not backend.validated
    run(exp, tmp_path / "run", backend)
    class NoBackend:
        def __getattr__(self, name):
            raise AssertionError(name)
    assert run(exp, tmp_path / "run", NoBackend())["new_attempts"] == 0


def test_identity_mismatch_stops_before_any_intervention(tmp_path):
    exp, backend = experiment(), FakeBackend(identity_change=True)
    with pytest.raises(ValueError, match="parity"):
        run(exp, tmp_path / "run", backend)
    assert [x[0]["mode"] for x in backend.calls] == ["raw", "identity"]
    with pytest.raises(ValueError, match="parity"):
        inspect(exp, tmp_path / "run")


@pytest.mark.parametrize("failure", [RuntimeError("simulator failed"), KeyboardInterrupt()])
def test_failed_or_interrupted_attempt_requires_reserved_explicit_retry(tmp_path, failure):
    exp = experiment(reserve=1)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        run(exp, tmp_path / "run", FakeBackend(fail=failure))
    state = inspect(exp, tmp_path / "run")
    assert state["attempts_used"] == 1 and state["completed_rollouts"] == 0
    with pytest.raises(ValueError, match="explicit retry"):
        run(exp, tmp_path / "run", FakeBackend())
    result = run(exp, tmp_path / "run", FakeBackend(), retry_incomplete=True)
    assert result["attempts_used"] == 9 and result["completed_rollouts"] == 8


def test_no_retry_reserve_and_wrong_approval_no_gpu(tmp_path):
    exp, backend = experiment(), FakeBackend()
    with pytest.raises(ValueError, match="Approved"):
        execute(exp, tmp_path / "run", backend, approved_hash="wrong")
    assert not backend.validated and not (tmp_path / "run").exists()
    with pytest.raises(RuntimeError):
        run(exp, tmp_path / "run", FakeBackend(fail=RuntimeError("bad")))
    with pytest.raises(ValueError, match="retry reserve"):
        run(exp, tmp_path / "run", backend, retry_incomplete=True)
    assert not backend.validated


@pytest.mark.parametrize("kind", ["num_actions", "success", "exception", "nan", "action_shape", "seed", "raw_hook"])
def test_invalid_outcomes_never_become_robot_failures(tmp_path, kind):
    def mutate(result, condition, directory):
        if kind == "num_actions": result["episode"]["num_actions"] = 2
        elif kind == "success": result["episode"]["success"] = 1
        elif kind == "exception": result["episode"]["caught_exception"] = "crashed"
        elif kind == "nan": result["actions"][0][0] = float("nan")
        elif kind == "action_shape": result["actions"][0].pop()
        elif kind == "seed": result["episode"]["seed"] += 1
        else: result["hook_metrics"] = {}
    exp = experiment()
    with pytest.raises(RuntimeError, match="no task failure"):
        run(exp, tmp_path / "run", FakeBackend(mutate=mutate))
    row = json.loads(next((tmp_path / "run").rglob("result.json")).read_text())
    assert row["status"] == "failed" and "result" not in row


@pytest.mark.parametrize("kind", ["mode", "control", "context", "sequence", "rows", "norm", "finite", "inactive", "summary", "false_ablation", "direction_hash"])
def test_perrow_hook_evidence_fail_closed(tmp_path, kind):
    def mutate(metrics, records, condition):
        if condition["mode"] != "matched_decoder":
            return
        row = records[0]
        if kind == "mode": row["mode"] = "anchor"
        elif kind == "control": row["control_feature_id"] = 13
        elif kind == "context": row["task_id"] += 1
        elif kind == "sequence": row["forward_index"] = 2
        elif kind == "rows": row["num_rows"] = 2
        elif kind == "norm": row["control_realized_norm"] = [2.]
        elif kind == "finite": row["finite"] = [False]
        elif kind == "inactive": row["anchor_active"] = [False]
        elif kind == "summary": metrics["active_rows"] = 6
        elif kind == "false_ablation": metrics["control_is_feature_ablation"] = True
        else: metrics["direction_binding_hash"] = "e" * 64
    exp, backend = experiment(), FakeBackend(hook_mutate=mutate)
    with pytest.raises(RuntimeError, match="no task failure"):
        run(exp, tmp_path / "run", backend)
    assert len(backend.calls) == 4
    state = inspect(exp, tmp_path / "run")
    assert state["completed_rollouts"] == 3 and state["attempts_used"] == 4


def test_hook_file_and_signed_receipt_tampering(tmp_path):
    exp = experiment()
    run(exp, tmp_path / "run", FakeBackend())
    file = next((tmp_path / "run" / "attempts" / "control-12").rglob("matched_records.jsonl"))
    original = file.read_text()
    file.write_text(original + "\n")
    with pytest.raises(ValueError, match="modified"):
        load_results(exp, tmp_path / "run")
    file.write_text(original)
    receipt = next((tmp_path / "run").rglob("result.json"))
    data = json.loads(receipt.read_text())
    data["result"]["episode"]["success"] = not data["result"]["episode"]["success"]
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect(exp, tmp_path / "run")


def test_unknown_root_attempt_and_symlink_entries_rejected(tmp_path):
    exp = experiment()
    run(exp, tmp_path / "run", FakeBackend(), max_new_attempts=1)
    extra = tmp_path / "run" / "unknown.txt"
    extra.write_text("x")
    with pytest.raises(ValueError, match="Unknown"):
        inspect(exp, tmp_path / "run")
    extra.unlink()
    attempt = next((tmp_path / "run").rglob("started.json")).parent
    extra = attempt / "unknown.txt"
    extra.write_text("x")
    with pytest.raises(ValueError, match="Unknown"):
        inspect(exp, tmp_path / "run")
    extra.unlink()
    extra.symlink_to(attempt / "started.json")
    with pytest.raises(ValueError, match="Symlink"):
        inspect(exp, tmp_path / "run")


def test_nonblocking_lock_and_readonly_missing_status(tmp_path):
    exp = experiment()
    assert inspect(exp, tmp_path / "missing")["status"] == "not_started"
    assert not (tmp_path / "missing").exists()
    root = tmp_path / "run"
    root.mkdir()
    with (root / ".execution.lock").open("w") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="locked"):
            run(exp, root, FakeBackend())


def test_actual_native_hook_evidence_roundtrip_with_cpu_simulator_fixture(fixture_runtime):
    from event_sae.research.head_matched.native import NativeBackend
    fixture = fixture_runtime
    exp = experiment()
    exp["cases"] = fixture.cases[:1]
    exp["total_rollouts"] = exp["max_attempts"] = 4
    exp["identity"] = fixture.identity
    for row in exp["conditions"]:
        if row["mode"] != "raw":
            row["feature_id"] = 0
        if row["mode"] == "matched_decoder":
            row["control_feature_id"] = 1
    backend = NativeBackend(fixture.cfg, fixture.identity)
    backend.validate_cases(exp["cases"])
    summaries = {}
    for row in exp["conditions"]:
        if row["mode"] != "raw":
            result = backend.run_case(row, exp["cases"][0], fixture.root / f"certificate-{row['mode']}")
            summaries[row["condition_id"]] = result["hook_metrics"]
    backend.close()
    exp["validation"] = {"details": {"hook_summaries": summaries}}
    exp["execution_hash"] = fingerprint({key: value for key, value in exp.items() if key != "execution_hash"})
    backend = NativeBackend(fixture.cfg, fixture.identity)
    try:
        assert run(exp, fixture.root / "actual-journal", backend)["completed_rollouts"] == 4
        assert len(load_results(exp, fixture.root / "actual-journal")) == 4
    finally:
        backend.close()
