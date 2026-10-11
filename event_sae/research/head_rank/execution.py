"""Dependency-free, fail-closed case journal for approved robot experiments.

The backend owns native model/simulator execution. This module owns accounting,
receipts, exact raw/identity gating, and safe resume. A simulator exception is
never converted into a task failure. An interrupted attempt consumes budget.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import re

from event_sae.research.output_head.provenance import (
    atomic_write_json, fingerprint, read_json, sha256_file,
)


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _verify_contract(experiment):
    """Lightweight boundary guard; root's verify_experiment has fuller rules."""
    if not isinstance(experiment, dict) or experiment.get("schema_version") != "head_rank_execution_v1":
        raise ValueError("Unsupported execution schema")
    expected = _digest(experiment.get("execution_hash"), "execution_hash")
    if fingerprint({k: v for k, v in experiment.items() if k != "execution_hash"}) != expected:
        raise ValueError("Execution hash mismatch")
    _digest(experiment.get("plan_hash"), "plan_hash")
    if experiment.get("code_identity", {}).get("dirty") is not False:
        raise ValueError("Execution requires frozen clean code identity")
    conditions, cases = experiment.get("conditions"), experiment.get("cases")
    if not isinstance(conditions, list) or not conditions or not isinstance(cases, list) or not cases:
        raise ValueError("Execution requires conditions and cases")
    modes, identifiers = [], set()
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("Malformed condition")
        mode, fid = condition.get("mode"), condition.get("feature_id")
        alpha = _number(condition.get("alpha"), "condition alpha")
        if mode == "raw":
            wanted = "raw"
            if fid is not None or alpha != 1.:
                raise ValueError("Raw condition must have no feature and alpha=1")
        elif mode in ("identity", "intervention"):
            _int(fid, "condition feature_id")
            wanted = f"{'identity' if mode == 'identity' else 'feature'}-{fid}"
            if fid >= 32768 or alpha != (1. if mode == "identity" else 0.):
                raise ValueError("Only identity or full-removal conditions are supported")
        else:
            raise ValueError("Unknown condition mode")
        if condition.get("condition_id") != wanted or wanted in identifiers:
            raise ValueError("Condition ID/mode mismatch or duplicate")
        identifiers.add(wanted)
        modes.append(mode)
    if modes.count("raw") != 1 or modes.count("identity") != 1 or "intervention" not in modes:
        raise ValueError("Exactly one raw, one identity and at least one feature are required")
    identities, states = set(), set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("suite"), str) or not case["suite"]:
            raise ValueError("Malformed case suite")
        for key in ("task_id", "task_episode_idx", "seed"):
            _int(case.get(key), f"case {key}")
        _digest(case.get("initial_state_sha256"), "initial_state_sha256")
        key = (case["suite"], case["task_id"], case["task_episode_idx"])
        state = (case["suite"], case["task_id"], case["initial_state_sha256"])
        if key in identities or state in states:
            raise ValueError("Duplicate case or initial state")
        identities.add(key)
        states.add(state)
    total = len(conditions) * len(cases)
    if _int(experiment.get("total_rollouts"), "total_rollouts", 1) != total:
        raise ValueError("total_rollouts differs from frozen condition/case product")
    if _int(experiment.get("max_attempts"), "max_attempts", 1) < total:
        raise ValueError("max_attempts cannot be below total_rollouts")


def _no_symlink(path):
    path = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Symlink paths are not allowed in an execution journal: {path}")
    return path.resolve()


def _root(path):
    root = _no_symlink(path)
    if root in (Path("/"), Path.home()):
        raise ValueError("Filesystem/home roots cannot be execution journals")
    if root.exists() and not root.is_dir():
        raise ValueError("Execution output must be a directory")
    return root


@contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = _no_symlink(root / ".execution.lock")
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Execution journal is busy/locked: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _signed(payload, field):
    return {**payload, field: fingerprint(payload)}


def _read_signed(path, field):
    _no_symlink(path)
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get(field) != fingerprint({k: v for k, v in payload.items() if k != field}):
        raise ValueError(f"Journal hash mismatch: {path}")
    return payload


def _binding(experiment, condition, case, attempt):
    return {"execution_hash": experiment["execution_hash"], "plan_hash": experiment["plan_hash"],
            "condition": condition, "case": case, "attempt_index": attempt}


def _check_binding(payload, binding):
    if any(payload.get(key) != value for key, value in binding.items()):
        raise ValueError("Case receipt/attempt provenance mismatch")


def _validate_result(result, condition, case, directory):
    if not isinstance(result, dict) or not isinstance(result.get("episode"), dict):
        raise ValueError("Backend must return an episode dictionary")
    result = deepcopy(result)
    episode = result["episode"]
    for key in ("suite", "task_id", "task_episode_idx", "initial_state_sha256", "seed"):
        if episode.get(key) != case[key] or (key in ("task_id", "task_episode_idx", "seed") and type(episode.get(key)) is not int):
            raise ValueError(f"Backend episode differs from requested case: {key}")
    if type(episode.get("success")) is not bool or "caught_exception" not in episode or episode["caught_exception"] is not None:
        raise ValueError("Episode success must be boolean and caught_exception must be explicitly null")
    count = _int(episode.get("num_actions"), "num_actions", 1)
    actions = result.get("actions")
    if not isinstance(actions, list) or len(actions) != count:
        raise ValueError("Actions must match episode num_actions")
    for action in actions:
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError("Every native action must have seven coordinates")
        for value in action:
            _number(value, "action coordinate")
    metrics, evidence = result.get("hook_metrics"), result.get("hook_evidence")
    if condition["mode"] == "raw":
        if metrics is not None or evidence is not None:
            raise ValueError("Raw condition must not contain intervention hooks")
    else:
        if not isinstance(metrics, dict) or not isinstance(evidence, dict):
            raise ValueError("Edited conditions require hook metrics and file evidence")
        if type(metrics.get("feature_id")) is not int or metrics["feature_id"] != condition["feature_id"]:
            raise ValueError("Hook feature_id differs from condition")
        if _number(metrics.get("alpha"), "hook alpha") != condition["alpha"]:
            raise ValueError("Hook alpha differs from condition")
        forwards = _int(metrics.get("num_forwards"), "hook num_forwards", 1)
        steps = _int(metrics.get("num_environment_steps"), "hook num_environment_steps", 1)
        active = _int(metrics.get("active_feature_values"), "active_feature_values")
        post = _int(metrics.get("post_intervention_active_feature_values"), "post active values")
        maximum = _number(metrics.get("max_feature_activation"), "max_feature_activation", 0)
        post_max = _number(metrics.get("post_intervention_max_feature_activation"), "post max activation", 0)
        delta = _number(metrics.get("absolute_feature_delta"), "absolute_feature_delta", 0)
        if condition["alpha"] == 0 and (post != 0 or post_max != 0):
            raise ValueError("Full-removal hook did not set feature activation to zero")
        if condition["alpha"] == 1 and (post != active or post_max != maximum or delta != 0):
            raise ValueError("Identity hook changed feature activations")
        if steps != count or forwards < steps:
            raise ValueError("Hook must cover every policy action")
        relative = evidence.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Hook evidence path is required")
        path = Path(relative)
        if not path.is_absolute():
            path = directory / path
        path = _no_symlink(path)
        try:
            relative = path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("Hook evidence must stay inside its attempt directory") from exc
        if not path.is_file() or sha256_file(path) != _digest(evidence.get("sha256"), "hook sha256"):
            raise ValueError("Hook evidence is missing or its hash differs")
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) != steps:
            raise ValueError("Hook evidence record count differs from executed action count")
        seen_steps, previous_forward = set(), 0
        for position, row in enumerate(records):
            if not isinstance(row, dict) or any(row.get(key) != value for key, value in {
                "feature_id": condition["feature_id"], "alpha": condition["alpha"],
                "task_id": case["task_id"], "task_episode_idx": case["task_episode_idx"],
            }.items()):
                raise ValueError("Hook evidence condition/case differs")
            step = _int(row.get("step_in_episode"), "hook step")
            forward = _int(row.get("first_forward_idx"), "first_forward_idx", 1)
            if (step != position or step in seen_steps or not previous_forward < forward <= forwards
                    or (position == 0 and forward != 1)):
                raise ValueError("Hook evidence step/forward sequence is invalid")
            seen_steps.add(step)
            previous_forward = forward
        result["hook_evidence"]["path"] = str(relative)
    fingerprint(result)  # Reject any other non-JSON/non-finite payload metadata.
    return result


def _journal_experiment(root, experiment, *, create):
    path = _no_symlink(root / "experiment.json")
    if path.exists():
        if read_json(path) != experiment:
            raise ValueError("Output journal belongs to a different/modified experiment")
    elif create:
        if any(child.name != ".execution.lock" for child in root.iterdir()):
            raise ValueError("Unrelated files or orphan attempts without their frozen experiment")
        atomic_write_json(path, experiment)
    else:
        raise ValueError("Missing frozen experiment journal")


def _scan(experiment, root):
    """Validate every attempt, including failed history; never choose a best run."""
    completed, attempts = {}, {}
    base = _no_symlink(root / "attempts")
    allowed_conditions = {row["condition_id"] for row in experiment["conditions"]}
    allowed_cases = {fingerprint(row) for row in experiment["cases"]}
    if base.exists():
        for child in base.iterdir():
            _no_symlink(child)
            if not child.is_dir() or child.name not in allowed_conditions:
                raise ValueError("Unknown condition entry in execution journal")
            for case_path in child.iterdir():
                _no_symlink(case_path)
                if not case_path.is_dir() or case_path.name not in allowed_cases:
                    raise ValueError("Unknown case entry in execution journal")
    for condition in experiment["conditions"]:
        for case in experiment["cases"]:
            key = (condition["condition_id"], fingerprint(case))
            parent = base / key[0] / key[1]
            directories = sorted(parent.iterdir()) if parent.exists() else []
            attempts[key] = len(directories)
            for number, directory in enumerate(directories, 1):
                _no_symlink(directory)
                if not directory.is_dir() or directory.name != f"attempt-{number:06d}":
                    raise ValueError("Attempt names must be contiguous and immutable")
                binding = _binding(experiment, condition, case, number)
                started = _read_signed(directory / "started.json", "started_hash")
                _check_binding(started, binding)
                if started.get("schema_version") != "head_rank_case_started_v1" or started.get("status") != "started":
                    raise ValueError("Malformed attempt start record")
                result_path = directory / "result.json"
                if not result_path.exists():
                    continue
                receipt = _read_signed(result_path, "receipt_hash")
                _check_binding(receipt, binding)
                if receipt.get("schema_version") != "head_rank_case_receipt_v1":
                    raise ValueError("Unsupported case receipt schema")
                if receipt.get("status") == "completed":
                    if key in completed or number != len(directories):
                        raise ValueError("Completed cases cannot be retried or superseded")
                    normalized = _validate_result(receipt.get("result"), condition, case, directory)
                    if normalized != receipt["result"]:
                        raise ValueError("Stored evidence paths must be attempt-relative")
                    completed[key] = receipt
                elif receipt.get("status") != "failed" or not isinstance(receipt.get("error"), dict):
                    raise ValueError("Unrecognized attempt result; not a task failure")
    if sum(attempts.values()) > experiment["max_attempts"]:
        raise ValueError("Journal exceeds approved attempt budget")
    return completed, attempts


def _identity_gate(experiment, completed, *, require_complete):
    raw = next(row for row in experiment["conditions"] if row["mode"] == "raw")
    identity = next(row for row in experiment["conditions"] if row["mode"] == "identity")
    ready = True
    for case in experiment["cases"]:
        key = fingerprint(case)
        a, b = completed.get((raw["condition_id"], key)), completed.get((identity["condition_id"], key))
        if a is None or b is None:
            ready = False
            continue
        ar, br = a["result"], b["result"]
        if ar["actions"] != br["actions"] or ar["episode"]["success"] != br["episode"]["success"]:
            raise ValueError("Raw/identity exact action or outcome parity failed; feature runs are forbidden")
    if require_complete and not ready:
        raise ValueError("All raw/identity cases must pass before feature runs")
    return ready


def _load(experiment, root):
    _journal_experiment(root, experiment, create=False)
    completed, attempts = _scan(experiment, root)
    if any(condition["mode"] == "intervention" and any((condition["condition_id"], fingerprint(case)) in completed
           for case in experiment["cases"]) for condition in experiment["conditions"]):
        _identity_gate(experiment, completed, require_complete=True)
    else:
        _identity_gate(experiment, completed, require_complete=False)
    return completed, attempts


def load_completed_results(experiment, output_root):
    """Return condition-ID → all validated receipts in frozen case order.

No model is constructed. Missing/failed cases are errors, not task failures.
Receipts contain ``result.episode``, ``result.actions`` and hook evidence.
"""
    _verify_contract(experiment)
    root = _root(output_root)
    if not root.is_dir():
        raise ValueError("Execution output directory does not exist")
    with _lock(root):
        completed, _ = _load(experiment, root)
        if len(completed) != experiment["total_rollouts"]:
            raise ValueError("Experiment is incomplete; cannot analyze missing attempts as failures")
        _identity_gate(experiment, completed, require_complete=True)
        return {condition["condition_id"]: [completed[(condition["condition_id"], fingerprint(case))]
                for case in experiment["cases"]] for condition in experiment["conditions"]}


def inspect_experiment(experiment, output_root):
    """Validate journal state on CPU without treating corruption as incomplete.

    A nonexistent output is not created. This is useful before constructing a
    native backend: a fully completed journal needs no GPU/parity rerun merely
    to report completion. It does not certify a new simulator execution.
    """
    _verify_contract(experiment)
    root = _root(output_root)
    initial = {"status": "not_started", "completed_rollouts": 0,
               "total_rollouts": experiment["total_rollouts"], "attempts_used": 0,
               "pending_rollouts": experiment["total_rollouts"],
               "model_execution": False, "execution_hash": experiment["execution_hash"]}
    if not root.exists():
        return initial
    with _lock(root):
        if not (root / "experiment.json").exists():
            if any(child.name != ".execution.lock" for child in root.iterdir()):
                raise ValueError("Unrelated files or orphan attempts without their frozen experiment")
            return initial
        completed, attempts = _load(experiment, root)
        pending = experiment["total_rollouts"] - len(completed)
        return {"status": "completed" if pending == 0 else "incomplete",
                "completed_rollouts": len(completed), "total_rollouts": experiment["total_rollouts"],
                "attempts_used": sum(attempts.values()), "pending_rollouts": pending,
                "incomplete_attempt_cases": sum(count > 0 and key not in completed for key, count in attempts.items()),
                "model_execution": False, "execution_hash": experiment["execution_hash"]}


def execute_experiment(experiment, output_root, backend, *, approved_hash,
                       retry_incomplete=False, max_new_attempts=None):
    """Execute/resume without overwriting a case or silently spending retries."""
    _verify_contract(experiment)
    if approved_hash != experiment["execution_hash"]:
        raise ValueError("Approved execution hash is missing or stale")
    if type(retry_incomplete) is not bool:
        raise ValueError("retry_incomplete must be boolean")
    if max_new_attempts is not None:
        _int(max_new_attempts, "max_new_attempts")
    root = _root(output_root)
    with _lock(root):
        _journal_experiment(root, experiment, create=True)
        completed, attempts = _load(experiment, root)
        new_attempts, backend_validated = 0, False

        def summary(status):
            return {"status": status, "execution_hash": experiment["execution_hash"],
                    "completed_rollouts": len(completed), "total_rollouts": experiment["total_rollouts"],
                    "attempts_used": sum(attempts.values()), "max_attempts": experiment["max_attempts"],
                    "new_attempts": new_attempts, "model_execution": new_attempts > 0,
                    "output_root": str(root)}

        controls = [next(row for row in experiment["conditions"] if row["mode"] == mode)
                    for mode in ("raw", "identity")]
        features = [row for row in experiment["conditions"] if row["mode"] == "intervention"]
        # A two-attempt smoke is one complete matched control pair. All control
        # pairs must still pass before the first feature intervention is run.
        jobs = ([(condition, case) for case in experiment["cases"] for condition in controls]
                + [(condition, case) for condition in features for case in experiment["cases"]])
        for condition, case in jobs:
            if condition["mode"] == "intervention":
                _identity_gate(experiment, completed, require_complete=True)
            key = (condition["condition_id"], fingerprint(case))
            if key in completed:
                continue
            if attempts[key] and not retry_incomplete:
                raise ValueError("Incomplete/failed attempt requires explicit retry_incomplete and a frozen retry reserve")
            if sum(attempts.values()) >= experiment["max_attempts"]:
                raise ValueError("Approved total attempt budget exhausted")
            if max_new_attempts is not None and new_attempts >= max_new_attempts:
                return summary("paused")
            # Reserve enough attempts for every still-unrun required case.
            pending = experiment["total_rollouts"] - len(completed)
            if sum(attempts.values()) + pending > experiment["max_attempts"]:
                raise ValueError("No approved retry reserve remains; remaining cases would exceed the total budget")
            if not backend_validated:
                backend.validate_cases(deepcopy(experiment["cases"]))
                backend_validated = True
            attempt = attempts[key] + 1
            directory = root / "attempts" / key[0] / key[1] / f"attempt-{attempt:06d}"
            _no_symlink(directory)
            directory.mkdir(parents=True, exist_ok=False)
            binding = _binding(experiment, condition, case, attempt)
            atomic_write_json(directory / "started.json", _signed({
                "schema_version": "head_rank_case_started_v1", "status": "started", **binding}, "started_hash"))
            attempts[key] += 1
            new_attempts += 1
            try:
                result = backend.run_case(deepcopy(condition), deepcopy(case), directory)
                result = _validate_result(result, condition, case, directory)
            except Exception as exc:
                atomic_write_json(directory / "result.json", _signed({
                    "schema_version": "head_rank_case_receipt_v1", "status": "failed", **binding,
                    "error": {"type": type(exc).__name__, "message": str(exc)[:4000]},
                }, "receipt_hash"))
                raise RuntimeError(f"Case attempt failed; no task failure was recorded: {exc}") from exc
            receipt = _signed({"schema_version": "head_rank_case_receipt_v1", "status": "completed",
                               **binding, "result": result}, "receipt_hash")
            atomic_write_json(directory / "result.json", receipt)
            completed[key] = receipt
            if condition["mode"] == "identity":
                _identity_gate(experiment, completed, require_complete=False)
        return summary("completed")
