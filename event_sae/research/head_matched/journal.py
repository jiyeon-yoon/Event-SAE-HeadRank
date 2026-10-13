"""Fail-closed, resumable journal for local norm-matched direction diagnostics.

This journal never authorizes GPU work by itself: workflow verifies the frozen
code, current runtime certificate and explicit budget before constructing the
backend. Simulator or numerical failures consume an attempt, not an outcome.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from event_sae.research.head_rank.execution import (
    _binding, _check_binding, _digest, _int, _lock, _no_symlink, _number,
    _read_signed, _root, _signed,
)
from event_sae.research.output_head.provenance import (
    atomic_write_json, fingerprint, read_json, sha256_file,
)


def _contract(experiment):
    if not isinstance(experiment, dict) or experiment.get("schema_version") != "head_matched_execution_v1":
        raise ValueError("Unsupported matched execution schema")
    expected = _digest(experiment.get("execution_hash"), "execution_hash")
    if fingerprint({k: v for k, v in experiment.items() if k != "execution_hash"}) != expected:
        raise ValueError("Execution hash mismatch")
    _digest(experiment.get("plan_hash"), "plan_hash")
    conditions, cases = experiment.get("conditions"), experiment.get("cases")
    if not isinstance(conditions, list) or not conditions or not isinstance(cases, list) or not cases:
        raise ValueError("Conditions and cases are required")
    ids, modes, anchors = set(), [], set()
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("Malformed matched condition")
        name = condition.get("condition_id")
        if (not isinstance(name, str) or not name or name in ids
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name)):
            raise ValueError("Condition IDs must be unique safe path components")
        ids.add(name)
        mode, alpha, fid = condition.get("mode"), _number(condition.get("alpha"), "alpha"), condition.get("feature_id")
        if mode == "raw":
            if fid is not None or alpha != 1 or condition.get("control_feature_id") is not None:
                raise ValueError("Raw condition may not apply an edit")
        elif mode in ("identity", "anchor", "matched_decoder"):
            _int(fid, "feature_id")
            if fid >= 32768 or alpha != (1 if mode == "identity" else 0):
                raise ValueError("Unsupported anchor/identity alpha or feature")
            anchors.add(fid)
            control = condition.get("control_feature_id")
            if mode == "matched_decoder":
                _int(control, "control_feature_id")
                if control == fid or control >= 32768:
                    raise ValueError("Matched decoder must use a distinct valid direction")
            elif control is not None:
                raise ValueError("Only matched decoder conditions may name a control direction")
        else:
            raise ValueError("Unsupported matched condition mode")
        modes.append(mode)
    if any(modes.count(mode) != 1 for mode in ("raw", "identity", "anchor")) or not modes.count("matched_decoder") or len(anchors) != 1:
        raise ValueError("One anchor, raw, identity and at least one matched decoder are required")
    controls = [row["control_feature_id"] for row in conditions if row["mode"] == "matched_decoder"]
    if len(controls) != len(set(controls)):
        raise ValueError("Duplicate matched decoder direction")
    keys, states = set(), set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("suite"), str) or not case["suite"]:
            raise ValueError("Malformed evaluation case")
        for key in ("task_id", "task_episode_idx", "seed"):
            _int(case.get(key), key)
        _digest(case.get("initial_state_sha256"), "initial_state_sha256")
        key = case["suite"], case["task_id"], case["task_episode_idx"]
        state = case["suite"], case["task_id"], case["initial_state_sha256"]
        if key in keys or state in states:
            raise ValueError("Duplicate case or initial state")
        keys.add(key)
        states.add(state)
    total = len(cases) * len(conditions)
    if _int(experiment.get("total_rollouts"), "total_rollouts", 1) != total:
        raise ValueError("Rollout total differs from cases times conditions")
    if _int(experiment.get("max_attempts"), "max_attempts", 1) < total:
        raise ValueError("Attempt budget cannot be below rollout total")


def _close(a, b):
    # The live tensor diagnostics are FP32; serialized norms are exact values
    # of those floats, but subtraction here happens in Python FP64.
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


def _validate_hook_evidence(metrics, records, condition, case, action_count):
    """Independently recompute every row's norm gate from recorded norms."""
    binding = metrics.get("direction_binding")
    if not isinstance(binding, dict) or fingerprint(binding) != _digest(metrics.get("direction_binding_hash"), "direction hash"):
        raise ValueError("Direction binding is missing or modified")
    for key, expected in {"mode": condition["mode"], "anchor_feature_id": condition["feature_id"],
            "support": "current_original_forward_anchor_activation",
            "norm_scope": "per_original_forward_row_same_incoming_hidden"}.items():
        if binding.get(key) != expected:
            raise ValueError("Hook direction binding differs from condition")
    _digest(binding.get("sae_checkpoint_hash"), "direction SAE hash")
    if condition["mode"] == "matched_decoder":
        if binding.get("control_feature_id") != condition["control_feature_id"] or binding.get("sign") != "negative_decoder_unit":
            raise ValueError("Decoder direction identity/sign differs")
        _digest(binding.get("direction_fp32_sha256"), "direction vector hash")
    elif any(key in binding for key in ("control_feature_id", "sign", "direction_fp32_sha256")):
        raise ValueError("Anchor/identity binding unexpectedly names a control")
    policy = condition.get("norm_policy")
    if not isinstance(policy, dict) or set(policy) != {"atol", "rtol"}:
        raise ValueError("Frozen per-row norm policy is required")
    atol, rtol = (_number(policy[key], key, 0) for key in ("atol", "rtol"))
    expected_common = {"condition_id": condition["condition_id"], "mode": condition["mode"],
        "feature_id": condition["feature_id"], "control_feature_id": condition.get("control_feature_id"),
        "norm_policy": policy, "encoder_grouping": "original_forward"}
    if (metrics.get("schema_version") != "head_matched_hook_v1"
            or any(metrics.get(key) != value for key, value in expected_common.items())
            or _number(metrics.get("alpha"), "hook alpha") != condition["alpha"]
            or metrics.get("cross_trajectory_norm_match") is not False
            or metrics.get("control_is_feature_ablation") is not False):
        raise ValueError("Matched hook metrics do not describe the frozen intervention")
    forwards = _int(metrics.get("num_forwards"), "num_forwards", 1)
    if not isinstance(records, list) or len(records) != forwards:
        raise ValueError("Hook forward evidence count mismatch")
    if forwards != 7 * action_count:
        raise ValueError("Every seven-token policy action must have seven hook forwards")
    counts = {"num_rows": 0, "active_rows": 0, "edited_rows": 0}
    maxima = {name: 0. for name in ("pair_norm_abs_error", "pair_norm_rel_error", "anchor_cast_abs_error", "control_cast_abs_error")}
    arrays = ("anchor_active", "intended_norm", "anchor_realized_norm", "control_realized_norm",
              "control_pre_cast_norm", "pair_norm_abs_error", "pair_norm_rel_error",
              "anchor_cast_abs_error", "control_cast_abs_error", "pre_cast_abs_error", "eligible", "off_support", "finite")
    for position, record in enumerate(records):
        if (not isinstance(record, dict) or record.get("schema_version") != "head_matched_forward_v1"
                or any(record.get(key) != value for key, value in expected_common.items())
                or record.get("direction_binding_hash") != metrics["direction_binding_hash"]
                or record.get("norm_scope") != "per_original_forward_row_same_incoming_hidden"
                or record.get("cross_trajectory_norm_match") is not False
                or record.get("task_id") != case["task_id"] or type(record.get("task_id")) is not int
                or record.get("task_episode_idx") != case["task_episode_idx"] or type(record.get("task_episode_idx")) is not int
                or _int(record.get("step_in_episode"), "hook step") != position // 7
                or _int(record.get("forward_index"), "hook forward", 1) != position + 1):
            raise ValueError("Hook evidence condition/context/forward sequence differs")
        n = _int(record.get("num_rows"), "num_rows", 1)
        if position % 7 and n != 1:
            raise ValueError("Cached generation forwards must have exactly one row")
        if any(not isinstance(record.get(name), list) or len(record[name]) != n for name in arrays):
            raise ValueError("Hook per-row norm evidence has malformed lengths")
        if _int(record.get("norm_invalid_rows"), "norm_invalid_rows") or _int(record.get("off_support_rows"), "off_support_rows"):
            raise ValueError("Hook records invalid or off-support edits")
        for row in range(n):
            for name in ("anchor_active", "eligible", "off_support", "finite"):
                if type(record[name][row]) is not bool:
                    raise ValueError("Hook mask must be boolean")
            if not record["eligible"][row] or not record["finite"][row] or record["off_support"][row]:
                raise ValueError("Ineligible/nonfinite/off-support norm row")
            numeric = {name: _number(record[name][row], name, 0) for name in arrays
                       if name not in ("anchor_active", "eligible", "off_support", "finite")}
            intended, anchor, actual, precast = (numeric[name] for name in
                ("intended_norm", "anchor_realized_norm", "control_realized_norm", "control_pre_cast_norm"))
            recomputed = {"pair_norm_abs_error": abs(actual - anchor),
                "pair_norm_rel_error": abs(actual - anchor) / anchor if anchor else 0.,
                "anchor_cast_abs_error": abs(anchor - intended), "control_cast_abs_error": abs(actual - intended),
                "pre_cast_abs_error": abs(precast - intended)}
            if any(not _close(numeric[name], value) for name, value in recomputed.items()):
                raise ValueError("Norm error evidence differs from independently computed values")
            if (recomputed["pre_cast_abs_error"] > 1e-6 + 1e-5 * intended
                    or recomputed["anchor_cast_abs_error"] > atol + rtol * intended
                    or recomputed["control_cast_abs_error"] > atol + rtol * intended
                    or recomputed["pair_norm_abs_error"] > atol + rtol * anchor):
                raise ValueError("Same-state per-row norm tolerance exceeded")
            if not record["anchor_active"][row] and (intended or anchor or actual):
                raise ValueError("Inactive anchor row was changed")
            if condition["mode"] == "identity" and (intended or anchor or actual or precast):
                raise ValueError("Identity hook changed hidden rows")
            counts["num_rows"] += 1
            counts["active_rows"] += int(record["anchor_active"][row])
            counts["edited_rows"] += int(actual > 0)
            for name in maxima:
                maxima[name] = max(maxima[name], numeric[name])
    if any(_int(metrics.get(name), name) != value for name, value in counts.items()):
        raise ValueError("Hook summary row counts differ from evidence")
    if _int(metrics.get("norm_invalid_rows"), "norm_invalid_rows") or _int(metrics.get("off_support_rows"), "off_support_rows"):
        raise ValueError("Hook summary contains invalid rows")
    if any(not _close(_number(metrics.get(f"max_{name}"), name, 0), value) for name, value in maxima.items()):
        raise ValueError("Hook summary maxima differ from evidence")


def _hook(result, condition, case, directory, experiment):
    """Validate numerical evidence using the new hook's strict schema."""
    metrics, evidence = result.get("hook_metrics"), result.get("hook_evidence")
    if condition["mode"] == "raw":
        if metrics is not None or evidence is not None:
            raise ValueError("Raw condition contains a hook")
        return
    if not isinstance(metrics, dict) or not isinstance(evidence, dict):
        raise ValueError("Edited conditions require hook metrics and evidence")
    name = evidence.get("path")
    if not isinstance(name, str) or not name:
        raise ValueError("Hook evidence path is missing")
    path = Path(name)
    if not path.is_absolute():
        path = directory / path
    path = _no_symlink(path)
    try:
        relative = path.relative_to(directory)
    except ValueError as exc:
        raise ValueError("Hook evidence is outside its attempt directory") from exc
    if not path.is_file() or sha256_file(path) != _digest(evidence.get("sha256"), "hook sha256"):
        raise ValueError("Hook evidence is missing or modified")
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    _validate_hook_evidence(metrics, records, condition, case, result["episode"]["num_actions"])
    expected_sae = experiment.get("identity", {}).get("sae_checkpoint_hash")
    certified = experiment.get("validation", {}).get("details", {}).get("hook_summaries", {}).get(condition["condition_id"], {})
    if (not expected_sae or metrics["direction_binding"].get("sae_checkpoint_hash") != expected_sae
            or not certified or metrics["direction_binding"] != certified.get("direction_binding")
            or metrics["direction_binding_hash"] != certified.get("direction_binding_hash")):
        raise ValueError("Live direction/SAE differs from current-device matched certificate")
    result["hook_evidence"]["path"] = str(relative)


def _result(result, condition, case, directory, experiment):
    if not isinstance(result, dict) or not isinstance(result.get("episode"), dict):
        raise ValueError("Backend must return an episode dictionary")
    result = deepcopy(result)
    episode = result["episode"]
    for key in ("suite", "task_id", "task_episode_idx", "initial_state_sha256", "seed"):
        if episode.get(key) != case[key] or (key in ("task_id", "task_episode_idx", "seed") and type(episode.get(key)) is not int):
            raise ValueError(f"Backend episode differs from frozen case: {key}")
    if type(episode.get("success")) is not bool or "caught_exception" not in episode or episode["caught_exception"] is not None:
        raise ValueError("Outcome requires boolean success and explicitly null caught_exception")
    count = _int(episode.get("num_actions"), "num_actions", 1)
    actions = result.get("actions")
    if not isinstance(actions, list) or len(actions) != count:
        raise ValueError("Action count differs from completed episode")
    for vector in actions:
        if not isinstance(vector, list) or len(vector) != 7:
            raise ValueError("Actions must have seven coordinates")
        for value in vector:
            _number(value, "action coordinate")
    _hook(result, condition, case, directory, experiment)
    fingerprint(result)
    return result


def _journal(root, experiment, *, create):
    allowed = {".execution.lock", "experiment.json", "attempts"}
    for entry in root.iterdir():
        _no_symlink(entry)
        if entry.name not in allowed:
            raise ValueError("Unknown entry in matched journal root")
    path = root / "experiment.json"
    if path.exists():
        if read_json(path) != experiment:
            raise ValueError("Journal belongs to a different or modified experiment")
    elif create:
        if any(entry.name != ".execution.lock" for entry in root.iterdir()):
            raise ValueError("Orphan attempts without frozen experiment")
        atomic_write_json(path, experiment)
    else:
        raise ValueError("Frozen experiment journal is missing")


def _scan(experiment, root):
    completed, attempts = {}, {}
    base = _no_symlink(root / "attempts")
    condition_ids = {row["condition_id"] for row in experiment["conditions"]}
    case_ids = {fingerprint(row) for row in experiment["cases"]}
    if base.exists():
        if not base.is_dir():
            raise ValueError("Attempts must be a directory")
        for child in base.iterdir():
            _no_symlink(child)
            if not child.is_dir() or child.name not in condition_ids:
                raise ValueError("Unknown condition in matched journal")
            for entry in child.iterdir():
                _no_symlink(entry)
                if not entry.is_dir() or entry.name not in case_ids:
                    raise ValueError("Unknown case in matched journal")
    for condition in experiment["conditions"]:
        for case in experiment["cases"]:
            key = condition["condition_id"], fingerprint(case)
            parent = base / key[0] / key[1]
            entries = sorted(parent.iterdir()) if parent.exists() else []
            attempts[key] = len(entries)
            for number, directory in enumerate(entries, 1):
                _no_symlink(directory)
                if not directory.is_dir() or directory.name != f"attempt-{number:06d}":
                    raise ValueError("Attempt directories must be contiguous and immutable")
                for child in directory.iterdir():
                    _no_symlink(child)
                    if child.name not in {"started.json", "result.json", "native"}:
                        raise ValueError("Unknown entry in matched attempt")
                binding = _binding(experiment, condition, case, number)
                started = _read_signed(directory / "started.json", "started_hash")
                _check_binding(started, binding)
                if started.get("schema_version") != "head_matched_case_started_v1" or started.get("status") != "started":
                    raise ValueError("Invalid matched attempt start")
                path = directory / "result.json"
                if not path.exists():
                    continue
                receipt = _read_signed(path, "receipt_hash")
                _check_binding(receipt, binding)
                if receipt.get("schema_version") != "head_matched_case_receipt_v1":
                    raise ValueError("Invalid matched case receipt schema")
                if receipt.get("status") == "completed":
                    if key in completed or number != len(entries):
                        raise ValueError("A completed case cannot be retried or replaced")
                    checked = _result(receipt.get("result"), condition, case, directory, experiment)
                    if checked != receipt["result"]:
                        raise ValueError("Stored evidence paths must be attempt-relative")
                    completed[key] = receipt
                elif receipt.get("status") != "failed" or not isinstance(receipt.get("error"), dict):
                    raise ValueError("Unknown attempt result")
    if sum(attempts.values()) > experiment["max_attempts"]:
        raise ValueError("Journal exceeds approved attempt budget")
    return completed, attempts


def _identity(experiment, completed, case, *, required):
    controls = {row["mode"]: row for row in experiment["conditions"] if row["mode"] in ("raw", "identity")}
    key = fingerprint(case)
    raw = completed.get((controls["raw"]["condition_id"], key))
    identity = completed.get((controls["identity"]["condition_id"], key))
    if raw is None or identity is None:
        if required:
            raise ValueError("This case requires complete raw/identity controls before intervention")
        return False
    a, b = raw["result"], identity["result"]
    if a["actions"] != b["actions"] or a["episode"]["success"] != b["episode"]["success"]:
        raise ValueError("Raw/identity exact action or outcome parity failed")
    return True


def _load(experiment, root):
    _journal(root, experiment, create=False)
    completed, attempts = _scan(experiment, root)
    for case in experiment["cases"]:
        has_intervention = any((row["condition_id"], fingerprint(case)) in completed
                               for row in experiment["conditions"] if row["mode"] in ("anchor", "matched_decoder"))
        _identity(experiment, completed, case, required=has_intervention)
    return completed, attempts


def load_results(experiment, root):
    """Return condition-ID -> signed receipts in frozen case order, CPU only."""
    _contract(experiment)
    root = _root(root)
    if not root.is_dir():
        raise ValueError("Execution output does not exist")
    with _lock(root):
        completed, _ = _load(experiment, root)
        if len(completed) != experiment["total_rollouts"]:
            raise ValueError("Incomplete experiment cannot be analyzed as task failures")
        return {condition["condition_id"]: [completed[(condition["condition_id"], fingerprint(case))]
                for case in experiment["cases"]] for condition in experiment["conditions"]}


def inspect(experiment, root):
    _contract(experiment)
    root = _root(root)
    initial = {"status": "not_started", "execution_hash": experiment["execution_hash"],
               "completed_rollouts": 0, "total_rollouts": experiment["total_rollouts"],
               "attempts_used": 0, "pending_rollouts": experiment["total_rollouts"], "model_execution": False}
    if not root.exists():
        return initial
    with _lock(root):
        if not (root / "experiment.json").exists():
            if any(entry.name != ".execution.lock" for entry in root.iterdir()):
                raise ValueError("Orphan or unknown journal files")
            return initial
        completed, attempts = _load(experiment, root)
        pending = experiment["total_rollouts"] - len(completed)
        return {**initial, "status": "completed" if pending == 0 else "incomplete",
                "completed_rollouts": len(completed), "attempts_used": sum(attempts.values()),
                "pending_rollouts": pending,
                "incomplete_attempt_cases": sum(count > 0 and key not in completed for key, count in attempts.items())}


def execute(experiment, root, backend, *, approved_hash, retry_incomplete=False, max_new_attempts=None):
    """Run each matched case raw -> identity -> anchor -> decoder controls.

The first four attempts therefore exercise all hook modes on one shared case.
Every later case independently passes exact raw/identity before intervention.
"""
    _contract(experiment)
    if approved_hash != experiment["execution_hash"]:
        raise ValueError("Approved execution hash is missing or stale")
    if type(retry_incomplete) is not bool:
        raise ValueError("retry_incomplete must be boolean")
    if max_new_attempts is not None:
        _int(max_new_attempts, "max_new_attempts")
    root = _root(root)
    with _lock(root):
        _journal(root, experiment, create=True)
        completed, attempts = _load(experiment, root)
        new_attempts, validated = 0, False

        def summary(status):
            return {"status": status, "execution_hash": experiment["execution_hash"],
                    "completed_rollouts": len(completed), "total_rollouts": experiment["total_rollouts"],
                    "attempts_used": sum(attempts.values()), "max_attempts": experiment["max_attempts"],
                    "new_attempts": new_attempts, "model_execution": new_attempts > 0, "output_root": str(root)}

        order = [next(row for row in experiment["conditions"] if row["mode"] == mode)
                 for mode in ("raw", "identity", "anchor")]
        order.extend(row for row in experiment["conditions"] if row["mode"] == "matched_decoder")
        for case in experiment["cases"]:
            for condition in order:
                key = condition["condition_id"], fingerprint(case)
                if key in completed:
                    continue
                if max_new_attempts is not None and new_attempts >= max_new_attempts:
                    return summary("paused")
                if condition["mode"] in ("anchor", "matched_decoder"):
                    _identity(experiment, completed, case, required=True)
                if attempts[key] and not retry_incomplete:
                    raise ValueError("Incomplete/failed attempt requires explicit retry_incomplete")
                pending = experiment["total_rollouts"] - len(completed)
                if sum(attempts.values()) + pending > experiment["max_attempts"]:
                    raise ValueError("No approved retry reserve remains")
                if sum(attempts.values()) >= experiment["max_attempts"]:
                    raise ValueError("Approved attempt budget exhausted")
                if not validated:
                    backend.validate_cases(deepcopy(experiment["cases"]))
                    validated = True
                number = attempts[key] + 1
                directory = root / "attempts" / key[0] / key[1] / f"attempt-{number:06d}"
                _no_symlink(directory)
                directory.mkdir(parents=True, exist_ok=False)
                binding = _binding(experiment, condition, case, number)
                atomic_write_json(directory / "started.json", _signed({
                    "schema_version": "head_matched_case_started_v1", "status": "started", **binding}, "started_hash"))
                attempts[key] += 1
                new_attempts += 1
                try:
                    outcome = _result(backend.run_case(deepcopy(condition), deepcopy(case), directory), condition, case, directory, experiment)
                except Exception as exc:
                    atomic_write_json(directory / "result.json", _signed({
                        "schema_version": "head_matched_case_receipt_v1", "status": "failed", **binding,
                        "error": {"type": type(exc).__name__, "message": str(exc)[:4000]}}, "receipt_hash"))
                    raise RuntimeError(f"Case attempt failed; no task failure was recorded: {exc}") from exc
                receipt = _signed({"schema_version": "head_matched_case_receipt_v1", "status": "completed",
                                   **binding, "result": outcome}, "receipt_hash")
                atomic_write_json(directory / "result.json", receipt)
                completed[key] = receipt
                if condition["mode"] == "identity":
                    _identity(experiment, completed, case, required=True)
        return summary("completed")
