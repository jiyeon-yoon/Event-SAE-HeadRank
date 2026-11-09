"""Condition-level, fail-closed execution for the team Baseline protocol.

One native call executes a whole ordered condition.  A partially completed
condition is never resumed by skipping cases: its global RNG/environment
history must be replayed from the beginning.  GPU work is opt-in and requires
both an approved plan hash and a successful, source-bound prefix smoke check.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import math
from pathlib import Path
import re
import subprocess
import time

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, read_json, sha256_file,
)

REPO = Path(__file__).resolve().parents[3]
PREFLIGHT_CASES = 2  # Only task0 trials 0/1; main raw/identity cover all 500.
PREFLIGHT_ROLLOUTS = 4 * PREFLIGHT_CASES + 2
PREFLIGHT_SCOPE = "first_task_two_case_prefix_smoke_not_full_suite_conformance"


def _signed(payload, field):
    return {**payload, field: fingerprint(payload)}


def _read_signed(path, field):
    value = read_json(path)
    if (not isinstance(value, dict) or value.get(field) != fingerprint(
            {key: item for key, item in value.items() if key != field})):
        raise ValueError(f"Invalid {field}: {path}")
    return value


def _code_identity():
    """Bind the clean revision and the actual implementation bytes."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO, text=True).strip()
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += sorted((REPO / "scripts/openvla").glob("*team500*.py"))
    files += [Path(__file__).with_name("baseline_source_manifest.json")]
    return {"commit": commit, "dirty": bool(dirty), "files": {
        str(path.relative_to(REPO)): sha256_file(path) for path in files}}


def _load_plan(path, scores_path=None):
    from .protocol import verify_plan, verify_plan_scores
    plan = read_json(path)
    verify_plan(plan)
    source = scores_path or plan.get("reference_bindings", {}).get("scores_path")
    if not isinstance(source, (str, Path)) or not str(source):
        raise ValueError("Frozen source scores must be available to verify actual Top-5 selection")
    if (plan.get("score_file_sha256") is None
            or sha256_file(source) != plan["score_file_sha256"]):
        raise ValueError("Source score file differs from the approved prediction plan")
    verify_plan_scores(plan, read_json(source))
    return plan


def _runtime_config(plan, path=None):
    if path is None:
        config = deepcopy(plan["runtime_config"])
    else:
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            config = read_json(path)
    if fingerprint(config) != plan["runtime_config_hash"]:
        raise ValueError("Runtime config differs from the frozen plan")
    return config


def _approval(plan, approved_plan_hash, expected_code_revision):
    if approved_plan_hash != plan["plan_hash"]:
        raise ValueError("Exact approved plan hash is required")
    current = _code_identity()
    if current["dirty"]:
        raise ValueError("Live execution requires a clean, committed implementation")
    if (not isinstance(expected_code_revision, str)
            or current["commit"] != expected_code_revision):
        raise ValueError("Expected code revision differs from current implementation")
    return current


def _root(path, protected=()):
    absolute = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ValueError("Execution journals cannot use symlink paths")
    value = assert_safe_output(absolute, protected)
    if value in (Path("/"), Path.home()):
        raise ValueError("Unsafe journal root")
    if value.exists() and not value.is_dir():
        raise ValueError("Journal root is not a directory")
    return value


def _config_inputs(config):
    return [config.get("sae_checkpoint"),
            config.get("baseline_config", {}).get("model", {}).get("checkpoint")]


@contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".execution.lock"
    if lock.is_symlink():
        raise ValueError("Symlink lock is not allowed")
    with lock.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Condition journal is busy/locked") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_once(path, value):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    atomic_write_json(path, value)


def _condition_id(condition):
    identifier = condition.get("condition_id")
    if not isinstance(identifier, str) or not re.fullmatch(
            r"raw|identity-[0-9]+|feature-[0-9]+", identifier):
        raise ValueError("Unsafe condition identifier")
    return identifier


def _native_call(config, condition, cases, output, baseline_root, *, reference=False,
                 expected_runtime_identity=None):
    from .native import run_condition
    return run_condition(config, condition, cases, output, baseline_root, reference=reference,
                         expected_runtime_identity=expected_runtime_identity)


def validate_receipt(receipt, condition, cases):
    """Strict stored-data checks in addition to the native runner's checks."""
    if not isinstance(receipt, dict) or receipt.get("status") != "completed":
        raise ValueError("Native condition did not complete")
    if receipt.get("condition") != condition:
        raise ValueError("Native condition identity differs from request")
    if receipt.get("case_manifest_hash") != fingerprint(cases):
        raise ValueError("Native ordered-case hash differs from request")
    runtime = receipt.get("runtime_identity")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("Missing actual runtime identity")
    episodes = receipt.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(cases):
        raise ValueError("Native episode count differs from frozen cases")
    for row, case in zip(episodes, cases):
        if not isinstance(row, dict):
            raise ValueError("Malformed native episode")
        for key in ("task_id", "task_episode_idx", "initial_state_sha256"):
            if row.get(key) != case[key]:
                raise ValueError(f"Native episode/order differs: {key}")
        if (type(row.get("task_id")) is not int
                or type(row.get("task_episode_idx")) is not int
                or type(row.get("success")) is not bool
                or "caught_exception" not in row
                or row["caught_exception"] is not None):
            raise ValueError("Episode has an exception or invalid outcome")
        actions, count = row.get("actions"), row.get("num_actions")
        if (type(count) is not int or count < 1 or not isinstance(actions, list)
                or count != len(actions)):
            raise ValueError("Empty or inconsistent native action trace")
        for action in actions:
            if not isinstance(action, list) or len(action) != 7:
                raise ValueError("Action must contain seven coordinates")
            if any(isinstance(x, bool) or not isinstance(x, (float, int))
                   or not math.isfinite(x) for x in action):
                raise ValueError("Nonfinite or invalid native action")
    if condition["mode"] != "raw":
        metrics = receipt.get("hook_metrics")
        if (not isinstance(metrics, dict)
                or metrics.get("feature_id") != condition["feature_id"]
                or metrics.get("alpha") != condition["alpha"]
                or type(metrics.get("num_forwards")) is not int
                or metrics["num_forwards"] < 1):
            raise ValueError("Missing or mismatched live intervention evidence")
        if condition["alpha"] == 0 and (
                metrics.get("post_intervention_active_feature_values") != 0
                or metrics.get("post_intervention_max_feature_activation") != 0):
            raise ValueError("Target feature was not fully suppressed")
        if condition["alpha"] == 1 and metrics.get("absolute_feature_delta") != 0:
            raise ValueError("Identity hook changed the target feature")
    elif receipt.get("hook_metrics") is not None:
        raise ValueError("Raw receipt must not have an intervention hook")
    fingerprint(receipt)  # Reject other non-JSON/nonfinite metadata as well.
    return receipt


def _traces(receipt):
    return [{key: row[key] for key in (
        "task_id", "task_episode_idx", "initial_state_sha256", "success",
        "num_actions", "actions")} for row in receipt["episodes"]]


def compare_traces(first, second):
    left, right = _traces(first), _traces(second)
    if len(left) != len(right) or fingerprint(left) != fingerprint(right):
        raise ValueError("Exact native actions, lengths or outcomes do not match")
    return {"status": "passed", "cases": len(left), "exact_action_trace": True,
            "trace_sha256": fingerprint(left)}


def _same_runtime(first, second):
    if first != second:
        differing = sorted(key for key in set(first) | set(second)
                           if first.get(key) != second.get(key))
        raise ValueError(f"Actual runtime changed: {differing}")


def _suppression_proof(receipt):
    metrics = receipt["hook_metrics"]
    active, delta = metrics.get("active_feature_values"), metrics.get("absolute_feature_delta")
    if (type(active) is not int or active <= 0
            or isinstance(delta, bool) or not isinstance(delta, (float, int))
            or not math.isfinite(delta) or delta <= 0):
        raise ValueError("Suppression smoke test must actually edit an active target feature")
    return {"status": "passed", "cases": 2, "hook_metrics": metrics}


def preflight(plan_path, output_root, baseline_root, runtime_config_path=None, *,
              execute=False, approved_plan_hash=None, expected_code_revision=None,
              max_rollouts=PREFLIGHT_ROLLOUTS, scores_path=None):
    """Run ten explicitly authorized prefix smoke episodes, not a random pilot.

    Each call uses only task0 trials 0/1, a legal prefix of the stock loop.
    Four raw/reference/repeat/identity conditions plus one alpha-zero check
    cost ten episodes. This does not test task transitions or establish full
    suite Baseline conformance. The separate main raw/identity gate still
    compares all 500 cases before any feature evaluation is permitted.
    """
    plan = _load_plan(plan_path, scores_path)
    config = _runtime_config(plan, runtime_config_path)
    if not execute:
        return {"status": "not_executed", "plan_hash": plan["plan_hash"],
                "preflight_rollouts": PREFLIGHT_ROLLOUTS, "model_execution": False,
                "claim_scope": PREFLIGHT_SCOPE,
                "required": "--execute --approved-plan-hash HASH --expected-code-revision COMMIT"}
    if type(max_rollouts) is not int or max_rollouts != PREFLIGHT_ROLLOUTS:
        raise ValueError("Prefix smoke preflight requires the exact explicit 10-rollout budget")
    code = _approval(plan, approved_plan_hash, expected_code_revision)
    root = _root(output_root, [plan_path, runtime_config_path, baseline_root, REPO,
                              *_config_inputs(config)])
    cases = plan["cases"][:PREFLIGHT_CASES]
    expected = [(0, trial) for trial in range(PREFLIGHT_CASES)]
    if [(c["task_id"], c["task_episode_idx"]) for c in cases] != expected:
        raise ValueError("Smoke preflight must use task0 trials 0/1 in the final plan order")
    raw = next(c for c in plan["conditions"] if c["mode"] == "raw")
    identity = next(c for c in plan["conditions"] if c["mode"] == "identity")
    smoke = next(c for c in plan["conditions"] if c["mode"] == "intervention"
                 and c["feature_id"] == identity["feature_id"])
    with _lock(root):
        if any(p.name != ".execution.lock" for p in root.iterdir()):
            raise FileExistsError("Preflight requires a fresh evidence directory")
        binding = {"schema_version": "team500_preflight_binding_v2",
                   "plan_hash": plan["plan_hash"], "runtime_config_hash": fingerprint(config),
                   "code_identity": code, "case_manifest_hash": fingerprint(plan["cases"]),
                   "prefix_manifest_hash": fingerprint(cases), "max_rollouts": max_rollouts,
                   "preflight_scope": PREFLIGHT_SCOPE}
        _write_once(root / "binding.json", binding)
        receipts = {}
        runtime = None
        for name, condition, selected, reference in (
                ("baseline-raw", raw, cases, True),
                ("adapter-raw", raw, cases, False),
                ("repeat-raw", raw, cases, False),
                ("identity", identity, cases, False),
                ("suppression-smoke", smoke, plan["cases"][:2], False)):
            if _code_identity() != code:
                raise ValueError("Implementation changed during preflight")
            directory = root / name
            receipt = _native_call(config, condition, selected, directory,
                                   baseline_root, reference=reference,
                                   expected_runtime_identity=runtime)
            validate_receipt(receipt, condition, selected)
            if runtime is None:
                runtime = receipt["runtime_identity"]
            else:
                _same_runtime(runtime, receipt["runtime_identity"])
            if receipt.get("synthetic") is not False:
                raise ValueError("Synthetic native output cannot certify a live runtime")
            signed = _signed(receipt, "receipt_hash")
            _write_once(root / f"{name}.json", signed)
            receipts[name] = signed
        checks = {
            "baseline_adapter": compare_traces(receipts["baseline-raw"], receipts["adapter-raw"]),
            "raw_repeat": compare_traces(receipts["adapter-raw"], receipts["repeat-raw"]),
            "raw_identity": compare_traces(receipts["adapter-raw"], receipts["identity"]),
            "suppression": _suppression_proof(receipts["suppression-smoke"]),
        }
        if _code_identity() != code:
            raise ValueError("Implementation changed during preflight")
        certificate = _signed({**binding, "schema_version": "team500_preflight_v2",
            "status": "passed", "synthetic": False, "runtime_identity": runtime,
            "checks": checks, "completed_rollouts": PREFLIGHT_ROLLOUTS,
            "receipt_hashes": {key: value["receipt_hash"] for key, value in receipts.items()},
            "claim_scope": PREFLIGHT_SCOPE},
            "preflight_hash")
        _write_once(root / "preflight.json", certificate)
    return {"status": "passed", "preflight": str(root / "preflight.json"),
            "preflight_hash": certificate["preflight_hash"], "model_execution": True,
            "completed_rollouts": PREFLIGHT_ROLLOUTS, "claim_scope": PREFLIGHT_SCOPE}


def _certificate(path, plan, config, code):
    certificate = _read_signed(path, "preflight_hash")
    if (certificate.get("schema_version") != "team500_preflight_v2"
            or certificate.get("status") != "passed" or certificate.get("synthetic") is not False
            or certificate.get("plan_hash") != plan["plan_hash"]
            or certificate.get("runtime_config_hash") != fingerprint(config)
            or certificate.get("case_manifest_hash") != fingerprint(plan["cases"])
            or certificate.get("prefix_manifest_hash") != fingerprint(plan["cases"][:PREFLIGHT_CASES])
            or certificate.get("completed_rollouts") != PREFLIGHT_ROLLOUTS
            or certificate.get("max_rollouts") != PREFLIGHT_ROLLOUTS
            or certificate.get("preflight_scope") != PREFLIGHT_SCOPE
            or certificate.get("claim_scope") != PREFLIGHT_SCOPE
            or certificate.get("code_identity") != code):
        raise ValueError("Preflight is missing, stale, synthetic or bound to a different experiment")
    directory = Path(path).resolve().parent
    expected = {"baseline-raw", "adapter-raw", "repeat-raw", "identity", "suppression-smoke"}
    if set(certificate.get("receipt_hashes", {})) != expected:
        raise ValueError("Preflight receipt set is incomplete")
    receipts = {}
    raw = next(c for c in plan["conditions"] if c["mode"] == "raw")
    identity = next(c for c in plan["conditions"] if c["mode"] == "identity")
    smoke = next(c for c in plan["conditions"] if c["mode"] == "intervention"
                 and c["feature_id"] == identity["feature_id"])
    for name, digest in certificate["receipt_hashes"].items():
        receipt = _read_signed(directory / f"{name}.json", "receipt_hash")
        if receipt["receipt_hash"] != digest:
            raise ValueError("Preflight native evidence was changed")
        condition = smoke if name == "suppression-smoke" else identity if name == "identity" else raw
        selected = plan["cases"][:2] if name == "suppression-smoke" else plan["cases"][:PREFLIGHT_CASES]
        validate_receipt(receipt, condition, selected)
        if receipt.get("synthetic") is not False:
            raise ValueError("Synthetic preflight evidence cannot authorize live work")
        _same_runtime(certificate["runtime_identity"], receipt["runtime_identity"])
        receipts[name] = receipt
    checks = {
        "baseline_adapter": compare_traces(receipts["baseline-raw"], receipts["adapter-raw"]),
        "raw_repeat": compare_traces(receipts["adapter-raw"], receipts["repeat-raw"]),
        "raw_identity": compare_traces(receipts["adapter-raw"], receipts["identity"]),
        "suppression": _suppression_proof(receipts["suppression-smoke"]),
    }
    if checks != certificate.get("checks"):
        raise ValueError("Preflight checks do not match the actual stored native evidence")
    return certificate


def _attempts(plan, root, binding):
    completed, incomplete = {}, []
    used = 0
    directory = root / "attempts"
    if not directory.exists():
        return completed, incomplete, used
    allowed = {_condition_id(c): c for c in plan["conditions"]}
    for cid in sorted(directory.iterdir()):
        if cid.is_symlink() or not cid.is_dir() or cid.name not in allowed:
            raise ValueError("Unrecognized condition evidence directory")
        attempts = sorted(cid.iterdir())
        for number, attempt in enumerate(attempts, 1):
            if cid.name in completed:
                raise ValueError("A completed condition cannot have a later attempt")
            if (attempt.is_symlink() or not attempt.is_dir()
                    or attempt.name != f"attempt-{number:06d}"):
                raise ValueError("Condition attempts are malformed or non-contiguous")
            started = _read_signed(attempt / "started.json", "started_hash")
            if (started.get("journal_binding_hash") != fingerprint(binding)
                    or started.get("condition") != allowed[cid.name]
                    or started.get("attempt") != number):
                raise ValueError("Condition attempt has a different provenance binding")
            used += len(plan["cases"])
            result_path = attempt / "result.json"
            failure_path = attempt / "failure.json"
            if failure_path.exists():
                failure = _read_signed(failure_path, "failure_hash")
                if (failure.get("started_hash") != started["started_hash"]
                        or failure.get("status") != "failed_not_a_task_failure"
                        or result_path.exists()):
                    raise ValueError("Invalid condition failure evidence")
            if result_path.exists():
                result = _read_signed(result_path, "result_hash")
                if (result.get("started_hash") != started["started_hash"]
                        or result.get("condition") != allowed[cid.name]):
                    raise ValueError("Condition result differs from its attempt")
                validate_receipt(result["receipt"], allowed[cid.name], plan["cases"])
                if cid.name in completed:
                    raise ValueError("Multiple completed outcomes for one condition are not allowed")
                completed[cid.name] = result
            else:
                incomplete.append((cid.name, attempt))
    if used > binding["max_rollout_attempts"]:
        raise ValueError("Attempt journal exceeds its declared budget")
    return completed, incomplete, used


def _journal_binding(plan, config, code, certificate, retry_budget_conditions):
    if type(retry_budget_conditions) is not int or not 0 <= retry_budget_conditions <= 10:
        raise ValueError("retry_budget_conditions must be an explicit integer between 0 and 10")
    return {"schema_version": "team500_execution_binding_v1", "plan_hash": plan["plan_hash"],
            "runtime_config_hash": fingerprint(config), "code_identity": code,
            "case_manifest_hash": fingerprint(plan["cases"]),
            "preflight_hash": certificate["preflight_hash"],
            "runtime_identity": certificate["runtime_identity"],
            "retry_budget_conditions": retry_budget_conditions,
            "max_rollout_attempts": plan["total_rollouts"] + retry_budget_conditions * len(plan["cases"]),
            "retry_policy": "explicit_whole_condition_restart_no_case_skipping"}


def run_experiment(plan_path, output_root, baseline_root, *, runtime_config_path=None,
                   preflight_path=None, approved_plan_hash=None, expected_code_revision=None,
                   execute=False, max_new_conditions=None, retry_incomplete=False,
                   retry_budget_conditions=0, scores_path=None):
    plan = _load_plan(plan_path, scores_path)
    config = _runtime_config(plan, runtime_config_path)
    if max_new_conditions is not None and (
            type(max_new_conditions) is not int or max_new_conditions < 1):
        raise ValueError("max_new_conditions must be a positive integer")
    if not execute:
        return {"status": "not_executed", "plan_hash": plan["plan_hash"],
                "total_rollouts": plan["total_rollouts"], "conditions": len(plan["conditions"]),
                "model_execution": False, "preflight_required": True,
                "preflight_rollouts": PREFLIGHT_ROLLOUTS,
                "full_controls_included_rollouts": 1000,
                "required": "--execute --approved-plan-hash HASH --expected-code-revision COMMIT --preflight FILE"}
    code = _approval(plan, approved_plan_hash, expected_code_revision)
    if preflight_path is None:
        raise ValueError("A successful current-runtime preflight certificate is required")
    certificate = _certificate(preflight_path, plan, config, code)
    binding = _journal_binding(plan, config, code, certificate, retry_budget_conditions)
    root = _root(output_root, [plan_path, runtime_config_path, baseline_root, preflight_path, REPO,
                              *_config_inputs(config)])
    with _lock(root):
        manifest = root / "experiment.json"
        saved = {"binding": binding, "plan": plan}
        if manifest.exists():
            if read_json(manifest) != saved:
                raise ValueError("Journal belongs to another plan/runtime/source/retry budget")
        else:
            if any(p.name != ".execution.lock" for p in root.iterdir()):
                raise ValueError("Existing evidence directory has no matching experiment manifest")
            _write_once(manifest, saved)
        completed, incomplete, used = _attempts(plan, root, binding)
        live_runtime = binding["runtime_identity"]
        for result in completed.values():
            _same_runtime(live_runtime, result["receipt"]["runtime_identity"])
        raw = next(c for c in plan["conditions"] if c["mode"] == "raw")
        identity = next(c for c in plan["conditions"] if c["mode"] == "identity")
        raw_id, identity_id = raw["condition_id"], identity["condition_id"]
        if identity_id in completed and raw_id not in completed:
            raise ValueError("Identity cannot complete before raw")
        if any(c["mode"] == "intervention" and c["condition_id"] in completed
               for c in plan["conditions"]) and identity_id not in completed:
            raise ValueError("Feature outcomes exist before the full raw/identity gate")
        if identity_id in completed:
            compare_traces(completed[raw_id]["receipt"], completed[identity_id]["receipt"])
        new_conditions = 0
        for condition in plan["conditions"]:
            cid = _condition_id(condition)
            if cid in completed:
                continue
            if max_new_conditions is not None and new_conditions >= max_new_conditions:
                break
            prior = [p for key, p in incomplete if key == cid]
            if prior and not retry_incomplete:
                raise ValueError("Incomplete condition requires --retry-incomplete; no cases will be skipped")
            if used + len(plan["cases"]) > binding["max_rollout_attempts"]:
                raise ValueError("No reserved whole-condition retry budget remains")
            if condition["mode"] == "intervention":
                if raw_id not in completed or identity_id not in completed:
                    raise ValueError("Full raw and identity conditions must pass before any feature")
                compare_traces(completed[raw_id]["receipt"], completed[identity_id]["receipt"])
            if _code_identity() != code:
                raise ValueError("Implementation changed during execution")
            base = root / "attempts" / cid
            base.mkdir(parents=True, exist_ok=True)
            number = len(list(base.iterdir())) + 1
            directory = base / f"attempt-{number:06d}"
            directory.mkdir()
            started = _signed({"journal_binding_hash": fingerprint(binding),
                "condition": condition, "attempt": number,
                "reserved_rollout_attempts": len(plan["cases"])}, "started_hash")
            _write_once(directory / "started.json", started)
            used += len(plan["cases"])
            tick = time.perf_counter()
            try:
                receipt = _native_call(config, condition, plan["cases"], directory / "native", baseline_root,
                                       expected_runtime_identity=live_runtime)
                validate_receipt(receipt, condition, plan["cases"])
                if receipt.get("synthetic") is not False:
                    raise ValueError("Synthetic outputs cannot become completed live outcomes")
                _same_runtime(live_runtime, receipt["runtime_identity"])
                if _code_identity() != code:
                    raise ValueError("Implementation changed during execution")
                if condition["mode"] == "identity":
                    compare_traces(completed[raw_id]["receipt"], receipt)
                result = _signed({"condition": condition, "started_hash": started["started_hash"],
                    "status": "completed", "receipt": receipt,
                    "elapsed_seconds": time.perf_counter() - tick}, "result_hash")
                _write_once(directory / "result.json", result)
            except Exception as exc:
                _write_once(directory / "failure.json", _signed({"started_hash": started["started_hash"],
                    "status": "failed_not_a_task_failure", "error_type": type(exc).__name__,
                    "error": str(exc)}, "failure_hash"))
                raise
            completed[cid] = result
            new_conditions += 1
        done = len(completed) == len(plan["conditions"])
        return {"status": "completed" if done else "paused", "plan_hash": plan["plan_hash"],
                "completed_conditions": len(completed), "conditions": len(plan["conditions"]),
                "completed_rollouts": len(completed) * len(plan["cases"]),
                "total_rollouts": plan["total_rollouts"], "reserved_rollout_attempts_used": used,
                "max_rollout_attempts": binding["max_rollout_attempts"],
                "new_conditions": new_conditions, "model_execution": new_conditions > 0,
                "output_root": str(root)}


def load_completed(plan_path, output_root, scores_path=None):
    """Read-only verification of stored evidence. Does not import a simulator."""
    plan = _load_plan(plan_path, scores_path)
    root = _root(output_root)
    saved = read_json(root / "experiment.json")
    if saved.get("plan") != plan or saved.get("binding", {}).get("plan_hash") != plan["plan_hash"]:
        raise ValueError("Stored journal differs from plan")
    completed, incomplete, used = _attempts(plan, root, saved["binding"])
    runtime = saved["binding"]["runtime_identity"]
    for value in completed.values():
        _same_runtime(runtime, value["receipt"]["runtime_identity"])
    raw = next(c["condition_id"] for c in plan["conditions"] if c["mode"] == "raw")
    identity = next(c["condition_id"] for c in plan["conditions"] if c["mode"] == "identity")
    if identity in completed:
        if raw not in completed:
            raise ValueError("Identity has no raw reference")
        compare_traces(completed[raw]["receipt"], completed[identity]["receipt"])
    return plan, completed, incomplete, used, saved["binding"]


def status(plan_path, output_root, scores_path=None):
    plan, completed, incomplete, used, binding = load_completed(plan_path, output_root, scores_path)
    return {"status": "completed" if len(completed) == len(plan["conditions"]) else "incomplete",
            "plan_hash": plan["plan_hash"], "completed_conditions": len(completed),
            "conditions": len(plan["conditions"]), "completed_rollouts": len(completed) * 500,
            "total_rollouts": plan["total_rollouts"],
            "pending_rollouts": plan["total_rollouts"] - len(completed) * 500,
            "reserved_rollout_attempts_used": used, "max_rollout_attempts": binding["max_rollout_attempts"],
            "incomplete_condition_attempts": sum(cid not in completed for cid, _ in incomplete),
            "historical_incomplete_attempts": len(incomplete), "model_execution": False,
            "partial_condition_cases_not_counted_as_completed": True}
