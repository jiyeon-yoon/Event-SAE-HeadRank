"""Prepare/freeze/approve the isolated matched-direction diagnostic.

Only run_execution constructs a native backend. Preparation binds every local
input by content and requires clean committed source plus fresh matched-hook
validation; an inherited feature-removal certificate alone is insufficient.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, git_identity, read_json, sha256_file,
)
from event_sae.research.head_rank.workflow import REPO, safe_output
from .protocol import ANALYSIS, NORM_POLICY, SCOPE, direction_scores, integer, signed, verify_plan


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA256: {name}")


def _predictions(values, plan):
    names = [row["condition_id"] for row in plan["conditions"] if row["mode"] not in ("raw", "identity")]
    if not isinstance(values, list) or [row.get("condition_id") for row in values] != names:
        raise ValueError("Predictions differ from all five frozen direction conditions")
    for row in values:
        mean, tasks = row.get("mean_head_kl"), row.get("task_mean_head_kl")
        if (isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < 0
                or not isinstance(tasks, dict) or set(tasks) != {str(i) for i in range(10)}):
            raise ValueError("Prediction KL and ten task scores are required")
        for value in tasks.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Task KL must be finite and nonnegative")
        if not math.isclose(mean, math.fsum(tasks.values()) / 10, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Prediction mean does not reproduce equal-task scores")


def _certificates(plan, scores, validation, identity, implementation):
    signed(scores, "score_hash")
    signed(validation, "validation_hash")
    if scores.get("schema_version") != "head_matched_scores_v1" or validation.get("schema_version") != "head_matched_validation_v1":
        raise ValueError("Matched scores and new matched-hook validation are required")
    if scores.get("status") != "scored" or validation.get("status") != "passed":
        raise ValueError("Matched live validation has not passed")
    for document in (scores, validation):
        if (document.get("plan_hash") != plan["plan_hash"] or document.get("identity") != identity
                or document.get("implementation") != implementation or document.get("synthetic", False) is not False):
            raise ValueError("Plan/device/source differs between matched certificates")
        bindings = document.get("source_bindings")
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError("Matched certificate input content bindings are required")
    _predictions(scores.get("predictions"), plan)
    raw = scores.get("score_result")
    if (not isinstance(raw, dict) or raw.get("schema_version") != "head_rank_scores_v1"
            or scores.get("norm_policy") != NORM_POLICY or raw.get("control_count") != 4
            or raw.get("feature_ids") != [plan["anchor_feature_id"]]
            or raw.get("norm_match", {}).get("atol") != NORM_POLICY["atol"]
            or raw.get("norm_match", {}).get("rtol") != NORM_POLICY["rtol"]
            or raw.get("diagnostics", {}).get("num_norm_invalid_readouts") != 0
            or raw.get("diagnostics", {}).get("num_norm_invalid_controls") != 0
            or raw.get("control_feature_ids", {}).get(str(plan["anchor_feature_id"])) != plan["control_feature_ids"]):
        raise ValueError("Matched scoring lacks complete valid fixed-direction evidence")
    if direction_scores(raw, plan["anchor_feature_id"]) != scores["predictions"]:
        raise ValueError("Frozen predictions do not replay from query scores")
    metadata = raw.get("query_metadata", [])
    if (not metadata or raw["feature_scores"][0].get("num_readouts") != 7 * len(metadata)
            or raw["feature_scores"][0].get("num_readouts") != plan["source_readouts"]
            or fingerprint(metadata) != plan["source_query_metadata_hash"]):
        raise ValueError("Matched score query/readout counts differ")
    details = validation.get("details", {})
    expected_names = [row["condition_id"] for row in plan["conditions"]]
    if (details.get("schema_version") != "head_matched_runtime_v1" or details.get("status") != "passed"
            or details.get("synthetic") is not False or details.get("identity") != identity
            or details.get("conditions_hash") != fingerprint(plan["conditions"])
            or details.get("conditions") != expected_names or details.get("raw_identity_action_match") is not True
            or details.get("fixture_sha256") != validation["source_bindings"].get("runtime_inputs", {}).get("sha256")):
        raise ValueError("Matched runtime certificate body is absent, failed or unbound")
    observations = details.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Matched runtime certificate lacks actual forward observations")
    groups = {name: {} for name in expected_names}
    modes = {row["condition_id"]: row["mode"] for row in plan["conditions"]}
    for row in observations:
        if (not isinstance(row, dict) or row.get("condition_id") not in groups
                or row.get("mode") != modes[row["condition_id"]] or row.get("status") != "passed"
                or row.get("hidden_exact_equal") is not True or row.get("grouped_logits", {}).get("status") != "passed"):
            raise ValueError("Matched runtime observation failed or differs from conditions")
        call = integer(row.get("call_index"), "call_index")
        integer(row.get("row_count"), "row_count", 1)
        if type(row.get("active_anchor")) is not bool:
            raise ValueError("Actual anchor activity must be explicit")
        groups[row["condition_id"]].setdefault(call, []).append(row)
    call_count = len(groups[expected_names[0]])
    if not call_count or details.get("model_action_queries") != call_count * len(expected_names):
        raise ValueError("Matched runtime action query accounting differs")
    for name, calls in groups.items():
        if set(calls) != set(range(call_count)):
            raise ValueError("Runtime conditions did not use identical complete fixture calls")
        for rows in calls.values():
            if len(rows) != 7 or rows[0]["row_count"] <= 1 or any(row["row_count"] != 1 for row in rows[1:]):
                raise ValueError("Runtime lacks prefill plus six cached forward checks")
        if modes[name] in ("anchor", "matched_decoder") and not any(row["active_anchor"] for rows in calls.values() for row in rows):
            raise ValueError("Runtime never exercised an active planned intervention")
    action_hashes = details.get("action_trace_hashes", {})
    if set(action_hashes) != set(expected_names):
        raise ValueError("Runtime lacks every condition's action trace hash")
    for name, value in action_hashes.items():
        _digest(value, name)
    raw_name = next(name for name in expected_names if modes[name] == "raw")
    identity_name = next(name for name in expected_names if modes[name] == "identity")
    if action_hashes[raw_name] != action_hashes[identity_name]:
        raise ValueError("Runtime raw/identity action hashes differ")
    summaries, evidence = details.get("hook_summaries", {}), details.get("hook_evidence", {})
    edited_names = set(expected_names) - {raw_name}
    if set(summaries) != edited_names or set(evidence) != edited_names:
        raise ValueError("Runtime hook evidence is incomplete")
    for condition in plan["conditions"]:
        name = condition["condition_id"]
        if name == raw_name:
            continue
        summary = summaries[name]
        binding = summary.get("direction_binding", {})
        if (summary.get("schema_version") != "head_matched_hook_v1"
                or summary.get("direction_binding_hash") != fingerprint(binding)
                or binding.get("sae_checkpoint_hash") != identity.get("sae_checkpoint_hash")
                or binding.get("mode") != condition["mode"] or binding.get("anchor_feature_id") != condition["feature_id"]
                or summary.get("mode") != condition["mode"] or summary.get("feature_id") != condition["feature_id"]
                or summary.get("control_feature_id") != condition.get("control_feature_id")
                or summary.get("norm_invalid_rows") != 0 or summary.get("off_support_rows") != 0
                or summary.get("num_forwards") != call_count * 7):
            raise ValueError("Runtime hook summary is inconsistent with approved direction")
        if condition["mode"] == "identity" and summary.get("edited_rows") != 0:
            raise ValueError("Runtime identity edited hidden rows")
        if condition["mode"] == "matched_decoder":
            if binding.get("control_feature_id") != condition["control_feature_id"] or binding.get("sign") != "negative_decoder_unit":
                raise ValueError("Runtime control direction differs")
            _digest(binding.get("direction_fp32_sha256"), "runtime direction vector")
        path = evidence[name].get("path")
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Runtime evidence must use safe relative paths")
        _digest(evidence[name].get("sha256"), "runtime hook evidence")
        if validation["source_bindings"].get(f"evidence/{path}", {}).get("sha256") != evidence[name]["sha256"]:
            raise ValueError("Runtime hook file was not content-bound to validation")


def _validate_sources(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Execution input bindings are required")
    for name, source in sources.items():
        if not isinstance(source, dict) or not isinstance(source.get("path"), str) or not Path(source["path"]).is_absolute():
            raise ValueError(f"Execution input needs an absolute path: {name}")
        _digest(source.get("sha256"), name)


def verify_experiment(experiment):
    """Pure archival verification; does not require original paths or a GPU."""
    from .journal import _contract
    _contract(experiment)
    if (experiment.get("synthetic") is not False or experiment.get("model_download") is not False
            or experiment.get("scientific_scope") != SCOPE or experiment.get("analysis") != ANALYSIS
            or experiment.get("norm_policy") != NORM_POLICY or experiment.get("processor_preflight") != "passed_offline"):
        raise ValueError("Unsupported matched execution scope/analysis")
    plan = experiment.get("plan")
    verify_plan(plan)
    if (experiment.get("plan_hash") != plan["plan_hash"] or experiment.get("conditions") != plan["conditions"]
            or experiment.get("cases") != plan["cases"] or experiment.get("total_rollouts") != plan["total_rollouts"]):
        raise ValueError("Execution cases or conditions differ from frozen plan")
    reserve = integer(experiment.get("retry_allowance"), "retry_allowance")
    if experiment.get("max_attempts") != experiment["total_rollouts"] + reserve:
        raise ValueError("Frozen attempt/retry budget mismatch")
    cfg, identity, implementation = experiment.get("config"), experiment.get("identity"), experiment.get("implementation")
    if not isinstance(cfg, dict) or experiment.get("config_hash") != fingerprint(cfg):
        raise ValueError("Execution configuration binding mismatch")
    if (cfg["rollout"]["env_seed"] != plan["base_seed"] or cfg["rollout"]["alpha"] != 0
            or cfg["rollout"]["hook_start_step"] != 0 or cfg["scoring"]["alpha"] != 0
            or cfg["scoring"]["pair_batch_size"] != 1 or cfg["scoring"]["reduction_dtype"] != "float32"
            or cfg["scoring"]["negative_tolerance"] != 1e-6 or cfg["model"]["allow_download"] is not False
            or cfg["model"]["expected_live_hidden_dtype"] != "bfloat16"
            or cfg["rollout"].get("collect_activations") is not False
            or cfg["rollout"].get("save_trajectory_records") is not False):
        raise ValueError("Execution must retain pinned all-step BF16 reference protocol")
    if experiment["max_attempts"] > integer(cfg["rollout"]["max_total_rollouts"], "max_total_rollouts", 1):
        raise ValueError("Frozen execution exceeds explicit total attempt cap")
    if not isinstance(identity, dict) or not isinstance(implementation, dict) or not implementation:
        raise ValueError("Numerical and implementation identities are required")
    _certificates(plan, experiment.get("scores"), experiment.get("validation"), identity, implementation)
    if experiment.get("predictions") != experiment["scores"]["predictions"]:
        raise ValueError("Execution predictions changed after scoring")
    code = experiment.get("code_identity", {})
    if code.get("dirty") is not False or not re.fullmatch("[0-9a-f]{40}", code.get("commit", "")):
        raise ValueError("Execution requires a clean frozen Git revision")
    if experiment.get("policy_config_hash") != fingerprint(experiment.get("effective_policy")):
        raise ValueError("Native policy hash mismatch")
    _validate_sources(experiment.get("sources"))
    if not isinstance(experiment.get("output_root"), str) or not Path(experiment["output_root"]).is_absolute():
        raise ValueError("Absolute run output path required")
    sources = experiment["sources"]
    for label in ("config", "plan", "scores", "validation", "sae", "sae_config", "head_weights", "head_manifest", "base_eval_config"):
        if label not in sources:
            raise ValueError(f"Required execution content source is missing: {label}")
    for name, expected in plan["parent_content_hashes"].items():
        if sources.get(name, {}).get("sha256") != expected:
            raise ValueError(f"Execution model/SAE content differs from frozen parent: {name}")
    for label, certificate in (("scores", experiment["scores"]), ("validation", experiment["validation"])):
        for name, entry in _binding_files(certificate["source_bindings"]).items():
            if sources.get(f"{label}_input/{name}") != entry:
                raise ValueError("Embedded certificate source binding differs from execution inputs")
    if not any(key.startswith("snapshot/") and key.endswith(".safetensors") for key in sources):
        raise ValueError("Model content shard bindings required")
    if not any(key.startswith("runtime_inputs/") for key in sources):
        raise ValueError("Runtime validation input content bindings required")


def _binding_files(bindings, prefix=""):
    """Read nested certificate metadata, retaining every explicit file binding."""
    result = {}
    if isinstance(bindings, dict):
        if "path" in bindings or "sha256" in bindings:
            _validate_sources({prefix: bindings})
            result[prefix] = {"path": bindings["path"], "sha256": bindings["sha256"]}
        else:
            for name, value in bindings.items():
                result.update(_binding_files(value, f"{prefix}/{name}".strip("/")))
    elif isinstance(bindings, list):
        for number, value in enumerate(bindings):
            result.update(_binding_files(value, f"{prefix}/{number}".strip("/")))
    return result


def _check_sources(sources):
    _validate_sources(sources)
    for name, source in sources.items():
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"Execution source missing or changed: {name}")


def _run_path(experiment, *, contract_path=None, new=False):
    cfg = experiment["config"]
    protected = [entry["path"] for entry in experiment["sources"].values()]
    protected += [cfg["inputs"].get("local_model_snapshot"), cfg["inputs"].get("dense_dir"),
                  cfg["output"]["root_dir"], contract_path, REPO, REPO.parent / "Event-SAE-Head"]
    protected += experiment.get("protected_roots", [])
    target = assert_safe_output(experiment["output_root"], protected)
    original = Path(experiment["output_root"]).absolute()
    if any(path.is_symlink() for path in (original, *original.parents)):
        raise ValueError("Run journal cannot use symlink paths")
    if new and target.exists():
        raise FileExistsError("Choose a new matched run output directory")
    return target


def build_experiment(plan, scores, validation, *, cfg, identity, implementation, code_identity,
                     policy, sources, output_root, retry_allowance=0, protected_roots=()):
    doc = deepcopy({"schema_version": "head_matched_execution_v1", "plan": plan,
        "scores": scores, "validation": validation, "plan_hash": plan["plan_hash"],
        "conditions": plan["conditions"], "cases": plan["cases"], "predictions": scores["predictions"],
        "analysis": ANALYSIS, "norm_policy": NORM_POLICY, "total_rollouts": plan["total_rollouts"],
        "retry_allowance": retry_allowance, "max_attempts": plan["total_rollouts"] + retry_allowance,
        "config": cfg, "config_hash": fingerprint(cfg), "identity": identity, "implementation": implementation,
        "code_identity": code_identity, "effective_policy": policy, "policy_config_hash": fingerprint(policy),
        "sources": sources, "output_root": str(Path(output_root).resolve()), "protected_roots": list(protected_roots),
        "processor_preflight": "passed_offline",
        "scientific_scope": SCOPE, "synthetic": False, "model_download": False,
        "status": "prepared_not_executed"})
    doc["execution_hash"] = fingerprint(doc)
    verify_experiment(doc)
    return doc


def _processor_preflight(cfg):
    """CPU-only actual native processor load, with network access disabled."""
    from event_sae.research.output_head.runtime import _offline_environment
    from event_sae.research.head_rank.native import _base_policy
    _offline_environment()
    from event_sae.openvla.eval.model import get_processor
    processor = get_processor(_base_policy(cfg))
    if processor is None:
        raise ValueError("Native offline processor preflight did not return a processor")


def prepare_execution(config, plan_file, scores_file, validation_file, output, run_output,
                      *, max_rollouts=140, retry_allowance=0):
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import numerical_identity, load_parity, _sae_paths, _head_path
    from event_sae.research.head_rank.native import effective_policy
    from .workflow import implementation_identity, check_runtime_bindings
    integer(max_rollouts, "max_rollouts", 1)
    integer(retry_allowance, "retry_allowance")
    code = git_identity(REPO)
    if code.get("dirty") is not False:
        raise ValueError("Freeze reviewed source in a clean commit before paid execution; no automatic commit")
    plan, scores, validation = (read_json(path) for path in (plan_file, scores_file, validation_file))
    verify_plan(plan)
    if plan["total_rollouts"] + retry_allowance > max_rollouts:
        raise ValueError("Requested matched run and retry reserve exceed explicit rollout cap")
    cfg = load_config(config)
    cfg["rollout"]["max_total_rollouts"] = max_rollouts
    identity, implementation = numerical_identity(cfg), implementation_identity()
    _certificates(plan, scores, validation, identity, implementation)
    load_parity(cfg, identity)
    parent_sources = check_runtime_bindings(cfg, plan, config_path=config)
    _processor_preflight(cfg)
    policy = effective_policy(cfg)
    weights, sae_config = _sae_paths(cfg)
    head = _head_path(cfg)
    if head.is_file():
        head = head.parent
    paths = {"config": config, "plan": plan_file, "scores": scores_file, "validation": validation_file,
        "sae": weights, "sae_config": sae_config, "head_weights": head / "output_head.safetensors",
        "head_manifest": head / "output_head_manifest.json", "base_eval_config": cfg["rollout"]["base_eval_config"],
        "generation_manifest": cfg["inputs"]["generation_manifest"],
        "head_parity": Path(cfg["output"]["root_dir"]) / "validation/runtime_parity.json",
        "edit_parity": Path(cfg["output"]["root_dir"]) / "validation/edit_parity.json"}
    snapshot = Path(cfg["inputs"]["local_model_snapshot"])
    assets = sorted(path for path in snapshot.iterdir() if path.is_file() and
                    path.suffix in {".safetensors", ".json", ".py", ".model", ".txt"})
    if not any(path.suffix == ".safetensors" for path in assets):
        raise ValueError("Local model snapshot contains no safetensors")
    paths.update({f"snapshot/{path.name}": path for path in assets})
    runtime_path = cfg["inputs"].get("runtime_inputs")
    if not runtime_path:
        raise ValueError("Runtime validation inputs are required")
    runtime_path = Path(runtime_path)
    if runtime_path.is_file():
        paths[f"runtime_inputs/{runtime_path.name}"] = runtime_path
    elif runtime_path.is_dir():
        files = sorted(path for path in runtime_path.rglob("*") if path.is_file())
        if not files:
            raise ValueError("Runtime input directory is empty")
        paths.update({f"runtime_inputs/{path.relative_to(runtime_path)}": path for path in files})
    else:
        raise ValueError("Runtime validation inputs are missing")
    sources = {name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)} for name, path in paths.items()}
    for name, entry in parent_sources.items():
        if name in sources and sources[name] != entry:
            raise ValueError("Current parent content changed while preparing execution")
        sources[name] = entry
    for label, certificate in (("scores", scores), ("validation", validation)):
        bindings = _binding_files(certificate["source_bindings"])
        if not bindings:
            raise ValueError("Certificate must bind actual input files")
        _check_sources(bindings)
        sources.update({f"{label}_input/{name}": entry for name, entry in bindings.items()})
    _check_sources(sources)
    protected = [str(weights.parent.resolve()), str(snapshot.resolve()), str(head.resolve())]
    target = safe_output(output, [*(entry["path"] for entry in sources.values()), *protected, run_output, REPO])
    doc = build_experiment(plan, scores, validation, cfg=cfg, identity=identity, implementation=implementation,
        code_identity=code, policy=policy, sources=sources, output_root=run_output,
        retry_allowance=retry_allowance, protected_roots=protected)
    _run_path(doc, contract_path=target, new=True)
    if git_identity(REPO) != code or implementation_identity() != implementation:
        raise ValueError("Source changed during matched preparation")
    atomic_write_json(target, doc)
    return {"status": "prepared_not_executed", "experiment": str(target), "execution_hash": doc["execution_hash"],
            "code_revision": code["commit"], "conditions": len(doc["conditions"]),
            "total_rollouts": doc["total_rollouts"], "max_attempts": doc["max_attempts"],
            "new_rollouts": 0, "model_execution": False}


def status(experiment_path, *, run_root=None):
    from .journal import inspect
    doc = read_json(experiment_path)
    verify_experiment(doc)
    target = Path(run_root) if run_root is not None else _run_path(doc, contract_path=experiment_path)
    return inspect(doc, target)


def run_execution(experiment_path, *, approved_hash, expected_revision,
                  retry_incomplete=False, max_new_attempts=None):
    from .journal import execute, inspect
    from .workflow import implementation_identity
    doc = read_json(experiment_path)
    verify_experiment(doc)
    if not approved_hash or approved_hash != doc["execution_hash"]:
        raise ValueError("Explicit approval must match complete matched execution hash")
    if not expected_revision or expected_revision != doc["code_identity"]["commit"]:
        raise ValueError("Expected code revision differs from frozen matched execution")
    target = _run_path(doc, contract_path=experiment_path)
    state = inspect(doc, target)
    if state["status"] == "completed":
        return {**state, "new_rollouts": 0, "model_execution": False,
                "scientific_scope": "stored receipt/action verification only; runtime not rerun"}
    if git_identity(REPO) != doc["code_identity"] or implementation_identity() != doc["implementation"]:
        raise ValueError("Source revision/cleanliness/implementation changed")
    _check_sources(doc["sources"])
    from event_sae.research.output_head.workflow import numerical_identity, load_parity
    from event_sae.research.head_rank.native import effective_policy
    from .native import NativeBackend
    cfg = doc["config"]
    identity = numerical_identity(cfg)
    if identity != doc["identity"] or effective_policy(cfg) != doc["effective_policy"]:
        raise ValueError("Current device/model/policy differs from frozen matched execution")
    _certificates(doc["plan"], doc["scores"], doc["validation"], identity, doc["implementation"])
    load_parity(cfg, identity)
    backend = NativeBackend(cfg, identity)
    try:
        return execute(doc, target, backend, approved_hash=approved_hash,
                       retry_incomplete=retry_incomplete, max_new_attempts=max_new_attempts)
    finally:
        backend.close()
