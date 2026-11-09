#!/usr/bin/env python3
"""Isolated whole-condition driver for an external, byte-pinned Baseline.

This file intentionally does not import the research checkout's event_sae.
The reference module is loaded by filename and contains only stdlib code.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


def _reference():
    file = Path(__file__).resolve().parents[2] / "event_sae/research/team500/reference.py"
    spec = importlib.util.spec_from_file_location("_team500_reference_contract", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_once(path, payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Result publication is all-or-nothing and never overwrites an old receipt.
    temporary = path.with_suffix(".incomplete.json")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def collect_episodes(outcome, cases, contract):
    """Validate stock runner output without converting exceptions to failures."""
    root = Path(outcome.run_dir).resolve()
    path = Path(outcome.episode_results_path).resolve()
    if path != root / "episode_results.jsonl":
        raise ValueError("Stock runner episode evidence is outside its run directory")
    rows = _read_rows(path)
    actions = json.loads((root / "actions.json").read_text())
    episodes = []
    for row in rows:
        vectors = actions.get(row.get("task_description"), {}).get(str(row.get("task_episode_idx")))
        episodes.append({key: row.get(key) for key in (
            "task_id", "task_episode_idx", "initial_state_sha256", "success", "num_actions",
            "caught_exception")})
        episodes[-1]["actions"] = vectors
    contract.validate_episode_rows(episodes, cases)
    return episodes, {"run_dir": str(root), "episode_results": str(path),
                      "episode_results_sha256": contract.sha256_file(path),
                      "actions_sha256": contract.sha256_file(root / "actions.json")}


def collect_hook_evidence(run_dir, condition, episodes, metrics, contract):
    """Bind stock hook records and prove coverage of every executed step.

    Per-step records prove scheduling/targeting, while the byte-pinned hook's
    accumulated post-activation metrics cover all actually executed forwards
    and all flattened token rows. Generation can end early at EOS, so no
    assumption of exactly seven model forwards per action is imposed here.
    """
    total_steps = sum(row["num_actions"] for row in episodes)
    if (type(metrics.get("num_environment_steps")) is not int
            or metrics["num_environment_steps"] != total_steps
            or type(metrics.get("num_forwards")) is not int
            or metrics["num_forwards"] < total_steps
            or type(metrics.get("num_tokens")) is not int
            or metrics["num_tokens"] < metrics["num_forwards"]):
        raise ValueError("Hook aggregate coverage differs from executed policy steps/tokens")
    filename = f"intervene_feat{condition['feature_id']}_alpha{float(condition['alpha'])}_records.jsonl"
    path = Path(run_dir) / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Stock hook per-step evidence is missing")
    count, previous_forward = 0, 0
    with path.open() as stream:
        for episode_num, episode in enumerate(episodes, 1):
            for step in range(episode["num_actions"]):
                line = stream.readline()
                if not line.strip():
                    raise ValueError("Hook per-step evidence is incomplete")
                row = json.loads(line)
                expected = {"episode_num": episode_num, "task_id": episode["task_id"],
                            "task_episode_idx": episode["task_episode_idx"],
                            "step_in_episode": step, "feature_id": condition["feature_id"],
                            "alpha": float(condition["alpha"])}
                if any(type(row.get(key)) is not int or row[key] != value for key, value in expected.items()
                       if key != "alpha") or row.get("alpha") != expected["alpha"]:
                    raise ValueError("Hook target or per-step order differs from executed episodes")
                first_forward = row.get("first_forward_idx")
                if (type(first_forward) is not int or first_forward <= previous_forward
                        or first_forward > metrics["num_forwards"]):
                    raise ValueError("Hook forward evidence is duplicated or out of order")
                previous_forward = first_forward
                count += 1
        if stream.read().strip():
            raise ValueError("Hook per-step evidence has duplicated or unexpected extra records")
    if count != total_steps:
        raise ValueError("Hook record count differs from executed policy steps")
    return {"path": str(path.resolve()), "sha256": contract.sha256_file(path),
            "records": count, "all_executed_policy_steps_covered": True,
            "token_scope": "all_flattened_token_rows_in_byte_pinned_stock_hook",
            "layer_idx": 31, "hook_start_step": 0}


def install_observers(runner, cases, task_ids, trials, contract, expected_backend, expected_unnorm):
    """Observe calls while retaining the stock loop and callable behavior.

    No observer draws random numbers or changes a numerical tensor.  The
    source hash proves the environment's initial env.seed(0) inside the helper;
    any later env.seed call is trapped. The conformance comparison separately
    checks every stored action against the raw reference.
    """
    events, models = [], []
    state_counter = 0
    originals = {name: getattr(runner, name) for name in (
        "set_seed_everywhere", "load_model", "get_libero_env", "resolve_task_ids")}

    def seed(value):
        events.append({"event": "global_seed", "value": value})
        if value != 0 or any(e["event"] == "model_loaded" for e in events):
            raise ValueError("Global seed must be zero once before model loading")
        return originals["set_seed_everywhere"](value)

    def load(cfg):
        model = originals["load_model"](cfg)
        text = model.language_model
        actual = getattr(text.config, "_attn_implementation", None)
        if actual is None:
            actual = getattr(model.config, "_attn_implementation", None)
        if actual != expected_backend:
            raise ValueError(f"Actual attention backend {actual!r} differs from {expected_backend!r}")
        stats = getattr(model, "norm_stats", {})
        key = cfg.env.task_suite_name if cfg.env.task_suite_name in stats else f"{cfg.env.task_suite_name}_no_noops"
        if key != expected_unnorm or key not in stats:
            raise ValueError("Actual policy unnormalization key differs from the frozen protocol")
        if str(text.lm_head.weight.dtype) != "torch.bfloat16":
            raise ValueError("Actual model head must use BF16")
        if bool(getattr(model, "is_loaded_in_8bit", False)) or bool(getattr(model, "is_loaded_in_4bit", False)):
            raise ValueError("Quantized policies are not allowed")
        if getattr(model, "training", False):
            raise ValueError("The stock loaded model is unexpectedly in training mode")
        if len(text.model.layers) != 32 or int(text.lm_head.weight.shape[1]) != 4096:
            raise ValueError("Actual decoder layer/hidden dimensions differ")
        events.append({"event": "model_loaded", "attention_backend": actual,
                       "unnorm_key": key, "hidden_dtype": str(text.lm_head.weight.dtype)})
        models.append(model)
        return model

    def resolve(selected, count):
        value = originals["resolve_task_ids"](selected, count)
        if value != task_ids:
            raise ValueError("Stock task resolution differs from the frozen order")
        return value

    def env_factory(task, resolution=256):
        env, description = originals["get_libero_env"](task, resolution=resolution)
        number = sum(e["event"] == "environment_created" for e in events)
        if number >= len(task_ids):
            raise ValueError("Unexpected additional environment")
        task_id = task_ids[number]
        events.append({"event": "environment_created", "task_id": task_id,
                       "seed": 0, "seed_source": "byte_pinned_get_libero_env"})
        old_reset, old_state = env.reset, env.set_init_state

        def reset(*args, **kwargs):
            events.append({"event": "reset", "task_id": task_id})
            return old_reset(*args, **kwargs)

        def state(value, *args, **kwargs):
            nonlocal state_counter
            if state_counter >= len(cases):
                raise ValueError("Unexpected extra initial state")
            expected = cases[state_counter]
            digest = runner._initial_state_sha256(value)
            if expected["task_id"] != task_id or digest != expected["initial_state_sha256"]:
                raise ValueError("Live initial state/order differs from the frozen plan")
            previous = events[-1] if events else {}
            if previous.get("event") != "reset" or previous.get("task_id") != task_id:
                raise ValueError("Each state must immediately follow its environment reset")
            events.append({"event": "set_init_state", "task_id": task_id,
                           "task_episode_idx": expected["task_episode_idx"], "sha256": digest})
            state_counter += 1
            return old_state(value, *args, **kwargs)

        def forbidden_seed(*args, **kwargs):
            raise ValueError("Episode-level env.seed is forbidden under the team Baseline protocol")

        env.reset, env.set_init_state, env.seed = reset, state, forbidden_seed
        return env, description

    runner.set_seed_everywhere, runner.load_model = seed, load
    runner.get_libero_env, runner.resolve_task_ids = env_factory, resolve

    def finish():
        for name, function in originals.items():
            setattr(runner, name, function)
        if (sum(e["event"] == "global_seed" for e in events) != 1 or len(models) != 1
                or sum(e["event"] == "environment_created" for e in events) != len(task_ids)
                or sum(e["event"] == "reset" for e in events) != len(cases)
                or state_counter != len(cases)):
            raise ValueError("Seed/model/environment lifecycle count does not match the stock protocol")
        if events[0] != {"event": "global_seed", "value": 0}:
            raise ValueError("The first stock-runner event must set seed=0")
        return {"status": "verified", "events": events, "task_ids": task_ids,
                "num_trials_per_task": trials, "resume_policy": "restart_whole_condition"}

    return finish, models


def validate_remote_code(config, contract):
    remote = {}
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name.startswith("transformers_modules.") and file and str(file).endswith(".py"):
            item = Path(file)
            remote[name] = {"file": item.name, "sha256": contract.sha256_file(item)}
    expected = config.get("remote_code_files", {})
    must = {"configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"}
    if not isinstance(expected, dict) or not must <= set(expected):
        raise ValueError("Require pinned hashes for the three actual remote-code Python files")
    for filename in must:
        actual = [item["sha256"] for item in remote.values() if item["file"] == filename]
        if not actual or any(value != expected[filename] for value in actual):
            raise ValueError(f"Imported remote code differs: {filename}")
    return remote


def static_runtime_identity(torch, config, source, content, contract):
    """Resolve package/source metadata before the stock runner seeds RNG.

    find_spec may import package parents. Doing that only before the stock
    global seed avoids hidden metadata-import RNG changes after seeding.
    No model parameters, CUDA tensors or simulator state are created here.
    """
    versions = {}
    for package in ("torch", "torchvision", "transformers", "tokenizers", "numpy", "tensorflow",
                    "mujoco", "robosuite", "gym", "Pillow", "dictionary-learning", "libero"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "no_distribution_metadata"
    try:
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Cannot record actual GPU identity") from exc
    libraries = {}
    for module_name in ("dictionary_learning.dictionary", "dictionary_learning.trainers.batch_top_k",
                        "libero.libero.benchmark", "libero.libero.envs", "robosuite"):
        # Raw has no SAE hook and need not import the trainer. Hash the same
        # installed source in every condition rather than hashing only modules
        # that happened to be imported on that condition's execution path.
        module_spec = importlib.util.find_spec(module_name)
        file = module_spec.origin if module_spec is not None else None
        if file and Path(file).is_file():
            libraries[module_name] = contract.sha256_file(file)
    flags = {key: os.environ.get(key) for key in (
        "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID", "TF_ENABLE_ONEDNN_OPTS", "TF_FORCE_GPU_ALLOW_GROWTH",
        "CUBLAS_WORKSPACE_CONFIG")}
    return {"schema_version": "team500_actual_runtime_v1", "baseline": source,
            "model_files": content["model_files"], "sae_sha256": content["sae_sha256"],
            "sae_config_sha256": content["sae_config_sha256"],
            "package_versions": versions, "runtime_library_sources": libraries,
            "python": platform.python_version(), "platform": platform.platform(),
            "gpu": gpu, "cuda": torch.version.cuda,
            "attention_backend": config["expected_attention_backend"],
            "unnorm_key": config["expected_unnorm_key"], "dtype": "bfloat16",
            "quantized": False, "center_crop": True, "environment_flags": flags}


def runtime_identity(torch, model, processor, config, source, content, contract, static=None):
    base = static if static is not None else static_runtime_identity(torch, config, source, content, contract)
    return {**base, "remote_code": validate_remote_code(config, contract),
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "model_class": type(model).__module__ + "." + type(model).__name__,
            "processor_class": type(processor).__module__ + "." + type(processor).__name__}


def assert_runtime_identity(expected, actual, *, subset=False):
    if expected is None:
        return
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Expected runtime identity must be a nonempty certificate identity")
    keys = set(actual) if subset else set(expected) | set(actual)
    different = sorted(key for key in keys if actual.get(key) != expected.get(key))
    if different:
        raise ValueError(f"Runtime changed before the first environment: {different}")


def install_processor_observer(runner, models, config, source, content, contract, torch,
                               static, expected):
    original = runner.get_processor
    processors, observed = [], []

    def observer(policy):
        processor = original(policy)
        if len(models) != 1:
            raise ValueError("Processor must follow exactly one stock model load")
        actual = runtime_identity(torch, models[0], processor, config, source, content, contract, static)
        assert_runtime_identity(expected, actual)
        processors.append(processor)
        observed.append(actual)
        return processor

    runner.get_processor = observer

    def restore():
        runner.get_processor = original

    return restore, processors, observed


def execute_request(request_path):
    contract = _reference()
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    supplied_hash = request.pop("request_hash", None)
    if supplied_hash != contract.fingerprint(request):
        raise ValueError("Native request hash differs")
    if request.get("schema_version") != "team500_native_request_v1":
        raise ValueError("Unsupported native request")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 must be set before launching Python")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise ValueError("Model/processor execution must be fully offline")
    target = Path(request["output_dir"]).resolve()
    if target != request_path.parent or (target / "native-result.json").exists():
        raise ValueError("Native evidence directory must be a fresh request directory")
    if {p.name for p in target.iterdir()} - {"request.json", "child.stdout.log"}:
        raise ValueError("Native directory contains a previous attempt; restart in a new directory")
    root = Path(request["baseline_root"]).resolve()
    source = contract.verify_baseline(root)
    if request["baseline_source"] != source:
        raise ValueError("Baseline source changed after request approval")
    config = copy.deepcopy(request["config"])
    content = contract.validate_config(config)
    cases, task_ids, trials = contract.normalize_cases(request["cases"])
    if request["task_ids"] != task_ids or request["num_trials_per_task"] != trials:
        raise ValueError("Request task/trial metadata differs from the ordered prefix")
    condition = contract.validate_condition(request["condition"], reference=request["reference"])
    if (config["expected_attention_backend"] == "sdpa"
            and importlib.util.find_spec("flash_attn") is not None):
        raise ValueError("The stock Baseline would attempt FlashAttention. Use the agreed no-flash-attn "
                         "SDPA runtime; this driver will not silently change its import behavior.")
    # This process must never resolve the research worktree's modified runner.
    sys.path = [str(root)] + [entry for entry in sys.path if entry != str(Path(__file__).resolve().parents[2])]
    import event_sae
    if Path(event_sae.__file__).resolve() != root / "event_sae/__init__.py":
        raise ValueError("The wrong event_sae package was imported")
    import torch
    from libero.libero import benchmark
    from event_sae.openvla.eval import config as stock_config, model as stock_model, runner
    from event_sae.openvla.intervene import apply_resid_post_feature_perturb_hook
    if not torch.cuda.is_available():
        raise ValueError("A supported CUDA device is required for native execution")
    # Validate live states before the stock runner resets global RNG and loads weights.
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for case in cases:
        values = suite.get_task_init_states(case["task_id"])
        if runner._initial_state_sha256(values[case["task_episode_idx"]]) != case["initial_state_sha256"]:
            raise ValueError("Current LIBERO initial-state bytes differ from the frozen plan")
    base = config["baseline_config"]
    base["env"].update(task_ids=task_ids, num_trials_per_task=trials)
    base["logging"].update(root_dir=str(target / "native"), save_actions=True,
                           save_video=False, save_trajectory_records=False)
    cfg = stock_config.RunConfig(
        model=stock_config.ModelConfig(**base["model"]), env=stock_config.EnvConfig(**base["env"]),
        logging=stock_config.LoggingConfig(**base["logging"]),
        sae_collect=stock_config.SAECollectConfig(**base["sae_collect"]))
    expected_runtime = request.get("expected_runtime_identity")
    # All metadata imports take place before eval_libero's one global seed.
    static_runtime = static_runtime_identity(torch, config, source, content, contract)
    assert_runtime_identity(expected_runtime, static_runtime, subset=True)
    handles = []
    finish, models = install_observers(runner, cases, task_ids, trials, contract,
                                      config["expected_attention_backend"], config["expected_unnorm_key"])
    restore_processor, processors, observed_identities = install_processor_observer(
        runner, models, config, source, content, contract, torch, static_runtime, expected_runtime)

    def applier(*, model, cfg, run_dir, log_file):
        handle = apply_resid_post_feature_perturb_hook(
            model=model, layer_idx=31, sae_checkpoint_path=config["sae_checkpoint"],
            feature_idx=condition["feature_id"], alpha=float(condition["alpha"]),
            hook_start_step=0, run_dir=run_dir, log_file=log_file)
        handles.append(handle)
        return [handle]

    started = time.perf_counter()
    try:
        # Both paths call the byte-identical stock loop once. Reference has no
        # intervention applier, adapter adds only the stock feature hook.
        if request["reference"]:
            outcome = runner.eval_libero(cfg)
        else:
            outcome = runner.eval_libero(cfg, extra_hook_applier=None if condition["mode"] == "raw" else applier)
        lifecycle = finish()
        if len(models) != 1 or len(processors) != 1:
            raise ValueError("One model/processor pair must be loaded per whole condition")
        if (target / "native").resolve() not in Path(outcome.run_dir).resolve().parents:
            raise ValueError("Stock runner returned evidence outside the condition output")
        episodes, evidence = collect_episodes(outcome, cases, contract)
        actual_identity = runtime_identity(torch, models[0], processors[0], config, source, content, contract,
                                           static_runtime)
        if len(observed_identities) != 1 or actual_identity != observed_identities[0]:
            raise ValueError("Runtime identity changed during the condition")
        metrics = handles[0].summary() if handles else None
        if condition["mode"] != "raw":
            if len(handles) != 1 or metrics.get("num_forwards", 0) <= 0:
                raise ValueError("The intended stock feature hook did not execute")
            if metrics.get("feature_id") != condition["feature_id"] or metrics.get("alpha") != condition["alpha"]:
                raise ValueError("Hook target differs from condition")
            if condition["alpha"] == 0 and (metrics.get("post_intervention_active_feature_values") != 0
                                            or metrics.get("post_intervention_max_feature_activation") != 0):
                raise ValueError("Target feature was not completely suppressed")
            if condition["alpha"] == 1 and metrics.get("absolute_feature_delta") != 0:
                raise ValueError("Identity intervention changed feature values")
            evidence["hook_records"] = collect_hook_evidence(
                outcome.run_dir, condition, episodes, metrics, contract)
        if source != contract.verify_baseline(root):
            raise ValueError("Baseline source changed while running")
        receipt = {"schema_version": "team500_native_receipt_v1", "status": "completed", "synthetic": False,
                   "request_hash": supplied_hash, "condition": condition,
                   "case_manifest_hash": contract.fingerprint(cases), "episodes": episodes,
                   "exceptions_checked": True, "runtime_identity": actual_identity,
                   "hook_metrics": metrics, "lifecycle": lifecycle, "source": source,
                   "driver_sha256": contract.sha256_file(Path(__file__).resolve()),
                   "evidence": evidence, "baseline_config": asdict(cfg),
                   "elapsed_seconds": time.perf_counter() - started,
                   "resume_policy": "restart_whole_condition_never_skip_episodes"}
        _write_once(target / "native-result.json", receipt)
        return {"status": "completed", "conditions": 1, "rollouts": len(episodes),
                "output": str(target / "native-result.json")}
    finally:
        restore_processor()
        for handle in handles:
            handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--allow-model-execution", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_model_execution:
        print(json.dumps({"status": "not_executed", "model_execution": False}))
        return 0
    try:
        print(json.dumps(execute_request(args.request), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error": str(exc), "task_failure_substituted": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
