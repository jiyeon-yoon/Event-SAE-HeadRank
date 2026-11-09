"""Dependency-free contract for the team's pinned, unmodified Baseline runner.

The numerical reference is an external checkout.  This module never imports
its package, initializes a simulator, or downloads a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

BASELINE_REVISION = "56e9f012aa88532f9880259b371d9300a8013b9e"
MODEL_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
MODEL_CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
SAE_SHA256 = "18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6"
REFERENCE_MANIFEST = Path(__file__).with_name("baseline_source_manifest.json")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline(root):
    """Check revision AND actual source bytes, including uncommitted edits."""
    root = Path(root).expanduser().resolve()
    manifest = json.loads(REFERENCE_MANIFEST.read_text())
    if manifest.get("revision") != BASELINE_REVISION:
        raise ValueError("Bundled Baseline reference revision is inconsistent")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual_revision != BASELINE_REVISION:
        raise ValueError("Use the exact pinned team Baseline checkout, not a current branch")
    for name, expected in manifest["files"].items():
        file = root / name
        if not file.is_file() or file.is_symlink() or sha256_file(file) != expected:
            raise ValueError(f"Pinned Baseline source differs: {name}")
    return {"revision": actual_revision, "source_files": manifest["files"],
            "source_manifest_sha256": sha256_file(REFERENCE_MANIFEST),
            "seed_protocol": "stock_baseline_global0_once_before_model_load_v1",
            "environment_lifecycle": "one_env_per_task_reused_in_trial_order"}


def normalize_cases(cases):
    """Only a contiguous prefix of the canonical 10 × 50 execution is legal.

    The stock loop has one uniform num_trials_per_task.  A pilot can therefore
    be part of task 0 alone, or an integer number of complete 50-trial tasks.
    In particular, 2 cases/task is NOT a pilot of the final global-RNG run.
    """
    if not isinstance(cases, list) or not 1 <= len(cases) <= 500:
        raise ValueError("Require 1..500 explicitly ordered cases")
    expected = [(task, trial) for task in range(10) for trial in range(50)]
    rows = []
    for number, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError("Each case must be a mapping")
        task, trial = row.get("task_id"), row.get("task_episode_idx")
        if type(task) is not int or type(trial) is not int or (task, trial) != expected[number]:
            raise ValueError("Cases must preserve the exact task-major 0..49 Baseline order")
        if row.get("suite", "libero_spatial") != "libero_spatial":
            raise ValueError("Only LIBERO-Spatial cases are supported")
        if "seed" in row and row["seed"] != 0:
            raise ValueError("Old hashed per-episode case seeds are not the team Baseline protocol")
        digest = row.get("initial_state_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Every current initial state needs a SHA256")
        rows.append({"suite": "libero_spatial", "task_id": task,
                     "task_episode_idx": trial, "initial_state_sha256": digest})
    count = len(rows)
    if count <= 50:
        task_ids, trials = [0], count
    elif count % 50 == 0:
        task_ids, trials = list(range(count // 50)), 50
    else:
        raise ValueError("Stock Baseline pilots must use full task blocks after task 0")
    return rows, task_ids, trials


def validate_condition(condition, *, reference=False):
    if not isinstance(condition, dict):
        raise ValueError("Condition must be a mapping")
    mode, fid, alpha = condition.get("mode"), condition.get("feature_id"), condition.get("alpha")
    if not isinstance(condition.get("condition_id"), str) or not condition["condition_id"]:
        raise ValueError("Condition identifier is required")
    if mode == "raw":
        if fid is not None or type(alpha) not in (int, float) or alpha != 1:
            raise ValueError("Raw condition requires feature_id=null and alpha=1")
    elif mode in ("identity", "intervention"):
        if type(fid) is not int or not 0 <= fid < 32768:
            raise ValueError("Invalid SAE feature index")
        if type(alpha) not in (int, float) or alpha != (1 if mode == "identity" else 0):
            raise ValueError("Only alpha=1 identity or alpha=0 complete removal is supported")
    else:
        raise ValueError("Unknown condition mode")
    if reference and mode != "raw":
        raise ValueError("The independent stock reference is raw only")
    return {key: condition[key] for key in ("condition_id", "mode", "feature_id", "alpha")}


def validate_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("baseline_config"), dict):
        raise ValueError("Require baseline_config in the team500 runtime configuration")
    base = config["baseline_config"]
    model, env = base.get("model", {}), base.get("env", {})
    if (model.get("family") != "openvla" or model.get("revision") != MODEL_REVISION
            or model.get("code_revision") != MODEL_CODE_REVISION
            or model.get("center_crop") is not True
            or model.get("load_in_8bit") is not False or model.get("load_in_4bit") is not False):
        raise ValueError("The pinned unquantized, center-cropped OpenVLA policy is required")
    snapshot = Path(model.get("checkpoint", "")).expanduser().resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise ValueError("An existing complete local model snapshot is required")
    if (env.get("task_suite_name") != "libero_spatial" or type(env.get("seed")) is not int
            or env["seed"] != 0 or env.get("num_steps_wait") != 10):
        raise ValueError("Baseline environment requires LIBERO-Spatial, seed=0, wait=10")
    if "per_episode_seed" in env or "trial_indices" in env:
        raise ValueError("Do not pass the HeadRank per-case seed or trial adapter")
    if config.get("expected_attention_backend") not in ("sdpa", "flash_attention_2"):
        raise ValueError("Freeze an explicit expected attention backend")
    if config.get("expected_unnorm_key") != "libero_spatial":
        raise ValueError("Freeze the exact libero_spatial unnormalization key")
    remote = config.get("remote_code_files")
    required_remote = {"configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"}
    if (not isinstance(remote, dict) or not required_remote <= set(remote)
            or any(not isinstance(remote[name], str) or len(remote[name]) != 64 for name in required_remote)):
        raise ValueError("Pin the actual remote-code file hashes before model execution")
    if base.get("sae_collect", {}).get("enabled") is not False:
        raise ValueError("Activation collection must remain disabled")
    files = config.get("model_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Pinned model content hashes are required")
    must = {"config.json", "dataset_statistics.json", "model.safetensors.index.json",
            *(f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5))}
    if not must <= set(files):
        raise ValueError("Model hashes must include config, statistics, index and all four shards")
    for name, expected in files.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid model content manifest")
        if not (snapshot / rel).is_file() or sha256_file(snapshot / rel) != expected:
            raise ValueError(f"Model content hash mismatch: {name}")
    sae = Path(config.get("sae_checkpoint", "")).expanduser().resolve()
    if not sae.is_file() or sha256_file(sae) != SAE_SHA256:
        raise ValueError("The frozen Spatial layer31 SAE checksum differs")
    sae_config = sae.parent / "config.json"
    if not sae_config.is_file():
        raise ValueError("SAE trainer config.json is required beside ae.pt")
    trainer = json.loads(sae_config.read_text()).get("trainer", {})
    if any(trainer.get(key) != value for key, value in {
        "layer": 31, "activation_dim": 4096, "dict_size": 32768,
        "submodule_name": "post_mlp_residual", "dict_class": "BatchTopKSAE", "k": 64,
    }.items()):
        raise ValueError("SAE trainer metadata differs from the frozen layer31 dictionary")
    return {"snapshot": str(snapshot), "model_files": files,
            "sae_sha256": SAE_SHA256, "sae_config_sha256": sha256_file(sae_config)}


def validate_episode_rows(episodes, cases):
    if not isinstance(episodes, list) or len(episodes) != len(cases):
        raise ValueError("Incomplete episode receipt")
    for row, case in zip(episodes, cases):
        for name in ("task_id", "task_episode_idx", "initial_state_sha256"):
            if row.get(name) != case[name]:
                raise ValueError(f"Episode case/order mismatch: {name}")
        actions = row.get("actions")
        if row.get("caught_exception") is not None or type(row.get("success")) is not bool:
            raise ValueError("Exceptional episodes cannot be recorded as task failures")
        if (not isinstance(actions, list) or not actions or type(row.get("num_actions")) is not int
                or row["num_actions"] != len(actions) or len(actions) > 220):
            raise ValueError("Empty, incomplete or overlong action trace")
        for vector in actions:
            if (not isinstance(vector, list) or len(vector) != 7
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                raise ValueError("Action vectors must contain seven finite real values")


def compare_traces(first, second):
    """Exact equality, not tolerance, rounded SR, or only final success flags."""
    if first.get("runtime_identity") != second.get("runtime_identity"):
        raise ValueError("Trace comparison runtime identities differ")
    if first.get("case_manifest_hash") != second.get("case_manifest_hash"):
        raise ValueError("Trace comparison case manifests differ")
    a, b = first.get("episodes"), second.get("episodes")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        raise ValueError("Trace comparison episode counts differ")
    keys = ("task_id", "task_episode_idx", "initial_state_sha256", "success", "num_actions", "actions")
    for index, (left, right) in enumerate(zip(a, b)):
        if any(left.get(key) != right.get(key) for key in keys):
            raise ValueError(f"Exact stored action/outcome parity failed at case {index}")
    return {"status": "passed", "cases": len(a), "exact_action_trace_equality": True,
            "historical_cross_hardware_parity": "not_claimed"}
