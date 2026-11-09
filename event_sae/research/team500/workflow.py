"""Setup and cache preparation for the explicitly overlapping Spatial-500 study.

No policy execution or downloads occur on import. The scoring cache is separate
from the historical 250-episode cache. Its numerical-validation fixtures are not
behavioral evaluations; only the external stock Baseline executes the latter.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import time

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, atomic_write_text, fingerprint,
    read_json, sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
PUBLIC_EVENT_SHA = "a82c23c588a12e897c20fec31e31263a111d23fba8c67731de24e5fdafd571f3"
BASELINE_COMMIT = "56e9f012aa88532f9880259b371d9300a8013b9e"


def write_once(path, payload):
    path = Path(path)
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"Existing setup file differs; use a new run directory: {path}")
    else:
        atomic_write_json(path, payload)


def paths(workspace, run_root=None):
    from scripts.openvla.headfull import paths as source_paths
    workspace = Path(workspace).expanduser().resolve()
    result = source_paths(workspace)
    result["run"] = Path(run_root or workspace / "head-team500-v1").expanduser().resolve()
    result["event_scores"] = result["events"] / "scores/event_feature_scores.pt"
    return result


def download_inputs(workspace, *, execute=False):
    """Explicit, pinned HF downloads. Never creates a Pod or launches inference."""
    from scripts.openvla.headfull import download, EVENT_REPO, EVENT_REVISION
    result = download(Path(workspace), execute=execute)
    if execute:
        from huggingface_hub import snapshot_download
        p = paths(workspace)
        snapshot_download(EVENT_REPO, repo_type="dataset", revision=EVENT_REVISION,
                          local_dir=p["events"], allow_patterns=["scores/event_feature_scores.pt"])
        if sha256_file(p["event_scores"]) != PUBLIC_EVENT_SHA:
            raise ValueError("Pinned public Spatial-500 Event matrix content mismatch")
        result.update(event_scores=str(p["event_scores"]), event_scores_sha256=PUBLIC_EVENT_SHA)
    return result


def snapshot_hashes(snapshot):
    """Verify all four current model shards against immutable pinned Hub OIDs."""
    from event_sae.research.head_rank.snapshot import PINNED_SHARDS
    snapshot = Path(snapshot).expanduser().resolve()
    for name, wanted in PINNED_SHARDS.items():
        file = snapshot / name
        if file.stat().st_size != wanted["size"] or sha256_file(file) != wanted["sha256"]:
            raise ValueError(f"Pinned model shard content differs: {name}")
    required = ("config.json", "dataset_statistics.json", "model.safetensors.index.json",
                "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json")
    for name in required:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(snapshot / name)
    # No data files, model shards or tokenizer assets are trusted by mtime alone.
    return {file.name: sha256_file(file) for file in sorted(snapshot.iterdir())
            if file.is_file() and not file.name.startswith(".")}


def remote_code_hashes():
    from scripts.openvla.headfull import CODE_REVISION
    cache = Path(os.environ.get("HF_HUB_CACHE") or
                 Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    source = cache / "models--openvla--openvla-7b" / "snapshots" / CODE_REVISION
    return {name: sha256_file(source / name) for name in
            ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py")}


def ordered_registry_cases(registry):
    episodes = registry.get("episodes", [])
    cases = [{"suite": row.get("suite", registry.get("suite")),
              "task_id": row["task_id"], "task_episode_idx": row["task_episode_idx"],
              "initial_state_sha256": row["initial_state_sha256"]} for row in episodes]
    cases.sort(key=lambda row: (row["task_id"], row["task_episode_idx"]))
    expected = [(task, trial) for task in range(10) for trial in range(50)]
    if [(r["task_id"], r["task_episode_idx"]) for r in cases] != expected:
        raise ValueError("Require exactly tasks 0..9, trials 0..49 in fixed order")
    if any(r["suite"] != "libero_spatial" or
           not isinstance(r["initial_state_sha256"], str) or len(r["initial_state_sha256"]) != 64
           for r in cases):
        raise ValueError("Invalid current LIBERO state identity")
    return cases


def setup(workspace, run_root, baseline_root, *, allow_libero=False):
    """Read current initial states; no policy weights are loaded for evaluation."""
    if not allow_libero:
        raise ValueError("Setup needs --allow-libero to inspect current benchmark states")
    import yaml
    from scripts.openvla.headfull import check_merged, SAE_SHA, MODEL_REVISION, CODE_REVISION
    from event_sae.research.output_head.config import validate_config
    from event_sae.research.output_head.state_registry import build_state_registry, build_generation_manifest
    from .native import verify_baseline

    p = paths(workspace, run_root)
    baseline = verify_baseline(baseline_root)
    assert_safe_output(p["run"], [p["inputs"], p["events"], p["model"], baseline_root])
    check_merged(p)
    if sha256_file(p["sae"]) != SAE_SHA or sha256_file(p["event_scores"]) != PUBLIC_EVENT_SHA:
        raise ValueError("Pinned SAE or public full500 Event score differs")
    p["run"].mkdir(parents=True, exist_ok=True)
    inputs = p["run"] / "inputs"
    # A fresh setup validates current LIBERO files, not historical collection state bytes.
    registry = build_state_registry(p["merged"], expected_shards=369)
    registry_path, generation_path = inputs / "source-episodes.json", inputs / "generation.json"
    if registry_path.exists():
        previous = read_json(registry_path)
        if previous != registry:
            raise ValueError("Live source registry changed; do not overwrite a frozen experiment")
    write_once(registry_path, registry)
    write_once(generation_path, build_generation_manifest(registry))
    cases = ordered_registry_cases(registry)
    cases_document = {"schema_version": "team500_cases_v1", "cases": cases,
                      "order": "task_ascending_then_trial_ascending",
                      "historical_state_identity_verified": registry["historical_state_identity_verified"],
                      "selection_evaluation_state_overlap": True}
    write_once(inputs / "cases.json", cases_document)
    baseline_config = {
        "model": {"family": "openvla", "checkpoint": str(p["model"]),
                  "revision": MODEL_REVISION, "code_revision": CODE_REVISION,
                  "load_in_8bit": False, "load_in_4bit": False, "center_crop": True},
        "env": {"task_suite_name": "libero_spatial", "num_steps_wait": 10,
                "num_trials_per_task": 50, "seed": 0, "task_ids": list(range(10))},
        "logging": {"root_dir": "logs", "save_video": False, "save_actions": True,
                    "save_prompt_records": True, "save_trajectory_records": False},
        "sae_collect": {"enabled": False, "mode": "dense", "layer_idxs": "31",
                        "flush_every": 50000, "sae_checkpoint": "", "topk": 64,
                        "rows_per_shard": 20000},
    }
    runtime = {"schema_version": "team500_runtime_v1", "baseline_config": baseline_config,
               "sae_checkpoint": str(p["sae"]), "expected_attention_backend": "sdpa",
               "expected_unnorm_key": "libero_spatial", "model_files": snapshot_hashes(p["model"]),
               "remote_code_files": remote_code_hashes(), "baseline_commit": BASELINE_COMMIT,
               "sae_sha256": SAE_SHA}
    write_once(inputs / "runtime-config.json", runtime)
    # Stock-compatible YAML is useful for the independent reference execution.
    baseline_yaml = inputs / "baseline-eval.yaml"
    text = yaml.safe_dump(baseline_config, sort_keys=False)
    if baseline_yaml.exists() and baseline_yaml.read_text() != text:
        raise ValueError("Frozen Baseline config differs")
    if not baseline_yaml.exists():
        atomic_write_text(baseline_yaml, text)
    head = validate_config({
        "experiment_name": "team500_head_numeric_validation_and_cache_only",
        "inputs": {"dense_dir": str(p["merged"] / "sae_activations/post_mlp_residual"),
                   "source_episode_manifest": str(registry_path), "sae_checkpoint": str(p["sae"]),
                   "generation_manifest": str(generation_path), "local_model_snapshot": str(p["model"]),
                   "head_export_metadata": str(inputs / "head.json"),
                   "event_scores_path": str(p["event_scores"])},
        "sampling": {"mode": "pilot", "task_ids": list(range(10)),
                     "max_episodes_per_task": 50, "max_steps_per_episode": 8},
        "scoring": {"device": "cuda:0", "max_scored_pairs": 10000000},
        "selection": {"top_k": 5, "random_audit_features": 0},
        "rollout": {"base_eval_config": str(ROOT / "configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml"),
                    "max_total_rollouts": 8500},
        "output": {"root_dir": str(p["run"] / "head-cache")},
    })
    head_path = p["run"] / "head-config.yaml"
    if head_path.exists():
        existing = yaml.safe_load(head_path.read_text())
        # Later head export/input capture only fill these numerical fixture paths.
        for key in ("output_head_bundle", "runtime_inputs", "head_export_metadata"):
            head["inputs"][key] = existing["inputs"].get(key)
        if existing != head:
            raise ValueError("Existing head configuration differs; use a new output directory")
    else:
        atomic_write_text(head_path, yaml.safe_dump(head, sort_keys=False))
    receipt = {"schema_version": "team500_setup_v1", "baseline": baseline,
               "selection_episodes": 500, "evaluation_cases": 500, "top_k": 5,
               "methods": ["mean_head_kl", "tail_head_kl", "event_max"],
               "selection_evaluation_state_overlap": True, "sae_training_overlap": True,
               "historical_state_identity_verified": registry["historical_state_identity_verified"],
               "historical_generation_flags_verified": False,
               "runtime_config_hash": fingerprint(runtime), "case_hash": fingerprint(cases),
               "head_config": str(head_path), "model_execution": False}
    write_once(p["run"] / "setup.json", receipt)
    return {"status": "configured_not_executed", "run_root": str(p["run"]),
            "head_config": str(head_path), "runtime_config": str(inputs / "runtime-config.json"),
            "cases": str(inputs / "cases.json"), "methods": receipt["methods"],
            "selection_episodes": 500, "evaluation_cases": 500, "top_k": 5,
            "selection_evaluation_state_overlap": True, "model_execution": False}


def prepare_cache(config_path, *, execute=False):
    if not execute:
        return {"status": "not_executed", "policy_execution": False,
                "next": "Explicit --execute permits SAE encoding of existing activation files only"}
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.readouts import prepare_readout_cache
    from event_sae.research.output_head.workflow import _sae_paths, sae_implementation_identity
    from event_sae.openvla.activations import load_batch_topk_sae
    from .readouts import build_full500_manifest
    started = time.perf_counter()
    cfg = load_config(config_path)
    dense = Path(cfg["inputs"]["dense_dir"])
    registry = read_json(cfg["inputs"]["source_episode_manifest"])
    generation = read_json(cfg["inputs"]["generation_manifest"])
    sample = build_full500_manifest(dense / "activation_index.jsonl", registry, generation, dense_dir=dense)
    root = Path(cfg["output"]["root_dir"])
    write_once(root / "sample_manifest.json", sample)
    weights, _ = _sae_paths(cfg)
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights),
                          "encoder_implementation": sae_implementation_identity(type(sae))})
    cache = prepare_readout_cache(sample, sae, root / "readouts", dense_dir=dense,
        device=cfg["scoring"]["device"], sae_id=sae_id, hidden_dtype="bfloat16",
        encoder_mode="original_forward", resume=True)
    return {"status": "prepared", "cache": str(root / "readouts"),
            "num_readouts": len(cache["records"]), "policy_execution": False,
            "elapsed_seconds": time.perf_counter() - started}


def freeze_plan(scores_path, cases_path, runtime_path, output):
    from .protocol import build_plan
    scores, cases_document, runtime = map(read_json, (scores_path, cases_path, runtime_path))
    cases = cases_document["cases"] if isinstance(cases_document, dict) else cases_document
    target = assert_safe_output(output, [scores_path, cases_path, runtime_path])
    if target.exists():
        raise FileExistsError(target)
    plan = build_plan(scores, cases, runtime,
        score_file_sha256=sha256_file(scores_path),
        reference_bindings={"scores_path": str(Path(scores_path).resolve()),
                            "cases_path": str(Path(cases_path).resolve()),
                            "cases_file_sha256": sha256_file(cases_path),
                            "runtime_config_path": str(Path(runtime_path).resolve()),
                            "runtime_config_file_sha256": sha256_file(runtime_path)})
    atomic_write_json(target, plan)
    return {"status": "planned_not_executed", "output": str(target), "plan_hash": plan["plan_hash"],
            "total_rollouts": plan["total_rollouts"], "conditions": len(plan["conditions"]),
            "rankings": plan["methods"], "new_rollouts": 0}
