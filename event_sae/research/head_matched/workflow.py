"""Non-overwriting current-device scoring and hook validation for the diagnostic.

No simulator is launched here. Original full-suite scores/outcomes stay frozen;
five direction predictions are recomputed on their original discovery cache.
"""
from __future__ import annotations

from pathlib import Path
import time

from event_sae.research.output_head.provenance import (
    atomic_write_json, fingerprint, read_json, sha256_file,
)
from event_sae.research.head_rank.workflow import REPO, safe_output
from .protocol import NORM_POLICY, direction_scores, signed, verify_plan


def implementation_identity():
    from event_sae.research.head_rank.workflow import implementation_identity as frozen_identity
    files = sorted(Path(__file__).parent.glob("*.py")) + [REPO / "scripts/openvla/headmatched.py"]
    return {"frozen_head_rank": frozen_identity(),
            "files": {str(p.relative_to(REPO)): sha256_file(p) for p in files}}


def binding(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Missing local input: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def check_bindings(bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Source content bindings required")
    for name, item in bindings.items():
        if (not isinstance(item, dict) or not Path(item["path"]).is_absolute()
                or binding(item["path"]) != item):
            raise ValueError(f"Source missing or changed: {name}")


def check_runtime_bindings(cfg, plan, config_path=None):
    """Content, not timestamp, equality to the parent policy/SAE/head."""
    from event_sae.research.output_head.workflow import _head_path, _sae_paths, required_input
    verify_plan(plan)
    check_bindings(plan["source_bindings"])
    weights, sae_config = _sae_paths(cfg)
    head = _head_path(cfg)
    head = head.parent if head.is_file() else head
    snapshot = required_input(cfg, "local_model_snapshot")
    candidates = {"base_eval_config": cfg["rollout"]["base_eval_config"],
                  "sae": weights, "sae_config": sae_config,
                  "head_weights": head / "output_head.safetensors",
                  "head_manifest": head / "output_head_manifest.json",
                  "generation_manifest": required_input(cfg, "generation_manifest")}
    candidates.update({k: snapshot / k.split("/", 1)[1] for k in plan["parent_content_hashes"] if k.startswith("snapshot/")})
    actual = {k: binding(p) for k, p in candidates.items()}
    for key, expected in plan["parent_content_hashes"].items():
        if actual[key]["sha256"] != expected:
            raise ValueError(f"Current policy differs from frozen parent content: {key}")
    if config_path is not None:
        actual["config"] = binding(config_path)
    actual["runtime_inputs"] = binding(required_input(cfg, "runtime_inputs"))
    from event_sae.research.output_head.workflow import output_root
    actual.update({name: binding(output_root(cfg) / "validation" / filename)
                   for name, filename in (("head_parity", "runtime_parity.json"), ("edit_parity", "edit_parity.json"))})
    return actual


def _check_settings(cfg, identity, plan):
    if (cfg["scoring"]["alpha"] != 0 or cfg["scoring"]["pair_batch_size"] != 1
            or cfg["scoring"]["reduction_dtype"] != "float32"
            or cfg["scoring"]["negative_tolerance"] != 1e-6):
        raise ValueError("Reference isolated-row complete removal and FP32 KL required")
    # Backend/device can change only with fresh scoring and fresh validation.
    # Semantic, extraction and grouping identities cannot change.
    for key in ("model_revision", "code_revision", "processor_identity", "sae_checkpoint_hash",
                "sae_config_hash", "encoder_implementation_hash", "head_weights_hash", "head_manifest_hash",
                "generation_manifest_hash", "layer_idx", "capture_target", "mapping_version",
                "hidden_dtype", "reduction_dtype", "edit_backend", "encoder_grouping", "arithmetic_mode",
                "pair_batch_size", "negative_tolerance", "alpha", "preprocessing", "unnorm_key"):
        if identity.get(key) != plan["source_identity"].get(key):
            raise ValueError(f"Scoring protocol changed from frozen discovery data: {key}")


def _reference_cache(root, plan, identity):
    from event_sae.research.output_head.readouts import load_readout_cache
    paths = {"source_scores_sha256": root / "scores/scores.json",
             "source_sample_sha256": root / "sample_manifest.json",
             "source_cache_manifest_sha256": root / "readouts/manifest.json"}
    sources = {name: binding(path) for name, path in paths.items()}
    if any(sources[k]["sha256"] != plan["source_reference"][k] for k in paths):
        raise ValueError("Discovery cache, sample or source scores changed")
    sample, old = read_json(paths["source_sample_sha256"]), read_json(paths["source_scores_sha256"])
    if (old["scope"] != plan["source_scope"]
            or old["identity"]["cache_manifest_hash"] != sources["source_cache_manifest_sha256"]["sha256"]
            or sample["sample_hash"] != old["scope"]["sample_manifest_hash"]
            or identity["sae_checkpoint_hash"] != old["scope"]["sae_sha256"]
            or identity["head_weights_hash"] != old["identity"]["head_weights_hash"]):
        raise ValueError("Frozen cache/sample/SAE/head bindings differ")
    cache = load_readout_cache(root / "readouts")
    if (cache.get("synthetic") is not False or cache["records"] != sample["readouts"]
            or cache["dict_size"] != old["scope"]["dictionary_size"]):
        raise ValueError("Real exact original readout records required")
    return cache, sample, sources


def score_workflow(config, plan_path, reference_result, output, *, max_pairs=100000, progress=None):
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import numerical_identity, load_parity, _sae_paths, _head_path, sae_implementation_identity
    from event_sae.research.output_head.head import load_output_head
    from event_sae.openvla.activations import load_batch_topk_sae
    from event_sae.research.head_rank.scoring import score_cache, select_controls
    started = time.perf_counter()
    plan, cfg, root = read_json(plan_path), load_config(config), Path(reference_result)
    verify_plan(plan)
    weights, _ = _sae_paths(cfg)
    target = safe_output(output, [config, plan_path, root, weights.parent,
        cfg["inputs"].get("local_model_snapshot"), _head_path(cfg), cfg["output"]["root_dir"],
        *[v["path"] for v in plan["source_bindings"].values()]])
    identity, implementation = numerical_identity(cfg), implementation_identity()
    _check_settings(cfg, identity, plan)
    parity = load_parity(cfg, identity)
    sources = check_runtime_bindings(cfg, plan, config)
    sources.update(plan["source_bindings"])
    sources["plan"] = binding(plan_path)
    cache, sample, reference_sources = _reference_cache(root, plan, identity)
    sources.update(reference_sources)
    active = sorted({int(fid) for ids in cache["feature_ids"] for fid in ids.tolist()})
    anchor = plan["anchor_feature_id"]
    if select_controls(active, anchor, count=4, seed=1729) != plan["control_feature_ids"]:
        raise ValueError("Frozen discovery active universe changed control selection")
    cache["control_pool_feature_ids"] = active
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights), "encoder_implementation": sae_implementation_identity(type(sae))})
    if cache["manifest"]["sae_id"] != sae_id or cache["manifest"]["sample_hash"] != sample["sample_hash"]:
        raise ValueError("SAE/grouping differs from original full discovery cache")
    head = load_output_head(_head_path(cfg), device=cfg["scoring"]["device"])
    raw = score_cache(cache, head, sae, feature_ids=[anchor], control_count=4, seed=1729,
                      max_pairs=max_pairs, current_identity=identity, parity_reports=parity, progress=progress,
                      norm_atol=NORM_POLICY["atol"], norm_rtol=NORM_POLICY["rtol"])
    raw["score_hash"] = fingerprint(raw)
    if (fingerprint(raw["query_metadata"]) != plan["source_query_metadata_hash"]
            or raw["feature_scores"][0]["num_readouts"] != plan["source_readouts"]):
        raise ValueError("Current direction scoring did not retain the full frozen discovery population")
    predictions = direction_scores(raw, anchor)
    if raw["feature_scores"][0]["control_feature_ids"] != plan["control_feature_ids"]:
        raise ValueError("Scorer did not evaluate the four frozen controls")
    result = {"schema_version": "head_matched_scores_v1", "status": "scored",
              "plan_hash": plan["plan_hash"], "identity": identity, "implementation": implementation,
              "source_bindings": sources, "predictions": predictions, "score_result": raw,
              "norm_policy": NORM_POLICY, "scientific_scope": plan["scientific_scope"],
              "sample_scope": "all_original_discovery_readouts_one_anchor_four_controls",
              "synthetic": False, "new_rollouts": 0, "model_execution": False,
              "output_head_execution": True, "sae_training": False, "model_download": False,
              "elapsed_seconds": time.perf_counter() - started}
    if implementation != implementation_identity() or identity != numerical_identity(cfg):
        raise ValueError("Code or numerical identity changed during scoring")
    check_bindings(sources)
    result["score_hash"] = fingerprint(result)
    atomic_write_json(target, result)
    return {"status": "scored", "output": str(target), "score_hash": result["score_hash"],
            "new_rollouts": 0, "predictions": predictions, "diagnostics": raw["diagnostics"],
            "elapsed_seconds": result["elapsed_seconds"]}


def validate_workflow(config, plan_path, output, *, max_model_calls=32):
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import numerical_identity, load_parity, _sae_paths, _head_path
    from .native import validate_matched_runtime
    started = time.perf_counter()
    cfg, plan = load_config(config), read_json(plan_path)
    verify_plan(plan)
    weights, _ = _sae_paths(cfg)
    target = safe_output(output, [config, plan_path, weights.parent,
        cfg["inputs"].get("local_model_snapshot"), _head_path(cfg), cfg["output"]["root_dir"],
        *[v["path"] for v in plan["source_bindings"].values()]])
    identity, implementation = numerical_identity(cfg), implementation_identity()
    _check_settings(cfg, identity, plan)
    load_parity(cfg, identity)
    sources = check_runtime_bindings(cfg, plan, config)
    sources.update(plan["source_bindings"])
    sources["plan"] = binding(plan_path)
    evidence = target.with_name(target.stem + "-evidence")
    details = validate_matched_runtime(cfg, plan["conditions"], evidence, max_model_calls=max_model_calls)
    if (details.get("status") != "passed" or details.get("identity") != identity
            or details.get("conditions_hash") != fingerprint(plan["conditions"])
            or details.get("fixture_sha256") != sources["runtime_inputs"]["sha256"]):
        raise ValueError("Matched live validation failed or certificate binding differs")
    sources.update({f"evidence/{p.relative_to(evidence)}": binding(p) for p in sorted(evidence.rglob("*.jsonl"))})
    if implementation != implementation_identity() or identity != numerical_identity(cfg):
        raise ValueError("Code or numerical identity changed during validation")
    check_bindings(sources)
    result = {"schema_version": "head_matched_validation_v1", "status": "passed",
              "plan_hash": plan["plan_hash"], "identity": identity, "implementation": implementation,
              "source_bindings": sources, "details": details, "synthetic": False,
              "new_rollouts": 0, "model_execution": True,
              "elapsed_seconds": time.perf_counter() - started}
    result["validation_hash"] = fingerprint(result)
    atomic_write_json(target, result)
    return {"status": "passed", "output": str(target), "validation_hash": result["validation_hash"],
            "new_rollouts": 0, "model_execution": True, "model_action_queries": details["model_action_queries"],
            "elapsed_seconds": result["elapsed_seconds"]}
