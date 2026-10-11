"""Read-only frozen-input adapter for additive response descriptors."""
from __future__ import annotations

import time
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from event_sae.research.head_rank.workflow import DEVELOPMENT_FEATURES, REPO, safe_output, subset_cache
from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, implementation_fingerprint, read_json, sha256_file,
)


def descriptor_implementation_identity():
    """Snapshot this scorer independently of the older HeadRank artifact hash."""
    files = [Path(__file__), Path(__file__).with_name("descriptors.py")]
    files += [REPO / "event_sae/research" / name for name in (
        "head_rank/scoring.py", "head_rank/workflow.py", "head_followup/interventions.py",
        "head_followup/metrics.py", "head_followup/stage1.py",
    )]
    return {"schema_version": "head_response_descriptor_implementation_v1",
            "base_implementation_hash": implementation_fingerprint(),
            "files": {str(path.relative_to(REPO)): sha256_file(path) for path in sorted(files)}}


def validate_descriptor_payload(payload, *, source_scores, reference_scores, bindings):
    """Pure JSON/NumPy semantic validation for full-readout calibration input.

    Integrity hashes alone do not establish a population or meaning. Check the
    complete task/query Cartesian product, numerical-source identities and the
    actual native decode, not merely a producer's ``full`` or ``verified`` flag.
    Development16/custom feature subsets remain valid for development fitting;
    they cannot advertise themselves as the complete all-active universe.
    """
    import numpy as np
    from event_sae.research.head_followup.stage1 import validate_mapping

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def numeric(value, name, lower=None, upper=None):
        require(not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value), f"Descriptor {name} must be finite numeric data")
        require((lower is None or value >= lower) and (upper is None or value <= upper),
                f"Descriptor {name} outside valid range")
        return value

    def id_list(values, name, upper=None):
        require(isinstance(values, list) and bool(values)
                and all(type(value) is int and value >= 0 and (upper is None or value < upper)
                        for value in values), f"Descriptor {name} must be nonempty integer IDs")
        require(values == sorted(set(values)), f"Descriptor {name} must be canonical and unique")
        return values

    require(isinstance(payload, dict) and payload.get("descriptor_hash") == fingerprint(
        {key: value for key, value in payload.items() if key != "descriptor_hash"}), "Descriptor integrity mismatch")
    require(payload.get("schema_version") == "head_response_descriptors_v1"
            and payload.get("synthetic") is False and payload.get("full_readouts") is True
            and payload.get("sampling_mode") == "full", "Require real full-readout descriptors, not smoke/synthetic data")
    scope = bindings["scope"]
    require(payload.get("scope") == scope == source_scores.get("scope") == reference_scores.get("scope"),
            "Descriptor/source/reference scope mismatch")
    width = scope.get("dictionary_size")
    require(type(width) is int and width > 0 and payload.get("dictionary_size") == width,
            "Descriptor dictionary size mismatch")
    tasks = id_list(payload.get("task_ids"), "task_ids")
    require(tasks == sorted(scope["task_ids"]), "Descriptor task population mismatch")
    features = id_list(payload.get("feature_ids"), "feature_ids", width)
    require(payload.get("scored_feature_ids") == features, "Descriptor scored/header feature IDs differ")
    frozen = id_list(payload.get("frozen_active_feature_ids"), "frozen_active_feature_ids", width)
    require(frozen == source_scores.get("feature_ids") and set(features).issubset(frozen),
            "Descriptor features differ from frozen source universe")
    feature_set = payload.get("feature_set")
    if feature_set == "all-active":
        require(features == frozen and payload.get("development_only") is False,
                "All-active descriptor declaration is incomplete/developmental")
    else:
        require(feature_set in ("development16", "custom") and payload.get("development_only") is True,
                "Descriptor subset must be explicitly developmental")
        if feature_set == "development16":
            require(features == list(DEVELOPMENT_FEATURES), "Development16 descriptor panel differs from historical IDs")
    for name, expected in (("source_sample_sha256", bindings["sample_manifest_sha256"]),
                           ("reference_scores_sha256", bindings["reference_scores_sha256"]),
                           ("readout_record_hash", bindings["readout_record_hash"])):
        require(payload.get(name) == expected, f"Descriptor {name} source mismatch")
    require(payload.get("source_cache_manifest_sha256") == source_scores.get("source_cache_manifest_sha256")
            == reference_scores.get("identity", {}).get("cache_manifest_hash")
            and payload.get("source_cache_manifest_sha256") is not None, "Descriptor frozen cache binding mismatch")
    identity, source_identity = payload.get("identity", {}), source_scores.get("identity", {})
    for name in ("sae_checkpoint_hash", "head_weights_hash", "model_revision", "code_revision", "layer_idx"):
        require(source_identity.get(name) is not None and identity.get(name) == source_identity[name],
                f"Descriptor numerical source differs: {name}")
    require(identity["sae_checkpoint_hash"] == scope["sae_sha256"], "Descriptor SAE/scope mismatch")
    for name, expected in (("encoder_grouping", "original_forward"), ("pair_batch_size", 1),
                           ("reduction_dtype", "float32"), ("negative_tolerance", 1e-6), ("alpha", 0.)):
        require(identity.get(name) == expected, f"Descriptor numerical semantics differ: {name}")
    require(payload.get("action_dimension_order") == list(range(7))
            and payload.get("scalar_denominator") == "all_task_readouts"
            and payload.get("position_denominator") == "all_task_queries_per_position"
            and payload.get("aggregation") == "per_task_equal_episode_query_means_including_all_inactive_zeros",
            "Descriptor position/aggregation semantics mismatch")
    require(payload.get("signed_shift_definition") == "E[edited_normalized_bin|action_tokens]-E[baseline_normalized_bin|action_tokens]"
            and payload.get("absolute_shift_definition") == "mean_of_absolute_per_readout_expected_bin_shift_not_absolute_of_mean"
            and payload.get("edit_norm_definition") == "realized_post_hidden_dtype_cast_L2"
            and payload.get("metric_reduction_dtype") == "float32"
            and payload.get("response_scope") == "fixed_prefix_token_bin_response_not_physical_harm_or_rollout_prediction",
            "Descriptor response semantics mismatch")
    require(payload.get("outcome_labels_used") is False and type(payload.get("new_rollouts")) is int
            and payload["new_rollouts"] == 0 and payload.get("model_regeneration") is False
            and payload.get("model_download") is False and payload.get("sae_training") is False,
            "Descriptor must use frozen inputs without outcomes/regeneration/training")
    queries = payload.get("query_metadata")
    require(isinstance(queries, list) and bool(queries) and queries == source_scores.get("query_metadata"),
            "Descriptor query population differs from complete source queries")
    task_counts, episodes, keys = Counter(), defaultdict(Counter), []
    for number, query in enumerate(queries):
        require(isinstance(query, dict), "Descriptor query must be an object")
        task, uid, step = query.get("task_id"), query.get("episode_uid"), query.get("step_in_episode")
        require(type(task) is int and task in tasks and isinstance(uid, str) and bool(uid)
                and type(step) is int and step >= 0 and type(query.get("query_index")) is int
                and query["query_index"] == number and type(query.get("num_action_positions")) is int
                and query["num_action_positions"] == 7, "Malformed descriptor query metadata")
        keys.append((task, uid, step))
        task_counts[task] += 1
        episodes[task][uid] += 1
    require(keys == sorted(set(keys)) and sorted(task_counts) == tasks,
            "Descriptor queries are duplicated/noncanonical or missing tasks")
    require(all(len(set(counts.values())) == 1 for counts in episodes.values()),
            "Descriptor query counts are unbalanced across episodes within task")
    sample, n_readouts = payload.get("scoring_sample", {}), len(queries) * 7
    require(sample.get("readouts_used") == sample.get("frozen_readouts") == n_readouts
            == source_scores.get("scoring_sample", {}).get("readouts_used")
            and sample.get("discovery_episodes_used") == sum(map(len, episodes.values())),
            "Descriptor full readout/episode population mismatch")
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == len(features) * len(tasks), "Descriptor feature/task population incomplete")
    seen, active_pairs = set(), 0
    arrays = ("signed_action_bin_shift", "absolute_action_bin_shift", "action_argmax_flip", "action_conditional_kl",
              "action_mass_abs_change", "signed_action_mass_change")
    for row in rows:
        require(isinstance(row, dict), "Descriptor row must be an object")
        key = row.get("feature_id"), row.get("task_id")
        require(type(key[0]) is int and type(key[1]) is int and key[0] in features and key[1] in tasks
                and key not in seen, "Descriptor row duplicated or outside header population")
        seen.add(key)
        n = task_counts[key[1]]
        require(row.get("num_queries") == n and row.get("num_readouts") == 7 * n,
                "Descriptor row query/readout denominator mismatch")
        active = row.get("num_active_readouts")
        require(type(active) is int and 0 <= active <= 7 * n, "Descriptor active count invalid")
        active_pairs += active
        frequency = numeric(row.get("readout_activation_frequency"), "readout_activation_frequency", 0, 1)
        require(math.isclose(frequency, active / (7 * n), abs_tol=1e-12, rel_tol=0),
                "Descriptor activation count/frequency mismatch")
        for name in ("mean_head_kl", "mean_edit_norm", "mean_intended_edit_norm", "mean_readout_activation", "mean_cast_norm_abs_error"):
            numeric(row.get(name), name, 0)
        numeric(row.get("full_vocab_argmax_flip_rate"), "full_vocab_argmax_flip_rate", 0, 1)
        for name in arrays:
            vector = row.get(name)
            require(isinstance(vector, list) and len(vector) == 7, f"Descriptor {name} requires seven dimensions")
            lo, hi = {"signed_action_bin_shift": (-2, 2), "absolute_action_bin_shift": (0, 2),
                      "action_argmax_flip": (0, 1), "action_conditional_kl": (0, None),
                      "action_mass_abs_change": (0, 1), "signed_action_mass_change": (-1, 1)}[name]
            for value in vector:
                numeric(value, name, lo, hi)
        require(all(absolute + 1e-7 >= abs(signed) for signed, absolute in
                    zip(row["signed_action_bin_shift"], row["absolute_action_bin_shift"])),
                "Descriptor mean absolute shift is smaller than absolute mean shift")
        if active == 0:
            require(all(value == 0 for name in arrays for value in row[name])
                    and all(row[name] == 0 for name in ("mean_head_kl", "mean_edit_norm", "full_vocab_argmax_flip_rate",
                            "mean_intended_edit_norm", "mean_readout_activation", "mean_cast_norm_abs_error")),
                    "Inactive descriptor responses must be exact zero")
    diagnostics = payload.get("diagnostics", {})
    require(diagnostics.get("planned_pairs") == diagnostics.get("executed_pairs") == active_pairs
            and diagnostics.get("num_readouts") == n_readouts, "Descriptor execution/count diagnostics mismatch")
    mapping = payload.get("action_mapping", {})
    require(isinstance(mapping, dict), "Descriptor mapping must be an object")
    validate_mapping(mapping, {"weights_sha256": source_identity["head_weights_hash"],
                              "model_revision": source_identity["model_revision"],
                              "code_revision": source_identity["code_revision"]})
    require(mapping.get("verified") is True and mapping.get("synthetic", False) is False
            and type(mapping.get("base_vocab_size")) is int and mapping["base_vocab_size"] == 32000
            and type(mapping.get("n_action_bins")) is int and mapping["n_action_bins"] == 256
            and type(mapping.get("head_vocab_size")) is int and mapping["head_vocab_size"] == 32064
            and mapping.get("tokenizer_sha256") == mapping.get("evidence", {}).get("tokenizer.json"),
            "Descriptor native action decode metadata mismatch")
    edges = np.linspace(-1., 1., 256)
    centers = (edges[:-1] + edges[1:]) / 2
    expected_decode = {
        "token_id_to_bin_center": {str(token): float(centers[min(max(32000 - token - 1, 0), 254)])
                                   for token in range(31744, 32000)},
        "decode_rule": "center[clip(base_vocab_size-token_id-1,0,n_action_bins-2)]",
        "num_action_tokens": 256, "num_distinct_bin_centers": 255,
        "conditional_argmax_scope": "token_identity_including_endpoint_aliases_not_physical_action_identity",
    }
    require(payload.get("action_bin_decode") == expected_decode, "Descriptor token-to-bin decode differs from verified native mapping")
    require(isinstance(payload.get("action_mapping_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", payload["action_mapping_sha256"]) is not None,
            "Descriptor mapping file hash is invalid")


def descriptor_workflow(config, reference_result, mapping_path, output, *, feature_ids=None,
                        full_readouts=True, max_pairs=5000000, progress=None):
    """Fresh parity plus verified local mapping; never re-encode or download.

    None selects the historical development16. The explicit string
    ``all-active`` opts into the frozen active universe; an ID collection is
    a custom development subset. Even all-active descriptors are response
    measurements, never prospective or closed-loop evidence.
    """
    from event_sae.openvla.activations import load_batch_topk_sae
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.head import load_output_head
    from event_sae.research.output_head.readouts import load_readout_cache
    from event_sae.research.output_head.workflow import (
        _head_path, _sae_paths, load_parity, numerical_identity, sae_implementation_identity,
    )
    from .descriptors import score_descriptor_cache

    if type(full_readouts) is not bool:
        raise ValueError("full_readouts must be boolean")
    if isinstance(feature_ids, str) and feature_ids != "all-active":
        raise ValueError("feature_ids string must be the explicit all-active opt-in")
    cfg, root = load_config(config), Path(reference_result)
    weights, _ = _sae_paths(cfg)
    sources = [root, config, mapping_path, weights.parent, cfg["output"]["root_dir"],
               _head_path(cfg), cfg["inputs"].get("local_model_snapshot")]
    target = safe_output(output, sources)
    assert_safe_output(cfg["output"]["root_dir"], [root, REPO.parent / "Event-SAE-Head"])
    if (cfg["scoring"]["alpha"] != 0 or cfg["scoring"]["pair_batch_size"] != 1
            or cfg["scoring"]["reduction_dtype"] != "float32"
            or cfg["scoring"]["negative_tolerance"] != 1e-6
            or cfg["scoring"]["edit_backend"] != "reference"):
        raise ValueError("Descriptors require reference removal, isolated rows and FP32 KL tolerance=1e-6")
    source_code = descriptor_implementation_identity()
    identity = numerical_identity(cfg)
    reports = load_parity(cfg, identity)  # Current runtime root, never archived validation.
    source_files = {"reference_scores_sha256": root / "scores/scores.json",
                    "source_sample_sha256": root / "sample_manifest.json",
                    "source_cache_manifest_sha256": root / "readouts/manifest.json",
                    "action_mapping_sha256": Path(mapping_path)}
    source_hashes = {key: sha256_file(path) for key, path in source_files.items()}
    sample, old = read_json(source_files["source_sample_sha256"]), read_json(source_files["reference_scores_sha256"])
    if sample.get("synthetic") is not False or old.get("synthetic") is not False:
        raise ValueError("Descriptor workflow requires real frozen sample/reference artifacts")
    if old.get("identity_hash") != fingerprint(old.get("identity")):
        raise ValueError("Frozen reference numerical identity hash mismatch")
    for name in ("model_revision", "code_revision", "layer_idx", "capture_target", "alpha"):
        if identity.get(name) is None or identity[name] != old["identity"].get(name):
            raise ValueError(f"Frozen reference/current numerical binding mismatch: {name}")
    if (source_hashes["source_cache_manifest_sha256"] != old["identity"]["cache_manifest_hash"]
            or sample["sample_hash"] != old["scope"]["sample_manifest_hash"]
            or identity["sae_checkpoint_hash"] != old["scope"]["sae_sha256"]
            or identity["head_weights_hash"] != old["identity"]["head_weights_hash"]):
        raise ValueError("Frozen cache/sample/SAE/head binding mismatch")
    cache = load_readout_cache(root / "readouts")
    if (cache.get("synthetic") is not False or cache["manifest"].get("synthetic", False) is not False
            or cache["records"] != sample["readouts"]
            or cache["dict_size"] != old["scope"]["dictionary_size"]
            or sorted({row["task_id"] for row in cache["records"]}) != sorted(old["scope"]["task_ids"])):
        raise ValueError("Require real complete cache with exact frozen scope/sample records")
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights),
                          "encoder_implementation": sae_implementation_identity(type(sae))})
    if cache["manifest"]["sae_id"] != sae_id or cache["manifest"]["sample_hash"] != sample["sample_hash"]:
        raise ValueError("SAE/grouping differs from frozen cache")
    active_ids = sorted({int(fid) for ids in cache["feature_ids"] for fid in ids.tolist()})
    feature_set = "development16" if feature_ids is None else "all-active" if feature_ids == "all-active" else "custom"
    selected = list(DEVELOPMENT_FEATURES) if feature_ids is None else active_ids if feature_set == "all-active" else feature_ids
    full_count = len(cache["records"])
    if not full_readouts:
        cache = subset_cache(cache)
    head = load_output_head(_head_path(cfg), device=cfg["scoring"]["device"])
    mapping = read_json(mapping_path)
    started = time.perf_counter()
    result = score_descriptor_cache(cache, head, sae, mapping, feature_ids=selected, max_pairs=max_pairs,
                                    current_identity=identity, parity_reports=reports, progress=progress)
    result.update(status="scored", scope=old["scope"], identity=identity, implementation=source_code,
        **source_hashes, source_scores_sha256=source_hashes["reference_scores_sha256"],
        source_scores_binding_role="frozen_reference_scores_not_new_headrank_full_scores",
        parity_report_hashes={key: fingerprint(report) for key, report in reports.items()},
        full_readouts=full_readouts, sampling_mode="full" if full_readouts else "smoke",
        feature_set=feature_set, development_only=feature_set != "all-active" or not full_readouts,
        frozen_active_feature_ids=active_ids, scored_feature_ids=result["feature_ids"],
        readout_record_hash=fingerprint(cache["records"]),
        scoring_sample={"readouts_used": len(cache["records"]), "frozen_readouts": full_count,
                        "discovery_episodes_used": len({(r["task_id"], r["episode_uid"]) for r in cache["records"]})},
        source_readonly_roots=[str(Path(path).resolve()) for path in sources if path],
        elapsed_seconds=time.perf_counter() - started)
    if source_code != descriptor_implementation_identity():
        raise ValueError("Descriptor scorer source changed during execution; mixed-code results are not saved")
    if source_hashes != {key: sha256_file(path) for key, path in source_files.items()}:
        raise ValueError("Descriptor input changed during execution; mixed-input results are not saved")
    result["descriptor_hash"] = fingerprint(result)
    atomic_write_json(target, result)
    return {"status": "scored", "output": str(target), "output_sha256": sha256_file(target),
            "descriptor_hash": result["descriptor_hash"], "features": len(result["feature_ids"]),
            "tasks": len(result["task_ids"]), "full_readouts": full_readouts,
            "development_only": result["development_only"], "feature_set": feature_set,
            "diagnostics": result["diagnostics"], "elapsed_seconds": result["elapsed_seconds"], "new_rollouts": 0}
