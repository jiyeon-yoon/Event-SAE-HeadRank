"""Exact Mean/Tail Head-KL and pinned full500 Event-max, no new rollout.

Head scores summarize eight sampled queries per inherited episode. Event
scores use the original full timelines and frozen clusters. The population
matches, but timestep support and the meaning of the two scores differ.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, read_json, sha256_file,
)
from .protocol import (
    SCORE_SCHEMA, SAE_SHA, DICT_SIZE, UNIVERSE, MODEL_REVISION, CODE_REVISION,
    ordered_cases, validate_scores,
)
from .readouts import validate_full500_sample

PUBLIC_EVENT_SHA = "a82c23c588a12e897c20fec31e31263a111d23fba8c67731de24e5fdafd571f3"


def implementation_identity():
    root = Path(__file__).resolve().parents[3]
    files = sorted(Path(__file__).parent.glob("*.py")) + [
        root / "event_sae/research/head_rank/scoring.py",
        root / "event_sae/research/output_head/readouts.py",
        root / "event_sae/research/output_head/head.py",
        root / "event_sae/research/output_head/sensitivity.py",
        root / "event_sae/research/head_followup/interventions.py",
    ]
    return {str(p.relative_to(root)): sha256_file(p) for p in files if p.is_file()}


def validate_event_payload(payload, sample, topk, actual_event_sha):
    """Never reinterpret a 250-episode comparator as the public full500 export."""
    import torch

    validate_full500_sample(sample)
    if actual_event_sha != PUBLIC_EVENT_SHA:
        raise ValueError("Require the pinned original full500 Event score SHA; no renamed/rebound 250 scores")
    if (topk.get("format") != "token_topk_sparse_v1" or topk.get("layer") != 31
            or topk.get("dict_size") != DICT_SIZE or topk.get("topk") != 64
            or topk.get("sae_sha256") != SAE_SHA
            or topk.get("activation_index_sha256") != sample.get("source_index_file_sha256")):
        raise ValueError("Public Event Top-K metadata differs from the exact Head source index/SAE")
    if topk.get("capture_target") not in (None, "post_mlp_residual"):
        raise ValueError("Wrong public Event hook target")
    if topk.get("capture_target") is None and Path(topk.get("dense_dir", "")).parts[-2:] != (
            "sae_activations", "post_mlp_residual"):
        raise ValueError("Legacy null hook metadata requires original post_mlp_residual path evidence")
    source = payload.get("source", {})
    if (source.get("dict_size") != DICT_SIZE or source.get("layer") != 31
            or source.get("allowed_episode_nums") is not None or payload.get("window_size") != 5
            or payload.get("step_mapping") != "inference_step"):
        raise ValueError("Event payload is not the unfiltered original full500 matrix")
    rows = payload.get("row_keys")
    matrix = payload.get("matrix_raw")
    if (not isinstance(rows, list) or not rows or not isinstance(matrix, torch.Tensor)
            or matrix.ndim != 2 or list(matrix.shape) != [len(rows), DICT_SIZE]
            or not torch.isfinite(matrix).all()):
        raise ValueError("Malformed public Event matrix/row keys")
    task_set = set(range(10))
    if {r.get("task_id") for r in rows} != task_set:
        raise ValueError("Event matrix must include every Spatial task")
    keep = []
    for index, row in enumerate(rows):
        coverage = row.get("episode_coverage")
        if (isinstance(coverage, bool) or not isinstance(coverage, (float, int))
                or not math.isfinite(coverage) or not 0 <= coverage <= 1):
            raise ValueError("Every Event cluster row requires explicit episode coverage")
        if coverage >= .5:
            keep.append(index)
    if not keep or {rows[index]["task_id"] for index in keep} != task_set:
        raise ValueError("Every task requires at least one canonical Event cluster")
    episodes = {r["episode_num"]: r["task_id"] for r in sample["readouts"]}
    events = payload.get("selected_events", [])
    if not events or any(e.get("episode_num") not in episodes
                         or e.get("task_id") != episodes[e["episode_num"]] for e in events):
        raise ValueError("Event input episode/task identities do not match the full500 Head source")
    if payload.get("selection_counts", {}).get("skipped_missing_window_vectors", 0) != 0:
        raise ValueError("Public Event matrix omitted required activation windows")
    vector = matrix[keep].float().max(dim=0).values.tolist()
    return vector, {"source_sha256": actual_event_sha,
                    "aggregation": "max_over_canonical_task_event_cluster_scores",
                    "min_episode_coverage": .5, "canonical_rows": len(keep),
                    "source_episodes": 500, "source_index_sha256": topk["activation_index_sha256"],
                    "sae_sha256": SAE_SHA,
                    "cluster_scope": "frozen_inherited_full500_clusters",
                    "timestep_scope": "full_inherited_timeline_not_eight_query_head_sample",
                    "historical_runtime_flags_verified": False,
                    "topk_truncation": "inherited_per_token_top64_event_input_not_lossless_head_latents"}


def complete_dictionary_scores(active_result, event_vector, *, dictionary_size=DICT_SIZE):
    """Inactive Head features are exactly zero, but retain their Event score."""
    if len(event_vector) != dictionary_size:
        raise ValueError("Event vector width differs from the dictionary")
    active = {}
    for row in active_result["feature_scores"]:
        fid = row.get("feature_id")
        if type(fid) is not int or not 0 <= fid < dictionary_size or fid in active:
            raise ValueError("Duplicate/out-of-range active feature ID")
        active[fid] = row
    result = []
    for fid in range(dictionary_size):
        row = active.get(fid)
        values = {"feature_id": fid, "mean_head_kl": float(row["mean_head_kl"]) if row else 0.0,
                  "tail_head_kl": float(row["tail_head_kl"]) if row else 0.0,
                  "event_max": float(event_vector[fid]),
                  "num_active_readouts": int(row.get("num_active_readouts", 0)) if row else 0,
                  "head_inactive_zero": row is None}
        if not all(math.isfinite(values[key]) for key in ("mean_head_kl", "tail_head_kl", "event_max")):
            raise ValueError("Nonfinite dictionary score")
        result.append(values)
    return result


def score_head_only_cache(cache, head, sae, *, feature_ids=None, tail_fraction=.1,
                          max_pairs=10_000_000, current_identity=None,
                          parity_reports=None, progress=None):
    """Exact original Mean/Tail arithmetic without unused direction controls.

    This isolated specialization preserves the original head-rank scorer's
    canonical traversal, residual-preserving ``apply_edit``, isolated native
    dtype head calls, FP32 KL and Python-fsum reductions. Only the additional
    control-direction calls and the unrequested direction-excess statistic
    are omitted. Inactive positions/queries remain zero in all denominators.
    It neither re-encodes the SAE nor substitutes a Taylor approximation.
    """
    import torch
    from event_sae.research.head_followup.interventions import apply_edit
    from event_sae.research.head_rank.scoring import (
        _feature_ids, _integer, _parity, aggregate_queries, query_index,
    )
    from event_sae.research.output_head.sensitivity import _validate_sparse_cache, full_vocab_kl

    hidden, records, ids, values, width = _validate_sparse_cache(cache)
    if cache.get("encoder_grouping") != "original_forward":
        raise ValueError("Team500 requires original_forward sparse latents")
    synthetic = _parity(cache, head, current_identity, parity_reports)
    _integer(max_pairs, "max_pairs", 1)
    groups, queries, row_to_query = query_index(records)
    aggregate_queries([0.] * len(queries), queries, tail_fraction)
    observed = sorted({fid for row in ids for fid in row.tolist()})
    selected = _feature_ids(feature_ids if feature_ids is not None else observed, width, "feature_ids")
    selected_set = set(selected)
    per_row = [[fid for fid in row.tolist() if fid in selected_set] for row in ids]
    active_pairs = sum(len(row) for row in per_row)
    if active_pairs > max_pairs:
        raise ValueError(f"Team500 Head pair budget exceeded: {active_pairs} > {max_pairs}")
    accumulated = {fid: {} for fid in selected}
    counts = {fid: 0 for fid in selected}
    device = head.head_weight.device
    executed, baseline_calls, max_cast_error = 0, 0, 0.

    with torch.inference_mode():
        for completed, index in enumerate((row for group in groups for row in group), 1):
            anchors = per_row[index]
            if anchors:
                h = hidden[index:index + 1].to(device=device, dtype=head.hidden_dtype)
                z = torch.zeros((1, width), dtype=torch.float32, device=device)
                z[0, ids[index].to(device)] = values[index].to(device)
                base = head(h)
                baseline_calls += 1
                query = row_to_query[index]
                for fid in anchors:
                    edited, diagnostic = apply_edit(h, sae, fid, 0., encoded=z)
                    real_kl = float(full_vocab_kl(base, head(edited))[0])
                    max_cast_error = max(max_cast_error, float(diagnostic["cast_norm_abs_error"][0]))
                    accumulated[fid].setdefault(query, []).append(real_kl)
                    counts[fid] += 1
                    executed += 1
            if progress and (completed % 100 == 0 or completed == len(records)):
                progress({"readouts": completed, "total_readouts": len(records),
                          "executed_pairs": executed, "total_pairs": active_pairs})
    if executed != active_pairs:
        raise RuntimeError("Team500 executed-pair count differs from preflight")
    rows = []
    for fid in selected:
        query_values = [0.] * len(queries)
        for number, row in sorted(accumulated[fid].items()):
            query_values[number] = math.fsum(row) / 7
        reduced = aggregate_queries(query_values, queries, tail_fraction)
        rows.append({"feature_id": fid, "mean_head_kl": reduced["mean"],
                     "tail_head_kl": reduced["tail"], "task_scores": reduced["tasks"],
                     "num_active_readouts": counts[fid], "num_readouts": len(records),
                     "num_active_queries": len(accumulated[fid]), "num_queries": len(queries)})
    return {"schema_version": "team500_head_only_scores_v1", "synthetic": synthetic,
            "feature_scores": rows, "feature_ids": selected, "dictionary_size": width,
            "aggregation": "seven_positions_mean_then_equal_task_query_mean_balanced_episodes",
            "tail_aggregation": "per_task_top_ceil_fraction_query_mean_then_equal_task_mean_including_inactive_zeros",
            "tail_fraction": float(tail_fraction), "direction_control_calls": 0,
            "diagnostics": {"active_pairs": active_pairs, "planned_pairs": active_pairs,
                            "executed_pairs": executed, "baseline_head_calls": baseline_calls,
                            "max_cast_norm_abs_error": max_cast_error, "direction_control_calls": 0,
                            "device": str(device), "hidden_dtype": str(head.hidden_dtype),
                            "encoder_grouping": "original_forward", "pair_batch_size": 1},
            "new_rollouts": 0, "model_regeneration": False}


def score_workflow(config_path, event_scores_path, output_path, *, smoke=False,
                   max_pairs=10_000_000, progress=None, event_topk_manifest=None):
    """Require fresh local head/edit parity, then score only the new full500 cache."""
    import torch
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import (
        numerical_identity, load_parity, _head_path, _sae_paths, sae_implementation_identity,
    )
    from event_sae.research.output_head.head import load_output_head
    from event_sae.research.output_head.readouts import load_readout_cache
    from event_sae.openvla.activations import load_batch_topk_sae
    from event_sae.research.head_rank.workflow import subset_cache

    cfg = load_config(config_path)
    root = Path(cfg["output"]["root_dir"])
    weights, sae_config = _sae_paths(cfg)
    topk_path = (Path(event_topk_manifest) if event_topk_manifest else
                 Path(event_scores_path).resolve().parents[1] / "topk/manifest.json")
    destination = assert_safe_output(output_path, [root, config_path, event_scores_path, topk_path,
                                                   weights.parent, cfg["inputs"]["dense_dir"]])
    if destination.exists():
        raise FileExistsError("Choose a new Team500 score path; existing outputs are immutable")
    if (cfg["model"]["revision"] != MODEL_REVISION or cfg["model"]["code_revision"] != CODE_REVISION
            or cfg["scope"]["layer_idx"] != 31 or cfg["scope"]["capture_target"] != "post_mlp_residual"
            or cfg["model"]["allow_download"] is not False or sha256_file(weights) != SAE_SHA):
        raise ValueError("Frozen OpenVLA/SAE identity differs")
    if (cfg["scoring"]["alpha"] != 0 or cfg["scoring"]["pair_batch_size"] != 1
            or cfg["scoring"]["arithmetic_mode"] != "isolated_row_bf16_reference_v1"
            or cfg["scoring"]["reduction_dtype"] != "float32"):
        raise ValueError("Team500 requires exact isolated-row BF16 edits and FP32 KL reductions")
    sample = validate_full500_sample(read_json(root / "sample_manifest.json"))
    if sample["source_index_file_sha256"] != sha256_file(sample["source_index_path"]):
        raise ValueError("Source activation index changed after full500 sampling")
    identity = numerical_identity(cfg)
    parity = load_parity(cfg, identity)
    cache = load_readout_cache(root / "readouts")
    if (cache.get("synthetic") is not False or cache.get("records") != sample["readouts"]
            or cache.get("dict_size") != DICT_SIZE or cache.get("encoder_grouping") != "original_forward"
            or cache["manifest"].get("sample_hash") != sample["sample_hash"]):
        raise ValueError("Cache does not match the complete real full500 original-forward sample")
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights), "encoder_implementation": sae_implementation_identity(type(sae))})
    if cache["manifest"].get("sae_id") != sae_id:
        raise ValueError("Cache SAE checkpoint/encoder implementation differs")
    event_sha = sha256_file(event_scores_path)
    payload = torch.load(event_scores_path, map_location="cpu", weights_only=True)
    event_vector, event_binding = validate_event_payload(payload, sample, read_json(topk_path), event_sha)
    event_binding.update(topk_manifest_sha256=sha256_file(topk_path), source_path=str(Path(event_scores_path).resolve()))
    active_ids = sorted({int(fid) for ids in cache["feature_ids"] for fid in ids.tolist()})
    if smoke:
        cache = subset_cache(cache, episodes_per_task=1, seed=1729)
    head = load_output_head(_head_path(cfg), device=cfg["scoring"]["device"])
    implementation = implementation_identity()
    start = time.perf_counter()
    scored = score_head_only_cache(cache, head, sae, feature_ids=active_ids,
                                   max_pairs=max_pairs, current_identity=identity,
                                   parity_reports=parity, progress=progress)
    current_cases = ordered_cases(sorted(({
        "suite": row["suite"], "task_id": row["task_id"], "task_episode_idx": row["task_episode_idx"],
        "initial_state_sha256": row["initial_state_sha256"]} for row in sample["discovery_episodes"]),
        key=lambda r: (r["task_id"], r["task_episode_idx"])))
    result = {
        "schema_version": SCORE_SCHEMA, "status": "scored", "synthetic": False,
        "sampling_mode": "smoke" if smoke else "full500", "development_only": bool(smoke),
        "feature_scores": complete_dictionary_scores(scored, event_vector),
        "active_feature_ids": active_ids, "dictionary_size": DICT_SIZE,
        "scope": {"task_ids": list(range(10)), "discovery_episodes": 500, "episodes_per_task": 50,
            "dictionary_size": DICT_SIZE, "sae_sha256": SAE_SHA, "candidate_universe": UNIVERSE,
            "discovery_manifest_hash": sample["discovery_manifest_hash"],
            "discovery_cases_hash": fingerprint(current_cases),
            "sample_manifest_hash": sample["sample_hash"], "source_index_sha256": sample["source_index_file_sha256"],
            "discovery_evaluation_state_overlap": True, "sae_training_evaluation_overlap": True,
            "historical_state_identity_verified": False, "historical_runtime_flags_verified": False,
            "sampled_queries_per_episode": 8, "head_timestep_scope": "eight_evenly_spaced_queries_per_episode",
            "event_timestep_scope": "full_inherited_timeline"},
        "identity": identity, "event_max_binding": event_binding,
        "diagnostics": {**scored["diagnostics"], "dictionary_features": DICT_SIZE,
                        "readout_active_features": len(active_ids), "readouts": len(cache["records"]),
                        "scored_episodes": len({r["episode_uid"] for r in cache["records"]})},
        "aggregation": scored["aggregation"], "tail_aggregation": scored["tail_aggregation"],
        "tail_fraction": .1, "inactive_handling": "exact_zero_Head_scores_without_excluding_Event_candidates",
        "source_cache_manifest_sha256": sha256_file(root / "readouts/manifest.json"),
        "source_sample_sha256": sha256_file(root / "sample_manifest.json"),
        "sae_config_sha256": sha256_file(sae_config), "implementation": implementation,
        "elapsed_seconds": time.perf_counter() - start,
        "new_rollouts": 0, "model_regeneration": False, "sae_training": False,
        "direction_controls": "not_computed_no_direction_excess_in_this_study",
        "scientific_scope": "retrospective_full500_overlap_output_sensitivity_not_held_out_prediction",
    }
    if implementation_identity() != implementation:
        raise ValueError("Scoring source changed while executing")
    result["score_hash"] = fingerprint(result)
    if not smoke:
        validate_scores(result)
    atomic_write_json(destination, result)
    return {"status": "scored", "output": str(destination), "sampling_mode": result["sampling_mode"],
            "development_only": bool(smoke), "features": DICT_SIZE, "active_features": len(active_ids),
            "score_hash": result["score_hash"], "elapsed_seconds": result["elapsed_seconds"],
            "diagnostics": result["diagnostics"], "new_rollouts": 0}
