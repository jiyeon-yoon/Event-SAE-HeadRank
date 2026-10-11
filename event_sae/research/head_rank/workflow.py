"""Read-only frozen-input adapter for Tail and direction-excess scoring.

New outputs never go inside the source result tree. This module neither reads
success labels for scoring nor launches a simulator, download, or SAE training.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, read_json, sha256_file,
    implementation_fingerprint,
)

REPO = Path(__file__).resolve().parents[3]
DEVELOPMENT_FEATURES = (2723, 2875, 3165, 4449, 5169, 7024, 13724, 14146,
                        18471, 22257, 22593, 24360, 24830, 25653, 28858, 30729)


def implementation_identity():
    """Include uncommitted new code, not merely the inherited Git commit."""
    files = sorted(Path(__file__).parent.glob("*.py")) + [
        REPO / "scripts/openvla/headrank.py",
        REPO / "event_sae/research/head_followup/interventions.py",
    ]
    return {"base_implementation_hash": implementation_fingerprint(),
            "files": {str(p.relative_to(REPO)): sha256_file(p) for p in files}}


def safe_output(output, sources=()):
    # Also protect the original checkout if it exists beside this independent clone.
    previous = REPO.parent / "Event-SAE-Head"
    protected = list(sources) + ([previous] if previous.exists() else [])
    target = assert_safe_output(output, protected)
    if target.exists():
        raise FileExistsError(f"Output exists; choose a new path: {target}")
    return target


def load_event_max(path, expected_scope):
    """Exact discovery-bound Event-max, NOT legacy mean or public-500 rebinding."""
    import math
    import torch
    from event_sae.research.output_head.candidates import validate_score_scope

    path = Path(path)
    receipt_path = path.with_suffix(path.suffix + ".binding.json")
    proof = read_json(receipt_path)
    source_hash = sha256_file(path)
    if proof.get("output_sha256") != source_hash:
        raise ValueError("Event score hash differs from its binding receipt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    derivation = payload.get("derivation", {})
    if (payload.get("synthetic") is not False
            or derivation.get("operation") != "discovery_only_reaggregation"
            or derivation.get("signature_hash") != proof.get("signature_hash")
            or fingerprint(derivation.get("signature")) != proof.get("signature_hash")):
        raise ValueError("Require intact discovery-only Event derivation, not the public 500 aggregate")
    if derivation["signature"].get("scope") != payload.get("scope"):
        raise ValueError("Event signature scope differs from payload scope")
    required = ("task_ids", "dictionary_size", "discovery_manifest_hash", "split_manifest_hash", "sae_sha256")
    if any(expected_scope.get(k) is None for k in required):
        raise ValueError("Head discovery scope is incomplete")
    validate_score_scope(payload.get("scope", {}), expected_scope)
    rows, matrix = payload.get("row_keys"), payload.get("matrix_raw")
    if not isinstance(rows, list) or not rows or not isinstance(matrix, torch.Tensor):
        raise ValueError("Event-max requires explicit row_keys and matrix_raw")
    if (matrix.ndim != 2 or list(matrix.shape) != [len(rows), expected_scope["dictionary_size"]]
            or not torch.isfinite(matrix).all()):
        raise ValueError("Malformed Event raw matrix")
    if sorted({r.get("task_id") for r in rows}) != sorted(expected_scope["task_ids"]):
        raise ValueError("Event row tasks differ from Head task scope")
    keep = []
    for i, row in enumerate(rows):
        value = row.get("episode_coverage")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Event row coverage must be explicit and finite")
        if value >= .5:
            keep.append(i)
    if not keep or {rows[i]["task_id"] for i in keep} != set(expected_scope["task_ids"]):
        raise ValueError("Every task needs a canonical Event row at coverage >= 0.5")
    vector = matrix[keep].float().max(dim=0).values.tolist()
    return vector, {
        "source_sha256": source_hash, "receipt_sha256": sha256_file(receipt_path),
        "signature_hash": proof["signature_hash"], "scope": payload["scope"],
        "aggregation": "max_over_canonical_task_event_rows", "min_episode_coverage": .5,
        "canonical_rows": len(keep),
        "timestep_comparability": "same_discovery_episodes_not_same_sampled_timesteps",
    }


def subset_cache(cache, *, episodes_per_task=1, seed=1729):
    """Seeded whole episodes for smoke only; keep all seven positions per query."""
    if type(episodes_per_task) is not int or episodes_per_task < 1:
        raise ValueError("episodes_per_task must be a positive integer")
    tasks = {}
    for row in cache["records"]:
        tasks.setdefault(row["task_id"], set()).add(row["episode_uid"])
    rng = random.Random(seed)
    selected = set()
    for task, episodes in sorted(tasks.items()):
        if len(episodes) < episodes_per_task:
            raise ValueError("Too few discovery episodes for smoke sampling")
        selected.update((task, uid) for uid in rng.sample(sorted(episodes), episodes_per_task))
    indices = [i for i, r in enumerate(cache["records"]) if (r["task_id"], r["episode_uid"]) in selected]
    output = dict(cache)
    output["hidden"] = cache["hidden"][indices]
    for name in ("records", "feature_ids", "feature_values"):
        output[name] = [cache[name][i] for i in indices]
    output["control_pool_feature_ids"] = sorted({int(fid) for ids in cache["feature_ids"] for fid in ids.tolist()})
    return output


def score_workflow(config, reference_result, event_scores, output, *, full_readouts=False,
                   feature_set="development", max_pairs=100000, progress=None):
    """Explicit local scoring only, requiring fresh current-device parity."""
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import (
        numerical_identity, load_parity, _head_path, _sae_paths, sae_implementation_identity,
    )
    from event_sae.research.output_head.head import load_output_head
    from event_sae.research.output_head.readouts import load_readout_cache
    from event_sae.openvla.activations import load_batch_topk_sae
    from .scoring import score_cache

    if feature_set not in ("development", "all-active"):
        raise ValueError("Use development or all-active feature set")
    cfg, root = load_config(config), Path(reference_result)
    weights, _ = _sae_paths(cfg)
    target = safe_output(output, [root, config, event_scores, weights.parent, cfg["output"]["root_dir"]])
    assert_safe_output(cfg["output"]["root_dir"], [root, REPO.parent / "Event-SAE-Head"])
    if cfg["scoring"]["alpha"] != 0 or cfg["scoring"]["pair_batch_size"] != 1:
        raise ValueError("Complete removal with isolated-row reference arithmetic is required")
    if cfg["scoring"]["reduction_dtype"] != "float32" or cfg["scoring"]["negative_tolerance"] != 1e-6:
        raise ValueError("HeadRank v1 requires FP32 KL reduction and negative_tolerance=1e-6")
    identity = numerical_identity(cfg)
    reports = load_parity(cfg, identity)
    sample, old = read_json(root / "sample_manifest.json"), read_json(root / "scores/scores.json")
    if (sha256_file(root / "readouts/manifest.json") != old["identity"]["cache_manifest_hash"]
            or sample["sample_hash"] != old["scope"]["sample_manifest_hash"]
            or identity["sae_checkpoint_hash"] != old["scope"]["sae_sha256"]
            or identity["head_weights_hash"] != old["identity"]["head_weights_hash"]):
        raise ValueError("Frozen cache/sample/SAE/head binding mismatch")
    event_vector, event_binding = load_event_max(event_scores, old["scope"])
    cache = load_readout_cache(root / "readouts")
    if (cache.get("synthetic") is not False or cache["records"] != sample["readouts"]
            or cache["dict_size"] != old["scope"]["dictionary_size"]):
        raise ValueError("Require real cache with exact frozen sample records")
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights), "encoder_implementation": sae_implementation_identity(type(sae))})
    if cache["manifest"]["sae_id"] != sae_id or cache["manifest"]["sample_hash"] != sample["sample_hash"]:
        raise ValueError("SAE/grouping differs from frozen cache")
    active_ids = sorted({int(fid) for ids in cache["feature_ids"] for fid in ids.tolist()})
    features = list(DEVELOPMENT_FEATURES) if feature_set == "development" else active_ids
    if not full_readouts:
        cache = subset_cache(cache)
    cache["control_pool_feature_ids"] = active_ids
    head = load_output_head(_head_path(cfg), device=cfg["scoring"]["device"])
    source_code = implementation_identity()
    started = time.perf_counter()
    result = score_cache(cache, head, sae, feature_ids=features, max_pairs=max_pairs,
                         current_identity=identity, parity_reports=reports, progress=progress)
    old_rows = {r["feature_id"]: r for r in old["feature_scores"]}
    for row in result["feature_scores"]:
        row["event_max"] = event_vector[row["feature_id"]]
        for metric in ("mean_edit_norm", "mean_readout_activation", "readout_activation_frequency"):
            row["reference_" + metric] = old_rows[row["feature_id"]][metric]
    replay_error = None
    if full_readouts:
        replay_error = max(abs(r["mean_head_kl"] - old_rows[r["feature_id"]]["head_full_vocab_kl_fixed_prefix"])
                           for r in result["feature_scores"])
    result.update(status="scored", sampling_mode="full" if full_readouts else "smoke",
        feature_set=feature_set, development_only=feature_set == "development" or not full_readouts,
        scope=old["scope"], event_max_binding=event_binding,
        identity=identity, implementation=source_code,
        readout_record_hash=fingerprint(cache["records"]),
        scoring_sample={"discovery_episodes_used": len({r["episode_uid"] for r in cache["records"]}),
                        "readouts_used": len(cache["records"]),
                        "event_max_matches_scoring_episode_population": full_readouts},
        source_scores_sha256=sha256_file(root / "scores/scores.json"),
        source_sample_sha256=sha256_file(root / "sample_manifest.json"),
        source_cache_manifest_sha256=sha256_file(root / "readouts/manifest.json"),
        frozen_active_feature_ids=active_ids, scored_feature_ids=features,
        source_readonly_roots=[str(root.resolve()), str(weights.parent.resolve()), str(Path(event_scores).parent.resolve())],
        elapsed_seconds=time.perf_counter() - started,
        replay_mean_head_max_abs_error=replay_error,
        new_rollouts=0, model_download=False, sae_training=False)
    if source_code != implementation_identity():
        raise ValueError("Scorer source changed during execution; do not publish mixed-code scores")
    result["score_hash"] = fingerprint(result)
    atomic_write_json(target, result)
    return {"status": "scored", "output": str(target), "new_rollouts": 0,
            "sampling_mode": result["sampling_mode"], "features": len(features),
            "elapsed_seconds": result["elapsed_seconds"], "score_hash": result["score_hash"],
            "development_only": result["development_only"],
            "diagnostics": result["diagnostics"],
            "next_gate": "inspect_norm_eligibility_and_timing_before_expanding"}


def plan_workflow(scores_path, output, *, registry, eval_manifest=None, feature_count=8,
                  cases_per_task=5, primary_metric="tail_head_kl"):
    """Freeze predictions without reading any raw/edited success labels."""
    from .protocol import plan_panel
    scores = read_json(scores_path)
    saved_hash = scores.get("score_hash")
    if saved_hash != fingerprint({k: v for k, v in scores.items() if k != "score_hash"}):
        raise ValueError("Scores missing/mismatched integrity hash")
    if scores.get("development_only") is not False or scores.get("feature_set") != "all-active":
        raise ValueError("Blind selection requires full all-active scores, not development/smoke")
    validate_full_score_population(scores)
    if scores.get("implementation") != implementation_identity():
        raise ValueError("Source changed after scoring; inspect differences before freezing a new plan")
    known = read_json(registry)
    if known.get("reviewed_for_team_prior_outcomes") is not True:
        raise ValueError("Confirm the additional known-feature registry with your team first")
    additional = known.get("feature_ids")
    if not isinstance(additional, list):
        raise ValueError("Registry needs feature_ids, even if an explicitly reviewed empty list")
    sources = [scores_path, registry, eval_manifest, *scores.get("source_readonly_roots", [])]
    target = safe_output(output, sources)
    evaluation = read_json(eval_manifest) if eval_manifest else None
    cases = evaluation.get("cases") if evaluation else None
    bindings = {"scores_sha256": sha256_file(scores_path), "score_hash": saved_hash,
                "registry_sha256": sha256_file(registry), "implementation": implementation_identity(),
                "event_max_binding": scores["event_max_binding"], "score_scope": scores["scope"],
                "eval_manifest_sha256": sha256_file(eval_manifest) if eval_manifest else None,
                "evaluation_label_history": evaluation.get("evaluation_labels_previously_used") if evaluation else None}
    plan = plan_panel(scores, additional_evaluated_ids=additional, feature_count=feature_count,
        cases_per_task=cases_per_task, task_ids=scores["scope"]["task_ids"],
        primary_metric=primary_metric, reference_bindings=bindings, evaluation_cases=cases)
    atomic_write_json(target, plan)
    return {"status": plan["status"], "output": str(target), "plan_hash": plan["plan_hash"],
            "new_rollouts": 0, "rollout_execution": "requires_prepare_run_and_explicit_execution_approval"}


def validate_full_score_population(scores):
    """Do not let a partial restore masquerade as an all-active full score file."""
    if scores.get("schema_version") != "head_rank_scores_v1" or scores.get("sampling_mode") != "full":
        raise ValueError("Full HeadRank score schema required")
    ids = [r["feature_id"] for r in scores.get("feature_scores", [])]
    if (not ids or len(set(ids)) != len(ids)
            or ids != scores.get("frozen_active_feature_ids")
            or ids != scores.get("scored_feature_ids") or ids != scores.get("feature_ids")):
        raise ValueError("Score ID population differs from complete frozen active universe")
    sample = scores.get("scoring_sample", {})
    queries = scores.get("query_metadata", [])
    readouts = sample.get("readouts_used")
    if (sample.get("event_max_matches_scoring_episode_population") is not True
            or not queries or type(readouts) is not int or readouts != 7 * len(queries)
            or any(q.get("num_action_positions") != 7 for q in queries)
            or [q.get("query_index") for q in queries] != list(range(len(queries)))
            or len({(q.get("task_id"), q.get("episode_uid"), q.get("step_in_episode")) for q in queries}) != len(queries)
            or sorted({q.get("task_id") for q in queries}) != sorted(scores["scope"]["task_ids"])
            or any(r.get("num_readouts") != readouts or r.get("num_queries") != len(queries)
                   for r in scores["feature_scores"])):
        raise ValueError("Incomplete query/readout population in full scores")
    if len({q["episode_uid"] for q in queries}) != sample.get("discovery_episodes_used"):
        raise ValueError("Scoring episode population differs from query metadata")
