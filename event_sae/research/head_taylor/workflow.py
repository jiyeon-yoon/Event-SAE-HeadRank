"""Read-only source binding and separate outputs for the Taylor comparison."""
from __future__ import annotations

from pathlib import Path

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, read_json, sha256_file,
    implementation_fingerprint, git_identity,
)

REPO = Path(__file__).resolve().parents[3]


def implementation_identity():
    files = sorted(Path(__file__).parent.glob("*.py")) + [
        REPO / "scripts/openvla/headtaylor.py",
        REPO / "event_sae/research/head_rank/workflow.py",
        REPO / "event_sae/research/head_rank/scoring.py",
    ]
    return {"base_implementation_hash": implementation_fingerprint(),
            "files": {str(p.relative_to(REPO)): sha256_file(p) for p in files if p.is_file()}}


def score_workflow(config, reference_result, baseline_scores, output, *, full_readouts=False,
                   max_readouts=20000, max_active_pairs=1000000, benchmark_kl=False,
                   progress=None):
    if full_readouts and benchmark_kl:
        raise ValueError("Use --benchmark-kl only on smoke; full reference rerun is not implicit")
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import (
        numerical_identity, load_parity, _head_path, _sae_paths, sae_implementation_identity,
    )
    from event_sae.research.output_head.head import load_output_head
    from event_sae.research.output_head.readouts import load_readout_cache
    from event_sae.research.head_rank.workflow import (
        subset_cache, safe_output, validate_full_score_population,
    )
    from event_sae.openvla.activations import load_batch_topk_sae
    from .scoring import score_cache, benchmark_exact_kl
    from .provenance import protected_source_roots

    cfg, root = load_config(config), Path(reference_result).resolve()
    weights, _ = _sae_paths(cfg)
    target = safe_output(output, [root, config, baseline_scores, weights.parent,
                                 cfg["output"]["root_dir"]])
    assert_safe_output(target, protected_source_roots())
    assert_safe_output(cfg["output"]["root_dir"], [root])
    if (cfg["scoring"]["alpha"] != 0 or cfg["scoring"]["pair_batch_size"] != 1
            or cfg["scoring"]["reduction_dtype"] != "float32"
            or cfg["scoring"]["negative_tolerance"] != 1e-6):
        raise ValueError("Frozen alpha0 isolated-row FP32-KL runtime config required")
    identity = numerical_identity(cfg)
    reports = load_parity(cfg, identity)
    source_paths = [root / "sample_manifest.json", root / "scores/scores.json",
                    root / "readouts/manifest.json", Path(baseline_scores), weights,
                    weights.parent / "config.json"]
    before = {str(path): sha256_file(path) for path in source_paths}
    sample, old = read_json(source_paths[0]), read_json(source_paths[1])
    baseline = read_json(baseline_scores)
    if baseline.get("score_hash") != fingerprint({k: v for k, v in baseline.items() if k != "score_hash"}):
        raise ValueError("Historical baseline scores have an invalid integrity hash")
    validate_full_score_population(baseline)
    if (baseline.get("synthetic") is not False or baseline.get("development_only") is not False
            or baseline.get("feature_set") != "all-active"
            or baseline["scope"] != old["scope"]
            or baseline["source_cache_manifest_sha256"] != before[str(source_paths[2])]
            or baseline["source_scores_sha256"] != before[str(source_paths[1])]
            or baseline["source_sample_sha256"] != before[str(source_paths[0])]):
        raise ValueError("Historical baselines do not bind the identical full frozen sample")
    if (before[str(source_paths[2])] != old["identity"]["cache_manifest_hash"]
            or sample["sample_hash"] != old["scope"]["sample_manifest_hash"]
            or identity["sae_checkpoint_hash"] != old["scope"]["sae_sha256"]
            or identity["head_weights_hash"] != old["identity"]["head_weights_hash"]):
        raise ValueError("Frozen cache/sample/SAE/head binding mismatch")
    for key in ("model_revision", "code_revision", "layer_idx", "head_weights_hash", "sae_checkpoint_hash"):
        if baseline["identity"].get(key) != identity.get(key):
            raise ValueError(f"Baseline and current Taylor identity differ: {key}")
    cache = load_readout_cache(root / "readouts")
    if (cache.get("synthetic") is not False or cache["records"] != sample["readouts"]
            or cache["dict_size"] != old["scope"]["dictionary_size"]):
        raise ValueError("Require the real exact original-forward sample")
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights),
                          "encoder_implementation": sae_implementation_identity(type(sae))})
    if (cache["manifest"]["sae_id"] != sae_id
            or cache["manifest"]["sample_hash"] != sample["sample_hash"]):
        raise ValueError("SAE/grouping differs from the frozen cache")
    active_ids = sorted({int(fid) for row in cache["feature_ids"] for fid in row.tolist()})
    if active_ids != baseline["frozen_active_feature_ids"]:
        raise ValueError("Baseline active universe differs from the exact readout cache")
    if fingerprint(cache["records"]) != baseline["readout_record_hash"]:
        raise ValueError("Baseline readout order/identity differs from the cache")
    if not full_readouts:
        cache = subset_cache(cache)
    head = load_output_head(_head_path(cfg), device=cfg["scoring"]["device"])
    code = implementation_identity()
    result = score_cache(cache, head, sae, max_readouts=max_readouts,
                         max_active_pairs=max_active_pairs, current_identity=identity,
                         parity_reports=reports, progress=progress)
    if full_readouts and result["scored_feature_ids"] != active_ids:
        raise ValueError("Incomplete full Taylor active universe")
    old_rows = {row["feature_id"]: row for row in baseline["feature_scores"]}
    for row in result["features"]:
        previous = old_rows[row["feature_id"]]
        for key in ("mean_head_kl", "tail_head_kl", "event_max"):
            row[key] = previous[key]
        row["readout_activation_frequency"] = previous["reference_readout_activation_frequency"]
        row["mean_edit_norm"] = previous["reference_mean_edit_norm"]
    if benchmark_kl:
        result["benchmark_exact_kl"] = benchmark_exact_kl(cache, head, sae,
            max_active_pairs=max_active_pairs, progress=progress)
        result["benchmark_exact_kl"]["taylor_including_preparation_seconds"] = result["diagnostics"]["total_seconds"]
        result["benchmark_exact_kl"]["timing_is_smoke_only"] = True
    if code != implementation_identity() or any(sha256_file(path) != digest for path, digest in before.items()):
        raise ValueError("Source code or frozen source input changed during scoring")
    result.update(status="scored", sampling_mode="full" if full_readouts else "smoke",
                  development_only=not full_readouts, scope=old["scope"], identity=identity,
                  implementation=code, git=git_identity(REPO),
                  frozen_active_feature_ids=active_ids, source_sha256=before,
                  source_readonly_roots=[str(root), str(weights.parent.resolve()),
                                         str(Path(baseline_scores).resolve())],
                  scoring_sample={"readouts_used": len(cache["records"]),
                                  "discovery_episodes_used": len({r["episode_uid"] for r in cache["records"]}),
                                  "same_sample_as_frozen_baselines": full_readouts},
                  readout_record_hash=fingerprint(cache["records"]),
                  baseline_scores_hash=baseline["score_hash"],
                  baseline_origin="historical_frozen_full_scores_not_recomputed_current_device",
                  event_max_binding=baseline["event_max_binding"],
                  model_download=False,
                  scientific_scope="retrospective selected-panel comparison; no unseen-feature outcome claim")
    result["score_hash"] = fingerprint(result)
    atomic_write_json(target, result)
    return {"status": "scored", "output": str(target), "features": len(result["features"]),
            "sampling_mode": result["sampling_mode"], "development_only": result["development_only"],
            "score_hash": result["score_hash"], "diagnostics": result["diagnostics"], "new_rollouts": 0}
