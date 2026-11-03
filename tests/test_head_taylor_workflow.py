"""Adapter/source protection tests; the actual GPU path remains unexecuted."""
import copy
import json

import pytest
import torch

from event_sae.research.head_taylor import workflow
from event_sae.research.output_head.provenance import fingerprint, sha256_file


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(tmp_path, monkeypatch):
    from event_sae.research.output_head import config, workflow as native, head, readouts
    from event_sae.research.head_taylor import scoring
    from event_sae.openvla import activations
    root = tmp_path / "reference"
    weights = tmp_path / "sae/ae.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"sae fixture")
    save(weights.parent / "config.json", {})
    cfg_path = tmp_path / "config/runtime.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text("fixture")
    cfg = {"scoring": {"device": "cpu", "alpha": 0., "pair_batch_size": 1,
                       "reduction_dtype": "float32", "negative_tolerance": 1e-6},
           "output": {"root_dir": str(tmp_path / "runtime")}}
    rows = [{"task_id": 0, "episode_uid": "a", "step_in_episode": 0,
             "action_dim_index": i} for i in range(7)]
    sample = {"sample_hash": "s", "readouts": rows}
    save(root / "sample_manifest.json", sample)
    save(root / "readouts/manifest.json", {})
    sae = object()
    identity = {"sae_checkpoint_hash": sha256_file(weights), "head_weights_hash": "h",
                "model_revision": "m", "code_revision": "c", "layer_idx": 31}
    scope = {"sample_manifest_hash": "s", "sae_sha256": sha256_file(weights),
             "dictionary_size": 3, "task_ids": [0]}
    old = {"scope": scope, "identity": {"cache_manifest_hash": sha256_file(root / "readouts/manifest.json"),
                                       "head_weights_hash": "h"}}
    save(root / "scores/scores.json", old)
    baseline = {"schema_version": "head_rank_scores_v1", "sampling_mode": "full",
        "synthetic": False, "development_only": False, "feature_set": "all-active",
        "scope": scope, "identity": copy.deepcopy(identity), "feature_ids": [0], "frozen_active_feature_ids": [0],
        "scored_feature_ids": [0], "readout_record_hash": fingerprint(rows),
        "source_cache_manifest_sha256": sha256_file(root / "readouts/manifest.json"),
        "source_scores_sha256": sha256_file(root / "scores/scores.json"),
        "source_sample_sha256": sha256_file(root / "sample_manifest.json"),
        "scoring_sample": {"readouts_used": 7, "discovery_episodes_used": 1,
                           "event_max_matches_scoring_episode_population": True},
        "query_metadata": [{"query_index": 0, "task_id": 0, "episode_uid": "a",
                            "step_in_episode": 0, "num_action_positions": 7}],
        "feature_scores": [{"feature_id": 0, "num_queries": 1, "num_readouts": 7,
            "mean_head_kl": .2, "tail_head_kl": .3, "event_max": .4,
            "reference_mean_edit_norm": .5, "reference_readout_activation_frequency": 1.}],
        "event_max_binding": {"aggregation": "max_over_canonical_task_event_rows"}}
    baseline["score_hash"] = fingerprint(baseline)
    baseline_path = tmp_path / "baseline/full-scores.json"
    save(baseline_path, baseline)
    cache = {"synthetic": False, "records": rows, "dict_size": 3,
        "hidden": torch.ones(7, 2), "feature_ids": [torch.tensor([0])] * 7,
        "feature_values": [torch.tensor([.1])] * 7,
        "manifest": {"sample_hash": "s", "sae_id": fingerprint({"weights": sha256_file(weights),
                                                                  "encoder_implementation": "enc"})}}
    monkeypatch.setattr(config, "load_config", lambda p: copy.deepcopy(cfg))
    monkeypatch.setattr(native, "_sae_paths", lambda c: (weights, weights.parent / "config.json"))
    monkeypatch.setattr(native, "_head_path", lambda c: tmp_path / "head")
    monkeypatch.setattr(native, "numerical_identity", lambda c: copy.deepcopy(identity))
    monkeypatch.setattr(native, "load_parity", lambda c, i: {})
    monkeypatch.setattr(native, "sae_implementation_identity", lambda cls: "enc")
    monkeypatch.setattr(readouts, "load_readout_cache", lambda p: copy.deepcopy(cache))
    monkeypatch.setattr(activations, "load_batch_topk_sae", lambda p, device: (sae, {}))
    monkeypatch.setattr(head, "load_output_head", lambda p, device: object())
    monkeypatch.setattr(scoring, "score_cache", lambda *a, **kw: {
        "synthetic": False, "features": [{"feature_id": 0, "taylor_abs_nll": .1}],
        "scored_feature_ids": [0], "diagnostics": {}, "new_rollouts": 0})
    return cfg_path, root, baseline_path, baseline, weights


def test_frozen_binding_and_exact_metric_join_without_modifying_inputs(tmp_path, monkeypatch):
    cfg, root, baseline, _, weights = fixture(tmp_path, monkeypatch)
    before = {p: sha256_file(p) for p in (cfg, baseline, weights, root / "scores/scores.json")}
    output = tmp_path / "outputs/full.json"
    result = workflow.score_workflow(cfg, root, baseline, output, full_readouts=True)
    data = json.loads(output.read_text())
    assert data["features"][0]["event_max"] == .4
    assert data["features"][0]["mean_head_kl"] == .2
    assert data["development_only"] is False and result["new_rollouts"] == 0
    assert data["score_hash"] == fingerprint({k: v for k, v in data.items() if k != "score_hash"})
    assert all(sha256_file(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError):
        workflow.score_workflow(cfg, root, baseline, output, full_readouts=True)


@pytest.mark.parametrize("change", ["tamper", "scope", "source", "identity", "records"])
def test_mismatched_inputs_are_blocked(tmp_path, monkeypatch, change):
    cfg, root, path, baseline, _ = fixture(tmp_path, monkeypatch)
    if change == "tamper": baseline["feature_scores"][0]["event_max"] = 1.
    if change == "scope": baseline["scope"]["sample_manifest_hash"] = "wrong"
    if change == "source": baseline["source_cache_manifest_sha256"] = "wrong"
    if change == "identity": baseline["identity"]["model_revision"] = "wrong"
    if change == "records": baseline["readout_record_hash"] = "wrong"
    if change != "tamper":
        baseline["score_hash"] = fingerprint({k: v for k, v in baseline.items() if k != "score_hash"})
    save(path, baseline)
    with pytest.raises(ValueError):
        workflow.score_workflow(cfg, root, path, tmp_path / "out.json", full_readouts=True)


def test_source_output_overlap_and_full_timing_rejected(tmp_path, monkeypatch):
    cfg, root, path, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="overlap"):
        workflow.score_workflow(cfg, root, path, root / "new.json", full_readouts=True)
    with pytest.raises(ValueError, match="smoke"):
        workflow.score_workflow("missing", "missing", "missing", "out", full_readouts=True, benchmark_kl=True)
