"""Metadata and score-joining tests. No simulator, native policy or GPU runs."""
import copy
import json
from pathlib import Path

import pytest
import torch

from event_sae.research.output_head.provenance import fingerprint, sha256_file
from event_sae.research.output_head.readouts import MAPPING_VERSION, discovery_population_hash
from event_sae.research.team500.protocol import DICT_SIZE, SAE_SHA
from event_sae.research.team500.readouts import MODE, build_full500_manifest, validate_full500_sample
from event_sae.research.team500.scoring import (
    PUBLIC_EVENT_SHA, complete_dictionary_scores, score_head_only_cache, validate_event_payload,
)


def sample_fixture():
    episodes, rows = [], []
    for task in range(10):
        for trial in range(50):
            number = task * 50 + trial
            uid = fingerprint([task, trial, "episode"])
            state = fingerprint([task, trial, "state"])
            episodes.append({"episode_uid": uid, "source_run_id": "fixture", "suite": "libero_spatial",
                             "task_id": task, "task_episode_idx": trial, "initial_state_sha256": state})
            for step in range(8):
                for dimension in range(7):
                    rows.append({**episodes[-1], "episode_num": number, "step_in_episode": step,
                                 "action_dim_index": dimension})
    gen = {"schema_version": "output_head_generation_interpretation_v1", "layer_idx": 31,
           "action_dim": 7, "batch_size": 1, "padding": "none", "use_cache": True,
           "evidence": {"historical_runtime_flags_verified": False,
                        "source_registry_hash": "c" * 64, "source_index_sha256": "b" * 64,
                        "mapping_assumptions": {"batch_size": 1, "padding": "none", "use_cache": True}}}
    spec = {"mode": MODE, "task_ids": list(range(10)), "max_episodes_per_task": 50,
            "max_steps_per_episode": 8}
    result = {"schema_version": "output_head_readout_manifest_v1", "mapping_version": MAPPING_VERSION,
              "source_index_hash": "a" * 64, "source_index_file_sha256": "b" * 64,
              "source_registry_hash": "c" * 64,
              "sample_spec": spec, "generation_spec": gen, "synthetic": False,
              "discovery_episodes": episodes, "readouts": rows, "split_manifest_hash": None,
              "discovery_manifest_hash": discovery_population_hash(episodes),
              "state_identity_scope": {"historical_state_identity_verified": False,
                  "historical_runtime_flags_verified": False, "discovery_evaluation_state_overlap": True,
                  "sae_training_evaluation_overlap": True}}
    rehash_sample(result)
    return result


def rehash_sample(sample):
    sample["mapping_hash"] = fingerprint({"index": sample["source_index_hash"],
        "generation": sample["generation_spec"], "mapping_version": sample["mapping_version"]})
    sample["sample_hash"] = fingerprint({"mapping_hash": sample["mapping_hash"],
        "sample": sample["sample_spec"], "readouts": sample["readouts"]})


def public_fixture(sample):
    topk = {"format": "token_topk_sparse_v1", "layer": 31, "dict_size": DICT_SIZE,
            "topk": 64, "sae_sha256": SAE_SHA,
            "activation_index_sha256": sample["source_index_file_sha256"],
            "capture_target": None, "dense_dir": "/old/run/sae_activations/post_mlp_residual"}
    matrix = torch.zeros(11, DICT_SIZE)
    matrix[0, 8] = 4.
    matrix[1, 8] = 9.
    matrix[10, 8] = 100.  # Noncanonical row must not win Event-max.
    payload = {"source": {"dict_size": DICT_SIZE, "layer": 31, "allowed_episode_nums": None},
               "window_size": 5, "step_mapping": "inference_step", "matrix_raw": matrix,
               "row_keys": [{"task_id": task, "episode_coverage": .5} for task in range(10)]
                           + [{"task_id": 0, "episode_coverage": .49}],
               "selected_events": [{"task_id": row["task_id"], "episode_num": row["task_id"] * 50 + row["task_episode_idx"]}
                                   for row in sample["discovery_episodes"]],
               "selection_counts": {"skipped_missing_window_vectors": 0}}
    return payload, topk


def test_exact_full500_sample_preserves_unknown_history_and_overlap():
    sample = sample_fixture()
    assert validate_full500_sample(sample) is sample
    assert len(sample["readouts"]) == 28000
    assert sample["state_identity_scope"]["historical_runtime_flags_verified"] is False


@pytest.mark.parametrize("mutation", [
    lambda s: s["sample_spec"].update(mode="pilot"),
    lambda s: s["discovery_episodes"].pop(),
    lambda s: s["readouts"].pop(),
    lambda s: s["readouts"][0].update(action_dim_index=1),
    lambda s: s["state_identity_scope"].update(historical_runtime_flags_verified=True),
])
def test_reject_250_smoke_incomplete_queries_and_history_laundering(mutation):
    sample = sample_fixture()
    mutation(sample)
    rehash_sample(sample)
    with pytest.raises(ValueError):
        validate_full500_sample(sample)


def test_event_max_is_canonical_cluster_max_not_mean_or_max_activation():
    sample = sample_fixture()
    payload, topk = public_fixture(sample)
    values, binding = validate_event_payload(payload, sample, topk, PUBLIC_EVENT_SHA)
    assert values[8] == 9. and len(values) == DICT_SIZE
    assert binding["canonical_rows"] == 10
    assert binding["source_episodes"] == 500
    assert binding["timestep_scope"] != "eight_query_head_sample"


@pytest.mark.parametrize("field", ["sha", "index", "sae", "filter", "coverage", "missing"])
def test_public_event_pin_population_and_metadata_bindings_cannot_drift(field):
    sample = sample_fixture()
    payload, topk = public_fixture(sample)
    sha = PUBLIC_EVENT_SHA
    if field == "sha": sha = "0" * 64
    elif field == "index": topk["activation_index_sha256"] = "c" * 64
    elif field == "sae": topk["sae_sha256"] = "d" * 64
    elif field == "filter": payload["source"]["allowed_episode_nums"] = list(range(250))
    elif field == "coverage": payload["row_keys"][0]["episode_coverage"] = float("nan")
    elif field == "missing": payload["selection_counts"]["skipped_missing_window_vectors"] = 1
    with pytest.raises(ValueError):
        validate_event_payload(payload, sample, topk, sha)


def test_event_only_feature_remains_candidate_when_head_readout_inactive():
    scored = {"feature_scores": [{"feature_id": 2, "mean_head_kl": 4., "tail_head_kl": 7.,
                                  "num_active_readouts": 3}]}
    rows = complete_dictionary_scores(scored, [0., 100., 3., 0.], dictionary_size=4)
    assert rows[1]["event_max"] == 100.
    assert rows[1]["mean_head_kl"] == rows[1]["tail_head_kl"] == 0.
    assert rows[1]["head_inactive_zero"] is True
    assert rows[2]["num_active_readouts"] == 3
    assert [r["feature_id"] for r in rows] == [0, 1, 2, 3]


def test_full500_mapper_uses_complete_forwards_and_does_not_read_tensors(tmp_path):
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "shard.pt").write_bytes(b"not a torch file: metadata mapper must not deserialize")
    source, episodes, offset = [], [], 0
    for task in range(10):
        for trial in range(50):
            number = task * 50 + trial
            state = fingerprint([task, trial])
            episodes.append({"source_run_id": "fixture", "episode_num": number, "suite": "libero_spatial",
                             "task_id": task, "task_episode_idx": trial, "initial_state_sha256": state,
                             "current_runtime_state_sha256": state, "historical_initial_state_sha256": None})
            for step in range(8):
                for dimension in range(7):
                    length = 2 if dimension == 0 else 1
                    source.append({"source_run_id": "fixture", "episode_num": number,
                                   "task_id": task, "task_episode_idx": trial, "step_in_episode": step,
                                   "global_forward_idx": len(source), "layer_idx": 31,
                                   "shard_path": "shard.pt", "row_start": offset, "row_end": offset + length})
                    offset += length
    index = dense / "activation_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in source))
    registry = {"schema_version": "output_head_current_state_registry_v1", "episodes": episodes,
                "suite": "libero_spatial", "historical_state_identity_verified": False,
                "source_files": {"activation_index": {"sha256": sha256_file(index)}}}
    registry["manifest_hash"] = fingerprint(registry)
    gen = sample_fixture()["generation_spec"]
    gen["evidence"].update(source_registry_hash=registry["manifest_hash"],
                           source_index_sha256=sha256_file(index))
    sample = build_full500_manifest(index, registry, gen, dense_dir=dense)
    assert len(sample["readouts"]) == 28000 and sample["readouts"][0]["source_row"] == 1
    assert sample["readouts"][0]["row_end"] - sample["readouts"][0]["row_start"] == 2
    assert sample["split_manifest_hash"] is None
    assert sample["generation_spec"]["evidence"]["historical_runtime_flags_verified"] is False
    # A duplicate forward does not become a second sample silently.
    source[1]["global_forward_idx"] = 0
    index.write_text("".join(json.dumps(row) + "\n" for row in source))
    registry["source_files"]["activation_index"]["sha256"] = sha256_file(index)
    registry["manifest_hash"] = fingerprint({k: v for k, v in registry.items() if k != "manifest_hash"})
    gen["evidence"].update(source_registry_hash=registry["manifest_hash"],
                           source_index_sha256=sha256_file(index))
    with pytest.raises(ValueError, match="Duplicate"):
        build_full500_manifest(index, registry, gen, dense_dir=dense)


class _ScoringSAE:
    def __init__(self):
        self.calls = 0
        self.columns = torch.tensor([[1., .1], [.2, 1.], [.7, -.7], [-.6, .8]])

    def encode(self, value):
        raise AssertionError("Never re-encode frozen cached original-forward latents")

    def decode(self, value):
        self.calls += 1
        return value @ self.columns.to(value.device) + torch.tensor([.1, -.2], device=value.device)


class _ScoringHead:
    def __init__(self, dtype):
        self.head_weight = torch.tensor([[1., .1], [-.2, .8], [-.7, -.4]], dtype=dtype)
        self.hidden_dtype, self.manifest, self.calls = dtype, {}, 0

    def __call__(self, hidden):
        self.calls += 1
        return (hidden @ self.head_weight.T).float()


def _scoring_cache(dtype):
    records, ids, values = [], [], []
    # Unequal task counts, inactive query/positions and multiple active IDs
    # exercise exactly the denominator/tail behavior of the original kernel.
    for task, steps in ((0, 11), (1, 3)):
        for step in range(steps):
            for dimension in range(7):
                active = {0: .5 + step / 10, 1: 1.3} if dimension % 2 == 0 and step % 3 else {}
                if step == 1:
                    active[2] = .2
                records.append({"task_id": task, "episode_uid": f"episode-{task}",
                                "step_in_episode": step, "action_dim_index": dimension})
                ids.append(torch.tensor(sorted(active), dtype=torch.int64))
                values.append(torch.tensor([active[fid] for fid in sorted(active)], dtype=torch.float32))
    # Deliberately reverse rows: canonical query traversal must still match.
    return {"records": list(reversed(records)),
            "hidden": torch.tensor([[2., 1.]] * len(records), dtype=dtype),
            "feature_ids": list(reversed(ids)), "feature_values": list(reversed(values)),
            "dict_size": 4, "synthetic": True, "encoder_grouping": "original_forward",
            "control_pool_feature_ids": [0, 1, 2, 3]}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tail_fraction", [.1, .3, 1.])
def test_head_only_scores_exactly_equal_original_control_scorer(dtype, tail_fraction):
    from event_sae.research.head_rank.scoring import score_cache

    cache = _scoring_cache(dtype)
    old_head, new_head = _ScoringHead(dtype), _ScoringHead(dtype)
    old = score_cache(cache, old_head, _ScoringSAE(), feature_ids=[0, 1, 2, 3],
                      control_count=2, tail_fraction=tail_fraction)
    new = score_head_only_cache(cache, new_head, _ScoringSAE(), feature_ids=[0, 1, 2, 3],
                               tail_fraction=tail_fraction)
    for reference, actual in zip(old["feature_scores"], new["feature_scores"]):
        for field in ("feature_id", "mean_head_kl", "tail_head_kl", "task_scores",
                      "num_active_readouts", "num_active_queries", "num_readouts", "num_queries"):
            assert actual[field] == reference[field]  # Exact, not approximate.
    assert new["diagnostics"]["baseline_head_calls"] == old["diagnostics"]["baseline_head_calls"]
    assert new["diagnostics"]["executed_pairs"] * 3 == old["diagnostics"]["executed_pairs"]
    assert new_head.calls == new["diagnostics"]["active_pairs"] + new["diagnostics"]["baseline_head_calls"]
    assert new["direction_control_calls"] == 0


def test_head_only_budget_is_checked_before_any_head_or_decoder_call():
    head, sae = _ScoringHead(torch.float32), _ScoringSAE()
    with pytest.raises(ValueError, match="pair budget"):
        score_head_only_cache(_scoring_cache(torch.float32), head, sae, max_pairs=1)
    assert head.calls == sae.calls == 0


def test_head_only_real_cache_still_requires_runtime_parity():
    cache = _scoring_cache(torch.float32)
    cache["synthetic"] = False
    with pytest.raises(ValueError, match="runtime parity"):
        score_head_only_cache(cache, _ScoringHead(torch.float32), _ScoringSAE())
