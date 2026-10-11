"""Synthetic CPU descriptor checks, not scientific rollout/parity evidence."""
from copy import deepcopy

import pytest
import torch

from event_sae.research.head_predict.descriptors import action_bin_mapping, score_descriptor_cache
from event_sae.research.head_predict.descriptor_workflow import (
    descriptor_implementation_identity, descriptor_workflow, validate_descriptor_payload,
)
from event_sae.research.head_followup.stage1 import CODE_REV, MODEL_REV, SOURCE_HASHES
from event_sae.research.output_head.provenance import atomic_write_json, fingerprint, read_json, sha256_file
from event_sae.research.output_head.sensitivity import full_vocab_kl


class ToyHead:
    def __init__(self):
        self.head_weight = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]])
        self.hidden_dtype = torch.float32
        self.manifest = {}
        self.calls = 0

    def __call__(self, hidden):
        self.calls += 1
        return hidden @ self.head_weight.T


class ToySAE:
    def __init__(self):
        self.directions = torch.tensor([[1., 0.], [-1., 0.], [0., 0.], [0., 1.]])
        self.calls = 0

    def encode(self, hidden):
        raise AssertionError("Original-forward sparse latents must never be re-encoded")

    def decode(self, encoded):
        self.calls += 1
        return encoded @ self.directions + torch.tensor([.25, -.25])


def mapping_fixture():
    # Canonical IDs are deliberately not bin-value order: lower ID is +1.
    return {"synthetic": True, "action_token_ids": [1, 0],
            "token_id_to_bin_center": {"0": 1., "1": -1.}}


def cache_fixture(specs=None):
    specs = specs or [(0, "a", 0, [{0: 1.}] * 7), (0, "a", 1, [{}] * 7)]
    records, features = [], []
    for task, episode, step, positions in specs:
        for dim, values in enumerate(positions):
            records.append({"task_id": task, "episode_uid": episode, "step_in_episode": step,
                            "action_dim_index": dim})
            features.append(values)
    return {"hidden": torch.tensor([[.5, -1.]] * len(records)), "records": records,
            "feature_ids": [torch.tensor(sorted(row), dtype=torch.int64) for row in features],
            "feature_values": [torch.tensor([row[fid] for fid in sorted(row)], dtype=torch.float32)
                               for row in features],
            "dict_size": 4, "encoder_grouping": "original_forward", "synthetic": True}


def score(cache=None, **kwargs):
    return score_descriptor_cache(cache or cache_fixture(), ToyHead(), ToySAE(), mapping_fixture(), **kwargs)


def rows(result):
    return {(row["feature_id"], row["task_id"]): row for row in result["rows"]}


def test_known_signed_response_scalar_and_inactive_zero_denominators():
    head, sae = ToyHead(), ToySAE()
    result = score_descriptor_cache(cache_fixture(), head, sae, mapping_fixture(), feature_ids=[0, 3])
    row = rows(result)[0, 0]
    baseline, edited = torch.tensor([[.5, -.5, -1.]]), torch.tensor([[-.5, .5, -1.]])
    expected_shift = -2 * float(torch.tanh(torch.tensor(.5)))
    assert row["mean_head_kl"] == pytest.approx(float(full_vocab_kl(baseline, edited)[0]) / 2)
    assert row["signed_action_bin_shift"] == pytest.approx([expected_shift / 2] * 7)
    assert row["absolute_action_bin_shift"] == pytest.approx([-expected_shift / 2] * 7)
    assert row["action_argmax_flip"] == [.5] * 7
    assert row["action_conditional_kl"] == pytest.approx([float(full_vocab_kl(baseline[:, :2], edited[:, :2])[0]) / 2] * 7)
    assert row["mean_edit_norm"] == row["readout_activation_frequency"] == row["full_vocab_argmax_flip_rate"] == .5
    inactive = rows(result)[3, 0]
    assert inactive["signed_action_bin_shift"] == [0.] * 7
    assert inactive["absolute_action_bin_shift"] == inactive["action_conditional_kl"] == [0.] * 7
    assert inactive["mean_head_kl"] == inactive["mean_edit_norm"] == inactive["readout_activation_frequency"] == 0
    assert result["diagnostics"] == {"planned_pairs": 7, "executed_pairs": 7, "baseline_head_calls": 7, "num_readouts": 14}
    assert head.calls == sae.calls == 14
    assert result["synthetic"] is True and result["new_rollouts"] == 0
    assert result["model_regeneration"] is False and result["outcome_labels_used"] is False
    assert "not_physical_harm" in result["response_scope"]


def test_per_task_position_grouping_and_mean_absolute_not_absolute_mean():
    # Opposite cached states around zero produce exactly opposite shifts.
    cache = cache_fixture([(0, "a", 0, [{0: 1.}] + [{}] * 6),
                           (0, "a", 1, [{0: 1.}] + [{}] * 6),
                           (1, "b", 0, [{}] * 7)])
    # A nonlinear synthetic decoder returns a sign-changing edit when another
    # cached latent changes; no re-encoding is allowed for either query.
    class ContextSAE(ToySAE):
        def decode(self, encoded):
            self.calls += 1
            return torch.stack((encoded[:, 0] * (1 - 2 * encoded[:, 3]), encoded[:, 3]), dim=-1)
    cache["feature_ids"][7] = torch.tensor([0, 3])
    cache["feature_values"][7] = torch.tensor([1., 1.])
    cache["hidden"][7, 0] = -.5
    result = score_descriptor_cache(cache, ToyHead(), ContextSAE(), mapping_fixture(), feature_ids=[0])
    row = rows(result)[0, 0]
    assert row["signed_action_bin_shift"] == pytest.approx([0.] * 7, abs=1e-7)
    assert row["absolute_action_bin_shift"][0] > .9
    assert row["absolute_action_bin_shift"][1:] == [0.] * 6
    assert row["readout_activation_frequency"] == pytest.approx(1 / 7)
    assert rows(result)[0, 1]["mean_head_kl"] == 0


def test_noop_active_decoder_and_all_inactive_are_zero():
    active = score(cache_fixture([(0, "a", 0, [{2: 1.}] * 7)]), feature_ids=[2])
    row = active["rows"][0]
    assert row["readout_activation_frequency"] == 1
    assert row["mean_head_kl"] == row["mean_edit_norm"] == 0
    assert row["signed_action_bin_shift"] == row["action_argmax_flip"] == [0.] * 7
    head, sae = ToyHead(), ToySAE()
    result = score_descriptor_cache(cache_fixture(), head, sae, mapping_fixture(), feature_ids={3})
    assert head.calls == sae.calls == result["diagnostics"]["executed_pairs"] == 0


def test_order_invariance_for_records_sparse_ids_mapping_ids_and_selected_features():
    cache = cache_fixture([(0, "a", 0, [{0: 1., 1: .25}] * 7), (1, "b", 0, [{}] * 7)])
    initial = score(cache, feature_ids=[0, 1])
    permutation = list(reversed(range(len(cache["records"]))))
    cache["hidden"] = cache["hidden"][permutation]
    for name in ("records", "feature_ids", "feature_values"):
        cache[name] = [cache[name][index] for index in permutation]
    for name in ("feature_ids", "feature_values"):
        cache[name] = [value.flip(0) for value in cache[name]]
    mapping = mapping_fixture()
    mapping["action_token_ids"].reverse()
    permuted = score_descriptor_cache(cache, ToyHead(), ToySAE(), mapping, feature_ids=[1, 0])
    assert permuted["rows"] == initial["rows"]
    assert permuted["query_metadata"] == initial["query_metadata"]
    assert permuted["action_bin_decode"] == initial["action_bin_decode"]


@pytest.mark.parametrize("mutation", ["budget", "max_bool", "grouping", "truncated", "missing_dimension", "duplicate_dimension",
                                      "unequal_episodes", "hidden_nan", "latent_nan", "duplicate_features", "bad_mapping"])
def test_preflight_before_any_head_or_decode(mutation):
    head, sae, cache, mapping = ToyHead(), ToySAE(), cache_fixture(), mapping_fixture()
    kwargs = {"feature_ids": [0]}
    if mutation == "budget": kwargs["max_pairs"] = 6
    elif mutation == "max_bool": kwargs["max_pairs"] = True
    elif mutation == "grouping": cache["encoder_grouping"] = "readout_only_validated"
    elif mutation == "truncated": cache["truncated"] = True
    elif mutation == "missing_dimension": cache["records"][0]["step_in_episode"] = 2
    elif mutation == "duplicate_dimension": cache["records"][0]["action_dim_index"] = 1
    elif mutation == "unequal_episodes":
        cache = cache_fixture([(0, "a", 0, [{0: 1.}] * 7), (0, "a", 1, [{}] * 7), (0, "b", 0, [{}] * 7)])
    elif mutation == "hidden_nan": cache["hidden"][0, 0] = float("nan")
    elif mutation == "latent_nan": cache["feature_values"][0][0] = float("nan")
    elif mutation == "duplicate_features": kwargs["feature_ids"] = [0, 0]
    elif mutation == "bad_mapping": mapping["token_id_to_bin_center"]["0"] = float("nan")
    with pytest.raises(ValueError):
        score_descriptor_cache(cache, head, sae, mapping, **kwargs)
    assert head.calls == sae.calls == 0


def test_real_cache_cannot_borrow_synthetic_mapping_or_archived_passed_flag():
    cache, head, sae = cache_fixture(), ToyHead(), ToySAE()
    cache["synthetic"] = False
    with pytest.raises(ValueError, match="current head/edit"):
        score_descriptor_cache(cache, head, sae, mapping_fixture(), feature_ids=[0],
                               parity_reports={"head": {"passed": True}, "edit": {"passed": True}})
    assert head.calls == sae.calls == 0
    with pytest.raises(ValueError, match="unverified/stale"):
        action_bin_mapping(mapping_fixture(), {}, 3, synthetic=False)


def test_stale_runtime_identity_rejected_before_head_or_decoder():
    cache, head, sae = cache_fixture(), ToyHead(), ToySAE()
    cache["synthetic"] = False
    head.manifest = {"weights_sha256": "current-head", "model_revision": "current-model"}
    with pytest.raises(ValueError, match="parity identity differs"):
        score_descriptor_cache(cache, head, sae, mapping_fixture(), feature_ids=[0],
                               current_identity={"head_weights_hash": "old-head"},
                               parity_reports={"head": {"passed": True}, "edit": {"passed": True}})
    assert head.calls == sae.calls == 0


def test_nonfinite_head_and_decoder_responses_are_rejected():
    head, sae = ToyHead(), ToySAE()
    head.head_weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        score_descriptor_cache(cache_fixture(), head, sae, mapping_fixture(), feature_ids=[0])
    assert sae.calls == 0
    head, sae = ToyHead(), ToySAE()
    sae.directions[0, 0] = float("inf")
    with pytest.raises(ValueError, match="decoder"):
        score_descriptor_cache(cache_fixture(), head, sae, mapping_fixture(), feature_ids=[0])


def native_mapping_fixture():
    head = {"weights_sha256": "h" * 64, "model_revision": MODEL_REV, "code_revision": CODE_REV, "vocab_size": 32064}
    mapping = {"schema_version": "head_action_mapping_v1", "verified": True, "model_revision": MODEL_REV,
               "code_revision": CODE_REV, "head_weights_sha256": head["weights_sha256"],
               "tokenizer_sha256": SOURCE_HASHES["tokenizer.json"], "evidence": SOURCE_HASHES,
               "action_token_ids": list(range(31744, 32000)), "head_vocab_size": 32064,
               "base_vocab_size": 32000, "n_action_bins": 256}
    mapping["mapping_hash"] = fingerprint(mapping)
    return mapping, head


def test_native_decode_uses_255_centers_and_preserves_clipped_endpoint_alias():
    mapping, head = native_mapping_fixture()
    ids, centers, metadata = action_bin_mapping(mapping, head, 32064)
    by_id = dict(zip(ids, centers))
    assert by_id[31744] == by_id[31745] == pytest.approx(254 / 255)
    assert by_id[31999] == pytest.approx(-254 / 255)
    assert by_id[31872] == pytest.approx(0.)
    assert metadata["num_action_tokens"] == 256 and metadata["num_distinct_bin_centers"] == 255


@pytest.mark.parametrize("key,value", [("base_vocab_size", 31999), ("n_action_bins", 257),
                                       ("head_vocab_size", 32065), ("verified", False),
                                       ("tokenizer_sha256", "changed"), ("synthetic", True)])
def test_rehashed_mapping_cannot_change_native_decode(key, value):
    mapping, head = native_mapping_fixture()
    mapping[key] = value
    mapping["mapping_hash"] = fingerprint({key: value for key, value in mapping.items() if key != "mapping_hash"})
    with pytest.raises(ValueError):
        action_bin_mapping(mapping, head, 32064)


def test_progress_reports_final_inactive_rows():
    updates = []
    score(feature_ids=[0], progress=updates.append)
    assert updates[-1] == {"readouts": 14, "total_readouts": 14, "executed_pairs": 7, "total_pairs": 7}


def workflow_fixture(tmp_path, monkeypatch):
    from event_sae.research.output_head import config as config_module, workflow, head, readouts
    from event_sae.openvla import activations
    from event_sae.research.head_predict import descriptors
    root, output, config = tmp_path / "frozen", tmp_path / "result.json", tmp_path / "config.yaml"
    config.write_text("mock local config")
    weights = tmp_path / "sae" / "ae.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"frozen synthetic checkpoint")
    runtime = tmp_path / "current-runtime"
    cfg = {"output": {"root_dir": str(runtime)}, "inputs": {"local_model_snapshot": None},
           "scoring": {"alpha": 0., "pair_batch_size": 1, "reduction_dtype": "float32",
                       "negative_tolerance": 1e-6, "device": "cpu", "edit_backend": "reference"}}
    cache = cache_fixture()
    cache["synthetic"] = False
    cache["dict_size"] = 32768
    cache["manifest"] = {"sae_id": fingerprint({"weights": sha256_file(weights), "encoder_implementation": "toy"}),
                         "sample_hash": "s", "synthetic": False}
    atomic_write_json(root / "sample_manifest.json", {"sample_hash": "s", "readouts": cache["records"], "synthetic": False})
    atomic_write_json(root / "readouts/manifest.json", cache["manifest"])
    identity = {"sae_checkpoint_hash": sha256_file(weights), "head_weights_hash": "h",
                "model_revision": MODEL_REV, "code_revision": CODE_REV, "layer_idx": 31,
                "capture_target": "post_mlp_residual", "alpha": 0.}
    old = {"scope": {"sample_manifest_hash": "s", "sae_sha256": sha256_file(weights), "dictionary_size": 32768,
                     "task_ids": [0]},
           "identity": {**identity, "cache_manifest_hash": sha256_file(root / "readouts/manifest.json")}, "synthetic": False}
    old["identity_hash"] = fingerprint(old["identity"])
    atomic_write_json(root / "scores/scores.json", old)
    mapping = tmp_path / "mapping.json"
    atomic_write_json(mapping, mapping_fixture())
    monkeypatch.setattr(config_module, "load_config", lambda _: cfg)
    monkeypatch.setattr(workflow, "_sae_paths", lambda _: (weights, weights.parent / "config.json"))
    monkeypatch.setattr(workflow, "_head_path", lambda _: root / "head")
    monkeypatch.setattr(workflow, "numerical_identity", lambda _: identity)
    seen = {}
    def parity(config, current):
        seen["parity_root"] = config["output"]["root_dir"]
        return {"head": {"fixture": True}, "edit": {"fixture": True}}
    monkeypatch.setattr(workflow, "load_parity", parity)
    monkeypatch.setattr(workflow, "sae_implementation_identity", lambda _: "toy")
    monkeypatch.setattr(readouts, "load_readout_cache", lambda _: cache)
    monkeypatch.setattr(activations, "load_batch_topk_sae", lambda *a, **kw: (ToySAE(), None))
    monkeypatch.setattr(head, "load_output_head", lambda *a, **kw: ToyHead())
    def scorer(*args, **kwargs):
        seen["feature_ids"] = kwargs["feature_ids"]
        return {"schema_version": "head_response_descriptors_v1", "synthetic": False,
                "feature_ids": sorted(kwargs["feature_ids"]), "task_ids": [0], "rows": [],
                "diagnostics": {"fixture_only": True}, "new_rollouts": 0}
    monkeypatch.setattr(descriptors, "score_descriptor_cache", scorer)
    return config, root, mapping, output, seen, cfg, old


def test_workflow_default_development_current_parity_and_integrity(tmp_path, monkeypatch):
    from event_sae.research.head_rank.workflow import DEVELOPMENT_FEATURES
    config, root, mapping, output, seen, cfg, old = workflow_fixture(tmp_path, monkeypatch)
    summary = descriptor_workflow(config, root, mapping, output)
    saved = read_json(output)
    assert seen["feature_ids"] == list(DEVELOPMENT_FEATURES)
    assert seen["parity_root"] == cfg["output"]["root_dir"]
    assert saved["development_only"] is True and saved["full_readouts"] is True
    assert saved["source_scores_sha256"] == saved["reference_scores_sha256"] == sha256_file(root / "scores/scores.json")
    assert "source_score_hash" not in saved
    assert saved["scope"] == old["scope"]
    assert saved["descriptor_hash"] == fingerprint({k: v for k, v in saved.items() if k != "descriptor_hash"})
    assert summary["output_sha256"] == sha256_file(output)
    assert saved["implementation"] == descriptor_implementation_identity()
    with pytest.raises(FileExistsError):
        descriptor_workflow(config, root, mapping, output)


def test_workflow_all_active_is_explicit_and_source_tree_is_readonly(tmp_path, monkeypatch):
    config, root, mapping, output, seen, cfg, _ = workflow_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="overlaps"):
        descriptor_workflow(config, root, mapping, root / "new.json")
    result = descriptor_workflow(config, root, mapping, output, feature_ids="all-active")
    assert seen["feature_ids"] == [0]
    assert result["feature_set"] == "all-active" and result["development_only"] is False


def test_workflow_stale_binding_fails_without_output(tmp_path, monkeypatch):
    config, root, mapping, output, _, _, old = workflow_fixture(tmp_path, monkeypatch)
    old["identity"]["head_weights_hash"] = "old-head"
    old["identity_hash"] = fingerprint(old["identity"])
    atomic_write_json(root / "scores/scores.json", old, overwrite=True)
    with pytest.raises(ValueError, match="binding mismatch"):
        descriptor_workflow(config, root, mapping, output)
    assert not output.exists()


def semantic_payload_fixture():
    """Construct test metadata, not genuine policy/parity evidence."""
    payload = score(feature_ids=[0, 3])
    mapping, head = native_mapping_fixture()
    _, _, decode = action_bin_mapping(mapping, head, 32064)
    scope = {"dictionary_size": 4, "task_ids": [0], "sae_sha256": "s" * 64,
             "sample_manifest_hash": "m" * 64, "split_manifest_hash": "p" * 64,
             "discovery_manifest_hash": "d" * 64}
    identity = {"sae_checkpoint_hash": scope["sae_sha256"], "head_weights_hash": head["weights_sha256"],
                "model_revision": MODEL_REV, "code_revision": CODE_REV, "layer_idx": 31,
                "encoder_grouping": "original_forward", "pair_batch_size": 1,
                "reduction_dtype": "float32", "negative_tolerance": 1e-6, "alpha": 0.}
    bindings = {"scope": scope, "sample_manifest_sha256": "a" * 64,
                "reference_scores_sha256": "r" * 64, "readout_record_hash": "q" * 64}
    payload.update(synthetic=False, full_readouts=True, sampling_mode="full", scope=scope,
                   feature_set="custom", development_only=True, scored_feature_ids=[0, 3],
                   frozen_active_feature_ids=[0, 3], identity=identity,
                   source_sample_sha256=bindings["sample_manifest_sha256"],
                   reference_scores_sha256=bindings["reference_scores_sha256"],
                   readout_record_hash=bindings["readout_record_hash"], source_cache_manifest_sha256="c" * 64,
                   scoring_sample={"readouts_used": 14, "frozen_readouts": 14, "discovery_episodes_used": 1},
                   action_mapping=mapping, action_bin_decode=decode, action_mapping_sha256="b" * 64)
    source = {"scope": scope, "identity": deepcopy(identity), "feature_ids": [0, 3],
              "source_cache_manifest_sha256": "c" * 64, "query_metadata": deepcopy(payload["query_metadata"]),
              "scoring_sample": {"readouts_used": 14}}
    reference = {"scope": scope, "identity": {"cache_manifest_hash": "c" * 64}}
    payload["descriptor_hash"] = fingerprint(payload)
    return payload, source, reference, bindings


def test_full_descriptor_semantic_validator_accepts_explicit_development_subset():
    payload, source, reference, bindings = semantic_payload_fixture()
    validate_descriptor_payload(payload, source_scores=source, reference_scores=reference, bindings=bindings)


@pytest.mark.parametrize("mutation", ["smoke", "synthetic", "false_allactive", "wrong_development16", "header_ids",
    "task_ids", "dimension_order", "query_subset", "duplicate_query", "unbalanced", "row_subset", "row_count",
    "active_frequency", "cache_hash", "sae", "head", "decoder", "outcomes", "mapping_bins", "bad_abs", "bad_flip"])
def test_rehashed_descriptor_cannot_masquerade_as_full_bound_response(mutation):
    payload, source, reference, bindings = semantic_payload_fixture()
    if mutation == "smoke": payload["full_readouts"] = False
    elif mutation == "synthetic": payload["synthetic"] = True
    elif mutation == "false_allactive": payload["feature_set"] = "all-active"
    elif mutation == "wrong_development16": payload["feature_set"] = "development16"
    elif mutation == "header_ids": payload["feature_ids"] = [0]
    elif mutation == "task_ids": payload["task_ids"] = [1]
    elif mutation == "dimension_order": payload["action_dimension_order"].reverse()
    elif mutation == "query_subset": payload["query_metadata"].pop()
    elif mutation == "duplicate_query":
        payload["query_metadata"][1]["step_in_episode"] = 0
        source["query_metadata"] = deepcopy(payload["query_metadata"])
    elif mutation == "unbalanced":
        extra = {**payload["query_metadata"][0], "query_index": 2, "episode_uid": "b"}
        payload["query_metadata"].append(extra)
        source["query_metadata"] = deepcopy(payload["query_metadata"])
    elif mutation == "row_subset": payload["rows"].pop()
    elif mutation == "row_count": payload["rows"][0]["num_readouts"] = 7
    elif mutation == "active_frequency": payload["rows"][0]["readout_activation_frequency"] = .4
    elif mutation == "cache_hash": payload["source_cache_manifest_sha256"] = "changed"
    elif mutation == "sae": payload["identity"]["sae_checkpoint_hash"] = "changed"
    elif mutation == "head": payload["identity"]["head_weights_hash"] = "changed"
    elif mutation == "decoder": payload["action_bin_decode"]["token_id_to_bin_center"]["31744"] = 1.
    elif mutation == "outcomes": payload["outcome_labels_used"] = True
    elif mutation == "mapping_bins":
        payload["action_mapping"]["n_action_bins"] = 257
        payload["action_mapping"]["mapping_hash"] = fingerprint({k: v for k, v in payload["action_mapping"].items() if k != "mapping_hash"})
    elif mutation == "bad_abs": payload["rows"][0]["absolute_action_bin_shift"][0] = 0.
    elif mutation == "bad_flip": payload["rows"][0]["action_argmax_flip"][0] = 1.5
    payload["descriptor_hash"] = fingerprint({k: v for k, v in payload.items() if k != "descriptor_hash"})
    with pytest.raises(ValueError):
        validate_descriptor_payload(payload, source_scores=source, reference_scores=reference, bindings=bindings)
