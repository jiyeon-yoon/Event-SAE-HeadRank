"""Synthetic CPU checks, not evidence of OpenVLA/robot experiment success."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from event_sae.research.head_followup.action_scores import score_action_cache
from event_sae.research.head_followup.design import design_summary, select_decoder_controls
from event_sae.research.head_followup.interventions import (
    PhaseRule, apply_edit, decoder_direction, phase_gate, random_direction,
)
from event_sae.research.head_followup.metrics import GeneralityAccumulator, action_metrics, calibrate_generality
from event_sae.research.head_followup.hooks import register_followup_hook
from event_sae.research.head_followup.temporal import (
    load_temporal_cache, prepare_temporal_cache, temporal_statistics, validate_sequences,
)
from event_sae.research.output_head.head import OutputHeadBundle
from event_sae.research.output_head.readouts import build_readout_manifest
from event_sae.research.output_head.sensitivity import edit_feature_reference

ROOT = Path(__file__).resolve().parents[1]


class SAE:
    training = False
    directions = torch.tensor([[1., .5], [-.4, .7], [.1, .2]])

    def encode(self, hidden):
        return torch.cat([hidden.relu(), torch.zeros_like(hidden[:, :1])], -1)

    def decode(self, z):
        return z @ self.directions.to(z.device) + .125


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alpha", [0., .25, .5, .75, 1.])
def test_dose_matches_legacy_reference(dtype, alpha):
    h = torch.tensor([[1.13, -.2], [0., 1.1]], dtype=dtype)
    actual, info = apply_edit(h, SAE(), 0, alpha)
    assert torch.equal(actual, edit_feature_reference(h, SAE(), 0, alpha))
    assert info["realized_norm"][1] == 0
    if alpha == 1:
        assert torch.equal(actual, h)


@pytest.mark.parametrize("alpha", [-.1, 1.1, float("nan"), True])
def test_invalid_dose_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        apply_edit(torch.ones(1, 2), SAE(), 0, alpha)


def test_direction_match_is_per_row_not_average_and_keeps_anchor_inactivity():
    h = torch.tensor([[1., .4], [2., .3], [0., .2]])
    original, original_info = apply_edit(h, SAE(), 0, 0.)
    matched, info = apply_edit(h, SAE(), 0, 0., direction=torch.tensor([0., 3.]))
    torch.testing.assert_close(info["intended_norm"], original_info["intended_norm"])
    torch.testing.assert_close((matched - h).norm(dim=-1), (original - h).norm(dim=-1))
    assert torch.equal(matched[2], h[2])
    assert not torch.equal(matched[0], original[0])


def test_latent_grouping_not_reencoded_when_original_z_supplied():
    class BatchSAE(SAE):
        def encode(self, h):
            raise AssertionError("must not re-encode the readout alone")
    z = torch.tensor([[2., 0., 0.]])
    _, info = apply_edit(torch.ones(1, 2), BatchSAE(), 0, .5, encoded=z)
    assert info["intended_norm"][0] > 0


def test_seeded_direction_is_fixed_and_decoder_bias_is_removed():
    a = random_direction(32, 22)
    assert torch.equal(a, random_direction(32, 22))
    assert not torch.equal(a, random_direction(32, 23))
    assert float(a.norm()) == pytest.approx(1.)
    expected = SAE.directions[1] / SAE.directions[1].norm()
    torch.testing.assert_close(decoder_direction(SAE(), 1, 3), expected)


@pytest.mark.parametrize("direction", [[0., 0.], [1., 2., 3.], [float("nan"), 1.]])
def test_invalid_control_directions_fail(direction):
    with pytest.raises(ValueError, match="direction"):
        apply_edit(torch.ones(1, 2), SAE(), 0, 0., direction=direction)


def test_phases_use_pre_action_target_state_not_time_thirds():
    rule = PhaseRule(.04)
    obs = {"target_grasped": False, "target_placed": False, "eef_target_distance": .2}
    assert rule.classify(obs) == "approach"
    assert rule.classify({**obs, "eef_target_distance": .02}) == "grasp"
    assert rule.classify({**obs, "target_grasped": True}) == "transport"
    assert rule.classify({**obs, "target_placed": True}) == "other"
    assert phase_gate("grasp", "grasp")
    assert not phase_gate("grasp", "transport")
    with pytest.raises(ValueError, match="verified"):
        rule.classify({"step_in_episode": 50})
    with pytest.raises(ValueError, match="phase"):
        phase_gate("grasp", None)


def hook_fixture():
    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return (hidden, "untouched")
    layer = Layer()
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])),
                            _sae_hook_context={"task_id": 0, "task_episode_idx": 0, "step_in_episode": 0})
    return model, layer


@pytest.mark.parametrize("alpha", [0., .5, .75, 1.])
def test_native_hook_matches_reference_and_preserves_tuple_and_removes(tmp_path, alpha):
    model, layer = hook_fixture()
    hidden = torch.tensor([[[1., .2], [.3, -.2]]], dtype=torch.bfloat16)
    handle = register_followup_hook(model, SAE(), layer_idx=0, feature_id=0, alpha=alpha,
                                    evidence_path=tmp_path / "evidence.jsonl")
    actual, rest = layer(hidden)
    assert torch.equal(actual, edit_feature_reference(hidden, SAE(), 0, alpha))
    assert rest == "untouched"
    assert handle.summary()["num_forwards"] == 1
    handle.remove()
    handle.remove()
    assert torch.equal(layer(hidden)[0], hidden)


def test_phase_hook_uses_one_observation_per_query_and_aborts_unknown_context(tmp_path):
    from event_sae.openvla.intervene import SAEHookError
    model, layer = hook_fixture()
    seen = []
    def provider(context):
        seen.append(context["step_in_episode"])
        return "grasp" if context["step_in_episode"] == 0 else "transport"
    handle = register_followup_hook(model, SAE(), layer_idx=0, feature_id=0, alpha=0,
        evidence_path=tmp_path / "evidence.jsonl", target_phase="grasp", phase_provider=provider)
    hidden = torch.ones(1, 1, 2)
    try:
        assert not torch.equal(layer(hidden)[0], hidden)
        layer(hidden)
        assert seen == [0]
        model._sae_hook_context["step_in_episode"] = 1
        assert torch.equal(layer(hidden)[0], hidden)
        assert seen == [0, 1]
        model._sae_hook_context = {}
        with pytest.raises(SAEHookError, match="context"):
            layer(hidden)
    finally:
        handle.remove()
    assert handle.summary()["enabled_forwards"] == 2


def test_phase_hook_does_not_guess_missing_adapter(tmp_path):
    model, _ = hook_fixture()
    with pytest.raises(ValueError, match="phase provider"):
        register_followup_hook(model, SAE(), layer_idx=0, feature_id=0, alpha=0, target_phase="grasp",
                               evidence_path=tmp_path / "absent.jsonl")
    assert not (tmp_path / "absent.jsonl").exists()


def test_action_kl_detects_conditioning_blind_spot_with_mass_metric():
    p = torch.tensor([[10., 2., 1.]])
    q = torch.tensor([[0., 2., 1.]])
    result = action_metrics(p, q, [1, 2])
    assert result["action_conditional_kl"].item() == 0
    assert result["full_vocab_kl"].item() > 1
    assert result["action_mass_abs_change"].item() > .5
    assert result["conditional_action_argmax_flip"].item() == 0


@pytest.mark.parametrize("ids", [[], [1, 1], [-1], [3], [True], [1.5]])
def test_action_mapping_validation(ids):
    with pytest.raises(ValueError, match="mapping"):
        action_metrics(torch.ones(1, 3), torch.ones(1, 3), ids)


def test_action_scores_use_all_readout_denominators_and_keep_inactive_features():
    hidden = torch.tensor([[1., -.2], [0., 1.]])
    sae = SAE()
    z = sae.encode(hidden)
    ids = [torch.nonzero(row, as_tuple=True)[0] for row in z]
    cache = {"hidden": hidden, "dict_size": 3, "synthetic": True, "encoder_grouping": "original_forward",
             "records": [{"task_id": i, "episode_uid": str(i), "step_in_episode": 0, "action_dim_index": 0} for i in range(2)],
             "feature_ids": ids, "feature_values": [z[i, ind] for i, ind in enumerate(ids)]}
    head = OutputHeadBundle(torch.ones(2), torch.tensor([[1., 0.], [0., 1.], [-1., .2]]),
        {"norm": {"implementation": "llama_rms_norm_v1", "eps": 1e-6}, "target_layer": 0, "num_layers": 1,
         "hidden_dtype": "float32"})
    result = score_action_cache(cache, head, sae, {"action_token_ids": [1, 2]}, max_pairs=2)
    expected = action_metrics(head(hidden[:1]), head(edit_feature_reference(hidden[:1], sae, 0, 0.)), [1, 2])
    assert result["feature_scores"][0]["action_conditional_kl"] == pytest.approx(expected["action_conditional_kl"].item() / 2)
    assert result["feature_scores"][2]["action_conditional_kl"] == 0
    assert result["feature_scores"][2]["mean_edited_action_mass"] == result["baseline_mean_action_mass"]
    with pytest.raises(ValueError, match="budget"):
        score_action_cache(cache, head, sae, {"action_token_ids": [1, 2]}, max_pairs=1)
    with pytest.raises(ValueError, match="verified"):
        score_action_cache({**cache, "synthetic": False}, head, sae, {"action_token_ids": [1, 2]}, max_pairs=2)


def test_temporal_hysteresis_active_episode_denominators_and_reset():
    stats = GeneralityAccumulator(3)
    stats.add_episode("a", list(range(5)), torch.tensor([[0., .05, 0.], [.2, .05, 0.], [.05, 0., 0.], [0., 0., 0.], [.3, 0., 0.]]))
    stats.add_episode("b", [0, 1], torch.tensor([[.2, 0., 0.], [0., 0., 0.]]))
    rows = stats.result()["feature_statistics"]
    assert rows[0]["episode_coverage"] == 1
    assert rows[0]["mean_onset_count"] == 1.5
    assert rows[0]["mean_episode_max_activation"] == pytest.approx(.25)
    assert rows[0]["relative_run_length"] == pytest.approx((3 / 2 / 5 + 1 / 1 / 2) / 2)
    assert rows[1]["episode_coverage"] == .5
    assert rows[1]["relative_run_length"] is None
    assert rows[1]["status"] == "undefined_run_length"
    assert rows[2]["status"] == "inactive"
    with pytest.raises(ValueError, match="unique"):
        stats.add_episode("a", [0], torch.zeros(1, 3))
    with pytest.raises(ValueError, match="consecutive"):
        stats.add_episode("c", [0, 3], torch.ones(2, 3))


def temporal_fixture(tmp_path):
    source = tmp_path / "dense"
    source.mkdir()
    rows, offset = [], 0
    for step in range(3):
        for dimension in range(7):
            n = 3 if dimension == 0 else 1
            rows.append({"layer_idx": 31, "shard_path": "dense.pt", "row_start": offset, "row_end": offset + n,
                         "episode_num": 1, "step_in_episode": step, "task_id": 0, "task_episode_idx": 0,
                         "global_forward_idx": len(rows)})
            offset += n
    dense = torch.ones(offset, 2)
    torch.save(dense, source / "dense.pt")
    generation = {"layer_idx": 31, "action_dim": 7, "batch_size": 1, "padding": "none", "use_cache": True,
                  "source_run_id": "synthetic", "suite": "toy", "evidence": "synthetic"}
    manifest = build_readout_manifest(rows, {"synthetic": True}, generation, dense_dir=source)
    return manifest


class ManyActiveSAE:
    training = False

    def encode(self, hidden):
        # Batch dependent with >64 nonzeros; single-readout encoding is wrong.
        return torch.ones(len(hidden), 70, dtype=torch.float32) * len(hidden)


def test_full_temporal_cache_keeps_over64_and_original_forward_grouping(tmp_path):
    manifest = temporal_fixture(tmp_path)
    output = tmp_path / "temporal"
    metadata = prepare_temporal_cache(manifest, ManyActiveSAE(), output, sae_sha256="a" * 64, hidden_dtype="float32", rows_per_shard=5)
    assert metadata["num_readouts"] == 21
    cache = load_temporal_cache(output)
    assert all(len(ids) == 70 for ids in cache["feature_ids"])
    assert cache["feature_values"][0][0] == 3
    assert cache["feature_values"][1][0] == 1
    stats = temporal_statistics(cache)
    assert stats["num_episodes"] == 1
    assert stats["feature_statistics"][0]["mean_episode_max_activation"] == pytest.approx(9 / 7)
    assert stats["feature_statistics"][0]["relative_run_length"] == 1.
    with pytest.raises(FileExistsError):
        prepare_temporal_cache(manifest, ManyActiveSAE(), output, sae_sha256="a" * 64)
    with pytest.raises(ValueError, match="overlaps"):
        prepare_temporal_cache(manifest, ManyActiveSAE(), Path(manifest["dense_dir"]) / "cache", sae_sha256="a" * 64)


def test_temporal_refuses_sampled_manifest_and_corrupted_shards(tmp_path):
    manifest = temporal_fixture(tmp_path)
    sampled = copy.deepcopy(manifest)
    sampled["sample_spec"]["max_steps_per_episode"] = 8
    with pytest.raises(ValueError, match="eight-step"):
        prepare_temporal_cache(sampled, ManyActiveSAE(), tmp_path / "bad", sae_sha256="a" * 64)
    output = tmp_path / "temporal"
    prepare_temporal_cache(manifest, ManyActiveSAE(), output, sae_sha256="a" * 64)
    shard = output / "latents_000000.pt"
    with shard.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_temporal_cache(output)


def test_temporal_requires_agreement_with_old_sample_not_just_same_checkpoint_name(tmp_path):
    manifest = temporal_fixture(tmp_path)
    row = manifest["readouts"][0]
    reference = {"hidden": torch.ones(1, 2), "dict_size": 70, "records": [row],
                 "feature_ids": [torch.arange(70)], "feature_values": [torch.full((70,), 3.)],
                 "encoder_grouping": "original_forward"}
    output = tmp_path / "good"
    metadata = prepare_temporal_cache(manifest, ManyActiveSAE(), output, sae_sha256="a" * 64,
                                      reference_cache=reference)
    assert metadata["frozen_head_reference_exact"] is True
    assert metadata["frozen_head_reference_rows_checked"] == 1
    reference["feature_values"][0][0] = 99
    with pytest.raises(ValueError, match="differs from frozen"):
        prepare_temporal_cache(manifest, ManyActiveSAE(), tmp_path / "bad", sae_sha256="a" * 64,
                               reference_cache=reference)


def test_sequence_missing_steps_or_dimensions_rejected(tmp_path):
    records = temporal_fixture(tmp_path)["readouts"]
    with pytest.raises(ValueError, match="dimensions"):
        validate_sequences(records[:-1])
    with pytest.raises(ValueError, match="nonconsecutive"):
        validate_sequences([r for r in records if r["step_in_episode"] != 1])


def test_generality_calibration_requires_independent_labels_and_refits_folds():
    rows = [{"feature_id": i, "status": "ok", "episode_coverage": (i + 1) / 32,
             "mean_onset_count": 1 + i / 10, "mean_episode_max_activation": 5 + i,
             "relative_run_length": 1 / (i + 1)} for i in range(32)]
    stats = {"feature_statistics": rows}
    labels = {"schema_version": "drvla_discovery_labels_v1", "used_outcomes": False, "used_head_rank": False,
              "split_role": "discovery", "evaluator": "synthetic test", "labels": [
        {"feature_id": i, "label": "general" if i >= 15 else "episode_specific", "evidence": "synthetic"} for i in range(30)]}
    result = calibrate_generality(stats, labels, excluded_feature_ids=[])
    assert len(result["feature_scores"]) == 32
    assert result["sae_retrained"] is False
    assert 0 <= result["leave_one_feature_out_accuracy"] <= 1
    with pytest.raises(ValueError, match="previously measured"):
        calibrate_generality(stats, labels, excluded_feature_ids=[1])
    with pytest.raises(ValueError, match="no-outcome"):
        calibrate_generality(stats, {**labels, "used_outcomes": True}, excluded_feature_ids=[])


def test_all_five_design_costs_are_explicit_not_approved():
    spec = json.loads((ROOT / "configs/research/openvla/head_followup_all5.json").read_text())
    result = design_summary(spec)
    assert result["status"] == "design_only_not_executable"
    assert result["execute"] is False
    assert result["known_conditions"] == 34
    assert result["rollouts_without_automatic_old_result_reuse"] == {"min": 6800, "max": 8000}
    assert len({c["id"] for c in result["conditions"]}) == 34
    assert 18471 in spec["anchor_features"]  # keep the old zero-drop counterexample
    complete = design_summary({**spec, "action_top3": [1, 2, 3], "drvla_top3": [4, 5, 6]})
    assert complete["rollouts_without_automatic_old_result_reuse"] == {"min": 8000, "max": 8000}
    assert len(complete["pending_rankings"]) == 0
    with pytest.raises(ValueError, match="execute"):
        design_summary({**spec, "execute": True})


def test_control_selection_not_based_on_outcomes_and_is_deterministic():
    a = select_decoder_controls(list(range(30)), [1, 2, 3], 44)
    assert a == select_decoder_controls(list(reversed(range(30))), [1, 2, 3], 44)
    assert not set(a.values()) & {1, 2, 3}


def test_cli_is_safe_without_execute_and_has_no_rollout_command(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/openvla/headfollowup.py")]
    process = subprocess.run(command + ["temporal", "--config", "not-present.yaml", "--output", str(tmp_path / "absent")],
                             capture_output=True, text=True)
    assert process.returncode == 0
    assert json.loads(process.stdout)["status"] == "not_executed"
    assert not (tmp_path / "absent").exists()
    process = subprocess.run(command + ["run", "--execute"], capture_output=True, text=True)
    assert process.returncode == 2


def test_headrank_limits_inherited_runtime_changes_to_explicit_adapters():
    # HeadRank is an independent clone, not the frozen Head checkout. Its only
    # allowed native extensions are caller-owned runtime/SAE reuse, per-case
    # seed records and cleanup. The original numerical modules stay frozen.
    # test_head_rank_native exercises both legacy and cached hook arithmetic.
    baseline = "81487a07509ee10433c41c19d4bbcd079b9a4968"
    result = subprocess.run(["git", "diff", "--name-only", baseline, "--", "event_sae/research/output_head",
                             "event_sae/openvla", "scripts/openvla/output_head_sensitivity.py"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) <= {
        "event_sae/openvla/eval/runner.py", "event_sae/openvla/intervene.py",
    }
