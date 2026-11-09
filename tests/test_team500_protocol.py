"""CPU protocol fixtures only; these are not GPU/LIBERO evidence."""
import copy

import pytest

from event_sae.research.output_head.provenance import fingerprint
from event_sae.research.team500.protocol import (
    BASELINE_COMMIT, CODE_REVISION, DICT_SIZE, MODEL_REVISION, SAE_SHA,
    SCORE_SCHEMA, UNIVERSE, build_plan, ordered_cases, validate_runtime_config,
    validate_scores, verify_plan, verify_plan_scores,
)


def cases_fixture():
    return [{"suite": "libero_spatial", "task_id": task, "task_episode_idx": trial,
             "initial_state_sha256": fingerprint([task, trial])}
            for task in range(10) for trial in range(50)]


def runtime_fixture():
    return {"schema_version": "team500_runtime_v1", "baseline_config": {
        "model": {"family": "openvla", "checkpoint": "/tmp/frozen-model",
                  "revision": MODEL_REVISION, "code_revision": CODE_REVISION,
                  "center_crop": True, "load_in_8bit": False, "load_in_4bit": False},
        "env": {"task_suite_name": "libero_spatial", "seed": 0,
                "num_steps_wait": 10, "num_trials_per_task": 50, "task_ids": "all"},
        "sae_collect": {"enabled": False},
        "logging": {"save_actions": True, "save_prompt_records": True, "save_video": False}},
        "sae_checkpoint": "/tmp/sae/ae.pt", "expected_attention_backend": "sdpa",
        "expected_unnorm_key": "libero_spatial", "model_files": {"config.json": "a" * 64}}


def scores_fixture():
    rows = [{"feature_id": fid, "mean_head_kl": 0., "tail_head_kl": 0., "event_max": 0.}
            for fid in range(DICT_SIZE)]
    # Mean/Tail same set, different order; Event shares two, adds three.
    for rank, fid in enumerate([8, 7, 6, 5, 4]):
        rows[fid]["mean_head_kl"] = float(10 - rank)
    for rank, fid in enumerate([8, 6, 7, 5, 4]):
        rows[fid]["tail_head_kl"] = float(10 - rank)
    for rank, fid in enumerate([6, 8, 2, 10, 12]):
        rows[fid]["event_max"] = float(10 - rank)
    score = {"schema_version": SCORE_SCHEMA, "status": "scored", "sampling_mode": "full500",
             "development_only": False, "synthetic": False, "feature_scores": rows,
             "scope": {"task_ids": list(range(10)), "discovery_episodes": 500, "episodes_per_task": 50,
                       "dictionary_size": DICT_SIZE, "sae_sha256": SAE_SHA, "candidate_universe": UNIVERSE,
                       "discovery_evaluation_state_overlap": True, "sae_training_evaluation_overlap": True,
                       "historical_state_identity_verified": False, "historical_runtime_flags_verified": False,
                       "discovery_manifest_hash": "b" * 64, "sample_manifest_hash": "c" * 64,
                       "source_index_sha256": "d" * 64, "discovery_cases_hash": fingerprint(cases_fixture())}}
    score["score_hash"] = fingerprint(score)
    return score


def rehash(payload, key):
    payload[key] = fingerprint({k: v for k, v in payload.items() if k != key})
    return payload


def test_exact_global_top5_union_not_15_separate_or_16_audit_features():
    score = scores_fixture()
    plan = build_plan(score, cases_fixture(), runtime_fixture(), score_file_sha256="e" * 64)
    assert plan["methods"]["mean_head_kl"] == [8, 7, 6, 5, 4]
    assert plan["methods"]["tail_head_kl"] == [8, 6, 7, 5, 4]
    assert plan["feature_ids"] == [2, 4, 5, 6, 7, 8, 10, 12]
    assert len(plan["conditions"]) == 10 and plan["total_rollouts"] == 5000
    assert plan["conditions"][0]["condition_id"] == "raw"
    assert plan["conditions"][1] == {"condition_id": "identity-8", "mode": "identity", "feature_id": 8, "alpha": 1.}
    assert plan["baseline_commit"] == BASELINE_COMMIT
    assert plan["selection_evaluation_overlap"] is True
    assert plan["new_rollouts"] == 0 and plan["execute"] is False
    verify_plan(plan)
    verify_plan_scores(plan, score)


def test_tied_scores_break_on_feature_id_and_never_exclude_known_ids():
    score = scores_fixture()
    for row in score["feature_scores"]:
        for key in ("mean_head_kl", "tail_head_kl", "event_max"):
            row[key] = 0.
    rehash(score, "score_hash")
    plan = build_plan(score, cases_fixture(), runtime_fixture())
    assert all(ids == [0, 1, 2, 3, 4] for ids in plan["methods"].values())
    assert plan["known_feature_exclusion"] is False
    assert plan["total_rollouts"] == 3500


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(sampling_mode="smoke"),
    lambda s: s.update(development_only=True),
    lambda s: s["scope"].update(discovery_episodes=250),
    lambda s: s["scope"].update(discovery_evaluation_state_overlap=False),
    lambda s: s["scope"].update(historical_runtime_flags_verified=True),
    lambda s: s["feature_scores"].pop(),
    lambda s: s["feature_scores"][0].update(mean_head_kl=float("inf")),
])
def test_smoke_250_incomplete_or_false_provenance_rejected(mutation):
    score = scores_fixture()
    mutation(score)
    if not any(isinstance(r.get("mean_head_kl"), float) and r["mean_head_kl"] == float("inf")
               for r in score["feature_scores"]):
        rehash(score, "score_hash")
    with pytest.raises(ValueError):
        validate_scores(score)


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop(),
    lambda c: c.reverse(),
    lambda c: c[1].update(task_episode_idx=0),
    lambda c: c[0].update(seed=3905563605),
    lambda c: c[0].update(initial_state_sha256=None),
])
def test_all500_order_and_current_hashes_are_mandatory(mutation):
    cases = cases_fixture()
    mutation(cases)
    with pytest.raises(ValueError):
        ordered_cases(cases)


@pytest.mark.parametrize("mutation", [
    lambda r: r["baseline_config"]["env"].update(seed=7),
    lambda r: r["baseline_config"]["env"].update(per_episode_seed=True),
    lambda r: r["baseline_config"]["env"].update(per_episode_seed=False),
    lambda r: r["baseline_config"]["env"].update(trial_indices=None),
    lambda r: r["baseline_config"]["env"].update(num_trials_per_task=20),
    lambda r: r["baseline_config"]["env"].update(task_ids=[1, 0]),
    lambda r: r["baseline_config"]["model"].update(load_in_4bit=True),
    lambda r: r.update(expected_attention_backend="auto"),
    lambda r: r.update(model_files={}),
])
def test_runtime_contract_cannot_silently_drift(mutation):
    runtime = runtime_fixture()
    mutation(runtime)
    with pytest.raises(ValueError):
        validate_runtime_config(runtime)


def test_tampering_rejected_and_rehashed_changed_ranking_requires_source_check():
    score = scores_fixture()
    plan = build_plan(score, cases_fixture(), runtime_fixture())
    broken = copy.deepcopy(plan)
    broken["total_rollouts"] += 500
    with pytest.raises(ValueError):
        verify_plan(broken)
    # Even a self-consistent alternate order must not replace original Top-5 predictions.
    changed = copy.deepcopy(plan)
    changed["methods"]["tail_head_kl"][1:3] = list(reversed(changed["methods"]["tail_head_kl"][1:3]))
    rehash(changed, "plan_hash")
    verify_plan(changed)
    with pytest.raises(ValueError, match="rankings"):
        verify_plan_scores(changed, score)


def test_eval_state_hash_must_match_full500_discovery_not_only_counts():
    score = scores_fixture()
    cases = cases_fixture()
    cases[0]["initial_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="discovery current-state"):
        build_plan(score, cases, runtime_fixture())
