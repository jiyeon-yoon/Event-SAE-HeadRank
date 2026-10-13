from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from event_sae.research.output_head.provenance import fingerprint, atomic_write_json
from event_sae.research.head_matched import protocol as p


def seal(doc, field):
    doc.pop(field, None)
    doc[field] = fingerprint(doc)
    return doc


def fixture_scores():
    queries = [{"query_index": 2 * task + i, "task_id": task, "episode_uid": f"task-{task}",
                "step_in_episode": i, "num_action_positions": 7} for task in range(10) for i in range(2)]
    ids = [1, 2, 3, 4, 5]
    rows = [{"feature_id": fid, "mean_head_kl": .5, "direction_excess_status": "ok",
             "norm_invalid_readouts": 0, "norm_invalid_controls": 0,
             "control_feature_ids": [2, 3, 4, 5] if fid == 1 else [i for i in ids if i != fid],
             "num_readouts": 140, "num_queries": 20} for fid in ids]
    doc = {"schema_version": "head_rank_scores_v1", "sampling_mode": "full", "synthetic": False,
           "development_only": False, "feature_set": "all-active", "norm_match": p.NORM_POLICY,
           "feature_scores": rows, "feature_ids": ids, "frozen_active_feature_ids": ids, "scored_feature_ids": ids,
           "query_metadata": queries,
           "feature_query_scores": {"1": [{"query_index": 2 * t, "mean_head_kl": 1.,
                                          "control_query_kls": [2., 3., 4., 5.]} for t in range(10)]},
           "scoring_sample": {"readouts_used": 140, "discovery_episodes_used": 10,
                              "event_max_matches_scoring_episode_population": True},
           "scope": {"task_ids": list(range(10))}, "identity": {}, "implementation": {},
           "source_scores_sha256": "a" * 64, "source_sample_sha256": "b" * 64,
           "source_cache_manifest_sha256": "c" * 64}
    return seal(doc, "score_hash")


def fixture_parent(scores):
    from event_sae.research.head_rank.native import episode_seed
    keys = ["base_eval_config", "sae", "sae_config", "head_weights", "head_manifest", "generation_manifest"]
    keys += [f"snapshot/model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
    return {"prediction_plan": {"source_bindings": {"score_hash": scores["score_hash"]}, "panel_feature_ids": [1]},
            "cases": [{"suite": "libero_spatial", "task_id": t, "task_episode_idx": i,
                       "seed": episode_seed(7, t, i), "initial_state_sha256": fingerprint([t, i])}
                      for t in range(10) for i in range(20)],
            "execution_hash": "a" * 64, "base_seed": 7, "split_evidence": {},
            "sources": {k: {"path": f"/old/{k}", "sha256": "b" * 64} for k in keys}}


def plan_fixture(monkeypatch):
    import event_sae.research.head_rank.runtime as old_runtime
    monkeypatch.setattr(old_runtime, "verify_experiment", lambda _: None)
    scores = fixture_scores()
    return p.build_plan(scores, fixture_parent(scores), anchor=1)


def test_direction_means_keep_inactive_queries():
    values = p.direction_scores(fixture_scores(), 1)
    assert [r["mean_head_kl"] for r in values] == [.5, 1., 1.5, 2., 2.5]
    assert all(r["task_mean_head_kl"]["0"] == r["mean_head_kl"] for r in values)


@pytest.mark.parametrize("change", ["duplicate", "nonfinite", "ineligible", "bad_mean"])
def test_bad_scores_rejected(change):
    scores = fixture_scores()
    if change == "duplicate":
        scores["feature_query_scores"]["1"].append(scores["feature_query_scores"]["1"][0])
    elif change == "nonfinite":
        scores["feature_query_scores"]["1"][0]["control_query_kls"][0] = -1
    elif change == "ineligible":
        scores["feature_scores"][0]["norm_invalid_controls"] = 1
    else:
        scores["feature_scores"][0]["mean_head_kl"] = .2
    seal(scores, "score_hash")
    with pytest.raises(ValueError):
        p.direction_scores(scores, 1)


def test_plan_is_bounded_seeded_and_preserves_parent(monkeypatch):
    plan = plan_fixture(monkeypatch)
    assert plan["total_rollouts"] == 140
    assert len(plan["conditions"]) == 7
    assert plan["purpose"] == "smoke_diagnostic_no_power_claim"
    assert plan == plan_fixture(monkeypatch)
    assert plan["automatic_reuse_of_old_outcomes"] is False
    assert plan["scientific_scope"] == p.SCOPE
    p.verify_plan(plan)


@pytest.mark.parametrize("change", ["duplicate_case", "condition", "seed", "budget", "missing_shard", "scope"])
def test_resigned_plan_semantic_tampering_rejected(monkeypatch, change):
    plan = plan_fixture(monkeypatch)
    if change == "duplicate_case":
        plan["cases"][1] = deepcopy(plan["cases"][0])
    elif change == "condition":
        plan["conditions"][3]["control_feature_id"] = 999
    elif change == "seed":
        plan["cases"][0]["seed"] += 1
    elif change == "budget":
        plan["total_rollouts"] = 139
    elif change == "missing_shard":
        del plan["parent_content_hashes"]["snapshot/model-00001-of-00004.safetensors"]
    else:
        plan["norm_scope"] = "equal_trajectories"
    seal(plan, "plan_hash")
    with pytest.raises(ValueError):
        p.verify_plan(plan)


def test_cli_score_validate_dry_preview_never_load_inputs(capsys):
    path = Path(__file__).parents[1] / "scripts/openvla/headmatched.py"
    spec = importlib.util.spec_from_file_location("headmatched_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["score", "--config", "missing", "--plan", "missing", "--reference-result", "missing", "--output", "missing"]) == 0
    assert cli.main(["validate", "--config", "missing", "--plan", "missing", "--output", "missing"]) == 0
    assert capsys.readouterr().out.count("not_executed") == 2


def test_plan_output_never_overwrites(monkeypatch, tmp_path):
    import event_sae.research.head_rank.runtime as old_runtime
    monkeypatch.setattr(old_runtime, "verify_experiment", lambda _: None)
    scores = fixture_scores()
    src, parent, out = tmp_path / "scores.json", tmp_path / "parent.json", tmp_path / "plan.json"
    atomic_write_json(src, scores)
    atomic_write_json(parent, fixture_parent(scores))
    p.plan_workflow(src, parent, out, anchor=1)
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        p.plan_workflow(src, parent, out, anchor=1)
    assert out.read_bytes() == before
