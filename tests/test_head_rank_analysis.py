"""Synthetic receipt fixtures only; not GPU or simulator parity evidence."""
from copy import deepcopy
import hashlib

import pytest

from event_sae.research.head_rank.analysis import analyze_results, render_report
from event_sae.research.head_rank.protocol import plan_panel
from event_sae.research.output_head.provenance import fingerprint


def rehash(doc, key):
    doc.pop(key, None)
    doc[key] = fingerprint(doc)
    return doc


def fixture(*, feature_count=4, same_rankings=False, same_outcomes=False):
    rows = [{"feature_id": fid, "mean_head_kl": float(fid), "tail_head_kl": float(20 - fid),
             "direction_excess_kl": float((fid * 5) % 17), "event_max": float((fid * 7) % 19)}
            for fid in range(20)]
    if same_rankings:
        for row in rows:
            for metric in ("mean_head_kl", "tail_head_kl", "direction_excess_kl", "event_max"):
                row[metric] = float(row["feature_id"])
    cases = [{"suite": "libero_spatial", "task_id": task, "task_episode_idx": index,
              "initial_state_sha256": hashlib.sha256(f"{task}-{index}".encode()).hexdigest(), "seed": 42}
             for task in (0, 1) for index in (0, 1)]
    plan = plan_panel({"feature_scores": rows, "synthetic": False}, feature_count=feature_count,
                       task_ids=[0, 1], cases_per_task=2, evaluation_cases=cases)
    panel = plan["panel_feature_ids"]
    conditions = [{"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.0},
                  {"condition_id": "identity", "mode": "identity", "feature_id": panel[0], "alpha": 1.0}]
    conditions += [{"condition_id": f"feature-{fid}", "mode": "intervention", "feature_id": fid, "alpha": 0.0}
                   for fid in panel]
    experiment = rehash({"schema_version": "head_rank_execution_v1", "synthetic": True,
        "prediction_plan": plan, "plan_hash": plan["plan_hash"], "conditions": conditions, "cases": cases,
        "total_rollouts": len(conditions) * len(cases),
        "analysis": {"bootstrap_replicates": 80, "bootstrap_seed": 1729, "confidence_level": .95}}, "execution_hash")
    completed = {}
    raw = [True, False, True, False]
    outcomes = [[False, False, False, False], [True, True, True, True], [True, False, False, False], raw]
    for condition in conditions:
        values = raw if condition["mode"] in ("raw", "identity") or same_outcomes else outcomes[panel.index(condition["feature_id"]) % 4]
        completed[condition["condition_id"]] = []
        for index, case in enumerate(cases):
            result = {"episode": {**case, "success": values[index], "caught_exception": None, "num_actions": 1},
                      "actions": [[.1, .2, .3, .4, .5, .6, 1.]], "hook_metrics": None, "hook_evidence": None}
            completed[condition["condition_id"]].append(rehash({
                "schema_version": "head_rank_case_receipt_v1", "status": "completed", "attempt_index": 1,
                "execution_hash": experiment["execution_hash"], "plan_hash": plan["plan_hash"],
                "condition": condition, "case": case, "result": result}, "receipt_hash"))
    return experiment, completed


def test_signed_effects_include_repairs_and_equal_task_weighting():
    experiment, completed = fixture()
    result = analyze_results(experiment, completed, bootstrap_replicates=80)
    effects = {row["feature_id"]: row for row in result["feature_effects"]}
    first, second, third, fourth = experiment["prediction_plan"]["panel_feature_ids"]
    assert effects[first]["drop"] == .5
    assert effects[second]["drop"] == -.5
    assert effects[third]["drop"] == .25
    assert effects[fourth]["drop"] == 0
    assert effects[first]["n_success_to_failure"] == 2
    assert effects[second]["n_failure_to_success"] == 2
    assert result["identity_validation"]["num_pairs"] == 4
    assert result["synthetic"] is True
    assert result["new_rollouts"] == 0
    assert result == analyze_results(experiment, completed, bootstrap_replicates=80)


def test_missing_original_topk_stays_null_without_substitution():
    experiment, completed = fixture(feature_count=2)
    result = analyze_results(experiment, completed, bootstrap_replicates=80)
    for row in result["predictor_comparison"]:
        assert row["missing_topk_feature_ids"]
        assert row["topk_mean_drop"] is None
        assert row["topk_mean_drop_ci"]["status"] == "incomplete_coverage"
        assert row["topk_substitution_performed"] is False
    assert all(row["difference"] is None for row in result["topk_contrasts"])


def test_shared_bootstrap_identical_rankings_have_exactly_zero_contrasts():
    experiment, completed = fixture(feature_count=8, same_rankings=True)
    result = analyze_results(experiment, completed, bootstrap_replicates=80)
    for contrast in result["spearman_contrasts"] + result["topk_contrasts"]:
        assert contrast["difference"] == 0
        assert contrast["ci"]["low"] == contrast["ci"]["high"] == 0
    assert result["uncertainty"]["bootstrap_replicates"] == 80
    assert not result["uncertainty"]["simultaneous_intervals"]


def test_constant_outcomes_have_undefined_spearman_not_zero():
    experiment, completed = fixture(same_outcomes=True)
    result = analyze_results(experiment, completed, bootstrap_replicates=80)
    for row in result["predictor_comparison"]:
        assert row["spearman"] is None
        assert row["spearman_status"] == "undefined"
        assert row["spearman_ci"]["undefined_replicates"] == 80
    assert all(row["difference"] is None for row in result["spearman_contrasts"])


def test_result_order_does_not_change_pairing():
    experiment, completed = fixture()
    a = analyze_results(experiment, completed, bootstrap_replicates=80)
    for rows in completed.values():
        rows.reverse()
    assert analyze_results(experiment, completed, bootstrap_replicates=80) == a


@pytest.mark.parametrize("change", ["missing_condition", "extra_condition", "missing_case", "duplicate_case"])
def test_incomplete_or_duplicate_outcomes_rejected(change):
    experiment, completed = fixture()
    if change == "missing_condition":
        completed.pop("identity")
    elif change == "extra_condition":
        completed["extra"] = completed["raw"]
    elif change == "missing_case":
        completed["raw"].pop()
    elif change == "duplicate_case":
        completed["raw"][1] = deepcopy(completed["raw"][0])
    with pytest.raises(ValueError):
        analyze_results(experiment, completed, bootstrap_replicates=80)


@pytest.mark.parametrize("change", ["actions", "outcome", "count"])
def test_identity_requires_exact_actions_and_outcomes_not_same_success_rate(change):
    experiment, completed = fixture()
    if change == "actions":
        completed["identity"][0]["result"]["actions"][0][0] += .01
    elif change == "outcome":
        # Keep identity SR identical to raw by swapping two opposite outcomes.
        completed["identity"][0]["result"]["episode"]["success"] = False
        completed["identity"][1]["result"]["episode"]["success"] = True
    elif change == "count":
        completed["identity"][0]["result"]["episode"]["num_actions"] = 2
        completed["identity"][0]["result"]["actions"].append([0.] * 7)
    for row in completed["identity"]:
        rehash(row, "receipt_hash")
    with pytest.raises(ValueError, match="identity"):
        analyze_results(experiment, completed, bootstrap_replicates=80)


@pytest.mark.parametrize("change", ["caught_exception", "success_nonbool", "nonfinite_action", "bad_dimension", "case_seed"])
def test_invalid_episode_records_rejected(change):
    experiment, completed = fixture()
    row = completed["raw"][0]
    if change == "caught_exception":
        row["result"]["episode"]["caught_exception"] = "renderer failed"
    elif change == "success_nonbool":
        row["result"]["episode"]["success"] = 1
    elif change == "nonfinite_action":
        # The JSON integrity layer rejects nonfinite numbers before statistics.
        row["result"]["actions"][0][0] = float("inf")
        with pytest.raises(ValueError):
            analyze_results(experiment, completed, bootstrap_replicates=80)
        return
    elif change == "bad_dimension":
        row["result"]["actions"][0].pop()
    elif change == "case_seed":
        row["result"]["episode"]["seed"] += 1
    rehash(row, "receipt_hash")
    with pytest.raises(ValueError):
        analyze_results(experiment, completed, bootstrap_replicates=80)


def test_plan_execution_and_receipt_hash_mismatches_rejected():
    original_experiment, original_completed = fixture()
    for target in ("plan", "execution", "receipt"):
        experiment, completed = deepcopy(original_experiment), deepcopy(original_completed)
        if target == "plan":
            experiment["prediction_plan"]["panel_feature_ids"].reverse()
            rehash(experiment, "execution_hash")
        elif target == "execution":
            experiment["total_rollouts"] += 1
        else:
            completed["raw"][0]["result"]["episode"]["success"] = False
        with pytest.raises(ValueError, match="hash"):
            analyze_results(experiment, completed, bootstrap_replicates=80)


def test_rehashed_wrong_receipt_plan_binding_still_rejected():
    experiment, completed = fixture()
    row = completed["raw"][0]
    row["plan_hash"] = "a" * 64
    rehash(row, "receipt_hash")
    with pytest.raises(ValueError, match="binding mismatch"):
        analyze_results(experiment, completed, bootstrap_replicates=80)


def test_analysis_cannot_posthoc_change_frozen_bootstrap_parameters():
    experiment, completed = fixture()
    with pytest.raises(ValueError, match="differs from the frozen"):
        analyze_results(experiment, completed, bootstrap_replicates=2000)


def test_report_explains_claim_limits_and_percentage_points():
    experiment, completed = fixture()
    result = analyze_results(experiment, completed, bootstrap_replicates=80)
    text = render_report(result)
    assert "10 percentage points" in text
    assert "not all 32,768" in text
    assert "lower-ranked measured features are never substituted" in text
    assert "not multiplicity-adjusted" in text
    assert "Negative means improvement" in text
    assert "semantic meaning" in text
