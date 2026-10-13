"""Synthetic CPU-only tests; toy outcomes are never scientific evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest

from event_sae.research.head_matched.analysis import analyze, analyze_results, _interval, _spearman_draws
from event_sae.research.head_matched.protocol import ANALYSIS, conditions
from event_sae.research.output_head.provenance import fingerprint


def signed(document, key):
    value = {k: v for k, v in document.items() if k != key}
    value[key] = fingerprint(value)
    return value


def fixture():
    from test_head_matched_journal import hook_payload
    conds = conditions(9, [10, 11, 12, 13])
    cases = [{"suite": "libero_spatial", "task_id": t, "task_episode_idx": i,
              "seed": 100 + t * 2 + i, "initial_state_sha256": f"{t * 2 + i:064x}"}
             for t in range(10) for i in range(2)]
    predictions = [{"condition_id": row["condition_id"], "mean_head_kl": float(5 - j),
                    "task_mean_head_kl": {str(t): float(5 - j) for t in range(10)}}
                   for j, row in enumerate(conds[2:])]
    exp = signed({"schema_version": "head_matched_execution_v1", "plan_hash": "a" * 64,
        "conditions": conds, "cases": cases, "predictions": predictions, "analysis": dict(ANALYSIS),
        "total_rollouts": 140, "max_attempts": 140, "synthetic": False,
        "identity": {"sae_checkpoint_hash": "b" * 64},
        "validation": {"details": {"hook_summaries": {
            row["condition_id"]: hook_payload(row, cases[0])[0]
            for row in conds if row["mode"] != "raw"}}}}, "execution_hash")
    completed = {}
    for j, condition in enumerate(conds):
        rows = []
        failures = 0 if j < 2 else 10 - (j - 2) * 2
        for i, case in enumerate(cases):
            result = {"episode": {**case, "success": i >= failures, "caught_exception": None, "num_actions": 1},
                      "actions": [[0.] * 7], "hook_metrics": None, "hook_evidence": None}
            rows.append(signed({"execution_hash": exp["execution_hash"], "plan_hash": exp["plan_hash"],
                "condition": condition, "case": case, "attempt_index": 1, "status": "completed",
                "result": result}, "receipt_hash"))
        completed[condition["condition_id"]] = rows
    return exp, completed


def rebind(exp, completed):
    exp = signed(exp, "execution_hash")
    for rows in completed.values():
        for i, row in enumerate(rows):
            row["execution_hash"] = exp["execution_hash"]
            rows[i] = signed(row, "receipt_hash")
    return exp, completed


def test_five_direction_primary_secondary_and_exact_counts():
    exp, completed = fixture()
    result = analyze_results(exp, completed)
    assert result["primary"]["spearman"] == pytest.approx(1)
    assert [row["net_harm_cases"] for row in result["direction_effects"]] == [10, 8, 6, 4, 2]
    assert [row["signed_drop"] for row in result["direction_effects"]] == [.5, .4, .3, .2, .1]
    assert result["secondary"]["difference"] == .25
    assert result["total_rollouts"] == 140 and result["new_rollouts"] == 0
    assert not result["model_execution"]
    assert result["identity_validation"]["num_pairs"] == 20
    assert all(row["repair_cases"] == 0 for row in result["direction_effects"])
    assert all(len(row["task_effects"]) == 10 for row in result["direction_effects"])
    assert analyze_results(exp, completed) == result


def test_reordering_receipts_does_not_break_paired_case_identity():
    exp, completed = fixture()
    original = analyze_results(exp, completed)
    for rows in completed.values():
        rows.reverse()
    assert analyze_results(exp, completed) == original


def test_exact_integer_ties_and_undefined_correlations():
    values = _spearman_draws([1, 2, 3, 4, 5], [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
    assert values[0] == pytest.approx(np.corrcoef([1, 2, 3, 4, 5], [2.5, 2.5, 2.5, 2.5, 5])[0, 1])
    assert np.isnan(values[1])
    assert np.isnan(_spearman_draws([1, 1, 1, 1, 1], [[1, 2, 3, 4, 5]])[0])
    interval = _interval([.4, np.nan])
    assert interval["low"] is None and interval["undefined_replicates"] == 1


def test_signed_repairs_not_absolute_damage():
    exp, completed = fixture()
    # Raw and identity fail on all cases; every edited condition succeeds.
    for rows in completed.values():
        for i, receipt in enumerate(rows):
            receipt["result"]["episode"]["success"] = receipt["condition"]["mode"] not in ("raw", "identity")
            rows[i] = signed(receipt, "receipt_hash")
    result = analyze_results(exp, completed)
    assert all(row["signed_drop"] == -1 for row in result["direction_effects"])
    assert all(row["repair_cases"] == 20 for row in result["direction_effects"])
    assert result["primary"]["spearman"] is None
    assert result["primary"]["ci"]["undefined_replicates"] == 2000
    assert result["secondary"]["difference"] == 0


@pytest.mark.parametrize("kind", ["missing", "duplicate", "failure", "identity_actions", "identity_success", "wrong_case", "hash", "exception"])
def test_bad_receipts_are_not_outcomes(kind):
    exp, completed = fixture()
    rows = completed[exp["conditions"][2]["condition_id"]]
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows[0] = deepcopy(rows[1])
    elif kind == "failure": rows[0] = signed({**rows[0], "status": "failed"}, "receipt_hash")
    elif kind.startswith("identity"):
        identity = completed["identity-9"][0]
        if kind == "identity_actions": identity["result"]["actions"][0][0] = .1
        else: identity["result"]["episode"]["success"] = False
        completed["identity-9"][0] = signed(identity, "receipt_hash")
    elif kind == "wrong_case":
        rows[0]["result"]["episode"]["seed"] += 1
        rows[0] = signed(rows[0], "receipt_hash")
    elif kind == "hash": rows[0]["result"]["episode"]["success"] = True
    else:
        rows[0]["result"]["episode"]["caught_exception"] = "bad"
        rows[0] = signed(rows[0], "receipt_hash")
    with pytest.raises(ValueError):
        analyze_results(exp, completed)


@pytest.mark.parametrize("kind", ["settings", "missing_prediction", "prediction_id", "task_scope", "prediction_average", "negative_prediction"])
def test_wrong_frozen_prediction_contract_rejected(kind):
    exp, completed = fixture()
    if kind == "settings": exp["analysis"]["bootstrap_replicates"] = 10
    elif kind == "missing_prediction": exp["predictions"].pop()
    elif kind == "prediction_id": exp["predictions"][0]["condition_id"] = "wrong"
    elif kind == "task_scope": exp["predictions"][0]["task_mean_head_kl"].pop("0")
    elif kind == "prediction_average": exp["predictions"][0]["mean_head_kl"] = 2
    else: exp["predictions"][0]["mean_head_kl"] = -1
    exp, completed = rebind(exp, completed)
    with pytest.raises(ValueError):
        analyze_results(exp, completed)


def test_complete_journal_integration_and_nonoverwrite(tmp_path):
    from test_head_matched_journal import FakeBackend
    from event_sae.research.head_matched.journal import execute
    exp, _ = fixture()
    root = tmp_path / "run"
    execute(exp, root, FakeBackend(), approved_hash=exp["execution_hash"])
    out = analyze(exp, root, tmp_path / "analysis")
    assert out["status"] == "analyzed" and out["new_rollouts"] == 0
    data = json.loads((tmp_path / "analysis" / "analysis.json").read_text())
    assert data["analysis_hash"] == fingerprint({k: v for k, v in data.items() if k != "analysis_hash"})
    assert data["observed_norms"]["raw"]["hook_applied"] is False
    assert data["observed_norms"]["anchor-9"]["num_rows"] == 140
    assert data["observed_norms"]["anchor-9"]["sum_row_norms"]["control_realized_norm"] == 140
    assert data["observed_norms"]["identity-9"]["sum_row_norms"]["control_realized_norm"] == 0
    assert data["observed_norms"]["decoder-9-control-10"]["trajectory_edit_budgets_matched"] is False
    with pytest.raises(FileExistsError):
        analyze(exp, root, tmp_path / "analysis")


def test_incomplete_journal_no_output(tmp_path):
    from test_head_matched_journal import FakeBackend
    from event_sae.research.head_matched.journal import execute
    exp, _ = fixture()
    root, output = tmp_path / "run", tmp_path / "analysis"
    execute(exp, root, FakeBackend(), approved_hash=exp["execution_hash"], max_new_attempts=1)
    with pytest.raises(ValueError, match="Incomplete"):
        analyze(exp, root, output)
    assert not output.exists()
