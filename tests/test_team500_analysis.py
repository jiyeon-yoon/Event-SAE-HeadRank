"""Pure paired-statistics tests; no simulator/GPU execution."""
from copy import deepcopy
import hashlib

import pytest

from event_sae.research.team500.analysis import analyze_receipts, markdown_report, _spearman


def fixture():
    cases = [{"suite": "libero_spatial", "task_id": t, "task_episode_idx": i,
              "initial_state_sha256": hashlib.sha256(f"{t}:{i}".encode()).hexdigest()}
             for t in range(10) for i in range(50)]
    methods = {"mean_head_kl": [1, 2, 3, 4, 5], "tail_head_kl": [2, 1, 3, 4, 5],
               "event_max": [2, 3, 4, 5, 6]}
    conditions = [{"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.},
                  {"condition_id": "identity-1", "mode": "identity", "feature_id": 1, "alpha": 1.}]
    conditions += [{"condition_id": f"feature-{i}", "mode": "intervention", "feature_id": i,
                    "alpha": 0.} for i in range(1, 7)]
    plan = {"plan_hash": "a" * 64, "conditions": conditions, "cases": cases,
            "feature_ids": list(range(1, 7)), "methods": methods, "total_rollouts": 4000}
    raw = [{**case, "success": case["task_episode_idx"] < 40, "num_actions": 1,
            "actions": [[0., 0., 0., 0., 0., 0., -1.]]} for case in cases]
    receipts = {c["condition_id"]: {"episodes": deepcopy(raw)} for c in conditions}
    # Feature1 harms all 400 originally successful cases.
    for row in receipts["feature-1"]["episodes"]:
        row["success"] = False
    # Feature2 repairs all 100 failures.
    for row in receipts["feature-2"]["episodes"]:
        row["success"] = True
    # Feature3 swaps five successes and five failures per task: zero net, 20% flips.
    for row in receipts["feature-3"]["episodes"]:
        trial = row["task_episode_idx"]
        if trial < 5 or 40 <= trial < 45:
            row["success"] = not row["success"]
    return plan, receipts


def test_paired_effects_signed_improvement_and_flip_cancellation():
    plan, receipts = fixture()
    result = analyze_receipts(plan, receipts, bootstrap_replicates=100)
    rows = {r["feature_id"]: r for r in result["feature_effects"]}
    assert rows[1]["signed_drop"] == .8
    assert rows[2]["signed_drop"] == -.2
    assert rows[3]["signed_drop"] == 0 and rows[3]["outcome_flip_rate"] == .2
    assert rows[3]["harm_cases"] == 50 and rows[3]["repair_cases"] == 50
    assert result["raw_successes"] == 400 and result["pairs_per_feature"] == 500


def test_shared_draws_make_same_top5_sets_exact_zero_contrast():
    plan, receipts = fixture()
    result = analyze_receipts(plan, receipts, bootstrap_replicates=100)
    contrast = result["top_k_contrasts"][-1]
    assert contrast["same_top_k_feature_set"] is True
    assert contrast["top_k_signed_drop_difference"] == 0
    assert contrast["ci"] == {"low": 0., "high": 0.}
    report = markdown_report(result)
    assert "동시에 제거" in report and "독립적인 일반화" in report


def test_missing_frozen_top5_never_substituted():
    plan, receipts = fixture()
    del receipts["feature-6"]
    with pytest.raises(ValueError, match="every originally planned"):
        analyze_receipts(plan, receipts, bootstrap_replicates=100)


def test_identity_failure_blocks_all_analysis():
    plan, receipts = fixture()
    receipts["identity-1"]["episodes"][0]["actions"][0][0] = .0001
    with pytest.raises(ValueError, match="Exact native actions"):
        analyze_receipts(plan, receipts, bootstrap_replicates=100)


def test_integer_ties_and_undefined_rank_correlation():
    assert _spearman([1, 2, 3], [0, 0, 1]) == pytest.approx(.866025403784)
    assert _spearman([1, 2, 3], [0, 0, 0]) is None
    plan, receipts = fixture()
    scores = {fid: {m: float(7 - fid) for m in plan["methods"]} for fid in range(1, 7)}
    result = analyze_receipts(plan, receipts, bootstrap_replicates=100, score_rows=scores)
    assert result["predictor_comparison"][0]["panel_signed_drop_spearman"] is not None


def test_default_analysis_loads_bound_scores_without_repeating_optional_argument(tmp_path, monkeypatch):
    import json
    from event_sae.research.team500 import analysis
    from event_sae.research.output_head.provenance import atomic_write_json, fingerprint, sha256_file
    plan, receipts = fixture()
    rows = [{"feature_id": fid, **{m: float(7 - fid) for m in plan["methods"]}}
            for fid in range(1, 7)]
    source = {"feature_scores": rows}
    source["score_hash"] = fingerprint(source)
    score_path = tmp_path / "frozen-scores.json"
    atomic_write_json(score_path, source)
    plan.update(scores_hash=source["score_hash"], score_file_sha256=sha256_file(score_path),
                reference_bindings={"scores_path": str(score_path)})
    completed = {key: {"receipt": value, "result_hash": fingerprint(value)}
                 for key, value in receipts.items()}
    monkeypatch.setattr(analysis, "load_completed", lambda *args: (
        plan, completed, [], 4000, {"runtime_identity": {"cpu_mock_only": True}}))
    target = tmp_path / "analysis"
    result = analysis.analyze_experiment(tmp_path / "plan.json", tmp_path / "runs", target,
                                         bootstrap_replicates=100)
    saved = json.load(open(target / "analysis.json"))
    assert result["status"] == "analyzed"
    assert saved["predictor_comparison"][0]["panel_signed_drop_spearman"] is not None
    assert saved["score_file_sha256"] == sha256_file(score_path)
