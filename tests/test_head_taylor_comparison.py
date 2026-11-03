import copy
import json

import pytest

from event_sae.research.head_taylor.comparison import (
    METRICS, compare_data, compare_workflow, read_panel, spearman,
)
from event_sae.research.output_head.provenance import fingerprint


def scores():
    data = {"schema_version": "head_taylor_scores_v1", "sampling_mode": "full", "synthetic": True,
            "development_only": False, "features": [
                {"feature_id": fid, **{m: float(5 - fid) for m in METRICS}}
                for fid in range(4)]}
    data["score_hash"] = fingerprint(data)
    return data


def panel(ids=(0, 1, 2), *, raw=(True, True, False, False), changes=None):
    if changes is None:
        changes = {0: (False, True, False, False),
                   1: (True, True, True, False), 2: (True, True, False, False)}
    paired, effects = [], []
    for fid in ids:
        edit = changes[fid]
        for at, (r, e) in enumerate(zip(raw, edit)):
            paired.append({"feature_id": fid, "suite": "suite", "task_id": at // 2,
                "task_episode_idx": at % 2, "initial_state_sha256": "state" + str(at),
                "seed": 0, "raw_success": r, "edit_success": e, "drop": int(r) - int(e)})
        h, r = sum(a and not b for a, b in zip(raw, edit)), sum(not a and b for a, b in zip(raw, edit))
        effects.append({"feature_id": fid, "drop": (h - r) / len(raw), "num_pairs": len(raw),
            "n_success_to_failure": h, "n_failure_to_success": r,
            "n_success_to_success": sum(a and b for a, b in zip(raw, edit)),
            "n_failure_to_failure": sum(not a and not b for a, b in zip(raw, edit))})
    return {"synthetic": True, "feature_ids": list(ids), "feature_effects": effects, "paired_rows": paired}


def test_spearman_ties_and_undefined():
    assert spearman([1, 1, 3, 4], [2, 2, 6, 8])["value"] == pytest.approx(1.)
    assert spearman([1, 1, 1], [0, 1, 2])["status"] == "constant_rank"
    assert spearman([1, 2], [2, 1])["status"] == "insufficient_features"
    assert spearman([1, 2, 3], [3, 2, 1])["value"] == pytest.approx(-1.)
    with pytest.raises(ValueError):
        spearman([1, 2, float("nan")], [1, 2, 3])


def test_separate_raw_baselines_and_outcomes():
    second = panel(raw=(True, False, False, False), changes={
        0: (False, False, False, False), 1: (True, False, True, False), 2: (True, False, False, False)})
    result = compare_data(scores(), [("original", panel()), ("later", second)])
    assert len(result["panels"]) == 2
    assert result["panels"][0]["feature_effects"][0]["raw_successes"] == 2
    assert result["panels"][1]["feature_effects"][0]["raw_successes"] == 1
    assert result["new_rollouts"] == 0 and not result["model_execution"]


def test_topk_never_substitutes_lower_measured_features():
    data = scores()
    data["features"][3]["taylor_abs_nll"] = 100.
    data["score_hash"] = fingerprint({k: v for k, v in data.items() if k != "score_hash"})
    result = compare_data(data, [("original", panel())])
    row = result["panels"][0]["comparisons"][0]
    assert row["original_topk_feature_ids"] == [3, 0, 1]
    assert row["measured_topk_feature_ids"] == [0, 1]
    assert row["topk_coverage"] == pytest.approx(2 / 3)
    assert row["topk_signed_drop"] is None and row["topk_outcome_flip"] is None


def test_same_complete_feature_panel_for_every_metric():
    data = scores()
    data["features"][1]["event_max"] = None
    data["score_hash"] = fingerprint({k: v for k, v in data.items() if k != "score_hash"})
    result = compare_data(data, [("original", panel())])
    assert result["panels"][0]["excluded_from_common_panel"] == [1]
    assert all(r["panel_feature_ids"] == [0, 2] for r in result["panels"][0]["comparisons"])


def test_harm_and_repair_cancel_net_not_flip():
    data = panel(changes={0: (False, True, True, False),
                          1: (True, True, True, False), 2: (True, True, False, False)})
    effect = read_panel(data)[0]
    assert effect["signed_drop"] == 0.
    assert effect["outcome_flip"] == .5
    assert effect["success_to_failure"] == effect["failure_to_success"] == 1


def test_task_weighted_outcomes_not_pooled_when_task_sizes_differ():
    data = panel(ids=(0,), changes={0: (False, True, False, False)})
    # One harmed case in task 0; two unchanged cases in task 1.  Equal-task
    # loss/flip is 0.5, whereas pooled loss/flip is only 1/3.
    data["paired_rows"].pop(1)
    data["feature_effects"][0].update(drop=.5, num_pairs=3, n_success_to_success=0)
    effect = read_panel(data)[0]
    assert effect["signed_drop"] == .5
    assert effect["outcome_flip"] == .5
    assert effect["pooled_outcome_flip"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("mutation", [
    lambda d: d["paired_rows"].append(copy.deepcopy(d["paired_rows"][0])),
    lambda d: d["feature_effects"][0].update(n_success_to_failure=99),
    lambda d: d["feature_effects"][0].update(drop=.9),
    lambda d: d["paired_rows"][0].update(raw_success=1),
    lambda d: d["paired_rows"][0].update(drop=-1),
    lambda d: d["feature_ids"].append(0),
    lambda d: d.update(paired_rows=[]),
    lambda d: d["paired_rows"][4].update(initial_state_sha256="other"),
])
def test_reject_invalid_panel(mutation):
    data = panel()
    mutation(data)
    with pytest.raises(ValueError):
        read_panel(data)


def test_tampered_score_and_analysis_hash_rejected():
    data = scores()
    data["features"][0]["taylor_abs_nll"] = 100.
    with pytest.raises(ValueError, match="score_hash"):
        compare_data(data, [("original", panel())])
    data = panel()
    data["analysis_hash"] = "tampered"
    with pytest.raises(ValueError, match="analysis_hash"):
        read_panel(data)


def test_smoke_is_labeled_development():
    data = scores()
    data.update(sampling_mode="smoke", development_only=True)
    data["score_hash"] = fingerprint({k: v for k, v in data.items() if k != "score_hash"})
    assert compare_data(data, [("original", panel())])["development_only"] is True


def test_workflow_nonoverwriting_hashes_and_report(tmp_path):
    scorefile, panelfile = tmp_path / "scores.json", tmp_path / "panel.json"
    scorefile.write_text(json.dumps(scores()))
    panelfile.write_text(json.dumps(panel()))
    target = tmp_path / "output"
    result = compare_workflow(scorefile, [("original", panelfile)], target)
    assert result["status"] == "compared"
    saved = json.loads((target / "comparison.json").read_text())
    assert saved["comparison_hash"] == fingerprint({k: v for k, v in saved.items() if k != "comparison_hash"})
    assert "retrospective" in (target / "report.md").read_text()
    with pytest.raises(ValueError, match="new output"):
        compare_workflow(scorefile, [("original", panelfile)], target)
    with pytest.raises(ValueError, match="overlaps"):
        compare_workflow(scorefile, [("original", panelfile)], tmp_path)


def test_topk_boundary_tie_is_exposed():
    data = scores()
    for row in data["features"]:
        row["taylor_abs_nll"] = 1.
    data["score_hash"] = fingerprint({k: v for k, v in data.items() if k != "score_hash"})
    row = compare_data(data, [("original", panel())])["panels"][0]["comparisons"][0]
    assert row["topk_boundary_tie"] is True
    assert row["original_topk_feature_ids"] == [0, 1, 2]


def real_scores(protected):
    data = scores()
    data.update(synthetic=False, source_readonly_roots=[str(protected)],
        scored_feature_ids=[0, 1, 2, 3], frozen_active_feature_ids=[0, 1, 2, 3],
        scoring_sample={"same_sample_as_frozen_baselines": True})
    return data


def rehash(data):
    data["score_hash"] = fingerprint({k: v for k, v in data.items() if k != "score_hash"})
    return data


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d.update(sampling_mode="smoke", development_only=True), "full, non-development"),
    (lambda d: d.update(development_only=True), "full, non-development"),
    (lambda d: d.update(scoring_sample={"same_sample_as_frozen_baselines": False}), "same sample"),
    (lambda d: d.update(scored_feature_ids=[0, 1]), "complete frozen"),
    (lambda d: d.update(frozen_active_feature_ids=[0, 1]), "complete frozen"),
    (lambda d: d.update(source_readonly_roots=[]), "protected source"),
])
def test_real_workflow_rejects_incomplete_comparison(tmp_path, mutation, message):
    data = real_scores(tmp_path / "reference")
    mutation(data)
    scorefile, panelfile = tmp_path / "scores.json", tmp_path / "panel.json"
    scorefile.write_text(json.dumps(rehash(data)))
    panelfile.write_text(json.dumps(panel()))
    with pytest.raises(ValueError, match=message):
        compare_workflow(scorefile, [("original", panelfile)], tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_real_workflow_rejects_unpinned_or_missing_panel(tmp_path):
    scorefile, panelfile = tmp_path / "scores.json", tmp_path / "panel.json"
    scorefile.write_text(json.dumps(rehash(real_scores(tmp_path / "reference"))))
    panelfile.write_text(json.dumps(panel()))
    with pytest.raises(ValueError, match="both exact pinned"):
        compare_workflow(scorefile, [("original", panelfile)], tmp_path / "output")


def test_protected_original_result_sibling_cannot_receive_report(tmp_path):
    root = tmp_path / "read_only_results"
    root.mkdir()
    scorefile, panelfile = tmp_path / "scores.json", root / "outcomes.json"
    scorefile.write_text(json.dumps(rehash(real_scores(root))))
    panelfile.write_text(json.dumps(panel()))
    with pytest.raises(ValueError, match="overlaps"):
        compare_workflow(scorefile, [("original", panelfile)], root / "new_analysis")
    assert not (root / "new_analysis").exists()


def test_real_workflow_accepts_only_both_verified_panels(tmp_path, monkeypatch):
    from event_sae.research.head_taylor import comparison
    from event_sae.research.output_head.provenance import sha256_file
    first, second = panel(), panel(raw=(True, False, False, False), changes={
        0: (False, False, False, False), 1: (True, False, True, False), 2: (True, False, False, False)})
    first["synthetic"] = second["synthetic"] = False
    oldfile, newfile, scorefile = tmp_path / "old.json", tmp_path / "new.json", tmp_path / "scores.json"
    oldfile.write_text(json.dumps(first))
    newfile.write_text(json.dumps(second))
    scorefile.write_text(json.dumps(rehash(real_scores(tmp_path / "reference"))))
    # Test fixtures stand in for the independently pinned public bytes.
    monkeypatch.setattr(comparison, "PINNED_PANEL_SHAS", {sha256_file(oldfile), sha256_file(newfile)})
    result = compare_workflow(scorefile, [("original", oldfile), ("later", newfile)], tmp_path / "output")
    assert result["status"] == "compared"
    saved = json.loads((tmp_path / "output" / "comparison.json").read_text())
    assert saved["behavior_source_binding"] == "both_fixed_v1_panel_sha256_verified"
