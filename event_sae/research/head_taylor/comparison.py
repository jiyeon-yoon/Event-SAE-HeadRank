"""Read-only, selected-panel comparisons against existing paired outcomes.

The two historical cohorts keep their own raw baseline.  No new behavior is
inferred for an unmeasured globally ranked feature.  Statistics here are
descriptive point estimates, not a prospective prediction benchmark.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, atomic_write_text, fingerprint,
    read_json, sha256_file,
)

METRICS = (
    "taylor_abs_nll", "mean_head_kl", "tail_head_kl", "event_max",
    "readout_activation_frequency", "mean_edit_norm",
)
SCOPE = (
    "retrospective selected-panel comparison; historical cohorts and raw baselines "
    "are kept separate; no unmeasured/global behavioral outcome is inferred"
)
PINNED_PANEL_SHAS = {
    "d356c597dadf5f9ae3b23716d5f1cf7aeecb8a2b9783b18e790d73cd399b4a7e",
    "2968cb060c054ad2d56c09f6bab388b0d767c595ea8e3f23fe2a061a336046c7",
}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, low=None, high=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _verify_hash(data, name):
    if data.get(name) != fingerprint({k: v for k, v in data.items() if k != name}):
        raise ValueError(f"{name} is absent, stale, or tampered")


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[start]] == values[order[end]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for at in order[start:end]:
            result[at] = average
        start = end
    return result


def spearman(first, second):
    """Average-rank Spearman, with explicit undefined small/constant cases."""
    if len(first) != len(second):
        raise ValueError("Spearman arrays have unequal lengths")
    if len(first) < 3:
        return {"value": None, "status": "insufficient_features", "n": len(first)}
    first = [_number(v, "Spearman input") for v in first]
    second = [_number(v, "Spearman input") for v in second]
    x, y = _ranks(first), _ranks(second)
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    x, y = [v - mx for v in x], [v - my for v in y]
    denominator = math.sqrt(math.fsum(v * v for v in x) * math.fsum(v * v for v in y))
    if denominator == 0:
        return {"value": None, "status": "constant_rank", "n": len(first)}
    value = math.fsum(a * b for a, b in zip(x, y)) / denominator
    return {"value": max(-1., min(1., value)), "status": "ok", "n": len(first)}


def _scores(data):
    if data.get("schema_version") != "head_taylor_scores_v1":
        raise ValueError("expected head_taylor_scores_v1")
    _verify_hash(data, "score_hash")
    if data.get("sampling_mode") not in ("smoke", "full"):
        raise ValueError("unknown Taylor sampling mode")
    if type(data.get("development_only")) is not bool:
        raise ValueError("Taylor development_only flag is required")
    if data["sampling_mode"] == "smoke" and not data["development_only"]:
        raise ValueError("smoke scores cannot be declared non-development")
    rows = data.get("features")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty Taylor feature scores required")
    result = {}
    for row in rows:
        fid = _integer(row.get("feature_id"), "feature_id")
        if fid in result:
            raise ValueError("duplicate scored feature")
        result[fid] = {}
        for metric in METRICS:
            value = row.get(metric)
            result[fid][metric] = None if value is None else _number(value, metric, 0.)
    return result


def _case_key(row):
    suite = row.get("suite")
    state = row.get("initial_state_sha256")
    if not isinstance(suite, str) or not suite or not isinstance(state, str) or not state:
        raise ValueError("paired outcome suite/state identity is missing")
    return (suite, _integer(row.get("task_id"), "task_id"),
            _integer(row.get("task_episode_idx"), "task_episode_idx"), state,
            _integer(row.get("seed"), "episode seed"))


def read_panel(data):
    """Validate paired booleans and reconstruct task-weighted effects.

    Both shipped Exp3 outcome formats have ``paired_rows`` and
    ``feature_effects``.  Aggregate counts alone are not accepted because they
    cannot verify common raw cases or equal-task weighting.
    """
    if "analysis_hash" in data:
        _verify_hash(data, "analysis_hash")
    if type(data.get("synthetic")) is not bool:
        raise ValueError("outcome synthetic flag is required")
    ids = data.get("feature_ids")
    if not isinstance(ids, list) or not ids or any(type(v) is not int or v < 0 for v in ids):
        raise ValueError("valid outcome feature_ids required")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate outcome feature IDs")
    rows = data.get("paired_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("paired_rows required to validate behavioral outcomes")
    by_feature = {fid: {} for fid in ids}
    for row in rows:
        fid = _integer(row.get("feature_id"), "paired feature_id")
        if fid not in by_feature:
            raise ValueError("paired outcome feature not in declared panel")
        key = _case_key(row)
        if key in by_feature[fid]:
            raise ValueError("duplicate paired outcome case")
        if type(row.get("raw_success")) is not bool or type(row.get("edit_success")) is not bool:
            raise ValueError("success outcomes must be Boolean")
        if row.get("drop") != int(row["raw_success"]) - int(row["edit_success"]):
            raise ValueError("stored paired drop differs from Boolean outcomes")
        by_feature[fid][key] = row
    keys = sorted(by_feature[ids[0]])
    if not keys:
        raise ValueError("panel feature has no cases")
    raw = {key: by_feature[ids[0]][key]["raw_success"] for key in keys}
    tasks = defaultdict(list)
    for key in keys:
        tasks[(key[0], key[1])].append(key)
    effects = data.get("feature_effects")
    if not isinstance(effects, list) or any(not isinstance(row, dict) for row in effects):
        raise ValueError("outcome feature_effects list required")
    effect_ids = [row.get("feature_id") for row in effects]
    if len(set(effect_ids)) != len(effect_ids) or set(effect_ids) != set(ids):
        raise ValueError("outcome effects do not match declared feature IDs")
    stored = {row["feature_id"]: row for row in effects}
    result = {}
    fields = ("n_success_to_failure", "n_failure_to_success", "n_success_to_success", "n_failure_to_failure")
    for fid in sorted(ids):
        current = by_feature[fid]
        if sorted(current) != keys or any(current[key]["raw_success"] != raw[key] for key in keys):
            raise ValueError("panel features must share identical cases and raw outcomes")
        h = sum(raw[key] and not current[key]["edit_success"] for key in keys)
        r = sum(not raw[key] and current[key]["edit_success"] for key in keys)
        ss = sum(raw[key] and current[key]["edit_success"] for key in keys)
        ff = len(keys) - h - r - ss
        for name, value in zip(fields, (h, r, ss, ff)):
            if _integer(stored[fid].get(name), name) != value:
                raise ValueError("stored transition count differs from paired outcomes")
        count = stored[fid].get("num_pairs", stored[fid].get("num_valid_pairs"))
        if _integer(count, "num_pairs", 1) != len(keys):
            raise ValueError("stored pair count differs from paired outcomes")
        drop_by_task, flip_by_task = [], []
        for selected in tasks.values():
            harm = sum(raw[key] and not current[key]["edit_success"] for key in selected)
            repair = sum(not raw[key] and current[key]["edit_success"] for key in selected)
            drop_by_task.append((harm - repair) / len(selected))
            flip_by_task.append((harm + repair) / len(selected))
        drop = math.fsum(drop_by_task) / len(tasks)
        flip = math.fsum(flip_by_task) / len(tasks)
        if abs(_number(stored[fid].get("drop"), "stored drop", -1., 1.) - drop) > 1e-10:
            raise ValueError("stored task-weighted drop differs from paired outcomes")
        result[fid] = {"feature_id": fid, "num_pairs": len(keys), "num_tasks": len(tasks),
            "signed_drop": drop, "outcome_flip": flip, "success_to_failure": h,
            "failure_to_success": r, "raw_successes": sum(raw.values()),
            "pooled_outcome_flip": (h + r) / len(keys)}
    return result


def compare_data(scores, panels, *, top_k=3):
    """Pure comparison for tests and workflows; panels remain separate."""
    _integer(top_k, "top_k", 1)
    lookup = _scores(scores)
    if top_k > len(lookup):
        raise ValueError("top_k exceeds scored population")
    if not panels or len({name for name, _ in panels}) != len(panels):
        raise ValueError("one or more uniquely named panels required")
    output = []
    for name, data in panels:
        if not isinstance(name, str) or not name or any(c in name for c in "\n\r|"):
            raise ValueError("invalid panel name")
        effects = read_panel(data)
        # Use exactly one complete-case panel for all predictors, not a different
        # feature population for whichever method happens to omit a score.
        common = sorted(fid for fid in effects if fid in lookup
                        and all(lookup[fid][m] is not None for m in METRICS))
        comparisons = []
        for metric in METRICS:
            ranked = sorted((fid for fid in lookup if lookup[fid][metric] is not None),
                            key=lambda fid: (-lookup[fid][metric], fid))
            top = ranked[:top_k]
            measured = [fid for fid in top if fid in effects]
            complete = len(top) == top_k and len(measured) == top_k
            boundary_tie = (len(ranked) > top_k
                            and lookup[ranked[top_k - 1]][metric] == lookup[ranked[top_k]][metric])
            comparisons.append({"metric": metric, "panel_feature_ids": common,
                "signed_drop_spearman": spearman([lookup[fid][metric] for fid in common],
                                                [effects[fid]["signed_drop"] for fid in common]),
                "outcome_flip_spearman": spearman([lookup[fid][metric] for fid in common],
                                                 [effects[fid]["outcome_flip"] for fid in common]),
                "top_k": top_k, "scored_metric_population": len(ranked),
                "original_topk_feature_ids": top, "topk_boundary_tie": boundary_tie,
                "measured_topk_feature_ids": measured, "topk_coverage": len(measured) / top_k,
                "topk_signed_drop": math.fsum(effects[fid]["signed_drop"] for fid in top) / top_k if complete else None,
                "topk_outcome_flip": math.fsum(effects[fid]["outcome_flip"] for fid in top) / top_k if complete else None,
                "topk_effect_status": "complete" if complete else "unmeasured_original_topk"})
        output.append({"name": name, "synthetic": data["synthetic"],
            "behavior_feature_ids": sorted(effects), "num_behavior_features": len(effects),
            "common_panel_feature_ids": common, "num_common_features": len(common),
            "excluded_from_common_panel": sorted(set(effects) - set(common)),
            "num_pairs_per_feature": next(iter(effects.values()))["num_pairs"],
            "feature_effects": [effects[fid] for fid in sorted(effects)], "comparisons": comparisons})
    return {"schema_version": "head_taylor_comparison_v1", "new_rollouts": 0,
        "model_execution": False, "score_hash": scores["score_hash"],
        "synthetic": scores.get("synthetic", False),
        "sampling_mode": scores["sampling_mode"], "development_only": scores["development_only"],
        "scope": SCOPE, "uncertainty": "descriptive point estimates; no confidence intervals or hypothesis tests",
        "rank_ties": "average ranks for correlation; lower feature ID breaks Top-K ties",
        "flip_definition": "equally task-weighted mean(success-to-failure + failure-to-success)/cases within task",
        "population": "scored cached-active feature pool, not the entire dictionary",
        "panels": output}


def _fmt(value):
    return "undefined" if value is None else f"{value:.4f}"


def report_markdown(result):
    lines = ["# Taylor importance — existing behavioral panel comparison", "",
        "New rollouts: 0. Model/simulator execution: no.", "",
        f"Sampling: {result['sampling_mode']}. Development only: {result['development_only']}.", "",
        "Taylor is a frozen-SAE adaptation of a pruning criterion, not a reproduction of the original CNN experiments.",
        "Taylor uses a new smooth-head FP32 gradient; Mean/Tail Head-KL are archival BF16 finite-removal scores. "
        "The gradient is not the exact derivative of quantized BF16 computation.",
        "Historical baseline elapsed times are not compared with Taylor timing on the new GPU. "
        "Only a separately labeled same-device benchmark can support a timing comparison.", ""]
    for panel in result["panels"]:
        lines += [f"## {panel['name']}", "",
            f"Behavior features: {panel['num_behavior_features']}; common complete-score panel: {panel['num_common_features']}; "
            f"paired cases per feature: {panel['num_pairs_per_feature']}.", "",
            f"Excluded from common panel (missing score/inactive pool): {panel['excluded_from_common_panel']}.", "",
            "| Predictor | Net-loss Spearman | Outcome-flip Spearman | Original Top-K | Coverage | Top-K net drop | Top-K flip |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
        for row in panel["comparisons"]:
            lines.append(f"| {row['metric']} | {_fmt(row['signed_drop_spearman']['value'])} | "
                f"{_fmt(row['outcome_flip_spearman']['value'])} | {row['original_topk_feature_ids']} | "
                f"{row['topk_coverage']:.3f} | {_fmt(row['topk_signed_drop'])} | {_fmt(row['topk_outcome_flip'])} |")
        lines.append("")
    lines += ["## Interpretation limits", "",
        "- Cohorts retain their own raw baseline; they are never pooled into one behavioral dataset.",
        "- Correlations use the same explicitly listed complete-score feature panel for every predictor.",
        "- Global Top-K means Top-K of the supplied scored pool. Missing effects remain undefined; measured lower ranks are never substituted.",
        "- These panels were selected using previous rankings and outcomes were already available. This is retrospective, not independent confirmation or all-feature prediction accuracy.",
        "- Values are descriptive point estimates, without uncertainty intervals or multiple-comparison claims.",
        "- A positive signed drop means harm; an outcome flip also includes improvements and is not a measure of damage alone.",
        "- Historical collection metadata, SAE training overlap, seed protocol and hardware differences are not repaired by this analysis.",
        "- This report neither establishes a new method's novelty nor semantic or task-specific feature meaning.", ""]
    return "\n".join(lines)


def compare_workflow(scores_path, panels, output, *, top_k=3):
    """Write a new CPU-only report from ``[(panel_name, JSON_path), ...]``."""
    if not isinstance(panels, (list, tuple)) or not panels:
        raise ValueError("at least one named outcome panel required")
    from .provenance import protected_source_roots
    paths = [Path(path) for _, path in panels]
    assert_safe_output(output, protected_source_roots())
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("duplicate outcome panel file")
    scores = read_json(scores_path)
    _scores(scores)  # Verify integrity before trusting protected input roots.
    if type(scores.get("synthetic")) is not bool:
        raise ValueError("Taylor synthetic flag is required")
    protected = scores.get("source_readonly_roots", [])
    if not isinstance(protected, list) or any(not isinstance(path, str) or not path for path in protected):
        raise ValueError("source_readonly_roots must be a pathname list")
    if not scores["synthetic"]:
        if not protected:
            raise ValueError("real comparison requires protected source_readonly_roots")
        if scores["sampling_mode"] != "full" or scores["development_only"]:
            raise ValueError("real comparison requires full, non-development Taylor scores")
        sample = scores.get("scoring_sample")
        if not isinstance(sample, dict) or sample.get("same_sample_as_frozen_baselines") is not True:
            raise ValueError("real comparison requires the same sample as frozen baselines")
        feature_ids = [row["feature_id"] for row in scores["features"]]
        for key in ("scored_feature_ids", "frozen_active_feature_ids"):
            actual = scores.get(key)
            if (not isinstance(actual, list) or any(type(fid) is not int for fid in actual)
                    or actual != feature_ids):
                raise ValueError("real comparison requires the complete frozen active feature universe")
    target = assert_safe_output(output, [scores_path, *paths, *protected])
    if target.exists():
        raise ValueError("comparison requires a new output directory")
    panel_shas = [sha256_file(path) for path in paths]
    if not scores["synthetic"] and (len(panel_shas) != 2 or set(panel_shas) != PINNED_PANEL_SHAS):
        raise ValueError("real v1 comparison requires both exact pinned historical outcome panel files")
    result = compare_data(scores, [(name, read_json(path)) for (name, _), path in zip(panels, paths)], top_k=top_k)
    if not scores["synthetic"] and any(panel["synthetic"] for panel in result["panels"]):
        raise ValueError("real comparison cannot use synthetic behavioral panels")
    result["behavior_source_binding"] = "synthetic_test_inputs" if scores["synthetic"] else "both_fixed_v1_panel_sha256_verified"
    result["sources"] = {"scores": {"path": str(Path(scores_path).resolve()), "sha256": sha256_file(scores_path)},
        "panels": [{"name": name, "path": str(path.resolve()), "sha256": sha256_file(path)}
                   for (name, _), path in zip(panels, paths)]}
    result["comparison_hash"] = fingerprint(result)
    target.mkdir(parents=True, exist_ok=False)
    atomic_write_json(target / "comparison.json", result)
    atomic_write_text(target / "report.md", report_markdown(result))
    return {"status": "compared", "output": str(target), "new_rollouts": 0,
            "model_execution": False, "panels": len(result["panels"]),
            "comparison_hash": result["comparison_hash"]}
