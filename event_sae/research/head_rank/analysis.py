"""Read-only analysis of a complete, prospectively frozen feature panel.

The bootstrap shares each within-task case resample across every feature and
predictor.  Its uncertainty is conditional on these fixed tasks, selected
features, frozen scores and SAE; it is not dictionary-wide prediction evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any, Mapping

from event_sae.research.output_head.provenance import (
    atomic_write_json, atomic_write_text, fingerprint, read_json, sha256_file,
)
from event_sae.research.output_head.results import (
    _bootstrap_drops, _interval, spearman_correlation,
)
from .protocol import METRICS, verify_plan


def _mean(values):
    if not values:
        raise ValueError("Cannot average an empty population")
    return math.fsum(values) / len(values)


def _hash_checked(document, key):
    if not isinstance(document, Mapping) or document.get(key) != fingerprint(
        {k: value for k, value in document.items() if k != key}
    ):
        raise ValueError(f"Missing or mismatched {key}")


def _analysis_parameters(experiment, bootstrap_replicates):
    settings = experiment.get("analysis", {})
    count, seed, level = (settings.get("bootstrap_replicates"),
                          settings.get("bootstrap_seed"), settings.get("confidence_level"))
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Experiment must freeze positive bootstrap_replicates and bootstrap_seed")
    if isinstance(level, bool) or not isinstance(level, (float, int)) or level != .95:
        raise ValueError("This analysis requires a frozen 0.95 confidence level")
    if type(bootstrap_replicates) is not int or bootstrap_replicates != count:
        raise ValueError("Analysis bootstrap_replicates differs from the frozen experiment")
    return count, seed, level


def _validated_outcomes(experiment, completed):
    """Defense-in-depth pairing checks after the disk receipt validator.

    The production loader also verifies hook evidence files and runtime
    provenance. This function intentionally does not replace that loader.
    """
    _hash_checked(experiment, "execution_hash")
    plan = experiment.get("prediction_plan")
    verify_plan(plan)
    if experiment.get("plan_hash") != plan["plan_hash"]:
        raise ValueError("Experiment/prediction plan hash mismatch")
    if not plan.get("evaluation_cases"):
        raise ValueError("Behavioral analysis requires frozen evaluation cases")
    conditions = experiment.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Experiment requires conditions")
    by_condition = {}
    for condition in conditions:
        cid = condition.get("condition_id")
        if not isinstance(cid, str) or not cid or cid in by_condition:
            raise ValueError("Conditions need distinct nonempty condition_id")
        by_condition[cid] = condition
    if set(completed) != set(by_condition):
        raise ValueError("Incomplete or extra condition results")
    raw_ids = [cid for cid, item in by_condition.items() if item.get("mode") == "raw"]
    identity_ids = [cid for cid, item in by_condition.items() if item.get("mode") == "identity"]
    feature_conditions = {item.get("feature_id"): cid for cid, item in by_condition.items()
                          if item.get("mode") == "intervention"}
    if (len(raw_ids) != 1 or len(identity_ids) != 1
            or len(feature_conditions) + 2 != len(conditions)
            or set(feature_conditions) != set(plan["panel_feature_ids"])):
        raise ValueError("Require exactly raw, identity and each frozen feature once")
    for fid, cid in feature_conditions.items():
        if by_condition[cid].get("alpha") != 0.0:
            raise ValueError(f"Feature {fid}: analysis requires complete alpha=0 removal")
    if by_condition[identity_ids[0]].get("alpha") != 1.0:
        raise ValueError("Identity must use alpha=1")
    if by_condition[raw_ids[0]].get("alpha") != 1.0 or by_condition[raw_ids[0]].get("feature_id") is not None:
        raise ValueError("Raw must use alpha=1 and no feature")
    cases = experiment.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Experiment requires frozen cases")
    case_keys = [fingerprint(case) for case in cases]
    if len(set(case_keys)) != len(cases):
        raise ValueError("Duplicate frozen evaluation cases")
    plan_cases = {(case["suite"], case["task_id"], case["task_episode_idx"],
                   case["initial_state_sha256"]): case for case in plan["evaluation_cases"]}
    if len(plan_cases) != len(cases) or set(plan_cases) != {
        (case["suite"], case["task_id"], case["task_episode_idx"], case["initial_state_sha256"])
        for case in cases
    }:
        raise ValueError("Executed evaluation cases differ from frozen prediction cases")
    for case in cases:
        original = plan_cases[(case["suite"], case["task_id"], case["task_episode_idx"],
                               case["initial_state_sha256"])]
        if "seed" in original and case.get("seed") != original["seed"]:
            raise ValueError("Execution seed differs from the prediction plan")
        if type(case.get("seed")) is not int or case["seed"] < 0:
            raise ValueError("Execution cases require explicit nonnegative seed")
    if experiment.get("total_rollouts") != len(cases) * len(conditions):
        raise ValueError("Frozen rollout count differs from condition/case product")
    normalized, receipt_hashes = {}, {}
    for cid, condition in by_condition.items():
        rows = completed[cid]
        if not isinstance(rows, list) or len(rows) != len(cases):
            raise ValueError(f"Condition {cid}: missing or extra case receipts")
        normalized[cid], receipt_hashes[cid] = {}, []
        for receipt in rows:
            _hash_checked(receipt, "receipt_hash")
            if (receipt.get("status") != "completed"
                    or receipt.get("execution_hash") != experiment["execution_hash"]
                    or receipt.get("plan_hash") != plan["plan_hash"]
                    or receipt.get("condition") != condition):
                raise ValueError("Receipt execution/plan/condition binding mismatch")
            case = receipt.get("case", {})
            key = fingerprint(case)
            if key not in case_keys or key in normalized[cid]:
                raise ValueError("Receipt has duplicate or unplanned evaluation case")
            result = receipt.get("result", {})
            episode, actions = result.get("episode", {}), result.get("actions")
            for name in ("suite", "task_id", "task_episode_idx", "initial_state_sha256", "seed"):
                if episode.get(name) != case.get(name):
                    raise ValueError(f"Receipt episode/case mismatch: {name}")
            if type(episode.get("success")) is not bool:
                raise ValueError("Episode success must be a boolean")
            if "caught_exception" not in episode or episode["caught_exception"] is not None:
                raise ValueError("Runtime exceptions cannot be treated as task failures")
            count = episode.get("num_actions")
            if type(count) is not int or count < 1 or not isinstance(actions, list) or len(actions) != count:
                raise ValueError("Action trace length does not match episode num_actions")
            if any(not isinstance(action, list) or len(action) != 7 or any(
                isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                for value in action) for action in actions):
                raise ValueError("Action trace must contain finite seven-dimensional actions")
            normalized[cid][key] = result
            receipt_hashes[cid].append(receipt["receipt_hash"])
    raw, identity = normalized[raw_ids[0]], normalized[identity_ids[0]]
    for key in case_keys:
        if (raw[key]["actions"] != identity[key]["actions"]
                or raw[key]["episode"]["success"] != identity[key]["episode"]["success"]
                or raw[key]["episode"]["num_actions"] != identity[key]["episode"]["num_actions"]):
            raise ValueError("Raw/identity exact action and outcome parity failed")
    return plan, cases, raw, feature_conditions, normalized, receipt_hashes


def analyze_results(experiment: Mapping[str, Any], completed: Mapping[str, Any], *,
                    bootstrap_replicates: int = 2000) -> dict[str, Any]:
    """Pure fixed-panel statistics for already disk-validated receipts.

    This helper is useful for synthetic CPU tests. Production callers must use
    analyze_experiment, which additionally verifies on-disk hook evidence.
    """
    count, seed, level = _analysis_parameters(experiment, bootstrap_replicates)
    plan, cases, raw, feature_conditions, conditions, receipt_hashes = _validated_outcomes(experiment, completed)
    panel = sorted(plan["panel_feature_ids"])
    effects, paired_rows = [], []
    for fid in panel:
        by_task, switches = defaultdict(list), Counter()
        edited = conditions[feature_conditions[fid]]
        for case in cases:
            key = fingerprint(case)
            before, after = int(raw[key]["episode"]["success"]), int(edited[key]["episode"]["success"])
            by_task[(case["suite"], case["task_id"])].append((before, after))
            switches[(before, after)] += 1
            paired_rows.append({"feature_id": fid, **{k: case[k] for k in (
                "suite", "task_id", "task_episode_idx", "initial_state_sha256", "seed")},
                "protocol_id": experiment["execution_hash"], "raw_success": bool(before),
                "edit_success": bool(after), "drop": before - after})
        task_effects = [{"suite": suite, "task_id": task, "num_pairs": len(pairs),
                         "sr_raw": _mean([a for a, _ in pairs]), "sr_edit": _mean([b for _, b in pairs]),
                         "drop": _mean([a - b for a, b in pairs])}
                        for (suite, task), pairs in sorted(by_task.items())]
        effects.append({"feature_id": fid, "num_pairs": len(cases), "num_tasks": len(by_task),
                        "sr_raw": _mean([row["sr_raw"] for row in task_effects]),
                        "sr_edit": _mean([row["sr_edit"] for row in task_effects]),
                        "drop": _mean([row["drop"] for row in task_effects]),
                        "n_success_to_failure": switches[(1, 0)], "n_failure_to_success": switches[(0, 1)],
                        "n_success_to_success": switches[(1, 1)], "n_failure_to_failure": switches[(0, 0)],
                        "task_effects": task_effects})
    draws = _bootstrap_drops({"paired_rows": paired_rows}, count, seed)
    effects_by_id = {row["feature_id"]: row for row in effects}
    for row in effects:
        row["drop_ci"] = _interval(draws[row["feature_id"]], level)
    eligible_scores = {row["feature_id"]: row for row in plan["eligible_feature_scores"]}
    comparisons, correlation_draws, topk_draws = [], {}, {}
    correlation_values, topk_values = {}, {}
    y = [effects_by_id[fid]["drop"] for fid in panel]
    for metric in METRICS:
        x = [eligible_scores[fid][metric] for fid in panel]
        value = spearman_correlation(x, y)
        correlations = [spearman_correlation(x, [draws[fid][index] for fid in panel]) for index in range(count)]
        correlation_draws[metric], correlation_values[metric] = correlations, value
        info = plan["top_k_availability"][metric]
        wanted = info["feature_ids"]
        missing = sorted(set(wanted) - set(panel))
        complete = len(wanted) == plan["top_k"] and not missing
        if complete:
            topk_values[metric] = _mean([effects_by_id[fid]["drop"] for fid in wanted])
            topk_draws[metric] = [_mean([draws[fid][index] for fid in wanted]) for index in range(count)]
            interval = _interval(topk_draws[metric], level)
        else:
            topk_values[metric] = None
            interval = {"low": None, "high": None, "status": "incomplete_coverage",
                        "valid_replicates": 0, "undefined_replicates": count}
        comparisons.append({"predictor": metric, "population_scope": "fixed_selected_feature_panel",
            "feature_ids": panel, "num_features": len(panel), "spearman": value,
            "spearman_status": "undefined" if value is None else "ok",
            "spearman_ci": _interval(correlations, level),
            "score_tie_fraction": 1 - len(set(x)) / len(x),
            "drop_tie_fraction": 1 - len(set(y)) / len(y),
            "top_k": plan["top_k"], "original_topk_feature_ids": wanted,
            "missing_topk_feature_ids": missing, "topk_coverage": (len(wanted) - len(missing)) / plan["top_k"],
            "topk_mean_drop": topk_values[metric], "topk_mean_drop_ci": interval,
            "topk_substitution_performed": False})
    primary = plan["primary_metric"]
    rank_contrasts, topk_contrasts = [], []
    for baseline in ("mean_head_kl", "event_max"):
        a, b = correlation_values[primary], correlation_values[baseline]
        paired = [None if first is None or second is None else first - second
                  for first, second in zip(correlation_draws[primary], correlation_draws[baseline])]
        rank_contrasts.append({"first": primary, "second": baseline,
                              "difference": None if a is None or b is None else a - b,
                              "ci": _interval(paired, level),
                              "interpretation": "selected_panel_spearman_difference_not_global_prediction_accuracy"})
        if primary in topk_draws and baseline in topk_draws:
            topk_contrasts.append({"first": primary, "second": baseline,
                "difference": topk_values[primary] - topk_values[baseline],
                "ci": _interval([a - b for a, b in zip(topk_draws[primary], topk_draws[baseline])], level)})
        else:
            topk_contrasts.append({"first": primary, "second": baseline, "difference": None,
                "ci": {"low": None, "high": None, "status": "incomplete_coverage"}})
    result = {
        "schema_version": "head_rank_behavior_analysis_v1", "status": "analyzed",
        "synthetic": bool(experiment.get("synthetic", False)),
        "execution_hash": experiment["execution_hash"], "plan_hash": plan["plan_hash"],
        "primary_metric": primary, "prediction_target": plan["prediction_target"],
        "primary_statistic": "spearman_on_fixed_selected_feature_panel",
        "primary_comparators": ["mean_head_kl", "event_max"],
        "feature_ids": panel, "task_ids": plan["task_ids"], "num_cases_per_feature": len(cases),
        "total_rollouts": experiment["total_rollouts"], "new_rollouts": 0,
        "split_evidence": dict(experiment.get("split_evidence", {})),
        "identity_validation": {"status": "passed", "num_pairs": len(cases),
                                "scope": "exact_stored_action_trace_and_boolean_success_not_success_rate_only"},
        "feature_effects": effects, "paired_rows": paired_rows,
        "predictor_comparison": comparisons, "spearman_contrasts": rank_contrasts,
        "topk_contrasts": topk_contrasts,
        "source_receipt_hashes": {key: sorted(values) for key, values in receipt_hashes.items()},
        "uncertainty": {"bootstrap_replicates": count, "bootstrap_seed": seed, "confidence_level": level,
                        "resampling": "one_shared_initial_state_resample_within_each_fixed_task_for_all_features",
                        "task_scope": "fixed_tasks", "feature_scope": "fixed_selected_panel",
                        "score_uncertainty_included": False, "simultaneous_intervals": False,
                        "multiple_comparisons": "pointwise_intervals; no_familywise_superiority_claim"},
        "scope": [
            "Behavioral importance is signed success-rate loss under this exact single-feature removal protocol",
            "Negative drop means improvement; harmful and beneficial changes are not collapsed into absolute effects",
            "Feature panel was selected by multiple rankings/disagreement/random sampling; it is not a representative dictionary census",
            "No all-feature recall, calibrated percentage-point prediction or held-out-task/model generalization is established",
            "Top-K effects and contrasts are undefined when any originally selected Top-K feature lacks outcomes",
            "Output sensitivity and success-rate loss do not establish semantic meaning or distinguish purposeful behavior from generic policy damage",
            "Intervals condition on this frozen panel and fixed tasks, do not guarantee power, and are not simultaneous across comparisons",
            "Failure to exclude zero is not equivalence; pilot outcomes used to change a method require new features for confirmation",
        ],
    }
    if "calibrated_predictions" in plan.get("source_bindings", {}):
        from event_sae.research.head_predict.evaluation import evaluate_prediction_attachment
        result["calibrated_prediction_comparison"] = evaluate_prediction_attachment(plan, effects)
    result["analysis_hash"] = fingerprint(result)
    return result


def _display(value):
    return "undefined" if value is None else f"{value:.4f}"


def render_report(result):
    split = result.get("split_evidence", {})
    lines = ["# HeadRank — frozen-panel behavioral validation", "",
             f"Execution: `{result['execution_hash']}`.",
             f"Prediction plan: `{result['plan_hash']}`.", "",
             f"Synthetic: {result['synthetic']}. Completed rollouts: {result['total_rollouts']}. "
             f"Features: {len(result['feature_ids'])}; paired cases per feature: {result['num_cases_per_feature']}.",
             "Analysis launches no model, simulator or new rollout.", "",
             f"Historical collection-state identity verified: {split.get('historical_state_identity_verified', 'unavailable')}. "
             f"Prior evaluation-label use: {split.get('evaluation_labels_previously_used', 'unavailable')}.",
             "These are new feature outcomes on fixed tasks/states, not a newly held-out-task study.", "",
             "Raw/identity verification: exact stored action traces and success outcomes match for every case.", "",
             "## Signed feature effects", "",
             "Drop = equally task-weighted mean(raw success − ablation success). Negative means improvement; 0.10 means 10 percentage points.", "",
             "| Feature | Raw SR | Ablated SR | Signed drop | 95% CI | Harm / repair cases |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for row in result["feature_effects"]:
        ci = row["drop_ci"]
        lines.append(f"| {row['feature_id']} | {_display(row['sr_raw'])} | {_display(row['sr_edit'])} | "
                     f"{_display(row['drop'])} | [{_display(ci['low'])}, {_display(ci['high'])}] | "
                     f"{row['n_success_to_failure']} / {row['n_failure_to_success']} |")
    lines += ["", "## Frozen predictor comparison", "",
              f"Primary metric: `{result['primary_metric']}`. Correlations refer only to the complete evaluated panel, not all 32,768 features.", "",
              "| Predictor | Panel Spearman | Original Top-K coverage | Top-K signed drop |",
              "| --- | ---: | ---: | ---: |"]
    for row in result["predictor_comparison"]:
        lines.append(f"| {row['predictor']} | {_display(row['spearman'])} | "
                     f"{_display(row['topk_coverage'])} | {_display(row['topk_mean_drop'])} |")
    lines += ["", "Missing Top-K effects stay undefined; lower-ranked measured features are never substituted.", "",
              "## Primary score versus baselines", "",
              "Positive differences favor the primary score in this panel. Intervals are pointwise, not multiplicity-adjusted.", "",
              "| Contrast | Spearman difference | 95% CI | Top-K drop difference | 95% CI |",
              "| --- | ---: | --- | ---: | --- |"]
    for rank, top in zip(result["spearman_contrasts"], result["topk_contrasts"]):
        a, b = rank["ci"], top["ci"]
        lines.append(f"| {rank['first']} − {rank['second']} | {_display(rank['difference'])} | "
                     f"[{_display(a['low'])}, {_display(a['high'])}] | {_display(top['difference'])} | "
                     f"[{_display(b['low'])}, {_display(b['high'])}] |")
    prediction = result.get("calibrated_prediction_comparison")
    if prediction is not None:
        lines += ["", "## Frozen calibrated predictor (secondary)", "",
                  f"Predictor: `{prediction['predictor']}`. Predictions were bound into the plan before execution.",
                  f"Mean absolute signed-drop error: {_display(prediction['mae'])}; "
                  f"panel Spearman: {_display(prediction['spearman'])}.",
                  "This is a secondary prediction test on the Tail-selected panel, not a dictionary-wide accuracy claim.", "",
                  "| Feature | Predicted drop | Observed drop | Error |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {row['feature_id']} | {_display(row['predicted_drop'])} | "
                  f"{_display(row['observed_drop'])} | {_display(row['error'])} |"
                  for row in prediction["predictions"]]
    lines += ["", "## Scope and limitations", ""]
    lines += [f"- {item}" for item in result["scope"]]
    return "\n".join(lines) + "\n"


def analyze_experiment(experiment_path, run_root, output, bootstrap_replicates=2000):
    """Validate complete disk receipts, then write new immutable CPU analysis."""
    from .execution import load_completed_results
    from .runtime import verify_experiment
    from .workflow import safe_output

    module_root = Path(__file__).resolve().parent
    implementation_paths = {
        "head_rank/analysis.py": module_root / "analysis.py",
        "head_rank/protocol.py": module_root / "protocol.py",
        "head_rank/execution.py": module_root / "execution.py",
        "head_rank/runtime.py": module_root / "runtime.py",
        "output_head/results.py": module_root.parent / "output_head/results.py",
        "head_predict/evaluation.py": module_root.parent / "head_predict/evaluation.py",
    }
    implementation = {name: sha256_file(path) for name, path in implementation_paths.items()}
    experiment = read_json(experiment_path)
    verify_experiment(experiment)
    protected = [experiment_path, run_root]
    protected.extend(item["path"] for item in experiment["sources"].values())
    protected.extend(experiment.get("protected_roots", []))
    target = safe_output(output, protected)
    completed = load_completed_results(experiment, run_root)
    result = analyze_results(experiment, completed, bootstrap_replicates=bootstrap_replicates)
    result["experiment_file_sha256"] = sha256_file(experiment_path)
    # An archived experiment may be reanalyzed with newer CPU tooling. Record
    # the actual analysis implementation instead of silently attributing it to
    # the old rollout-code identity; no current GPU/runtime is required here.
    if implementation != {name: sha256_file(path) for name, path in implementation_paths.items()}:
        raise ValueError("Analysis implementation changed while computing results")
    result["analysis_implementation"] = implementation
    result.pop("analysis_hash")
    result["analysis_hash"] = fingerprint(result)
    # Exclusive directory creation prevents a second analysis from overwriting
    # either outputs or inputs if two processes race after the guard.
    target.mkdir(parents=True, exist_ok=False)
    atomic_write_json(target / "analysis.json", result)
    atomic_write_text(target / "report.md", render_report(result))
    return {"status": "analyzed", "output": str(target), "analysis_hash": result["analysis_hash"],
            "feature_count": len(result["feature_ids"]), "total_rollouts": result["total_rollouts"],
            "new_rollouts": 0, "model_execution": False}
