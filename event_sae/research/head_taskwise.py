"""Retrospective CPU-only, fixed-panel task-conditioned ranking diagnostics.

No model, simulator, download, intervention or score fitting is performed.
Task conditioning is selected after outcomes were inspected, so this is not a
new confirmatory evaluation. Integer within-task net counts preserve ties.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random

from event_sae.research.output_head.provenance import (
    atomic_write_json, atomic_write_text, fingerprint, read_json, sha256_file,
)

METRICS = ("mean_head_kl", "tail_head_kl", "event_max")
SCOPE_KEYS = ("task_ids", "dictionary_size", "discovery_manifest_hash",
              "split_manifest_hash", "sae_sha256")
CASE_KEYS = ("suite", "task_id", "task_episode_idx", "initial_state_sha256", "seed")


def _checked_hash(document, name):
    if document.get(name) != fingerprint({k: v for k, v in document.items() if k != name}):
        raise ValueError(f"Missing or mismatched {name}")


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _case_key(row):
    try:
        return tuple(row[key] for key in CASE_KEYS)
    except KeyError as exc:
        raise ValueError(f"Incomplete paired case: {exc}") from exc


def event_task_vectors(rows, matrix, task_ids, dictionary_size):
    """Same canonical coverage rule as frozen Event-max, partitioned by task.

    Matrix conversion is CPU-only. No semantic phase labels are inferred from
    event clusters. This pure helper is independently testable without torch.
    """
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float32)
    if (not isinstance(rows, list) or matrix.shape != (len(rows), dictionary_size)
            or not np.isfinite(matrix).all() or not rows):
        raise ValueError("Malformed Event row matrix")
    if set(row.get("task_id") for row in rows) != set(task_ids):
        raise ValueError("Event row task scope mismatch")
    keep = []
    for index, row in enumerate(rows):
        coverage = row.get("episode_coverage")
        if not _finite(coverage) or not 0 <= coverage <= 1:
            raise ValueError("Event episode coverage must be explicit and finite")
        if coverage >= .5:
            keep.append(index)
    by_task = {task: [i for i in keep if rows[i]["task_id"] == task] for task in task_ids}
    if any(not indices for indices in by_task.values()):
        raise ValueError("Every task needs canonical Event rows at coverage >= 0.5")
    return (matrix[keep].max(axis=0),
            {task: matrix[indices].max(axis=0) for task, indices in by_task.items()},
            {str(task): len(indices) for task, indices in by_task.items()})


def load_events(path, scores):
    # Reuse the historical bound loader for SHA, discovery signature and scope
    # checks. It lazily imports torch and only reads tensors on CPU.
    import torch
    from event_sae.research.head_rank.workflow import load_event_max

    global_vector, binding = load_event_max(path, scores["scope"])
    if binding != scores.get("event_max_binding"):
        raise ValueError("Event binding differs from frozen HeadRank score binding")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    global_replay, tasks, counts = event_task_vectors(
        payload["row_keys"], payload["matrix_raw"].float().numpy(),
        scores["scope"]["task_ids"], scores["scope"]["dictionary_size"])
    if global_replay.tolist() != global_vector:
        raise ValueError("Global Event-max aggregation replay mismatch")
    return global_vector, tasks, {"binding": binding, "canonical_rows_per_task": counts,
        "semantic_phase_labels_used": False,
        "aggregation": "within_task_max_over_same_canonical_event_rows"}


def validate_documents(scores, analysis, experiment):
    """Check intact bindings and exact integer paired outcomes, not runtime parity."""
    from event_sae.research.head_rank.protocol import verify_plan
    from event_sae.research.output_head.candidates import validate_score_scope

    for document, key in ((scores, "score_hash"), (analysis, "analysis_hash"),
                          (experiment, "execution_hash")):
        _checked_hash(document, key)
        if document.get("synthetic") is not False:
            raise ValueError("This workflow requires explicitly real artifacts")
    if scores.get("sampling_mode") != "full" or scores.get("development_only") is not False:
        raise ValueError("Full, non-development scoring is required")
    plan = experiment["prediction_plan"]
    verify_plan(plan)
    if (analysis.get("execution_hash") != experiment["execution_hash"]
            or analysis.get("plan_hash") != plan["plan_hash"]
            or experiment.get("plan_hash") != plan["plan_hash"]):
        raise ValueError("Analysis/execution/plan binding mismatch")
    bindings = plan["source_bindings"]
    if bindings.get("score_hash") != scores["score_hash"]:
        raise ValueError("Prediction plan binds a different score artifact")
    for key in SCOPE_KEYS:
        if scores.get("scope", {}).get(key) is None:
            raise ValueError(f"Incomplete score scope: {key}")
    validate_score_scope(scores["scope"], bindings["score_scope"])
    if bindings.get("event_max_binding") != scores.get("event_max_binding"):
        raise ValueError("Plan/Event binding mismatch")
    if scores["scope"]["sae_sha256"] != experiment["sources"]["sae"]["sha256"]:
        raise ValueError("SAE identity mismatch")
    panel, task_ids = sorted(plan["panel_feature_ids"]), sorted(plan["task_ids"])
    if analysis.get("feature_ids") != panel or sorted(analysis.get("task_ids", [])) != task_ids:
        raise ValueError("Analysis panel/task scope mismatch")
    if sorted(scores["scope"]["task_ids"]) != task_ids:
        raise ValueError("Task scope differs across scores and behavior")
    cases = {_case_key(row): row for row in experiment["cases"]}
    if len(cases) != len(experiment["cases"]) or {k[1] for k in cases} != set(task_ids):
        raise ValueError("Duplicate cases or missing task")
    if (analysis.get("status") != "analyzed" or analysis.get("num_cases_per_feature") != len(cases)
            or analysis.get("total_rollouts") != experiment.get("total_rollouts")
            or experiment.get("total_rollouts") != len(cases) * (len(panel) + 2)
            or analysis.get("identity_validation", {}).get("status") != "passed"
            or analysis["identity_validation"].get("num_pairs") != len(cases)):
        raise ValueError("Incomplete archived analysis or missing raw/identity verification")
    rows = {fid: {} for fid in panel}
    shared_raw = {}
    for row in analysis["paired_rows"]:
        fid, case = row.get("feature_id"), _case_key(row)
        if fid not in rows or case not in cases or case in rows[fid]:
            raise ValueError("Extra, duplicated or unplanned paired observation")
        if (type(row.get("raw_success")) is not bool or type(row.get("edit_success")) is not bool
                or row.get("protocol_id") != experiment["execution_hash"]):
            raise ValueError("Malformed or wrong-protocol outcome")
        delta = int(row["raw_success"]) - int(row["edit_success"])
        if type(row.get("drop")) is not int or row["drop"] != delta:
            raise ValueError("Paired delta is not exact success-count difference")
        if case in shared_raw and shared_raw[case] != row["raw_success"]:
            raise ValueError("Raw outcomes differ across paired features")
        shared_raw[case] = row["raw_success"]
        rows[fid][case] = row
    effects = {row["feature_id"]: row for row in analysis["feature_effects"]}
    if set(effects) != set(panel) or len(effects) != len(analysis["feature_effects"]):
        raise ValueError("Missing/duplicate archived feature effects")
    for fid in panel:
        if set(rows[fid]) != set(cases):
            raise ValueError("Incomplete paired feature observations")
        switches = Counter((r["raw_success"], r["edit_success"]) for r in rows[fid].values())
        if (effects[fid]["n_success_to_failure"] != switches[(True, False)]
                or effects[fid]["n_failure_to_success"] != switches[(False, True)]):
            raise ValueError("Paired outcomes disagree with archived summary counts")
        task_effects = {row["task_id"]: row for row in effects[fid]["task_effects"]}
        if len(task_effects) != len(effects[fid]["task_effects"]) or set(task_effects) != set(task_ids):
            raise ValueError("Missing/duplicate archived task effects")
        for task in task_ids:
            observations = [row for case, row in rows[fid].items() if case[1] == task]
            effect = task_effects[task]
            if (effect["num_pairs"] != len(observations)
                    or not math.isclose(effect["drop"], sum(row["drop"] for row in observations)
                                        / len(observations), rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Paired outcomes disagree with archived task effects")
    return panel, task_ids, rows


def score_matrices(scores, panel, tasks, event_global, event_tasks):
    import numpy as np

    by_id = {row["feature_id"]: row for row in scores["feature_scores"]}
    if len(by_id) != len(scores["feature_scores"]) or set(panel) - set(by_id):
        raise ValueError("Missing/duplicate feature scores")
    matrix = np.empty((6, len(tasks), len(panel)), dtype=np.float64)
    names = [f"{scope}_{metric}" for scope in ("global", "taskwise") for metric in METRICS]
    for fi, fid in enumerate(panel):
        row = by_id[fid]
        task_rows = {v["task_id"]: v for v in row["task_scores"]}
        if len(task_rows) != len(row["task_scores"]) or set(task_rows) != set(tasks):
            raise ValueError("Missing/duplicate task-conditioned score")
        if row["event_max"] != event_global[fid]:
            raise ValueError("Frozen feature Event-max differs from bound Event matrix")
        for ti, task in enumerate(tasks):
            values = [row[m] for m in METRICS] + [task_rows[task]["mean"],
                task_rows[task]["tail"], float(event_tasks[task][fid])]
            if not all(_finite(v) for v in values):
                raise ValueError("Nonfinite predictor score")
            matrix[:, ti, fi] = values
    return names, matrix


def rank_correlations(x_scores, y_counts):
    """Batched exact tie ranks; x [metric,task,feature], y [draw,task,feature]."""
    import numpy as np
    from scipy.stats import rankdata

    xrank = rankdata(np.asarray(x_scores), axis=-1, method="average")
    yrank = rankdata(np.asarray(y_counts), axis=-1, method="average")
    xrank -= xrank.mean(axis=-1, keepdims=True)
    yrank -= yrank.mean(axis=-1, keepdims=True)
    numerator = np.einsum("mtf,btf->bmt", xrank, yrank)
    denominator = np.sqrt(np.sum(xrank * xrank, axis=-1)[None, :, :]
                          * np.sum(yrank * yrank, axis=-1)[:, None, :])
    out = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _interval(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    result = {"valid_replicates": len(valid), "undefined_replicates": len(values) - len(valid),
              "low": None, "high": None, "confidence_level": .95}
    # Do not condition a macro statistic silently on favorable nonconstant
    # tasks/replicates. If any full-task statistic is undefined, no CI is issued.
    if len(valid) != len(values):
        result["status"] = "undefined_replicates_no_interval"
    elif not len(valid):
        result["status"] = "no_replicates"
    else:
        result.update(low=float(np.quantile(valid, .025)), high=float(np.quantile(valid, .975)),
                      status="degenerate" if np.ptp(valid) == 0 else "ok")
    return result


def _number(value):
    return float(value) if math.isfinite(value) else None


def compute_taskwise(panel, tasks, paired_rows, predictor_names, predictor_scores, *,
                     replicates=2000, seed=1729):
    import numpy as np

    if type(replicates) is not int or replicates < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive bootstrap replicate count and nonnegative integer seed required")
    rng = random.Random(seed)
    counts = np.empty((replicates + 1, len(tasks), len(panel)), dtype=np.int64)
    task_outcomes = []
    for ti, task in enumerate(tasks):
        keys = sorted(k for k in paired_rows[panel[0]] if k[1] == task)
        if not keys:
            raise ValueError("Task has no paired cases")
        deltas = np.array([[paired_rows[fid][key]["drop"] for fid in panel] for key in keys], dtype=np.int64)
        counts[0, ti] = deltas.sum(axis=0)
        task_outcomes.append({"task_id": task, "num_cases": len(keys), "features": [
            {"feature_id": fid, "net_harm_cases": int(counts[0, ti, fi]),
             "signed_drop": int(counts[0, ti, fi]) / len(keys),
             "harm_cases": int((deltas[:, fi] == 1).sum()),
             "repair_cases": int((deltas[:, fi] == -1).sum())}
            for fi, fid in enumerate(panel)]})
        for bi in range(replicates):
            indices = [rng.randrange(len(keys)) for _ in keys]
            counts[bi + 1, ti] = deltas[indices].sum(axis=0)
    correlations = rank_correlations(predictor_scores, counts)
    # np.mean (not nanmean) fixes the denominator to all prespecified tasks.
    macro = correlations.mean(axis=-1)
    comparisons = []
    for mi, name in enumerate(predictor_names):
        comparisons.append({"predictor": name, "macro_spearman": _number(macro[0, mi]),
            "macro_ci": _interval(macro[1:, mi]),
            "per_task": [{"task_id": task, "spearman": _number(correlations[0, mi, ti]),
                          "ci": _interval(correlations[1:, mi, ti])}
                         for ti, task in enumerate(tasks)]})
    pairs = [(3 + i, i) for i in range(3)] + [(3, 5), (4, 5), (4, 3)]
    contrasts = []
    for first, second in pairs:
        values = macro[:, first] - macro[:, second]
        contrasts.append({"first": predictor_names[first], "second": predictor_names[second],
                          "difference": _number(values[0]), "ci": _interval(values[1:])})
    return {"comparisons": comparisons, "contrasts": contrasts, "task_outcomes": task_outcomes,
        "uncertainty": {"bootstrap_replicates": replicates, "seed": seed,
            "resampling": "shared_case_indices_within_fixed_task_across_all_features_and_predictors",
            "rng_draw_order": "task_then_replicate_then_sorted_case_count_python_random_randrange",
            "macro_denominator": len(tasks),
            "undefined_policy": "any_undefined_task_makes_macro_undefined_no_nanmean; any_undefined_draw_withholds_ci",
            "outcome_ties": "exact_integer_net_harm_counts_within_task",
            "fixed_tasks": True, "fixed_feature_panel": True, "score_uncertainty_included": False,
            "simultaneous_intervals": False}}


def render_report(result):
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"
    def ci(row):
        return f"[{fmt(row['low'])}, {fmt(row['high'])}] ({row['status']})"
    lines = ["# Task-conditioned HeadRank reanalysis", "",
        "Retrospective diagnostic only. New rollouts: 0. GPU/model execution: False.", "",
        "Target: signed success-rate loss, not absolute behavior change.",
        "Statistic: equal-task mean of within-task Spearman over the same fixed feature panel.",
        "This is NOT pooled correlation over feature-task cells, and differs from suite-level rho.", "",
        "| Predictor | Macro Spearman | Pointwise 95% CI |",
        "| --- | ---: | --- |"]
    for row in result["comparisons"]:
        lines.append(f"| {row['predictor']} | {fmt(row['macro_spearman'])} | {ci(row['macro_ci'])} |")
    lines += ["", "| Contrast | Difference | Pointwise 95% CI |", "| --- | ---: | --- |"]
    for row in result["contrasts"]:
        lines.append(f"| {row['first']} - {row['second']} | {fmt(row['difference'])} | {ci(row['ci'])} |")
    lines += ["", "## Limitations", ""] + [f"- {line}" for line in result["limitations"]]
    return "\n".join(lines) + "\n"


def analyze_taskwise(*, scores_path, analysis_path, experiment_path, event_path, output,
                     replicates=2000, seed=1729):
    from event_sae.research.head_rank.workflow import safe_output

    inputs = {"scores": Path(scores_path), "analysis": Path(analysis_path),
              "experiment": Path(experiment_path), "events": Path(event_path)}
    target = safe_output(output, list(inputs.values()))
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    scores, analysis, experiment = [read_json(inputs[k]) for k in ("scores", "analysis", "experiment")]
    if (hashes["scores"] != experiment["sources"]["scores"]["sha256"]
            or hashes["scores"] != experiment["prediction_plan"]["source_bindings"]["scores_sha256"]
            or hashes["experiment"] != analysis["experiment_file_sha256"]):
        raise ValueError("Input file SHA differs from frozen source bindings")
    panel, tasks, paired = validate_documents(scores, analysis, experiment)
    global_event, task_event, event_info = load_events(inputs["events"], scores)
    names, matrix = score_matrices(scores, panel, tasks, global_event, task_event)
    result = compute_taskwise(panel, tasks, paired, names, matrix, replicates=replicates, seed=seed)
    result.update(schema_version="head_taskwise_retrospective_v1", status="analyzed", synthetic=False,
        new_rollouts=0, model_execution=False, gpu_required=False, development_only=True,
        task_ids=tasks, feature_ids=panel, source_sha256=hashes,
        score_hash=scores["score_hash"], source_analysis_hash=analysis["analysis_hash"],
        execution_hash=experiment["execution_hash"], score_scope=scores["scope"], event_info=event_info,
        source_integrity="signed_archived_analysis_and_exact_paired_counts_checked_no_simulator_rerun",
        implementation_sha256=sha256_file(__file__),
        predictor_values={name: matrix[mi].tolist() for mi, name in enumerate(names)},
        limitations=[
            "Task conditioning was chosen after inspecting outcomes; retrospective, not confirmatory.",
            "The same globally selected 16-feature panel is used, not a representative feature census.",
            "No feature-task independence assumption: cases are paired and resampled together within each fixed task.",
            "Intervals condition on fixed tasks, frozen features and scores; score/discovery uncertainty is not included.",
            "Pointwise intervals are not adjusted across the multiple exploratory comparisons.",
            "Canonical Event clusters are unlabeled; task-conditioned Event-max is not stage-specific semantic analysis.",
            "Head uses eight sampled times per discovery episode; Event uses all retained discovery timesteps.",
            "Inherited full500 clusters and pretrained SAE are not discovery-only fitted.",
            "No new-feature, held-out-task/model generalization or calibrated percentage-point prediction is established.",
            "Only this execution's outcomes are used; earlier execution outcomes are not pooled.",
            "Zero-net loss need not mean no behavioral effect; harms and repairs may cancel."])
    if hashes != {name: sha256_file(path) for name, path in inputs.items()}:
        raise ValueError("Inputs changed during analysis")
    result["analysis_hash"] = fingerprint(result)
    target.mkdir(parents=True, exist_ok=False)
    atomic_write_json(target / "taskwise-analysis.json", result)
    atomic_write_text(target / "report.md", render_report(result))
    return {"status": "analyzed", "output": str(target), "analysis_hash": result["analysis_hash"],
            "features": len(panel), "tasks": len(tasks), "new_rollouts": 0,
            "model_execution": False, "gpu_required": False, "development_only": True}
