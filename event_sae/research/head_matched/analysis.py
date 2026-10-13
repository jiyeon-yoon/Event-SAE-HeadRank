"""CPU-only analysis of complete norm-matched direction diagnostics.

Five directions from one already evaluated anchor are not five unseen feature
ablations. All uncertainty is conditional on this fixed, retrospective design.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

from event_sae.research.output_head.provenance import (
    atomic_write_json, atomic_write_text, fingerprint, sha256_file,
)
from .protocol import ANALYSIS, SCOPE, signed


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _interval(values):
    import numpy as np
    data = np.asarray(values, dtype=float)
    valid = data[np.isfinite(data)]
    result = {"low": None, "high": None, "confidence_level": .95,
              "valid_replicates": len(valid), "undefined_replicates": len(data) - len(valid)}
    if len(valid) != len(data):
        result["status"] = "undefined_replicates_no_unconditional_interval"
    elif not len(valid):
        result["status"] = "undefined"
    else:
        result.update(low=float(np.quantile(valid, .025)), high=float(np.quantile(valid, .975)),
                      status="degenerate" if np.ptp(valid) == 0 else "ok")
    return result


def _spearman_draws(x, integer_counts):
    import numpy as np
    from scipy.stats import rankdata
    xr = rankdata(np.asarray(x), method="average")
    yr = rankdata(np.asarray(integer_counts), method="average", axis=-1)
    xr -= xr.mean()
    yr -= yr.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((xr * xr).sum() * (yr * yr).sum(axis=-1))
    values = np.full(len(yr), np.nan)
    np.divide((yr * xr).sum(axis=-1), denominator, out=values, where=denominator > 0)
    return values


def _validated(experiment, completed):
    signed(experiment, "execution_hash")
    if experiment.get("analysis") != ANALYSIS:
        raise ValueError("Matched analysis settings must equal the frozen protocol")
    condition_rows = experiment["conditions"]
    conditions = {c["condition_id"]: c for c in condition_rows}
    modes = Counter(c["mode"] for c in condition_rows)
    if len(conditions) != 7 or modes != {"raw": 1, "identity": 1, "anchor": 1, "matched_decoder": 4}:
        raise ValueError("Require raw, identity, anchor and exactly four matched decoder directions")
    if set(completed) != set(conditions):
        raise ValueError("Missing or extra result conditions")
    predictions = experiment.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 5:
        raise ValueError("Five current-runtime frozen direction predictions are required")
    directions = [c["condition_id"] for c in condition_rows if c["mode"] in ("anchor", "matched_decoder")]
    anchor = next(c["condition_id"] for c in condition_rows if c["mode"] == "anchor")
    predictions_by_id = {r["condition_id"]: r for r in predictions}
    if len(predictions_by_id) != 5 or set(predictions_by_id) != set(directions):
        raise ValueError("Prediction conditions differ from evaluated directions")
    cases = {fingerprint(c): c for c in experiment["cases"]}
    if not cases or len(cases) != len(experiment["cases"]):
        raise ValueError("Empty or duplicate cases")
    task_counts = Counter(c["task_id"] for c in cases.values())
    if set(task_counts) != set(range(10)) or len(set(task_counts.values())) != 1:
        raise ValueError("Matched protocol requires ten fixed tasks with equal paired case counts")
    if experiment.get("total_rollouts") != len(cases) * 7:
        raise ValueError("Rollout budget does not match paired conditions and cases")
    for name in directions:
        prediction = predictions_by_id[name]
        values = prediction.get("task_mean_head_kl")
        if (not _number(prediction.get("mean_head_kl")) or prediction["mean_head_kl"] < 0
                or not isinstance(values, dict) or set(values) != {str(t) for t in task_counts}
                or not all(_number(v) and v >= 0 for v in values.values())
                or not math.isclose(prediction["mean_head_kl"], math.fsum(values.values()) / 10,
                                    rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError("Invalid or inconsistent equal-task current prediction")
    normalized = {}
    hashes = {}
    for cid, condition in conditions.items():
        rows = completed[cid]
        if not isinstance(rows, list) or len(rows) != len(cases):
            raise ValueError("Incomplete results cannot become task failures")
        normalized[cid], hashes[cid] = {}, []
        for row in rows:
            signed(row, "receipt_hash")
            key = fingerprint(row["case"])
            if (key not in cases or key in normalized[cid] or row.get("condition") != condition
                    or row.get("execution_hash") != experiment["execution_hash"]
                    or row.get("plan_hash") != experiment["plan_hash"] or row.get("status") != "completed"):
                raise ValueError("Receipt is duplicated, incomplete or bound to another experiment")
            result = row["result"]
            ep = result["episode"]
            if any(ep.get(k) != cases[key][k] for k in ("suite", "task_id", "task_episode_idx", "seed", "initial_state_sha256")):
                raise ValueError("Episode/case mismatch")
            if type(ep.get("success")) is not bool or "caught_exception" not in ep or ep["caught_exception"] is not None:
                raise ValueError("Invalid outcome or runtime error")
            count = ep.get("num_actions")
            actions = result.get("actions")
            if (type(count) is not int or count < 1 or not isinstance(actions, list) or len(actions) != count
                    or any(not isinstance(a, list) or len(a) != 7 or not all(_number(v) for v in a) for a in actions)):
                raise ValueError("Incomplete/nonfinite seven-dimensional action trace")
            normalized[cid][key] = result
            hashes[cid].append(row["receipt_hash"])
    raw_id = next(c["condition_id"] for c in condition_rows if c["mode"] == "raw")
    identity_id = next(c["condition_id"] for c in condition_rows if c["mode"] == "identity")
    for key in cases:
        raw, identity = normalized[raw_id][key], normalized[identity_id][key]
        if raw["actions"] != identity["actions"] or raw["episode"]["success"] != identity["episode"]["success"]:
            raise ValueError("Raw/identity exact action and outcome parity failed")
    return directions, anchor, predictions_by_id, cases, normalized, raw_id, hashes


def analyze_results(experiment, completed):
    """Pure testable statistics; production must first validate journal evidence."""
    import numpy as np

    directions, anchor, predictions, cases, outcomes, raw_id, receipt_hashes = _validated(experiment, completed)
    task_ids = list(range(10))
    grouped = {task: sorted(k for k, c in cases.items() if c["task_id"] == task) for task in task_ids}
    n = len(grouped[0])
    raw = np.array([[int(outcomes[raw_id][k]["episode"]["success"]) for k in grouped[t]] for t in task_ids])
    edited = np.array([[[int(outcomes[name][k]["episode"]["success"]) for name in directions]
                       for k in grouped[t]] for t in task_ids])
    deltas = raw[:, :, None] - edited
    reps, seed = ANALYSIS["bootstrap_replicates"], ANALYSIS["bootstrap_seed"]
    counts = np.empty((reps + 1, 5), dtype=np.int64)
    task_draws = np.empty((reps + 1, 10, 5), dtype=np.int64)
    task_draws[0] = deltas.sum(axis=1)
    counts[0] = task_draws[0].sum(axis=0)
    rng = random.Random(seed)
    for bi in range(1, reps + 1):
        for ti in task_ids:
            indices = [rng.randrange(n) for _ in range(n)]
            task_draws[bi, ti] = deltas[ti, indices].sum(axis=0)
        counts[bi] = task_draws[bi].sum(axis=0)
    drops = counts / (10 * n)
    kl = [predictions[name]["mean_head_kl"] for name in directions]
    correlations = _spearman_draws(kl, counts)
    anchor_index = directions.index(anchor)
    control_indices = [i for i in range(5) if i != anchor_index]
    # Integer numerator avoids introducing false signed-effect ties/cancellation.
    contrast = (4 * counts[:, anchor_index] - counts[:, control_indices].sum(axis=1)) / (40 * n)
    effects = []
    paired = []
    for di, name in enumerate(directions):
        task_effects = []
        for ti in task_ids:
            task_effects.append({"task_id": ti, "num_pairs": n, "raw_successes": int(raw[ti].sum()),
                "edited_successes": int(edited[ti, :, di].sum()),
                "net_harm_cases": int(task_draws[0, ti, di]), "signed_drop": float(task_draws[0, ti, di] / n),
                "harm_cases": int((deltas[ti, :, di] == 1).sum()), "repair_cases": int((deltas[ti, :, di] == -1).sum())})
            for ci, key in enumerate(grouped[ti]):
                paired.append({"condition_id": name, **cases[key], "raw_success": bool(raw[ti, ci]),
                    "edit_success": bool(edited[ti, ci, di]), "drop": int(deltas[ti, ci, di])})
        effects.append({"condition_id": name, "mode": "anchor" if name == anchor else "matched_decoder",
            "num_pairs": 10 * n, "raw_success_rate": float(raw.sum() / (10 * n)),
            "edited_success_rate": float(edited[:, :, di].sum() / (10 * n)),
            "net_harm_cases": int(counts[0, di]), "signed_drop": float(drops[0, di]),
            "drop_ci": _interval(drops[1:, di]), "harm_cases": int((deltas[:, :, di] == 1).sum()),
            "repair_cases": int((deltas[:, :, di] == -1).sum()), "task_effects": task_effects,
            "frozen_mean_head_kl": predictions[name]["mean_head_kl"]})
    per_task_rank = []
    for ti in task_ids:
        values = _spearman_draws([predictions[name]["task_mean_head_kl"][str(ti)] for name in directions], task_draws[:, ti])
        per_task_rank.append({"task_id": ti, "spearman": float(values[0]) if math.isfinite(values[0]) else None,
                              "ci": _interval(values[1:]), "status": "descriptive_secondary"})
    return {"schema_version": "head_matched_analysis_v1", "status": "analyzed", "synthetic": False,
        "execution_hash": experiment["execution_hash"], "plan_hash": experiment["plan_hash"],
        "scientific_scope": SCOPE, "new_rollouts": 0, "model_execution": False,
        "total_rollouts": experiment["total_rollouts"], "num_directions": 5,
        "cases_per_task": n, "paired_cases_per_direction": 10 * n, "task_ids": task_ids,
        "purpose": "smoke_diagnostic_no_power_claim" if n < 20 else "retrospective_direction_diagnostic_no_power_guarantee",
        "identity_validation": {"status": "passed", "num_pairs": 10 * n, "exact_actions_and_success": True},
        "frozen_predictions": [predictions[name] for name in directions], "direction_effects": effects,
        "paired_rows": paired,
        "primary": {"statistic": "spearman_five_frozen_current_direction_KLs_vs_signed_success_loss",
            "spearman": float(correlations[0]) if math.isfinite(correlations[0]) else None,
            "ci": _interval(correlations[1:]), "population_size": 5,
            "outcome_ties": "exact_integer_net_harm_counts_equivalent_to_equal_task_means"},
        "secondary": {"statistic": "anchor_signed_loss_minus_mean_four_control_losses",
            "difference": float(contrast[0]), "ci": _interval(contrast[1:]),
            "positive_means": "anchor_caused_more_net_success_loss_than_average_control"},
        "per_task_direction_correlation": per_task_rank,
        "source_receipt_hashes": {key: sorted(value) for key, value in receipt_hashes.items()},
        "uncertainty": {**ANALYSIS,
            "resampling": "case_resample_shared_across_all_five_directions_within_each_fixed_task",
            "rng_draw_order": "replicate_then_sorted_task_then_fingerprint_sorted_case_count",
            "fixed_tasks": True, "fixed_directions": True, "score_uncertainty_included": False,
            "simultaneous_intervals": False,
            "undefined_policy": "constant_outcomes_or_scores_have_undefined_rho; do_not_discard_undefined_draws_for_CI"},
        "limitations": [
            "This is a retrospective single-anchor/four-direction diagnostic, not unseen-feature predictive validation.",
            "At two cases per task there are 140 rollouts; this is smoke-scale evidence without established statistical power.",
            "Decoder controls use the anchor support and magnitude; they are not removals of those control features.",
            "Norm matching holds per row on the same incoming hidden state within tolerance, not across diverging trajectories.",
            "Different trajectories may have different activation frequencies, episode lengths and cumulative edit budgets.",
            "Only five preselected directions are correlated; no all-feature, held-out-task/model or semantic-importance inference.",
            "Full-vocabulary fixed-prefix KL is not physical action error and is unsigned, whereas success loss is signed.",
            "Bootstrap conditions on fixed tasks, states, directions and scores; intervals are pointwise, not simultaneous.",
            "Zero observed effect or degenerate interval does not establish equivalence or population-level absence of effect.",
            "Old outcomes are not reused; failures, missing results and invalid norm rows cannot count as task failures."]}


def _norm_summaries(completed, root):
    """Streaming summaries of evidence already checked by journal.load_results."""
    root = Path(root).resolve()
    summaries = {}
    for cid, receipts in completed.items():
        if receipts[0]["condition"]["mode"] == "raw":
            summaries[cid] = {"hook_applied": False}
            continue
        counts = {"num_rows": 0, "active_rows": 0, "edited_rows": 0, "norm_invalid_rows": 0,
                  "off_support_rows": 0, "num_forwards": 0}
        totals = {key: 0. for key in ("intended_norm", "anchor_realized_norm", "control_realized_norm")}
        maxima = {key: 0. for key in ("pair_norm_abs_error", "pair_norm_rel_error", "anchor_cast_abs_error", "control_cast_abs_error")}
        evidence_hashes = []
        for receipt in receipts:
            metrics = receipt["result"]["hook_metrics"]
            for key in counts:
                counts[key] += metrics[key]
            for key in maxima:
                maxima[key] = max(maxima[key], metrics["max_" + key])
            evidence = receipt["result"]["hook_evidence"]
            directory = root / "attempts" / cid / fingerprint(receipt["case"]) / f"attempt-{receipt['attempt_index']:06d}"
            path = (directory / evidence["path"]).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256_file(path) != evidence["sha256"]:
                raise ValueError("Hook evidence changed or escaped attempt after journal validation")
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        for key in totals:
                            totals[key] += math.fsum(row[key])
            if sha256_file(path) != evidence["sha256"]:
                raise ValueError("Hook evidence changed while summarizing norms")
            evidence_hashes.append(evidence["sha256"])
        summaries[cid] = {"hook_applied": True, **counts,
            "maxima": maxima, "sum_row_norms": totals,
            "mean_norm_over_all_hook_rows": {key: value / counts["num_rows"] for key, value in totals.items()},
            "norm_policy": receipts[0]["condition"]["norm_policy"],
            "source_evidence_hashes": sorted(evidence_hashes),
            "trajectory_edit_budgets_matched": False,
            "sum_is_descriptive_not_cross_trajectory_matching_certificate": True}
    return summaries


def render_report(result):
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"
    def ci(value):
        return f"[{fmt(value['low'])}, {fmt(value['high'])}] ({value['status']})"
    lines = ["# Matched-direction diagnostic", "",
        f"Completed rollouts: {result['total_rollouts']}. Directions: 5. Paired cases per direction: {result['paired_cases_per_direction']}.",
        "New model/rollout execution during analysis: 0. Raw/identity exact traces and outcomes: passed.",
        "Retrospective smoke/diagnostic evidence, not new-feature predictive validation.", "",
        "| Direction | Frozen KL | Raw SR | Edited SR | Signed drop | 95% CI | Harm / repair |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
    for row in result["direction_effects"]:
        lines.append(f"| {row['condition_id']} | {row['frozen_mean_head_kl']:.6g} | {row['raw_success_rate']:.4f} | {row['edited_success_rate']:.4f} | {row['signed_drop']:.4f} | {ci(row['drop_ci'])} | {row['harm_cases']} / {row['repair_cases']} |")
    lines += ["", f"Primary direction Spearman: {fmt(result['primary']['spearman'])}; 95% CI {ci(result['primary']['ci'])}.",
        f"Anchor minus four controls' mean signed loss: {fmt(result['secondary']['difference'])}; 95% CI {ci(result['secondary']['ci'])}.",
        "", "Recorded norm summaries and task-level effects are in analysis.json.",
        "Per-row matching does not certify equal cumulative perturbation on diverging trajectories.",
        "", "## Limitations", ""] + [f"- {line}" for line in result["limitations"]]
    return "\n".join(lines) + "\n"


def analyze(experiment, root, output):
    """Read complete signed journal; publish immutable analysis without GPU calls."""
    from .journal import load_results
    from event_sae.research.head_rank.workflow import safe_output

    protected = [root] + [row["path"] for row in experiment.get("sources", {}).values() if isinstance(row, dict) and "path" in row]
    target = safe_output(output, protected)
    completed = load_results(experiment, root)
    result = analyze_results(experiment, completed)
    result["observed_norms"] = _norm_summaries(completed, root)
    module_root = Path(__file__).parent
    result["analysis_implementation"] = {name: sha256_file(module_root / name) for name in ("analysis.py", "journal.py", "protocol.py")}
    result["source_integrity"] = "journal_receipts_actions_hook_norm_evidence_and_raw_identity_validated"
    result["analysis_hash"] = fingerprint(result)
    target.mkdir(parents=True, exist_ok=False)
    atomic_write_json(target / "analysis.json", result)
    atomic_write_text(target / "report.md", render_report(result))
    return {"status": "analyzed", "output": str(target), "analysis_hash": result["analysis_hash"],
            "total_rollouts": result["total_rollouts"], "directions": 5, "new_rollouts": 0,
            "model_execution": False, "scientific_scope": SCOPE}
