"""Small, CPU-only signed-drop calibration with feature-grouped validation.

These are development diagnostics, not independent validation of a new method.
Outcome labels are a separate input: only the explicit descriptor allowlist can
enter a regression. Feature/task identifiers are join keys, never covariates.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


HEAD_COLUMNS = ("mean_head_kl", "full_vocab_argmax_flip_rate", "mean_edit_norm",
                "readout_activation_frequency")
ACTION_COLUMNS = tuple(f"{name}_{index}" for name in (
    "signed_action_bin_shift", "absolute_action_bin_shift", "action_argmax_flip")
    for index in range(7))
BASE_MODELS = {
    "zero": (),
    "kl_ridge": ("mean_head_kl",),
    "event_size_frequency_ridge": ("event_max", "mean_edit_norm", "readout_activation_frequency"),
    "head_ridge": HEAD_COLUMNS,
}
CATASTROPHIC_IDS = frozenset({7024, 24830})
TARGET = "raw_success_rate_minus_ablation_success_rate"


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data; missing/NaN values are not imputed")
    return float(value)


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _key(row):
    return integer(row.get("feature_id"), "feature_id"), integer(row.get("task_id"), "task_id")


def unique_rows(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} must be a nonempty list")
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError(f"Duplicate {name} feature/task key {key}")
        result[key] = row
    return result


def paired_task_labels(paired, *, task_ids, feature_ids=None):
    """Recompute signed harm-minus-repair targets from identical paired cases.

    The case key includes state digest and seed. Every feature must have exactly
    the same raw cases AND raw outcomes. Summaries are cross-checked, never used
    as a substitute for missing pairs.
    """
    features = paired.get("feature_ids")
    if (not isinstance(features, list) or not features or len(set(features)) != len(features)
            or any(type(fid) is not int or fid < 0 for fid in features)):
        raise ValueError("Labels require distinct explicit feature_ids")
    if feature_ids is not None and set(features) != set(feature_ids):
        raise ValueError("Historical label feature population differs from the declared development panel")
    protocol = paired.get("protocol", {})
    if protocol.get("alpha") != 0 or protocol.get("allow_missing_pairs") is not False:
        raise ValueError("Labels require complete alpha=0 ablation pairs")
    pid = paired.get("protocol_id")
    if not pid or protocol.get("protocol_id") != pid:
        raise ValueError("Label protocol binding mismatch")
    if set(protocol.get("expected_feature_ids", [])) != set(features):
        raise ValueError("Label protocol feature population mismatch")
    by_feature = {fid: {} for fid in features}
    for row in paired.get("paired_rows", []):
        fid = integer(row.get("feature_id"), "feature_id")
        task = integer(row.get("task_id"), "task_id")
        if fid not in by_feature or task not in task_ids or row.get("protocol_id") != pid:
            raise ValueError("Pair outside declared feature/task/protocol population")
        state, suite = row.get("initial_state_sha256"), row.get("suite")
        if not isinstance(state, str) or len(state) != 64 or not isinstance(suite, str) or not suite:
            raise ValueError("Pairs require an explicit suite and state digest")
        key = (suite, task, integer(row.get("task_episode_idx"), "task_episode_idx"), state,
               integer(row.get("seed"), "seed"))
        if key in by_feature[fid]:
            raise ValueError("Duplicate paired case")
        raw, edit = row.get("raw_success"), row.get("edit_success")
        if type(raw) is not bool or type(edit) is not bool:
            raise ValueError("Pair success labels must be explicit booleans")
        if finite(row.get("drop"), "paired drop") != int(raw) - int(edit):
            raise ValueError("Inconsistent pair drop; expected raw minus edited success")
        by_feature[fid][key] = (raw, edit)
    raw_cases = {key: value[0] for key, value in by_feature[features[0]].items()}
    if not raw_cases or {key[1] for key in raw_cases} != set(task_ids):
        raise ValueError("Missing paired tasks/cases")
    if paired.get("num_shared_cases") != len(raw_cases):
        raise ValueError("Shared case count mismatch")
    for pairs in by_feature.values():
        if {key: value[0] for key, value in pairs.items()} != raw_cases:
            raise ValueError("Features do not have identical raw case identities and outcomes")
    # The frozen protocol uses one suite and state identity per task/episode.
    protocol_cases = protocol.get("eval_cases", [])
    expected = {(r["suite"], r["task_id"], r["task_episode_idx"], r["initial_state_sha256"])
                for r in protocol_cases}
    observed = {(key[0], key[1], key[2], key[3]) for key in raw_cases}
    if not expected or len(expected) != len(protocol_cases) or observed != expected or len(observed) != len(raw_cases):
        raise ValueError("Paired cases differ from frozen protocol evaluation cases")
    effects = paired.get("feature_effects", [])
    effects_by_id = {r.get("feature_id"): r for r in effects}
    if len(effects_by_id) != len(effects) or set(effects_by_id) != set(features):
        raise ValueError("Duplicate/missing feature effect summary")
    labels = []
    for fid in sorted(features):
        summary = effects_by_id[fid]
        if summary.get("alpha") != 0 or summary.get("protocol_id") != pid:
            raise ValueError("Feature effect alpha/protocol mismatch")
        summaries = unique_rows([{**r, "feature_id": fid} for r in summary.get("task_effects", [])], "task effects")
        if set(summaries) != {(fid, task) for task in task_ids}:
            raise ValueError("Missing/extra task effect summaries")
        feature_labels = []
        for task in sorted(task_ids):
            values = [value for key, value in by_feature[fid].items() if key[1] == task]
            n = len(values)
            raw_sr = sum(a for a, _ in values) / n
            edit_sr = sum(b for _, b in values) / n
            harm, repair = sum(a and not b for a, b in values), sum(not a and b for a, b in values)
            label = {"feature_id": fid, "task_id": task, "raw_sr": raw_sr, "edit_sr": edit_sr,
                     "target_drop": (harm - repair) / n, "num_pairs": n,
                     "n_success_to_failure": harm, "n_failure_to_success": repair}
            saved = summaries[(fid, task)]
            for key, value in (("sr_raw", raw_sr), ("sr_edit", edit_sr), ("drop", label["target_drop"])):
                if not math.isclose(finite(saved.get(key), key), value, abs_tol=1e-12):
                    raise ValueError(f"Inconsistent task effect {key}")
            if saved.get("num_valid_pairs") != n:
                raise ValueError("Task effect pair count mismatch")
            feature_labels.append(label)
        for key, target_key in (("sr_raw", "raw_sr"), ("sr_edit", "edit_sr"), ("drop", "target_drop")):
            if not math.isclose(finite(summary.get(key), key), np.mean([r[target_key] for r in feature_labels]), abs_tol=1e-12):
                raise ValueError(f"Inconsistent feature effect {key}")
        for key in ("n_success_to_failure", "n_failure_to_success"):
            if summary.get(key) != sum(r[key] for r in feature_labels):
                raise ValueError(f"Inconsistent feature transition count {key}")
        if summary.get("num_valid_pairs") != sum(r["num_pairs"] for r in feature_labels):
            raise ValueError("Feature effect pair count mismatch")
        labels.extend(feature_labels)
    return labels


def fit_ridge(x, y, *, ridge_alpha=1.0, input_columns=None):
    """Fixed-alpha ridge with an unpenalized intercept; train-only z-scores."""
    alpha = finite(ridge_alpha, "ridge_alpha")
    if alpha <= 0:
        raise ValueError("ridge_alpha must be positive; no parameter search is performed")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not x.shape[1]:
        raise ValueError("Ridge needs nonempty 2D descriptors and one target per row")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("NaN/infinite descriptors or labels are not imputed")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    constant = scale == 0
    scale[constant] = 1.0
    standardized = (x - mean) / scale
    intercept = float(y.mean())
    coef = np.linalg.solve(standardized.T @ standardized + alpha * np.eye(x.shape[1]),
                           standardized.T @ (y - intercept))
    coef[constant] = 0.0
    if not np.isfinite(coef).all():
        raise ValueError("Non-finite ridge fit")
    return {"kind": "ridge", "ridge_alpha": alpha, "input_columns": list(input_columns or []),
            "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
            "intercept": intercept, "constant_columns": np.flatnonzero(constant).tolist(),
            "training_rows": len(y), "standardization": "training_fold_only_population_std_ddof0"}


def predict_ridge(model, x):
    x = np.asarray(x, dtype=float)
    mean, scale, coef = (np.asarray(model[name], dtype=float) for name in ("mean", "scale", "coef"))
    if (x.ndim != 2 or x.shape[1:] != mean.shape or mean.shape != scale.shape or mean.shape != coef.shape
            or not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.isfinite(coef).all() or (scale <= 0).any()):
        raise ValueError("Invalid/non-finite ridge inputs or preprocessing")
    prediction = ((x - mean) / scale) @ coef + finite(model.get("intercept"), "intercept")
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions")
    return prediction


def clip_drop(prediction, raw_sr):
    p, raw = np.asarray(prediction, dtype=float), np.asarray(raw_sr, dtype=float)
    if p.shape != raw.shape or not np.isfinite(p).all() or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("Clipping requires finite equal-shaped predictions and raw SR in [0, 1]")
    return np.clip(p, raw - 1.0, raw)


def _ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def feature_metrics(rows):
    by_feature = defaultdict(list)
    for row in rows:
        by_feature[row["feature_id"]].append(row)
    # fsum preserves exact cancellation of matching harm/repair rates. Ordinary
    # ndarray mean can turn a genuine zero into -1e-18 and break rank ties.
    aggregated = [{"feature_id": fid, "target_drop": math.fsum(r["target_drop"] for r in values) / len(values),
                   "predicted_drop": math.fsum(r["predicted_drop"] for r in values) / len(values),
                   "num_tasks": len(values)} for fid, values in sorted(by_feature.items())]
    if not aggregated:
        return {"num_features": 0, "feature_mean_mae": None, "spearman": None,
                "spearman_status": "empty_population", "per_feature": []}
    a, b = ([r[name] for r in aggregated] for name in ("target_drop", "predicted_drop"))
    ranks_a, ranks_b = _ranks(a), _ranks(b)
    constant = len(aggregated) < 2 or np.std(ranks_a) == 0 or np.std(ranks_b) == 0
    return {"num_features": len(aggregated), "feature_mean_mae": float(np.mean(np.abs(np.subtract(a, b)))),
            "spearman": None if constant else float(np.corrcoef(ranks_a, ranks_b)[0, 1]),
            "spearman_status": "undefined_constant_or_too_few_features" if constant else "ok",
            "per_feature": aggregated}


def descriptor_matrix(rows, columns):
    # This explicit allowlist is the only route into a fitted predictor.
    allowed = {"event_max", *HEAD_COLUMNS, *ACTION_COLUMNS}
    if not columns or set(columns) - allowed or len(set(columns)) != len(columns):
        raise ValueError("Predictor contains forbidden or duplicate input columns")
    return np.asarray([[finite(row.get(name), name) for name in columns] for row in rows], dtype=float)


def calibrate_rows(descriptor_rows, label_rows, *, ridge_alpha=1.0, with_response=False):
    """LOFO diagnostics and a separate all-development-label deployment fit."""
    descriptors, labels = unique_rows(descriptor_rows, "descriptor"), unique_rows(label_rows, "label")
    if set(descriptors) != set(labels):
        raise ValueError("Descriptor/label feature-task join has missing or extra rows")
    keys = sorted(labels)
    feature_ids, tasks = sorted({f for f, _ in keys}), sorted({t for _, t in keys})
    if len(feature_ids) < 3 or set(keys) != {(f, t) for f in feature_ids for t in tasks}:
        raise ValueError("Require at least three features and a complete feature/task product")
    y = np.asarray([finite(labels[key].get("target_drop"), "target_drop") for key in keys])
    raw = np.asarray([finite(labels[key].get("raw_sr"), "raw_sr") for key in keys])
    if not np.allclose(y, clip_drop(y, raw), rtol=0, atol=1e-12):
        raise ValueError("Signed drop outside its raw-SR feasibility interval")
    for task in tasks:
        if len({labels[(fid, task)]["raw_sr"] for fid in feature_ids}) != 1:
            raise ValueError("Raw task baseline is inconsistent across features")
    models = dict(BASE_MODELS)
    if with_response:
        models.update(response_ridge=HEAD_COLUMNS + ACTION_COLUMNS,
                      event_plus_head_ridge=("event_max",) + HEAD_COLUMNS)
    xrows, ids = [descriptors[key] for key in keys], np.asarray([key[0] for key in keys])
    diagnostics, final_models = {}, {}
    for name, columns in models.items():
        x = descriptor_matrix(xrows, columns) if columns else None
        folds, predictions = [], []
        def run_fold(train_ids, test_ids):
            train, test = np.isin(ids, train_ids), np.isin(ids, test_ids)
            if set(train_ids) & set(test_ids) or not train.any() or not test.any():
                raise ValueError("Feature holdout split is empty or overlapping")
            fit = fit_ridge(x[train], y[train], ridge_alpha=ridge_alpha, input_columns=columns) if columns else {"kind": "zero"}
            predicted = predict_ridge(fit, x[test]) if columns else np.zeros(int(test.sum()))
            clipped = clip_drop(predicted, raw[test])
            rows = [{"feature_id": keys[i][0], "task_id": keys[i][1], "target_drop": float(y[i]),
                     "raw_sr": float(raw[i]), "predicted_drop": float(clipped[j]),
                     "unclipped_predicted_drop": float(predicted[j])}
                    for j, i in enumerate(np.flatnonzero(test))]
            return {"training_feature_ids": list(train_ids), "held_out_feature_ids": list(test_ids),
                    "training_rows": int(train.sum()), "test_rows": int(test.sum()), "fit": fit}, rows
        for held_out in feature_ids:
            fold, rows = run_fold([fid for fid in feature_ids if fid != held_out], [held_out])
            folds.append(fold)
            predictions.extend(rows)
        noncat_ids = [fid for fid in feature_ids if fid not in CATASTROPHIC_IDS]
        noncat_rows = [r for r in predictions if r["feature_id"] in noncat_ids]
        diagnostic = {"input_columns": list(columns), "folds": folds, "oof_task_predictions": predictions,
                      "all_features": feature_metrics(predictions),
                      "noncatastrophic_subset_of_full_lofo": feature_metrics(noncat_rows)}
        if len(noncat_ids) >= 3:
            subset_predictions, subset_folds = [], []
            for held_out in noncat_ids:
                fold, rows = run_fold([fid for fid in noncat_ids if fid != held_out], [held_out])
                subset_folds.append(fold)
                subset_predictions.extend(rows)
            diagnostic["noncatastrophic_only_lofo"] = {
                **feature_metrics(subset_predictions), "folds": subset_folds,
                "oof_task_predictions": subset_predictions,
                "excluded_from_training_and_evaluation": sorted(CATASTROPHIC_IDS & set(feature_ids))}
        held_out = sorted(CATASTROPHIC_IDS & set(feature_ids))
        if len(held_out) == 2 and noncat_ids:
            fold, rows = run_fold(noncat_ids, held_out)
            diagnostic["joint_catastrophic_holdout"] = {**feature_metrics(rows), "fold": fold, "task_predictions": rows}
        else:
            diagnostic["joint_catastrophic_holdout"] = {"status": "unavailable_both_fixed_ids_required"}
        diagnostics[name] = diagnostic
        final_models[name] = (fit_ridge(x, y, ridge_alpha=ridge_alpha, input_columns=columns)
                              if columns else {"kind": "zero", "input_columns": []})
    report = {"target": TARGET, "independent_feature_units": len(feature_ids), "feature_task_rows": len(keys),
              "task_ids": tasks, "training_feature_ids": feature_ids, "ridge_alpha": finite(ridge_alpha, "ridge_alpha"),
              "models": diagnostics, "noncatastrophic_feature_ids": noncat_ids,
              "clipping": "per-task [shared_raw_SR - 1, shared_raw_SR]",
              "equal_budget": "same labels, task rows, held-out feature groups and fixed ridge alpha for every model",
              "hyperparameter_search": False, "feature_or_task_id_fitted_parameters": False,
              "missing_data_policy": "reject_nonfinite_or_missing_no_imputation",
              "constant_columns_policy": "train_fold_scale_one_coefficient_zero",
              "gate": {"status": "diagnostic_only_requires_independent_validation", "rollouts_authorized": False,
                       "confirmatory_claim": False, "published_new_method_claim": False},
              "limitations": ["Repeated development on previously inspected historical outcomes",
                              "Selected 16-feature panel is not a random dictionary sample",
                              "Shared task cases do not create independent feature labels",
                              "No model selection or rollout authorization from these metrics alone",
                              "Both full-LOFO subset and retrained noncatastrophic-only LOFO are diagnostic"]}
    model = {"target": TARGET, "training_feature_ids": feature_ids, "task_ids": tasks,
             "raw_sr_by_task": {str(t): labels[(feature_ids[0], t)]["raw_sr"] for t in tasks},
             "models": final_models, "fit_scope": "all_historical_development_labels_not_oof",
             "requires_disjoint_inference_feature_ids": True, "with_response": with_response}
    return report, model
