"""Paired fixed-task analysis of the complete team500 condition journal.

No model or simulator is imported. Uncertainty resamples initial states within
the ten fixed tasks, using the same draws for every measured feature/method.
"""
from __future__ import annotations

import math
from pathlib import Path
import random

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, atomic_write_text, fingerprint, read_json,
    sha256_file,
)
from .execution import compare_traces, load_completed

METHODS = ("mean_head_kl", "tail_head_kl", "event_max")


def _quantile(values, probability):
    ordered = sorted(values)
    at = (len(ordered) - 1) * probability
    low = math.floor(at)
    high = math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _interval(values, confidence):
    alpha = (1 - confidence) / 2
    return {"low": _quantile(values, alpha), "high": _quantile(values, 1 - alpha)}


def _ranks(values):
    positions = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.] * len(values)
    left = 0
    while left < len(positions):
        right = left + 1
        while right < len(positions) and values[positions[right]] == values[positions[left]]:
            right += 1
        rank = (left + right - 1) / 2 + 1
        for i in positions[left:right]:
            ranks[i] = rank
        left = right
    return ranks


def _spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    a, b = _ranks(x), _ranks(y)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    if va == 0 or vb == 0:
        return None
    return sum((v - ma) * (w - mb) for v, w in zip(a, b)) / math.sqrt(va * vb)


def analyze_receipts(plan, receipts, *, bootstrap_replicates=2000,
                     bootstrap_seed=1729, confidence_level=.95, score_rows=None):
    """Pure statistical core; receipts were validated by the journal loader."""
    if (type(bootstrap_replicates) is not int or bootstrap_replicates < 100
            or type(bootstrap_seed) is not int
            or isinstance(confidence_level, bool)
            or not isinstance(confidence_level, (int, float))
            or not 0 < confidence_level < 1):
        raise ValueError("Invalid bootstrap configuration")
    conditions, cases = plan["conditions"], plan["cases"]
    required = {c["condition_id"] for c in conditions}
    if set(receipts) != required:
        raise ValueError("Analysis requires every originally planned condition; no substitutions")
    raw_id = next(c["condition_id"] for c in conditions if c["mode"] == "raw")
    identity_id = next(c["condition_id"] for c in conditions if c["mode"] == "identity")
    identity_check = compare_traces(receipts[raw_id], receipts[identity_id])
    raw = receipts[raw_id]["episodes"]
    if len(raw) != 500 or len(cases) != 500:
        raise ValueError("Analysis requires the exact full500 paired case population")
    tasks = {}
    for index, case in enumerate(cases):
        tasks.setdefault(case["task_id"], []).append(index)
    if sorted(tasks) != list(range(10)) or any(len(ids) != 50 for ids in tasks.values()):
        raise ValueError("Require exactly ten tasks with fifty paired states each")
    raw_values = [int(row["success"]) for row in raw]
    feature_conditions = {c["feature_id"]: c["condition_id"]
                          for c in conditions if c["mode"] == "intervention"}
    features = sorted(feature_conditions)
    if features != sorted(plan["feature_ids"]):
        raise ValueError("Measured feature union differs from frozen plan")
    effects, flip_values, counts, feature_rows = {}, {}, {}, []
    for fid in features:
        edited = receipts[feature_conditions[fid]]["episodes"]
        if len(edited) != len(raw):
            raise ValueError("Incomplete feature outcome array")
        values = [int(row["success"]) for row in edited]
        delta = [base - changed for base, changed in zip(raw_values, values)]
        flips = [int(value != 0) for value in delta]
        harm, repair = delta.count(1), delta.count(-1)
        effects[fid], flip_values[fid], counts[fid] = delta, flips, harm - repair
        per_task = [{"task_id": task, "pairs": len(ids),
                     "raw_successes": sum(raw_values[i] for i in ids),
                     "ablated_successes": sum(values[i] for i in ids),
                     "signed_drop": sum(delta[i] for i in ids) / len(ids)}
                    for task, ids in sorted(tasks.items())]
        feature_rows.append({"feature_id": fid, "pairs": len(raw),
            "raw_success_rate": sum(raw_values) / len(raw),
            "ablated_success_rate": sum(values) / len(values),
            "signed_drop": (harm - repair) / len(raw),
            "harm_cases": harm, "repair_cases": repair,
            "outcome_flip_rate": (harm + repair) / len(raw), "tasks": per_task})
    draws = {fid: [] for fid in features}
    flip_draws = {fid: [] for fid in features}
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_replicates):
        indices = [rng.choice(ids) for _, ids in sorted(tasks.items()) for _ in range(len(ids))]
        for fid in features:
            draws[fid].append(sum(effects[fid][i] for i in indices) / len(indices))
            flip_draws[fid].append(sum(flip_values[fid][i] for i in indices) / len(indices))
    for row in feature_rows:
        row["signed_drop_ci"] = _interval(draws[row["feature_id"]], confidence_level)
        row["outcome_flip_ci"] = _interval(flip_draws[row["feature_id"]], confidence_level)
    method_rows, method_draws = [], {}
    for method in METHODS:
        ids = plan["methods"][method]
        if len(ids) != 5 or not set(ids) <= set(features):
            raise ValueError("Every frozen Top-5 must have complete measured outcomes")
        method_draws[method] = [math.fsum(draws[fid][i] for fid in ids) / 5
                                for i in range(bootstrap_replicates)]
        row = {"method": method, "top_k": 5, "feature_ids": ids,
               "top_k_coverage": 1., "top_k_mean_signed_drop": sum(counts[fid] for fid in ids) / 2500,
               "top_k_signed_drop_ci": _interval(method_draws[method], confidence_level),
               "top_k_mean_flip_rate": sum(sum(flip_values[fid]) for fid in ids) / 2500,
               "panel_signed_drop_spearman": None, "panel_flip_spearman": None}
        if score_rows is not None:
            missing = [fid for fid in features if fid not in score_rows or method not in score_rows[fid]]
            if missing:
                raise ValueError(f"Missing full-panel scores: {missing}")
            x = [score_rows[fid][method] for fid in features]
            # Integer outcome counts preserve exact ties across task summation.
            row["panel_signed_drop_spearman"] = _spearman(x, [counts[fid] for fid in features])
            row["panel_flip_spearman"] = _spearman(x, [sum(flip_values[fid]) for fid in features])
        method_rows.append(row)
    contrasts = []
    for first, second in (("mean_head_kl", "event_max"), ("tail_head_kl", "event_max"),
                          ("tail_head_kl", "mean_head_kl")):
        distribution = [a - b for a, b in zip(method_draws[first], method_draws[second])]
        a, b = [next(row for row in method_rows if row["method"] == method)
                for method in (first, second)]
        contrasts.append({"first": first, "second": second,
            "top_k_signed_drop_difference": a["top_k_mean_signed_drop"] - b["top_k_mean_signed_drop"],
            "ci": _interval(distribution, confidence_level),
            "same_top_k_feature_set": set(plan["methods"][first]) == set(plan["methods"][second])})
    return {"schema_version": "team500_analysis_v1", "status": "analyzed",
            "plan_hash": plan["plan_hash"], "synthetic": False,
            "total_rollouts": plan["total_rollouts"], "feature_count": len(features),
            "pairs_per_feature": 500, "raw_successes": sum(raw_values),
            "raw_success_rate": sum(raw_values) / len(raw_values),
            "identity_validation": identity_check, "feature_effects": feature_rows,
            "predictor_comparison": method_rows, "top_k_contrasts": contrasts,
            "uncertainty": {"bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed, "confidence_level": confidence_level,
                "resampling": "same_indices_for_all_features_within_each_fixed_task",
                "scope": "conditional_on_fixed_ten_tasks_and_frozen_selected_feature_union",
                "score_uncertainty_included": False, "multiple_comparisons": "pointwise_unadjusted"},
            "interpretation": "retrospective_full500_overlap_protocol_alignment_not_held_out",
            "limitations": [
                "Discovery and evaluation use the same 500 initial states; SAE/clusters were also inherited from full500.",
                "Top-K is the mean of separate single-feature removals, not simultaneous five-feature removal.",
                "Signed drop is raw minus ablated success rate; negative values mean improvement.",
                "Output sensitivity and success-rate changes do not establish feature semantics or purposeful behavior.",
                "This selected Top-K union does not establish all-feature prediction accuracy or held-out-task/model generalization.",
                "Pointwise intervals are not multiplicity-adjusted and do not prove equivalence when zero is included.",
                "Historical cross-GPU outcomes are not required to match; within-runtime implementation conformance is checked separately.",
            ], "new_rollouts": 0, "model_execution": False}


def markdown_report(result):
    def pc(value):
        return f"{100 * value:.2f}"
    def corr(value):
        return "정의 불가" if value is None else f"{value:.4f}"
    ci_label = f"{100 * result['uncertainty']['confidence_level']:g}%"
    lines = ["# Exp3 — 팀 Baseline 조건의 500개·Top-5 비교", "",
        f"Plan: `{result['plan_hash']}`", "",
        f"완료 평가: {result['total_rollouts']:,}회. 특징 {result['feature_count']}개, 특징당 대응 초기 상태 500개.",
        f"원본 성공률: {pc(result['raw_success_rate'])}% ({result['raw_successes']}/500).", "",
        "원본과 무변경 대조의 모든 행동 명령·길이·성공 여부가 일치하였다.",
        "이 분석은 저장된 결과만 사용하며 새 모델 추론이나 로봇 실행을 수행하지 않는다.", "",
        "## 특징별 제거 효과", "",
        "성공률 하락 = 원본 − 제거 후 성공률. 음수는 개선을 의미한다.", "",
        f"| 특징 | 제거 후 성공률(%) | 하락(%p) | {ci_label} 구간(%p) | 성공→실패 / 실패→성공 |",
        "| --- | ---: | ---: | --- | ---: |"]
    for row in result["feature_effects"]:
        ci = row["signed_drop_ci"]
        lines.append(f"| {row['feature_id']} | {pc(row['ablated_success_rate'])} | {pc(row['signed_drop'])} | [{pc(ci['low'])}, {pc(ci['high'])}] | {row['harm_cases']} / {row['repair_cases']} |")
    lines += ["", "## 선정 방법 비교", "",
        f"| 방법 | 원래 Top-5 | 평균 하락(%p) | {ci_label} 구간(%p) | 후보군 내 하락 순위 상관 | 후보군 내 변화율 순위 상관 |",
        "| --- | --- | ---: | --- | ---: | ---: |"]
    for row in result["predictor_comparison"]:
        ci = row["top_k_signed_drop_ci"]
        lines.append(f"| {row['method']} | {row['feature_ids']} | {pc(row['top_k_mean_signed_drop'])} | [{pc(ci['low'])}, {pc(ci['high'])}] | {corr(row['panel_signed_drop_spearman'])} | {corr(row['panel_flip_spearman'])} |")
    lines += ["", "## 대응 차이", "",
        f"| 비교(앞 − 뒤) | Top-5 평균 하락 차이(%p) | {ci_label} 구간(%p) | Top-5 집합 동일 |",
        "| --- | ---: | --- | --- |"]
    for row in result["top_k_contrasts"]:
        ci = row["ci"]
        lines.append(f"| {row['first']} − {row['second']} | {pc(row['top_k_signed_drop_difference'])} | [{pc(ci['low'])}, {pc(ci['high'])}] | {row['same_top_k_feature_set']} |")
    lines += ["", "## 해석 범위", "",
        "- 같은 500개 초기 상태로 선정과 평가를 수행하였으며, 독립적인 일반화 검증이 아니다.",
        "- 순위 상관은 실제 평가한 Top-5 합집합 안의 Spearman 상관이며, 전체 사전의 예측 정확도가 아니다.",
        "- 각 특징을 하나씩 제거한 효과의 평균이다. 다섯 특징을 동시에 제거한 결과가 아니다.",
        "- Mean과 Tail의 Top-5 집합이 같으면 두 방법의 Top-5 평균 효과도 같다.",
        "- 신뢰구간은 고정 과제·후보군 안에서 초기 상태를 대응 재표집한 결과이며, 다중비교 보정은 하지 않았다.",
        "- 기존 다른 GPU의 성공률과 숫자까지 같아졌다는 의미가 아니다.",
        "- 높은 출력 변화나 제거 효과만으로 특징의 의미·회로 역할을 확정하지 않는다.", ""]
    return "\n".join(lines)


def analyze_experiment(plan_path, output_root, analysis_output, *, scores_path=None,
                       bootstrap_replicates=2000, bootstrap_seed=1729, confidence_level=.95):
    plan, completed, _, _, binding = load_completed(plan_path, output_root, scores_path)
    if set(completed) != {c["condition_id"] for c in plan["conditions"]}:
        raise ValueError("Analysis is blocked until all frozen conditions complete")
    effective_scores_path = scores_path or plan.get("reference_bindings", {}).get("scores_path")
    if effective_scores_path is None:
        raise ValueError("Analysis requires the frozen full-score artifact")
    source = read_json(effective_scores_path)
    if (source.get("score_hash") != plan["scores_hash"]
            or source.get("score_hash") != fingerprint({k: v for k, v in source.items() if k != "score_hash"})
            or (plan.get("score_file_sha256") is not None
                and sha256_file(effective_scores_path) != plan["score_file_sha256"])):
        raise ValueError("Analysis scores differ from the frozen score artifact")
    rows = {row["feature_id"]: row for row in source["feature_scores"]}
    result = analyze_receipts(plan, {key: value["receipt"] for key, value in completed.items()},
        bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level, score_rows=rows)
    result["runtime_identity"] = binding["runtime_identity"]
    result["score_file_sha256"] = sha256_file(effective_scores_path)
    result["receipt_hashes"] = {key: value["result_hash"] for key, value in completed.items()}
    result["analysis_hash"] = fingerprint(result)
    output = assert_safe_output(analysis_output, [plan_path, output_root, effective_scores_path])
    if output.exists():
        raise FileExistsError("Analysis output must be a fresh directory")
    atomic_write_json(output / "analysis.json", result)
    atomic_write_text(output / "report.md", markdown_report(result))
    return {"status": "analyzed", "output": str(output), "analysis_hash": result["analysis_hash"],
            "total_rollouts": result["total_rollouts"], "feature_count": result["feature_count"],
            "new_rollouts": 0, "model_execution": False}
