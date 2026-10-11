"""Bind outcome-blind predictions to a guarded, explicit native execution.

Preparation hashes inputs and checks existing current-device parity. Only
``run_execution`` invokes the native backend. Archived analysis needs neither
the original paths nor CUDA: ``verify_experiment`` is a pure contract check.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import re

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, git_identity, read_json, sha256_file,
)
from .protocol import plan_panel, verify_plan, selection_options
from .compatibility import validate_scoring_implementation, verify_scoring_bindings
from .workflow import REPO, implementation_identity, safe_output, validate_full_score_population

ANALYSIS = {"bootstrap_replicates": 2000, "bootstrap_seed": 1729, "confidence_level": .95}


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA256: {name}")
    return value


def _signed(doc, field):
    if not isinstance(doc, dict) or doc.get(field) != fingerprint({k: v for k, v in doc.items() if k != field}):
        raise ValueError(f"Missing or changed {field}")


def _conditions(plan):
    first = plan["panel_feature_ids"][0]
    return [
        {"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.0},
        {"condition_id": f"identity-{first}", "mode": "identity", "feature_id": first, "alpha": 1.0},
        *[{"condition_id": f"feature-{fid}", "mode": "intervention", "feature_id": fid, "alpha": 0.0}
          for fid in plan["panel_feature_ids"]],
    ]


def _seeded_cases(plan, base_seed):
    from .native import episode_seed
    result = deepcopy(plan["evaluation_cases"])
    for case in result:
        seed = episode_seed(base_seed, case["task_id"], case["task_episode_idx"])
        if "seed" in case and case["seed"] != seed:
            raise ValueError("Frozen case seed differs from condition-independent native RNG")
        case["seed"] = seed
    return result


def verify_experiment(experiment):
    """Pure semantic check; hashes detect changes, not human outcome blindness."""
    _signed(experiment, "execution_hash")
    if (experiment.get("schema_version") != "head_rank_execution_v1"
            or experiment.get("synthetic") is not False
            or experiment.get("model_download") is not False):
        raise ValueError("Unsupported real/offline execution contract")
    plan = experiment["prediction_plan"]
    verify_plan(plan)
    if plan["status"] != "predictions_frozen_not_runtime_approved":
        raise ValueError("Actual evaluation cases must be frozen first")
    if experiment.get("plan_hash") != plan["plan_hash"] or experiment.get("conditions") != _conditions(plan):
        raise ValueError("Execution conditions differ from the frozen feature panel")
    base_seed = _int(experiment.get("base_seed"), "base_seed")
    if (experiment.get("seed_algorithm") != "sha256_seed_task_trial_v1"
            or experiment.get("cases") != _seeded_cases(plan, base_seed)):
        raise ValueError("Execution cases/seeds differ from the prediction plan")
    total = len(experiment["conditions"]) * len(experiment["cases"])
    reserve = _int(experiment.get("retry_budget"), "retry_budget")
    if (experiment.get("total_rollouts") != total or total != plan["budget"]["total_rollouts"]
            or experiment.get("max_attempts") != total + reserve):
        raise ValueError("Frozen rollout/retry budget mismatch")
    cfg = experiment["config"]
    if experiment.get("config_hash") != fingerprint(cfg) or cfg["rollout"]["env_seed"] != base_seed:
        raise ValueError("Execution config binding mismatch")
    if (cfg["rollout"]["alpha"] != 0 or cfg["rollout"]["hook_start_step"] != 0
            or cfg["scoring"]["alpha"] != 0 or cfg["scoring"]["pair_batch_size"] != 1
            or cfg["model"]["allow_download"] is not False
            or cfg["model"]["expected_live_hidden_dtype"] != "bfloat16"
            or cfg["scoring"]["reduction_dtype"] != "float32"
            or cfg["scoring"]["negative_tolerance"] != 1e-6):
        raise ValueError("Execution must retain the BF16 all-step/full-removal reference protocol")
    if experiment["max_attempts"] > _int(cfg["rollout"]["max_total_rollouts"], "max_total_rollouts", 1):
        raise ValueError("Execution exceeds approved rollout cap including retries")
    code = experiment["code_identity"]
    if code.get("dirty") is not False or not re.fullmatch(r"[0-9a-f]{40}", code.get("commit", "")):
        raise ValueError("Execution requires a clean pinned code revision")
    bindings = plan["source_bindings"]
    if bindings.get("implementation") != experiment.get("implementation"):
        raise ValueError("Prediction and execution implementations differ")
    verify_scoring_bindings(bindings, experiment["implementation"])
    if "calibrated_predictions" in bindings:
        from ..head_predict.evaluation import validate_prediction_attachment
        validate_prediction_attachment(plan)
    if experiment.get("policy_config_hash") != fingerprint(experiment["effective_policy"]):
        raise ValueError("Native policy hash mismatch")
    if experiment.get("analysis") != ANALYSIS:
        raise ValueError("Analysis must use the predeclared shared-bootstrap settings")
    evidence = experiment["split_evidence"]
    scope = bindings["score_scope"]
    if (evidence.get("selection_eval_overlap") is not False
            or evidence.get("split_manifest_hash") != scope.get("split_manifest_hash")
            or scope.get("task_ids") != plan["task_ids"]
            or scope.get("sae_sha256") != experiment["numerical_identity"].get("sae_checkpoint_hash")):
        raise ValueError("Execution split/task/SAE differs from score scope")
    _digest(evidence.get("split_manifest_hash"), "split_manifest_hash")
    if not isinstance(evidence.get("historical_state_identity_verified"), bool):
        raise ValueError("Historical state verification must remain explicit")
    for name, source in experiment["sources"].items():
        _digest(source.get("sha256"), name)
        if not isinstance(source.get("path"), str) or not Path(source["path"]).is_absolute():
            raise ValueError("Execution input paths must be absolute")
    if not experiment["sources"] or not Path(experiment["output_root"]).is_absolute():
        raise ValueError("Execution sources/output path are required")
    # The exception is deliberately archival: it verifies the embedded proof,
    # not the availability of paths on the machine inspecting old results.
    has_receipt = "snapshot_receipt" in experiment
    has_source = "snapshot_receipt" in experiment["sources"]
    if has_receipt != has_source or has_receipt != ("snapshot_receipt_file_sha256" in experiment):
        raise ValueError("Snapshot receipt and its execution source binding must appear together")
    if has_receipt:
        from .snapshot import verify_snapshot_receipt
        receipt = experiment["snapshot_receipt"]
        if (not isinstance(receipt, dict)
                or receipt.get("runtime_implementation") != experiment["implementation"]
                or receipt.get("scoring_implementation") != bindings.get("scoring_implementation", experiment["implementation"])):
            raise ValueError("Snapshot receipt implementation differs from the score/execution bindings")
        receipt_file_sha256 = _digest(experiment["snapshot_receipt_file_sha256"], "snapshot_receipt_file_sha256")
        if experiment["sources"]["snapshot_receipt"]["sha256"] != receipt_file_sha256:
            raise ValueError("Snapshot receipt file binding differs from the frozen source")
        verify_snapshot_receipt(receipt,
            score_hash=bindings["score_hash"], scores_sha256=bindings["scores_sha256"],
            current_identity=experiment["numerical_identity"], sources=experiment["sources"])


def build_experiment(plan, *, cfg, numerical_identity, policy, code_identity,
                     sources, split_evidence, output_root, retry_budget=0,
                     snapshot_receipt=None, snapshot_receipt_file_sha256=None):
    """Assemble a contract from already checked inputs; never executes a model."""
    verify_plan(plan)
    if not plan.get("evaluation_cases"):
        raise ValueError("Freeze actual evaluation cases before preparing execution")
    cases = _seeded_cases(plan, cfg["rollout"]["env_seed"])
    conditions = _conditions(plan)
    total = len(cases) * len(conditions)
    experiment = deepcopy({
        "schema_version": "head_rank_execution_v1", "prediction_plan": plan,
        "plan_hash": plan["plan_hash"], "conditions": conditions, "cases": cases,
        "total_rollouts": total, "retry_budget": retry_budget, "max_attempts": total + retry_budget,
        "base_seed": cfg["rollout"]["env_seed"], "seed_algorithm": "sha256_seed_task_trial_v1",
        "config": cfg, "config_hash": fingerprint(cfg), "numerical_identity": numerical_identity,
        "effective_policy": policy, "policy_config_hash": fingerprint(policy),
        "code_identity": code_identity, "implementation": plan["source_bindings"]["implementation"],
        "sources": sources, "split_evidence": split_evidence, "output_root": str(Path(output_root).resolve()),
        "analysis": ANALYSIS, "synthetic": False, "model_download": False,
        "status": "ready_for_explicit_execution",
        "scientific_scope": "prospective_new_feature_panel_on_fixed_tasks_not_confirmatory_task_generalization",
    })
    if snapshot_receipt is not None:
        experiment["snapshot_receipt"] = deepcopy(snapshot_receipt)
        experiment["snapshot_receipt_file_sha256"] = snapshot_receipt_file_sha256
    elif snapshot_receipt_file_sha256 is not None:
        raise ValueError("Snapshot receipt file hash requires an embedded receipt")
    experiment["execution_hash"] = fingerprint(experiment)
    verify_experiment(experiment)
    return experiment


def select_cases(split_path, output, *, cases_per_task=5, seed=1729):
    """Outcome-blind subset of the original frozen evaluation split only."""
    from event_sae.research.output_head.splits import validate_split_manifest
    _int(cases_per_task, "cases_per_task", 1)
    _int(seed, "seed")
    split = read_json(split_path)
    validate_split_manifest(split)
    rows, selected = split["splits"]["evaluation"], []
    for task in sorted({r["task_id"] for r in rows}):
        pool = sorted((r for r in rows if r["task_id"] == task), key=lambda r: r["task_episode_idx"])
        if len(pool) < cases_per_task:
            raise ValueError("Requested case count exceeds the frozen evaluation split")
        rng = random.Random(f"headrank-cases-v1:{seed}:{task}")
        selected.extend({k: r[k] for k in ("suite", "task_id", "task_episode_idx", "initial_state_sha256")}
                        for r in rng.sample(pool, cases_per_task))
    selected.sort(key=lambda r: (r["task_id"], r["task_episode_idx"]))
    result = {"schema_version": "head_rank_evaluation_cases_v1", "cases": selected,
              "split_manifest_hash": split["manifest_hash"], "discovery_eval_overlap": False,
              "evaluation_labels_previously_used": split["evaluation_labels_previously_used"],
              "historical_state_identity_verified": split.get("historical_state_identity_verified", False),
              "selection": {"algorithm": "seeded_without_replacement_per_task_v1", "seed": seed,
                            "cases_per_task": cases_per_task, "uses_outcomes": False}}
    atomic_write_json(safe_output(output, [split_path]), result)
    return {"status": "cases_frozen", "output": str(Path(output).resolve()), "cases": len(selected), "new_rollouts": 0}


def _check_score_plan(plan, scores, evaluation, registry):
    _signed(scores, "score_hash")
    validate_full_score_population(scores)
    if scores.get("development_only") is not False or scores.get("feature_set") != "all-active":
        raise ValueError("Only complete all-active discovery scores can authorize a new panel")
    if registry.get("reviewed_for_team_prior_outcomes") is not True:
        raise ValueError("Team known-feature registry has not been reviewed")
    current_implementation = implementation_identity()
    compatibility = validate_scoring_implementation(scores["implementation"], current_implementation)
    bindings = plan["source_bindings"]
    verify_scoring_bindings(bindings, current_implementation)
    if "calibrated_predictions" in bindings:
        from ..head_predict.evaluation import validate_prediction_attachment
        validate_prediction_attachment(plan)
    if "scoring_implementation" in bindings:
        if (bindings["scoring_implementation"] != scores["implementation"]
                or bindings.get("scorer_compatibility") != compatibility):
            raise ValueError("Frozen immutable scorer provenance differs from the scores")
    elif scores["implementation"] != current_implementation:
        raise ValueError("Legacy score reuse requires explicit separate scoring/runtime bindings")
    if (bindings.get("implementation") != current_implementation
            or bindings.get("score_hash") != scores["score_hash"]
            or bindings.get("score_scope") != scores["scope"]
            or bindings.get("event_max_binding") != scores["event_max_binding"]
            or bindings.get("evaluation_label_history") != evaluation.get("evaluation_labels_previously_used")):
        raise ValueError("Frozen plan no longer matches score/evaluation provenance")
    replay = plan_panel(scores, additional_evaluated_ids=registry["feature_ids"],
        feature_count=len(plan["panel_feature_ids"]), seed=plan["selection"]["seed"], top_k=plan["top_k"],
        task_ids=plan["task_ids"], cases_per_task=plan["cases_per_task"], primary_metric=plan["primary_metric"],
        reference_bindings=bindings, evaluation_cases=evaluation["cases"], **selection_options(plan))
    if replay != plan:
        raise ValueError("Predictions do not replay exactly from complete frozen scores and cases")


def _check_split(cfg, split, sample, scores, plan, evaluation):
    from event_sae.research.output_head.splits import validate_selection_split
    mode = "followup" if split.get("mode") == "followup" else "confirmatory"
    validate_selection_split(split, sample["discovery_episodes"], stage="score", mode=mode)
    validate_selection_split(split, plan["evaluation_cases"], stage="evaluation", mode=mode)
    expected = split["manifest_hash"]
    if (evaluation.get("discovery_eval_overlap") is not False
            or evaluation.get("split_manifest_hash") != expected
            or scores["scope"].get("split_manifest_hash") != expected
            or sample.get("split_manifest_hash") != expected
            or scores["scope"].get("discovery_manifest_hash") != sample.get("discovery_manifest_hash")
            or scores["scope"].get("sample_manifest_hash") != sample.get("sample_hash")
            or sorted(scores["scope"]["task_ids"]) != plan["task_ids"]
            or any(c["suite"] != cfg["scope"]["suite"] for c in plan["evaluation_cases"])):
        raise ValueError("Discovery/sample/evaluation split binding mismatch")
    # Do not promote reused state-label history to a fresh confirmatory claim.
    if evaluation.get("evaluation_labels_previously_used") != split["evaluation_labels_previously_used"]:
        raise ValueError("Evaluation label history differs from the original frozen split")
    return {"split_manifest_hash": expected, "selection_eval_overlap": False,
            "historical_state_identity_verified": split.get("historical_state_identity_verified", False),
            "evaluation_labels_previously_used": split["evaluation_labels_previously_used"],
            "mode": mode, "new_feature_outcomes_used_for_selection": False}


def _check_sources(sources):
    for name, source in sources.items():
        if not Path(source["path"]).is_file() or sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"Execution source missing or changed: {name}")


def _run_path(experiment, *, contract_path=None, new=False):
    cfg = experiment["config"]
    protected = [s["path"] for s in experiment["sources"].values()]
    protected += [cfg["inputs"].get("local_model_snapshot"), cfg["inputs"].get("dense_dir"),
                  cfg["output"]["root_dir"], contract_path, REPO, REPO.parent / "Event-SAE-Head"]
    protected += experiment.get("protected_roots", [])
    target = assert_safe_output(experiment["output_root"], protected)
    # The journal itself disallows symlink ancestors too, before any mkdir.
    original = Path(experiment["output_root"]).absolute()
    if any(p.is_symlink() for p in (original, *original.parents)):
        raise ValueError("Execution journal cannot use symlink paths")
    if new and target.exists():
        raise FileExistsError("Choose a new run output directory")
    return target


def prepare_execution(config, plan_path, scores_path, registry_path, eval_manifest,
                      reference_result, split_path, output, run_output, *, retry_budget=0,
                      max_rollouts=None, snapshot_receipt_path=None):
    """No rollout/model forward. Bind score, split, files, code and current parity."""
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import numerical_identity, load_parity, _sae_paths, _head_path
    from .native import effective_policy
    _int(retry_budget, "retry_budget")
    code = git_identity(REPO)
    if code["dirty"]:
        raise ValueError("Freeze reviewed development in a clean commit before preparing paid execution; no automatic commit")
    cfg = load_config(config)
    if max_rollouts is not None:
        cfg["rollout"]["max_total_rollouts"] = _int(max_rollouts, "max_rollouts", 1)
    plan, scores = read_json(plan_path), read_json(scores_path)
    verify_plan(plan)
    evaluation, registry = read_json(eval_manifest), read_json(registry_path)
    _check_score_plan(plan, scores, evaluation, registry)
    for key, path in (("scores_sha256", scores_path), ("registry_sha256", registry_path),
                      ("eval_manifest_sha256", eval_manifest)):
        if plan["source_bindings"].get(key) != sha256_file(path):
            raise ValueError(f"Frozen input changed: {key}")
    root = Path(reference_result).resolve()
    sample = read_json(root / "sample_manifest.json")
    for key, relative in (("source_sample_sha256", "sample_manifest.json"),
                          ("source_scores_sha256", "scores/scores.json"),
                          ("source_cache_manifest_sha256", "readouts/manifest.json")):
        if scores.get(key) != sha256_file(root / relative):
            raise ValueError(f"Restored reference differs from scoring source: {key}")
    split = read_json(split_path)
    split_evidence = _check_split(cfg, split, sample, scores, plan, evaluation)
    cfg["sampling"].update(mode=split_evidence["mode"], split_manifest=str(Path(split_path).resolve()),
                           task_ids=plan["task_ids"])
    cfg["rollout"]["eval_manifest"] = str(Path(eval_manifest).resolve())
    identity = numerical_identity(cfg)
    receipt_file_sha256 = sha256_file(snapshot_receipt_path) if snapshot_receipt_path is not None else None
    receipt = read_json(snapshot_receipt_path) if snapshot_receipt_path is not None else None
    if receipt is not None:
        from .snapshot import verify_snapshot_receipt
        verify_snapshot_receipt(receipt, score_hash=scores["score_hash"],
            scores_sha256=sha256_file(scores_path), old_identity=scores.get("identity"), current_identity=identity)
    elif scores.get("identity") != identity:
        raise ValueError("Current runtime identity differs from scoring; a valid --snapshot-receipt is required for verified snapshot timestamp-only differences")
    load_parity(cfg, identity)
    policy = effective_policy(cfg)
    weights, sae_config = _sae_paths(cfg)
    head = _head_path(cfg)
    if head.is_file():
        head = head.parent
    paths = {"config": config, "prediction_plan": plan_path, "scores": scores_path,
             "registry": registry_path, "evaluation": eval_manifest, "split": split_path,
             "sample": root / "sample_manifest.json", "reference_scores": root / "scores/scores.json",
             "cache_manifest": root / "readouts/manifest.json", "sae": weights, "sae_config": sae_config,
             "head_weights": head / "output_head.safetensors", "head_manifest": head / "output_head_manifest.json",
             "base_eval_config": cfg["rollout"]["base_eval_config"],
             "generation_manifest": cfg["inputs"]["generation_manifest"],
             "head_parity": Path(cfg["output"]["root_dir"]) / "validation/runtime_parity.json",
             "edit_parity": Path(cfg["output"]["root_dir"]) / "validation/edit_parity.json"}
    snapshot = Path(cfg["inputs"]["local_model_snapshot"])
    # Full content hashes, not merely mtimes, pin weights and processor assets.
    assets = sorted(p for p in snapshot.iterdir() if p.is_file() and
                    p.suffix in {".safetensors", ".json", ".py", ".model", ".txt"})
    if not any(p.suffix == ".safetensors" for p in assets):
        raise ValueError("Native snapshot requires local safetensor weights")
    paths.update({"snapshot/" + p.name: p for p in assets})
    if receipt is not None:
        paths["snapshot_receipt"] = snapshot_receipt_path
    sources = {name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)} for name, path in paths.items()}
    if receipt is not None:
        if sources["snapshot_receipt"]["sha256"] != receipt_file_sha256:
            raise ValueError("Snapshot receipt changed during execution preparation")
        verify_snapshot_receipt(receipt, score_hash=scores["score_hash"],
            scores_sha256=sha256_file(scores_path), old_identity=scores["identity"], current_identity=identity,
            sources=sources)
    protected = [str(root), str(weights.parent.resolve()), str(snapshot.resolve()),
                 str(head.resolve()), *scores.get("source_readonly_roots", [])]
    target = safe_output(output, [*paths.values(), *protected, run_output, REPO])
    experiment = build_experiment(plan, cfg=cfg, numerical_identity=identity, policy=policy,
        code_identity=code, sources=sources, split_evidence=split_evidence,
        output_root=run_output, retry_budget=retry_budget, snapshot_receipt=receipt,
        snapshot_receipt_file_sha256=receipt_file_sha256)
    experiment["protected_roots"] = protected
    experiment["execution_hash"] = fingerprint({k: v for k, v in experiment.items() if k != "execution_hash"})
    _run_path(experiment, contract_path=target, new=True)
    if git_identity(REPO) != code or implementation_identity() != plan["source_bindings"]["implementation"]:
        raise ValueError("Source changed during preparation")
    atomic_write_json(target, experiment)
    return {"status": "prepared_not_executed", "experiment": str(target),
            "execution_hash": experiment["execution_hash"], "code_revision": code["commit"],
            "conditions": len(experiment["conditions"]), "total_rollouts": experiment["total_rollouts"],
            "max_attempts": experiment["max_attempts"], "new_rollouts": 0}


def run_execution(experiment_path, *, approved_hash, expected_revision,
                  retry_incomplete=False, max_new_attempts=None):
    from .execution import execute_experiment, inspect_experiment
    experiment = read_json(experiment_path)
    verify_experiment(experiment)
    if not approved_hash or approved_hash != experiment["execution_hash"]:
        raise ValueError("Explicit approval must match the complete execution hash")
    if not expected_revision or expected_revision != experiment["code_identity"]["commit"]:
        raise ValueError("Explicit expected code revision differs from the prepared revision")
    target = _run_path(experiment, contract_path=experiment_path)
    progress = inspect_experiment(experiment, target)
    if progress["status"] == "completed":
        return {**progress, "new_rollouts": 0, "model_execution": False,
                "scientific_scope": "stored receipt/action verification only; runtime parity not rerun"}
    if git_identity(REPO) != experiment["code_identity"] or implementation_identity() != experiment["implementation"]:
        raise ValueError("Source revision/cleanliness/implementation changed; execution blocked")
    _check_sources(experiment["sources"])
    from event_sae.research.output_head.workflow import numerical_identity, load_parity
    from .native import NativeBackend, effective_policy
    cfg = experiment["config"]
    identity = numerical_identity(cfg)
    if identity != experiment["numerical_identity"] or effective_policy(cfg) != experiment["effective_policy"]:
        raise ValueError("Current device/model/policy differs from the frozen execution")
    load_parity(cfg, identity)
    backend = NativeBackend(cfg, identity)
    try:
        return execute_experiment(experiment, target, backend, approved_hash=approved_hash,
                                  retry_incomplete=retry_incomplete, max_new_attempts=max_new_attempts)
    finally:
        backend.close()
