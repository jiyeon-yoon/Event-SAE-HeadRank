#!/usr/bin/env python3
"""Isolated HeadRank scoring, frozen plans and explicitly approved native evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    budget = commands.add_parser("budget", help="historical cost extrapolation, no execution")
    budget.add_argument("--features", type=int, default=8)
    budget.add_argument("--tasks", type=int, default=10)
    budget.add_argument("--cases-per-task", type=int, default=5)
    score = commands.add_parser("score", help="score existing hidden cache with fresh base parity")
    score.add_argument("--config", required=True)
    score.add_argument("--reference-result", required=True)
    score.add_argument("--event-scores", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--feature-set", choices=("development", "all-active"), default="development")
    score.add_argument("--full-readouts", action="store_true")
    score.add_argument("--max-pairs", type=int, default=100000)
    score.add_argument("--execute", action="store_true")
    plan = commands.add_parser("plan", help="freeze unseen feature selection; no rollout execution")
    plan.add_argument("--scores", required=True)
    plan.add_argument("--registry", required=True)
    plan.add_argument("--eval-manifest")
    plan.add_argument("--features", type=int)
    plan.add_argument("--cases-per-task", type=int)
    plan.add_argument("--preset", choices=("full10",),
                      help="16 feature conditions + raw/identity, 10 tasks x 20 cases = 3600 rollouts")
    plan.add_argument("--selection-strategy", choices=("round_robin", "topk_union"))
    plan.add_argument("--include-excess-topk", action="store_true",
                      help="also guarantee Excess top-K coverage (secondary, not the default primary claim)")
    plan.add_argument("--seed", type=int, default=1729)
    plan.add_argument("--top-k", type=int, default=3)
    plan.add_argument("--primary-metric", choices=("tail_head_kl", "direction_excess_kl"), default="tail_head_kl")
    plan.add_argument("--output", required=True)
    verify = commands.add_parser("verify-plan", help="check frozen predictions, not GPU/runtime parity")
    verify.add_argument("--plan", required=True)
    cases = commands.add_parser("cases", help="select evaluation states from the frozen split, without outcomes")
    cases.add_argument("--split", required=True)
    cases.add_argument("--cases-per-task", type=int, default=5)
    cases.add_argument("--seed", type=int, default=1729)
    cases.add_argument("--output", required=True)
    snapshot_check = commands.add_parser("snapshot-check", help="hash pinned model files and current parity for timestamp-only score reuse; no model execution")
    for name in ("config", "scores", "reference-result", "output"):
        snapshot_check.add_argument("--" + name, required=True)
    prepare = commands.add_parser("prepare-run", help="bind predictions/cases/code/current parity; no model forward")
    for name in ("config", "plan", "scores", "registry", "eval-manifest", "reference-result", "split", "output", "run-output"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--retry-budget", type=int, default=0)
    prepare.add_argument("--max-rollouts", type=int, help="explicit cap including retry reserve, otherwise config cap")
    prepare.add_argument("--snapshot-receipt", help="optional verified timestamp-only snapshot reuse receipt; never relaxes device/parity checks")
    run = commands.add_parser("run", help="dry preview unless --execute plus exact hash and revision are supplied")
    run.add_argument("--experiment", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--approved-execution-hash")
    run.add_argument("--expected-code-revision")
    run.add_argument("--retry-incomplete", action="store_true")
    run.add_argument("--max-new-attempts", type=int, help="stop cleanly at a case boundary; does not expand budget")
    status = commands.add_parser("status", help="CPU verification of journal progress; never starts the model")
    status.add_argument("--experiment", required=True)
    status.add_argument("--run-root", help="optional relocated archive journal")
    analyze = commands.add_parser("analyze", help="CPU-only paired analysis after every frozen case completes")
    analyze.add_argument("--experiment", required=True)
    analyze.add_argument("--run-root", help="optional relocated archive journal")
    analyze.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "budget":
            from event_sae.research.head_rank.protocol import estimate_budget
            result = estimate_budget(args.features, args.tasks, args.cases_per_task)
        elif args.command == "score":
            if not args.execute:
                result = {"status": "not_executed", "new_rollouts": 0,
                          "next": "prepare separate runtime config, validate on current GPU, then add --execute",
                          "model_download": False}
            else:
                from event_sae.research.head_rank.workflow import score_workflow
                result = score_workflow(args.config, args.reference_result, args.event_scores, args.output,
                    feature_set=args.feature_set, full_readouts=args.full_readouts, max_pairs=args.max_pairs,
                    progress=lambda p: print(json.dumps(p), file=sys.stderr, flush=True))
        elif args.command == "plan":
            from event_sae.research.head_rank.workflow import plan_workflow
            if args.preset == "full10":
                for name, wanted in (("features", 16), ("cases_per_task", 20),
                                     ("selection_strategy", "topk_union")):
                    if getattr(args, name) not in (None, wanted):
                        raise ValueError(f"--preset full10 requires {name}={wanted}")
                    setattr(args, name, wanted)
                from event_sae.research.output_head.provenance import read_json
                scope = read_json(args.scores).get("scope", {})
                if sorted(scope.get("task_ids", [])) != list(range(10)):
                    raise ValueError("--preset full10 requires exactly task IDs 0..9")
            else:
                args.features = 8 if args.features is None else args.features
                args.cases_per_task = 5 if args.cases_per_task is None else args.cases_per_task
                args.selection_strategy = args.selection_strategy or "round_robin"
            result = plan_workflow(args.scores, args.output, registry=args.registry,
                eval_manifest=args.eval_manifest, feature_count=args.features,
                cases_per_task=args.cases_per_task, primary_metric=args.primary_metric,
                selection_strategy=args.selection_strategy, include_excess_topk=args.include_excess_topk,
                seed=args.seed, top_k=args.top_k, require_full10=args.preset == "full10")
        elif args.command == "verify-plan":
            from event_sae.research.head_rank.protocol import verify_plan
            from event_sae.research.output_head.provenance import read_json
            saved = read_json(args.plan)
            verify_plan(saved)
            result = {"status": "verified", "plan_hash": saved["plan_hash"],
                      "runtime_approved": False, "new_rollouts": 0}
        elif args.command == "cases":
            from event_sae.research.head_rank.runtime import select_cases
            result = select_cases(args.split, args.output, cases_per_task=args.cases_per_task, seed=args.seed)
        elif args.command == "snapshot-check":
            from event_sae.research.head_rank.snapshot import create_snapshot_receipt
            result = create_snapshot_receipt(args.config, args.scores, args.reference_result, args.output)
        elif args.command == "prepare-run":
            from event_sae.research.head_rank.runtime import prepare_execution
            result = prepare_execution(args.config, args.plan, args.scores, args.registry, args.eval_manifest,
                args.reference_result, args.split, args.output, args.run_output,
                retry_budget=args.retry_budget, max_rollouts=args.max_rollouts,
                snapshot_receipt_path=args.snapshot_receipt)
        else:
            from event_sae.research.head_rank.runtime import verify_experiment
            from event_sae.research.output_head.provenance import read_json
            saved = read_json(args.experiment)
            verify_experiment(saved)
            if args.command == "run":
                if not args.execute:
                    result = {"status": "not_executed", "execution_hash": saved["execution_hash"],
                        "conditions": saved["conditions"], "total_rollouts": saved["total_rollouts"],
                        "max_attempts": saved["max_attempts"], "new_rollouts": 0,
                        "required": "--execute --approved-execution-hash HASH --expected-code-revision COMMIT"}
                else:
                    from event_sae.research.head_rank.runtime import run_execution
                    result = run_execution(args.experiment, approved_hash=args.approved_execution_hash,
                        expected_revision=args.expected_code_revision, retry_incomplete=args.retry_incomplete,
                        max_new_attempts=args.max_new_attempts)
            elif args.command == "status":
                from event_sae.research.head_rank.execution import inspect_experiment
                result = inspect_experiment(saved, args.run_root or saved["output_root"])
            else:
                from event_sae.research.head_rank.analysis import analyze_experiment
                result = analyze_experiment(args.experiment, args.run_root or saved["output_root"], args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "command": args.command, "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
