#!/usr/bin/env python3
"""Retrospective norm-matched direction diagnostic, isolated from frozen runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="freeze seven conditions and seeded states, no execution")
    for name in ("scores", "parent-experiment", "output"):
        plan.add_argument("--" + name, required=True)
    plan.add_argument("--anchor", type=int, default=30452)
    plan.add_argument("--cases-per-task", type=int, default=2)
    plan.add_argument("--seed", type=int, default=1729)
    verify = commands.add_parser("verify-plan")
    verify.add_argument("--plan", required=True)
    score = commands.add_parser("score", help="current-device five-direction prediction, no rollout")
    for name in ("config", "plan", "reference-result", "output"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--execute", action="store_true")
    score.add_argument("--max-pairs", type=int, default=100000)
    validate = commands.add_parser("validate", help="actual cached-input hook/head validation; no simulator")
    for name in ("config", "plan", "output"):
        validate.add_argument("--" + name, required=True)
    validate.add_argument("--allow-model-execution", action="store_true")
    validate.add_argument("--max-model-calls", type=int, default=32)
    prepare = commands.add_parser("prepare-run", help="freeze clean code, files, current GPU certificates and budget")
    for name in ("config", "plan", "scores", "validation", "output", "run-output"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--max-rollouts", type=int, default=140)
    prepare.add_argument("--retry-allowance", type=int, default=0)
    run = commands.add_parser("run", help="dry preview unless explicit execution/hash/revision supplied")
    run.add_argument("--experiment", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--approved-execution-hash")
    run.add_argument("--expected-code-revision")
    run.add_argument("--max-new-attempts", type=int)
    run.add_argument("--retry-incomplete", action="store_true")
    status = commands.add_parser("status", help="verify journal progress, CPU only")
    status.add_argument("--experiment", required=True)
    status.add_argument("--run-root", help="relocated archive journal, read only")
    analyze = commands.add_parser("analyze", help="paired five-direction diagnostic, CPU only")
    analyze.add_argument("--experiment", required=True)
    analyze.add_argument("--run-root", help="relocated archive journal, read only")
    analyze.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        from event_sae.research.output_head.provenance import read_json
        if args.command == "plan":
            from event_sae.research.head_matched.protocol import plan_workflow
            result = plan_workflow(args.scores, args.parent_experiment, args.output,
                anchor=args.anchor, cases_per_task=args.cases_per_task, seed=args.seed)
        elif args.command == "verify-plan":
            from event_sae.research.head_matched.protocol import verify_plan
            plan = read_json(args.plan)
            verify_plan(plan)
            result = {"status": "verified_not_runtime_approved", "plan_hash": plan["plan_hash"],
                      "total_rollouts": plan["total_rollouts"], "new_rollouts": 0}
        elif args.command == "score":
            if not args.execute:
                result = {"status": "not_executed", "new_rollouts": 0, "required": "--execute"}
            else:
                from event_sae.research.head_matched.workflow import score_workflow
                result = score_workflow(args.config, args.plan, args.reference_result, args.output,
                    max_pairs=args.max_pairs,
                    progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True))
        elif args.command == "validate":
            if not args.allow_model_execution:
                result = {"status": "not_executed", "new_rollouts": 0, "required": "--allow-model-execution"}
            else:
                from event_sae.research.head_matched.workflow import validate_workflow
                result = validate_workflow(args.config, args.plan, args.output, max_model_calls=args.max_model_calls)
        elif args.command == "prepare-run":
            from event_sae.research.head_matched.runtime import prepare_execution
            result = prepare_execution(args.config, args.plan, args.scores, args.validation,
                args.output, args.run_output, max_rollouts=args.max_rollouts, retry_allowance=args.retry_allowance)
        else:
            from event_sae.research.head_matched.runtime import verify_experiment
            experiment = read_json(args.experiment)
            verify_experiment(experiment)
            if args.command == "run":
                if not args.execute:
                    result = {"status": "not_executed", "execution_hash": experiment["execution_hash"],
                              "code_revision": experiment["code_identity"]["commit"],
                              "conditions": experiment["conditions"], "total_rollouts": experiment["total_rollouts"],
                              "max_attempts": experiment["max_attempts"], "new_rollouts": 0,
                              "required": "--execute --approved-execution-hash HASH --expected-code-revision COMMIT"}
                else:
                    from event_sae.research.head_matched.runtime import run_execution
                    result = run_execution(args.experiment, approved_hash=args.approved_execution_hash,
                        expected_revision=args.expected_code_revision, retry_incomplete=args.retry_incomplete,
                        max_new_attempts=args.max_new_attempts)
            elif args.command == "status":
                from event_sae.research.head_matched.journal import inspect
                result = inspect(experiment, args.run_root or experiment["output_root"])
            else:
                from event_sae.research.head_matched.analysis import analyze
                result = analyze(experiment, args.run_root or experiment["output_root"], args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "command": args.command, "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
