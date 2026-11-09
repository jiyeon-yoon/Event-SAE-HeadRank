#!/usr/bin/env python3
"""All-500, Top-5 Mean/Tail/Event-max study with the stock team Baseline.

Help, setup, planning and status never execute a policy. Long execution requires
a clean reviewed commit, exact plan approval, and the current-device preflight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="explicit pinned existing-input downloads; no GPU")
    download.add_argument("--workspace", default="/workspace")
    download.add_argument("--execute", action="store_true")
    merge = sub.add_parser("merge", help="verify/link existing 500-episode activation inputs")
    merge.add_argument("--workspace", default="/workspace")
    setup = sub.add_parser("setup", help="freeze all500 input/state/runtime identities; no policy")
    setup.add_argument("--workspace", default="/workspace")
    setup.add_argument("--run-root", default="/workspace/head-team500-v1")
    setup.add_argument("--baseline-root", required=True)
    setup.add_argument("--allow-libero", action="store_true")
    reference = sub.add_parser("verify-baseline", help="revision and source-content check only")
    reference.add_argument("--baseline-root", required=True)
    prepare = sub.add_parser("prepare-cache", help="derive full500 readouts from original activations")
    prepare.add_argument("--head-config", required=True)
    prepare.add_argument("--execute", action="store_true")
    score = sub.add_parser("score", help="score Mean/Tail/Event-max using fresh numerical parity")
    score.add_argument("--head-config", required=True)
    score.add_argument("--event-scores", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--max-pairs", type=int, default=10000000)
    score.add_argument("--smoke", action="store_true", help="development only; cannot authorize full500")
    score.add_argument("--execute", action="store_true")
    plan = sub.add_parser("plan", help="freeze actual Top-5 union and exact rollout budget")
    for name in ("scores", "cases", "runtime-config", "output"):
        plan.add_argument("--" + name, required=True)
    check = sub.add_parser("verify-plan")
    check.add_argument("--plan", required=True)
    check.add_argument("--scores", help="same score bytes relocated from the frozen path")
    for command in ("preflight", "run"):
        opt = sub.add_parser(command, help="dry preview unless explicit execution approval is supplied")
        for name in ("plan", "output-root", "baseline-root"):
            opt.add_argument("--" + name, required=True)
        opt.add_argument("--runtime-config")
        opt.add_argument("--scores", help="same score bytes relocated from the frozen path")
        opt.add_argument("--execute", action="store_true")
        opt.add_argument("--approved-plan-hash")
        opt.add_argument("--expected-code-revision")
        if command == "preflight":
            opt.add_argument("--max-rollouts", type=int, default=402)
        else:
            opt.add_argument("--preflight", required=True)
            opt.add_argument("--max-new-conditions", type=int)
            opt.add_argument("--retry-incomplete", action="store_true")
            opt.add_argument("--retry-budget-conditions", type=int, default=0)
    status = sub.add_parser("status", help="verify condition-level completion without GPU")
    status.add_argument("--plan", required=True)
    status.add_argument("--output-root", required=True)
    status.add_argument("--scores")
    analysis = sub.add_parser("analyze", help="paired analysis from the completed stored conditions")
    analysis.add_argument("--plan", required=True)
    analysis.add_argument("--output-root", required=True)
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--scores")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            from event_sae.research.team500.workflow import download_inputs
            result = download_inputs(args.workspace, execute=args.execute)
        elif args.command == "merge":
            from scripts.openvla.headfull import merge
            result = merge(Path(args.workspace))
        elif args.command == "setup":
            from event_sae.research.team500.workflow import setup
            result = setup(args.workspace, args.run_root, args.baseline_root,
                           allow_libero=args.allow_libero)
        elif args.command == "verify-baseline":
            from event_sae.research.team500.native import verify_baseline
            result = {"status": "source_verified_not_gpu_validated",
                      "reference": verify_baseline(args.baseline_root), "model_execution": False}
        elif args.command == "prepare-cache":
            from event_sae.research.team500.workflow import prepare_cache
            result = prepare_cache(args.head_config, execute=args.execute)
        elif args.command == "score":
            if not args.execute:
                result = {"status": "not_executed", "new_rollouts": 0,
                          "next": "Prepare new all500 cache, validate numerical parity, then --execute"}
            else:
                from event_sae.research.team500.scoring import score_workflow
                result = score_workflow(args.head_config, args.event_scores, args.output,
                    smoke=args.smoke, max_pairs=args.max_pairs,
                    progress=lambda value: print(json.dumps(value), file=sys.stderr, flush=True))
        elif args.command == "plan":
            from event_sae.research.team500.workflow import freeze_plan
            result = freeze_plan(args.scores, args.cases, args.runtime_config, args.output)
        elif args.command == "verify-plan":
            from event_sae.research.team500.execution import _load_plan
            value = _load_plan(args.plan, args.scores)
            result = {"status": "verified", "plan_hash": value["plan_hash"],
                      "total_rollouts": value["total_rollouts"], "runtime_approved": False}
        elif args.command in ("preflight", "run"):
            from event_sae.research.team500.execution import preflight, run_experiment
            common = dict(runtime_config_path=args.runtime_config, execute=args.execute,
                          approved_plan_hash=args.approved_plan_hash,
                          expected_code_revision=args.expected_code_revision,
                          scores_path=args.scores)
            if args.command == "preflight":
                result = preflight(args.plan, args.output_root, args.baseline_root,
                                   max_rollouts=args.max_rollouts, **common)
            else:
                result = run_experiment(args.plan, args.output_root, args.baseline_root,
                    preflight_path=args.preflight, retry_incomplete=args.retry_incomplete,
                    retry_budget_conditions=args.retry_budget_conditions,
                    max_new_conditions=args.max_new_conditions, **common)
        elif args.command == "status":
            from event_sae.research.team500.execution import status
            result = status(args.plan, args.output_root, scores_path=args.scores)
        else:
            from event_sae.research.team500.analysis import analyze_experiment
            result = analyze_experiment(args.plan, args.output_root, args.output,
                                        scores_path=args.scores)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, KeyError,
            subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "blocked", "command": args.command, "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
