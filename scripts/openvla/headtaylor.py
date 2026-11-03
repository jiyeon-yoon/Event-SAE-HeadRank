#!/usr/bin/env python3
"""Isolated Taylor scoring and retrospective comparison; never a robot rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="pinned minimal inputs, dry unless --execute")
    download.add_argument("--workspace", default="/workspace/head-taylor-v1")
    download.add_argument("--execute", action="store_true")
    score = commands.add_parser("score", help="new Taylor scores from lossless frozen readouts")
    for name in ("config", "reference-result", "baseline-scores", "output"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--full-readouts", action="store_true", help="all 250 discovery episodes; default is smoke only")
    score.add_argument("--max-readouts", type=int, default=20000)
    score.add_argument("--max-active-pairs", type=int, default=1000000)
    score.add_argument("--benchmark-kl", action="store_true", help="same-device exact-KL timing; smoke only")
    score.add_argument("--execute", action="store_true")
    compare = commands.add_parser("compare", help="CPU only; never combines cohort baselines")
    compare.add_argument("--scores", required=True)
    compare.add_argument("--panel", action="append", required=True, metavar="NAME=JSON")
    compare.add_argument("--output", required=True)
    compare.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            from event_sae.research.head_taylor.inputs import download_inputs
            result = download_inputs(args.workspace, execute=args.execute)
        elif args.command == "score":
            if not args.execute:
                result = {"status": "not_executed", "new_rollouts": 0,
                          "model_download": False, "next": "validate current device, then add --execute"}
            else:
                from event_sae.research.head_taylor.workflow import score_workflow
                result = score_workflow(args.config, args.reference_result, args.baseline_scores, args.output,
                    full_readouts=args.full_readouts, max_readouts=args.max_readouts,
                    max_active_pairs=args.max_active_pairs, benchmark_kl=args.benchmark_kl,
                    progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True))
        else:
            from event_sae.research.head_taylor.comparison import compare_workflow
            panels = []
            for item in args.panel:
                if "=" not in item:
                    raise ValueError("--panel must be NAME=JSON_PATH")
                name, path = item.split("=", 1)
                if not name or not path:
                    raise ValueError("--panel requires a nonempty NAME and JSON_PATH")
                panels.append((name, path))
            result = compare_workflow(args.scores, panels, args.output, top_k=args.top_k)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "command": args.command, "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
