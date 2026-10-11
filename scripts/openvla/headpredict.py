#!/usr/bin/env python3
"""Frozen-head descriptors and feature-held-out calibration; never robot rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptors = commands.add_parser("descriptors", help="new output responses on the existing cache; no new rollouts")
    for name in ("config", "reference-result", "mapping", "output"):
        descriptors.add_argument("--" + name, required=True)
    descriptors.add_argument("--feature-set", choices=("development", "all-active"), default="development")
    descriptors.add_argument("--feature-ids", nargs="+", type=int,
                             help="explicit small panel instead of recomputing every active feature")
    descriptors.add_argument("--smoke", action="store_true", help="balanced one-episode-per-task development check only")
    descriptors.add_argument("--max-pairs", type=int, default=800000)
    descriptors.add_argument("--execute", action="store_true")
    calibrate = commands.add_parser("calibrate", help="CPU feature-held-out diagnostics using historical paired outcomes")
    for name in ("scores", "reference-result", "output"):
        calibrate.add_argument("--" + name, required=True)
    calibrate.add_argument("--descriptors")
    calibrate.add_argument("--ridge-alpha", type=float, default=1.0)
    predict = commands.add_parser("predict", help="offline predictions only, excluding training/known feature IDs")
    for name in ("model", "scores", "reference-result", "registry", "output"):
        predict.add_argument("--" + name, required=True)
    predict.add_argument("--descriptors")
    predict.add_argument("--predictor", default="head_ridge")
    predict.add_argument("--feature-ids", nargs="+", type=int,
                         help="freeze predictions only for these previously unevaluated feature IDs")
    attach = commands.add_parser("attach", help="bind predictions into a NEW Tail plan before running it")
    for name in ("plan", "predictions", "model", "output"):
        attach.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "descriptors":
            if args.feature_ids is not None and args.feature_set != "development":
                raise ValueError("Use --feature-ids or --feature-set all-active, not both")
            if not args.execute:
                result = {"status": "not_executed", "new_rollouts": 0, "model_download": False,
                          "feature_set": args.feature_set,
                          "next": "verify action mapping and current-device parity, then add --execute"}
            else:
                from event_sae.research.head_predict.descriptor_workflow import descriptor_workflow
                result = descriptor_workflow(args.config, args.reference_result, args.mapping, args.output,
                    feature_ids=args.feature_ids if args.feature_ids is not None else (
                        "all-active" if args.feature_set == "all-active" else None),
                    full_readouts=not args.smoke, max_pairs=args.max_pairs,
                    progress=lambda p: print(json.dumps(p), file=sys.stderr, flush=True))
        elif args.command == "calibrate":
            from event_sae.research.head_predict.calibration_workflow import calibration_workflow
            result = calibration_workflow(args.scores, args.reference_result, args.output,
                descriptors_path=args.descriptors, ridge_alpha=args.ridge_alpha)
        elif args.command == "predict":
            from event_sae.research.head_predict.calibration_workflow import predict_workflow
            result = predict_workflow(args.model, args.scores, args.reference_result, args.output,
                registry=args.registry, descriptors_path=args.descriptors, predictor=args.predictor,
                feature_ids=args.feature_ids)
        else:
            from event_sae.research.head_predict.evaluation import attach_predictions
            result = attach_predictions(args.plan, args.predictions, args.model, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "command": args.command, "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
