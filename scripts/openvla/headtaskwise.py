#!/usr/bin/env python3
"""CPU-only retrospective taskwise HeadRank comparison; never starts rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scores", "analysis", "experiment", "event-scores", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    try:
        from event_sae.research.head_taskwise import analyze_taskwise
        result = analyze_taskwise(scores_path=args.scores, analysis_path=args.analysis,
            experiment_path=args.experiment, event_path=args.event_scores, output=args.output,
            replicates=args.bootstrap_replicates, seed=args.seed)
    except (ValueError, KeyError, OSError, TypeError, ImportError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc), "new_rollouts": 0}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
