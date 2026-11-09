"""Fresh-process, whole-condition execution of the pinned team Baseline.

No model/simulator imports occur here.  Completed episodes are never skipped
on restart: the caller must create a fresh whole-condition attempt directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from event_sae.research.output_head.provenance import assert_safe_output, atomic_write_json
from .reference import (compare_traces, fingerprint, normalize_cases, sha256_file,
                        validate_condition, validate_episode_rows, verify_baseline)

DRIVER = Path(__file__).resolve().parents[3] / "scripts/openvla/team500_driver.py"


def run_condition(config, condition, ordered_cases, output_dir, baseline_root, reference=False,
                  expected_runtime_identity=None):
    """Execute exactly one condition from seed/model-load/environment start.

    Invocation itself is the internal execution authorization. The public CLI
    must enforce a frozen plan, source revision and explicit approval first.
    """
    condition = validate_condition(condition, reference=reference)
    cases, task_ids, trials = normalize_cases(ordered_cases)
    baseline = Path(baseline_root).expanduser().resolve()
    source = verify_baseline(baseline)
    snapshot = config.get("baseline_config", {}).get("model", {}).get("checkpoint")
    target = assert_safe_output(output_dir, [baseline, snapshot, config.get("sae_checkpoint")])
    if target.exists() and any(target.iterdir()):
        raise ValueError("Whole-condition attempt output must be fresh; never resume by skipping cases")
    target.mkdir(parents=True, exist_ok=True)
    request = {"schema_version": "team500_native_request_v1", "config": config,
               "condition": condition, "cases": cases, "task_ids": task_ids,
               "num_trials_per_task": trials, "reference": bool(reference),
               "baseline_root": str(baseline), "output_dir": str(target),
               "baseline_source": source, "expected_runtime_identity": expected_runtime_identity}
    request["request_hash"] = fingerprint(request)
    request_path, result_path = target / "request.json", target / "native-result.json"
    atomic_write_json(request_path, request)
    env = dict(os.environ)
    env.update(PYTHONHASHSEED="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", PYTHONPATH=str(baseline))
    # Do not inherit a parent research checkout through the working directory.
    with (target / "child.stdout.log").open("x") as log:
        completed = subprocess.run([sys.executable, str(DRIVER), "--request", str(request_path),
                                    "--allow-model-execution"], cwd=baseline, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Pinned Baseline condition failed; no task failure was substituted. "
                           f"Inspect {target / 'child.stdout.log'}")
    if not result_path.is_file():
        raise ValueError("Native process ended without a complete receipt")
    result = json.loads(result_path.read_text())
    if (result.get("status") != "completed" or result.get("request_hash") != request["request_hash"]
            or result.get("condition") != condition
            or result.get("case_manifest_hash") != fingerprint(cases)
            or result.get("exceptions_checked") is not True):
        raise ValueError("Native receipt does not match the approved request")
    validate_episode_rows(result.get("episodes"), cases)
    if source != verify_baseline(baseline):
        raise ValueError("Baseline source changed during condition execution")
    if result.get("source") != source or result.get("driver_sha256") != sha256_file(DRIVER):
        raise ValueError("Native process used a different baseline/driver source")
    return result


__all__ = ["run_condition", "compare_traces", "normalize_cases", "verify_baseline"]
