"""Offline guards and Event-max binding; no robot/GPU experiment is performed."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from event_sae.research.head_rank.workflow import (
    load_event_max, safe_output, subset_cache, validate_full_score_population,
)
from event_sae.research.output_head.provenance import fingerprint, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def event_fixture(tmp_path):
    scope = {"task_ids": [0, 1], "dictionary_size": 3, "discovery_manifest_hash": "d" * 64,
             "split_manifest_hash": "e" * 64, "sae_sha256": "f" * 64}
    signature = {"schema_version": "test", "scope": scope}
    payload = {"synthetic": False, "scope": scope,
        "matrix_raw": torch.tensor([[9., 1., 0.], [0., 8., 0.], [0., 0., 100.]]),
        "row_keys": [{"task_id": 0, "episode_coverage": .8},
                     {"task_id": 1, "episode_coverage": .7}, {"task_id": 1, "episode_coverage": .1}],
        "derivation": {"operation": "discovery_only_reaggregation", "signature": signature,
                       "signature_hash": fingerprint(signature)}}
    path = tmp_path / "events.pt"
    save_event(path, payload)
    return path, scope, payload


def save_event(path, payload):
    torch.save(payload, path)
    path.with_suffix(".pt.binding.json").write_text(json.dumps({
        "output_sha256": sha256_file(path), "signature_hash": payload["derivation"]["signature_hash"]}))


def test_event_is_max_not_mean_and_filters_low_coverage(tmp_path):
    path, scope, _ = event_fixture(tmp_path)
    values, binding = load_event_max(path, scope)
    assert values == [9., 8., 0.]
    assert binding["canonical_rows"] == 2
    assert binding["aggregation"] == "max_over_canonical_task_event_rows"


@pytest.mark.parametrize("bad", ["hash", "scope", "signature", "public500", "missing_task", "nan", "rebound"])
def test_event_rejects_invalid_or_different_population(tmp_path, bad):
    path, scope, payload = event_fixture(tmp_path)
    if bad == "hash":
        path.write_bytes(b"wrong")
    else:
        if bad == "scope": payload["scope"]["discovery_manifest_hash"] = "a" * 64
        elif bad == "signature": payload["derivation"]["signature"]["extra"] = 1
        elif bad == "public500": payload["derivation"]["operation"] = "public_500"
        elif bad == "missing_task": payload["row_keys"][1]["episode_coverage"] = .1
        elif bad == "nan": payload["matrix_raw"][0, 0] = float("nan")
        elif bad == "rebound":
            payload["derivation"]["signature"] = {"scope": {**scope, "discovery_manifest_hash": "a" * 64}}
            payload["derivation"]["signature_hash"] = fingerprint(payload["derivation"]["signature"])
        save_event(path, payload)
    expected = {**scope, "discovery_manifest_hash": "d" * 64}
    with pytest.raises(ValueError):
        load_event_max(path, expected)


def test_smoke_keeps_complete_queries_and_full_control_pool():
    rows, ids = [], []
    for task in range(2):
        for ep in range(3):
            for step in range(8):
                for dim in range(7):
                    rows.append({"task_id": task, "episode_uid": f"{task}-{ep}",
                                 "step_in_episode": step, "action_dim_index": dim})
                    ids.append(torch.tensor([task * 3 + ep]))
    cache = {"records": rows, "hidden": torch.ones(len(rows), 2), "feature_ids": ids,
             "feature_values": [torch.ones(1) for _ in ids]}
    a, b = subset_cache(cache), subset_cache(cache)
    assert a["records"] == b["records"]
    assert len(a["records"]) == 2 * 8 * 7
    assert a["control_pool_feature_ids"] == list(range(6))
    assert len(cache["records"]) == 2 * 3 * 8 * 7


def test_safe_output_rejects_source_and_symlink_containment(tmp_path):
    source = tmp_path / "frozen"
    source.mkdir()
    alias = tmp_path / "link"
    alias.symlink_to(source)
    with pytest.raises(ValueError): safe_output(source / "new.json", [source])
    with pytest.raises(ValueError): safe_output(alias / "new.json", [source])
    with pytest.raises(ValueError): safe_output(tmp_path, [source])
    output = tmp_path / "new.json"
    assert safe_output(output, [source]) == output
    output.touch()
    with pytest.raises(FileExistsError): safe_output(output, [source])


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/openvla/headrank.py"), *args],
                          text=True, capture_output=True, check=False)


def test_cli_dry_run_does_not_touch_nonexistent_inputs(tmp_path):
    output = tmp_path / "out.json"
    result = cli("score", "--config", "absent.yaml", "--reference-result", "absent",
                 "--event-scores", "absent.pt", "--output", str(output))
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "not_executed"
    assert not output.exists()


def test_cli_budget_counts_controls_once():
    result = cli("budget", "--features", "8", "--tasks", "10", "--cases-per-task", "5")
    assert result.returncode == 0
    assert json.loads(result.stdout)["total_rollouts"] == 500


def test_cli_run_requires_an_explicit_experiment():
    assert cli("run").returncode == 2


@pytest.mark.parametrize("bad", [None, "ids", "query", "count", "episodes", "smoke"])
def test_full_population_cannot_be_partial_or_smoke(bad):
    d = {"schema_version": "head_rank_scores_v1", "sampling_mode": "full",
         "feature_ids": [1, 2], "frozen_active_feature_ids": [1, 2], "scored_feature_ids": [1, 2],
         "feature_scores": [{"feature_id": i, "num_readouts": 7, "num_queries": 1} for i in (1, 2)],
         "scope": {"task_ids": [0]},
         "scoring_sample": {"readouts_used": 7, "discovery_episodes_used": 1,
                            "event_max_matches_scoring_episode_population": True},
         "query_metadata": [{"query_index": 0, "task_id": 0, "episode_uid": "a", "step_in_episode": 0,
                             "num_action_positions": 7}]}
    if bad == "ids": d["feature_scores"].pop()
    elif bad == "query": d["query_metadata"][0]["num_action_positions"] = 6
    elif bad == "count": d["scoring_sample"]["readouts_used"] = 8
    elif bad == "episodes": d["scoring_sample"]["discovery_episodes_used"] = 2
    elif bad == "smoke": d["sampling_mode"] = "smoke"
    if bad:
        with pytest.raises(ValueError): validate_full_score_population(d)
    else:
        validate_full_score_population(d)
