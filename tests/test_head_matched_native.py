"""Real hook/runner with CPU fake model and fake LIBERO, no live approval."""
import json

import pytest
import torch

from test_head_rank_native import fixture_runtime  # Reuse independent existing simulator fixture.
from event_sae.openvla.intervene import SAEHookError
from event_sae.research.head_matched.native import NativeBackend, register_matched_hook


POLICY = {"atol": 1e-5, "rtol": .05}


def condition(mode, control=None):
    fid = None if mode == "raw" else 0
    return {"condition_id": "raw" if mode == "raw" else f"{mode}-0-{control}", "mode": mode,
            "feature_id": fid, "control_feature_id": control, "alpha": 1 if mode in {"raw", "identity"} else 0,
            "norm_policy": POLICY}


def test_reuses_policy_runs_real_hook_and_truthful_direction_evidence(fixture_runtime):
    f = fixture_runtime
    backend = NativeBackend(f.cfg, f.identity)
    backend.validate_cases(f.cases)
    raw = backend.run_case(condition("raw"), f.cases[0], f.root / "matched-raw")
    identity = backend.run_case(condition("identity"), f.cases[0], f.root / "matched-identity")
    anchor = backend.run_case(condition("anchor"), f.cases[0], f.root / "matched-anchor")
    control = backend.run_case(condition("matched_decoder", 1), f.cases[0], f.root / "matched-control")
    assert len(f.loads) == len(f.processors) == len(f.sae_loads) == 1
    assert raw["actions"] == identity["actions"] != anchor["actions"]
    assert raw["hook_metrics"] is raw["hook_evidence"] is None
    metrics = control["hook_metrics"]
    assert metrics["num_forwards"] == 7 and metrics["num_rows"] == 9
    assert metrics["norm_invalid_rows"] == metrics["off_support_rows"] == 0
    assert metrics["mode"] == "matched_decoder" and metrics["control_feature_id"] == 1
    assert metrics["control_is_feature_ablation"] is False
    assert "post_intervention_active_feature_values" not in metrics
    assert metrics["direction_binding"]["sign"] == "negative_decoder_unit"
    rows = [json.loads(line) for line in open(control["hook_evidence"]["path"])]
    assert len(rows) == 7
    assert [r["num_rows"] for r in rows] == [3, 1, 1, 1, 1, 1, 1]
    for row in rows:
        assert row["task_id"] == 0 and row["task_episode_idx"] == 0
        assert all(row["eligible"]) and not any(row["off_support"])
        assert len(row["anchor_realized_norm"]) == row["num_rows"]
        assert row["anchor_realized_norm"] == row["control_realized_norm"]
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not hasattr(f.model, "_sae_hook_context")
    backend.close()


def test_norm_violation_fatal_not_robot_failure(fixture_runtime):
    f = fixture_runtime
    backend = NativeBackend(f.cfg, f.identity)
    backend.validate_cases(f.cases)
    value = condition("matched_decoder", 2)
    value["norm_policy"] = {"atol": 0., "rtol": 0.}
    with pytest.raises(SAEHookError, match="norm gate"):
        backend.run_case(value, f.cases[0], f.root / "norm-invalid")
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not hasattr(f.model, "_sae_hook_context")
    assert f.envs[-1].closed == 1


def test_bad_condition_before_weight_load(fixture_runtime):
    f = fixture_runtime
    backend = NativeBackend(f.cfg, f.identity)
    backend.validate_cases(f.cases)
    for bad in (condition("intervention"), condition("matched_decoder", 0), condition("anchor", 1)):
        with pytest.raises(ValueError):
            backend.run_case(bad, f.cases[0], f.root / "never-loaded")
    assert not f.loads


def test_context_required_and_every_forward_records_dtype(fixture_runtime):
    f = fixture_runtime
    sae, _ = f.runtime.load_sae(f.root / "sae/ae.pt", "cpu")
    hook = register_matched_hook(f.model, sae, layer_idx=0, condition=condition("anchor"),
        evidence_path=f.root / "direct.jsonl", sae_checkpoint_hash=f.identity["sae_checkpoint_hash"], dictionary_size=3)
    try:
        with pytest.raises(SAEHookError, match="context"):
            f.model.language_model.model.layers[0](torch.ones(1, 1, 2, dtype=torch.bfloat16))
        f.model._sae_hook_context = {"task_id": 0, "task_episode_idx": 0, "step_in_episode": 0}
        with pytest.raises(SAEHookError, match="BF16"):
            f.model.language_model.model.layers[0](torch.ones(1, 1, 2))
    finally:
        hook.remove()


def test_cached_runtime_hook_and_head_checker_real_forward_cpu_fake(fixture_runtime, monkeypatch):
    from types import MethodType
    import numpy as np
    from event_sae.research.head_matched.native import _check_loaded_runtime
    f = fixture_runtime
    sae, _ = f.runtime.load_sae(f.root / "sae/ae.pt", "cpu")

    def predict(model, **kwargs):
        assert "input_ids" in kwargs
        for index in range(7):
            hidden = torch.ones(1, 3 if index == 0 else 1, 2, dtype=torch.bfloat16)
            edited = model.language_model.model.layers[0](hidden)[0]
            logits = model.language_model.lm_head(model.language_model.model.norm(edited))
        return np.array([float(logits[0, -1, 0]), 0., 0., 0., 0., 0., 1.])

    f.model.predict_action = MethodType(predict, f.model)
    conditions = [condition("raw"), condition("identity"), condition("anchor"), condition("matched_decoder", 1)]
    fixture = {"calls": [{"input_ids": torch.ones(1, 3, dtype=torch.long)}],
               "episodes": [{"task_id": 0, "task_episode_idx": 0, "step_in_episode": 0}]}
    report = _check_loaded_runtime(f.model, sae, f.head, cfg=f.cfg, identity=f.identity,
                                  conditions=conditions, fixture=fixture, evidence_dir=f.root / "cached-validation")
    assert report["status"] == "passed" and report["raw_identity_action_match"]
    assert report["model_action_queries"] == 4 and len(report["observations"]) == 28
    assert all(r["hidden_exact_equal"] and r["grouped_logits"]["status"] == "passed" for r in report["observations"])
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not f.model.language_model.lm_head._forward_hooks
    # A live hook error must fail even when its self-reported norm metrics say
    # valid: the independent original apply_edit reference is authoritative.
    import event_sae.research.head_matched.native as matched_native
    original = matched_native.matched_edit

    def corrupt(*args, **kwargs):
        edited, evidence = original(*args, **kwargs)
        return (edited + 1 if kwargs["mode"] == "anchor" else edited), evidence

    monkeypatch.setattr(matched_native, "matched_edit", corrupt)
    with pytest.raises(ValueError, match="independent original-forward"):
        _check_loaded_runtime(f.model, sae, f.head, cfg=f.cfg, identity=f.identity,
            conditions=conditions, fixture=fixture, evidence_dir=f.root / "corrupt-validation")
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not f.model.language_model.lm_head._forward_hooks


def test_cached_runtime_rejects_inactive_anchor_fixture(fixture_runtime):
    from types import MethodType
    import numpy as np
    from event_sae.research.head_matched.native import _check_loaded_runtime
    f = fixture_runtime
    sae, _ = f.runtime.load_sae(f.root / "sae/ae.pt", "cpu")

    def predict(model, **kwargs):
        for index in range(7):
            hidden = torch.zeros(1, 3 if index == 0 else 1, 2, dtype=torch.bfloat16)
            edited = model.language_model.model.layers[0](hidden)[0]
            model.language_model.lm_head(model.language_model.model.norm(edited))
        return np.zeros(7)

    f.model.predict_action = MethodType(predict, f.model)
    fixture = {"calls": [{"input_ids": torch.ones(1, 3, dtype=torch.long)}],
               "episodes": [{"task_id": 0, "task_episode_idx": 0, "step_in_episode": 0}]}
    with pytest.raises(ValueError, match="never activated"):
        _check_loaded_runtime(f.model, sae, f.head, cfg=f.cfg, identity=f.identity,
            conditions=[condition("anchor")], fixture=fixture, evidence_dir=f.root / "inactive-validation")
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not f.model.language_model.lm_head._forward_hooks
