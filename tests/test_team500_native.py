"""CPU-only contract tests. No CUDA, model weights, or simulator are used."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from event_sae.research.team500 import native, reference


def _driver():
    spec = importlib.util.spec_from_file_location("team500_test_driver", native.DRIVER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cases(count=500):
    return [{"suite": "libero_spatial", "task_id": task, "task_episode_idx": trial,
             "initial_state_sha256": reference.fingerprint([task, trial])}
            for task in range(10) for trial in range(50)][:count]


@pytest.mark.parametrize("count,tasks,trials", [(1, [0], 1), (2, [0], 2), (50, [0], 50),
                                                (100, [0, 1], 50), (500, list(range(10)), 50)])
def test_only_contiguous_stock_baseline_prefix(count, tasks, trials):
    normalized, actual_tasks, actual_trials = reference.normalize_cases(cases(count))
    assert actual_tasks == tasks and actual_trials == trials and len(normalized) == count


def test_casewise_or_cross_task_shortcuts_are_rejected():
    for data in (cases(52), list(reversed(cases(5))), cases(500)[1:],
                 [row for row in cases() if row["task_episode_idx"] < 2]):
        with pytest.raises(ValueError):
            reference.normalize_cases(data)
    hashed = cases(2)
    hashed[0]["seed"] = 3905563605
    with pytest.raises(ValueError, match="hashed"):
        reference.normalize_cases(hashed)


def test_condition_contract_and_reference_is_raw_only():
    raw = {"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1.0}
    identity = {"condition_id": "identity-7024", "mode": "identity", "feature_id": 7024, "alpha": 1.0}
    assert reference.validate_condition(raw, reference=True) == raw
    assert reference.validate_condition(identity) == identity
    for wrong in ({**identity, "alpha": 0}, {**raw, "feature_id": 0}, {**identity, "feature_id": True}):
        with pytest.raises(ValueError):
            reference.validate_condition(wrong)
    with pytest.raises(ValueError):
        reference.validate_condition(identity, reference=True)


def test_reference_manifest_is_bound_to_actual_original_sources():
    # Read-only inspection of the actual local pinned reference, not CUDA.
    root = Path(__file__).resolve().parents[2] / "Event-SAE-Baseline"
    if not root.exists():
        pytest.skip("separate Baseline checkout is not present")
    result = reference.verify_baseline(root)
    assert result["revision"] == reference.BASELINE_REVISION
    assert len(result["source_files"]) >= 10
    assert result["seed_protocol"] == "stock_baseline_global0_once_before_model_load_v1"


def test_baseline_source_mutation_is_rejected(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[2] / "Event-SAE-Baseline"
    if not original.exists():
        pytest.skip("separate Baseline checkout is not present")
    manifest = json.loads(reference.REFERENCE_MANIFEST.read_text())
    for name in manifest["files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    monkeypatch.setattr(reference.subprocess, "check_output", lambda *a, **k: reference.BASELINE_REVISION + "\n")
    reference.verify_baseline(tmp_path)
    (tmp_path / "event_sae/openvla/eval/runner.py").write_text("changed source")
    with pytest.raises(ValueError, match="source differs"):
        reference.verify_baseline(tmp_path)


class FakeEnv:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append("reset")
        return None

    def set_init_state(self, state):
        self.calls.append(("state", state))
        return {"synthetic": True}

    def seed(self, value):
        self.calls.append(("seed", value))


def observed_loop(count=100, extra_seed=False, omit_state=False):
    driver = _driver()
    selected, tasks, trials = reference.normalize_cases(cases(count))
    created = []

    def make_env(task, resolution=256):
        assert resolution == 256
        env = FakeEnv()
        env.seed(0)  # What the byte-pinned original helper does.
        created.append(env)
        return env, f"task{task}"

    fake_model = SimpleNamespace(language_model=SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        model=SimpleNamespace(layers=list(range(32))),
        lm_head=SimpleNamespace(weight=SimpleNamespace(dtype="torch.bfloat16", shape=(32064, 4096)))),
        norm_stats={"libero_spatial": {}}, training=False)
    runner = SimpleNamespace(set_seed_everywhere=lambda seed: None,
                             load_model=lambda cfg: fake_model, get_libero_env=make_env,
                             resolve_task_ids=lambda selected, count: selected,
                             _initial_state_sha256=lambda state: reference.fingerprint(state))
    finish, models = driver.install_observers(runner, selected, tasks, trials, reference,
                                              "sdpa", "libero_spatial")
    cfg = SimpleNamespace(env=SimpleNamespace(task_suite_name="libero_spatial"))
    runner.set_seed_everywhere(0)
    runner.load_model(cfg)
    assert runner.resolve_task_ids(tasks, 10) == tasks
    for task in tasks:
        env, _ = runner.get_libero_env(task, resolution=256)
        for trial in range(trials):
            if extra_seed:
                env.seed(0)
            env.reset()
            if not omit_state:
                env.set_init_state([task, trial])
    return finish, models, created


def test_observed_stock_lifecycle_preserves_one_environment_per_task():
    finish, models, environments = observed_loop(100)
    lifecycle = finish()
    assert len(models) == 1 and len(environments) == 2
    assert lifecycle["events"][0] == {"event": "global_seed", "value": 0}
    assert sum(e["event"] == "reset" for e in lifecycle["events"]) == 100
    assert all(env.calls[0] == ("seed", 0) for env in environments)
    assert all(sum(call == "reset" for call in env.calls) == 50 for env in environments)


def test_observer_rejects_episode_seed_and_incomplete_lifecycle():
    with pytest.raises(ValueError, match="Episode-level"):
        observed_loop(2, extra_seed=True)
    finish, _, _ = observed_loop(2, omit_state=True)
    with pytest.raises(ValueError, match="lifecycle count"):
        finish()


def receipt():
    row = {**cases(1)[0], "success": True, "num_actions": 1,
           "actions": [[0, 0, 0, 0, 0, 0, 1]], "caught_exception": None}
    row.pop("suite")
    return {"runtime_identity": {"fixture_only": True}, "episodes": [row],
            "case_manifest_hash": reference.fingerprint(cases(1))}


def test_trace_parity_requires_every_action_value_not_only_sr():
    a, b = receipt(), receipt()
    assert reference.compare_traces(a, b)["status"] == "passed"
    b["episodes"][0]["actions"][0][0] = 1e-10
    with pytest.raises(ValueError, match="Exact stored"):
        reference.compare_traces(a, b)
    b = receipt()
    b["runtime_identity"]["fixture_only"] = False
    with pytest.raises(ValueError, match="runtime"):
        reference.compare_traces(a, b)


def test_exception_is_not_a_task_failure():
    value = receipt()["episodes"]
    reference.validate_episode_rows(value, cases(1))
    value[0]["caught_exception"] = "synthetic setup failure"
    value[0]["success"] = False
    with pytest.raises(ValueError, match="Exceptional"):
        reference.validate_episode_rows(value, cases(1))


def test_child_dry_run_does_not_import_model_or_open_request(capsys):
    assert _driver().main(["--request", "/nonexistent/request.json"]) == 0
    assert json.loads(capsys.readouterr().out)["model_execution"] is False


def test_condition_restart_refuses_existing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "verify_baseline", lambda root: {"fixture": True})
    target = tmp_path / "attempt"
    target.mkdir()
    (target / "old-partial.json").write_text("{}")
    cfg = {"baseline_config": {"model": {"checkpoint": str(tmp_path / "snapshot")}},
           "sae_checkpoint": str(tmp_path / "ae.pt")}
    raw = {"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1}
    with pytest.raises(ValueError, match="never resume"):
        native.run_condition(cfg, raw, cases(2), target, tmp_path / "baseline")


def test_runtime_mismatch_stops_before_first_environment(monkeypatch):
    driver = _driver()
    created = []
    runner = SimpleNamespace(get_processor=lambda cfg: object(),
                             get_libero_env=lambda: created.append(True))
    actual = {"gpu": "fixture-GPU-A", "package_versions": {"torch": "fixture"}}
    monkeypatch.setattr(driver, "runtime_identity", lambda *a, **k: actual)
    restore, processors, identities = driver.install_processor_observer(
        runner, [object()], {}, {}, {}, reference, None, {},
        {**actual, "gpu": "fixture-GPU-B"})
    with pytest.raises(ValueError, match="before the first environment"):
        runner.get_processor(None)
    assert created == [] and processors == [] and identities == []
    restore()


def test_static_runtime_identity_can_be_checked_before_weights():
    driver = _driver()
    expected = {"gpu": "A", "package_versions": {"torch": "2"}, "model_class": "loaded.later"}
    driver.assert_runtime_identity(expected, {"gpu": "A", "package_versions": {"torch": "2"}}, subset=True)
    with pytest.raises(ValueError, match="before the first environment"):
        driver.assert_runtime_identity(expected, {"gpu": "B", "package_versions": {"torch": "2"}}, subset=True)


def test_subprocess_startup_is_offline_seeded_and_uses_external_baseline(tmp_path, monkeypatch):
    source = {"fixture": True}
    monkeypatch.setattr(native, "verify_baseline", lambda root: source)
    raw = {"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": 1}
    cfg = {"baseline_config": {"model": {"checkpoint": str(tmp_path / "snapshot")}},
           "sae_checkpoint": str(tmp_path / "ae.pt")}
    baseline = tmp_path / "baseline"
    target = tmp_path / "condition"
    calls = []

    def run(command, *, cwd, env, stdout, stderr, check):
        calls.append((command, cwd, env))
        request = json.loads((target / "request.json").read_text())
        result = {**receipt(), "status": "completed", "synthetic": False,
                  "request_hash": request["request_hash"], "condition": raw,
                  "source": source, "driver_sha256": reference.sha256_file(native.DRIVER),
                  "exceptions_checked": True}
        (target / "native-result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(native.subprocess, "run", run)
    result = native.run_condition(cfg, raw, cases(1), target, baseline,
                                  expected_runtime_identity={"fixture": True})
    assert result["status"] == "completed"
    command, cwd, env = calls[0]
    assert Path(cwd) == baseline.resolve()
    assert env["PYTHONHASHSEED"] == "0" and env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1" and env["PYTHONPATH"] == str(baseline.resolve())
    assert "--allow-model-execution" in command
    request = json.loads((target / "request.json").read_text())
    assert request["expected_runtime_identity"] == {"fixture": True}


def hook_records_fixture(tmp_path):
    condition = {"condition_id": "feature-7024", "mode": "intervention", "feature_id": 7024, "alpha": 0.0}
    episodes = [{"task_id": 0, "task_episode_idx": 0, "num_actions": 2},
                {"task_id": 0, "task_episode_idx": 1, "num_actions": 1}]
    rows = [{"episode_num": 1, "task_id": 0, "task_episode_idx": 0, "step_in_episode": 0,
             "feature_id": 7024, "alpha": 0.0, "first_forward_idx": 1},
            {"episode_num": 1, "task_id": 0, "task_episode_idx": 0, "step_in_episode": 1,
             "feature_id": 7024, "alpha": 0.0, "first_forward_idx": 8},
            {"episode_num": 2, "task_id": 0, "task_episode_idx": 1, "step_in_episode": 0,
             "feature_id": 7024, "alpha": 0.0, "first_forward_idx": 15}]
    file = tmp_path / "intervene_feat7024_alpha0.0_records.jsonl"
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return condition, episodes, rows, file, {"num_environment_steps": 3, "num_forwards": 21, "num_tokens": 870}


def test_hook_records_bind_every_executed_step_and_actual_file_hash(tmp_path):
    condition, episodes, rows, file, metrics = hook_records_fixture(tmp_path)
    result = _driver().collect_hook_evidence(tmp_path, condition, episodes, metrics, reference)
    assert result["records"] == 3 and result["all_executed_policy_steps_covered"]
    assert result["sha256"] == reference.sha256_file(file)
    assert result["layer_idx"] == 31 and result["hook_start_step"] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_target", "wrong_step", "forward_order", "extra"])
def test_missing_duplicate_or_out_of_order_hook_records_block(tmp_path, mutation):
    condition, episodes, rows, file, metrics = hook_records_fixture(tmp_path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "wrong_target":
        rows[0]["feature_id"] = 24830
    elif mutation == "wrong_step":
        rows[0]["step_in_episode"] = 1
    elif mutation == "forward_order":
        rows[1]["first_forward_idx"] = 1
    else:
        rows.append(copy.deepcopy(rows[-1]))
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Hook"):
        _driver().collect_hook_evidence(tmp_path, condition, episodes, metrics, reference)


def test_aggregate_hook_step_mismatch_blocks(tmp_path):
    condition, episodes, rows, file, metrics = hook_records_fixture(tmp_path)
    metrics["num_environment_steps"] = 2
    with pytest.raises(ValueError, match="aggregate coverage"):
        _driver().collect_hook_evidence(tmp_path, condition, episodes, metrics, reference)
