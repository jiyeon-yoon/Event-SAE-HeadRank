"""CPU-only native adapter tests with a fake simulator and the real SAE hook.

These tests never load OpenVLA, invoke CUDA or certify a real robot run.
"""
import copy
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from event_sae.openvla import intervene
from event_sae.research.head_rank import native
from event_sae.research.output_head.config import DEFAULTS
from event_sae.research.output_head.head import ExactRMSNorm, OutputHeadBundle
from event_sae.research.output_head.provenance import fingerprint, sha256_file


class FakeSAE:
    def encode(self, hidden):
        return torch.cat([hidden.clamp_min(0), torch.zeros(len(hidden), 1)], dim=-1)

    def decode(self, z):
        return z @ torch.tensor([[.5, 0.], [0., .5], [.1, .1]])


class FakeLayer(torch.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.layers = torch.nn.ModuleList([FakeLayer()])
        self.language_model.model.norm = ExactRMSNorm(torch.ones(2, dtype=torch.bfloat16), 1e-6)
        self.language_model.lm_head = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
        self.norm_stats = {"libero_spatial": {"fixture_only": True}}

    def predict_action(self, **kwargs):
        assert kwargs == {"unnorm_key": "libero_spatial", "do_sample": False}
        for index in range(7):
            h = torch.ones((1, 3 if index == 0 else 1, 2), dtype=torch.bfloat16)
            changed = self.language_model.model.layers[0](h)[0]
        return np.array([float(changed[0, -1, 0]), 0., 0., 0., 0., 0., 1.])


@pytest.fixture
def fixture_runtime(tmp_path, monkeypatch):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["scope"]["layer_idx"] = 0
    cfg["scoring"]["device"] = "cpu"
    cfg["model"].update(revision="fixture-model", code_revision="fixture-code")
    cfg["rollout"]["env_seed"] = 17
    snapshot, sae_path, head_path = tmp_path / "snapshot", tmp_path / "sae", tmp_path / "head"
    for path in (snapshot, sae_path, head_path):
        path.mkdir()
    (snapshot / "dataset_statistics.json").write_text(json.dumps({"libero_spatial": {"fixture_only": True}}))
    (sae_path / "ae.pt").write_text("synthetic weights not a loadable checkpoint")
    sae_config = {"trainer": {"dict_class": "BatchTopKSAE", "submodule_name": "post_mlp_residual",
                              "layer": 0, "activation_dim": 2, "dict_size": 3}}
    (sae_path / "config.json").write_text(json.dumps(sae_config))
    (head_path / "output_head.safetensors").write_text("synthetic head fixture")
    policy_path = tmp_path / "base.yaml"
    policy_path.write_text(yaml.safe_dump({
        "model": {"family": "openvla", "revision": "fixture-model", "code_revision": "fixture-code"},
        "env": {"num_steps_wait": 1},
    }))
    cfg["inputs"].update(local_model_snapshot=str(snapshot), sae_checkpoint=str(sae_path / "ae.pt"),
                         output_head_bundle=str(head_path))
    cfg["rollout"]["base_eval_config"] = str(policy_path)
    identity = {"model_revision": "fixture-model", "code_revision": "fixture-code",
                "model_checkpoint": cfg["model"]["checkpoint"],
                "device": "cpu", "hidden_dtype": "bfloat16", "attention_backend": "sdpa",
                "unnorm_key": "libero_spatial", "layer_idx": 0, "capture_target": "post_mlp_residual",
                "encoder_grouping": "original_forward", "pair_batch_size": 1,
                "preprocessing": {"center_crop": True, "num_steps_wait": 1},
                "sae_checkpoint_hash": sha256_file(sae_path / "ae.pt"),
                "sae_config_hash": sha256_file(sae_path / "config.json"),
                "head_weights_hash": sha256_file(head_path / "output_head.safetensors")}
    model = FakeModel()
    live = model.language_model
    head = OutputHeadBundle(live.model.norm.weight, live.lm_head.weight,
                            {"norm": {"implementation": "llama_rms_norm_v1", "eps": 1e-6},
                             "target_layer": 0, "num_layers": 1, "hidden_dtype": "bfloat16"})
    import event_sae.research.output_head.head as head_module
    monkeypatch.setattr(head_module, "load_output_head", lambda *a, **k: head)
    monkeypatch.setattr(intervene, "load_batch_topk_sae", lambda *a, **k: (FakeSAE(), sae_config))
    states = {0: [np.array([1., 2.]), np.array([3., 4.])], 1: [np.array([5., 6.])]}

    class Suite:
        n_tasks = 2

        def get_task_init_states(self, task):
            return states[task]

        def get_task(self, task):
            return task

    envs, loads, processors, sae_loads, handles, opened = [], [], [], [], [], []
    failures = {"reset": False, "action": False, "fatal_hook": False}

    def observation():
        return {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                "robot0_gripper_qpos": np.zeros(2)}

    class Env:
        def __init__(self):
            self.closed = 0
            self.seed_value = None
            self.steps = 0

        def seed(self, value):
            self.seed_value = value

        def reset(self):
            if failures["reset"]:
                raise RuntimeError("fixture reset failure")

        def set_init_state(self, value):
            return observation()

        def step(self, action):
            self.steps += 1
            return observation(), 0, self.steps >= 2, {}

        def close(self):
            self.closed += 1

    def env_factory(task, resolution):
        env = Env()
        envs.append(env)
        return env, f"fixture task {task}"

    def get_action(model, processor, cfg, obs, text, unnorm_key):
        if failures["fatal_hook"]:
            raise intervene.SAEHookError("fixture fatal hook")
        if failures["action"]:
            raise RuntimeError("fixture inference failure")
        return model.predict_action(unnorm_key=unnorm_key, do_sample=False)

    # Import the actual runner against fake external simulator/processor modules.
    # A unique module name avoids replacing any other test's runner import.
    libero = ModuleType("libero")
    submodule = ModuleType("libero.libero")
    submodule.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": Suite})
    libero.libero = submodule
    monkeypatch.setitem(sys.modules, "libero", libero)
    monkeypatch.setitem(sys.modules, "libero.libero", submodule)
    libero_utils = ModuleType("event_sae.openvla.eval.libero_utils")
    for key, value in {"get_libero_dummy_action": lambda: [0.] * 7, "get_libero_env": env_factory,
                       "get_libero_image": lambda obs, resize: np.zeros((2, 2, 3), dtype=np.uint8),
                       "quat2axisangle": lambda quat: np.zeros(3), "save_rollout_video": lambda *a, **k: None}.items():
        setattr(libero_utils, key, value)
    monkeypatch.setitem(sys.modules, libero_utils.__name__, libero_utils)
    model_module = ModuleType("event_sae.openvla.eval.model")
    model_module.get_action = get_action

    def load_model(run_cfg):
        assert run_cfg.model.local_files_only and run_cfg.model.attn_implementation == "sdpa"
        assert run_cfg.model.checkpoint == str(snapshot)
        loads.append(run_cfg)
        return model

    def get_processor(run_cfg):
        assert run_cfg.model.local_files_only
        processors.append(run_cfg)
        return object()

    model_module.load_model, model_module.get_processor = load_model, get_processor
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    source = Path(native.__file__).parents[2] / "openvla/eval/runner.py"
    spec = importlib.util.spec_from_file_location("head_rank_runner_fixture", source)
    runner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, runner)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.tqdm, "tqdm", lambda values: values)
    builtin_open = open

    def recording_open(*args, **kwargs):
        result = builtin_open(*args, **kwargs)
        opened.append(result)
        return result

    monkeypatch.setattr(runner, "open", recording_open, raising=False)

    def apply_hook(**kwargs):
        handle = intervene.apply_resid_post_feature_perturb_hook(**kwargs)
        handles.append(handle)
        return handle

    def load_sae(path, device):
        assert path == sae_path / "ae.pt" and device == "cpu"
        sae_loads.append(str(path))
        return FakeSAE(), copy.deepcopy(sae_config)

    runtime = SimpleNamespace(make_suite=lambda name: Suite(), hash_state=runner._initial_state_sha256,
                              eval_libero=runner.eval_libero, load_model=load_model, get_processor=get_processor,
                              load_sae=load_sae,
                              set_seed=runner.set_seed_everywhere, device="cpu", apply_hook=apply_hook)
    monkeypatch.setattr(native, "_runtime", lambda: runtime)
    cases = [{"suite": "libero_spatial", "task_id": task, "task_episode_idx": trial,
              "initial_state_sha256": runner._initial_state_sha256(value),
              "seed": native.episode_seed(17, task, trial)}
             for task in states for trial, value in enumerate(states[task])]
    return SimpleNamespace(cfg=cfg, identity=identity, model=model, head=head, runtime=runtime,
                           states=states, envs=envs, loads=loads, processors=processors, handles=handles,
                           sae_loads=sae_loads, sae_config=sae_config,
                           failures=failures, cases=cases, runner=runner, opened=opened, root=tmp_path)


def condition(mode, fid=0):
    name = "feature" if mode == "intervention" else mode
    return {"condition_id": mode if mode == "raw" else f"{name}-{fid}", "mode": mode,
            "feature_id": None if mode == "raw" else fid, "alpha": 0 if mode == "intervention" else 1}


def test_cases_verified_before_model_loading_and_bad_hash_blocks_all(fixture_runtime):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    cases = copy.deepcopy(f.cases)
    cases[-1]["initial_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        backend.validate_cases(cases)
    assert f.loads == [] and f.envs == []
    with pytest.raises(ValueError, match="validated frozen"):
        backend.run_case(condition("raw"), f.cases[0], f.root / "not-approved")
    assert f.loads == []


def test_one_model_reused_actual_hook_identity_actions_and_isolated_evidence(fixture_runtime):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    assert backend.validate_cases(f.cases)["model_execution"] is False
    assert f.loads == []
    # The journal owns started.json; the adapter writes in a fresh child.
    target = f.root / "raw"
    target.mkdir()
    (target / "started.json").write_text("{}")
    raw = backend.run_case(condition("raw"), f.cases[0], target)
    identical = backend.run_case(condition("identity"), f.cases[0], f.root / "identity")
    removed = backend.run_case(condition("intervention"), f.cases[0], f.root / "feature")
    assert len(f.loads) == len(f.processors) == 1
    assert len(f.sae_loads) == 1
    assert raw["actions"] == identical["actions"]
    assert raw["actions"] != removed["actions"]
    assert raw["hook_metrics"] is raw["hook_evidence"] is None
    assert identical["hook_metrics"]["num_forwards"] == 7
    assert removed["hook_metrics"]["post_intervention_active_feature_values"] == 0
    assert removed["hook_metrics"]["post_intervention_max_feature_activation"] == 0
    assert removed["hook_metrics"]["num_tokens"] == 9  # Full prefill + six cached positions.
    assert removed["hook_metrics"]["active_feature_values"] == 9
    evidence = removed["hook_evidence"]
    assert evidence["sha256"] == sha256_file(evidence["path"])
    lines = [json.loads(line) for line in Path(evidence["path"]).read_text().splitlines()]
    assert len(lines) == removed["hook_metrics"]["num_environment_steps"] == 1
    assert lines[0]["feature_id"] == 0 and lines[0]["alpha"] == 0
    assert all(env.closed == 1 and env.seed_value == f.cases[0]["seed"] for env in f.envs)
    assert removed["episode"]["seed"] == f.cases[0]["seed"]
    assert all(stream.closed for stream in f.opened)
    assert all(handle._removed and handle._records_file.closed for handle in f.handles)
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not hasattr(f.model, "_sae_hook_context")
    backend.close()
    assert backend._bundle is None
    assert backend._sae_bundle is None
    with pytest.raises(ValueError, match="closed"):
        backend.run_case(condition("raw"), f.cases[0], f.root / "after-close")


def test_native_original_hook_results_pass_complete_journal_and_resume(fixture_runtime):
    from event_sae.research.head_rank.execution import execute_experiment, load_completed_results
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    conditions = [condition("raw"), condition("identity"), condition("intervention")]
    experiment = {"schema_version": "head_rank_execution_v1", "plan_hash": "a" * 64,
                  "conditions": conditions, "cases": f.cases, "total_rollouts": 9, "max_attempts": 9,
                  "code_identity": {"commit": "b" * 40, "dirty": False}, "numerical_identity": f.identity,
                  "policy_config_hash": fingerprint(backend.effective_policy)}
    experiment["execution_hash"] = fingerprint(experiment)
    root = f.root / "complete-journal"
    result = execute_experiment(experiment, root, backend, approved_hash=experiment["execution_hash"])
    assert result["status"] == "completed" and result["completed_rollouts"] == 9
    rows = load_completed_results(experiment, root)
    assert set(rows) == {"raw", "identity-0", "feature-0"}
    assert all(len(receipts) == 3 for receipts in rows.values())
    assert len(f.loads) == 1 and len(f.envs) == 9
    assert all(env.closed == 1 for env in f.envs)
    backend.close()
    # Completed resume never touches even the now-closed native backend.
    resumed = execute_experiment(experiment, root, backend, approved_hash=experiment["execution_hash"])
    assert resumed["new_attempts"] == 0
    assert len(f.loads) == 1


@pytest.mark.parametrize("failure", ["reset", "fatal_hook", "action"])
def test_every_exception_closes_environment_hooks_and_files(fixture_runtime, failure):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    backend.validate_cases(f.cases)
    f.failures[failure] = True
    with pytest.raises((RuntimeError, ValueError)):
        backend.run_case(condition("intervention"), f.cases[0], f.root / f"broken-{failure}")
    assert f.envs[-1].closed == 1
    assert all(stream.closed for stream in f.opened)
    assert all(handle._removed and handle._records_file.closed for handle in f.handles)
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not hasattr(f.model, "_sae_hook_context")


def test_runner_default_loading_and_seed_record_unchanged_policy(fixture_runtime):
    f = fixture_runtime
    base = native._base_policy(f.cfg)
    base.env.task_ids, base.env.trial_indices, base.env.num_trials_per_task = [0], [0], 1
    base.logging.root_dir = str(f.root / "legacy-default")
    result = f.runner.eval_libero(base)
    row = json.loads(Path(result.episode_results_path).read_text())
    assert len(f.loads) == len(f.processors) == 1
    assert row["seed"] == f.cases[0]["seed"]
    assert f.envs[0].closed == 1


def test_effective_policy_is_shared_and_excludes_case_routing_without_native_import(fixture_runtime, monkeypatch):
    f = fixture_runtime
    monkeypatch.setattr(native, "_runtime", lambda: (_ for _ in ()).throw(AssertionError("unexpected import")))
    policy = native.effective_policy(f.cfg)
    backend = native.NativeBackend(f.cfg, f.identity)
    assert backend.effective_policy == policy
    assert policy["env"]["per_episode_seed"] is True
    assert policy["env"]["seed"] == 17
    assert not {"task_ids", "trial_indices", "num_trials_per_task"} & set(policy["env"])
    assert "logging" not in policy and "sae_collect" not in policy
    assert policy["model"]["local_files_only"] is True


@pytest.mark.parametrize("mutate,match", [
    (lambda cfg: cfg["model"].update(allow_download=True), "offline"),
    (lambda cfg: cfg["model"].update(expected_live_hidden_dtype="float32"), "BF16"),
    (lambda cfg: cfg["model"].update(unnorm_key="wrong"), "unnorm"),
    (lambda cfg: cfg["model"].update(revision="wrong"), "revisions"),
    (lambda cfg: cfg["rollout"].update(hook_start_step=1), "all-step"),
    (lambda cfg: cfg["rollout"].update(collect_activations=True), "recollect"),
])
def test_policy_restrictions_before_loading(fixture_runtime, mutate, match):
    f = fixture_runtime
    mutate(f.cfg)
    with pytest.raises(ValueError, match=match):
        native.NativeBackend(f.cfg, f.identity)
    assert f.loads == [] and f.envs == []


def test_actual_head_dtype_and_content_checked_before_environment(fixture_runtime):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    backend.validate_cases(f.cases)
    f.model.language_model.lm_head = f.model.language_model.lm_head.float()
    with pytest.raises(ValueError, match="dtype/device"):
        backend.run_case(condition("raw"), f.cases[0], f.root / "wrong-dtype")
    assert f.envs == []


def test_seed_unknown_case_duplicate_and_invalid_feature_cannot_run(fixture_runtime):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    bad = {**f.cases[0], "seed": f.cases[0]["seed"] + 1}
    with pytest.raises(ValueError, match="Case seed"):
        backend.validate_cases([bad])
    with pytest.raises(ValueError, match="Duplicate"):
        backend.validate_cases([f.cases[0], f.cases[0]])
    backend.validate_cases([f.cases[0]])
    with pytest.raises(ValueError, match="validated frozen"):
        backend.run_case(condition("raw"), f.cases[1], f.root / "unknown-case")
    with pytest.raises(ValueError, match="valid feature"):
        backend.run_case(condition("intervention", 999), f.cases[0], f.root / "bad-feature")
    assert f.loads == []


def test_inactive_feature_still_requires_hook_execution_not_activation(fixture_runtime):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    backend.validate_cases(f.cases)
    result = backend.run_case(condition("intervention", 2), f.cases[0], f.root / "inactive")
    assert result["hook_metrics"]["num_forwards"] == 7
    assert result["hook_metrics"]["active_feature_values"] == 0
    assert result["hook_evidence"] is not None


@pytest.mark.parametrize("alpha", [0., 1.])
def test_cached_sae_hook_matches_original_default_arithmetic(fixture_runtime, monkeypatch, alpha):
    f = fixture_runtime
    f.model._sae_hook_context = {"episode_num": 1, "step_in_episode": 0,
                                 "task_id": 0, "task_episode_idx": 0}
    hidden = torch.tensor([[[1.25, .625], [.25, 1.5]]], dtype=torch.bfloat16)
    outputs, summaries = [], []
    for cached in (False, True):
        directory = f.root / f"hook-{cached}"
        directory.mkdir()
        if cached:
            monkeypatch.setattr(intervene, "load_batch_topk_sae",
                                lambda *a, **k: (_ for _ in ()).throw(AssertionError("SAE reloaded")))
        handle = intervene.apply_resid_post_feature_perturb_hook(
            model=f.model, layer_idx=0, sae_checkpoint_path=f.cfg["inputs"]["sae_checkpoint"],
            feature_idx=0, alpha=alpha, hook_start_step=0, run_dir=str(directory), log_file=io.StringIO(),
            sae_bundle=(FakeSAE(), f.sae_config) if cached else None,
        )
        try:
            outputs.append(f.model.language_model.model.layers[0](hidden.clone())[0])
            summaries.append(handle.summary())
        finally:
            handle.remove()
    assert torch.equal(outputs[0], outputs[1])
    assert summaries[0] == summaries[1]


def test_cached_sae_keeps_original_layer_validation(fixture_runtime):
    f = fixture_runtime
    config = copy.deepcopy(f.sae_config)
    config["trainer"]["layer"] = 1
    with pytest.raises(ValueError, match="checkpoint is layer"):
        intervene.apply_resid_post_feature_perturb_hook(
            model=f.model, layer_idx=0, sae_checkpoint_path=f.cfg["inputs"]["sae_checkpoint"],
            feature_idx=0, alpha=0., hook_start_step=0, run_dir=str(f.root), log_file=io.StringIO(),
            sae_bundle=(FakeSAE(), config),
        )
    assert not f.model.language_model.model.layers[0]._forward_hooks


def test_existing_attempt_cannot_be_overwritten(fixture_runtime):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    backend.validate_cases(f.cases)
    target = f.root / "existing"
    (target / "native").mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        backend.run_case(condition("raw"), f.cases[0], target)
    assert f.loads == []


def test_live_registry_mutation_and_device_mismatch_block(fixture_runtime):
    f = fixture_runtime
    backend = native.NativeBackend(f.cfg, f.identity)
    f.runtime.device = "cuda:0"
    for _ in range(2):
        with pytest.raises(ValueError, match="device"):
            backend.validate_cases(f.cases)
    assert f.loads == []
    f.runtime.device = "cpu"
    backend.validate_cases(f.cases)
    f.states[0][0][0] += 1
    with pytest.raises(ValueError, match="initial-state hash mismatch"):
        backend.run_case(condition("raw"), f.cases[0], f.root / "changed-after-check")
    assert f.envs == []
