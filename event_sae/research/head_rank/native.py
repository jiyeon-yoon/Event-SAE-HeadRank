"""Offline-only, single-case native LIBERO adapter for the guarded journal.

This module does not approve or schedule a GPU run. Its caller must check the
frozen execution plan, fresh parity, code identity and attempt budget first.
All live initial states are checked before policy weights are loaded. One
policy/processor pair is then reused; each case owns a fresh environment,
original all-row intervention hook, and isolated evidence directory.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from event_sae.openvla.eval.config import load_config as load_eval_config
from event_sae.research.output_head.provenance import (
    assert_safe_output, fingerprint, read_json, read_jsonl, sha256_file,
)


def episode_seed(base_seed: int, task_id: int, trial: int) -> int:
    """Match the native runner's condition-independent episode RNG exactly."""
    for value in (base_seed, task_id, trial):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("seed/task/trial must be nonnegative integers")
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{task_id}:{trial}".encode()).digest()[:4], "big")


def _base_policy(cfg):
    snapshot_value = cfg["inputs"].get("local_model_snapshot")
    if not snapshot_value or not Path(snapshot_value).is_dir():
        raise ValueError("Native execution needs an existing local model snapshot")
    if cfg["model"].get("allow_download") is not False:
        raise ValueError("Native execution is offline only")
    snapshot = Path(snapshot_value).resolve()
    base = load_eval_config(cfg["rollout"]["base_eval_config"])
    if (base.model.family != "openvla" or base.model.revision != cfg["model"]["revision"]
            or base.model.code_revision != cfg["model"]["code_revision"]):
        raise ValueError("Base eval model family/revisions differ from the sensitivity protocol")
    if not base.model.revision or not base.model.code_revision:
        raise ValueError("Native execution requires pinned model and code revisions")
    if (base.model.load_in_8bit or base.model.load_in_4bit
            or cfg["model"]["expected_live_hidden_dtype"] != "bfloat16"):
        raise ValueError("Native research rollout requires unquantized BF16")
    if (cfg["scope"]["model_family"] != "openvla" or cfg["scope"]["suite"] != "libero_spatial"
            or cfg["scope"]["capture_target"] != "post_mlp_residual"
            or cfg["scope"]["require_final_decoder_layer"] is not True):
        raise ValueError("Native adapter only supports final-layer OpenVLA Spatial residual edits")
    if cfg["rollout"]["hook_start_step"] != 0 or cfg["rollout"]["alpha"] != 0:
        raise ValueError("Native feature conditions require all-step complete removal")
    if (cfg["rollout"].get("collect_activations") is not False
            or cfg["rollout"].get("save_trajectory_records") is not False):
        raise ValueError("Native prediction validation does not recollect activations/trajectories")
    episode_seed(cfg["rollout"]["env_seed"], 0, 0)  # Type/range check before imports.
    base.model.checkpoint = str(snapshot)
    base.model.attn_implementation = "sdpa"
    base.model.local_files_only = True
    base.env.task_suite_name = cfg["scope"]["suite"]
    base.env.seed = cfg["rollout"]["env_seed"]
    base.env.per_episode_seed = True
    base.sae_collect.enabled = False
    base.logging.save_video = bool(cfg["rollout"]["save_video"])
    base.logging.save_trajectory_records = False
    base.logging.save_prompt_records = True
    base.logging.save_actions = True
    statistics = read_json(snapshot / "dataset_statistics.json")
    effective_unnorm = (base.env.task_suite_name if base.env.task_suite_name in statistics
                        else f"{base.env.task_suite_name}_no_noops")
    if effective_unnorm != cfg["model"]["unnorm_key"] or effective_unnorm not in statistics:
        raise ValueError("Native rollout unnorm key differs from the sensitivity protocol")
    return base


def _policy_dict(base):
    result = asdict(base)
    result.pop("logging")
    result.pop("sae_collect")
    for name in ("task_ids", "trial_indices", "num_trials_per_task"):
        result["env"].pop(name)
    return result


def effective_policy(cfg):
    """Policy identity for the outer execution contract; no model/simulator load."""
    return _policy_dict(_base_policy(cfg))


def _runtime():
    """Import native dependencies only after execution-side preflight."""
    from libero.libero import benchmark
    from event_sae.openvla.eval import model, runner, utils
    from event_sae.openvla.activations import load_batch_topk_sae
    from event_sae.openvla.intervene import apply_resid_post_feature_perturb_hook
    return SimpleNamespace(
        make_suite=lambda name: benchmark.get_benchmark_dict()[name](),
        hash_state=runner._initial_state_sha256, eval_libero=runner.eval_libero,
        load_model=model.load_model, get_processor=model.get_processor,
        load_sae=load_batch_topk_sae,
        set_seed=utils.set_seed_everywhere, device=str(utils.DEVICE),
        apply_hook=apply_resid_post_feature_perturb_hook,
    )


class NativeBackend:
    """Caller-owned reusable policy; ``close`` releases it after the run."""

    def __init__(self, cfg, identity):
        from event_sae.research.output_head.workflow import _sae_paths
        self.cfg, self.identity = copy.deepcopy(cfg), copy.deepcopy(identity)
        self.base = _base_policy(self.cfg)
        self.effective_policy = _policy_dict(self.base)
        self.weights, self.sae_config = _sae_paths(self.cfg)
        trainer = read_json(self.sae_config).get("trainer", {})
        self.dictionary_size = trainer.get("dict_size")
        if (isinstance(self.dictionary_size, bool) or not isinstance(self.dictionary_size, int)
                or self.dictionary_size < 1 or trainer.get("layer") != cfg["scope"]["layer_idx"]
                or trainer.get("submodule_name") != "post_mlp_residual"):
            raise ValueError("Native SAE configuration does not match the approved layer/dictionary")
        expected = {
            "model_revision": cfg["model"]["revision"], "code_revision": cfg["model"]["code_revision"],
            "model_checkpoint": cfg["model"]["checkpoint"],
            "device": cfg["scoring"]["device"], "hidden_dtype": "bfloat16", "attention_backend": "sdpa",
            "unnorm_key": cfg["model"]["unnorm_key"], "layer_idx": cfg["scope"]["layer_idx"],
            "capture_target": "post_mlp_residual", "encoder_grouping": "original_forward",
            "pair_batch_size": 1,
            "preprocessing": {"center_crop": self.base.model.center_crop,
                              "num_steps_wait": self.base.env.num_steps_wait},
            "sae_checkpoint_hash": sha256_file(self.weights), "sae_config_hash": sha256_file(self.sae_config),
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"Native policy differs from current parity identity: {key}")
        self._native = None
        self._bundle = None
        self._sae_bundle = None
        self._validated = None
        self._closed = False

    def _imports(self):
        if self._closed:
            raise ValueError("Native backend has been closed")
        if self._native is None:
            from event_sae.research.output_head.runtime import _offline_environment
            _offline_environment()
            native = _runtime()
            if native.device != self.cfg["scoring"]["device"]:
                raise ValueError("Native runner device differs from current parity device")
            self._native = native
        return self._native

    def _case(self, case):
        if not isinstance(case, dict) or case.get("suite") != self.cfg["scope"]["suite"]:
            raise ValueError("Case suite differs from the native policy")
        task, trial = case.get("task_id"), case.get("task_episode_idx")
        seed = episode_seed(self.base.env.seed, task, trial)
        if type(case.get("seed")) is not int or case.get("seed") != seed:
            raise ValueError("Case seed differs from SHA256(base_seed:task:trial)")
        digest = case.get("initial_state_sha256")
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Case initial-state SHA256 is missing or invalid")
        return {"suite": case["suite"], "task_id": task, "task_episode_idx": trial,
                "initial_state_sha256": digest, "seed": seed}

    def validate_cases(self, cases):
        """Prove every live initial state before the first model-weight load."""
        if not isinstance(cases, list) or not cases:
            raise ValueError("At least one frozen evaluation case is required")
        normalized = [self._case(case) for case in cases]
        keys = [(case["task_id"], case["task_episode_idx"]) for case in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate native evaluation cases")
        declared = {fingerprint(case): case for case in normalized}
        if self._bundle is not None and declared != self._validated:
            raise ValueError("Cannot change the validated case set after loading the policy")
        native = self._imports()
        suite = native.make_suite(self.cfg["scope"]["suite"])
        for task in sorted({case["task_id"] for case in normalized}):
            if task >= suite.n_tasks:
                raise ValueError("Evaluation task exceeds the live LIBERO task suite")
            states = suite.get_task_init_states(task)
            for case in (case for case in normalized if case["task_id"] == task):
                trial = case["task_episode_idx"]
                if trial >= len(states) or native.hash_state(states[trial]) != case["initial_state_sha256"]:
                    raise ValueError(f"Live initial-state hash mismatch: task={task}, trial={trial}")
        self._validated = declared
        return {"status": "validated", "cases": len(normalized), "model_execution": False}

    def _validate_loaded_model(self, model):
        import torch
        from event_sae.research.output_head.head import load_output_head
        from event_sae.research.output_head.workflow import _head_path
        if bool(getattr(model, "is_loaded_in_8bit", False)) or bool(getattr(model, "is_loaded_in_4bit", False)):
            raise ValueError("Loaded policy is quantized")
        live = model.language_model
        if self.cfg["scope"]["layer_idx"] != len(live.model.layers) - 1:
            raise ValueError("Actual model target is not its final decoder layer")
        if (live.lm_head.weight.dtype != torch.bfloat16 or live.model.norm.weight.dtype != torch.bfloat16
                or str(live.lm_head.weight.device) != self.cfg["scoring"]["device"]):
            raise ValueError("Actual model dtype/device differs from BF16 parity")
        if self.cfg["model"]["unnorm_key"] not in model.norm_stats:
            raise ValueError("Loaded policy lacks the approved unnormalization key")
        head_path = _head_path(self.cfg)
        if head_path.is_file():
            head_path = head_path.parent
        if sha256_file(head_path / "output_head.safetensors") != self.identity.get("head_weights_hash"):
            raise ValueError("Current head weights differ from the runtime certificate")
        head = load_output_head(head_path, device=self.cfg["scoring"]["device"])
        if (live.lm_head.weight.dtype != head.head_weight.dtype or live.model.norm.weight.dtype != head.norm.weight.dtype
                or not torch.equal(live.lm_head.weight, head.head_weight)
                or not torch.equal(live.model.norm.weight, head.norm.weight)
                or float(live.model.norm.variance_epsilon) != head.norm.variance_epsilon
                or (live.lm_head.bias is None) != (head.head_bias is None)
                or (head.head_bias is not None and not torch.equal(live.lm_head.bias, head.head_bias))):
            raise ValueError("Loaded native output head differs from the validated exported head")

    def _load(self):
        native = self._imports()
        if self._validated is None:
            raise ValueError("All live cases must be validated before loading the model")
        if self._bundle is None:
            native.set_seed(self.base.env.seed)
            model = native.load_model(self.base).eval()
            processor = native.get_processor(self.base)
            self._validate_loaded_model(model)
            sae_bundle = native.load_sae(self.weights, device=self.cfg["scoring"]["device"])
            if (not isinstance(sae_bundle, tuple) or len(sae_bundle) != 2
                    or sae_bundle[1] != read_json(self.sae_config)):
                raise ValueError("Loaded cached SAE config differs from its approved local source")
            self._bundle = (model, processor)
            self._sae_bundle = sae_bundle
        return self._bundle

    def run_case(self, condition, case, output_dir):
        if self._closed:
            raise ValueError("Native backend has been closed")
        normalized = self._case(case)
        if self._validated is None or fingerprint(normalized) not in self._validated:
            raise ValueError("Case was not included in the validated frozen case set")
        mode, fid, alpha = condition.get("mode"), condition.get("feature_id"), condition.get("alpha")
        if mode not in {"raw", "identity", "intervention"}:
            raise ValueError("Unsupported native condition")
        if not isinstance(condition.get("condition_id"), str) or not condition["condition_id"]:
            raise ValueError("Native condition needs a condition_id")
        if mode == "raw":
            if fid is not None or isinstance(alpha, bool) or alpha != 1:
                raise ValueError("Raw condition must have no feature and alpha=1")
        elif (isinstance(fid, bool) or not isinstance(fid, int) or not 0 <= fid < self.dictionary_size
              or isinstance(alpha, bool) or alpha != (1 if mode == "identity" else 0)):
            raise ValueError("Identity/intervention must use a valid feature and alpha=1/0")
        target = assert_safe_output(output_dir, [self.weights, self.sae_config,
                                                  self.cfg["inputs"]["local_model_snapshot"],
                                                  self.cfg["inputs"].get("output_head_bundle")])
        native_dir = target / "native"
        if native_dir.exists():
            raise ValueError("Native case directory already exists; journal must create a fresh attempt")
        native_dir.mkdir(parents=True)
        native = self._imports()
        bundle = self._load()
        run_cfg = copy.deepcopy(self.base)
        run_cfg.env.task_ids = [normalized["task_id"]]
        run_cfg.env.trial_indices = [normalized["task_episode_idx"]]
        run_cfg.env.num_trials_per_task = 1
        run_cfg.logging.root_dir = str(native_dir)
        handles = []

        def applier(*, model, cfg, run_dir, log_file):
            handle = native.apply_hook(
                model=model, layer_idx=self.identity["layer_idx"], sae_checkpoint_path=str(self.weights),
                feature_idx=fid, alpha=float(alpha), hook_start_step=0, run_dir=run_dir, log_file=log_file,
                sae_bundle=self._sae_bundle,
            )
            handles.append(handle)
            return [handle]

        try:
            outcome = native.eval_libero(
                run_cfg, extra_hook_applier=None if mode == "raw" else applier,
                trial_indices_by_task={normalized["task_id"]: [normalized["task_episode_idx"]]},
                expected_initial_states={(normalized["task_id"], normalized["task_episode_idx"]):
                                         normalized["initial_state_sha256"]},
                runtime_bundle=bundle,
            )
        finally:
            for handle in handles:
                handle.remove()
            if hasattr(bundle[0], "_sae_hook_context"):
                del bundle[0]._sae_hook_context
        run_dir = Path(outcome.run_dir).resolve()
        if native_dir.resolve() not in run_dir.parents:
            raise ValueError("Native runner returned evidence outside its attempt directory")
        if Path(outcome.episode_results_path).resolve() != run_dir / "episode_results.jsonl":
            raise ValueError("Native episode evidence path differs from its isolated run directory")
        rows = read_jsonl(outcome.episode_results_path)
        if len(rows) != 1 or rows[0].get("caught_exception") is not None:
            raise ValueError("Exceptional/incomplete native rollout cannot count as a task failure")
        row = rows[0]
        if any(row.get(key) != value for key, value in normalized.items()):
            raise ValueError("Native episode does not match the approved case/seed")
        actions_by_task = read_json(run_dir / "actions.json")
        actions = actions_by_task.get(row.get("task_description"), {}).get(str(normalized["task_episode_idx"]))
        if (not isinstance(actions, list) or not actions or row.get("num_actions") != len(actions)
                or not isinstance(row.get("success"), bool)
                or any(not isinstance(vector, list) or len(vector) != 7 or
                       any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                           for value in vector) for vector in actions)):
            raise ValueError("Native action trace is empty, malformed or inconsistent")
        metrics, evidence = None, None
        if handles:
            if len(handles) != 1:
                raise ValueError("Exactly one native SAE intervention hook is required")
            metrics = handles[0].summary()
            if metrics.get("feature_id") != fid or metrics.get("alpha") != alpha or metrics.get("num_forwards", 0) <= 0:
                raise ValueError("Native hook did not execute the approved feature intervention")
            if alpha == 0 and (metrics.get("post_intervention_active_feature_values") != 0
                               or metrics.get("post_intervention_max_feature_activation") != 0):
                raise ValueError("Native hook did not suppress its target latent")
            path = run_dir / f"intervene_feat{fid}_alpha{float(alpha)}_records.jsonl"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Native hook records are missing")
            evidence = {"path": str(path), "sha256": sha256_file(path)}
        return {"episode": row, "actions": actions, "hook_metrics": metrics, "hook_evidence": evidence}

    def close(self):
        loaded_cuda = self._bundle is not None and self.cfg["scoring"]["device"].startswith("cuda")
        self._bundle = None
        self._sae_bundle = None
        self._native = None
        self._closed = True
        if loaded_cuda:
            import torch
            torch.cuda.empty_cache()
