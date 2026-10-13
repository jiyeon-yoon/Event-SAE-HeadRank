"""Guarded live adapter for truthful same-state decoder-direction controls.

The existing native loader/state checks are reused without changing the
frozen head_rank modules. Model loading/rollout require the outer approved
execution contract. A failed norm gate raises SAEHookError and is not a robot
task failure. No assertion is made that a control's own latent was ablated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from event_sae.openvla.intervene import SAEHookError, _load_and_validate_sae
from event_sae.research.head_followup.interventions import decoder_direction
from event_sae.research.head_rank.native import NativeBackend as FrozenNativeBackend
from event_sae.research.output_head.provenance import assert_safe_output, fingerprint, read_json, read_jsonl, sha256_file
from .kernels import matched_edit, validate_norm_policy


class MatchedHook:
    def __init__(self, handle, stream, counters):
        self.handle, self.stream, self.counters = handle, stream, counters
        self.removed = False

    def remove(self):
        if not self.removed:
            self.handle.remove()
            self.stream.close()
            self.removed = True

    def summary(self):
        return copy.deepcopy(self.counters)


def register_matched_hook(model, sae, *, layer_idx, condition, evidence_path,
                          sae_checkpoint_hash, dictionary_size):
    """Install one all-row final-layer hook and retain only norm evidence."""
    mode = condition["mode"]
    fid = condition["feature_id"]
    policy = validate_norm_policy(condition["norm_policy"])
    if mode not in {"identity", "anchor", "matched_decoder"}:
        raise ValueError("non-raw matched hook mode required")
    layers = model.language_model.model.layers
    if layer_idx != len(layers) - 1 or getattr(sae, "training", False):
        raise ValueError("frozen SAE and final decoder layer required")
    direction = None
    direction_binding = {"mode": mode, "anchor_feature_id": fid,
                         "sae_checkpoint_hash": sae_checkpoint_hash,
                         "support": "current_original_forward_anchor_activation",
                         "norm_scope": "per_original_forward_row_same_incoming_hidden"}
    if mode == "matched_decoder":
        cid = condition["control_feature_id"]
        direction = -decoder_direction(sae, cid, dictionary_size,
                                       device=model.language_model.lm_head.weight.device)
        direction_binding.update(control_feature_id=cid, sign="negative_decoder_unit",
            direction_fp32_sha256=hashlib.sha256(direction.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    binding_hash = fingerprint(direction_binding)
    path = Path(evidence_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", encoding="utf-8")
    counters = {"schema_version": "head_matched_hook_v1", "condition_id": condition["condition_id"],
                "mode": mode, "feature_id": fid, "control_feature_id": condition.get("control_feature_id"),
                "alpha": condition["alpha"], "norm_policy": policy, "num_forwards": 0, "num_rows": 0,
                "active_rows": 0, "edited_rows": 0, "norm_invalid_rows": 0, "off_support_rows": 0,
                "max_pair_norm_abs_error": 0., "max_pair_norm_rel_error": 0.,
                "max_anchor_cast_abs_error": 0., "max_control_cast_abs_error": 0.,
                "direction_binding": direction_binding, "direction_binding_hash": binding_hash,
                "encoder_grouping": "original_forward", "cross_trajectory_norm_match": False,
                "control_is_feature_ablation": False}

    @torch.inference_mode()
    def hook(module, inputs, output):
        del module, inputs
        try:
            context = getattr(model, "_sae_hook_context", None)
            keys = ("task_id", "task_episode_idx", "step_in_episode")
            if not isinstance(context, dict) or any(type(context.get(k)) is not int or context[k] < 0 for k in keys):
                raise ValueError("complete current task/trial/step context required")
            if (not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor)
                    or output[0].ndim != 3 or output[0].shape[0] != 1 or output[0].dtype != torch.bfloat16):
                raise ValueError("BF16 batch-one decoder output tuple required")
            hidden = output[0]
            flat = hidden.reshape(-1, hidden.shape[-1])
            result, metrics = matched_edit(flat, sae, fid, mode=mode, control_direction=direction, norm_policy=policy)
            counters["num_forwards"] += 1
            counters["num_rows"] += len(flat)
            counters["active_rows"] += int(metrics["anchor_active"].sum())
            counters["edited_rows"] += int((metrics["control_realized_norm"] > 0).sum())
            invalid = int((~metrics["eligible"]).sum())
            counters["norm_invalid_rows"] += invalid
            counters["off_support_rows"] += int(metrics["off_support"].sum())
            for name in ("pair_norm_abs_error", "pair_norm_rel_error", "anchor_cast_abs_error", "control_cast_abs_error"):
                counters[f"max_{name}"] = max(counters[f"max_{name}"], float(metrics[name].max()))
            evidence = {"schema_version": "head_matched_forward_v1", **{k: context[k] for k in keys},
                        "forward_index": counters["num_forwards"], "condition_id": condition["condition_id"],
                        "mode": mode, "feature_id": fid, "control_feature_id": condition.get("control_feature_id"),
                        "num_rows": len(flat), "direction_binding_hash": binding_hash, "norm_policy": policy,
                        "encoder_grouping": "original_forward", "norm_invalid_rows": invalid,
                        "off_support_rows": int(metrics["off_support"].sum())}
            evidence.update({key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                             for key, value in metrics.items()})
            stream.write(json.dumps(evidence, allow_nan=False) + "\n")
            stream.flush()
            if invalid:
                raise ValueError(f"same-state post-cast norm gate failed for {invalid}/{len(flat)} rows")
            return (result.reshape_as(hidden), *output[1:])
        except Exception as exc:
            if isinstance(exc, SAEHookError):
                raise
            raise SAEHookError(f"Matched direction intervention failed: {exc}") from exc

    try:
        handle = layers[layer_idx].register_forward_hook(hook)
    except Exception:
        stream.close()
        raise
    return MatchedHook(handle, stream, counters)


class NativeBackend(FrozenNativeBackend):
    """Offline loader/state verification inherited; intervention isolated here."""

    def run_case(self, condition, case, output_dir):
        if self._closed:
            raise ValueError("Native backend has been closed")
        normalized = self._case(case)
        if self._validated is None or fingerprint(normalized) not in self._validated:
            raise ValueError("Case was not included in the validated frozen case set")
        mode, fid, alpha = condition.get("mode"), condition.get("feature_id"), condition.get("alpha")
        cid = condition.get("control_feature_id")
        if mode not in {"raw", "identity", "anchor", "matched_decoder"}:
            raise ValueError("Unsupported matched condition")
        if not isinstance(condition.get("condition_id"), str) or not condition["condition_id"]:
            raise ValueError("Native condition needs a condition_id")
        if mode == "raw":
            if fid is not None or cid is not None or isinstance(alpha, bool) or alpha != 1:
                raise ValueError("Raw condition needs feature=None, control=None, alpha=1")
        else:
            if (type(fid) is not int or not 0 <= fid < self.dictionary_size
                    or isinstance(alpha, bool) or alpha != (1 if mode == "identity" else 0)):
                raise ValueError("Matched condition needs a valid anchor and alpha=1/0")
            validate_norm_policy(condition.get("norm_policy"))
            if mode == "matched_decoder":
                if type(cid) is not int or not 0 <= cid < self.dictionary_size or cid == fid:
                    raise ValueError("Control must be a different valid decoder feature")
            elif cid is not None:
                raise ValueError("Only matched_decoder may specify a control feature")
        target = assert_safe_output(output_dir, [self.weights, self.sae_config,
            self.cfg["inputs"]["local_model_snapshot"], self.cfg["inputs"].get("output_head_bundle")])
        native_dir = target / "native"
        if native_dir.exists():
            raise ValueError("Native case directory exists; a fresh attempt is required")
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
            del cfg, log_file
            sae, trainer, _ = _load_and_validate_sae(model=model, layer_idx=self.identity["layer_idx"],
                sae_checkpoint_path=str(self.weights), sae_bundle=self._sae_bundle)
            handle = register_matched_hook(model, sae, layer_idx=self.identity["layer_idx"], condition=condition,
                evidence_path=Path(run_dir) / "matched_records.jsonl",
                sae_checkpoint_hash=self.identity["sae_checkpoint_hash"], dictionary_size=trainer["dict_size"])
            handles.append(handle)
            return [handle]

        try:
            outcome = native.eval_libero(run_cfg, extra_hook_applier=None if mode == "raw" else applier,
                trial_indices_by_task={normalized["task_id"]: [normalized["task_episode_idx"]]},
                expected_initial_states={(normalized["task_id"], normalized["task_episode_idx"]): normalized["initial_state_sha256"]},
                runtime_bundle=bundle)
        finally:
            for handle in handles:
                handle.remove()
            if hasattr(bundle[0], "_sae_hook_context"):
                del bundle[0]._sae_hook_context
        run_dir = Path(outcome.run_dir).resolve()
        if native_dir.resolve() not in run_dir.parents:
            raise ValueError("Native evidence escaped its attempt directory")
        if Path(outcome.episode_results_path).resolve() != run_dir / "episode_results.jsonl":
            raise ValueError("Native episode evidence path mismatch")
        rows = read_jsonl(outcome.episode_results_path)
        if len(rows) != 1 or rows[0].get("caught_exception") is not None:
            raise ValueError("Exceptional/incomplete native rollout is not a task failure")
        row = rows[0]
        if any(row.get(key) != value for key, value in normalized.items()):
            raise ValueError("Native episode does not match the frozen case/seed")
        actions = read_json(run_dir / "actions.json").get(row.get("task_description"), {}).get(str(normalized["task_episode_idx"]))
        if (not isinstance(actions, list) or not actions or row.get("num_actions") != len(actions)
                or not isinstance(row.get("success"), bool)
                or any(not isinstance(vector, list) or len(vector) != 7 or any(isinstance(v, bool)
                    or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector) for vector in actions)):
            raise ValueError("Native action trace malformed/incomplete")
        metrics, evidence = None, None
        if mode != "raw":
            if len(handles) != 1:
                raise ValueError("Exactly one matched hook required")
            metrics = handles[0].summary()
            if (metrics["num_forwards"] <= 0 or metrics["mode"] != mode or metrics["feature_id"] != fid
                    or metrics["control_feature_id"] != cid or metrics["alpha"] != alpha
                    or metrics["norm_invalid_rows"] or metrics["off_support_rows"]):
                raise ValueError("Matched hook missing or norm-gate evidence invalid")
            if mode == "identity" and metrics["edited_rows"]:
                raise ValueError("Identity hook changed hidden rows")
            path = run_dir / "matched_records.jsonl"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Matched hook records missing")
            evidence = {"path": str(path), "sha256": sha256_file(path)}
        return {"episode": row, "actions": actions, "hook_metrics": metrics, "hook_evidence": evidence}


def _check_loaded_runtime(model, sae, head, *, cfg, identity, conditions, fixture, evidence_dir):
    """Actual forward/hook checker separated for CPU fake-model tests only."""
    from event_sae.research.head_followup.interventions import apply_edit
    from event_sae.research.output_head.head import compare_logits

    layers = model.language_model.model.layers
    device = cfg["scoring"]["device"]
    calls, episodes = fixture["calls"], fixture["episodes"]
    # Derive the latent width from the first real forward, never a guessed SAE
    # class attribute. Config was already validated by the outer function.
    from event_sae.research.output_head.workflow import _sae_paths
    _, sae_cfg = _sae_paths(cfg)
    dictionary_size = read_json(sae_cfg)["trainer"]["dict_size"]
    tolerances = {k: cfg["validation"][k] for k in ("atol", "rtol", "max_argmax_mismatch")}
    observations, summaries, action_traces, hook_evidence = [], {}, {}, {}
    for condition in conditions:
        mode, fid = condition["mode"], condition["feature_id"]
        cid = condition.get("control_feature_id")
        name = condition["condition_id"]
        direction = (-decoder_direction(sae, cid, dictionary_size, device=device)
                     if mode == "matched_decoder" else None)
        current, captures, actions = {}, [], []

        @torch.inference_mode()
        def before(_module, _inputs, output):
            h = output[0].detach()
            flat = h.reshape(-1, h.shape[-1])
            z = sae.encode(flat.float())
            if mode == "raw":
                expected, isolated = flat, flat[-1:]
            else:
                expected, _ = apply_edit(flat, sae, fid, condition["alpha"], direction=direction, encoded=z)
                isolated, _ = apply_edit(flat[-1:], sae, fid, condition["alpha"], direction=direction, encoded=z[-1:])
            current.update(expected=expected.reshape_as(h), rows=len(flat),
                           active=mode != "raw" and bool((z[:, fid] > 0).any()),
                           isolated_logits=head(isolated))

        @torch.inference_mode()
        def after(_module, _inputs, output):
            if not current:
                raise ValueError("Missing pre-intervention runtime reference")
            actual = output[0].detach()
            if not torch.equal(current["expected"], actual):
                raise ValueError("Actual matched hook hidden differs from independent original-forward apply_edit reference")
            current["grouped_logits"] = head(current["expected"]).reshape(-1, head.head_weight.shape[0])[-1:]

        @torch.inference_mode()
        def after_head(_module, _inputs, output):
            if "grouped_logits" not in current:
                raise ValueError("Missing post-intervention runtime reference")
            live_logits = output.reshape(-1, output.shape[-1])[-1:]
            grouped = compare_logits(current["grouped_logits"], live_logits, **tolerances)
            isolated = compare_logits(current["isolated_logits"], live_logits, **tolerances)
            row = {"condition_id": name, "mode": mode, "row_count": current["rows"],
                   "active_anchor": current["active"], "hidden_exact_equal": True,
                   "grouped_logits": grouped, "isolated_scoring_logits_diagnostic": isolated,
                   "status": grouped["status"]}
            captures.append(row)
            if grouped["status"] != "passed":
                raise ValueError("Matched runtime grouped logits differ from exported head")
            current.clear()

        pre = layers[-1].register_forward_hook(before)
        edit, post, head_hook = None, None, None
        try:
            if mode != "raw":
                edit = register_matched_hook(model, sae, layer_idx=cfg["scope"]["layer_idx"], condition=condition,
                    evidence_path=Path(evidence_dir) / name / "matched_records.jsonl",
                    sae_checkpoint_hash=identity["sae_checkpoint_hash"], dictionary_size=dictionary_size)
            post = layers[-1].register_forward_hook(after)
            head_hook = model.language_model.lm_head.register_forward_hook(after_head)
            for index, (call, episode) in enumerate(zip(calls, episodes)):
                kwargs = {k: v.to(device=device, dtype=head.hidden_dtype if v.is_floating_point() else v.dtype)
                          for k, v in call.items()}
                model._sae_hook_context = {k: episode[k] for k in ("task_id", "task_episode_idx", "step_in_episode")}
                start = len(captures)
                with torch.inference_mode():
                    action = model.predict_action(**kwargs, unnorm_key=cfg["model"]["unnorm_key"], do_sample=False)
                action_values = action.tolist()
                if len(action_values) != 7 or any(not math.isfinite(float(v)) for v in action_values):
                    raise ValueError("Runtime action output must be a finite seven-dimensional vector")
                actions.append(action_values)
                observed = captures[start:]
                if (len(observed) != 7 or observed[0]["row_count"] <= 1
                        or any(r["row_count"] != 1 for r in observed[1:])):
                    raise ValueError("Matched runtime requires one prefill and six cached forwards per action")
                for row in observed:
                    row["call_index"] = index
            if edit is not None:
                summaries[name] = edit.summary()
                if mode in {"anchor", "matched_decoder"} and not any(r["active_anchor"] for r in captures):
                    raise ValueError("Runtime fixture never activated the planned anchor; cannot certify this condition")
                path = Path(evidence_dir) / name / "matched_records.jsonl"
                hook_evidence[name] = {"path": str(path.relative_to(evidence_dir)), "sha256": sha256_file(path)}
            observations.extend(captures)
            action_traces[name] = actions
        finally:
            pre.remove()
            if edit is not None:
                edit.remove()
            if post is not None:
                post.remove()
            if head_hook is not None:
                head_hook.remove()
            if hasattr(model, "_sae_hook_context"):
                del model._sae_hook_context
    raws = [c["condition_id"] for c in conditions if c["mode"] == "raw"]
    identities = [c["condition_id"] for c in conditions if c["mode"] == "identity"]
    if len(raws) != 1 or not identities or any(action_traces[i] != action_traces[raws[0]] for i in identities):
        raise ValueError("Matched raw/identity cached action parity missing or failed")
    return {"status": "passed", "conditions": [c["condition_id"] for c in conditions],
            "observations": observations, "hook_summaries": summaries, "hook_evidence": hook_evidence,
            "action_trace_hashes": {name: fingerprint(trace) for name, trace in action_traces.items()},
            "raw_identity_action_match": True, "model_action_queries": len(calls) * len(conditions),
            "isolated_scoring_comparison_is_diagnostic_only": True}


def validate_matched_runtime(cfg, conditions, evidence_dir, *, max_model_calls=32):
    """Opt-in real cached-input validation, no LIBERO/new robot rollout.

Caller must require explicit model-execution authorization BEFORE this call.
Existing base parity is mandatory but insufficient: all planned matched
directions are now executed through the real hook. This function returns a
certificate body; the outer workflow binds/writes its signed artifact.
"""
    from event_sae.openvla.activations import load_batch_topk_sae
    from event_sae.research.output_head.head import load_output_head, resolve_dtype
    from event_sae.research.output_head.runtime import _offline_environment, _validate_frozen_split
    from event_sae.research.output_head.workflow import (_head_path, _sae_paths, numerical_identity,
                                                        load_parity, required_input)

    identity = numerical_identity(cfg)
    load_parity(cfg, identity)
    if not isinstance(conditions, list) or not conditions or any(not isinstance(c, dict) for c in conditions):
        raise ValueError("Frozen matched conditions required")
    names = [c.get("condition_id") for c in conditions]
    if any(not isinstance(n, str) or not n or "/" in n or "\\" in n or n in {".", ".."} for n in names) or len(set(names)) != len(names):
        raise ValueError("Unique safe condition names required")
    for c in conditions:
        if c.get("mode") not in {"raw", "identity", "anchor", "matched_decoder"}:
            raise ValueError("Unsupported runtime condition")
        mode, fid, cid = c["mode"], c.get("feature_id"), c.get("control_feature_id")
        if isinstance(c.get("alpha"), bool) or c.get("alpha") != (1 if mode in {"raw", "identity"} else 0):
            raise ValueError("Runtime condition alpha mismatch")
        if mode == "raw" and (fid is not None or cid is not None):
            raise ValueError("Raw runtime condition cannot name a feature")
        if mode != "raw" and (type(fid) is not int or fid < 0):
            raise ValueError("Runtime anchor must be a nonnegative integer")
        if mode == "matched_decoder" and (type(cid) is not int or cid < 0 or cid == fid):
            raise ValueError("Runtime control must differ from its anchor")
        if mode not in {"raw", "matched_decoder"} and cid is not None:
            raise ValueError("Runtime control is only valid for matched_decoder")
        if c["mode"] != "raw":
            validate_norm_policy(c.get("norm_policy"))
    if type(max_model_calls) is not int or max_model_calls < 1:
        raise ValueError("Explicit positive matched validation call budget required")
    fixture_path = required_input(cfg, "runtime_inputs")
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    calls, episodes = fixture.get("calls"), fixture.get("episodes")
    if (fixture.get("synthetic") is not False or not isinstance(calls, list) or not calls
            or not isinstance(episodes, list) or len(calls) != len(episodes)
            or len(calls) * len(conditions) > max_model_calls):
        raise ValueError("Real runtime fixture missing, inconsistent or exceeds matched call budget")
    for key in ("processor_identity", "model_revision", "code_revision"):
        if not identity.get(key) or fixture.get(key) != identity[key]:
            raise ValueError(f"Matched runtime fixture identity mismatch: {key}")
    for call, episode in zip(calls, episodes):
        if (not isinstance(call, dict) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in call.values())
                or "input_ids" not in call or call["input_ids"].ndim != 2 or call["input_ids"].shape[0] != 1
                or ("attention_mask" in call and not bool(call["attention_mask"].all()))):
            raise ValueError("Runtime calls must be finite unpadded batch-one preprocessed tensors")
        if any(type(episode.get(k)) is not int or episode[k] < 0 for k in ("task_id", "task_episode_idx", "step_in_episode")):
            raise ValueError("Runtime fixture requires task/trial/step context")
    _validate_frozen_split(cfg, episodes, stage="pilot")
    generation = read_json(required_input(cfg, "generation_manifest"))
    if generation.get("action_dim") != 7:
        raise ValueError("Matched runtime requires the verified seven-token action mapping")
    target = assert_safe_output(evidence_dir, [fixture_path, required_input(cfg, "local_model_snapshot"), _head_path(cfg)])
    if target.exists():
        raise ValueError("Matched runtime evidence directory must be new")
    target.mkdir(parents=True)
    _offline_environment()
    from transformers import AutoModelForVision2Seq
    device = cfg["scoring"]["device"]
    model = AutoModelForVision2Seq.from_pretrained(str(required_input(cfg, "local_model_snapshot")),
        local_files_only=True, trust_remote_code=True, code_revision=cfg["model"]["code_revision"],
        torch_dtype=resolve_dtype(cfg["model"]["expected_live_hidden_dtype"]),
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    try:
        model.norm_stats = read_json(required_input(cfg, "local_model_snapshot") / "dataset_statistics.json")
        validator = NativeBackend(cfg, identity)
        validator._validate_loaded_model(model)
        weights, _ = _sae_paths(cfg)
        sae, sae_cfg = load_batch_topk_sae(weights, device=device)
        _load_and_validate_sae(model=model, layer_idx=identity["layer_idx"], sae_checkpoint_path=str(weights), sae_bundle=(sae, sae_cfg))
        head = load_output_head(_head_path(cfg), device=device)
        report = _check_loaded_runtime(model, sae, head, cfg=cfg, identity=identity, conditions=conditions,
                                       fixture=fixture, evidence_dir=target)
        from . import kernels
        from event_sae.research.head_followup import interventions
        report.update(schema_version="head_matched_runtime_v1", synthetic=False, identity=identity,
                      conditions_hash=fingerprint(conditions), fixture_sha256=sha256_file(fixture_path),
                      implementation_hashes={"native.py": sha256_file(__file__),
                          "kernels.py": sha256_file(kernels.__file__), "interventions.py": sha256_file(interventions.__file__)},
                      new_rollouts=0, model_execution=True,
                      scope="cached real preprocessed calls; all-row same-state norm gate and hook/head parity; no simulator")
        return report
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
