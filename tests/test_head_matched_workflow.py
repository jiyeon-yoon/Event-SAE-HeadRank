from copy import deepcopy
import os
from pathlib import Path

import pytest

from event_sae.research.output_head.provenance import atomic_write_json, fingerprint
from event_sae.research.head_matched import workflow as w
from test_head_matched_protocol import plan_fixture, seal


def runtime_fixture(monkeypatch, tmp_path):
    from event_sae.research.output_head import workflow as base
    plan = plan_fixture(monkeypatch)
    source = tmp_path / "archived-score.json"
    atomic_write_json(source, {"archive": True})
    plan["source_bindings"] = {"scores": w.binding(source)}
    snapshot = tmp_path / "snapshot"
    head = tmp_path / "head"
    sae = tmp_path / "sae"
    validation = tmp_path / "runtime/validation"
    for folder in (snapshot, head, sae, validation):
        folder.mkdir(parents=True)
    paths = {"base_eval_config": tmp_path / "base.yaml", "sae": sae / "ae.pt",
             "sae_config": sae / "config.json", "head_weights": head / "output_head.safetensors",
             "head_manifest": head / "output_head_manifest.json", "generation_manifest": tmp_path / "generation.json"}
    paths.update({k: snapshot / k.split("/", 1)[1] for k in plan["parent_content_hashes"] if k.startswith("snapshot/")})
    for name, path in paths.items():
        atomic_write_json(path, {"name": name})
    plan["parent_content_hashes"] = {k: w.binding(path)["sha256"] for k, path in paths.items()}
    seal(plan, "plan_hash")
    fixture, config = tmp_path / "inputs.pt", tmp_path / "config.yaml"
    atomic_write_json(fixture, {"fixture": True})
    atomic_write_json(config, {"config": True})
    for name in ("runtime_parity.json", "edit_parity.json"):
        atomic_write_json(validation / name, {"parity": True})
    cfg = {"inputs": {"local_model_snapshot": str(snapshot), "generation_manifest": str(paths["generation_manifest"]),
                      "runtime_inputs": str(fixture)}, "output": {"root_dir": str(validation.parent)},
           "rollout": {"base_eval_config": str(paths["base_eval_config"])}}
    monkeypatch.setattr(base, "_sae_paths", lambda _: (paths["sae"], paths["sae_config"]))
    monkeypatch.setattr(base, "_head_path", lambda _: head)
    return plan, cfg, config, paths


def test_content_binding_accepts_mtime_only_move(monkeypatch, tmp_path):
    plan, cfg, config, paths = runtime_fixture(monkeypatch, tmp_path)
    sources = w.check_runtime_bindings(cfg, plan, config)
    shard = paths["snapshot/model-00001-of-00004.safetensors"]
    os.utime(shard, (7, 7))
    assert sources == w.check_runtime_bindings(cfg, plan, config)
    assert "runtime_inputs" in sources and "head_parity" in sources


def test_content_binding_rejects_weight_and_source_changes(monkeypatch, tmp_path):
    plan, cfg, config, paths = runtime_fixture(monkeypatch, tmp_path)
    atomic_write_json(paths["snapshot/model-00001-of-00004.safetensors"], {"changed": True}, overwrite=True)
    with pytest.raises(ValueError, match="parent content"):
        w.check_runtime_bindings(cfg, plan, config)
    with pytest.raises(ValueError, match="changed"):
        w.check_bindings({"x": {"path": str(config), "sha256": "1" * 64}})


def test_missing_runtime_parity_is_not_silently_assumed(monkeypatch, tmp_path):
    plan, cfg, config, _ = runtime_fixture(monkeypatch, tmp_path)
    # Only this test's small temporary diagnostic file is removed.
    (Path(cfg["output"]["root_dir"]) / "validation/edit_parity.json").unlink()
    with pytest.raises(ValueError, match="Missing local input"):
        w.check_runtime_bindings(cfg, plan, config)


def test_semantic_identity_must_match_even_if_gpu_changes():
    cfg = {"scoring": {"alpha": 0, "pair_batch_size": 1, "reduction_dtype": "float32", "negative_tolerance": 1e-6}}
    old = {"model_revision": "frozen", "torch_version": "old", "device": "cuda:0"}
    current = {**old, "torch_version": "new", "device": "cuda:1"}
    w._check_settings(cfg, current, {"source_identity": old})
    current["model_revision"] = "wrong"
    with pytest.raises(ValueError, match="model_revision"):
        w._check_settings(cfg, current, {"source_identity": old})


def test_reference_cache_file_hashes_and_records(monkeypatch, tmp_path):
    from event_sae.research.output_head import readouts
    root = tmp_path / "reference"
    root.mkdir()
    scope = {"sample_manifest_hash": "sample", "sae_sha256": "sae", "dictionary_size": 8}
    manifest = root / "readouts/manifest.json"
    atomic_write_json(manifest, {"manifest": True})
    records = [{"record": "frozen"}]
    atomic_write_json(root / "sample_manifest.json", {"sample_hash": "sample", "readouts": records})
    atomic_write_json(root / "scores/scores.json", {"scope": scope,
        "identity": {"cache_manifest_hash": w.binding(manifest)["sha256"], "head_weights_hash": "head"}})
    sources = {"source_scores_sha256": root / "scores/scores.json", "source_sample_sha256": root / "sample_manifest.json",
               "source_cache_manifest_sha256": manifest}
    plan = {"source_scope": scope, "source_reference": {k: w.binding(p)["sha256"] for k, p in sources.items()}}
    cache = {"synthetic": False, "records": records, "dict_size": 8}
    monkeypatch.setattr(readouts, "load_readout_cache", lambda _: cache)
    identity = {"sae_checkpoint_hash": "sae", "head_weights_hash": "head"}
    assert w._reference_cache(root, plan, identity)[0] == cache
    cache["synthetic"] = True
    with pytest.raises(ValueError, match="Real exact"):
        w._reference_cache(root, plan, identity)
    cache["synthetic"] = False
    atomic_write_json(manifest, {"manifest": False}, overwrite=True)
    with pytest.raises(ValueError, match="changed"):
        w._reference_cache(root, plan, identity)


def test_implementation_identity_covers_new_cli_kernel_and_frozen_scorer():
    identity = w.implementation_identity()
    assert "event_sae/research/head_matched/kernels.py" in identity["files"]
    assert "scripts/openvla/headmatched.py" in identity["files"]
    assert "event_sae/research/head_rank/scoring.py" in identity["frozen_head_rank"]["files"]
