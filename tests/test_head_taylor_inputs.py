import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from event_sae.research.head_taylor import inputs
from event_sae.research.output_head.provenance import fingerprint


def digest(body):
    return hashlib.sha256(body).hexdigest()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    archive, sae, config = b"pinned archive", b"pinned sae", b"pinned model config"
    baseline = {"features": [1, 2], "development_only": False}
    baseline["score_hash"] = fingerprint(baseline)
    baseline_body = json.dumps(baseline).encode()
    new_panel = b'{"feature_effects":[]}'
    for name, body in (("ARCHIVE_SHA256", archive), ("SAE_SHA256", sae),
                       ("BASELINE_SHA256", baseline_body), ("NEW_PANEL_SHA256", new_panel),
                       ("MODEL_CONFIG_SHA256", config)):
        monkeypatch.setattr(inputs, name, digest(body))
    monkeypatch.setattr(inputs, "BASELINE_SCORE_HASH", baseline["score_hash"])
    bodies = {
        (inputs.RESULT_REPO, inputs.RESULT_REVISION, "experiment.tar.gz"): archive,
        (inputs.RESULT_REPO, inputs.RESULT_REVISION, "manifest.json"):
            json.dumps({"archive": {"sha256": digest(archive)}}).encode(),
        (inputs.SAE_REPO, inputs.SAE_REVISION, "trainer_0/ae.pt"): sae,
        (inputs.SAE_REPO, inputs.SAE_REVISION, "trainer_0/config.json"): b"{}",
        (inputs.RESULT_REPO, inputs.BASELINE_REVISION, inputs.BASELINE_FILE): baseline_body,
        (inputs.RESULT_REPO, inputs.NEW_PANEL_REVISION, inputs.NEW_PANEL_FILE): new_panel,
    }
    calls = []
    model = tmp_path / "hub-model"
    model.mkdir()
    (model / "config.json").write_bytes(config)
    (model / "model-00001-of-00001.safetensors").write_bytes(b"model weight")
    (model / "model.safetensors.index.json").write_bytes(b"{}")
    code = tmp_path / "hub-code/models--openvla--openvla-7b/snapshots" / inputs.CODE_REVISION
    code.mkdir(parents=True)
    for name in inputs.CODE_FILES:
        (code / name).write_bytes(b"# pinned code\n")

    def download(**kwargs):
        calls.append(("file", kwargs))
        repo, revision, filename = (kwargs[key] for key in ("repo_id", "revision", "filename"))
        if repo == inputs.CODE_REPO:
            assert revision == inputs.CODE_REVISION
            return str(code / filename)
        body = bodies[(repo, revision, filename)]
        path = tmp_path / "hub-files" / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return str(path)

    def snapshot(**kwargs):
        calls.append(("snapshot", kwargs))
        assert kwargs["repo_id"] == inputs.MODEL_REPO
        assert kwargs["revision"] == inputs.MODEL_REVISION
        return str(model)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        hf_hub_download=download, snapshot_download=snapshot))
    return {"calls": calls, "bodies": bodies, "code": code}


def test_dry_plan_has_no_imports_directories_or_environment_changes(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.delenv("HF_HOME", raising=False)
    target = tmp_path / "not-created"
    result = inputs.download_inputs(target)
    assert result["status"] == "download_plan"
    assert result["dense_dataset_download"] is False
    assert result["new_rollouts"] == 0
    assert not target.exists()
    assert "HF_HOME" not in __import__("os").environ


def test_download_is_pinned_immutable_and_never_restores_or_executes(tmp_path, hub):
    target = tmp_path / "taylor"
    result = inputs.download_inputs(target, execute=True)
    paths = inputs.input_paths(target)
    assert result["status"] == "downloaded"
    assert result["runtime_validation"] == "not_run"
    assert result["restored"] is False
    assert not paths["restored"].exists()
    assert paths["archive"].read_bytes() == b"pinned archive"
    assert paths["sae_checkpoint"].read_bytes() == b"pinned sae"
    assert paths["baseline_scores"].exists()
    assert paths["new_panel_analysis"].exists()
    assert (paths["model"] / "model-00001-of-00001.safetensors").exists()
    assert (hub["code"].parent.parent / "refs/main").read_text().strip() == inputs.CODE_REVISION
    assert all(call[1]["revision"] != "main" for call in hub["calls"])
    assert {call[1]["filename"] for call in hub["calls"] if call[0] == "file" and
            call[1]["repo_id"] == inputs.CODE_REPO} == set(inputs.CODE_FILES)
    assert not list(target.rglob(".taylor-download-*"))
    before = paths["archive"].stat().st_mtime_ns
    assert inputs.download_inputs(target, execute=True)["status"] == "downloaded"
    assert paths["archive"].stat().st_mtime_ns == before


@pytest.mark.parametrize("name,value", [("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "true")])
def test_offline_execution_fails_without_writing(tmp_path, hub, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    root = tmp_path / "blocked"
    with pytest.raises(ValueError, match="offline mode was not changed"):
        inputs.download_inputs(root, execute=True)
    assert not root.exists()
    assert not hub["calls"]


def test_existing_conflicting_file_is_not_overwritten(tmp_path, hub):
    root = tmp_path / "run"
    destination = root / "bundle/experiment.tar.gz"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="Existing input differs"):
        inputs.download_inputs(root, execute=True)
    assert destination.read_bytes() == b"do not overwrite"


def test_wrong_download_hash_does_not_publish_bad_artifact(tmp_path, hub):
    hub["bodies"][(inputs.RESULT_REPO, inputs.RESULT_REVISION, "experiment.tar.gz")] = b"tampered"
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inputs.download_inputs(root, execute=True)
    assert not (root / "bundle/experiment.tar.gz").exists()


def test_symlinked_destination_cannot_escape_workspace(tmp_path):
    root, outside = tmp_path / "run", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "bundle").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "src"
    source.write_bytes(b"bytes")
    with pytest.raises(ValueError, match="escapes"):
        inputs._copy_verified(source, root / "bundle/file", root)
    assert not list(outside.iterdir())


def test_explicit_hf_cache_environment_is_preserved(tmp_path, hub, monkeypatch):
    home, cache, modules = (str(tmp_path / name) for name in ("my-home", "my-cache", "my-modules"))
    monkeypatch.setenv("HF_HOME", home)
    monkeypatch.setenv("HF_HUB_CACHE", cache)
    monkeypatch.setenv("HF_MODULES_CACHE", modules)
    result = inputs.download_inputs(tmp_path / "run", execute=True)
    assert result["hf_home"] == home
    assert result["hf_hub_cache"] == cache
    assert __import__("os").environ["HF_MODULES_CACHE"] == modules
    assert all(kwargs["cache_dir"] == cache for _, kwargs in hub["calls"])


def test_baseline_fingerprint_is_verified_even_with_a_matching_file_digest(tmp_path, hub, monkeypatch):
    invalid = b'{"score_hash":"not-the-right-hash"}'
    hub["bodies"][(inputs.RESULT_REPO, inputs.BASELINE_REVISION, inputs.BASELINE_FILE)] = invalid
    monkeypatch.setattr(inputs, "BASELINE_SHA256", digest(invalid))
    with pytest.raises(ValueError, match="Baseline score integrity"):
        inputs.download_inputs(tmp_path / "run", execute=True)


def test_model_revision_config_is_verified_before_copy(tmp_path, hub):
    model = tmp_path / "hub-model/config.json"
    model.write_bytes(b"different config")
    with pytest.raises(ValueError, match="model config"):
        inputs.download_inputs(tmp_path / "run", execute=True)
    assert not (tmp_path / "run/model").exists()


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_execute_requires_explicit_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="execute must be a bool"):
        inputs.download_inputs(tmp_path / "run", execute=value)
