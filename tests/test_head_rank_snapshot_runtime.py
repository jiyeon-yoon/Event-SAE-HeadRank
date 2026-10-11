"""Snapshot reuse remains a narrow opt-in gate, never paid/GPU execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from event_sae.research.head_rank import runtime
from event_sae.research.head_rank import native, snapshot, workflow
from event_sae.research.head_rank.protocol import plan_panel
from event_sae.research.output_head import workflow as old_workflow
from event_sae.research.output_head.provenance import fingerprint, read_json, sha256_file
from test_head_rank_runtime import CODE, IMPLEMENTATION, CPUFakeBackend, fixture_files, prepared, seal, write


def test_timestamp_mismatch_stays_blocked_without_receipt(tmp_path, monkeypatch):
    fixture = fixture_files(tmp_path, monkeypatch)
    changed = deepcopy(fixture["identity"])
    changed["local_snapshot_identity"] = {"model-1.safetensors": {"mtime_ns": 2, "size": 10}}
    monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: changed)
    with pytest.raises(ValueError, match="snapshot-receipt"):
        runtime.prepare_execution(**fixture["kwargs"])
    assert not fixture["kwargs"]["output"].exists()
    assert not fixture["parity_calls"]


@pytest.mark.parametrize("orphan", ["receipt", "source", "digest"])
def test_rehashed_orphan_snapshot_binding_is_rejected(tmp_path, monkeypatch, orphan):
    fixture, _, exp = prepared(tmp_path, monkeypatch)
    if orphan == "receipt":
        exp["snapshot_receipt"] = {"bad": "unpaired"}
    elif orphan == "source":
        exp["sources"]["snapshot_receipt"] = {"path": str(tmp_path / "receipt.json"), "sha256": "a" * 64}
    else:
        exp["snapshot_receipt_file_sha256"] = "a" * 64
    seal(exp, "execution_hash")
    with pytest.raises(ValueError, match="appear together"):
        runtime.verify_experiment(exp)


def test_legacy_contract_bytes_do_not_gain_receipt_fields(tmp_path, monkeypatch):
    _, _, exp = prepared(tmp_path, monkeypatch)
    assert "snapshot_receipt" not in exp
    assert "snapshot_receipt_file_sha256" not in exp
    assert "snapshot_receipt" not in exp["sources"]


def relocated_fixture(tmp_path, monkeypatch):
    """Tiny fake CPU model: only test constants replace 15GB production pins."""
    fixture = fixture_files(tmp_path, monkeypatch)
    kwargs, cfg, folder = fixture["kwargs"], fixture["cfg"], fixture["snapshot"]
    (folder / "model-1.safetensors").unlink()
    pins = {}
    for name in snapshot.PINNED_SHARDS:
        path = write(folder / name, {"synthetic_test_shard": name})
        pins[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    monkeypatch.setattr(snapshot, "PINNED_SHARDS", pins)
    historical = {name: pins[name]["sha256"] for name in sorted(pins)[2:]}
    monkeypatch.setattr(snapshot, "HISTORICAL_HEAD_HASHES", historical)
    evidence = {"files": {"config.json": sha256_file(folder / "config.json")},
                "model_revision": snapshot.PINNED_MODEL["revision"],
                "code_revision": snapshot.PINNED_MODEL["code_revision"]}
    processor_identity = "sha256:" + fingerprint(evidence)
    head = fixture["weights"].parent.parent / "head"
    write(head / "output_head_manifest.json", {"source_hashes": historical,
        "processor_identity": processor_identity, "processor_identity_evidence": evidence})
    identity = fixture["identity"]
    identity.update(model_checkpoint=snapshot.PINNED_MODEL["repo_id"],
        model_revision=snapshot.PINNED_MODEL["revision"], code_revision=snapshot.PINNED_MODEL["code_revision"],
        local_snapshot_verification_level="metadata_and_head_content_hashes_not_full_weight_rescan",
        processor_identity=processor_identity, sae_config_hash=sha256_file(fixture["weights"].parent / "config.json"),
        head_weights_hash=sha256_file(head / "output_head.safetensors"),
        head_manifest_hash=sha256_file(head / "output_head_manifest.json"),
        generation_manifest_hash=sha256_file(cfg["inputs"]["generation_manifest"]),
        local_snapshot_identity={name: {"mtime_ns": (folder / name).stat().st_mtime_ns - 10,
                                      "size": data["size"]} for name, data in pins.items()})
    identity["local_snapshot_identity"]["config.json"] = {"sha256": sha256_file(folder / "config.json")}
    current = deepcopy(identity)
    for name in pins:
        current["local_snapshot_identity"][name]["mtime_ns"] = (folder / name).stat().st_mtime_ns
    monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: deepcopy(current))
    monkeypatch.setattr(workflow, "implementation_identity", lambda: deepcopy(IMPLEMENTATION))
    write(kwargs["scores_path"], seal(fixture["scores"], "score_hash"))
    bindings = deepcopy(fixture["plan"]["source_bindings"])
    bindings.update(scores_sha256=sha256_file(kwargs["scores_path"]), score_hash=fixture["scores"]["score_hash"])
    fixture["plan"] = plan_panel(fixture["scores"], feature_count=4, task_ids=[0, 1], cases_per_task=1,
        reference_bindings=bindings, evaluation_cases=fixture["evaluation"]["cases"])
    write(kwargs["plan_path"], fixture["plan"])
    receipt_path = tmp_path / "snapshot-receipt.json"
    original_score_sha = sha256_file(kwargs["scores_path"])
    summary = snapshot.create_snapshot_receipt(kwargs["config"], kwargs["scores_path"], kwargs["reference_result"], receipt_path)
    assert summary["new_rollouts"] == 0
    assert sha256_file(kwargs["scores_path"]) == original_score_sha
    fixture.update(receipt_path=receipt_path, current_identity=current)
    return fixture


def test_real_receipt_prepare_and_cpu_fake_resume_then_archival_verification(tmp_path, monkeypatch):
    fixture = relocated_fixture(tmp_path, monkeypatch)
    result = runtime.prepare_execution(**fixture["kwargs"], snapshot_receipt_path=fixture["receipt_path"])
    assert result["new_rollouts"] == 0 and result["total_rollouts"] == 12
    exp = read_json(fixture["kwargs"]["output"])
    assert exp["numerical_identity"] == fixture["current_identity"]
    assert exp["snapshot_receipt"]["old_identity"] == fixture["identity"]
    assert exp["snapshot_receipt_file_sha256"] == sha256_file(fixture["receipt_path"])
    assert exp["snapshot_receipt"]["historical_full_model_content_verified"] is False
    monkeypatch.setattr(native, "NativeBackend", CPUFakeBackend)
    output = runtime.run_execution(fixture["kwargs"]["output"],
        approved_hash=exp["execution_hash"], expected_revision=CODE["commit"])
    assert output["status"] == "completed"
    # Original inputs may disappear after archival. Verification uses embedded
    # evidence; it must neither load models nor pretend to rerun numerical parity.
    fixture["weights"].unlink()
    fixture["receipt_path"].unlink()
    runtime.verify_experiment(exp)


@pytest.mark.parametrize("change", ["old_identity", "current_identity", "score_hash", "scores_sha256",
                                   "source", "missing_source", "runtime_implementation"])
def test_rehashed_receipt_for_other_inputs_cannot_prepare(tmp_path, monkeypatch, change):
    fixture = relocated_fixture(tmp_path, monkeypatch)
    receipt = read_json(fixture["receipt_path"])
    if change in ("old_identity", "current_identity"):
        first = sorted(snapshot.PINNED_SHARDS)[0]
        receipt[change]["local_snapshot_identity"][first]["mtime_ns"] += 1
    elif change in ("score_hash", "scores_sha256"):
        receipt[change] = "f" * 64
    elif change == "source":
        receipt["sources"]["head_parity"]["sha256"] = "f" * 64
    elif change == "missing_source":
        del receipt["sources"]["head_parity"]
    else:
        # Exact self-consistent fake scorer/runtime pair still cannot replace
        # the separately frozen execution implementation.
        receipt["runtime_implementation"] = receipt["scoring_implementation"] = {"other": True}
    write(fixture["receipt_path"], seal(receipt, "receipt_hash"))
    with pytest.raises(ValueError):
        runtime.prepare_execution(**fixture["kwargs"], snapshot_receipt_path=fixture["receipt_path"])
    assert not fixture["kwargs"]["output"].exists()


@pytest.mark.parametrize("change", ["sae", "parity", "receipt", "identity"])
def test_snapshot_receipt_does_not_relax_live_resume_integrity(tmp_path, monkeypatch, change):
    fixture = relocated_fixture(tmp_path, monkeypatch)
    runtime.prepare_execution(**fixture["kwargs"], snapshot_receipt_path=fixture["receipt_path"])
    exp = read_json(fixture["kwargs"]["output"])
    if change == "sae":
        write(fixture["weights"], {"changed": True})
    elif change == "parity":
        write(Path(fixture["cfg"]["output"]["root_dir"]) / "validation/runtime_parity.json", {"changed": True})
    elif change == "receipt":
        write(fixture["receipt_path"], {"changed": True})
    else:
        altered = deepcopy(fixture["current_identity"])
        altered["device"] = "other_gpu"
        monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: altered)
    with pytest.raises(ValueError):
        runtime.run_execution(fixture["kwargs"]["output"],
            approved_hash=exp["execution_hash"], expected_revision=CODE["commit"])
    assert not fixture["kwargs"]["run_output"].exists()
