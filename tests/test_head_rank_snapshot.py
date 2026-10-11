"""Snapshot relocation semantics; no downloaded model, CUDA, or simulator."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from event_sae.research.head_rank import snapshot as sn
from event_sae.research.head_rank.compatibility import LEGACY_SCORER_IMPLEMENTATION, validate_scoring_implementation
from event_sae.research.head_rank.workflow import implementation_identity
from event_sae.research.output_head.provenance import fingerprint, sha256_file


def seal(receipt):
    receipt["receipt_hash"] = fingerprint({k: v for k, v in receipt.items() if k != "receipt_hash"})
    return receipt


def receipt_fixture():
    """Pure fixture using real pinned constants, not an executable receipt."""
    evidence = {"model_revision": sn.PINNED_MODEL["revision"], "code_revision": sn.PINNED_MODEL["code_revision"],
                "files": {"config.json": "1" * 64, "tokenizer.model": "2" * 64, "missing.json": None}}
    identity = {"model_checkpoint": sn.PINNED_MODEL["repo_id"], "model_revision": sn.PINNED_MODEL["revision"],
        "code_revision": sn.PINNED_MODEL["code_revision"], "processor_identity": "sha256:" + fingerprint(evidence),
        "local_snapshot_verification_level": "metadata_and_head_content_hashes_not_full_weight_rescan",
        "local_snapshot_identity": {name: {"size": row["size"], "mtime_ns": 10} for name, row in sn.PINNED_SHARDS.items()},
        "device": "cuda:0", "torch_version": "2.2.0"}
    identity["local_snapshot_identity"]["config.json"] = {"sha256": "1" * 64}
    for field in sn._IDENTITY_SOURCES.values():
        identity[field] = "3" * 64
    current = deepcopy(identity)
    for name in sn.PINNED_SHARDS:
        current["local_snapshot_identity"][name]["mtime_ns"] = 20
    sources = {name: {"path": "/synthetic_fixture/" + name, "sha256": "3" * 64} for name in sn._CORE_SOURCES}
    sources["scores"]["sha256"] = "4" * 64
    for name, row in sn.PINNED_SHARDS.items():
        sources["snapshot/" + name] = {"path": "/synthetic_fixture/" + name, "sha256": row["sha256"]}
    for name, digest in evidence["files"].items():
        if digest is not None:
            sources["snapshot/" + name] = {"path": "/synthetic_fixture/" + name, "sha256": digest}
    implementation = implementation_identity()
    return seal({"schema_version": "head_rank_snapshot_receipt_v1", "status": "snapshot_verified_not_runtime_approved",
        "pinned_model": deepcopy(sn.PINNED_MODEL), "current_snapshot_shards": deepcopy(sn.PINNED_SHARDS),
        "historical_head_source_hashes": deepcopy(sn.HISTORICAL_HEAD_HASHES),
        "historical_unverified_shards": list(sn.HISTORICAL_UNVERIFIED), "historical_full_model_content_verified": False,
        "current_runtime_parity_verified": True, "score_document_mutated": False, "new_rollouts": 0,
        "model_execution": False, "scientific_scope": sn.SCIENTIFIC_SCOPE, "score_hash": "5" * 64,
        "scores_sha256": "4" * 64, "old_identity": identity, "current_identity": current,
        "changed_mtime_files": sorted(sn.PINNED_SHARDS), "match_mode": "mtime_only_relocation",
        "scoring_implementation": deepcopy(LEGACY_SCORER_IMPLEMENTATION), "runtime_implementation": implementation,
        "scorer_compatibility": validate_scoring_implementation(LEGACY_SCORER_IMPLEMENTATION, implementation),
        "processor_file_hashes": {n: h for n, h in evidence["files"].items() if h is not None},
        "processor_identity_evidence": evidence, "sources": sources})


def test_receipt_allows_only_explicit_mtime_relocation_without_mutation():
    receipt = receipt_fixture()
    original = deepcopy(receipt)
    assert sn.verify_snapshot_receipt(receipt, score_hash=receipt["score_hash"],
        scores_sha256=receipt["scores_sha256"], old_identity=receipt["old_identity"],
        current_identity=receipt["current_identity"], sources=receipt["sources"]) == original
    assert receipt == original
    receipt["current_identity"] = deepcopy(receipt["old_identity"])
    receipt.update(changed_mtime_files=[], match_mode="current_exact")
    sn.verify_snapshot_receipt(seal(receipt))


@pytest.mark.parametrize("change", ["library", "gpu", "size", "extra_shard", "json", "mtime_schema", "pin",
    "historical_promotion", "missing_old_evidence", "missing_source", "source_head", "source_shard", "tokenizer",
    "score_mutated", "no_parity", "fake_change_list", "different_revision", "fake_processor_evidence"])
def test_rehash_does_not_bypass_semantic_guards(change):
    receipt = receipt_fixture()
    name = sorted(sn.PINNED_SHARDS)[0]
    current = receipt["current_identity"]
    if change == "library": current["torch_version"] = "new"
    elif change == "gpu": current["device"] = "cuda:1"
    elif change == "size": current["local_snapshot_identity"][name]["size"] += 1
    elif change == "extra_shard": current["local_snapshot_identity"]["evil.safetensors"] = {"size": 1, "mtime_ns": 1}
    elif change == "json": current["local_snapshot_identity"]["config.json"]["sha256"] = "0" * 64
    elif change == "mtime_schema": current["local_snapshot_identity"][name]["mtime_ns"] = True
    elif change == "pin": receipt["current_snapshot_shards"][name]["sha256"] = "0" * 64
    elif change == "historical_promotion": receipt["historical_full_model_content_verified"] = True
    elif change == "missing_old_evidence": receipt["historical_head_source_hashes"] = {}
    elif change == "missing_source": receipt["sources"].pop("head_parity")
    elif change == "source_head": receipt["sources"]["head_weights"]["sha256"] = "0" * 64
    elif change == "source_shard": receipt["sources"]["snapshot/" + name]["sha256"] = "0" * 64
    elif change == "tokenizer": receipt["sources"]["snapshot/tokenizer.model"]["sha256"] = "0" * 64
    elif change == "score_mutated": receipt["score_document_mutated"] = True
    elif change == "no_parity": receipt["current_runtime_parity_verified"] = False
    elif change == "fake_change_list": receipt["changed_mtime_files"] = []
    elif change == "different_revision": current["model_revision"] = "0" * 40
    else:
        receipt["processor_identity_evidence"]["files"]["tokenizer.model"] = "0" * 64
        receipt["processor_file_hashes"]["tokenizer.model"] = "0" * 64
        receipt["sources"]["snapshot/tokenizer.model"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): sn.verify_snapshot_receipt(seal(receipt))


@pytest.mark.parametrize("field", ["score_hash", "scores_sha256", "old_identity", "current_identity", "sources"])
def test_receipt_must_match_execution_bindings(field):
    receipt = receipt_fixture()
    value = "f" * 64 if field.endswith("hash") or field == "scores_sha256" else {}
    with pytest.raises(ValueError): sn.verify_snapshot_receipt(receipt, **{field: value})


def creation_fixture(tmp_path, monkeypatch):
    """Small explicit fake content pins isolate filesystem logic from 15GB models."""
    from event_sae.research.output_head import config as config_module, workflow as old_workflow
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path
    snapshot = inputs / "snapshot"
    snapshot.mkdir()
    fakepins = {}
    for index, name in enumerate(sn.PINNED_SHARDS):
        data = f"SYNTHETIC TEST WEIGHT {index}".encode()
        (snapshot / name).write_bytes(data)
        fakepins[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(sn, "PINNED_SHARDS", fakepins)
    monkeypatch.setattr(sn, "HISTORICAL_HEAD_HASHES", {n: fakepins[n]["sha256"] for n in sorted(fakepins)[2:]})
    config_json = write(snapshot / "config.json", {"synthetic": True})
    (snapshot / "tokenizer.model").write_bytes(b"SYNTHETIC TOKENIZER")
    processor = {n: sha256_file(snapshot / n) for n in ("config.json", "tokenizer.model")}
    evidence = {"files": processor, "model_revision": sn.PINNED_MODEL["revision"],
                "code_revision": sn.PINNED_MODEL["code_revision"]}
    weights = inputs / "sae/ae.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"SYNTHETIC SAE")
    sae_config = write(weights.parent / "config.json", {"synthetic": True})
    head = inputs / "head"
    head.mkdir()
    (head / "output_head.safetensors").write_bytes(b"SYNTHETIC HEAD")
    write(head / "output_head_manifest.json", {"source_hashes": deepcopy(sn.HISTORICAL_HEAD_HASHES),
        "processor_identity": "sha256:" + fingerprint(evidence), "processor_identity_evidence": evidence})
    generation = write(inputs / "generation.json", {"synthetic": True})
    parity = inputs / "parity"
    for name in ("runtime_parity.json", "edit_parity.json"):
        write(parity / "validation" / name, {"synthetic": True})
    scope = {"sae_sha256": sha256_file(weights), "sample_manifest_hash": "a" * 64,
             "discovery_manifest_hash": "b" * 64, "split_manifest_hash": "c" * 64, "task_ids": [0]}
    ref = inputs / "reference"
    sample = write(ref / "sample_manifest.json", {"sample_hash": scope["sample_manifest_hash"],
        "discovery_manifest_hash": scope["discovery_manifest_hash"], "split_manifest_hash": scope["split_manifest_hash"]})
    oldscore = write(ref / "scores/scores.json", {"synthetic": True})
    cache = write(ref / "readouts/manifest.json", {"synthetic": True})
    identity = {"model_checkpoint": sn.PINNED_MODEL["repo_id"], "model_revision": sn.PINNED_MODEL["revision"],
        "code_revision": sn.PINNED_MODEL["code_revision"], "processor_identity": "sha256:" + fingerprint(evidence),
        "local_snapshot_verification_level": "metadata_and_head_content_hashes_not_full_weight_rescan",
        "local_snapshot_identity": {n: {"size": (snapshot / n).stat().st_size,
            "mtime_ns": (snapshot / n).stat().st_mtime_ns} for n in fakepins},
        "sae_checkpoint_hash": sha256_file(weights), "sae_config_hash": sha256_file(sae_config),
        "head_weights_hash": sha256_file(head / "output_head.safetensors"),
        "head_manifest_hash": sha256_file(head / "output_head_manifest.json"), "generation_manifest_hash": sha256_file(generation)}
    identity["local_snapshot_identity"]["config.json"] = {"sha256": sha256_file(config_json)}
    previous = deepcopy(identity)
    for n in fakepins: previous["local_snapshot_identity"][n]["mtime_ns"] = 1
    score = {"schema_version": "head_rank_scores_v1", "sampling_mode": "full", "feature_set": "all-active",
        "development_only": False, "synthetic": False, "status": "scored", "identity": previous,
        "implementation": deepcopy(LEGACY_SCORER_IMPLEMENTATION), "scope": scope,
        "feature_scores": [{"feature_id": 1, "num_readouts": 7, "num_queries": 1}],
        "feature_ids": [1], "scored_feature_ids": [1], "frozen_active_feature_ids": [1],
        "query_metadata": [{"query_index": 0, "task_id": 0, "episode_uid": "a", "step_in_episode": 0, "num_action_positions": 7}],
        "scoring_sample": {"event_max_matches_scoring_episode_population": True, "readouts_used": 7, "discovery_episodes_used": 1},
        "source_sample_sha256": sha256_file(sample), "source_scores_sha256": sha256_file(oldscore),
        "source_cache_manifest_sha256": sha256_file(cache)}
    score["score_hash"] = fingerprint(score)
    scores = write(inputs / "scores.json", score)
    cfg = {"inputs": {"local_model_snapshot": str(snapshot), "generation_manifest": str(generation)},
           "output": {"root_dir": str(parity)}}
    config = write(inputs / "cfg.json", cfg)
    monkeypatch.setattr(config_module, "load_config", lambda _: deepcopy(cfg))
    monkeypatch.setattr(old_workflow, "_sae_paths", lambda _: (weights, sae_config))
    monkeypatch.setattr(old_workflow, "_head_path", lambda _: head)
    monkeypatch.setattr(old_workflow, "numerical_identity", lambda _: deepcopy(identity))
    parity_calls = []
    monkeypatch.setattr(old_workflow, "load_parity", lambda c, i: parity_calls.append(deepcopy(i)))
    return {"args": (config, scores, ref, tmp_path / "receipt.json"), "identity": identity,
            "snapshot": snapshot, "head": head, "score": score, "parity_calls": parity_calls}


def test_create_receipt_content_verifies_without_mutating_inputs(tmp_path, monkeypatch):
    f = creation_fixture(tmp_path, monkeypatch)
    before = f["args"][1].read_bytes()
    mtimes = {p.name: p.stat().st_mtime_ns for p in f["snapshot"].iterdir()}
    result = sn.create_snapshot_receipt(*f["args"])
    assert result["status"] == "snapshot_verified_not_runtime_approved" and result["new_rollouts"] == 0
    assert len(f["parity_calls"]) == 2 and f["args"][1].read_bytes() == before
    assert {p.name: p.stat().st_mtime_ns for p in f["snapshot"].iterdir()} == mtimes
    receipt = json.loads(f["args"][3].read_text())
    sn.verify_snapshot_receipt(receipt)
    with pytest.raises(FileExistsError): sn.create_snapshot_receipt(*f["args"])


@pytest.mark.parametrize("change", ["shard", "tokenizer", "head_evidence", "score_source", "output_overlap", "parity"])
def test_create_refuses_bad_current_content_or_proof(tmp_path, monkeypatch, change):
    f = creation_fixture(tmp_path, monkeypatch)
    args = list(f["args"])
    if change == "shard": (f["snapshot"] / sorted(sn.PINNED_SHARDS)[0]).write_bytes(b"WRONG MODEL")
    elif change == "tokenizer": (f["snapshot"] / "tokenizer.model").write_bytes(b"WRONG TOKENIZER")
    elif change == "head_evidence":
        p = f["head"] / "output_head_manifest.json"
        d = json.loads(p.read_text()); d["source_hashes"] = {}; p.write_text(json.dumps(d))
    elif change == "score_source": (args[2] / "sample_manifest.json").write_text("{}")
    elif change == "output_overlap": args[3] = f["snapshot"] / "receipt.json"
    else:
        from event_sae.research.output_head import workflow
        def fail(*_): raise ValueError("No valid current parity")
        monkeypatch.setattr(workflow, "load_parity", fail)
    with pytest.raises(ValueError): sn.create_snapshot_receipt(*args)
    assert not Path(args[3]).exists()


def test_stable_hash_detects_mutation_during_hash(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"before")
    def mutate(p):
        digest = sha256_file(p)
        p.write_bytes(b"changed")
        return digest
    monkeypatch.setattr(sn, "sha256_file", mutate)
    with pytest.raises(ValueError, match="changed while hashing"):
        sn._stable_hash(path)


def test_creation_rejects_replaced_json_even_when_bytes_are_restored(tmp_path, monkeypatch):
    f = creation_fixture(tmp_path, monkeypatch)
    original_read = sn.read_json
    def replace_after_read(path):
        result = original_read(path)
        if Path(path) == f["args"][1]:
            content = Path(path).read_bytes()
            Path(path).unlink()
            Path(path).write_bytes(content)
        return result
    monkeypatch.setattr(sn, "read_json", replace_after_read)
    with pytest.raises(ValueError, match="JSON evidence was read"):
        sn.create_snapshot_receipt(*f["args"])
    assert not f["args"][3].exists()
