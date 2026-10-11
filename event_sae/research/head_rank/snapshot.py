"""Content-verified relocation of one pinned OpenVLA snapshot.

The old score and its metadata remain immutable. This is not a generic identity
bypass: only safetensor mtimes may differ, and current bytes must match the
fixed Hub revision. Old shards 1/2 lack historical content evidence; a receipt
must retain that limitation rather than manufacture retrospective proof.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from event_sae.research.output_head.provenance import (
    atomic_write_json, fingerprint, read_json, sha256_file,
)
from .compatibility import validate_scoring_implementation, verify_scoring_bindings

PINNED_MODEL = {
    "repo_id": "openvla/openvla-7b-finetuned-libero-spatial",
    "revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
    "code_revision": "47a0ec7fc4ec123775a391911046cf33cf9ed83f",
}
# LFS SHA-256 OIDs and sizes at the exact revision above, not mutable main.
PINNED_SHARDS = {
    "model-00001-of-00004.safetensors": {
        "sha256": "094ba8d798d0e13152230897e893d4d12ca3fe11ddb218f27545b1e6e7536134", "size": 4925122448},
    "model-00002-of-00004.safetensors": {
        "sha256": "086780bb21b61eee9fb54acc0b938017c599e9d741bcdda18da23c4617fa36d7", "size": 4947392496},
    "model-00003-of-00004.safetensors": {
        "sha256": "d696e042cf49abd0b5846e73cbd4931d75a5467ae44857bf8e54a90ed6588308", "size": 4947417456},
    "model-00004-of-00004.safetensors": {
        "sha256": "234f92229d4055749c17e904e4d3f1c9a9af3eb13f96f14eaaa4530c0c376473", "size": 262668432},
}
HISTORICAL_UNVERIFIED = sorted(PINNED_SHARDS)[:2]
HISTORICAL_HEAD_HASHES = {name: PINNED_SHARDS[name]["sha256"] for name in sorted(PINNED_SHARDS)[2:]}
SCIENTIFIC_SCOPE = (
    "Current pinned model content verified; immutable cached head-score inputs reused. "
    "Historical full-model shard 1/2 bytes are not retrospectively verified. "
    "No scoring, model forward, simulator, or rollout is performed by this check."
)
_CORE_SOURCES = {
    "scores", "sample", "reference_scores", "cache_manifest", "sae", "sae_config",
    "head_weights", "head_manifest", "generation_manifest", "head_parity", "edit_parity",
}
_IDENTITY_SOURCES = {
    "sae": "sae_checkpoint_hash", "sae_config": "sae_config_hash",
    "head_weights": "head_weights_hash", "head_manifest": "head_manifest_hash",
    "generation_manifest": "generation_manifest_hash",
}


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA256: {name}")
    return value


def _filename(name):
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Snapshot evidence must use plain filenames")
    return name


def _mtime_differences(old, current):
    if not isinstance(old, dict) or not isinstance(current, dict):
        raise ValueError("Both complete numerical identities are required")
    for identity in (old, current):
        if (identity.get("model_checkpoint") != PINNED_MODEL["repo_id"]
                or identity.get("model_revision") != PINNED_MODEL["revision"]
                or identity.get("code_revision") != PINNED_MODEL["code_revision"]):
            raise ValueError("Snapshot relocation only supports the pinned spatial model/revisions")
        if identity.get("local_snapshot_verification_level") != "metadata_and_head_content_hashes_not_full_weight_rescan":
            raise ValueError("Unsupported historical snapshot identity schema")
    if {k: v for k, v in old.items() if k != "local_snapshot_identity"} != {
            k: v for k, v in current.items() if k != "local_snapshot_identity"}:
        raise ValueError("Only safetensor mtimes may differ; numerical/model/library identity changed")
    a, b = old.get("local_snapshot_identity"), current.get("local_snapshot_identity")
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
        raise ValueError("Snapshot asset inventory changed")
    if {n for n in a if n.endswith(".safetensors")} != set(PINNED_SHARDS):
        raise ValueError("Require exactly the four pinned model shards")
    changed = []
    for name in sorted(a):
        _filename(name)
        if name in PINNED_SHARDS:
            for entry in (a[name], b[name]):
                if (not isinstance(entry, dict) or set(entry) != {"mtime_ns", "size"}
                        or type(entry["mtime_ns"]) is not int or entry["mtime_ns"] < 0
                        or type(entry["size"]) is not int or entry["size"] != PINNED_SHARDS[name]["size"]):
                    raise ValueError("Snapshot shard size/schema differs from the pinned model")
            if a[name]["mtime_ns"] != b[name]["mtime_ns"]:
                changed.append(name)
        elif (a[name] != b[name] or not isinstance(a[name], dict) or set(a[name]) != {"sha256"}):
            raise ValueError("Non-weight snapshot content identity changed")
        else:
            _digest(a[name]["sha256"], name)
    return changed


def verify_snapshot_receipt(receipt, *, score_hash=None, scores_sha256=None,
                            old_identity=None, current_identity=None, sources=None):
    """Pure archival semantics; live creation/runtime separately read file bytes.

    Hashes detect inconsistent artifacts, not human authenticity or missing
    historical observations. Optional sources are the execution source map.
    """
    if (not isinstance(receipt, dict) or receipt.get("receipt_hash") != fingerprint(
            {k: v for k, v in receipt.items() if k != "receipt_hash"})):
        raise ValueError("Snapshot receipt hash mismatch")
    if (receipt.get("schema_version") != "head_rank_snapshot_receipt_v1"
            or receipt.get("status") != "snapshot_verified_not_runtime_approved"
            or receipt.get("pinned_model") != PINNED_MODEL
            or receipt.get("current_snapshot_shards") != PINNED_SHARDS
            or receipt.get("historical_head_source_hashes") != HISTORICAL_HEAD_HASHES
            or receipt.get("historical_unverified_shards") != HISTORICAL_UNVERIFIED
            or receipt.get("historical_full_model_content_verified") is not False
            or receipt.get("current_runtime_parity_verified") is not True
            or receipt.get("score_document_mutated") is not False
            or receipt.get("new_rollouts") != 0 or type(receipt.get("new_rollouts")) is not int
            or receipt.get("model_execution") is not False
            or receipt.get("scientific_scope") != SCIENTIFIC_SCOPE):
        raise ValueError("Unsupported or weakened snapshot receipt semantics")
    for key, actual in (("score_hash", score_hash), ("scores_sha256", scores_sha256)):
        _digest(receipt.get(key), key)
        if actual is not None and receipt[key] != actual:
            raise ValueError(f"Snapshot receipt is bound to different {key}")
    for key, actual in (("old_identity", old_identity), ("current_identity", current_identity)):
        if actual is not None and receipt.get(key) != actual:
            raise ValueError(f"Snapshot receipt is bound to different {key}")
    changed = _mtime_differences(receipt.get("old_identity"), receipt.get("current_identity"))
    if (receipt.get("changed_mtime_files") != changed or receipt.get("match_mode") !=
            ("mtime_only_relocation" if changed else "current_exact")):
        raise ValueError("Snapshot receipt misstates allowed metadata differences")
    verify_scoring_bindings({"scoring_implementation": receipt.get("scoring_implementation"),
                             "scorer_compatibility": receipt.get("scorer_compatibility")},
                            receipt.get("runtime_implementation"))
    recorded = receipt.get("sources")
    processor = receipt.get("processor_file_hashes")
    if not isinstance(recorded, dict) or not isinstance(processor, dict) or not processor:
        raise ValueError("Snapshot receipt is missing bound source/processor evidence")
    identity = receipt["current_identity"]
    evidence = receipt.get("processor_identity_evidence")
    if (not isinstance(evidence, dict) or not isinstance(evidence.get("files"), dict)
            or "sha256:" + fingerprint(evidence) != identity.get("processor_identity")
            or evidence.get("model_revision") != PINNED_MODEL["revision"]
            or evidence.get("code_revision") != PINNED_MODEL["code_revision"]
            or processor != {n: h for n, h in evidence["files"].items() if h is not None}):
        raise ValueError("Processor evidence is not bound to the immutable identity")
    assets = {name: value["sha256"] for name, value in identity["local_snapshot_identity"].items()
              if name not in PINNED_SHARDS}
    for name, digest in processor.items():
        _filename(name)
        _digest(digest, name)
        if name in assets and assets[name] != digest:
            raise ValueError("Processor and snapshot hashes disagree")
        assets[name] = digest
    assets.update({name: row["sha256"] for name, row in PINNED_SHARDS.items()})
    required = _CORE_SOURCES | {"snapshot/" + name for name in assets}
    if set(recorded) != required:
        raise ValueError("Snapshot receipt source coverage is incomplete or unexpected")
    expected = {name: identity.get(field) for name, field in _IDENTITY_SOURCES.items()}
    expected.update({"snapshot/" + name: digest for name, digest in assets.items()})
    expected["scores"] = receipt["scores_sha256"]
    for name, row in recorded.items():
        if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()):
            raise ValueError("Snapshot source record must have an absolute path and digest")
        _digest(row["sha256"], name)
        if name in expected and row["sha256"] != expected[name]:
            raise ValueError(f"Snapshot source differs from its identity: {name}")
        if sources is not None and sources.get(name, {}).get("sha256") != row["sha256"]:
            raise ValueError(f"Execution source differs from snapshot receipt: {name}")
    return receipt


def _stable_hash(path):
    """Catch normal concurrent mutation while hashing without changing mtimes."""
    path = Path(path)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"Snapshot input changed while hashing: {path}")
    return digest, after


def create_snapshot_receipt(config, scores_path, reference_result, output):
    """Hash local files and verify existing parity; never download or run a model."""
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import numerical_identity, load_parity, _sae_paths, _head_path
    from .workflow import REPO, implementation_identity, safe_output, validate_full_score_population

    cfg = load_config(config)
    root = Path(reference_result).resolve()
    snapshot = Path(cfg["inputs"]["local_model_snapshot"]).resolve()
    weights, sae_config = _sae_paths(cfg)
    head = _head_path(cfg)
    head = head.parent if head.is_file() else head
    target = safe_output(output, [REPO, config, scores_path, root, snapshot, weights.parent, head,
                                 cfg["output"]["root_dir"], cfg["inputs"]["generation_manifest"]])
    scores_before, scores_stat = _stable_hash(scores_path)
    scores = read_json(scores_path)
    if scores.get("score_hash") != fingerprint({k: v for k, v in scores.items() if k != "score_hash"}):
        raise ValueError("Immutable score hash mismatch")
    validate_full_score_population(scores)
    if (scores.get("development_only") is not False or scores.get("feature_set") != "all-active"
            or scores.get("synthetic") is not False or scores.get("status") != "scored"):
        raise ValueError("A complete real all-active score file is required")
    implementation = implementation_identity()
    compatibility = validate_scoring_implementation(scores.get("implementation"), implementation)
    identity = numerical_identity(cfg)
    changed = _mtime_differences(scores.get("identity"), identity)
    load_parity(cfg, identity)
    paths = {"scores": scores_path, "sample": root / "sample_manifest.json",
             "reference_scores": root / "scores/scores.json", "cache_manifest": root / "readouts/manifest.json",
             "sae": weights, "sae_config": sae_config, "head_weights": head / "output_head.safetensors",
             "head_manifest": head / "output_head_manifest.json",
             "generation_manifest": cfg["inputs"]["generation_manifest"],
             "head_parity": Path(cfg["output"]["root_dir"]) / "validation/runtime_parity.json",
             "edit_parity": Path(cfg["output"]["root_dir"]) / "validation/edit_parity.json"}
    manifest_stat = Path(paths["head_manifest"]).stat()
    manifest = read_json(paths["head_manifest"])
    historical = {name: manifest.get("source_hashes", {}).get(name) for name in HISTORICAL_HEAD_HASHES}
    if historical != HISTORICAL_HEAD_HASHES:
        raise ValueError("Historical exported-head source hashes do not prove pinned shards 3/4")
    evidence = manifest.get("processor_identity_evidence", {})
    if (manifest.get("processor_identity") != identity.get("processor_identity")
            or "sha256:" + fingerprint(evidence) != identity.get("processor_identity")
            or evidence.get("model_revision") != PINNED_MODEL["revision"]
            or evidence.get("code_revision") != PINNED_MODEL["code_revision"]):
        raise ValueError("Exported processor evidence differs from the immutable identity")
    processor = {name: digest for name, digest in evidence.get("files", {}).items() if digest is not None}
    if not processor:
        raise ValueError("Exported processor file evidence is missing")
    for name in set(identity["local_snapshot_identity"]) | set(processor):
        paths["snapshot/" + _filename(name)] = snapshot / name
    sources, source_stats = {}, {}
    for name, path in paths.items():
        digest, stat = _stable_hash(path)
        prior_stat = {"scores": scores_stat, "head_manifest": manifest_stat}.get(name)
        if prior_stat is not None and any(getattr(prior_stat, field) != getattr(stat, field)
                for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError(f"Input changed since its JSON evidence was read: {name}")
        if name.startswith("snapshot/") and name.split("/", 1)[1] in PINNED_SHARDS:
            asset = name.split("/", 1)[1]
            if {"sha256": digest, "size": stat.st_size} != PINNED_SHARDS[asset]:
                raise ValueError(f"Current model shard differs from pinned Hub content: {asset}")
        sources[name] = {"path": str(Path(path).resolve()), "sha256": digest}
        source_stats[name] = stat
    if sources["scores"]["sha256"] != scores_before:
        raise ValueError("Score file changed while preparing relocation receipt")
    for source, field in (("sample", "source_sample_sha256"), ("reference_scores", "source_scores_sha256"),
                           ("cache_manifest", "source_cache_manifest_sha256")):
        if sources[source]["sha256"] != scores.get(field):
            raise ValueError(f"Restored score source differs: {source}")
    sample = read_json(paths["sample"])
    scope = scores["scope"]
    if (scope.get("sae_sha256") != identity["sae_checkpoint_hash"]
            or sample.get("sample_hash") != scope.get("sample_manifest_hash")
            or sample.get("discovery_manifest_hash") != scope.get("discovery_manifest_hash")
            or sample.get("split_manifest_hash") != scope.get("split_manifest_hash")):
        raise ValueError("Frozen score/sample/SAE scope differs")
    receipt = {"schema_version": "head_rank_snapshot_receipt_v1",
        "status": "snapshot_verified_not_runtime_approved", "pinned_model": deepcopy(PINNED_MODEL),
        "current_snapshot_shards": deepcopy(PINNED_SHARDS), "historical_head_source_hashes": historical,
        "historical_unverified_shards": list(HISTORICAL_UNVERIFIED),
        "historical_full_model_content_verified": False, "current_runtime_parity_verified": True,
        "score_document_mutated": False, "new_rollouts": 0, "model_execution": False,
        "scientific_scope": SCIENTIFIC_SCOPE, "score_hash": scores["score_hash"],
        "scores_sha256": scores_before, "old_identity": scores["identity"], "current_identity": identity,
        "changed_mtime_files": changed, "match_mode": "mtime_only_relocation" if changed else "current_exact",
        "scoring_implementation": scores["implementation"], "runtime_implementation": implementation,
        "scorer_compatibility": compatibility, "processor_file_hashes": processor,
        "processor_identity_evidence": evidence, "sources": sources}
    receipt["receipt_hash"] = fingerprint(receipt)
    verify_snapshot_receipt(receipt, old_identity=scores["identity"], current_identity=identity)
    if numerical_identity(cfg) != identity or implementation_identity() != implementation:
        raise ValueError("Current files/runtime/source changed during content verification")
    load_parity(cfg, identity)
    # Detect changes to small critical files (including parity) after their first
    # read. Full model files are checked again by prepare-run and on every run.
    for name, source in sources.items():
        final_stat = Path(source["path"]).stat()
        if any(getattr(final_stat, field) != getattr(source_stats[name], field)
               for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError(f"Input changed during snapshot verification: {name}")
        if not (name.startswith("snapshot/") and name.endswith(".safetensors")):
            if _stable_hash(source["path"])[0] != source["sha256"]:
                raise ValueError(f"Input changed during snapshot verification: {name}")
    atomic_write_json(target, receipt)
    return {"status": receipt["status"], "receipt": str(target), "receipt_hash": receipt["receipt_hash"],
            "match_mode": receipt["match_mode"], "changed_mtime_files": changed,
            "verified_model_shards": len(PINNED_SHARDS), "historical_full_model_content_verified": False,
            "new_rollouts": 0, "model_execution": False}
