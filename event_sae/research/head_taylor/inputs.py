"""Pinned, explicitly requested Taylor inputs; no network activity on import.

This fetches the selected readout backup, not the 281-GiB dense dataset.  The
frozen inputs are copied from Hugging Face's revision cache without overwriting
different local bytes.  Restoration and current-device runtime validation are
separate operations: downloading never authorizes scoring or a rollout.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from event_sae.research.output_head.provenance import (
    assert_safe_output, fingerprint, read_json, sha256_file,
)

RESULT_REPO = "jiyeony/event-sae-head-full10-results"
RESULT_REVISION = "69c33d6a943c253dc97cf9a3968095276f7e9540"
ARCHIVE_SHA256 = "07eeb5c182ee4e195eb526362db9232ae1717de4c94b9b6da2fd62ef288de8bc"
SAE_REPO = "jiyeony/event-sae-openvla-libero-spatial-layer31-paper"
SAE_REVISION = "adb776b08f5b8ec4ea67556cf3e3bc60d91d607a"
SAE_SHA256 = "18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6"
MODEL_REPO = "openvla/openvla-7b-finetuned-libero-spatial"
MODEL_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
MODEL_CONFIG_SHA256 = "a2de1404c39c5608cc11917904ec610bf55a35b670b02db2ad71efb04a80b39e"
CODE_REPO = "openvla/openvla-7b"
CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
CODE_FILES = ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py")
BASELINE_REVISION = "417c222030dab9c5c6e354a75ebb03f8057aa0b5"
BASELINE_FILE = "followups/headrank-20261011T104004Z/full-scores.json"
BASELINE_SHA256 = "aea94626851e0510b50085e4707b534d623867e12b8bf16068274f6d04fb2b4a"
BASELINE_SCORE_HASH = "ffad9a82643b3bead59333346156542820da7ffd8aad233ddf5c2da9eceba9f8"
NEW_PANEL_REVISION = "84bfc5fd633fdea0fd9b6c35cd52a605b19d7ead"
NEW_PANEL_FILE = "followups/headrank-tail-backup.UhyVeF/analysis.json"
NEW_PANEL_SHA256 = "2968cb060c054ad2d56c09f6bab388b0d767c595ea8e3f23fe2a061a336046c7"


def input_paths(workspace):
    """Return local input paths without creating directories or importing HF."""
    root = Path(workspace).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    protected = [repo]
    for name in ("Event-SAE-Baseline", "Event-SAE-Head", "Event-SAE-HeadRank", "Event-SAE-HeadCommand"):
        sibling = repo.parent / name
        if sibling.exists():
            protected.append(sibling)
    assert_safe_output(root, protected)
    return {
        "workspace": root,
        "bundle": root / "bundle",
        "archive": root / "bundle/experiment.tar.gz",
        "manifest": root / "bundle/manifest.json",
        "restored": root / "restored",
        "reference_result": root / "restored/results/full10-followup-v1",
        "sae": root / "sae",
        "sae_checkpoint": root / "sae/trainer_0/ae.pt",
        "sae_config": root / "sae/trainer_0/config.json",
        "model": root / "model",
        "baselines": root / "baselines",
        "baseline_scores": root / "baselines/full-scores.json",
        "new_panel": root / "new-panel",
        "new_panel_analysis": root / "new-panel/analysis.json",
    }


def _copy_verified(source, destination, root, expected_sha256=None):
    """Verify source and write once; reuse only byte-identical existing files."""
    source, destination, root = Path(source), Path(destination), Path(root)
    if not source.is_file():
        raise ValueError(f"Missing downloaded regular file: {source}")
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Input destination escapes the dedicated workspace: {destination}")
    digest = sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Downloaded SHA-256 mismatch: {destination.name}")
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise ValueError(f"Existing input differs; it was not overwritten: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".taylor-download-", dir=destination.parent)
    temporary = Path(name)
    try:
        with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024**2)
        if sha256_file(temporary) != digest:
            raise ValueError("Downloaded-input copy verification failed")
        # Linking an adjacent temporary file publishes it atomically without
        # replacing a file another process may have created in the meantime.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _offline():
    truthy = {"1", "ON", "YES", "TRUE"}
    return any(os.environ.get(name, "").upper() in truthy
               for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def download_inputs(workspace, execute=False):
    """Plan only by default; explicit execution downloads immutable revisions.

    Existing HF cache variables are respected.  If no HF_HOME was selected, a
    dedicated cache is established before importing the hub client.  Offline
    mode is never silently disabled.  Model files are materialized separately
    from the hub cache, so budget approximately 60 GiB including restoration.
    """
    if type(execute) is not bool:
        raise ValueError("execute must be a bool")
    paths = input_paths(workspace)
    summary = {
        "status": "download_plan", "execute": execute,
        "paths": {name: str(path) for name, path in paths.items()},
        "sources": {
            "results": {"repo": RESULT_REPO, "revision": RESULT_REVISION},
            "sae": {"repo": SAE_REPO, "revision": SAE_REVISION},
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "code": {"repo": CODE_REPO, "revision": CODE_REVISION},
            "baseline_scores": {"repo": RESULT_REPO, "revision": BASELINE_REVISION,
                                "filename": BASELINE_FILE},
            "new_panel": {"repo": RESULT_REPO, "revision": NEW_PANEL_REVISION,
                          "filename": NEW_PANEL_FILE},
        },
        "dense_dataset_download": False, "new_rollouts": 0,
        "model_execution": False, "sae_training": False,
        "runtime_validation": "not_run",
        "recommended_free_disk_gib": 60,
    }
    if not execute:
        return summary
    if _offline():
        raise ValueError("Downloads require HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; offline mode was not changed")
    root = paths["workspace"]
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "cache/hf"))
    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(os.environ["HF_HOME"]) / "modules"))
    from huggingface_hub import hf_hub_download, snapshot_download
    from scripts.openvla.headfull import pin_nested_code_reference

    def fetch(repo, revision, filename, destination, *, repo_type=None, expected=None):
        source = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                 repo_type=repo_type, cache_dir=str(cache))
        return _copy_verified(source, destination, root, expected)

    hashes = {}
    hashes["archive"] = fetch(RESULT_REPO, RESULT_REVISION, "experiment.tar.gz", paths["archive"],
                              repo_type="dataset", expected=ARCHIVE_SHA256)
    hashes["manifest"] = fetch(RESULT_REPO, RESULT_REVISION, "manifest.json", paths["manifest"],
                               repo_type="dataset")
    if read_json(paths["manifest"]).get("archive", {}).get("sha256") != ARCHIVE_SHA256:
        raise ValueError("Result manifest does not bind the pinned archive")
    hashes["sae"] = fetch(SAE_REPO, SAE_REVISION, "trainer_0/ae.pt", paths["sae_checkpoint"], expected=SAE_SHA256)
    hashes["sae_config"] = fetch(SAE_REPO, SAE_REVISION, "trainer_0/config.json", paths["sae_config"])
    hashes["baseline_scores"] = fetch(RESULT_REPO, BASELINE_REVISION, BASELINE_FILE,
                                      paths["baseline_scores"], repo_type="dataset", expected=BASELINE_SHA256)
    baseline = read_json(paths["baseline_scores"])
    if (baseline.get("score_hash") != BASELINE_SCORE_HASH or
            fingerprint({key: value for key, value in baseline.items() if key != "score_hash"}) != BASELINE_SCORE_HASH):
        raise ValueError("Baseline score integrity hash differs from the frozen experiment")
    hashes["new_panel_analysis"] = fetch(RESULT_REPO, NEW_PANEL_REVISION, NEW_PANEL_FILE,
                                         paths["new_panel_analysis"], repo_type="dataset", expected=NEW_PANEL_SHA256)
    snapshot = Path(snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION,
                                      cache_dir=str(cache), max_workers=2))
    if sha256_file(snapshot / "config.json") != MODEL_CONFIG_SHA256:
        raise ValueError("Pinned model config SHA-256 mismatch")
    model_files = [file for file in snapshot.rglob("*") if file.is_file()]
    if not model_files or not list(snapshot.glob("model-*.safetensors")):
        raise ValueError("Pinned model snapshot is missing model weight shards")
    hashes["model_files"] = {
        str(file.relative_to(snapshot)): _copy_verified(file, paths["model"] / file.relative_to(snapshot), root)
        for file in sorted(model_files)
    }
    code = []
    for name in CODE_FILES:
        code.append(Path(hf_hub_download(repo_id=CODE_REPO, filename=name,
                                         revision=CODE_REVISION, cache_dir=str(cache))))
    if len({file.parent for file in code}) != 1:
        raise ValueError("Custom-code files did not resolve to one pinned snapshot")
    pin_nested_code_reference(code[0].parent)
    summary.update(status="downloaded", sha256=hashes, hf_home=os.environ["HF_HOME"],
                   hf_hub_cache=str(cache), restored=False,
                   next="stage1-restore, separate runtime setup, then current-device numerical validation")
    return summary
