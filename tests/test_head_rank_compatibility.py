"""Pinned legacy scorer reuse, never a GPU or behavior evaluation."""
from copy import deepcopy
import ast
import hashlib
import subprocess

import pytest

from event_sae.research.head_rank import compatibility as compat
from event_sae.research.head_rank.workflow import REPO, implementation_identity
from event_sae.research.output_head.provenance import fingerprint


def test_static_manifest_matches_exact_readonly_legacy_git_objects():
    def blob(path):
        return subprocess.check_output(["git", "-C", str(REPO), "show", f"{compat.LEGACY_SCORER_COMMIT}:{path}"])
    expected = compat.LEGACY_SCORER_IMPLEMENTATION
    for path, digest in expected["files"].items():
        assert hashlib.sha256(blob(path)).hexdigest() == digest
    names = subprocess.check_output(["git", "-C", str(REPO), "ls-tree", "--name-only",
        f"{compat.LEGACY_SCORER_COMMIT}:event_sae/research/output_head"], text=True).splitlines()
    base_files = ["research/output_head/" + name for name in names if name.endswith(".py")]
    base_files += ["openvla/intervene.py", "openvla/activations.py", "openvla/eval/model.py",
                   "openvla/eval/runner.py", "openvla/eval/config.py"]
    base = {path: hashlib.sha256(blob("event_sae/" + path)).hexdigest() for path in base_files}
    assert fingerprint(base) == expected["base_implementation_hash"]
    adapter = blob("event_sae/research/head_rank/workflow.py").decode()
    assert compat.scorer_adapter_ast_sha256(adapter) == compat.LEGACY_ADAPTER_AST_SHA256


def test_legacy_reuse_separates_current_runtime_without_mutating_scorer():
    recorded = deepcopy(compat.LEGACY_SCORER_IMPLEMENTATION)
    original = deepcopy(recorded)
    current = implementation_identity()
    receipt = compat.validate_scoring_implementation(recorded, current)
    assert receipt["mode"] == "pinned_legacy_v1"
    assert receipt["legacy_scorer_commit"] == compat.LEGACY_SCORER_COMMIT
    assert recorded == original and recorded != current
    compat.verify_scoring_bindings({"scoring_implementation": recorded,
                                   "scorer_compatibility": receipt}, current)


def test_reviewed_adapter_change_only_adds_runtime_validator_source_binding():
    old_source = subprocess.check_output(["git", "-C", str(REPO), "show",
        f"{compat.LEGACY_SCORER_COMMIT}:event_sae/research/head_rank/workflow.py"], text=True)
    current_source = (REPO / "event_sae/research/head_rank/workflow.py").read_text()
    assert compat.scorer_adapter_ast_sha256(current_source) == compat.RUNTIME_BINDING_ADAPTER_AST_SHA256
    old_tree, current_tree = ast.parse(old_source), ast.parse(current_source)
    old_identity = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "implementation_identity")
    current_identity = next(n for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name == "implementation_identity")
    # The expected one-expression addition is the only change to this function.
    expected_identity = deepcopy(old_identity)
    file_list = next(n for n in ast.walk(expected_identity) if isinstance(n, ast.List))
    file_list.elts.append(ast.parse('REPO / "event_sae/research/head_predict/evaluation.py"', mode="eval").body)
    assert ast.dump(current_identity, include_attributes=False) == ast.dump(expected_identity, include_attributes=False)
    current_tree.body = [old_identity if n is current_identity else n for n in current_tree.body]
    assert compat.scorer_adapter_ast_sha256(ast.unparse(current_tree)) == compat.LEGACY_ADAPTER_AST_SHA256


@pytest.mark.parametrize("change", ["unknown_legacy", "base", "kernel", "intervention", "adapter"])
def test_legacy_exception_rejects_source_drift(monkeypatch, change):
    recorded, current = deepcopy(compat.LEGACY_SCORER_IMPLEMENTATION), implementation_identity()
    if change == "unknown_legacy":
        recorded["files"]["scripts/openvla/headrank.py"] = "0" * 64
    elif change == "base":
        current["base_implementation_hash"] = "0" * 64
    elif change == "kernel":
        current["files"]["event_sae/research/head_rank/scoring.py"] = "0" * 64
    elif change == "intervention":
        current["files"]["event_sae/research/head_followup/interventions.py"] = "0" * 64
    elif change == "adapter":
        monkeypatch.setattr(compat, "scorer_adapter_ast_sha256", lambda _: "0" * 64)
    with pytest.raises(ValueError):
        compat.validate_scoring_implementation(recorded, current)


@pytest.mark.parametrize("change", ["missing", "receipt", "identity", "current"])
def test_rehashed_legacy_binding_cannot_forge_supported_compatibility(change):
    current = implementation_identity()
    recorded = deepcopy(compat.LEGACY_SCORER_IMPLEMENTATION)
    bindings = {"scoring_implementation": recorded,
                "scorer_compatibility": compat.validate_scoring_implementation(recorded, current)}
    if change == "missing":
        bindings.pop("scorer_compatibility")
    elif change == "receipt":
        bindings["scorer_compatibility"]["legacy_scorer_commit"] = "a" * 40
    elif change == "identity":
        recorded["files"]["scripts/openvla/headrank.py"] = "a" * 64
    else:
        current["base_implementation_hash"] = "a" * 64
    with pytest.raises(ValueError):
        compat.verify_scoring_bindings(bindings, current)
