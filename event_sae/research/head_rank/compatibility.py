"""Narrow, read-only reuse of the immutable 183eaa9 v1 scorer artifacts.

This is a source-identity allowlist, not a score-file allowlist. It neither
rehashes/rebinds score documents nor exempts full-population, split, numerical
identity, parity, or clean-runtime-revision gates. The manifest below was
derived from the named Git objects, never from a mutable caller-supplied path.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from event_sae.research.output_head.provenance import fingerprint


LEGACY_SCORER_COMMIT = "183eaa9ff0f6ba44e5e0a403ecf40d8f335b8310"
LEGACY_SCORER_IMPLEMENTATION = {
    "base_implementation_hash": "71f4676206de3bcd8ce18713388eab2cbbe653800319b46e5e0f9971c2d0442a",
    "files": {
        "event_sae/research/head_followup/interventions.py": "8c12e9133e00b8f70b72070e58a129883a0ba14b53a401733abbefe9c20abae3",
        "event_sae/research/head_rank/__init__.py": "0a23caa64a2c04dc510a704feeb7812ebeeb16607e6d223ac42b70d98d85ea26",
        "event_sae/research/head_rank/analysis.py": "1726d5914fade81a69915d6330d0272e6b0726571910999994aeb1cb690b96b9",
        "event_sae/research/head_rank/execution.py": "b09fd273935d97c3a649449cf77583fb1e233b260f4d01b5e4db0103bf6f35b4",
        "event_sae/research/head_rank/native.py": "9dfe127e7ae6c1c8bbed27cf1ef90f7c2a859114a7e74a3fc5c58d3604fbf5e1",
        "event_sae/research/head_rank/protocol.py": "32188831fd622bfa8a1c38d7a4787cf8080f824ff54a39851d923525e0d89ad2",
        "event_sae/research/head_rank/runtime.py": "a21b0474bea96dbf3f205ae7bd6fc769f215ca7a4a1fadb4b51d46dd3f54e203",
        "event_sae/research/head_rank/scoring.py": "9b69234b06ce0572276398bb636da74e951ff4257fc21b7d49719579d024d0dd",
        "event_sae/research/head_rank/workflow.py": "abd2a887e05b4e0bd89198da8adaa3bf229d9e3deac9074c72d55d93f06e22f9",
        "scripts/openvla/headrank.py": "3236fa3d0c2d26533b9339b5882620ac4de7fca0613384fb1b73e0e6f01a9a1a",
    },
}
LEGACY_ADAPTER_AST_SHA256 = "0e24d209b58b8fa6c114dccececa5bc719ab84c412ed29cbbd49da3c1d0f8fe9"
# The sole permitted scoring-adapter change adds the pure calibrated-attachment
# validator to implementation_identity's fixed source list. No scoring or
# loading logic changed; tests compare the complete old/current adapter ASTs.
RUNTIME_BINDING_ADAPTER_AST_SHA256 = "d1658a61c9e99ce9a9c3f5304f22ea4dae38f64b2f6331e14ad24b56a582a7e5"
_KERNEL_FILES = ("event_sae/research/head_rank/scoring.py",
                 "event_sae/research/head_rank/__init__.py",
                 "event_sae/research/head_followup/interventions.py")


def scorer_adapter_ast_sha256(source: str) -> str:
    """All workflow source except its two explicitly non-scoring entry points.

Using an AST permits formatting-only changes, not modified scorer setup,
helpers, imports, globals, or score publication. No historical code executes.
"""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 or node.name not in {"plan_workflow", "validate_full_score_population"}]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _receipt(recorded, mode):
    return {"schema_version": "head_rank_scorer_compatibility_v1", "mode": mode,
            "scoring_implementation_sha256": fingerprint(recorded),
            "legacy_scorer_commit": LEGACY_SCORER_COMMIT if mode == "pinned_legacy_v1" else None,
            "score_document_mutated": False,
            "current_numerical_identity_and_parity_required": True}


def validate_scoring_implementation(recorded, current):
    """Accept exact current source, or this one supported unchanged old scorer."""
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Scoring implementation identity is missing")
    if recorded == current:
        return _receipt(recorded, "current_exact")
    if recorded != LEGACY_SCORER_IMPLEMENTATION:
        raise ValueError("Unsupported legacy scoring implementation; source changed after scoring")
    if (current.get("base_implementation_hash") != recorded["base_implementation_hash"]
            or any(current.get("files", {}).get(path) != recorded["files"][path] for path in _KERNEL_FILES)):
        raise ValueError("Legacy scorer reuse requires byte-identical numerical kernel and base dependencies")
    source = Path(__file__).with_name("workflow.py").read_text(encoding="utf-8")
    if scorer_adapter_ast_sha256(source) not in (LEGACY_ADAPTER_AST_SHA256, RUNTIME_BINDING_ADAPTER_AST_SHA256):
        raise ValueError("Legacy scoring workflow adapter changed; compatibility must be reviewed")
    return _receipt(recorded, "pinned_legacy_v1")


def verify_scoring_bindings(bindings, current):
    """Pure archival check; current source/device checks are separate runtime gates."""
    has_scoring = "scoring_implementation" in bindings
    has_receipt = "scorer_compatibility" in bindings
    if not has_scoring and not has_receipt:
        # Historical v1 plans bound score and runtime to one exact identity.
        return
    if not has_scoring or not has_receipt:
        raise ValueError("Separate scorer/runtime provenance is incomplete")
    recorded = bindings["scoring_implementation"]
    mode = "current_exact" if recorded == current else "pinned_legacy_v1"
    if mode == "pinned_legacy_v1":
        if (recorded != LEGACY_SCORER_IMPLEMENTATION
                or current.get("base_implementation_hash") != recorded["base_implementation_hash"]
                or any(current.get("files", {}).get(path) != recorded["files"][path] for path in _KERNEL_FILES)):
            raise ValueError("Unsupported frozen legacy scorer/runtime binding")
    if bindings["scorer_compatibility"] != _receipt(recorded, mode):
        raise ValueError("Scorer compatibility receipt does not match frozen identities")
