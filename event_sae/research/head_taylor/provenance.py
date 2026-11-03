"""Protect all existing experiment checkouts from new Taylor outputs."""
from pathlib import Path


def protected_source_roots():
    repo = Path(__file__).resolve().parents[3]
    names = ("Event-SAE-Baseline", "Event-SAE-Head", "Event-SAE-HeadRank",
             "Event-SAE-HeadCommand", "Event-SAE-Recollect", "Event-SAE-Pipeline")
    return [repo] + [repo.parent / name for name in names if (repo.parent / name).exists()]
