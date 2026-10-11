"""CLI smoke uses archived synthetic fixtures; no simulator, model or GPU."""
import json
import subprocess
import sys
from pathlib import Path

from event_sae.research.head_rank import native, runtime
from test_head_rank_runtime import CODE, CPUFakeBackend, prepared

CLI = Path(__file__).resolve().parents[1] / "scripts/openvla/headrank.py"


def cli(*args):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)


def test_cli_execution_is_dry_by_default_and_needs_explicit_approval(tmp_path, monkeypatch):
    fixture, _, _ = prepared(tmp_path, monkeypatch)
    path = fixture["kwargs"]["output"]
    preview = cli("run", "--experiment", path)
    assert preview.returncode == 0
    result = json.loads(preview.stdout)
    assert result["status"] == "not_executed" and result["new_rollouts"] == 0
    assert result["total_rollouts"] == 12
    denied = cli("run", "--experiment", path, "--execute")
    assert denied.returncode == 2
    assert "approval" in json.loads(denied.stdout)["error"]
    assert not fixture["kwargs"]["run_output"].exists()


def test_cli_status_never_creates_a_run_and_analysis_needs_complete_cases(tmp_path, monkeypatch):
    fixture, _, _ = prepared(tmp_path, monkeypatch)
    path, runs = fixture["kwargs"]["output"], fixture["kwargs"]["run_output"]
    status = cli("status", "--experiment", path)
    assert status.returncode == 0
    assert json.loads(status.stdout)["status"] == "not_started"
    assert not runs.exists()
    result = cli("analyze", "--experiment", path, "--output", tmp_path / "analysis")
    assert result.returncode == 2
    assert not (tmp_path / "analysis").exists()


def test_completed_journal_cli_verifies_and_analyzes_on_cpu(tmp_path, monkeypatch):
    fixture, _, experiment = prepared(tmp_path, monkeypatch)
    path = fixture["kwargs"]["output"]
    monkeypatch.setattr(native, "NativeBackend", CPUFakeBackend)
    runtime.run_execution(path, approved_hash=experiment["execution_hash"], expected_revision=CODE["commit"])
    # The subprocess has no mocked Git or native backend. Stored-result paths
    # must remain CPU-only and independent of original model availability.
    fixture["weights"].unlink()
    status = cli("status", "--experiment", path)
    assert status.returncode == 0, status.stdout + status.stderr
    assert json.loads(status.stdout)["completed_rollouts"] == 12
    output = tmp_path / "cli-analysis"
    result = cli("analyze", "--experiment", path, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["new_rollouts"] == 0
    assert (output / "report.md").is_file()


def test_cli_help_describes_connected_commands():
    result = cli("--help")
    assert result.returncode == 0
    for name in ("cases", "prepare-run", "run", "status", "analyze"):
        assert name in result.stdout
