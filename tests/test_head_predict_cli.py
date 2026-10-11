"""CLI gates are CPU-only and never represent GPU/scientific validation."""
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def cli(name, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/openvla" / name), *args],
                          text=True, capture_output=True, check=False)


def test_descriptor_dry_preview_never_reads_inputs_or_creates_output(tmp_path):
    output = tmp_path / "descriptors.json"
    result = cli("headpredict.py", "descriptors", "--config", "missing.yaml",
                 "--reference-result", "missing", "--mapping", "missing.json", "--output", str(output))
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "not_executed" and data["new_rollouts"] == 0
    assert data["feature_set"] == "development"
    assert not output.exists()


def test_full10_preset_cannot_silently_expand_budget(tmp_path):
    result = cli("headrank.py", "plan", "--preset", "full10", "--features", "6009",
                 "--scores", "absent", "--registry", "absent", "--output", str(tmp_path / "plan.json"))
    assert result.returncode == 2
    assert "features=16" in json.loads(result.stdout)["error"]
    assert not (tmp_path / "plan.json").exists()


def test_full10_requires_all_ten_tasks_before_planning(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"scope": {"task_ids": [0, 1]}}))
    result = cli("headrank.py", "plan", "--preset", "full10", "--scores", str(scores),
                 "--registry", "absent", "--output", str(tmp_path / "plan.json"))
    assert result.returncode == 2
    assert "0..9" in json.loads(result.stdout)["error"]
    assert not (tmp_path / "plan.json").exists()


def test_headpredict_has_no_robot_execution_command():
    result = cli("headpredict.py", "run", "--execute")
    assert result.returncode == 2


def test_historical_full10_budget_is_3600_not_all_active():
    result = cli("headrank.py", "budget", "--features", "16", "--tasks", "10", "--cases-per-task", "20")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["total_rollouts"] == 3600
    assert data["execute"] is False
