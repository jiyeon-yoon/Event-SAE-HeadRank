import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/openvla/headtaylor.py"


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_score_is_dry_without_execute_even_with_missing_inputs(tmp_path):
    output = tmp_path / "absent.json"
    result = cli("score", "--config", "missing.yaml", "--reference-result", "missing",
                 "--baseline-scores", "missing.json", "--output", str(output))
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "not_executed"
    assert not output.exists()


def test_no_run_command_or_implicit_rollout():
    assert cli("run").returncode == 2
    assert cli("--help").returncode == 0


def test_compare_malformed_panel_is_blocked(tmp_path):
    result = cli("compare", "--scores", "missing.json", "--panel", "bad",
                 "--output", str(tmp_path / "out"))
    assert result.returncode == 2
    assert "NAME=JSON_PATH" in json.loads(result.stdout)["error"]
