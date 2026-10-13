"""CPU-only tests; fixtures are toy statistics, never scientific evidence."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from event_sae.research.head_taskwise import (
    _checked_hash, _interval, compute_taskwise, event_task_vectors, load_events,
    rank_correlations, score_matrices,
    validate_documents,
)
from event_sae.research.output_head.provenance import fingerprint


class TaskwiseTests(unittest.TestCase):
    def test_exact_integer_outcome_ties(self):
        scores = np.array([[[1, 2, 3, 4, 5]]])
        counts = np.array([[[0, 0, 0, 0, 1]]])
        expected = np.corrcoef([1, 2, 3, 4, 5], [2.5, 2.5, 2.5, 2.5, 5])[0, 1]
        self.assertAlmostEqual(rank_correlations(scores, counts)[0, 0, 0], expected)
        self.assertTrue(np.isnan(rank_correlations(scores, counts * 0)[0, 0, 0]))

    def test_event_same_canonical_coverage_rule_per_task(self):
        rows = [{"task_id": 0, "episode_coverage": .5},
                {"task_id": 0, "episode_coverage": .49},
                {"task_id": 1, "episode_coverage": 1},
                {"task_id": 1, "episode_coverage": .7}]
        global_score, tasks, counts = event_task_vectors(rows, [[1, 2], [100, 100],
            [3, 1], [2, 5]], [0, 1], 2)
        self.assertEqual(global_score.tolist(), [3, 5])
        self.assertEqual(tasks[0].tolist(), [1, 2])
        self.assertEqual(tasks[1].tolist(), [3, 5])
        self.assertEqual(counts, {"0": 1, "1": 2})

    def test_event_missing_task_or_coverage_rejected(self):
        for rows in ([{"task_id": 0, "episode_coverage": .49}],
                     [{"task_id": 0}], [{"task_id": 0, "episode_coverage": True}]):
            with self.assertRaises(ValueError):
                event_task_vectors(rows, [[1, 2]], [0], 2)
        with self.assertRaises(ValueError):
            event_task_vectors([{"task_id": 0, "episode_coverage": 1}], [[1, 2]], [0, 1], 2)

    def test_binding_mismatch_stops_before_event_tensor_read(self):
        scores = {"scope": {}, "event_max_binding": {"source_sha256": "expected"}}
        with patch("event_sae.research.head_rank.workflow.load_event_max", return_value=([], {})), \
                patch("torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "binding differs"):
                load_events("not-needed.pt", scores)
            loader.assert_not_called()

    def test_source_hash_rejects_change(self):
        doc = {"value": 1}
        doc["score_hash"] = fingerprint(doc)
        _checked_hash(doc, "score_hash")
        doc["value"] = 2
        with self.assertRaises(ValueError):
            _checked_hash(doc, "score_hash")

    @staticmethod
    def documents():
        scope = {"task_ids": [0], "dictionary_size": 32768, "discovery_manifest_hash": "discovery",
                 "split_manifest_hash": "split", "sae_sha256": "sae"}
        scores = {"synthetic": False, "sampling_mode": "full", "development_only": False,
                  "scope": scope, "event_max_binding": {"x": "bound"}}
        scores["score_hash"] = fingerprint(scores)
        case = {"suite": "toy", "task_id": 0, "task_episode_idx": 0,
                "initial_state_sha256": "state", "seed": 0}
        plan = {"plan_hash": "plan", "panel_feature_ids": [1, 2], "task_ids": [0],
                "source_bindings": {"score_hash": scores["score_hash"], "score_scope": scope,
                                    "event_max_binding": scores["event_max_binding"]}}
        experiment = {"synthetic": False, "prediction_plan": plan, "plan_hash": "plan",
                      "sources": {"sae": {"sha256": "sae"}}, "cases": [case], "total_rollouts": 4}
        experiment["execution_hash"] = fingerprint(experiment)
        analysis = {"synthetic": False, "status": "analyzed", "num_cases_per_feature": 1,
            "total_rollouts": 4, "identity_validation": {"status": "passed", "num_pairs": 1},
            "execution_hash": experiment["execution_hash"], "plan_hash": "plan",
            "feature_ids": [1, 2], "task_ids": [0],
            "feature_effects": [{"feature_id": fid, "n_success_to_failure": 0,
                "n_failure_to_success": 0, "task_effects": [{"task_id": 0, "num_pairs": 1, "drop": 0}]}
                for fid in [1, 2]],
            "paired_rows": [{"feature_id": fid, **case, "protocol_id": experiment["execution_hash"],
                             "raw_success": True, "edit_success": True, "drop": 0} for fid in [1, 2]]}
        analysis["analysis_hash"] = fingerprint(analysis)
        return scores, analysis, experiment

    def test_paired_records_scope_and_summary_must_match(self):
        documents = self.documents()
        with patch("event_sae.research.head_rank.protocol.verify_plan"):
            self.assertEqual(validate_documents(*documents)[0], [1, 2])
            for corruption in ("task", "drop", "raw", "summary", "identity"):
                scores, analysis, experiment = deepcopy(documents)
                if corruption == "task":
                    analysis["paired_rows"][0]["task_id"] = 1
                elif corruption == "drop":
                    analysis["paired_rows"][0]["drop"] = 1e-18
                elif corruption == "raw":
                    analysis["paired_rows"][1].update(raw_success=False, edit_success=False)
                elif corruption == "summary":
                    analysis["feature_effects"][0]["task_effects"][0]["drop"] = .5
                else:
                    analysis["identity_validation"]["status"] = "failed"
                analysis["analysis_hash"] = fingerprint({k: v for k, v in analysis.items() if k != "analysis_hash"})
                with self.assertRaises(ValueError, msg=corruption):
                    validate_documents(scores, analysis, experiment)

    def test_mismatched_sae_or_split_fails_even_when_resigned(self):
        for key in ("sae_sha256", "split_manifest_hash", "discovery_manifest_hash"):
            scores, analysis, experiment = deepcopy(self.documents())
            # Separate source scope objects before altering a single source.
            scores["scope"] = dict(scores["scope"])
            scores["scope"][key] = "different"
            scores["score_hash"] = fingerprint({k: v for k, v in scores.items() if k != "score_hash"})
            experiment["prediction_plan"]["source_bindings"]["score_hash"] = scores["score_hash"]
            experiment["execution_hash"] = fingerprint({k: v for k, v in experiment.items() if k != "execution_hash"})
            analysis["execution_hash"] = experiment["execution_hash"]
            analysis["analysis_hash"] = fingerprint({k: v for k, v in analysis.items() if k != "analysis_hash"})
            with patch("event_sae.research.head_rank.protocol.verify_plan"):
                with self.assertRaisesRegex(ValueError, "scope mismatch"):
                    validate_documents(scores, analysis, experiment)

    @staticmethod
    def sample_scores():
        return {"feature_scores": [{"feature_id": 0, "mean_head_kl": 2,
            "tail_head_kl": 3, "event_max": 4, "task_scores": [
            {"task_id": 0, "mean": 1, "tail": 2}, {"task_id": 1, "mean": 3, "tail": 4}]}]}

    def test_score_task_scope_and_global_event_replay(self):
        scores = self.sample_scores()
        names, values = score_matrices(scores, [0], [0, 1], [4], {0: [4], 1: [2]})
        self.assertEqual(names[-1], "taskwise_event_max")
        self.assertEqual(values[-1].ravel().tolist(), [4, 2])
        with self.assertRaisesRegex(ValueError, "Event-max"):
            score_matrices(scores, [0], [0, 1], [5], {0: [4], 1: [2]})
        scores["feature_scores"][0]["task_scores"].pop()
        with self.assertRaisesRegex(ValueError, "task-conditioned"):
            score_matrices(scores, [0], [0, 1], [4], {0: [4], 1: [2]})

    def test_shared_bootstrap_identical_predictors_have_exact_zero_contrast(self):
        panel, tasks = [0, 1, 2], [0, 1]
        rows = {fid: {} for fid in panel}
        for task in tasks:
            for case in range(4):
                key = ("toy", task, case, str(case), 0)
                for fid in panel:
                    rows[fid][key] = {"drop": [-1, 0, 1][fid]}
        matrix = np.tile([1., 2., 3.], (6, 2, 1))
        names = [f"m{i}" for i in range(6)]
        out = compute_taskwise(panel, tasks, rows, names, matrix, replicates=32, seed=17)
        replay = compute_taskwise(panel, tasks, rows, names, matrix, replicates=32, seed=17)
        self.assertEqual(out, replay)
        self.assertEqual(out["uncertainty"]["macro_denominator"], 2)
        for contrast in out["contrasts"]:
            self.assertEqual(contrast["difference"], 0)
            self.assertEqual((contrast["ci"]["low"], contrast["ci"]["high"]), (0, 0))

    def test_undefined_macro_never_drops_constant_task(self):
        panel, tasks = [0, 1], [0, 1]
        rows = {0: {}, 1: {}}
        for task in tasks:
            key = ("toy", task, 0, "hash", 0)
            rows[0][key] = {"drop": 0}
            rows[1][key] = {"drop": task}
        out = compute_taskwise(panel, tasks, rows, [str(i) for i in range(6)],
            np.tile([1., 2.], (6, 2, 1)), replicates=4)
        self.assertIsNone(out["comparisons"][0]["macro_spearman"])
        self.assertEqual(out["comparisons"][0]["macro_ci"]["undefined_replicates"], 4)
        interval = _interval([0., .1, np.nan])
        self.assertIsNone(interval["low"])
        self.assertEqual(interval["status"], "undefined_replicates_no_interval")

    def test_cli_help_does_not_import_torch(self):
        root = Path(__file__).resolve().parents[1]
        code = "import sys; import event_sae.research.head_taskwise; assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        help_result = subprocess.run([sys.executable, "scripts/openvla/headtaskwise.py", "--help"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertNotIn("--execute", help_result.stdout)

    def test_existing_output_is_not_overwritten(self):
        from event_sae.research.head_taskwise import analyze_taskwise
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                analyze_taskwise(scores_path="a", analysis_path="b", experiment_path="c",
                    event_path="d", output=tmp)


if __name__ == "__main__":
    unittest.main()
