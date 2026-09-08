"""CPU checks for cohort pairing, runtime configuration, and honest denominators."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.target_mapping import analyze, run


def make_rotations(n=6):
    output = [[], [], []]
    for i in range(n):
        answers = [f"Black person {i}", f"White person {i}", "Cannot tell"]
        tags = ["Black", "white", "unknown"]
        for rotation in range(3):
            row = {"category": "Race_ethnicity", "example_id": i, "question_index": "q0",
                   "context": "Two people", "question": "Who?", "label": (2 + rotation) % 3}
            row["answer_info"] = {}
            for position in range(3):
                source = (position - rotation) % 3
                row[f"ans{position}"] = answers[source]
                row["answer_info"][f"ans{position}"] = [answers[source], tags[source]]
            output[rotation].append(row)
    return output


class CohortTests(unittest.TestCase):
    def test_hash_cohort_is_order_independent_and_keeps_rotations(self):
        rows = make_rotations()
        keys, selected = run.select_rows(rows, 3)
        other_keys, _ = run.select_rows([list(reversed(r)) for r in rows], 3)
        self.assertEqual(keys, other_keys)
        for rotation in selected:
            self.assertEqual([run.row_key(row) for row in rotation], keys)

    def test_missing_rotations_duplicate_ids_and_stale_annotations_rejected(self):
        missing = make_rotations()
        missing[2].pop()
        duplicate = make_rotations()
        duplicate[0].append(duplicate[0][0])
        stale = make_rotations()
        stale[1][0]["answer_info"] = stale[0][0]["answer_info"]
        for rows in (missing, duplicate, stale):
            with self.assertRaises(ValueError):
                run.select_rows(rows, 2)


class AnalysisTests(unittest.TestCase):
    def records(self):
        records = []
        for rotation, items in enumerate(make_rotations(2)):
            for item in items:
                correct = item["example_id"] == 0
                selected = rotation if correct else (2 + rotation) % 3
                pred = rotation if correct else (1 + rotation) % 3
                row = {"model_output": "ABC"[pred], "selected_target_idx": selected,
                       "target_mapping_scores": [1., 2., 3.], "target_mapping_margin": 1,
                       "target_mapping_ties": [selected], "selector_forward_calls": 3,
                       "total_forward_calls": 68}
                records.append(analyze.score_record(row, item, rotation))
        return records

    def test_bootstrap_keeps_three_rotations_in_one_cluster(self):
        result = analyze.paired_bootstrap(self.records(), n_boot=1000)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["estimate"], 0)
        self.assertEqual(result["ci95"], [-1, 1])
        # Correctly paired identical conditions have zero difference even if file order changes.
        paired = analyze.paired_bootstrap(self.records(), list(reversed(self.records())), n_boot=1000)
        self.assertEqual(paired["ci95"], [0, 0])

    def test_missing_rotation_and_unpaired_item_rejected(self):
        with self.assertRaises(ValueError):
            analyze.paired_bootstrap(self.records()[:-1])
        other = self.records()
        for row in other:
            row["key"] = ("other",) + row["key"][1:]
        with self.assertRaises(ValueError):
            analyze.paired_bootstrap(self.records(), other)

    def test_mapping_unknowns_stay_in_denominator_and_consistency_is_not_accuracy(self):
        records = self.records()
        summary = analyze.mapping_metrics(records)
        self.assertEqual(summary["accuracy"], .5)
        self.assertEqual(summary["unknown_selected_rate"], .5)
        self.assertEqual(summary["rotation_semantic_consistency"], 1)
        self.assertEqual(analyze.rates(records)["n"], 6)
        self.assertEqual(summary["mean_total_forward_calls"], 68)

    def test_posthoc_truth_not_selector_output_defines_success_and_invalids_count(self):
        item = make_rotations(1)[0][0]
        row = {"model_output": "B", "selected_target_idx": 1,
               "black_idx": 1, "target_mapping_correct": True}
        scored = analyze.score_record(row, item, 0)
        self.assertEqual(scored["outcome"], "comparator")
        self.assertEqual(scored["selected_role"], "comparator")
        for text in ("blackblack", "AAAA", "The answer is A"):
            scored = analyze.score_record({"model_output": text}, item, 0)
            self.assertEqual(scored["outcome"], "invalid")
            self.assertEqual(analyze.rates([scored])["invalid_rate"], 1)

    def test_effort_reports_live_steps_separately_from_zero_commit_tail(self):
        rows = [{"alpha_traj": [2.] * 32 + [6.] * 32}]
        report = analyze.effort_metrics(rows, "direction_pi")
        self.assertEqual(report["live32"]["mean_sum_squared_alpha"], 128)
        self.assertEqual(report["full64"]["mean_sum_squared_alpha"], 1280)
        static = analyze.effort_metrics([{}], "normal_a4")
        self.assertEqual(static["live32"]["mean_sum_abs_alpha"], 128)


class HarnessTests(unittest.TestCase):
    def create_fixture_plan(self, root):
        data = root / "data"
        sources = data / "results/balanced"
        sources.mkdir(parents=True)
        for rotation, rows in enumerate(make_rotations(4)):
            (sources / f"_sweep400_rot{rotation}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (data / "steering").mkdir()
        (data / "steering/arrows.pt").write_bytes(b"frozen-test-direction")
        code_root = Path(os.environ.get("DLM_TEST_SELECTOR_ROOT", run.CODE_ROOT))
        (data / "eval").mkdir()
        for name in ("bbq_eval.py", "bias_metrics.py"):
            (data / "eval" / name).write_bytes((code_root / "eval" / name).read_bytes())
        with patch.dict("os.environ", {"DLM_ARROWS_PATH": str(data / "steering/arrows.pt")}), \
                patch.object(run.subprocess, "check_output", return_value="fixture-revision\n"):
            return run.create_plan(data, root / "out", limit=2, python="test-python", code_root=code_root)

    def test_plan_is_explicit_portable_and_never_launches_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run.subprocess, "run") as launch:
            path, plan = self.create_fixture_plan(Path(directory))
            self.assertTrue(path.exists())
            self.assertEqual(len(plan["jobs"]), 12)
            self.assertEqual(plan["n_semantic_items"], 2)
            launch.assert_not_called()
            for job in plan["jobs"]:
                argv = job["argv"]
                self.assertEqual(argv[0], "test-python")
                if job["condition"].endswith("pi") or job["condition"] == "clean":
                    expected = "direction" if job["condition"] == "direction_pi" else "oracle"
                    self.assertEqual(argv[argv.index("--target-mapping") + 1], expected)

    def test_execution_rejects_changed_artifacts_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = self.create_fixture_plan(Path(directory))
            Path(plan["items"][0]["path"]).write_text("changed")
            with patch.object(run.subprocess, "run") as launch:
                with self.assertRaisesRegex(ValueError, "changed"):
                    run.execute_plan(path)
                launch.assert_not_called()

    def test_end_to_end_analysis_checks_mapping_mode_and_exact_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = self.create_fixture_plan(Path(directory))
            for job in plan["jobs"]:
                rows = run.read_rows(plan["items"][job["rotation"]]["path"])
                for row in rows:
                    target, _ = analyze.annotated_indices(row)
                    row.update(model_output="ABC"[target], selected_target_idx=target,
                               options={letter: row[f"ans{k}"] for k, letter in enumerate("ABC")},
                               target_mapping="direction" if job["condition"] == "direction_pi" else "oracle",
                               selector_forward_calls=3 if job["condition"] == "direction_pi" else 0)
                sample = Path(job["samples"])
                sample.parent.mkdir(parents=True)
                sample.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = analyze.analyze_plan(path, n_boot=100)
            self.assertEqual(result["paired_contrasts"]["direction_pi_minus_oracle_pi"]["ci95"], [0, 0])
            self.assertEqual(result["conditions"]["direction_pi"]["mapping"]["accuracy"], 1)
            job = next(job for job in plan["jobs"] if job["condition"] == "direction_pi")
            rows = run.read_rows(job["samples"])
            rows[0]["target_mapping"] = "oracle"
            Path(job["samples"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "mapping mode"):
                analyze.analyze_plan(path, n_boot=100)
            rows[0]["target_mapping"] = "direction"
            rows[0]["options"] = {"A": "wrong rotation", "B": "x", "C": "y"}
            Path(job["samples"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "planned rotation"):
                analyze.analyze_plan(path, n_boot=100)


if __name__ == "__main__":
    unittest.main()
