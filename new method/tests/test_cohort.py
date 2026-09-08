"""Offline cohort/scoring invariants, separate from the inference tests."""
import importlib.util
import os
from pathlib import Path
import unittest


SOURCE = Path(os.environ.get("DLM_NEW_METHOD_ROOT", Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("new_method_test_cohort", SOURCE / "cohort.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def item(index):
    return {"category": "Race_ethnicity", "example_id": index, "question_index": "q",
            "context": f"Context {index}", "question": "Who?", "context_condition": "ambig",
            "ans0": "Target person", "ans1": "Other person", "ans2": "Cannot tell",
            "answer_info": {"ans0": ["target", "Black"], "ans1": ["other", "White"],
                            "ans2": ["unknown", "unknown"]}, "label": 2}


class CohortTests(unittest.TestCase):
    def test_fit_and_known_prior_items_excluded_before_complete_rotations(self):
        rows = [item(index) for index in range(5)]
        records, metadata = C.prepare_cohort(rows, [rows[0]], [rows[1]], limit=2, rotations=3)
        self.assertEqual(metadata["available_semantic_items"], 3)
        self.assertEqual(metadata["n_semantic_items"], 2)
        self.assertEqual(len(records), 6)
        selected = {record["semantic_key"][1] for record in records}
        self.assertTrue(selected.isdisjoint({"0", "1"}))
        for record in records:
            row, rotation = record["row"], record["rotation"]
            self.assertEqual(row[f"ans{rotation}"], "Target person")
            self.assertEqual(C.annotated_indices(row), (rotation, (2 + rotation) % 3))
            self.assertEqual(row["label"], (2 + rotation) % 3)
        repeat, _ = C.prepare_cohort(rows[::-1], [rows[0]], [rows[1]], limit=2, rotations=3)
        self.assertEqual(records, repeat)

    def test_fit_overlap_rejected_for_custom_input_even_with_removed_ids(self):
        fit = item(0)
        visible = {field: fit[field] for field in C.VISIBLE_KEYS if field in fit}
        visible["ans0"], visible["ans2"] = visible["ans2"], visible["ans0"]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            C.prepare_cohort([visible], [fit], benchmark=False, rotations=1)

    def test_custom_unlabelled_rows_do_not_acquire_self_certified_target_scores(self):
        source = item(2)
        visible = {field: source[field] for field in C.VISIBLE_KEYS if field in source}
        records, _ = C.prepare_cohort([visible], [], benchmark=False, rotations=1)
        score = C.score_output({"model_output": "B", "selected_target_idx": 1}, records[0]["row"])
        self.assertEqual(score["pred_index"], 1)
        self.assertFalse(score["scoring_available"])
        self.assertIsNone(score["gap"])
        self.assertIsNone(score["mapping_correct"])

    def test_wrong_mapping_unknown_and_invalid_remain_in_outcome_denominators(self):
        row = item(3)
        records = []
        for rotation, output in enumerate(("B", "A", "unparseable prose")):
            rotated = C.rotate_row(row, rotation)
            record = {"semantic_key": list(C.semantic_key(row)), "rotation": rotation,
                      "arm": "auto_pi", "model_output": output,
                      "selected_target_idx": (1 + rotation) % 3,
                      "standalone_forward_calls": 68, "generation_forward_calls": 65,
                      "standalone_elapsed_s": 1.}
            record["evaluation"] = C.score_output(record, rotated)
            records.append(record)
        summary = C.summarize(records, 3)["auto_pi"]
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_scored"], 3)
        self.assertEqual(summary["counts"], {"target": 0, "comparator": 1, "abstain": 1, "invalid": 1})
        self.assertAlmostEqual(summary["gap"], -1 / 3)
        self.assertEqual(summary["mapping_accuracy"], 0.)
        self.assertEqual(summary["mapping_rotation_consistency"], 1.)
        with self.assertRaisesRegex(ValueError, "incomplete rotation"):
            C.summarize(records[:-1], 3)

    def test_paired_bootstrap_resamples_two_semantic_items_not_six_rotations(self):
        records = []
        for index in (0, 1):
            for arm in ("auto_pi", "auto_openloop"):
                picks_target = (index == 0) == (arm == "auto_pi")
                for rotation in range(3):
                    row = C.rotate_row(item(index), rotation)
                    predicted = (rotation + (0 if picks_target else 1)) % 3
                    record = {"semantic_key": list(C.semantic_key(row)), "rotation": rotation,
                              "arm": arm, "model_output": "ABC"[predicted],
                              "selected_target_idx": rotation, "standalone_forward_calls": 68,
                              "generation_forward_calls": 65, "standalone_elapsed_s": 1.}
                    record["evaluation"] = C.score_output(record, row)
                    records.append(record)
        result = C.summarize(records, 3, n_boot=2000, seed=7)["paired_auto_pi_minus_auto_openloop"]
        self.assertEqual(result["n_semantic_clusters"], 2)
        self.assertEqual(result["gap_difference"], 0.)
        self.assertEqual(result["ci95"], [-2., 2.])
        changed = [dict(record, semantic_key=["different", "key"])
                   if record["arm"] == "auto_openloop" and record["semantic_key"][1] == "1"
                   else record for record in records]
        with self.assertRaisesRegex(ValueError, "same semantic item keys"):
            C.summarize(changed, 3, n_boot=2)


if __name__ == "__main__":
    unittest.main()
